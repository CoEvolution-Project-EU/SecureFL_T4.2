from logging import INFO, WARNING
from typing import Optional

import numpy as np
import sklearn.metrics.pairwise as smp
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class FLDefenderStrategy(StrategyTrackingMixin, FedAvg):
    """FL-Defender robust aggregation strategy.

    Defends against targeted poisoning attacks (label-flip, backdoor) by:

      1. Computing pairwise cosine similarity of last-layer pseudo-gradients.
      2. Reducing dimensionality via PCA (n=2) on the similarity matrix.
      3. Scoring each client by cosine similarity to the median centroid.
      4. Accumulating scores across rounds into a persistent reputation history.
      5. Deriving normalised trust weights and performing a weighted aggregation.

    """

    def __init__(self, *args, **kwargs) -> None:
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

        self.n_clients: int = settings.client.num_clients
        self.score_history: np.ndarray = np.zeros(self.n_clients)
        self.rounds: int = 0
        self.detection_stats: dict = {}

    def initialize_parameters(self, client_manager):
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters is not None:
            self.global_weights = parameters_to_ndarrays(initial_parameters)
        return initial_parameters

    def _get_pca(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Standardise a similarity matrix and reduce it to ≤2 principal components.

        :param similarity_matrix: Square pairwise cosine-similarity matrix (n × n).
        :return: PCA-transformed feature matrix (n × k), where k = min(2, n).
        """
        scaled = StandardScaler().fit_transform(similarity_matrix)
        n_components = min(2, scaled.shape[0], scaled.shape[1])
        return PCA(n_components=n_components).fit_transform(scaled)

    def _unflatten(
        self, flat_array: np.ndarray, reference_ndarrays: list
    ) -> list:
        """Reconstruct layer-wise weight arrays from a flat 1-D parameter vector.

        :param flat_array: Flat array containing concatenated model parameters.
        :param reference_ndarrays: Layer arrays whose shapes serve as the template.
        :return: List of arrays reshaped to the original layer dimensions.
        """
        result, idx = [], 0
        for ref in reference_ndarrays:
            size = int(np.prod(ref.shape))
            result.append(flat_array[idx : idx + size].reshape(ref.shape))
            idx += size
        return result

    def _last_layer_grad(
        self, client_weights: list, global_weights: list
    ) -> np.ndarray:
        """Compute the pseudo-gradient of the final classification layer's weight matrix.

        Only the weight matrix (index ``-2``) is used; the bias (index ``-1``) is
        intentionally excluded, matching the original author's implementation::

            last_g = global_model[-2]
            grad   = (last_g - local_model[-2]).reshape(-1)

        The final-layer weights encode the class mapping directly and therefore
        carry the strongest signal for detecting targeted poisoning attacks.

        :param client_weights: Layer arrays received from a client.
        :param global_weights: Layer arrays of the current global model.
        :return: Flat pseudo-gradient vector for the last-layer weight matrix.
        """
        return (global_weights[-2] - client_weights[-2]).flatten()

    def aggregate_fit(self, server_round: int, results, failures):
        """Perform FL-Defender reputation-based robust aggregation.

        :param server_round: Current federated learning round index.
        :param results: Successful client fit results.
        :param failures: Failed or unresponsive client results.
        :return: Tuple of (aggregated Parameters, metrics dict).
        """
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Deserialise client updates
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        n = len(weights_results)

        # Initialise global weights on the first round if not yet set
        if getattr(self, "global_weights", None) is None:
            log(WARNING, "FL-Defender: global_weights not initialised; using first valid client.")
            self.global_weights = next(
                (
                    w for w, _ in weights_results
                    if np.all(np.isfinite(np.concatenate([a.flatten() for a in w])))
                ),
                weights_results[0][0],
            )

        log(
            INFO,
            "FL-Defender | round %d | last-layer norm: %.4f",
            server_round,
            np.linalg.norm(self.global_weights[-2]),
        )

        # Build per-client feature vectors and full-model flat arrays
        f_grads: list = []
        flat_weights: list = []

        for client_weights, _ in weights_results:
            # Detection feature: last-layer weight-matrix pseudo-gradient (no bias)
            f_grads.append(self._last_layer_grad(client_weights, self.global_weights))
            # Aggregation payload: full model flattened
            flat_weights.append(np.concatenate([w.flatten() for w in client_weights]))

        grads_np = np.stack(f_grads, axis=0)

        # Filter clients with non-finite gradients
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = int(np.sum(~valid_mask))

        if n_invalid > 0:
            log(WARNING, "FL-Defender: %d/%d clients have NaN/Inf gradients — excluding.", n_invalid, n)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                log(WARNING, "FL-Defender: all gradients invalid; falling back to simple mean.")
                mean_w = np.mean(np.stack(flat_weights, axis=0), axis=0)
                return ndarrays_to_parameters(self._unflatten(mean_w, weights_results[0][0])), {}

            grads_np = grads_np[valid_indices]
            flat_weights_valid = [flat_weights[i] for i in valid_indices]
            n_valid = len(flat_weights_valid)
        else:
            valid_indices = np.arange(n)
            flat_weights_valid = flat_weights
            n_valid = n

        # Steps 1–4: Cosine similarity → PCA → centroid → per-client scores
        cs = smp.cosine_similarity(grads_np) - np.eye(n_valid)   # remove self-similarity
        cs_pca = self._get_pca(cs)
        centroid = np.median(cs_pca, axis=0)                      # robust to Byzantine outliers
        scores = smp.cosine_similarity([centroid], cs_pca)[0]

        # Step 5: Accumulate reputation scores across rounds
        if n == self.n_clients and n_invalid == 0:
            self.score_history += scores
        else:
            self.score_history = np.zeros(self.n_clients)
            self.score_history[valid_indices] = scores

        # Step 6: Derive normalised trust weights via Q1 thresholding
        q1 = np.quantile(self.score_history, 0.25)
        trust = np.clip(self.score_history - q1, 0.0, None)
        max_trust = trust.max()
        if max_trust > 0:
            trust /= max_trust

        trust_weights = trust[valid_indices]

        honest_trusts = []
        malicious_trusts = []
        for v_idx, tw in zip(valid_indices, trust_weights):
            client_proxy, fit_res = results[v_idx]
            cid = fit_res.metrics.get("id", client_proxy.cid)
            ctype = fit_res.metrics.get("client_type", "Unknown")
            log(INFO, "[FL-Defender] Client %s (%s) Trust Score: %.4f", cid, ctype, tw)
            if ctype == "Honest":
                honest_trusts.append(tw)
            elif ctype == "Malicious":
                malicious_trusts.append(tw)

        # Forensic diagnostics: flag anomalous gradient magnitudes
        self._log_trap_analysis(grads_np, trust_weights)

        # Step 7: Trust-weighted aggregation
        total_trust = trust_weights.sum()
        if total_trust > 0:
            weighted_sum = sum(tw * fw for tw, fw in zip(trust_weights, flat_weights_valid))
            aggregated_flat = weighted_sum / total_trust
        else:
            aggregated_flat = np.mean(np.stack(flat_weights_valid, axis=0), axis=0)

        # Persist state and emit metrics
        active = int(np.sum(trust_weights > 0))
        self.detection_stats = {
            "total_clients": n,
            "valid_clients": n_valid,
            "invalid_clients": n_invalid,
            "active_clients": active,
            "filtered_out": int(np.sum(trust_weights == 0)),
            "active_ratio": float(active / n) if n > 0 else 0.0,
            "mean_trust": float(np.mean(trust_weights)),
        }
        self.rounds += 1

        metrics_aggregated = {
            "FLDef-active-ratio": self.detection_stats["active_ratio"],
            "FLDef-mean-trust": self.detection_stats["mean_trust"],
            "FLDef-mean-honest-trust": float(np.mean(honest_trusts)) if honest_trusts else 0.0,
            "FLDef-mean-malicious-trust": float(np.mean(malicious_trusts)) if malicious_trusts else 0.0,
        }
        self._log_results(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        aggregated_ndarrays = self._unflatten(aggregated_flat, weights_results[0][0])
        self.global_weights = aggregated_ndarrays
        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated

    # Diagnostics
    def _log_trap_analysis(
        self, grads_np: np.ndarray, trust_weights: np.ndarray
    ) -> None:
        """Log gradient-magnitude anomalies to support forensic analysis.

        Clients whose pseudo-gradient magnitude exceeds 3× the group mean are
        flagged as suspicious.  Whether they were suppressed (trust = 0) or
        slipped through (trust > 0) is reported separately.

        :param grads_np: Stacked last-layer pseudo-gradients (n_valid × d).
        :param trust_weights: Normalised trust scores for each valid client.
        """
        magnitudes = np.linalg.norm(grads_np, axis=1)
        mean_mag = float(np.mean(magnitudes))
        threshold = mean_mag * 3.0

        log(
            INFO,
            "FL-Defender | gradient magnitudes — mean: %.2f  max: %.2f  min: %.2f",
            mean_mag,
            float(np.max(magnitudes)),
            float(np.min(magnitudes)),
        )

        caught = bypassed = 0
        for idx, (mag, tw) in enumerate(zip(magnitudes, trust_weights)):
            if mag <= threshold:
                continue
            if tw > 0:
                bypassed += 1
                log(
                    WARNING,
                    "🚨 TRAP TRIGGERED — client %d: magnitude %.2f (%.1fx mean); trust %.4f",
                    idx, mag, mag / mean_mag, tw,
                )
            else:
                caught += 1
                log(
                    INFO,
                    "🛡️  ATTACK BLOCKED  — client %d: magnitude %.2f (%.1fx mean); trust %.4f",
                    idx, mag, mag / mean_mag, tw,
                )

        if caught or bypassed:
            log(
                INFO,
                "FL-Defender | anomalous clients — caught: %d  bypassed: %d",
                caught,
                bypassed,
            )
