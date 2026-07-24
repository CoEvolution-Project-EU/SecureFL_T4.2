from logging import INFO, WARNING

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
    """FL-Defender Strategy.

    Combating Targeted Attacks in Federated Learning.
    Uses PCA on cosine similarity matrix + accumulated reputation scoring.
    """

    def __init__(self, *args, **kwargs):
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
        """Standardize and apply PCA(n_components=2) to the similarity matrix."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(similarity_matrix)
        n_components = min(2, scaled.shape[0], scaled.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled)

    def _unflatten(self, flat_array: np.ndarray, reference_ndarrays: list) -> list:
        """Reconstruct layer-wise weight arrays from a flat parameter vector."""
        aggregated_ndarrays = []
        idx = 0
        for ref in reference_ndarrays:
            size = int(np.prod(ref.shape))
            aggregated_ndarrays.append(flat_array[idx : idx + size].reshape(ref.shape))
            idx += size
        return aggregated_ndarrays

    def aggregate_fit(self, server_round: int, results, failures):
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # 1. Deserialize and extract local weights from clients
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        n = len(weights_results)

        # Ensure global weights exist for pseudo-gradient calculation
        if getattr(self, "global_weights", None) is None:
            log(
                WARNING,
                "FL-Defender: global_weights missing! Falling back to first valid client.",
            )
            valid_idx = 0
            for i in range(n):
                if np.all(np.isfinite(np.concatenate([w.flatten() for w in weights_results[i][0]]))):
                    valid_idx = i
                    break
            self.global_weights = weights_results[valid_idx][0]

        flat_global_weights = np.concatenate([w.flatten() for w in self.global_weights])

        log(
            INFO,
            "FL-Defender Stateful Check: Round %d using full global_weights with norm %.4f",
            server_round,
            np.linalg.norm(flat_global_weights),
        )

        f_grads = []
        flat_weights = []
        for client_weights, _ in weights_results:
            # State preparation: flatten the complete model for trust-weighted aggregation
            flat_client = np.concatenate([w.flatten() for w in client_weights])
            flat_weights.append(flat_client)

            # Feature engineering: compute the pseudo-gradient across the entire model
            # to identify targeted manipulation
            grad = flat_global_weights - flat_client
            f_grads.append(grad)

        grads_np = np.stack(f_grads, axis=0)

        # Filter out clients with NaN/Inf gradients
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = int(np.sum(~valid_mask))

        if n_invalid > 0:
            log(
                WARNING,
                "FL-Defender: %d/%d clients have NaN/Inf gradients, filtering them out",
                n_invalid,
                n,
            )
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                log(WARNING, "FL-Defender: All gradients invalid, using simple mean")
                mean_weights = np.mean(np.stack(flat_weights, axis=0), axis=0)
                aggregated_ndarrays = self._unflatten(mean_weights, weights_results[0][0])
                return ndarrays_to_parameters(aggregated_ndarrays), {}

            grads_np = grads_np[valid_indices]
            flat_weights_valid = [flat_weights[i] for i in valid_indices]
            n_valid = len(flat_weights_valid)
        else:
            valid_indices = np.arange(n)
            flat_weights_valid = flat_weights
            n_valid = n

        # Step 1: Pairwise cosine similarity matrix, removing self-similarity
        cs = smp.cosine_similarity(grads_np) - np.eye(n_valid)

        # Step 2: Compress dimensionality via PCA to extract discriminative features
        cs_pca = self._get_pca(cs)

        # Step 3: Compute a Byzantine-resilient centroid using the median
        centroid = np.median(cs_pca, axis=0)

        # Step 4: Score each client by cosine similarity to the robust centroid
        scores = smp.cosine_similarity([centroid], cs_pca)[0]

        # Step 5: Accumulate scores across rounds to build long-term client reputation
        if n == self.n_clients and n_invalid == 0:
            self.score_history += scores
        else:
            self.score_history = np.zeros(self.n_clients)
            if n_invalid == 0:
                self.score_history[:n] = scores
            else:
                self.score_history[valid_indices] = scores

        # Step 6: Derive normalized trust weights via 25th-percentile thresholding
        q1 = np.quantile(self.score_history, 0.25)
        trust = np.clip(self.score_history - q1, 0, None)
        max_trust = trust.max()
        if max_trust > 0:
            trust /= max_trust

        trust_weights = trust[valid_indices]

        # Diagnostic trap analysis: flag clients with anomalous gradient magnitudes
        magnitudes = np.linalg.norm(grads_np, axis=1)
        mean_mag = np.mean(magnitudes)

        log(
            INFO,
            "[FL-Defender Trap Analysis] Weight Magnitudes — Mean: %.2f | Max: %.2f | Min: %.2f",
            mean_mag,
            np.max(magnitudes),
            np.min(magnitudes),
        )

        caught_count = 0
        bypassed_count = 0
        for idx, (mag, tw) in enumerate(zip(magnitudes, trust_weights)):
            if mag > mean_mag * 3:  # 3x the mean magnitude is suspicious
                if tw > 0:
                    bypassed_count += 1
                    log(
                        WARNING,
                        "🚨 TRAP TRIGGERED! Client %d has anomalous magnitude (%.2f vs mean %.2f) "
                        "but bypassed the PCA filter (Trust Score: %.4f).",
                        idx,
                        mag,
                        mean_mag,
                        tw,
                    )
                else:
                    caught_count += 1
                    log(
                        INFO,
                        "🛡️ ATTACK BLOCKED! Client %d has anomalous magnitude (%.2f vs mean %.2f) "
                        "and was caught by the PCA filter (Trust Score: %.4f).",
                        idx,
                        mag,
                        mean_mag,
                        tw,
                    )

        if caught_count > 0 or bypassed_count > 0:
            log(
                INFO,
                "[FL-Defender Trap Summary] Anomalous Clients Caught: %d | Bypassed: %d",
                caught_count,
                bypassed_count,
            )

        # Step 7: Trust-weighted aggregation over client parameters
        total_weight = trust_weights.sum()
        if total_weight > 0:
            weighted_sum = np.zeros_like(flat_weights_valid[0])
            for fw, tw in zip(flat_weights_valid, trust_weights):
                weighted_sum += tw * fw
            aggregated_flat = weighted_sum / total_weight
        else:
            aggregated_flat = np.mean(np.stack(flat_weights_valid, axis=0), axis=0)

        # Persist detection statistics for this round
        active_clients = int(np.sum(trust_weights > 0))
        self.detection_stats = {
            "total_clients": n,
            "valid_clients": n_valid,
            "invalid_clients": n_invalid,
            "active_clients": active_clients,
            "filtered_out": int(np.sum(trust_weights == 0)),
            "active_ratio": float(active_clients / n) if n > 0 else 0.0,
            "mean_trust": float(np.mean(trust_weights)),
        }
        self.rounds += 1

        metrics_aggregated = {
            "FLDef-active-ratio": self.detection_stats["active_ratio"],
            "FLDef-mean-trust": self.detection_stats["mean_trust"],
        }

        self._log_results(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        # Persist aggregated state to compute pseudo-gradients in the subsequent round
        aggregated_ndarrays = self._unflatten(aggregated_flat, weights_results[0][0])
        self.global_weights = aggregated_ndarrays
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
