from logging import INFO, WARNING

import numpy as np
import sklearn.metrics.pairwise as smp
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class FoolsGoldStrategy(StrategyTrackingMixin, FedAvg):
    """
    FoolsGold Robust Aggregation Strategy.

    Defends against Sybil-based data poisoning attacks in Federated Learning. 
    It evaluates the historical similarity of client gradient updates over time and 
    penalizes clients (by downweighting their contribution) that submit highly correlated 
    or identical gradients, a hallmark of colluding malicious actors.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

        self.n_clients: int = settings.client.num_clients
        self.rounds: int = 0

        # FoolsGold-specific hyperparameters
        self.use_memory: bool = settings.defence.fg_use_memory
        self.memory_size: int = settings.defence.fg_memory_size
        self.epsilon: float = settings.defence.fg_epsilon

        self.gradient_memory = None
        self.client_id_mapping: dict = {}

    def initialize_parameters(self, client_manager):
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters is not None:
            self.global_weights = parameters_to_ndarrays(initial_parameters)
        return initial_parameters

    def _unflatten(self, flat_array: np.ndarray, reference_ndarrays: list) -> list:
        """
        Reconstructs layer-wise weight arrays from a flattened 1D parameter vector.

        :param flat_array: The 1D NumPy array containing the aggregated global weights.
        :param reference_ndarrays: A list of original layer arrays used as a shape template.
        :return: A list of NumPy arrays correctly reshaped for the PyTorch model.
        """
        aggregated_ndarrays = []
        idx = 0
        for ref in reference_ndarrays:
            size = int(np.prod(ref.shape))
            aggregated_ndarrays.append(flat_array[idx : idx + size].reshape(ref.shape))
            idx += size
        return aggregated_ndarrays

    def aggregate_fit(self, server_round: int, results, failures):
        """
        Executes robust aggregation using the FoolsGold similarity-penalization algorithm.

        :param server_round: The current federated learning round.
        :param results: A list of parameter updates successfully received from active clients.
        :param failures: A list of encountered errors or unresponsive clients.
        :return: A tuple containing the aggregated global parameters and an empty metrics dictionary.
        """
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # 1. Deserialize and extract local weights from clients
        client_updates = []
        cids = []
        for client_proxy, fit_res in results:
            client_updates.append(parameters_to_ndarrays(fit_res.parameters))
            logical_id = str(fit_res.metrics.get("id", client_proxy.cid))
            if logical_id not in self.client_id_mapping:
                self.client_id_mapping[logical_id] = len(self.client_id_mapping)
            cids.append(self.client_id_mapping[logical_id])

        n = len(client_updates)

        # Grab the robust baseline
        if getattr(self, "global_weights", None) is None:
            log(WARNING, "FoolsGold: global_weights missing! Falling back to first valid client.")
            valid_idx = 0
            for i in range(n):
                if np.all(np.isfinite(np.concatenate([w.flatten() for w in client_updates[i]]))):
                    valid_idx = i
                    break
            self.global_weights = client_updates[valid_idx]

        flat_global_weights = np.concatenate([w.flatten() for w in self.global_weights])
        grad_dim = flat_global_weights.shape[0]

        # Initialize gradient memory if it doesn't exist
        if self.gradient_memory is None:
            if self.use_memory:
                self.gradient_memory = np.zeros((self.n_clients, grad_dim, self.memory_size))
            else:
                self.gradient_memory = np.zeros((self.n_clients, grad_dim))

        f_grads = []
        flat_weights = []
        for client_weights in client_updates:
            flat_client = np.concatenate([w.flatten() for w in client_weights])
            f_grads.append(flat_global_weights - flat_client)
            flat_weights.append(flat_client)

        grads_np = np.stack(f_grads, axis=0)

        # Filter NaNs
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = int(np.sum(~valid_mask))

        if n_invalid > 0:
            log(
                WARNING,
                "FoolsGold: %d/%d clients have NaN/Inf gradients, skipping them from memory.",
                n_invalid,
                n,
            )
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                log(WARNING, "FoolsGold: All gradients invalid, using simple mean")
                mean_weights = np.mean(np.stack(flat_weights, axis=0), axis=0)
                aggregated_ndarrays = self._unflatten(mean_weights, client_updates[0])
                return ndarrays_to_parameters(aggregated_ndarrays), {}
        else:
            valid_indices = np.arange(n)

        # Update gradient memory for valid clients
        memory_idx = self.rounds % self.memory_size
        for valid_idx in valid_indices:
            actual_cid = cids[valid_idx]
            if self.use_memory:
                self.gradient_memory[actual_cid, :, memory_idx] = grads_np[valid_idx]
            else:
                self.gradient_memory[actual_cid, :] = grads_np[valid_idx]

        # Prepare similarity gradients based on memory
        if self.use_memory:
            if self.rounds < self.memory_size:
                similarity_gradients = grads_np[valid_indices]
            else:
                similarity_gradients = np.sum(self.gradient_memory[cids, :, :], axis=2)[valid_indices]
        else:
            similarity_gradients = grads_np[valid_indices]

        # Compute FoolsGold weights
        n_valid = similarity_gradients.shape[0]

        # 1. Pairwise cosine similarity matrix
        cs = smp.cosine_similarity(similarity_gradients) - np.eye(n_valid)

        # 2. Maximum similarity per client
        maxcs = np.max(cs, axis=1) + self.epsilon

        # 3. Pardoning mechanism
        for i in range(n_valid):
            for j in range(n_valid):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # 4. Compute weights: 1 - max similarity
        wv = 1 - np.max(cs, axis=1)
        wv = np.clip(wv, 0, 1)

        # 5. Rescale so max weight is close to 1
        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        # 6. (Removed logit transform and 0.99 cap to match reference implementation)

        trust_weights = wv
        flat_weights_valid = [flat_weights[i] for i in valid_indices]

        # Log per-client trust weights
        honest_trusts = []
        malicious_trusts = []
        for i, v_idx in enumerate(valid_indices):
            tw = trust_weights[i]
            mcs = maxcs[i]
            client_proxy, fit_res = results[v_idx]
            cid = fit_res.metrics.get("id", client_proxy.cid)
            ctype = fit_res.metrics.get("client_type", "Unknown")
            log(INFO, "[FoolsGold] Client %s (%s) Trust Weight: %.4f | Max Cosine Sim: %.4f", cid, ctype, tw, mcs)
            if ctype == "Honest":
                honest_trusts.append(tw)
            elif ctype == "Malicious":
                malicious_trusts.append(tw)

        # Weighted aggregation
        total_weight = trust_weights.sum()
        if total_weight > 0:
            weighted_sum = np.zeros_like(flat_weights_valid[0])
            for fw, tw in zip(flat_weights_valid, trust_weights):
                weighted_sum += tw * fw
            aggregated_flat = weighted_sum / total_weight
        else:
            aggregated_flat = np.mean(np.stack(flat_weights_valid, axis=0), axis=0)

        active_clients = int(np.sum(trust_weights > 0))
        mean_trust = float(np.mean(trust_weights))
        self.rounds += 1

        metrics_aggregated = {
            "FG-active-ratio": float(active_clients / n) if n > 0 else 0.0,
            "FG-mean-trust": mean_trust,
            "FG-mean-honest-trust": float(np.mean(honest_trusts)) if honest_trusts else 0.0,
            "FG-mean-malicious-trust": float(np.mean(malicious_trusts)) if malicious_trusts else 0.0,
        }

        self._log_results(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        aggregated_ndarrays = self._unflatten(aggregated_flat, client_updates[0])
        self.global_weights = aggregated_ndarrays
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
