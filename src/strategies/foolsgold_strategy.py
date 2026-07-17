import json
from logging import INFO, WARNING

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
import wandb
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir


class FoolsGoldStrategy(FedAvg):
    """FoolsGold Strategy.

    Mitigating Sybils in Federated Learning Poisoning.
    Penalizes clients that submit highly similar gradients over time.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        self.model = model_config.model
        super().__init__(*args, **kwargs)

        self.save_path, self.run_dir = create_run_dir()
        if settings.general.use_wandb:
            self._init_wandb_project()

        self.best_acc_so_far = 0.0
        self.best_loss_so_far = None
        self.results = {}

        self.n_clients = settings.client.num_clients
        self.rounds = 0

        # FoolsGold Specific Hyperparameters
        self.use_memory = settings.defence.fg_use_memory
        self.memory_size = settings.defence.fg_memory_size
        self.epsilon = settings.defence.fg_epsilon

        # Initialize memory tracking
        self.gradient_memory = None
        self.client_id_mapping = {}

    def _init_wandb_project(self):
        if settings.attack.type is not None:
            name = f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-{settings.attack.type}"
            wandb.init(project=PROJECT_NAME, name=name)
        else:
            wandb.init(
                project=PROJECT_NAME,
                name=f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-No attack",
            )

    def _store_results(self, tag: str, results_dict) -> None:
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def initialize_parameters(self, client_manager):
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters is not None:
            self.global_weights = parameters_to_ndarrays(initial_parameters)
        return initial_parameters

    def _update_best_acc(self, server_round: int, accuracy, parameters: Parameters) -> None:
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %f", accuracy)
            model = self.model
            set_weights(model, parameters_to_ndarrays(parameters))
            file_name = f"model_state_acc_{accuracy}_round_{server_round}.pth"
            if hasattr(self, "best_model_path") and self.best_model_path and self.best_model_path.exists():
                import os

                try:
                    os.remove(self.best_model_path)
                except Exception:
                    pass
            self.best_model_path = self.save_path / file_name
            torch.save(model.state_dict(), self.best_model_path)

    def _store_results_and_log(self, server_round: int, tag: str, results_dict) -> None:
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})
        if settings.general.use_wandb:
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round: int, parameters: Parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)
        if self.best_loss_so_far is None or loss <= self.best_loss_so_far:
            self.best_loss_so_far = loss
            log(INFO, "💡 New best global loss found: %f", loss)
        self._store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self._store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def _unflatten(self, flat_array, reference_ndarrays):
        """Helper to unflatten array back to layer shapes."""
        original_shapes = [w.shape for w in reference_ndarrays]
        aggregated_ndarrays = []
        idx = 0
        for shape in original_shapes:
            size = int(np.prod(shape))
            layer_array = flat_array[idx : idx + size].reshape(shape)
            aggregated_ndarrays.append(layer_array)
            idx += size
        return aggregated_ndarrays

    def aggregate_fit(self, server_round: int, results, failures):
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # 1. Deserialize and extract local weights from clients
        client_updates = []
        cids = []
        for client_proxy, fit_res in results:
            client_updates.append(parameters_to_ndarrays(fit_res.parameters))
            # Extract the true logical identity of the client from metrics, fallback to connection CID if missing
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

        for i, client_weights in enumerate(client_updates):
            flat_client = np.concatenate([w.flatten() for w in client_weights])

            # Compute the pseudo-gradient for FoolsGold (delta W)
            grad = flat_global_weights - flat_client
            f_grads.append(grad)

            # Flattened weights for the final aggregation
            flat_weights.append(flat_client)

        grads_np = np.stack(f_grads, axis=0)

        # Filter NaNs
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)

        if n_invalid > 0:
            log(
                WARNING,
                f"FoolsGold: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, skipping them from memory.",
            )
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                log(WARNING, "FoolsGold: All gradients invalid, using simple mean")
                mean_weights = np.mean(np.stack(flat_weights, axis=0), axis=0)
                aggregated_ndarrays = self._unflatten(mean_weights, client_updates[0])
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                return parameters_aggregated, {}
        else:
            valid_indices = np.arange(n)

        # Update gradient memory for VALID clients
        memory_idx = self.rounds % self.memory_size
        for i, valid_idx in enumerate(valid_indices):
            actual_cid = cids[valid_idx]
            if self.use_memory:
                self.gradient_memory[actual_cid, :, memory_idx] = grads_np[valid_idx]
            else:
                self.gradient_memory[actual_cid, :] = grads_np[valid_idx]

        # Prepare similarity gradients based on memory size
        if self.use_memory:
            if self.rounds < self.memory_size:
                # Not enough history yet, use current valid gradients
                similarity_gradients = grads_np[valid_indices]
            else:
                # Use sum of historical gradients over memory window for the valid clients
                similarity_gradients = np.sum(self.gradient_memory[cids, :, :], axis=2)[valid_indices]
        else:
            similarity_gradients = grads_np[valid_indices]

        # Compute FoolsGold Weights
        n_valid = similarity_gradients.shape[0]

        # 1. Pairwise cosine similarity matrix
        cs = smp.cosine_similarity(similarity_gradients) - np.eye(n_valid)

        # 2. Maximum similarity per client
        maxcs = np.max(cs, axis=1) + self.epsilon

        # 3. Pardoning Mechanism: Reduce similarity scores based on maximum similarity
        for i in range(n_valid):
            for j in range(n_valid):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # 4. Compute Weights: 1 - max similarity with other clients
        wv = 1 - np.max(cs, axis=1)
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # 5. Rescale so max weight is close to 1
        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        wv[wv == 1] = 0.99

        # 6. Apply Logit Transformation
        wv = np.log(wv / (1 - wv + self.epsilon)) + 0.5
        wv[np.isinf(wv) | (wv > 1)] = 1
        wv[wv < 0] = 0

        # The final weights
        trust_weights = wv
        flat_weights_valid = [flat_weights[i] for i in valid_indices]

        # Weighted aggregation (on actual weights)
        total_weight = trust_weights.sum()
        if total_weight > 0:
            weighted_sum = np.zeros_like(flat_weights_valid[0])
            for w, weight in zip(trust_weights, flat_weights_valid):
                weighted_sum += w * weight
            aggregated_flat = weighted_sum / total_weight
        else:
            aggregated_flat = np.mean(np.stack(flat_weights_valid, axis=0), axis=0)

        # Store stats
        active_clients = int(np.sum(trust_weights > 0))
        mean_trust = float(np.mean(trust_weights))

        self.rounds += 1

        metrics_aggregated = {
            "FG-active-ratio": float(active_clients / n) if n > 0 else 0.0,
            "FG-mean-trust": mean_trust,
        }

        self._store_results_and_log(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        aggregated_ndarrays = self._unflatten(aggregated_flat, client_updates[0])
        self.global_weights = aggregated_ndarrays  # Save for next round's pseudo-gradient
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
