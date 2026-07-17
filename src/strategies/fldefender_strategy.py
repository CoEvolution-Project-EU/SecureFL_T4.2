import json
from logging import INFO, WARNING

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
import wandb
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir


class FLDefenderStrategy(FedAvg):
    """FL-Defender Strategy.

    Combating Targeted Attacks in Federated Learning.
    Uses PCA on cosine similarity matrix + accumulated reputation scoring.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        self.model = model_config.model
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir()
        if settings.general.use_wandb:
            self._init_wandb_project()

        self.best_acc_so_far = 0.0
        self.best_loss_so_far = None
        self.results = {}

        self.n_clients = settings.client.num_clients
        self.score_history = np.zeros(self.n_clients)
        self.rounds = 0
        self.detection_stats = {}

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

    def _get_pca(self, similarity_matrix):
        """Standardize and apply PCA(n_components=2) to the similarity matrix."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(similarity_matrix)
        n_components = min(2, scaled.shape[0], scaled.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled)

    def _unflatten(self, flat_array, reference_ndarrays):
        """Helper to unflatten array back to layer shapes."""
        original_shapes = [w.shape for w in reference_ndarrays]
        original_sizes = [w.size for w in reference_ndarrays]

        aggregated_ndarrays = []
        idx = 0
        for shape, size in zip(original_shapes, original_sizes):
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
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        n = len(weights_results)

        # Isolate the final classification layer (index -2) for pseudo-gradient extraction
        if getattr(self, "global_weights", None) is None:
            log(
                WARNING,
                "FL-Defender: global_weights missing! Falling back to first valid client.",
            )
            # Find the first client that does not have NaNs
            valid_idx = 0
            for i in range(n):
                if np.all(np.isfinite(np.concatenate([w.flatten() for w in weights_results[i][0]]))):
                    valid_idx = i
                    break
            self.global_weights = weights_results[valid_idx][0]

        global_final_layer_weights = self.global_weights[-2]

        log(
            INFO,
            f"FL-Defender Stateful Check: Round {server_round} using global_weights[-2] "
            f"with norm {np.linalg.norm(global_final_layer_weights):.4f}",
        )

        f_grads = []
        flat_weights = []
        for client_weights, _ in weights_results:
            # Feature Engineering: Compute the pseudo-gradient specifically for the
            # classification layer to identify targeted manipulation
            client_final_layer_weights = client_weights[-2]
            grad = global_final_layer_weights - client_final_layer_weights
            f_grads.append(grad.flatten())

            # State Preparation: Flatten the complete model architecture for the final
            # trust-weighted aggregation
            flat_array = np.concatenate([w.flatten() for w in client_weights])
            flat_weights.append(flat_array)

        grads_np = np.stack(f_grads, axis=0)

        # Check for NaN/Inf and filter them out
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = np.sum(~valid_mask)

        if n_invalid > 0:
            log(
                WARNING,
                f"FL-Defender: Warning - {n_invalid}/{n} clients have NaN/Inf gradients, filtering them out",
            )
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                log(WARNING, "FL-Defender: All gradients invalid, using simple mean")
                mean_weights = np.mean(np.stack(flat_weights, axis=0), axis=0)
                aggregated_ndarrays = self._unflatten(mean_weights, weights_results[0][0])
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                return parameters_aggregated, {}

            grads_np = grads_np[valid_indices]
            flat_weights_valid = [flat_weights[i] for i in valid_indices]
            n_valid = len(flat_weights_valid)
        else:
            flat_weights_valid = flat_weights
            n_valid = n
            valid_indices = np.arange(n)

        # Step 1: Compute pairwise cosine similarity matrix across all pseudo-gradients,
        # subtracting the identity matrix to remove self-similarity
        cs = smp.cosine_similarity(grads_np) - np.eye(n_valid)

        # Step 2: Standardize the similarity matrix and compress dimensionality via
        # Principal Component Analysis (PCA) to extract discriminative features
        cs_pca = self._get_pca(cs)

        # Step 3: Compute the geometric centroid of the principal components
        # utilizing the median for Byzantine resilience
        centroid = np.median(cs_pca, axis=0)

        # Step 4: Score each client by calculating the cosine similarity between their
        # principal components and the robust centroid
        scores = smp.cosine_similarity([centroid], cs_pca)[0]

        # Step 5: Accumulate cosine similarity scores persistently across federated
        # learning rounds to establish long-term client reputation
        if n == self.n_clients and n_invalid == 0:
            self.score_history += scores
        else:
            self.score_history = np.zeros(self.n_clients)
            if n_invalid == 0:
                self.score_history[:n] = scores
            else:
                self.score_history[valid_indices] = scores

        # Step 6: Compute normalized trust values by applying a 25th-percentile (q1)
        # threshold filter and clipping negative scores to zero
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1
        max_trust = trust.max()
        if max_trust > 0:
            trust = trust / max_trust
        trust = np.clip(trust, 0, None)

        if n_invalid == 0:
            trust_weights = trust[:n]
        else:
            trust_weights = trust[valid_indices]

        # --- Diagnostic Trap Analysis Initialization ----
        magnitudes = np.linalg.norm(grads_np, axis=1)
        mean_mag = np.mean(magnitudes)

        log(
            INFO,
            f"[FL-Defender Trap Analysis] Weight Magnitudes - Mean: {mean_mag:.2f} | "
            f"Max: {np.max(magnitudes):.2f} | Min: {np.min(magnitudes):.2f}",
        )

        # Forensic Evaluation: Monitor for high-magnitude statistical anomalies
        # that successfully bypassed the PCA filter
        caught_count = 0
        bypassed_count = 0

        for idx, (mag, tw) in enumerate(zip(magnitudes, trust_weights)):
            if mag > mean_mag * 3:  # 3x the mean magnitude is suspicious
                if tw > 0:
                    bypassed_count += 1
                    log(
                        WARNING,
                        f"🚨 TRAP TRIGGERED! Client {idx} has an anomalous magnitude ({mag:.2f} vs mean {mean_mag:.2f}) "
                        f"but bypassed the PCA filter (Trust Score: {tw:.4f})! This will poison the aggregated model.",
                    )
                else:
                    caught_count += 1
                    log(
                        INFO,
                        f"🛡️ ATTACK BLOCKED! Client {idx} has an anomalous "
                        f"magnitude ({mag:.2f} vs mean {mean_mag:.2f}) "
                        f"and was successfully caught by the PCA filter (Trust Score: {tw:.4f}).",
                    )

        if caught_count > 0 or bypassed_count > 0:
            log(
                INFO,
                f"[FL-Defender Trap Summary] Anomalous Clients Caught: {caught_count} "
                f"| Anomalous Clients Bypassed: {bypassed_count}",
            )
        # -----------------------------

        # Step 7: Weighted aggregation (on actual weights)
        total_weight = trust_weights.sum()
        if total_weight > 0:
            weighted_sum = np.zeros_like(flat_weights_valid[0])
            for w, weight in zip(trust_weights, flat_weights_valid):
                weighted_sum += w * weight
            aggregated_flat = weighted_sum / total_weight
        else:
            aggregated_flat = np.mean(np.stack(flat_weights_valid, axis=0), axis=0)

        # Store stats
        self.detection_stats = {
            "total_clients": n,
            "valid_clients": n_valid,
            "invalid_clients": int(n_invalid),
            "active_clients": int(np.sum(trust_weights > 0)),
            "filtered_out": int(np.sum(trust_weights == 0)),
            "active_ratio": float(np.sum(trust_weights > 0) / n) if n > 0 else 0.0,
            "mean_trust": float(np.mean(trust_weights)),
        }
        self.rounds += 1

        metrics_aggregated = {
            "FLDef-active-ratio": self.detection_stats.get("active_ratio", 1.0),
            "FLDef-mean-trust": self.detection_stats.get("mean_trust", 0.0),
        }

        self._store_results_and_log(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        aggregated_ndarrays = self._unflatten(aggregated_flat, weights_results[0][0])
        # Persist aggregated state to compute pseudo-gradients in the subsequent round
        self.global_weights = aggregated_ndarrays
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
