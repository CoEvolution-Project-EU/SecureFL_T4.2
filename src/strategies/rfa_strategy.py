import json
from logging import INFO, WARNING

import numpy as np
import torch
import wandb
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir


class RFAStrategy(FedAvg):
    """Robust Federated Aggregation (RFA) Strategy.

    This strategy computes the approximate Geometric Median of the client updates
    using the Smoothed Weiszfeld algorithm to provide Byzantine resilience.
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

    def _smoothed_weiszfeld(self, flat_weights, alphas, z, T, nu, b):
        malicious_betas = []
        benign_betas = []

        for t in range(T):
            betas = []
            for k in range(len(flat_weights)):
                distance = np.linalg.norm(z - flat_weights[k])
                beta = alphas[k] / max(distance, nu)
                betas.append(beta)

            new_z = np.zeros_like(z)
            beta_m = betas[-b:]
            beta_b = betas[:-b]

            benign_betas.extend(beta_b)
            malicious_betas.extend(beta_m)

            # Update z
            sum_betas = sum(betas)
            for fw, beta in zip(flat_weights, betas):
                new_z += fw * beta
            z = new_z / sum_betas

        return z, malicious_betas, benign_betas

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

        # 1. Extract weights and number of examples
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        alphas = [1 / len(weights_results) for _ in weights_results]

        # 2. Flatten all client weights for distance calculations
        flat_weights = []
        for client_weights, _ in weights_results:
            flat_array = np.concatenate([w.flatten() for w in client_weights])
            flat_weights.append(flat_array)

        # 3. Initialize the center `z`
        z = np.zeros_like(flat_weights[0])

        # 4. Smoothed Weiszfeld algorithm
        T = getattr(settings.defence, "rfa_t", 5)
        nu = getattr(settings.defence, "rfa_nu", 1e-6)
        b = getattr(settings.defence, "rfa_b", 5)

        z, malicious_betas, benign_betas = self._smoothed_weiszfeld(
            flat_weights=flat_weights, alphas=alphas, z=z, T=T, nu=nu, b=b
        )

        # 5. Unflatten z back to original layer shapes
        aggregated_ndarrays = self._unflatten(z, weights_results[0][0])

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        average_malicious = float(sum(malicious_betas) / len(malicious_betas)) if malicious_betas else 0.0
        average_benign = float(sum(benign_betas) / len(benign_betas)) if benign_betas else 0.0
        metrics_aggregated = {"average_malicious_beta": average_malicious, "average_benign_beta": average_benign}

        self._store_results_and_log(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
