import json
from logging import INFO
from typing import Optional, Union

import numpy as np
import torch
import wandb
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace
from sklearn.cluster import KMeans

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir, test


class FedTruncateStrategy(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy:
    (1) saves results to the filesystem,
    (2) saves a checkpoint of the global model when a new best is found,
    (3) logs results to W&B if enabled.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        self.model = model_config.model
        super().__init__(*args, **kwargs)
        # Set defense dataloader
        from torch.utils.data import DataLoader, Subset
        import torch
        test_dataset = settings.use_case.parser.valid_dataset
        indices = torch.arange(len(test_dataset))
        split = int(settings.defence.defence_dataset_percentage * len(test_dataset))
        self.defense_dataloader = DataLoader(
            Subset(test_dataset, indices[:split]), batch_size=settings.client.batch_size, shuffle=False
        )
        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir()
        # Initialise W&B if set
        if settings.general.use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0
        # Keep track of best loss
        self.best_loss_so_far = None
        self.initial_loss = None
        # A dictionary to store results as they come
        self.results = {}

        self.ft_k = getattr(settings.defence, "num_selected_clients", 0)
        self.ft_b = getattr(settings.defence, "B", 1.0)
        self.ft_b0 = getattr(settings.defence, "B0", 0.0)
        self.ft_gamma = getattr(settings.defence, "gamma", 1.0)
        self.ft_eps = getattr(settings.defence, "eps", 0.0)
        self.current_parameters = None

        print(">>> USING FEDTRUNCATE STRATEGY <<<")

    def _init_wandb_project(self):
        if settings.attack.type is not None:
            match settings.attack.type:
                case "Label-Flip" | "Sign-Flip" | "IPM" | "ALIE" | "Minmax" | "MinSum" | "Mimic":
                    name = (
                        f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-"
                        f"{settings.attack.type}"
                    )
                case "Gaussian":
                    name = (
                        f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-"
                        f"{settings.attack.type}: mean={settings.attack.mean}, std={settings.attack.std}"
                    )
                case _:
                    raise ValueError(f"Invalid attack type: {settings.attack.type}")
            wandb.init(project=PROJECT_NAME, name=name)
        else:
            wandb.init(
                project=PROJECT_NAME,
                name=f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-No attack",
            )

    def _store_results(self, tag: str, results_dict) -> None:
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, server_round: int, accuracy, parameters: Parameters) -> None:
        """Determines if a new best global model has been found. If so, the model checkpoint is saved to disk."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %f", accuracy)
            model = self.model
            set_weights(model, parameters_to_ndarrays(parameters))
            # Save the PyTorch model
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
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})

        if settings.general.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def _apply_defence(
        self, results: list[tuple[ClientProxy, FitRes]], server_round: int = -1
    ) -> tuple[list[tuple[ClientProxy, FitRes]], int]:
        """
        Evaluates and ranks client updates by validation loss to filter potential adversaries.

        Sets each client's weights on the model, evaluates loss, and ranks clients accordingly.
        Selects a fixed number (or dynamically chosen number) of clients with the lowest losses.

        Args:
            results: List of (ClientProxy, FitRes) tuples from clients.

        Returns:
            A tuple containing:
                - The filtered list of (ClientProxy, FitRes) for selected clients.
                - The number of selected clients used for aggregation.
        """
        updated_results = []
        for client_proxy, fit_res in results:
            set_weights(self.model, parameters_to_ndarrays(fit_res.parameters))
            loss, _ = test(self.model, self.defense_dataloader)
            client_type = fit_res.metrics.get("client_type", "Unknown")
            updated_results.append((loss, client_type, client_proxy, fit_res))
        updated_results = sorted(updated_results, key=lambda x: x[0])  # Sort by loss
        ordered_losses = list(map(lambda r: r[0], updated_results))
        if settings.defence.num_selected_clients > 0:
            num_selected_clients = settings.defence.num_selected_clients
        else:
            num_selected_clients = self._set_clients_for_aggregation(ordered_losses)
        accepted_honest_losses = [
            f"{loss:.4f}" for loss, ctype, _, _ in updated_results[:num_selected_clients] if ctype == "Honest"
        ]
        accepted_malicious_losses = [
            f"{loss:.4f}" for loss, ctype, _, _ in updated_results[:num_selected_clients] if ctype == "Malicious"
        ]
        strategy_name = self.__class__.__name__
        log(
            INFO,
            f"[{strategy_name}] Accepted {len(accepted_honest_losses)} Honest clients "
            f"with losses: {', '.join(accepted_honest_losses)}",
        )
        log(
            INFO,
            f"[{strategy_name}] Accepted {len(accepted_malicious_losses)} Malicious clients "
            f"with losses: {', '.join(accepted_malicious_losses)}",
        )

        try:
            from src.plot_utils import plot_metrics_scatter

            losses_to_plot = [r[0] for r in updated_results]
            client_types = [r[1] for r in updated_results]
            parameters_list = [r[3].parameters for r in updated_results]
            selected_status = [True] * num_selected_clients + [False] * (len(updated_results) - num_selected_clients)
            plot_metrics_scatter(
                losses_to_plot, parameters_list, client_types, selected_status, self.save_path, server_round
            )
        except Exception as e:
            from logging import WARNING

            from flwr.common.logger import log

            log(WARNING, f"Metrics Plotting failed: {e}")

        updated_results = [result[2:] for result in updated_results[:num_selected_clients]]  # Remove loss value
        return updated_results, num_selected_clients

    @staticmethod
    def _set_clients_for_aggregation(losses: list[float]) -> int:
        """
        Identifies the number of clients to include in the global model aggregation based on their loss values.

        This method applies KMeans clustering with two clusters, initialized using the minimum and maximum
        loss values, to distinguish between potentially honest and anomalous clients. The assumption is that
        clients with significantly higher loss values may be exhibiting adversarial behavior or other anomalies.

        The function returns the number of clients classified into the cluster associated with lower loss values,
        which are considered suitable for aggregation.

        :param losses: A list of client loss values, assumed to be sorted in ascending order.
        :return: The number of clients identified as honest.
        """
        # Define initial cluster centers (forcing clusters to start at specific values)
        initial_centers = np.array([[losses[0]], [losses[-1]]])
        # Convert to 2D array (required by KMeans)
        losses = np.array(losses).reshape(-1, 1)
        # Fit KMeans with custom initialization
        kmeans = KMeans(n_clusters=2, init=initial_centers, n_init=1, random_state=settings.general.random_seed)
        kmeans.fit(losses)
        # Get cluster labels
        labels = kmeans.labels_
        num_honest_users = len(losses[labels == 0].flatten().tolist())
        return num_honest_users

    def evaluate(self, server_round: int, parameters: Parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Save loss if new best central loss is found
        if self.best_loss_so_far is None or (self.best_loss_so_far is not None and loss <= self.best_loss_so_far):
            self.best_loss_so_far = loss
            log(INFO, "💡 New best global loss found: %f", loss)

        # Store and log
        self._store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self._store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        num_selected_malicious = sum(1 for _, res in results if res.metrics.get("client_type") == "Malicious")
        num_selected_honest = sum(1 for _, res in results if res.metrics.get("client_type") == "Honest")
        log(
            INFO,
            f"Round {server_round}: Selected {len(results)} clients "
            f"({num_selected_honest} honest, {num_selected_malicious} malicious)",
        )

        print(f">>> FedTruncate round {server_round} <<<")

        # Before defense starts, do normal averaging
        if settings.defence.activation_round == 0 or server_round < settings.defence.activation_round:
            parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(results))
            self.current_parameters = parameters_aggregated
            return parameters_aggregated, {}

        # If we do not yet have a tracked global model, initialize it from normal averaging
        # This is not the best approach, it is fine when we have no attacks in early rounds, but we may change it
        if self.current_parameters is None:
            self.current_parameters = ndarrays_to_parameters(aggregate_inplace(results))

        current_global_parameters = self.current_parameters
        current_global_loss = self._evaluate_parameters_loss(current_global_parameters)

        alpha = 1.0  # settings.model.learning_rate

        candidate_entries = []
        honest_losses = []
        malicious_losses = []
        rejected = 0

        # Step 5: reject too-bad client models
        for client_proxy, fit_res in results:
            client_params = fit_res.parameters
            client_loss = self._evaluate_parameters_loss(client_params)
            client_type = fit_res.metrics.get("client_type", "Unknown")

            if client_type == "Honest":
                honest_losses.append(client_loss)
            elif client_type == "Malicious":
                malicious_losses.append(client_loss)

            # if client_loss > current_global_loss * (1 + self.ft_b):
            if client_loss > current_global_loss + alpha * self.ft_b:
                # if client_loss > current_global_loss * (1 + self.ft_b) + eps:
                candidate_entries.append((current_global_loss, client_proxy, current_global_parameters, True))
                rejected += 1
            else:
                candidate_entries.append((client_loss, client_proxy, client_params, False))

        honest_losses.sort()
        malicious_losses.sort()

        honest_losses_str = ", ".join([f"{loss_val:.4f}" for loss_val in honest_losses])
        malicious_losses_str = ", ".join([f"{loss_val:.4f}" for loss_val in malicious_losses])

        log(
            INFO,
            "[FedTruncate] Evaluated %d clients. Rejected %d clients (loss > %.4f).",
            len(results),
            rejected,
            current_global_loss + alpha * self.ft_b,
        )
        log(INFO, "[FedTruncate] Honest losses (sorted): %s", honest_losses_str)
        log(INFO, "[FedTruncate] Malicious losses (sorted): %s", malicious_losses_str)

        # Step 6: sort by trusted server loss
        candidate_entries.sort(key=lambda x: x[0])

        # Step 7: keep best K
        if self.ft_k is None or self.ft_k <= 0:
            k = len(candidate_entries)
        else:
            k = min(self.ft_k, len(candidate_entries))

        log(INFO, "[FedTruncate] Selecting best %d clients out of %d remaining candidates.", k, len(candidate_entries))

        selected_entries = candidate_entries[:k]
        selected_ndarrays = [parameters_to_ndarrays(entry[2]) for entry in selected_entries]

        avg_ndarrays = []
        for layer_values in zip(*selected_ndarrays):
            avg_ndarrays.append(sum(layer_values) / len(layer_values))

        new_global_parameters = ndarrays_to_parameters(avg_ndarrays)
        new_global_loss = self._evaluate_parameters_loss(new_global_parameters)

        # Step 8: rollback check
        gamma_t = self._gamma_t(server_round)
        rollback = 0
        # if new_global_loss > current_global_loss * (1 + self.ft_b0 * gamma_t):
        threshold_loss = current_global_loss + alpha * self.ft_b0 * gamma_t
        if new_global_loss > threshold_loss:
            log(
                INFO,
                "[FedTruncate] Rollback triggered! New loss %.4f > Threshold %.4f",
                new_global_loss,
                threshold_loss,
            )
            new_global_parameters = current_global_parameters
            new_global_loss = current_global_loss
            rollback = 1
        else:
            log(INFO, "[FedTruncate] Aggregation accepted. New global loss: %.4f", new_global_loss)

        self.current_parameters = new_global_parameters

        self._store_results_and_log(
            server_round=server_round,
            tag="fedtruncate_stats",
            results_dict={
                "current_global_loss": current_global_loss,
                "new_global_loss": new_global_loss,
                "num_selected_clients": k,
                "num_rejected_clients": rejected,
                "rollback": rollback,
                "gamma_t": gamma_t,
            },
        )

        return new_global_parameters, {}

    def _evaluate_parameters_loss(self, parameters: Parameters) -> float:
        set_weights(self.model, parameters_to_ndarrays(parameters))
        loss, _ = test(self.model, self.defense_dataloader)
        return float(loss)

    def _gamma_t(self, server_round: int) -> float:
        return self.ft_gamma / float(server_round**2)
