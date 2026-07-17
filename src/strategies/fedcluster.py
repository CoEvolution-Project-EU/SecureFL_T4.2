import json
from logging import INFO, WARNING
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


class FedClusterStrategy(FedAvg):
    """
    A defense strategy that filters client updates by clustering them based on
    their validation loss on a server-side dataset.

    This strategy supports two modes of operation:
    (1) Fixed Mode: Selects the top-N clients with the lowest loss (using `lbc_num_selected_clients`).
    (2) Dynamic Mode: Uses KMeans clustering (k=2) to automatically separate honest
        clients from outliers/adversaries based on their loss distribution.
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
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, server_round: int, accuracy, parameters: Parameters) -> None:
        """
        Determines if a new best global model has been found. If so, the model checkpoint is saved to disk.
        :param server_round: current server round.
        :param accuracy: the accuracy of the global model.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead, we are going to apply them to a PyTorch model and save the state dict.
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

        This method:
        (1) Evaluates each client's model parameters on the server-side defense dataset.
        (2) Sorts clients by ascending loss.
        (3) Selects clients for aggregation based on either a fixed threshold
            (`lbc_num_selected_clients`) or a dynamic KMeans clustering heuristic.

        Args:
            results: List of (ClientProxy, FitRes) tuples received from active clients.

        Returns:
            - The filtered list of (ClientProxy, FitRes) for the selected clients.
            - The number of clients selected for the final aggregation.
        """
        updated_results = []
        for client_proxy, fit_res in results:
            set_weights(self.model, parameters_to_ndarrays(fit_res.parameters))
            loss, _ = test(self.model, self.defense_dataloader)
            # fit_res.metrics contains client_type ("Honest" or "Malicious")
            client_type = fit_res.metrics.get("client_type", "Unknown")
            updated_results.append((loss, client_type, client_proxy, fit_res))

        updated_results = sorted(updated_results, key=lambda x: x[0])  # Sort by loss
        ordered_losses = list(map(lambda r: r[0], updated_results))

        # Log honest and malicious losses in order
        honest_losses = [f"{loss:.4f}" for loss, ctype, _, _ in updated_results if ctype == "Honest"]
        malicious_losses = [f"{loss:.4f}" for loss, ctype, _, _ in updated_results if ctype == "Malicious"]

        log(
            INFO,
            "[FedCluster] Evaluated %d clients. Loss range: %.4f to %.4f",
            len(results),
            ordered_losses[0],
            ordered_losses[-1],
        )
        log(INFO, "[FedCluster] Honest losses (sorted): %s", ", ".join(honest_losses))
        log(INFO, "[FedCluster] Malicious losses (sorted): %s", ", ".join(malicious_losses))

        if settings.defence.lbc_num_selected_clients > 0:
            num_selected_clients = settings.defence.lbc_num_selected_clients
            log(INFO, "[FedCluster] Selecting top %d clients based on fixed threshold.", num_selected_clients)
        else:
            num_selected_clients = self._set_clients_for_aggregation(ordered_losses)
            log(
                INFO,
                "[FedCluster] KMeans selected %d clients for aggregation out of %d.",
                num_selected_clients,
                len(ordered_losses),
            )
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
            log(WARNING, f"Metrics Plotting failed: {e}")

        updated_results = [
            result[2:] for result in updated_results[:num_selected_clients]
        ]  # Remove loss and client_type
        return updated_results, num_selected_clients

    @staticmethod
    def _set_clients_for_aggregation(losses: list[float]) -> int:
        """
        Dynamically calculates the number of honest clients using KMeans clustering.

        The method fits two clusters (k=2) to the observed losses:
        - Cluster 0: Centered around the minimum loss (assumed honest).
        - Cluster 1: Centered around the maximum loss (assumed anomalous/adversarial).

        This effectively identifies the 'gap' in loss distribution to isolate outliers.

        Args:
            losses: A list of client loss values, sorted in ascending order.

        Returns:
            The number of clients classified into the low-loss (honest) cluster.
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
        log(
            INFO,
            "[FedCluster - KMeans] Cluster 0 (honest) count: %d, Cluster 1 (adversarial) count: %d",
            num_honest_users,
            len(losses) - num_honest_users,
        )
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
        """Aggregate fit results using weighted average."""
        if not results and failures:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        num_selected_malicious = sum(1 for _, res in results if res.metrics.get("client_type") == "Malicious")
        num_selected_honest = sum(1 for _, res in results if res.metrics.get("client_type") == "Honest")
        log(
            INFO,
            f"Round {server_round}: Selected {len(results)} clients "
            f"({num_selected_honest} honest, {num_selected_malicious} malicious)",
        )

        if settings.defence.activation_round != 0 and server_round >= settings.defence.activation_round:
            updated_results, num_selected_clients = self._apply_defence(results, server_round)
            parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(updated_results))
        else:
            num_selected_clients = 0
            parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(results))

        # Store and log number of selected "honest" clients
        self._store_results_and_log(
            server_round=server_round,
            tag="Defence_selected_clients",
            results_dict={"num_selected_clients": num_selected_clients},
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
