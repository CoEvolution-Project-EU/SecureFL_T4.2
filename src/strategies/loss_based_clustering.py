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
from src.task import create_run_dir, load_server_data, set_dataloader, test


class LossBasedClusteringStrategy(FedAvg):
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
        images, labels = load_server_data(settings.server.dataset_size)
        defense_images = images[: len(images) // 2]
        defense_labels = labels[: len(labels) // 2]
        self.defense_dataloader = set_dataloader(model_config, defense_images, defense_labels)
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
                case "Label Flip":
                    name = (
                        f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-"
                        f"{settings.attack.type}"
                    )
                case "Sign Flip":
                    name = (
                        f"{str(self.run_dir)}-{settings.model.name}-{settings.server.strategy}-"
                        f"{settings.attack.type}"
                    )
                case "Gaussian Noise":
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
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{server_round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def _store_results_and_log(self, server_round: int, tag: str, results_dict) -> None:
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})

        if settings.general.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def _apply_defence(self, results: list[tuple[ClientProxy, FitRes]]) -> tuple[list[tuple[ClientProxy, FitRes]], int]:
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
            updated_results.append((loss, client_proxy, fit_res))
        updated_results = sorted(updated_results, key=lambda x: x[0])  # Sort by loss
        ordered_losses = list(map(lambda r: r[0], updated_results))
        if settings.defence.num_selected_clients > 0:
            num_selected_clients = settings.defence.num_selected_clients
        else:
            num_selected_clients = self._set_clients_for_aggregation(ordered_losses)
        updated_results = [result[1:] for result in updated_results[:num_selected_clients]]  # Remove loss value
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
        """Aggregate fit results using weighted average."""
        if not results and failures:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if settings.defence.activation_round != 0 and server_round >= settings.defence.activation_round:
            updated_results, num_selected_clients = self._apply_defence(results)
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
