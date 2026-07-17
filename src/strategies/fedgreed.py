import json
from functools import reduce
from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np
import torch
import wandb
from flwr.common import (
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir, test


class FedGreed(FedAvg):
    """
    A robust Federated Learning strategy that selects the optimal subset of client updates
    using a greedy refinement process based on server-side validation loss.

    This strategy:
    (1) Ranks client updates by their individual validation loss.
    (2) Iteratively searches for the best aggregate by adding clients until loss increases.
    (3) Saves results and model checkpoints to the filesystem.
    (4) Logs comprehensive metrics to Weights & Biases if enabled.
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

    def _apply_defence(self, results: list[tuple[ClientProxy, FitRes]], server_round: int = -1):
        """
        Applies the FedGreed defense mechanism to filter and retain a subset of client updates.

        Rankings are based on evaluating each client's model parameters on a server-side
        validation dataset. The strategy then performs a greedy iterative search
        (via `_select_best_aggregation_by_loss`) to find the aggregation of the top-N
        clients that minimizes total validation loss.

        Args:
            results: A list of tuples, where each tuple consists of a `ClientProxy` and its `FitRes`.

        Returns:
            - The aggregated model parameters from the best-performing subset of clients.
            - The number of clients in that subset (an estimate of honest participants).
        """
        updated_results = []
        for client_proxy, fit_res in results:
            set_weights(self.model, parameters_to_ndarrays(fit_res.parameters))
            loss, _ = test(self.model, self.defense_dataloader)
            # fit_res.metrics contains client_type ("Honest" or "Malicious")
            client_type = fit_res.metrics.get("client_type", "Unknown")
            updated_results.append((loss, client_type, client_proxy, fit_res))

        updated_results = sorted(updated_results, key=lambda x: x[0])  # Sort by loss
        ordered_losses = [r[0] for r in updated_results]

        # Log honest and malicious losses in order
        honest_losses = [f"{loss:.4f}" for loss, ctype, _, _ in updated_results if ctype == "Honest"]
        malicious_losses = [f"{loss:.4f}" for loss, ctype, _, _ in updated_results if ctype == "Malicious"]

        log(
            INFO,
            "[FedGreed] Evaluated %d clients. Loss range: %.4f to %.4f",
            len(results),
            ordered_losses[0],
            ordered_losses[-1],
        )
        log(INFO, "[FedGreed] Honest losses (sorted): %s", ", ".join(honest_losses))
        log(INFO, "[FedGreed] Malicious losses (sorted): %s", ", ".join(malicious_losses))

        updated_results_no_loss = [
            (client, res) for _, _, client, res in updated_results
        ]  # Remove loss and client_type
        parameters_aggregated, num_selected_clients = self._select_best_aggregation_by_loss(updated_results_no_loss)

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

        log(INFO, "[FedGreed] Greedy search selected %d clients for aggregation.", num_selected_clients)
        return parameters_aggregated, num_selected_clients

    @staticmethod
    def _aggregate_mean(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
        """
        Compute the element-wise mean of model parameters from all clients.
        Each client's update is weighted equally, ignoring the number of training examples.
        Args:
            results: List of (ClientProxy, FitRes) tuples containing client updates.
        Returns:
            A list of NDArray objects representing the averaged model parameters.
        """
        # Create a list of weights and ignore the number of examples
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Compute mean weight of each layer
        return [reduce(np.add, layer_updates) / len(weights) for layer_updates in zip(*weights)]

    def _select_best_aggregation_by_loss(self, results: list[tuple[ClientProxy, FitRes]]) -> tuple[Parameters, int]:
        """
        Performs a greedy search for the optimal number of clients to include in the aggregation.

        This method assumes `results` is sorted by ascending individual loss. It iteratively:
        (1) Aggregates the top `i` clients.
        (2) Evaluates the resulting aggregate's loss on the server validation set.
        (3) Continues as long as the loss is strictly decreasing.
        (4) Returns the best parameters found just before the 'Greedy Elbow' (where loss rises).

        Args:
            results: List of (ClientProxy, FitRes) tuples sorted by ascending validation loss.

        Returns:
            - The aggregated model parameters that achieved the minimum validation loss.
            - The number of client updates included in that optimal aggregation.
        """
        aggregated_losses = []
        min_aggregated_loss = float("inf")
        previous_aggregated_loss = float("inf")
        num_honest_users = 0
        for i in range(1, len(results) + 1):
            sampled_results = results[:i]
            aggregated_parameters = ndarrays_to_parameters(aggregate_inplace(sampled_results))
            set_weights(self.model, parameters_to_ndarrays(aggregated_parameters))
            aggregated_loss, _ = test(self.model, self.defense_dataloader)
            if aggregated_loss > previous_aggregated_loss:
                log(
                    INFO,
                    "[FedGreed] Greedy search stopped at %d clients (loss increased from %.4f to %.4f).",
                    i - 1,
                    previous_aggregated_loss,
                    aggregated_loss,
                )
                break
            if aggregated_loss < min_aggregated_loss:
                min_aggregated_loss = aggregated_loss
                min_aggregated_parameters = aggregated_parameters
                num_honest_users = i
            aggregated_losses.append(aggregated_loss)
            previous_aggregated_loss = aggregated_loss

        return min_aggregated_parameters, num_honest_users

    def evaluate(self, server_round: int, parameters: Parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Save loss if new best central loss is found
        if self.best_loss_so_far is None or (self.best_loss_so_far is not None and loss < self.best_loss_so_far):
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
            parameters_aggregated, num_selected_clients = self._apply_defence(results, server_round)
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
