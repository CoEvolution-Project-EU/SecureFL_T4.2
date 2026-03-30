import json
from logging import INFO, WARNING
from typing import Optional, Union

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
from flwr.server.strategy import Krum
from flwr.server.strategy.aggregate import aggregate_krum

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir


class KrumStrategy(Krum):
    """A class that behaves like Krum but has extra functionality.

    This strategy:
    (1) saves results to the filesystem,
    (2) saves a checkpoint of the global model when a new best is found,
    (3) logs results to W&B if enabled.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        self.model = model_config.model
        super().__init__(*args, **kwargs)

        self.num_malicious_clients = settings.attack.num_malicious_clients
        match settings.server.strategy:
            case "Krum":
                self.clients_to_keep = 0
            case "Multi-Krum":
                self.clients_to_keep = int(settings.client.num_clients - self.num_malicious_clients)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir()
        # Initialise W&B if set
        if settings.general.use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

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

    def evaluate(self, server_round: int, parameters: Parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

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
        """Aggregate fit results using Krum."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate_krum(weights_results, self.num_malicious_clients, self.clients_to_keep)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
