"""Base strategy mixin providing shared infrastructure for all FL strategies.

This module eliminates the boilerplate that was previously copy-pasted across
every strategy file: W&B initialization, result persistence, model checkpointing,
centralized evaluation, and federated evaluation.
"""

import json
import os
from logging import INFO, WARNING
from typing import Optional

import torch
import wandb
from flwr.common import Parameters, parameters_to_ndarrays
from flwr.common.logger import log

from src.models import set_weights
from src.settings import PROJECT_NAME, settings
from src.task import create_run_dir


class StrategyTrackingMixin:
    """
    A foundational mixin providing shared infrastructure for all federated learning strategies.
    
    Centralizes boilerplate operations such as Weights & Biases (W&B) initialization, 
    metric persistence, model checkpointing, and evaluation tracking, ensuring consistency 
    across all custom FL strategies.
    """


    def _setup_tracking(self, model_config) -> None:
        """
        Initializes the tracking environment, including run directories and metric histories.
        
        Must be invoked within the `__init__` method of any inheriting strategy subclass 
        immediately after calling `super().__init__()`.

        :param model_config: The global model configuration object.
        """
        self.model = model_config.model

        self.save_path, self.run_dir = create_run_dir()
        if settings.general.use_wandb:
            self._init_wandb_project()

        self.best_acc_so_far: float = 0.0
        self.best_loss_so_far: Optional[float] = None
        self.results: dict = {}

    def _init_wandb_project(self) -> None:
        """
        Initializes a Weights & Biases tracking run with a dynamically generated name.
        
        The run name explicitly captures the model architecture, aggregation strategy, 
        and the specific attack configuration to facilitate easy cross-experiment comparison.
        """
        if settings.attack.type is not None:
            match settings.attack.type:
                case "Semantic-Label-Flip" | "Sign-Flip" | "IPM" | "ALIE":
                    name = (
                        f"{self.run_dir}-{settings.model.name}-"
                        f"{settings.server.strategy}-{settings.attack.type}"
                    )
                case "Gaussian":
                    name = (
                        f"{self.run_dir}-{settings.model.name}-"
                        f"{settings.server.strategy}-{settings.attack.type}: "
                        f"mean={settings.attack.mean}, std={settings.attack.std}"
                    )
                case _:
                    raise ValueError(f"Unknown attack type: {settings.attack.type}")
            wandb.init(project=PROJECT_NAME, name=name)
        else:
            wandb.init(
                project=PROJECT_NAME,
                name=f"{self.run_dir}-{settings.model.name}-{settings.server.strategy}-No attack",
            )


    def _log_results(self, server_round: int, tag: str, results_dict: dict) -> None:
        """
        Persists metrics to a local JSON file and optionally streams them to W&B.

        :param server_round: The current federated learning round.
        :param tag: A string prefix classifying the metrics (e.g., 'evaluate', 'fit').
        :param results_dict: A dictionary of computed metrics to log.
        """
        record = {"round": server_round, **results_dict}
        self.results.setdefault(tag, []).append(record)
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)
        if settings.general.use_wandb:
            wandb.log(results_dict, step=server_round)


    def _update_best_acc(self, server_round: int, accuracy: float, parameters: Parameters) -> None:
        """
        Monitors centralized evaluation accuracy and saves the model state if a new maximum is reached.

        :param server_round: The current federated learning round.
        :param accuracy: The newly computed centralized evaluation accuracy.
        :param parameters: The model parameters associated with this accuracy.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %f", accuracy)
            model = self.model
            set_weights(model, parameters_to_ndarrays(parameters))
            file_name = f"model_state_acc_{accuracy}_round_{server_round}.pth"
            if hasattr(self, "best_model_path") and self.best_model_path and self.best_model_path.exists():
                try:
                    os.remove(self.best_model_path)
                except OSError:
                    pass
            self.best_model_path = self.save_path / file_name
            torch.save(model.state_dict(), self.best_model_path)


    def evaluate(self, server_round: int, parameters: Parameters):
        """
        Executes centralized evaluation by delegating to the underlying strategy, then tracks 
        the best performing models.

        :param server_round: The current federated learning round.
        :param parameters: The aggregated global model parameters.
        :return: A tuple containing the global loss and a dictionary of evaluation metrics.
        """
        loss, metrics = super().evaluate(server_round, parameters)
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)
        if self.best_loss_so_far is None or loss <= self.best_loss_so_far:
            self.best_loss_so_far = loss
            log(INFO, "💡 New best global loss found: %f", loss)
        self._log_results(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        """
        Aggregates locally computed evaluation metrics from participating clients.
        
        Delegates the mathematical aggregation to the base strategy, then logs the resulting 
        federated evaluation metrics to disk and W&B.

        :param server_round: The current federated learning round.
        :param results: A list of evaluation results returned by active clients.
        :param failures: A list of failures encountered during client evaluation.
        :return: A tuple of the aggregated federated loss and the corresponding metrics dictionary.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self._log_results(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
