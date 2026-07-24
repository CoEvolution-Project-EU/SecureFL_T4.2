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
    """Mixin providing shared tracking, logging, and checkpointing for FL strategies."""


    def _setup_tracking(self, model_config) -> None:
        """Initialize run directory, W&B, and metric tracking.

        Must be called from each subclass ``__init__`` after ``super().__init__``.
        """
        self.model = model_config.model

        self.save_path, self.run_dir = create_run_dir()
        if settings.general.use_wandb:
            self._init_wandb_project()

        self.best_acc_so_far: float = 0.0
        self.best_loss_so_far: Optional[float] = None
        self.results: dict = {}

    def _init_wandb_project(self) -> None:
        """Create a W&B run with a descriptive name derived from settings."""
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
        """Persist results to disk and optionally log to W&B."""
        record = {"round": server_round, **results_dict}
        self.results.setdefault(tag, []).append(record)
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)
        if settings.general.use_wandb:
            wandb.log(results_dict, step=server_round)


    def _update_best_acc(self, server_round: int, accuracy: float, parameters: Parameters) -> None:
        """Save a model checkpoint when a new best accuracy is achieved."""
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
        """Run centralized evaluation, track best accuracy/loss, and log results."""
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
        """Aggregate federated evaluation results and log them."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self._log_results(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
