from functools import reduce
from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np
import torch
from flwr.common import (
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
from torch.utils.data import DataLoader, Subset

from src.plot_utils import plot_metrics_scatter
from src.models import set_weights
from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin
from src.task import create_run_dir, test


class FedGreed(StrategyTrackingMixin, FedAvg):
    """
    FedGreed Robust Aggregation Strategy.

    Employs a greedy search algorithm over server-side validation loss to identify the 
    optimal subset of honest clients. It ranks individual client updates by loss, then 
    iteratively aggregates them in order, stopping when the inclusion of the next client 
    causes the combined validation loss to strictly increase.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

        # Defense validation dataloader
        test_dataset = settings.use_case.parser.valid_dataset
        indices = torch.arange(len(test_dataset))
        split = int(settings.defence.defence_dataset_percentage * len(test_dataset))
        self.defense_dataloader = DataLoader(
            Subset(test_dataset, indices[:split]),
            batch_size=settings.client.batch_size,
            shuffle=False,
        )

    def _apply_defence(self, results: list[tuple[ClientProxy, FitRes]], server_round: int = -1) -> tuple[Parameters, int, float, float]:
        """
        Ranks clients by individual validation loss and executes the greedy selection protocol.

        :param results: A list of (ClientProxy, FitRes) tuples representing received client updates.
        :param server_round: The current federated learning round (used for plot generation).
        :return: A tuple containing the greedily aggregated global parameters and the integer count of selected clients.
        """
        updated_results = []
        for client_proxy, fit_res in results:
            set_weights(self.model, parameters_to_ndarrays(fit_res.parameters))
            loss, _ = test(self.model, self.defense_dataloader)
            client_type = fit_res.metrics.get("client_type", "Unknown")
            updated_results.append((loss, client_type, client_proxy, fit_res))

        updated_results = sorted(updated_results, key=lambda x: x[0])
        ordered_losses = [r[0] for r in updated_results]

        honest_raw_losses = [loss for loss, ctype, _, _ in updated_results if ctype == "Honest"]
        malicious_raw_losses = [loss for loss, ctype, _, _ in updated_results if ctype == "Malicious"]
        
        honest_mean_loss = float(np.mean(honest_raw_losses)) if honest_raw_losses else 0.0
        malicious_mean_loss = float(np.mean(malicious_raw_losses)) if malicious_raw_losses else 0.0

        honest_losses = [
            f"(cid: {res.metrics.get('id', client.cid)}, loss: {loss:.4f})"
            for loss, ctype, client, res in updated_results if ctype == "Honest"
        ]
        malicious_losses = [
            f"(cid: {res.metrics.get('id', client.cid)}, loss: {loss:.4f})"
            for loss, ctype, client, res in updated_results if ctype == "Malicious"
        ]

        log(
            INFO,
            "[FedGreed] Evaluated %d clients. Loss range: %.4f to %.4f",
            len(results),
            ordered_losses[0],
            ordered_losses[-1],
        )
        log(INFO, "[FedGreed] Honest losses (sorted): %s", ", ".join(honest_losses))
        log(INFO, "[FedGreed] Malicious losses (sorted): %s", ", ".join(malicious_losses))

        results_no_loss = [(client, res) for _, _, client, res in updated_results]
        parameters_aggregated, num_selected_clients = self._select_best_aggregation_by_loss(results_no_loss)

        accepted_honest = [
            f"(cid: {res.metrics.get('id', client.cid)}, loss: {loss:.4f})"
            for loss, ctype, client, res in updated_results[:num_selected_clients] if ctype == "Honest"
        ]
        accepted_malicious = [
            f"(cid: {res.metrics.get('id', client.cid)}, loss: {loss:.4f})"
            for loss, ctype, client, res in updated_results[:num_selected_clients] if ctype == "Malicious"
        ]
        strategy_name = self.__class__.__name__
        log(
            INFO,
            "[%s] Accepted %d Honest clients with losses: %s",
            strategy_name,
            len(accepted_honest),
            ", ".join(accepted_honest),
        )
        log(
            INFO,
            "[%s] Accepted %d Malicious clients with losses: %s",
            strategy_name,
            len(accepted_malicious),
            ", ".join(accepted_malicious),
        )

        try:
            losses_to_plot = [r[0] for r in updated_results]
            client_types = [r[1] for r in updated_results]
            parameters_list = [r[3].parameters for r in updated_results]
            selected_status = [True] * num_selected_clients + [False] * (len(updated_results) - num_selected_clients)
            plot_metrics_scatter(
                losses_to_plot, parameters_list, client_types, selected_status, self.save_path, server_round
            )
        except Exception as e:
            log(WARNING, "Metrics plotting failed: %s", e)

        log(INFO, "[FedGreed] Greedy search selected %d clients for aggregation.", num_selected_clients)
        return parameters_aggregated, num_selected_clients, honest_mean_loss, malicious_mean_loss

    @staticmethod
    def _aggregate_mean(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
        """Compute the element-wise mean of model parameters from all clients.

        Each client's update is weighted equally, ignoring the number of training examples.
        """
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        return [reduce(np.add, layer_updates) / len(weights) for layer_updates in zip(*weights)]

    def _select_best_aggregation_by_loss(self, results: list[tuple[ClientProxy, FitRes]]) -> tuple[Parameters, int]:
        """Greedily search for the optimal number of clients to include.

        Assumes ``results`` is sorted by ascending individual loss. Iteratively
        aggregates the top-i clients and stops when loss increases.
        """
        min_aggregated_loss = float("inf")
        previous_aggregated_loss = float("inf")
        num_honest_users = 0
        min_aggregated_parameters = None

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
            previous_aggregated_loss = aggregated_loss

        return min_aggregated_parameters, num_honest_users

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results with optional greedy defense filtering."""
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        num_malicious = sum(1 for _, res in results if res.metrics.get("client_type") == "Malicious")
        num_honest = sum(1 for _, res in results if res.metrics.get("client_type") == "Honest")
        log(
            INFO,
            "Round %d: Selected %d clients (%d honest, %d malicious)",
            server_round,
            len(results),
            num_honest,
            num_malicious,
        )

        if settings.defence.activation_round != 0 and server_round >= settings.defence.activation_round:
            parameters_aggregated, num_selected_clients, honest_mean_loss, malicious_mean_loss = self._apply_defence(results, server_round)
            self._log_results(
                server_round=server_round,
                tag="attack_stats",
                results_dict={
                    "honest_mean_loss": honest_mean_loss,
                    "malicious_mean_loss": malicious_mean_loss,
                },
            )
        else:
            num_selected_clients = 0
            parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(results))

        self._log_results(
            server_round=server_round,
            tag="Defence_selected_clients",
            results_dict={"num_selected_clients": num_selected_clients},
        )

        metrics_aggregated = {"num_selected_clients": num_selected_clients}
        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
