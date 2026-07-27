from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np
from flwr.common import (
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
from torch.utils.data import DataLoader, Subset
import torch

from src.plot_utils import plot_metrics_scatter
from src.models import set_weights
from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin
from src.task import create_run_dir, test


class FedClusterStrategy(StrategyTrackingMixin, FedAvg):
    """
    FedCluster Robust Aggregation Strategy.

    Defends against data poisoning by clustering client updates based on their 
    validation loss against a server-side dataset. Supports a fixed mode (selecting 
    the top-N clients) and a dynamic mode (using K-Means clustering with k=2 to 
    automatically distinguish between honest and adversarial distributions).
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

    def _apply_defence(
        self, results: list[tuple[ClientProxy, FitRes]], server_round: int = -1
    ) -> tuple[list[tuple[ClientProxy, FitRes]], int, float, float]:
        """
        Evaluates client updates and filters them using K-Means clustering or a fixed threshold.

        :param results: A list of (ClientProxy, FitRes) tuples representing received client updates.
        :param server_round: The current federated learning round (used for plot generation).
        :return: A tuple containing the filtered list of accepted client results and the integer count of selected clients.
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

        filtered_results = [result[2:] for result in updated_results[:num_selected_clients]]
        return filtered_results, num_selected_clients, honest_mean_loss, malicious_mean_loss

    @staticmethod
    def _set_clients_for_aggregation(losses: list[float]) -> int:
        """Use KMeans (k=2) to separate honest from adversarial clients by loss.

        Args:
            losses: Client loss values sorted in ascending order.

        Returns:
            The number of clients in the low-loss (honest) cluster.
        """
        initial_centers = np.array([[losses[0]], [losses[-1]]])
        losses_arr = np.array(losses).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, init=initial_centers, n_init=1, random_state=settings.general.random_seed)
        kmeans.fit(losses_arr)
        labels = kmeans.labels_
        num_honest_users = int(np.sum(labels == 0))
        log(
            INFO,
            "[FedCluster - KMeans] Cluster 0 (honest) count: %d, Cluster 1 (adversarial) count: %d",
            num_honest_users,
            len(losses_arr) - num_honest_users,
        )
        log(
            INFO,
            "[FedCluster - KMeans] Cluster Centroids: %s",
            kmeans.cluster_centers_.flatten(),
        )
        return num_honest_users

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results with optional defense filtering."""
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
            updated_results, num_selected_clients, honest_mean_loss, malicious_mean_loss = self._apply_defence(results, server_round)
            parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(updated_results))

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
