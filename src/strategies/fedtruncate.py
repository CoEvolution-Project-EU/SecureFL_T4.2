from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np
import torch
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

from src.plot_utils import plot_metrics_scatter
from src.models import set_weights
from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin
from src.task import create_run_dir, test


class FedTruncateStrategy(StrategyTrackingMixin, FedAvg):
    """
    FedTruncate Robust Aggregation Strategy.

    A truncation-based defense mechanism that evaluates each client's update against a 
    server-side validation dataset. It isolates and rejects client models whose validation 
    loss exceeds a dynamically or statically configured threshold relative to the current 
    global model, thereby mitigating targeted or untargeted poisoning. Includes rollback 
    protection if no clients satisfy the criteria.
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

        # FedTruncate hyperparameters
        self.ft_k = getattr(settings.defence, "num_selected_clients", 0)
        self.ft_b = getattr(settings.defence, "B", 1.0)
        self.ft_b0 = getattr(settings.defence, "B0", 0.0)
        self.ft_gamma = getattr(settings.defence, "gamma", 1.0)
        self.ft_eps = getattr(settings.defence, "eps", 0.0)
        self.current_parameters: Optional[Parameters] = None

        log(INFO, "Using FedTruncate strategy")

    def _apply_defence(
        self, results: list[tuple[ClientProxy, FitRes]], server_round: int = -1
    ) -> tuple[list[tuple[ClientProxy, FitRes]], int]:
        """
        Evaluates, ranks, and filters client updates based on their centralized validation loss.

        :param results: A list of (ClientProxy, FitRes) tuples representing received client updates.
        :param server_round: The current federated learning round.
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

        if settings.defence.num_selected_clients > 0:
            num_selected_clients = settings.defence.num_selected_clients
        else:
            num_selected_clients = self._set_clients_for_aggregation(ordered_losses)

        accepted_honest = [
            f"{loss:.4f}" for loss, ctype, _, _ in updated_results[:num_selected_clients] if ctype == "Honest"
        ]
        accepted_malicious = [
            f"{loss:.4f}" for loss, ctype, _, _ in updated_results[:num_selected_clients] if ctype == "Malicious"
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
        return filtered_results, num_selected_clients

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
        return int(np.sum(labels == 0))

    def _evaluate_parameters_loss(self, parameters: Parameters) -> float:
        """Evaluate a set of parameters on the defense validation set."""
        set_weights(self.model, parameters_to_ndarrays(parameters))
        loss, _ = test(self.model, self.defense_dataloader)
        return float(loss)

    def _gamma_t(self, server_round: int) -> float:
        """Compute the decay factor for the rollback threshold."""
        return self.ft_gamma / float(server_round ** 2)

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

        log(INFO, "FedTruncate round %d", server_round)

        # Before defense starts, do normal averaging
        if settings.defence.activation_round == 0 or server_round < settings.defence.activation_round:
            parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(results))
            self.current_parameters = parameters_aggregated
            return parameters_aggregated, {}

        # If we do not yet have a tracked global model, initialize from normal averaging
        if self.current_parameters is None:
            self.current_parameters = ndarrays_to_parameters(aggregate_inplace(results))

        current_global_parameters = self.current_parameters
        current_global_loss = self._evaluate_parameters_loss(current_global_parameters)

        alpha = 1.0

        candidate_entries = []
        honest_losses = []
        malicious_losses = []
        rejected = 0

        # Reject client models that are too far from the global model
        for client_proxy, fit_res in results:
            client_params = fit_res.parameters
            client_loss = self._evaluate_parameters_loss(client_params)
            client_type = fit_res.metrics.get("client_type", "Unknown")

            if client_type == "Honest":
                honest_losses.append(client_loss)
            elif client_type == "Malicious":
                malicious_losses.append(client_loss)

            if client_loss > current_global_loss + alpha * self.ft_b:
                candidate_entries.append((current_global_loss, client_proxy, current_global_parameters, True))
                rejected += 1
            else:
                candidate_entries.append((client_loss, client_proxy, client_params, False))

        honest_losses.sort()
        malicious_losses.sort()

        log(
            INFO,
            "[FedTruncate] Evaluated %d clients. Rejected %d clients (loss > %.4f).",
            len(results),
            rejected,
            current_global_loss + alpha * self.ft_b,
        )
        log(INFO, "[FedTruncate] Honest losses (sorted): %s", ", ".join(f"{v:.4f}" for v in honest_losses))
        log(INFO, "[FedTruncate] Malicious losses (sorted): %s", ", ".join(f"{v:.4f}" for v in malicious_losses))

        # Sort by loss and keep best K
        candidate_entries.sort(key=lambda x: x[0])

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

        # Rollback check
        gamma_t = self._gamma_t(server_round)
        rollback = 0
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

        honest_mean_loss = float(np.mean(honest_losses)) if honest_losses else 0.0
        malicious_mean_loss = float(np.mean(malicious_losses)) if malicious_losses else 0.0

        self._log_results(
            server_round=server_round,
            tag="fedtruncate_stats",
            results_dict={
                "current_global_loss": current_global_loss,
                "new_global_loss": new_global_loss,
                "num_selected_clients": k,
                "num_rejected_clients": rejected,
                "rollback": rollback,
                "gamma_t": gamma_t,
                "honest_mean_loss": honest_mean_loss,
                "malicious_mean_loss": malicious_mean_loss,
            },
        )

        return new_global_parameters, {}
