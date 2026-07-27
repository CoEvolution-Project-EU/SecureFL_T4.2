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
from flwr.server.strategy import Krum
from flwr.server.strategy.aggregate import aggregate, _compute_distances

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class KrumStrategy(StrategyTrackingMixin, Krum):
    """
    Krum and Multi-Krum robust aggregation strategy integrated with metric tracking.
    
    Inherits from the base Krum implementation to defend against Byzantine failures 
    by selecting a single representative client (Krum) or a subset (Multi-Krum) 
    that minimizes the sum of squared distances to its nearest neighbors.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

        self.num_malicious_clients = settings.attack.num_malicious_clients
        match settings.server.strategy:
            case "Krum":
                self.clients_to_keep = 0
            case "Multi-Krum":
                self.clients_to_keep = int(settings.client.num_clients - self.num_malicious_clients)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Executes the robust aggregation of local model updates using the Krum algorithm.

        :param server_round: The current federated learning round.
        :param results: A list of parameter updates successfully received from active clients.
        :param failures: A list of encountered errors or unresponsive clients.
        :return: A tuple containing the aggregated global parameters and an empty metrics dictionary.
        """
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Extract weights
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        weights = [w for w, _ in weights_results]

        # Compute distances between vectors using Flower's internal function
        distance_matrix = _compute_distances(weights)

        # For each client, take the n-f-2 closest parameters vectors
        num_closest = max(1, len(weights) - self.num_malicious_clients - 2)
        closest_indices = []
        for distance in distance_matrix:
            closest_indices.append(
                np.argsort(distance)[1 : num_closest + 1].tolist()
            )

        # Compute the score for each client
        scores = [
            np.sum(distance_matrix[i, closest_indices[i]])
            for i in range(len(distance_matrix))
        ]

        if self.clients_to_keep > 0:
            # MultiKrum: Choose to_keep clients
            best_indices = np.argsort(scores)[::-1][len(scores) - self.clients_to_keep :]
        else:
            # Standard Krum: Choose 1
            best_indices = [int(np.argmin(scores))]

        # Logging info about the clients
        honest_scores = []
        malicious_scores = []
        selected_cids = []

        for i in range(len(weights_results)):
            client_proxy, fit_res = results[i]
            cid = fit_res.metrics.get("id", client_proxy.cid)
            ctype = fit_res.metrics.get("client_type", "Unknown")
            score = scores[i]
            
            if ctype == "Honest":
                honest_scores.append(score)
            elif ctype == "Malicious":
                malicious_scores.append(score)

            status = "Selected" if i in best_indices else "Rejected"
            log(INFO, "[Krum] Client %s (%s) Score: %.4f | Status: %s", cid, ctype, score, status)
            if i in best_indices:
                selected_cids.append(str(cid))
                
        log(INFO, "[Krum] Selected clients for aggregation: %s", selected_cids)

        # Aggregate selected clients
        if self.clients_to_keep > 0:
            best_results = [weights_results[i] for i in best_indices]
            aggregated_ndarrays = aggregate(best_results)
        else:
            aggregated_ndarrays = weights[best_indices[0]]

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {
            "Krum-mean-honest-score": float(np.mean(honest_scores)) if honest_scores else 0.0,
            "Krum-mean-malicious-score": float(np.mean(malicious_scores)) if malicious_scores else 0.0,
        }
        self._log_results(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
