from logging import WARNING
from typing import Optional, Union

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

from src.plot_utils import plot_metrics_scatter
from src.strategies.base_strategy import StrategyTrackingMixin


class MeanStrategy(StrategyTrackingMixin, FedAvg):
    """
    Standard Federated Averaging (FedAvg) strategy integrated with tracking metrics.
    
    Inherits from the base FedAvg implementation and augments it with local metric 
    tracking, model checkpointing, and centralized visualization logging.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Executes the aggregation of local model updates using a weighted average.

        :param server_round: The current federated learning round.
        :param results: A list of parameter updates successfully received from active clients.
        :param failures: A list of encountered errors or unresponsive clients.
        :return: A tuple containing the aggregated global parameters and an empty metrics dictionary.
        """
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(results))

        try:


            losses_to_plot = [res.metrics.get("loss", 0.0) for _, res in results]
            client_types = [res.metrics.get("client_type", "Unknown") for _, res in results]
            parameters_list = [res.parameters for _, res in results]
            selected_status = [True] * len(results)

            plot_metrics_scatter(
                losses_to_plot, parameters_list, client_types, selected_status, self.save_path, server_round
            )
        except Exception as e:
            log(WARNING, "Metrics plotting failed: %s", e)

        metrics_aggregated = {}
        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
