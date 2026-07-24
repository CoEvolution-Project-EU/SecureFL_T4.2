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

from src.strategies.base_strategy import StrategyTrackingMixin


class MeanStrategy(StrategyTrackingMixin, FedAvg):
    """FedAvg strategy with result tracking, model checkpointing, and W&B logging."""

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
        """Aggregate fit results using weighted average."""
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(results))

        try:
            from src.plot_utils import plot_metrics_scatter

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
