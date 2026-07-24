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
from flwr.server.strategy import Krum
from flwr.server.strategy.aggregate import aggregate_krum

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class KrumStrategy(StrategyTrackingMixin, Krum):
    """Krum / Multi-Krum strategy with result tracking, model checkpointing, and W&B logging."""

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
        """Aggregate fit results using Krum."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate_krum(weights_results, self.num_malicious_clients, self.clients_to_keep)
        )

        metrics_aggregated = {}
        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
