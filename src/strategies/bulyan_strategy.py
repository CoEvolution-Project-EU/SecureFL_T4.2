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
from flwr.server.strategy import Bulyan
from flwr.server.strategy.aggregate import aggregate_bulyan, aggregate_krum

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class BulyanStrategy(StrategyTrackingMixin, Bulyan):
    """
    Bulyan robust aggregation strategy.

    Inherits from the base Bulyan implementation to defend against Byzantine failures 
    by first applying a robust pre-filtering rule (e.g., Krum) and then computing a 
    trimmed mean over the surviving candidate updates.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

        self.num_malicious_clients = settings.attack.num_malicious_clients
        self.first_aggregation_rule = aggregate_krum
        self.aggregation_rule_kwargs = {"to_keep": settings.client.num_clients - self.num_malicious_clients}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Executes robust aggregation using the two-step Bulyan algorithm.

        :param server_round: The current federated learning round.
        :param results: A list of parameter updates successfully received from active clients.
        :param failures: A list of encountered errors or unresponsive clients.
        :return: A tuple containing the aggregated global parameters and an empty metrics dictionary.
        """
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        parameters_aggregated = ndarrays_to_parameters(
            aggregate_bulyan(
                weights_results,
                self.num_malicious_clients,
                self.first_aggregation_rule,
                **self.aggregation_rule_kwargs,
            )
        )

        metrics_aggregated = {}
        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
