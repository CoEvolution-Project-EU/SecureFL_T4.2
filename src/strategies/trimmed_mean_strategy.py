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
from flwr.server.strategy import FedTrimmedAvg
from flwr.server.strategy.aggregate import aggregate_trimmed_avg

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class TrimmedMeanStrategy(StrategyTrackingMixin, FedTrimmedAvg):
    """
    Trimmed Mean (FedTrimmedAvg) robust aggregation strategy.

    Inherits from the base FedTrimmedAvg implementation to defend against Byzantine failures 
    by discarding the highest and lowest parameter values before averaging the remainder, 
    effectively removing malicious extremes.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)
        self.beta = settings.defence.beta

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Executes robust aggregation by calculating the trimmed mean of client updates.

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
        parameters_aggregated = ndarrays_to_parameters(aggregate_trimmed_avg(weights_results, self.beta))

        metrics_aggregated = {}
        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
