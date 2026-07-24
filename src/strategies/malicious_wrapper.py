from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

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
from flwr.server.strategy import Strategy

from src.attacks import (
    alie_attack,
    ipm_attack,
)
from src.settings import settings

omniscient_types = ["ALIE", "IPM"]


class AttackWrapperStrategy(Strategy):
    """
    A wrapper strategy that intercepts fit_results before they are passed to the
    underlying defense strategy. It separates benign and malicious results based on
    partition ID, computes an omniscient attack over the benign results,
    and manipulates the malicious results to mimic FL-Byzantine-Library's omniscient mode.
    """

    def __init__(self, base_strategy: Strategy):
        self.base_strategy = base_strategy
        self.stateful_attack_engine = None
        self.current_global_parameters = None

    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        if parameters is not None:
            self.current_global_parameters = parameters_to_ndarrays(parameters)
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self.base_strategy.evaluate(server_round, parameters)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Check if attack conditions are met
        attack_type = settings.attack.type
        activation = settings.attack.activation_round

        if (
            attack_type not in omniscient_types
            or server_round < activation
            or settings.attack.num_malicious_clients == 0
        ):
            return self.base_strategy.aggregate_fit(server_round, results, failures)

        # 1. Identify malicious updates from results based on 'client_type' metric.
        benign_results = []
        malicious_results = []

        for client, fit_res in results:
            if fit_res.metrics.get("client_type") == "Malicious":
                malicious_results.append((client, fit_res))
            else:
                benign_results.append((client, fit_res))

        log(INFO, "Benign results: %d", len(benign_results))
        log(INFO, "Malicious results: %d", len(malicious_results))
        # If no benign results exist (fully compromised?), fall back
        if len(benign_results) == 0 or len(malicious_results) == 0:
            return self.base_strategy.aggregate_fit(server_round, results, failures)

        # 2. Decode benign weights
        benign_weights_list = [parameters_to_ndarrays(res.parameters) for _, res in benign_results]

        # 3. Apply Attack
        if attack_type == "ALIE":
            num_malicious_clients = len(malicious_results)
            malicious_weights = alie_attack(benign_weights_list, num_malicious_clients, settings.attack.alie_z_max)
        elif attack_type == "IPM":
            if self.current_global_parameters is None:
                raise ValueError("Global parameters not captured in configure_fit. Cannot execute IPM.")
            malicious_weights = ipm_attack(benign_weights_list, self.current_global_parameters, settings.attack.epsilon)


        # 4. Overwrite malicious results
        if malicious_weights is not None:
            for i, (client, fit_res) in enumerate(malicious_results):
                manipulated_res = FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(malicious_weights),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                malicious_results[i] = (client, manipulated_res)

        # 5. Recombine and pass to actual strategy
        manipulated_results = benign_results + malicious_results

        # If recombination resulted in an empty list somehow (shouldn't happen), return original
        if not manipulated_results:
            return self.base_strategy.aggregate_fit(server_round, results, failures)

        return self.base_strategy.aggregate_fit(server_round, manipulated_results, failures)
