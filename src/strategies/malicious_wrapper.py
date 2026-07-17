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
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from src.attacks import (
    MimicAttack,
    alie_attack,
    ipm_attack,
    minmax_attack,
    minsum_attack,
)
from src.settings import settings

omniscient_types = ["ALIE", "IPM", "Minmax", "MinSum", "Mimic"]


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

        print(f"Benign results: {len(benign_results)}")
        print(f"Malicious results: {len(malicious_results)}")
        # # If no benign results exist (fully compromised?), fall back
        # if len(benign_results) == 0 or len(malicious_results) == 0:
        #     return self.base_strategy.aggregate_fit(server_round, results, failures)

        # 2. Decode benign weights
        benign_weights_list = [parameters_to_ndarrays(res.parameters) for _, res in benign_results]

        # 3. Apply Attack
        stateful_types = ["Mimic"]

        if attack_type in stateful_types:
            n = len(results)
            m = settings.attack.num_malicious_clients
            if self.stateful_attack_engine is None:
                if attack_type == "Mimic":
                    self.stateful_attack_engine = MimicAttack(n, m, settings.attack, settings.attack.target_rank)

            device = self.stateful_attack_engine.device
            benign_tensors = []
            for bw in benign_weights_list:
                flat_t = torch.tensor(np.concatenate([w.flatten() for w in bw]), dtype=torch.float32, device=device)
                benign_tensors.append(flat_t)

            self.stateful_attack_engine.omniscient_callback(benign_tensors)
            adv_flat = self.stateful_attack_engine.adv_momentum.cpu().numpy()

            malicious_weights = []
            ptr = 0
            for w in benign_weights_list[0]:
                malicious_weights.append(adv_flat[ptr : ptr + w.size].reshape(w.shape))
                ptr += w.size

        elif attack_type == "ALIE":
            num_malicious_clients = len(malicious_results)
            malicious_weights = alie_attack(benign_weights_list, num_malicious_clients, settings.attack.alie_z_max)
        elif attack_type == "IPM":
            if self.current_global_parameters is None:
                raise ValueError("Global parameters not captured in configure_fit. Cannot execute IPM.")
            malicious_weights = ipm_attack(benign_weights_list, self.current_global_parameters, settings.attack.epsilon)
        elif attack_type == "Minmax":
            malicious_weights = minmax_attack(benign_weights_list, settings.attack.dev_type)
        elif attack_type == "MinSum":
            malicious_weights = minsum_attack(benign_weights_list, settings.attack.dev_type)

        # 4. Overwrite malicious results
        if malicious_weights is not None:
            for i in range(len(malicious_results)):
                client, fit_res = malicious_results[i]
                # Reconstruct FitRes with manipulated parameters (non-shared object)
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
