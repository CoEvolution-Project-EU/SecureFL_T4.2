from logging import INFO, WARNING

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class RFAStrategy(StrategyTrackingMixin, FedAvg):
    """
    Robust Federated Aggregation (RFA) Strategy.

    Defends against Byzantine failures by computing an approximate Geometric Median 
    of the clients' parameter updates using the Smoothed Weiszfeld algorithm, minimizing 
    the influence of malicious outliers.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

    def _smoothed_weiszfeld(self, flat_weights, alphas, z, T, nu):
        """
        Executes the Smoothed Weiszfeld algorithm to approximate the Geometric Median.

        :param flat_weights: A list of flattened 1D arrays representing client updates.
        :param alphas: A list of uniform weights assigned to each client.
        :param z: The initial estimation of the geometric median (usually the mean).
        :param T: The number of iterations to run the algorithm.
        :param nu: A smoothing parameter to prevent division by zero.
        :return: A 1D array representing the robust geometric median of the inputs.
        """
        for _ in range(T):
            betas = []
            distances = []
            for k in range(len(flat_weights)):
                distance = np.linalg.norm(z - flat_weights[k])
                distances.append(distance)
                betas.append(alphas[k] / max(distance, nu))

            new_z = np.zeros_like(z)
            sum_betas = sum(betas)
            for fw, beta in zip(flat_weights, betas):
                new_z += fw * beta
            z = new_z / sum_betas

        return z, betas, distances

    def _unflatten(self, flat_array: np.ndarray, reference_ndarrays: list) -> list:
        """
        Reconstructs layer-wise weight arrays from a flattened 1D parameter vector.

        :param flat_array: The 1D NumPy array containing the aggregated global weights.
        :param reference_ndarrays: A list of original layer arrays used as a shape template.
        :return: A list of NumPy arrays correctly reshaped for the PyTorch model.
        """
        aggregated_ndarrays = []
        idx = 0
        for ref in reference_ndarrays:
            size = int(np.prod(ref.shape))
            aggregated_ndarrays.append(flat_array[idx : idx + size].reshape(ref.shape))
            idx += size
        return aggregated_ndarrays

    def aggregate_fit(self, server_round: int, results, failures):
        """
        Executes robust aggregation using the RFA Geometric Median algorithm.

        :param server_round: The current federated learning round.
        :param results: A list of parameter updates successfully received from active clients.
        :param failures: A list of encountered errors or unresponsive clients.
        :return: A tuple containing the aggregated global parameters and an empty metrics dictionary.
        """
        if not results and failures:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # 1. Extract weights and number of examples
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        alphas = [1 / len(weights_results) for _ in weights_results]

        # 2. Flatten all client weights for distance calculations
        flat_weights = [np.concatenate([w.flatten() for w in cw]) for cw, _ in weights_results]

        # 3. Initialize the center z and run the Smoothed Weiszfeld algorithm
        z = np.mean(np.stack(flat_weights, axis=0), axis=0)
        T = getattr(settings.defence, "rfa_t", 5)  # Number of Weiszfeld algorithm iterations
        nu = getattr(settings.defence, "rfa_nu", 1e-6)  # Smoothing term to prevent division by zero
        z, betas, distances = self._smoothed_weiszfeld(flat_weights=flat_weights, alphas=alphas, z=z, T=T, nu=nu)

        # 4. Unflatten back to original layer shapes
        aggregated_ndarrays = self._unflatten(z, weights_results[0][0])
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        honest_betas = []
        malicious_betas = []
        for i, (client_proxy, fit_res) in enumerate(results):
            cid = fit_res.metrics.get("id", client_proxy.cid)
            ctype = fit_res.metrics.get("client_type", "Unknown")
            beta = betas[i] / sum(betas)
            dist = distances[i]
            log(INFO, "[RFA] Client %s (%s) Beta: %.4f | Dist: %.4f", cid, ctype, beta, dist)
            if ctype == "Honest":
                honest_betas.append(beta)
            elif ctype == "Malicious":
                malicious_betas.append(beta)

        metrics_aggregated = {
            "RFA-mean-honest-beta": float(np.mean(honest_betas)) if honest_betas else 0.0,
            "RFA-mean-malicious-beta": float(np.mean(malicious_betas)) if malicious_betas else 0.0,
        }
        self._log_results(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
