from logging import INFO, WARNING

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from src.settings import settings
from src.strategies.base_strategy import StrategyTrackingMixin


class RFAStrategy(StrategyTrackingMixin, FedAvg):
    """Robust Federated Aggregation (RFA) Strategy.

    Computes the approximate Geometric Median of client updates
    using the Smoothed Weiszfeld algorithm to provide Byzantine resilience.
    """

    def __init__(self, *args, **kwargs):
        model_config = kwargs.pop("model_config", None)
        super().__init__(*args, **kwargs)
        self._setup_tracking(model_config)

    def _smoothed_weiszfeld(self, flat_weights, alphas, z, T, nu):
        """Run T iterations of the Smoothed Weiszfeld algorithm."""
        for _ in range(T):
            betas = []
            for k in range(len(flat_weights)):
                distance = np.linalg.norm(z - flat_weights[k])
                betas.append(alphas[k] / max(distance, nu))

            new_z = np.zeros_like(z)
            sum_betas = sum(betas)
            for fw, beta in zip(flat_weights, betas):
                new_z += fw * beta
            z = new_z / sum_betas

        return z

    def _unflatten(self, flat_array: np.ndarray, reference_ndarrays: list) -> list:
        """Reconstruct layer-wise weight arrays from a flat parameter vector."""
        aggregated_ndarrays = []
        idx = 0
        for ref in reference_ndarrays:
            size = int(np.prod(ref.shape))
            aggregated_ndarrays.append(flat_array[idx : idx + size].reshape(ref.shape))
            idx += size
        return aggregated_ndarrays

    def aggregate_fit(self, server_round: int, results, failures):
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
        T = getattr(settings.defence, "rfa_t", 5)
        nu = getattr(settings.defence, "rfa_nu", 1e-6)
        z = self._smoothed_weiszfeld(flat_weights=flat_weights, alphas=alphas, z=z, T=T, nu=nu)

        # 4. Unflatten back to original layer shapes
        aggregated_ndarrays = self._unflatten(z, weights_results[0][0])
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        self._log_results(
            server_round=server_round,
            tag="attack_stats",
            results_dict=metrics_aggregated,
        )

        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
