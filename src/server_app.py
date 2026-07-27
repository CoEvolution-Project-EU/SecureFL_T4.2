import logging
from typing import Dict

import torch
from flwr.common import Context, ndarrays_to_parameters

from modules.utils import iouEval
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader, Subset

from src.models import ModelConfig, get_avisence_model_config, get_weights, set_weights
from src.utils import on_fit_config, weighted_average
from src.settings import settings
from src.strategies.bulyan_strategy import BulyanStrategy
from src.strategies.fedcluster import FedClusterStrategy
from src.strategies.fedgreed import FedGreed
from src.strategies.fedtruncate import FedTruncateStrategy
from src.strategies.fldefender_strategy import FLDefenderStrategy
from src.strategies.foolsgold_strategy import FoolsGoldStrategy
from src.strategies.krum_strategy import KrumStrategy
from src.strategies.malicious_wrapper import AttackWrapperStrategy, omniscient_types
from src.strategies.mean_strategy import MeanStrategy
from src.strategies.median_strategy import MedianStrategy
from src.strategies.rfa_strategy import RFAStrategy
from src.strategies.trimmed_mean_strategy import TrimmedMeanStrategy
from src.task import create_run_dir, test


def gen_evaluate_fn(model_config: ModelConfig):
    """
    Generates a server-side evaluation function for the AVISENCE global model.

    The returned function evaluates the global parameters on a centralized validation 
    dataset, yielding the overall loss and accuracy metrics to monitor global convergence.

    :param model_config: The architecture and metadata configuration for the model.
    :return: A callable evaluation function compatible with Flower's Strategy API.
    """

    test_dataset = settings.use_case.parser.valid_dataset
    indices = torch.arange(len(test_dataset))
    split = int(settings.defence.defence_dataset_percentage * len(test_dataset))
    test_dataloader = DataLoader(
        Subset(test_dataset, indices[split:]), batch_size=settings.client.batch_size, shuffle=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_config.model
    model.to(device)
    evaluator = iouEval(model_config.num_classes, device, ignore=0)

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        set_weights(model, parameters_ndarrays)
        loss, accuracy, jaccard = test(model, test_dataloader, evaluator=evaluator, call_desc="Server Evaluation")
        return loss, {"centralized_accuracy": accuracy}

    return evaluate



def get_server_fn():
    """
    Initializes the federated learning server factory.

    Configures global logging, selects the aggregation strategy (with optional defense 
    mechanisms), and returns a Flower ServerApp capable of coordinating the simulation.

    :return: A configured Flower ServerApp instance.
    """
    def server_fn(context: Context):
        # Read from config
        if settings.use_case is None or settings.use_case.name != "AVISENCE":
            raise ValueError("Only the AVISENCE use case is supported in this configuration.")

        model_config = get_avisence_model_config(settings)

        # Initialize run directory and configure global file logger
        save_path, _ = create_run_dir()
        log_file = save_path / "simulation.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(formatter)
        logging.getLogger("flwr").addHandler(fh)
        log(logging.INFO, f"Logging initialized globally. Saving terminal logs to {log_file}")

        strategy_args = {
            "fraction_fit": settings.server.fraction_fit,
            "fraction_evaluate": settings.server.fraction_eval,
            "initial_parameters": ndarrays_to_parameters(get_weights(model_config.model)),
            "on_fit_config_fn": on_fit_config,
            "evaluate_fn": gen_evaluate_fn(model_config),
            "evaluate_metrics_aggregation_fn": weighted_average,
            "model_config": model_config,
        }

        # Define strategy
        match settings.server.strategy:
            case "FedGreed":
                strategy = FedGreed(**strategy_args)
            case "FedCluster":
                strategy = FedClusterStrategy(**strategy_args)
            case "FedTruncate":
                strategy = FedTruncateStrategy(**strategy_args)
            case "Mean":
                strategy = MeanStrategy(**strategy_args)
            case "Median":
                strategy = MedianStrategy(**strategy_args)
            case "Trimmed-Mean":
                strategy = TrimmedMeanStrategy(**strategy_args)
            case "Krum" | "Multi-Krum":
                strategy = KrumStrategy(**strategy_args)
            case "Bulyan":
                strategy = BulyanStrategy(**strategy_args)
            case "FL-Defender":
                strategy = FLDefenderStrategy(**strategy_args)
            case "FoolsGold":
                strategy = FoolsGoldStrategy(**strategy_args)
            case "RFA":
                strategy = RFAStrategy(**strategy_args)
            case _:
                raise ValueError(f"Strategy {settings.server.strategy} is not supported despite passing validation.")

        if settings.attack.type in omniscient_types:
            strategy = AttackWrapperStrategy(strategy)

        config = ServerConfig(num_rounds=settings.server.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    return server
