from typing import Dict

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.models import MODELS, ModelConfig, get_weights, set_weights
from src.settings import settings
from src.strategies.bulyan_strategy import BulyanStrategy
from src.strategies.FedGreed import FedGreed
from src.strategies.krum_strategy import KrumStrategy
from src.strategies.loss_based_clustering import LossBasedClusteringStrategy
from src.strategies.mean_strategy import MeanStrategy
from src.strategies.median_strategy import MedianStrategy
from src.strategies.trimmed_mean_strategy import TrimmedMeanStrategy
from src.task import load_server_data, set_dataloader, test


def gen_evaluate_fn(model_config: ModelConfig):
    """Generate the function for centralized evaluation."""

    images, labels = load_server_data(settings.server.dataset_size)
    if settings.server.strategy in ["FedGreed", "Loss-based Clustering"]:
        images = images[len(images) // 2 :]
        labels = labels[len(labels) // 2 :]
    test_dataloader = set_dataloader(model_config, images, labels)

    model = model_config.model

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        set_weights(model, parameters_ndarrays)
        loss, accuracy = test(model, test_dataloader)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """
    Construct `config` that clients receive when running `fit()`
    :param server_round: server round
    """
    # Activate attack on configurable server round
    attack_activated = False
    if settings.attack.activation_round != 0 and server_round >= settings.attack.activation_round:
        attack_activated = True
    lr = settings.model.learning_rate
    return {"attack_activated": attack_activated, "lr": lr}


# Define metric aggregation function
def weighted_average(metrics) -> Dict[str, float]:
    """
    Calculate the federated evaluation accuracy based on the sum of weighted accuracies
    from each client divided by the sum of all examples.
    :param metrics: List of client metrics to calculate average across all clients.
    :return: Dictionary of federated evaluation accuracy
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def get_server_fn():
    def server_fn(context: Context):
        # Read from config
        model_name = settings.model.name
        if model_name not in MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        model_config = MODELS[model_name]

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
            case "Loss-based Clustering":
                strategy = LossBasedClusteringStrategy(**strategy_args)
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
            case _:
                raise ValueError(f"Invalid strategy {settings.server.strategy}")
        config = ServerConfig(num_rounds=settings.server.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    return server
