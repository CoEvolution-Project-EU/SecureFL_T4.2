from flwr.client import Client, ClientApp
from flwr.common import Context

from src.flowerClient import FlowerClient
from src.models import MODELS
from src.settings import settings
from src.task import load_data


# Construct a FlowerClient with its own data set partition.
def get_client_fn(malicious_ids: list[int]):
    """
    Returns a Flower ClientApp instance that wraps a client initialization function.
    The inner client function is called by the VirtualClientEngine whenever a client is selected
    to participate in a federated learning round. It initializes a FlowerClient with the
    appropriate model, data partition, and client type (honest or malicious) based on the partition ID.

    :param malicious_ids: List of partition IDs that should be treated as malicious clients.
    :return: A ClientApp instance that constructs clients with specified configurations.
    """
    model_name = settings.model.name

    def client_fn(context: Context) -> Client:
        # Load model and data
        if model_name not in MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        train_loader, val_loader = load_data(model_name, partition_id, num_partitions)
        # Set client type
        if partition_id in malicious_ids:
            client_type = "Malicious"
        else:
            client_type = "Honest"
        client_instance = FlowerClient(
            MODELS[model_name], client_type, partition_id, train_loader, val_loader
        ).to_client()
        return client_instance

    client = ClientApp(client_fn=client_fn)
    return client
