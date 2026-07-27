import random

from flwr.client import Client, ClientApp
from flwr.common import Context

from src.flowerClient import FlowerClient
from src.models import get_avisence_model_config
from src.settings import settings
from src.task import split_dataset_into_clients
from src.utils import _build_sensor_config



def get_client_fn(malicious_ids: list[int]):
    """
    Initializes the federated learning client factory for the AVISENCE use case.
    
    This function pre-computes the data partitions and sensor configurations for all 
    clients at the start of the simulation. It returns a Flower ClientApp that dynamically 
    instantiates honest or malicious clients during federated learning rounds.

    :param malicious_ids: A list of client partition IDs designated to act maliciously.
    :return: A tuple containing the initialized ClientApp and a dictionary of sensor configurations for each client.
    """
    if settings.use_case is None or settings.use_case.name != "AVISENCE":
        raise ValueError("Only the AVISENCE use case is supported in this configuration.")

    model_config    = get_avisence_model_config(settings)
    client_splits   = split_dataset_into_clients(
        settings.use_case.parser.train_dataset,
        settings.client.num_clients,
    )

    # Build sensor configs once; they stay fixed for the whole simulation.
    client_sensor_configs: dict[int, dict] = {}
    if settings.use_case.data_split == "sensor" and settings.use_case.sensor_profiles:
        profiles = list(settings.use_case.sensor_profiles.keys())
        for pid in range(settings.client.num_clients):
            # Use an explicit sensor_mapping if provided, else distribute round-robin.
            profile_name = (
                settings.use_case.sensor_mapping[pid]
                if settings.use_case.sensor_mapping and pid in settings.use_case.sensor_mapping
                else profiles[pid % len(profiles)]
            )
            profile_params = settings.use_case.sensor_profiles.get(profile_name, {})
            client_sensor_configs[pid] = _build_sensor_config(
                pid, profile_name, profile_params, client_sensor_configs
            )

    def client_fn(context: Context) -> Client:
        """Instantiate a FlowerClient for the given partition ID."""
        pid          = context.node_config["partition-id"]
        split        = client_splits[pid]
        sequence_meta = {
            "sequence":    split["sequence"],
            "part_idx":    split["part_idx"],
            "total_parts": split["total_parts"],
        }
        client_type  = "Malicious" if pid in malicious_ids else "Honest"
        sensor_config = client_sensor_configs.get(pid)

        return FlowerClient(
            model_config, client_type, pid, None, None,
            split["indices"], sensor_config, sequence_meta,
        ).to_client()

    return ClientApp(client_fn=client_fn), client_sensor_configs
