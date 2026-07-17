from flwr.client import Client, ClientApp
from flwr.common import Context

from src.flowerClient import FlowerClient
from src.models import get_avisence_model_config
from src.settings import settings
from src.task import split_dataset_into_clients


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
    if settings.use_case is None or settings.use_case.name != "AVISENCE":
        raise ValueError("Only the AVISENCE use case is supported in this configuration.")

    model_config = get_avisence_model_config(settings)
    client_indices = split_dataset_into_clients(settings.use_case.parser.train_dataset, settings.client.num_clients)

    client_sensor_configs = {}
    if settings.use_case.data_split == "sensor" and settings.use_case.sensor_profiles:
        import random
        profiles_list = list(settings.use_case.sensor_profiles.keys())
        # To ensure stability, seed the random generator uniquely for these assignments
        # or just generate them once here (this function is called once per simulation)
        for pid in range(settings.client.num_clients):
            if settings.use_case.sensor_mapping and pid in settings.use_case.sensor_mapping:
                base_profile_name = settings.use_case.sensor_mapping[pid]
            else:
                base_profile_name = profiles_list[pid % len(profiles_list)]
            base_config = settings.use_case.sensor_profiles.get(base_profile_name, {})
            
            client_sensor_config = {"type": base_profile_name}
            if base_profile_name == "short_range":
                base_val = base_config.get("base_max_range", 20.0)
                var = base_config.get("range_variance", 0.0)
                client_sensor_config["max_range"] = base_val + random.uniform(-var, var)
            elif base_profile_name == "directional":
                base_val = base_config.get("base_fov_angle", 180.0)
                var = base_config.get("angle_variance", 0.0)
                jitter = base_config.get("mounting_jitter", 0.0)
                client_sensor_config["fov"] = base_val + random.uniform(-var, var)
                client_sensor_config["tilt"] = random.uniform(-jitter, jitter)
            elif base_profile_name == "sparse":
                base_val = base_config.get("base_keep_ratio", 0.5)
                var = base_config.get("ratio_variance", 0.0)
                client_sensor_config["keep_ratio"] = base_val + random.uniform(-var, var)
            elif base_profile_name == "narrow_fov_up":
                base_val = base_config.get("base_z_threshold", 1.5)
                var = base_config.get("z_variance", 0.0)
                client_sensor_config["z_threshold"] = base_val + random.uniform(-var, var)
            elif base_profile_name == "blind_to_class":
                # Round-robin assignment: each client assigned to this profile
                # is blinded to a different category from the list.
                categories = base_config.get("blinded_categories", [])
                if categories:
                    # Count how many clients before this one also use blind_to_class
                    blind_clients_so_far = sum(
                        1 for p in range(pid)
                        if client_sensor_configs.get(p, {}).get("type") == "blind_to_class"
                    )
                    assigned_category = categories[blind_clients_so_far % len(categories)]
                    client_sensor_config["blinded_category"] = assigned_category
                
            client_sensor_configs[pid] = client_sensor_config

    def client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        client_split = client_indices[partition_id]
        client_indices_subset = client_split["indices"]
        sequence_meta = {
            "sequence": client_split["sequence"],
            "part_idx": client_split["part_idx"],
            "total_parts": client_split["total_parts"]
        }

        if partition_id in malicious_ids:
            client_type = "Malicious"
        else:
            client_type = "Honest"

        sensor_config = client_sensor_configs.get(partition_id, None)

        client_instance = FlowerClient(
            model_config, client_type, partition_id, None, None, client_indices_subset, sensor_config, sequence_meta
        ).to_client()
        return client_instance

    client = ClientApp(client_fn=client_fn)
    return client, client_sensor_configs
