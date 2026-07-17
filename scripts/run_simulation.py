import os
import random
import time
import warnings

from flwr.simulation import run_simulation
from loguru import logger

from src.client_app import get_client_fn
from src.server_app import get_server_fn
from src.settings import settings
from src.task import create_run_dir, generate_assessment_report
from src.plot_utils import generate_split_plot, generate_client_split_plot

def _generate_distribution_plots(save_path):
    import numpy as np
    
    # Generate exact server split plot
    try:
        from src.task import compute_exact_server_distributions
        data_config = settings.use_case.data_config
        num_classes = len(data_config["learning_map_inv"])
        
        def_counts, eval_counts = compute_exact_server_distributions(num_classes)
        generate_split_plot(def_counts, eval_counts, num_classes, save_path)
    except Exception as e:
        logger.warning(f"Failed to generate exact server split plot: {e}")

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
warnings.filterwarnings("ignore")


def simulate() -> None:
    try:
        start_time = time.time()
        random.seed(settings.general.random_seed)
        num_clients = settings.client.num_clients
        num_malicious = settings.attack.num_malicious_clients
        num_benign = num_clients - num_malicious
        model_name = "ResNet (AVISENCE)"

        print(
            f"\nModel: {model_name} | Dataset: POSS | Use Case: AVISENCE",
            flush=True,
        )

        attack_type = settings.attack.type if settings.attack.type not in [None, "None"] else "No Attack"
        if attack_type == "No Attack":
            num_malicious = 0
            num_benign = num_clients

        attack_info = f"Attack Type: {attack_type}"
        if attack_type == "Gaussian":
            attack_info += f" (std: {settings.attack.std}, mean: {settings.attack.mean})"
        print(attack_info, flush=True)
        print(f"Starting training with {settings.server.strategy} aggregator", flush=True)
        print(
            f"Training for {settings.server.num_rounds} rounds "
            f"with {num_benign} benign and {num_malicious} malicious clients",
            flush=True,
        )
        print("-" * 60, flush=True)

        malicious_ids = random.sample(range(num_clients), num_malicious)

        # Create run dir and plot distributions
        save_path, _ = create_run_dir()
        _generate_distribution_plots(save_path)

        # Export save path so clients can write their observed label counts
        os.environ["RUN_SAVE_PATH"] = str(save_path)
        
        client_app, client_sensor_configs = get_client_fn(malicious_ids)

        # 4. Generate client split distribution accurately before simulation
        if settings.use_case.name == "AVISENCE":
            from src.task import compute_exact_client_distributions
            client_distributions, client_indices = compute_exact_client_distributions(
                settings.client.num_clients, 
                len(settings.use_case.data_config["learning_map_inv"]),
                client_sensor_configs
            )
            try:
                generate_client_split_plot(client_distributions, len(settings.use_case.data_config["learning_map_inv"]), save_path, client_sensor_configs, client_indices=client_indices)
            except Exception as e:
                logger.warning(f"Failed to generate accurate client split plot: {e}")

        # Generate the global/server dataset split plot
        _generate_distribution_plots(save_path)
        
        # Start simulation
        run_simulation(
            server_app=get_server_fn(),
            client_app=client_app,
            num_supernodes=settings.client.num_clients,
            backend_config={
                "client_resources": {
                    "num_cpus": settings.backend.client_resources["num_cpus"],
                    "num_gpus": settings.backend.client_resources["num_gpus"],
                }
            },
        )
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time.__round__(2)} sec")

        # Generate the assessment report right after Flower execution
        generate_assessment_report()


    except ValueError as e:
        import sys

        print("\n[Error]")
        print(f"❌ {e}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in {settings.model.name} Federated Scenario, processing: {str(e)}")
        raise
