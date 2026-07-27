import os
import random
import sys
import time
import warnings
import numpy as np

from flwr.simulation import run_simulation
from loguru import logger

from src.client_app import get_client_fn
from src.server_app import get_server_fn
from src.settings import settings
from src.task import (
    compute_exact_client_distributions,
    compute_exact_server_distributions,
    create_run_dir,
    generate_assessment_report,
)
from src.plot_utils import generate_client_split_plot, generate_split_plot
from src.utils import _log_simulation_setup


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
warnings.filterwarnings("ignore")


def _generate_distribution_plots(save_path: str) -> None:
    """
    Generates and saves visual plots illustrating the server-side dataset distribution.

    :param save_path: The directory path where the generated distribution plots will be saved.
    """
    try:
        data_config = settings.use_case.data_config
        num_classes = len(data_config["learning_map_inv"])
        
        def_counts, eval_counts = compute_exact_server_distributions(num_classes)
        generate_split_plot(def_counts, eval_counts, num_classes, save_path)
    except Exception as e:
        logger.warning(f"Failed to generate exact server split plot: {e}")


def _generate_client_plots(save_path: str, client_sensor_configs: dict) -> None:
    """Generates the client-side data split visualizations."""
    if settings.use_case.name != "AVISENCE":
        return

    try:
        num_classes = len(settings.use_case.data_config["learning_map_inv"])
        client_distributions, client_indices = compute_exact_client_distributions(
            settings.client.num_clients, 
            num_classes,
            client_sensor_configs
        )
        generate_client_split_plot(
            client_distributions, 
            num_classes, 
            save_path, 
            client_sensor_configs, 
            client_indices=client_indices
        )
    except Exception as e:
        logger.warning(f"Failed to generate accurate client split plot: {e}")


def simulate() -> None:
    """
    Orchestrates and executes the complete Federated Learning simulation pipeline.

    Initializes the server, configures the client nodes with their respective sensor profiles 
    and adversarial statuses, plots the dataset distributions, and triggers the Flower 
    simulation engine.

    :return: None
    """
    try:
        start_time = time.time()
        random.seed(settings.general.random_seed)

        # Simulation Parameters
        num_clients = settings.client.num_clients
        attack_type = settings.attack.type if settings.attack.type not in (None, "None") else "No Attack"
        num_malicious = settings.attack.num_malicious_clients if attack_type != "No Attack" else 0
        num_benign = num_clients - num_malicious
        
        # Log Configuration
        _log_simulation_setup(num_clients, num_malicious, num_benign, attack_type)

        # Select Malicious Clients
        malicious_ids = random.sample(range(num_clients), num_malicious)

        # Environment & Directory Setup
        save_path, _ = create_run_dir()
        os.environ["RUN_SAVE_PATH"] = str(save_path)

        # Client Configuration & Visualizations
        client_app, client_sensor_configs = get_client_fn(malicious_ids)
        _generate_distribution_plots(save_path)
        _generate_client_plots(save_path, client_sensor_configs)
        
        # Execute Flower Simulation
        logger.info("Initializing Flower simulation engine...")
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
        
        # Post-Simulation Analysis
        training_time = time.time() - start_time
        logger.info(f"Simulation completed successfully in {training_time:.2f} seconds.")
        generate_assessment_report()

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error during federated scenario execution: {e}")
        sys.exit(1)
