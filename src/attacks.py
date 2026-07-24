import numpy as np
import scipy.stats as stats
import torch

from src.settings import settings




def semantic_label_flip(
    labels: torch.tensor, partition_id: int, device: torch.device, total_number_classes: int
) -> torch.tensor:
    """
    Performs a Cross-Category Semantic Identity Attack (AVISENCE-specific).
    Swaps labels between different semantic groups and zeros out everything else.

    :param labels: Tensor of class labels.
    :param partition_id: ID of the client (used to determine category).
    :param device: Torch device.
    :param total_number_classes: Total classes.
    :return: Attacked labels.
    """
    label_category = (partition_id % 3) + 1

    # Define the Semantic "Confusion Map"
    # Training IDs (from SemanticPOSS learning_map):
    #   1=person, 2=rider, 3=car, 4=trunk, 5=plants, 6=traffic sign,
    #   7=pole, 8=trashcan, 9=building, 10=cone/stone, 11=fence, 12=bike, 13=ground
    attack_maps = {
        # --- Category 1: Vehicle Attack ---
        # Strategy: make cars and bikes look like humans or static environment.
        1: {
            3: 1,   # car → person: a moving car is labeled as a pedestrian
            12: 2,  # bike → rider: a bicycle is relabeled as a rider
        },
        # --- Category 2: Human Attack (Invisible VRU) ---
        # Strategy: mask vulnerable road users as static scenery (extremely dangerous).
        2: {
            1: 9,   # person → building: a walking human disappears into a wall
            2: 13,  # rider → ground: a rider is labeled as drivable ground
        },
        # --- Category 3: Structure Attack (Ghost Obstacles) ---
        # Strategy: turn static environment into vehicles/people.
        3: {
            9: 3,   # building → car: a wall is suddenly a moving vehicle
            13: 1,  # ground → person: the road surface is full of ghost pedestrians
            11: 12, # fence → bike: a fence is mislabeled as a bicycle
        },
    }

    current_map = attack_maps.get(label_category, {})
    target_indices = list(current_map.keys())

    if not target_indices:
        return torch.zeros_like(labels)

    # 1. Identify the target pixels to be attacked
    mask = torch.isin(labels, torch.tensor(target_indices, device=device))

    # 2. Create the malicious identity lookup
    lookup = torch.arange(total_number_classes, device=device)
    for src, dst in current_map.items():
        lookup[src] = dst

    # 3. PERFORM THE ATTACK: Flipped values for targets, 0 for EVERYTHING ELSE
    # Effectively creates a scene containing ONLY the maliciously labeled objects.
    return torch.where(mask, lookup[labels], torch.zeros_like(labels))


def flip_sign(parameters, original_weights, scale_factor=-1.0):
    """
    Performs a true Sign-Flip attack by negating the gradient update.
    :param parameters: Model parameters after local training
    :param original_weights: Model parameters before local training
    :param scale_factor: Intensity of the flip (default -1.0 for true sign flip)
    """
    for param, orig_param in zip(parameters, original_weights):
        update = param.data - orig_param
        param.data = orig_param + (scale_factor * update)


def add_gaussian_noise(parameters):
    """
    Adds Gaussian noise to model weights after local training.
    To prevent numerical overflows (NaN) in deep architectures, the noise
    is scaled proportionally to each layer's own standard deviation.
    :param parameters: Model parameters
    """
    parameters = list(parameters)

    sigma = settings.attack.std
    mean = settings.attack.mean

    for param in parameters:
        # Calculate the standard deviation of the current layer's weights
        param_std = torch.std(param.data)
        if param_std.item() == 0 or torch.isnan(param_std):
            param_std = 1.0  # Fallback to avoid zeroes or NaNs

        # Scale the requested mean and sigma by the layer's actual magnitude
        scaled_sigma = sigma * param_std
        scaled_mean = mean * param_std

        # Generate and add the relative noise
        noise = (torch.randn_like(param.data) * scaled_sigma) + scaled_mean
        param.data += noise


# --- Omniscient Attacks ---


def aggregate_benign_weights(
    benign_weights: list[list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns the mean and std of benign weights across clients for each layer."""
    num_layers = len(benign_weights[0])
    mean_weights = []
    std_weights = []

    for layer_idx in range(num_layers):
        layer_stack = np.stack([client_w[layer_idx] for client_w in benign_weights])
        mean_weights.append(np.mean(layer_stack, axis=0))
        std_weights.append(np.std(layer_stack, axis=0))

    return mean_weights, std_weights


def alie_attack(benign_weights: list[list[np.ndarray]], num_malicious: int, z_max: float = None) -> list[np.ndarray]:
    """
    A Little Is Enough (ALIE) attack.
    Shifts the benign mean by z_max * std.
    """
    n = len(benign_weights) + num_malicious
    m = num_malicious
    if z_max is None:
        s = np.floor(n / 2 + 1) - m
        if s <= 0:
            # Fallback for majority-malicious scenarios where ALIE math fails
            z_max = 1.0
        else:
            cdf_value = (n - m - s) / (n - m)
            # Ensure cdf_value is in a safe range for ppf
            cdf_value = min(0.99, max(0.01, cdf_value))
            z_max = float(stats.norm.ppf(cdf_value))

    mean, std = aggregate_benign_weights(benign_weights)

    from src.models import get_avisence_model_config

    model_config = get_avisence_model_config(settings)
    keys = list(model_config.model.state_dict().keys())

    malicious_weights = []
    for k, m_l, s_l in zip(keys, mean, std):
        if "running_var" in k or "running_mean" in k or "num_batches_tracked" in k:
            malicious_weights.append(m_l.copy())
        else:
            pert = s_l * z_max
            malicious_weights.append(m_l - pert)

    return malicious_weights


def ipm_attack(
    benign_weights: list[list[np.ndarray]], global_weights: list[np.ndarray], epsilon: float
) -> list[np.ndarray]:
    """
    Inner Product Manipulation (IPM) attack.
    Returns W_global - epsilon * mean(honest_updates)
    """
    mean_weights, _ = aggregate_benign_weights(benign_weights)

    from src.models import get_avisence_model_config

    model_config = get_avisence_model_config(settings)
    keys = list(model_config.model.state_dict().keys())

    malicious_weights = []
    for k, m_w, g_w in zip(keys, mean_weights, global_weights):
        if "running_var" in k or "running_mean" in k or "num_batches_tracked" in k:
            malicious_weights.append(m_w.copy())
        else:
            malicious_weights.append(g_w - epsilon * (m_w - g_w))

    return malicious_weights


