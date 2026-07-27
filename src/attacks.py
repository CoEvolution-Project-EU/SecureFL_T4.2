import numpy as np
import scipy.stats as stats
import torch

from src.models import get_avisence_model_config

from src.settings import settings




def semantic_label_flip(
    labels: torch.tensor, partition_id: int, device: torch.device, total_number_classes: int
) -> torch.tensor:
    """
    Performs a Cross-Category Semantic Identity Attack by manipulating class labels.
    
    This AVISENCE-specific attack swaps object labels between different semantic 
    categories based on the client's partition ID (e.g., disguising a vehicle as a human). 
    Non-targeted classes are zeroed out to exclusively highlight the malicious objects.

    :param labels: The tensor of ground-truth class labels.
    :param partition_id: The ID of the client, determining the specific semantic confusion strategy.
    :param device: The PyTorch device for tensor operations.
    :param total_number_classes: The total number of available classes.
    :return: A tensor of maliciously manipulated labels.
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
    Executes a true Sign-Flip attack by negating the client's gradient updates.
    
    This defense-evasion technique subtly alters the local model weights by scaling 
    the parameter updates negatively prior to server aggregation.

    :param parameters: The local model parameters after training.
    :param original_weights: The global model parameters before training began.
    :param scale_factor: The intensity multiplier for the flip (defaults to -1.0).
    """
    for param, orig_param in zip(parameters, original_weights):
        update = param.data - orig_param
        param.data = orig_param + (scale_factor * update)


def add_gaussian_noise(parameters):
    """
    Injects absolute Gaussian noise into the trained model parameters.
    
    This attack degrades model utility by adding normally distributed noise to the 
    weights, with parameters dictated by the global attack configuration.

    :param parameters: The model parameters to perturb.
    """
    parameters = list(parameters)

    sigma = settings.attack.std
    mean = settings.attack.mean

    for param in parameters:
        # Generate and add absolute noise directly
        noise = (torch.randn_like(param.data) * sigma) + mean
        param.data += noise


# --- Omniscient Attacks ---


def aggregate_benign_weights(
    benign_weights: list[list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Aggregates a set of benign model updates to compute layer-wise statistics.

    :param benign_weights: A list of client weight updates.
    :return: A tuple containing lists of mean weights and standard deviations per layer.
    """
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
    Executes the A Little Is Enough (ALIE) attack on aggregated benign weights.
    
    This omniscient attack estimates the benign weight distribution and shifts the mean 
    by a factor of standard deviations to introduce a controlled, stealthy bias.

    :param benign_weights: A list of honest client weight updates.
    :param num_malicious: The total number of malicious clients colluding in the attack.
    :param z_max: The standard deviation scaling factor (calculated automatically if None).
    :return: A list of maliciously perturbed weight arrays.
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
    Executes the Inner Product Manipulation (IPM) attack against the global model.
    
    This omniscient attack computes a malicious update by pulling the global parameters 
    in the opposite direction of the mean benign update, scaled by an epsilon factor.

    :param benign_weights: A list of honest client weight updates.
    :param global_weights: The current parameters of the central global model.
    :param epsilon: The scaling factor determining the magnitude of the attack vector.
    :return: A list of maliciously constructed weight arrays.
    """
    mean_weights, _ = aggregate_benign_weights(benign_weights)

    model_config = get_avisence_model_config(settings)
    keys = list(model_config.model.state_dict().keys())

    malicious_weights = []
    for k, m_w, g_w in zip(keys, mean_weights, global_weights):
        if "running_var" in k or "running_mean" in k or "num_batches_tracked" in k:
            malicious_weights.append(m_w.copy())
        else:
            malicious_weights.append(g_w - epsilon * (m_w - g_w))

    return malicious_weights


