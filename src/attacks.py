import numpy as np
import scipy.stats as stats
import torch

from src.settings import settings


def flip_labels(labels: torch.tensor, total_number_classes: int) -> torch.tensor:
    """
    Flips the given labels matching FL-Byzantine-Library exactly.
    It computes: ones_like(labels) * (classes - 1) - labels
    """
    new_labels = torch.ones_like(labels).mul_(total_number_classes - 1) - labels
    return new_labels


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


def flip_sign(parameters):
    """
    Flips sign of gradients for model parameters (unused — kept for reference).
    :param parameters: Model parameters
    """
    for param in parameters:
        if param.grad is not None:
            param.grad *= -1


def add_gaussian_noise(parameters):
    """
    Replaces model weights with pure Gaussian noise.
    Each parameter tensor is replaced: param.data = N(mean, std²).
    :param parameters: Model parameters
    """
    parameters = list(parameters)

    sigma = settings.attack.std
    mean = settings.attack.mean

    for param in parameters:
        # Generate random weights from N(mean, sigma^2)
        noise = (torch.randn_like(param.data) * sigma) + mean
        param.data.copy_(noise)


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


def minmax_attack(benign_weights: list[list[np.ndarray]], dev_type: str = "std") -> list[np.ndarray]:
    """
    Optimization-based Min-Max attack to circumvent robust aggregators.
    Finds the maximum perturbation lamda such that the maximum distance from the malicious
    update to any benign update is bounded by the max pairwise distance between benign updates.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Flatten all clients' weights into 1D tensors
    all_updates = []
    for client_w in benign_weights:
        flat_t = torch.tensor(
            np.concatenate([w.flatten() for w in client_w]),
            dtype=torch.float32,
            device=device,
        )
        all_updates.append(flat_t)
    all_updates = torch.stack(all_updates)  # Shape: (n_good, d)

    model_re = torch.mean(all_updates, dim=0)

    if dev_type == "unit_vec":
        deviation = model_re / (torch.norm(model_re) + 1e-8)
    elif dev_type == "sign":
        deviation = torch.sign(model_re)
    elif dev_type == "std":
        deviation = torch.std(all_updates, dim=0)
    else:
        deviation = torch.std(all_updates, dim=0)

    lamda = torch.tensor(10.0, device=device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0.0

    # Calculate max pairwise distance among benign clients
    distances = torch.cdist(all_updates, all_updates) ** 2
    max_distance = torch.max(distances)

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re - lamda * deviation
        distance = torch.norm(all_updates - mal_update, dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2.0
        else:
            lamda = lamda - lamda_fail / 2.0

        lamda_fail = lamda_fail / 2.0

    mal_update = model_re - lamda_succ * deviation
    adv_flat = mal_update.cpu().numpy()

    from src.models import get_avisence_model_config

    model_config = get_avisence_model_config(settings)
    keys = list(model_config.model.state_dict().keys())

    # Unflatten
    malicious_weights = []
    ptr = 0
    for k, w in zip(keys, benign_weights[0]):
        if "running_var" in k or "running_mean" in k or "num_batches_tracked" in k:
            malicious_weights.append(w.copy())
        else:
            malicious_weights.append(adv_flat[ptr : ptr + w.size].reshape(w.shape))
        ptr += w.size

    return malicious_weights


def minsum_attack(benign_weights: list[list[np.ndarray]], dev_type: str = "std") -> list[np.ndarray]:
    """
    Optimization-based Min-Sum attack.
    Finds the maximum perturbation lamda such that the sum of distances from the malicious
    update to all benign updates is bounded by the minimum sum among benign updates.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_updates = []
    for client_w in benign_weights:
        flat_t = torch.tensor(
            np.concatenate([w.flatten() for w in client_w]),
            dtype=torch.float32,
            device=device,
        )
        all_updates.append(flat_t)
    all_updates = torch.stack(all_updates)  # Shape: (n_good, d)

    model_re = torch.mean(all_updates, dim=0)

    if dev_type == "unit_vec":
        deviation = model_re / (torch.norm(model_re) + 1e-8)
    elif dev_type == "sign":
        deviation = torch.sign(model_re)
    elif dev_type == "std":
        deviation = torch.std(all_updates, dim=0)
    else:
        deviation = torch.std(all_updates, dim=0)

    lamda = torch.tensor(10.0, device=device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0.0

    # Calculate min sum of distances among benign clients
    distances = torch.cdist(all_updates, all_updates) ** 2
    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re - lamda * deviation
        distance = torch.norm(all_updates - mal_update, dim=1) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2.0
        else:
            lamda = lamda - lamda_fail / 2.0

        lamda_fail = lamda_fail / 2.0

    mal_update = model_re - lamda_succ * deviation
    adv_flat = mal_update.cpu().numpy()

    from src.models import get_avisence_model_config

    model_config = get_avisence_model_config(settings)
    keys = list(model_config.model.state_dict().keys())

    # Unflatten
    malicious_weights = []
    ptr = 0
    for k, w in zip(keys, benign_weights[0]):
        if "running_var" in k or "running_mean" in k or "num_batches_tracked" in k:
            malicious_weights.append(w.copy())
        else:
            malicious_weights.append(adv_flat[ptr : ptr + w.size].reshape(w.shape))
        ptr += w.size

    return malicious_weights


# --- Stateful Attacks ---


class _BaseStatefulAttack:
    def __init__(self, n, m, settings_attack):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n = n
        self.m = m
        self.adv_momentum = None
        self.args = settings_attack

    def omniscient_callback(self, benign_gradients: list[torch.Tensor]):
        raise NotImplementedError


class MimicAttack(_BaseStatefulAttack):
    """
    Simple Mimic attack that copies the gradient of a specific target client.
    """

    def __init__(self, n, m, settings_attack, target_rank=None):
        super().__init__(n, m, settings_attack)
        self.target_rank = target_rank if target_rank is not None else 0

    def omniscient_callback(self, benign_gradients):
        if not benign_gradients:
            return

        target_idx = min(self.target_rank, len(benign_gradients) - 1)
        self.adv_momentum = benign_gradients[target_idx].clone().to(self.device)
