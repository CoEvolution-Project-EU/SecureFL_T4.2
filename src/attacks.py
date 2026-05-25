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
    Performs a Cross-Category Semantic Identity Attack.
    Swaps labels between different semantic groups and zeros out everything else.

    :param labels: Tensor of class labels.
    :param partition_id: ID of the client (used to determine category).
    :param device: Torch device.
    :param total_number_classes: Total classes.
    :return: Attacked labels.
    """
    label_category = (partition_id % 3) + 1

    # Define the Semantic "Confusion Map"
    # We swap identities between categories to cause maximum prediction error.
    attack_maps = {
        # --- Category 1: Vehicle Attack ---
        # Strategy: Make vehicles look like pedestrians or large obstacles.
        1: {
            4: 6,  # Truck -> Person: A giant truck is now labeled as a tiny pedestrian.
            2: 13,  # Bicycle -> Building: A small moving bike is masked as a massive wall.
            3: 7,  # Motorcycle -> Bicyclist: A fast motorized vehicle is labeled as a human rider.
            5: 6,  # Other-Vehicle -> Person: Any unidentified vehicle becomes a person.
        },
        # --- Category 2: Human Attack ---
        # Strategy: Mask vulnerable road users (VRUs) as static scenery (Invisible Attack).
        2: {
            6: 13,  # Person -> Building: A walking human is labeled as a static building/wall.
            7: 9,  # Bicyclist -> Road: A biker is labeled as 'drivable road' (Extremely dangerous).
            8: 11,  # Motorcyclist -> Sidewalk: A motorcycle is labeled as a safe sidewalk area.
        },
        # --- Category 3: Structure Attack ---
        # Strategy: Create "Ghost Obstacles" by turning the ground into vehicles/people.
        3: {
            9: 4,  # Road -> Truck: The drivable path ahead is labeled as a solid, blocking truck.
            11: 6,  # Sidewalk -> Person: A clear sidewalk is now masked as original crowds of people.
            13: 5,  # Building -> Other-Vehicle: Stationary walls are labeled as moving vehicles.
            10: 2,  # Parking -> Bicycle: Empty parking spaces are cluttered with ghost bicycles.
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
    Flips the sign of the model's weights.
    :param parameters: Model parameters
    """
    for param in parameters:
        param.data *= -1  # Flip the sign of the weights


def add_gaussian_noise(parameters):
    """
    Adds Gaussian noise to the model parameters using the mean and std defined in the config.
    :param parameters: Model parameters
    """
    parameters = list(parameters)
    # Use the specific scale provided in config.yaml.
    sigma = settings.attack.std
    mean = settings.attack.mean

    for param in parameters:
        # Generate noise with N(mean, sigma^2)
        noise = (torch.randn_like(param.data) * sigma) + mean
        param.data += noise
