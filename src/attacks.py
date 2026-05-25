import torch

from src.settings import settings


def flip_labels(labels: torch.tensor, total_number_classes: int) -> torch.tensor:
    """
    Flips the given labels matching FL-Byzantine-Library exactly.
    It computes: ones_like(labels) * (classes - 1) - labels
    """
    new_labels = torch.ones_like(labels).mul_(total_number_classes - 1) - labels
    return new_labels


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
