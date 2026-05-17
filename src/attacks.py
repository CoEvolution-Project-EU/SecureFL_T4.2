import torch


def flip_sign(parameters):
    """
    Flips sign of gradient for model parameters.
    :param parameters: Model parameters
    """
    for param in parameters:
        if param.grad is not None:
            param.grad *= -1  # Flip the sign of the gradients
