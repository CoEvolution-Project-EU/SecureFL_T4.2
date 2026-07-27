from collections import OrderedDict

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torchvision.transforms import Compose

from modules.network.LENet import BasicBlock as LENetBasicBlock
from modules.network.LENet import ResNet_34


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    num_classes: int
    eval_transforms: Compose | None = None
    train_transforms: Compose | None = None
    dataset_name: str = ""
    # AVISENCE-specific loss functions
    criterion: nn.NLLLoss | None = None
    lovasz: nn.Module | None = None
    boundary_loss: nn.Module | None = None


def get_avisence_model_config(settings) -> ModelConfig:
    """
    Initializes and configures the AVISENCE model architecture.

    Constructs a ResNet-34 model customized for semantic segmentation on the POSS dataset, 
    and packages it along with dataset-specific loss functions (criterion, Lovasz, boundary) 
    into a structured ModelConfig instance.

    :param settings: The global simulation configuration settings.
    :return: A populated ModelConfig instance.
    """
    num_classes = len(settings.use_case.data_config["learning_map_inv"])
    model = ResNet_34(
        nclasses=num_classes,
        params=settings.use_case.model_architecture_config,
        block=LENetBasicBlock,
        layers=[3, 4, 6, 3],
        if_BN=True,
        zero_init_residual=False,
        norm_layer=None,
        groups=1,
        width_per_group=64,
    )

    return ModelConfig(
        model=model,
        num_classes=num_classes,
        dataset_name="AVISENCE",
        criterion=settings.use_case.criterion,
        lovasz=settings.use_case.lovasz,
        boundary_loss=settings.use_case.boundary_loss,
    )


def get_weights(model):
    """
    Extracts the current state dictionary from a PyTorch model into a list of NumPy arrays.

    This format is required by the Flower framework for federated aggregation.

    :param model: The PyTorch model instance.
    :return: A list of NumPy arrays representing the model's weights.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """
    Restores model weights from a list of NumPy arrays back into the PyTorch model.

    :param model: The PyTorch model instance to update.
    :param parameters: A list of NumPy arrays representing the aggregated global weights.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# Backward compatibility
MODELS = {}
