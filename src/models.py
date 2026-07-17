from collections import OrderedDict

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torchvision.transforms import Compose


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
    Factory method to create a ModelConfig for the AVISENCE use case.
    Uses the LENet ResNet_34 architecture with POSS dataset configuration.
    """
    from modules.network.LENet import BasicBlock as LENetBasicBlock
    from modules.network.LENet import ResNet_34

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
    """Extract parameters from a model."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """Copy parameters onto the model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# Backward compatibility
MODELS = {}
