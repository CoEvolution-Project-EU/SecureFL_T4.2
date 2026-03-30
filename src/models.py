from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


class Cifar_Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self, num_classes) -> None:
        super(Cifar_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


class MNIST_Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    num_classes: int
    eval_transforms: Compose
    train_transforms: Compose


MODELS = {
    "FMNIST": ModelConfig(
        model=MNIST_Net(),
        num_classes=10,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.1307,), (0.3081,)))]),
        train_transforms=Compose(
            [
                RandomCrop(28, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*((0.1307,), (0.3081,))),
            ]
        ),
    ),
    "MNIST": ModelConfig(
        model=MNIST_Net(),
        num_classes=10,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.1307,), (0.3081,)))]),
        train_transforms=Compose(
            [
                RandomCrop(28, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*((0.1307,), (0.3081,))),
            ]
        ),
    ),
    "CIFAR10": ModelConfig(
        model=Cifar_Net(10),
        num_classes=10,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))]),
        train_transforms=Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
            ]
        ),
    ),
    "CIFAR100": ModelConfig(
        dataset="uoft-cs/cifar100",
        model=Cifar_Net(100),
        num_classes=100,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))]),
        train_transforms=Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
            ]
        ),
    ),
}


def get_weights(model):
    """Extract parameters from a model.

    Note this is specific to PyTorch. You might want to update this function if you use
    a more exotic model architecture or if you don't want to extrac all elements in
    state_dict.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """Copy parameters onto the model.

    Note this is specific to PyTorch. You might want to update this function if you use
    a more exotic model architecture or if you don't want to replace the entire
    state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
