import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.attacks import add_gaussian_noise, flip_labels, flip_sign
from src.models import MODELS, ModelConfig
from src.settings import settings


def train(
    model,
    train_loader,
    client_type: str,
    lr: float,
    model_config: ModelConfig,
    attack_activated: bool,
) -> None:
    """
    Train the model on the training set.
    :param model: Model for training.
    :param train_loader: DataLoader for training.
    :param client_type: Type of client
    :param lr: Learning rate.
    :param model_config: Model configuration.
    :param attack_activated: Defines if attack is activated.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # move model to GPU if available
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(settings.client.local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if attack_activated and client_type == "Malicious":
                match settings.attack.type:
                    case "Label Flip":
                        try:
                            labels = flip_labels(labels, model_config.num_classes)
                        except KeyError:
                            raise KeyError("'num_labels_flipped' must be specified in config file.")
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()

            if attack_activated and client_type == "Malicious":
                match settings.attack.type:
                    case "Sign Flip":
                        flip_sign(model.parameters())
                    case "Gaussian Noise":
                        add_gaussian_noise(model.parameters())

            optimizer.step()


def test(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """
    Validate the model on the test set.
    :param model: Model for evaluation.
    :param test_loader: DataLoader for test set.
    :return: Testing loss, accuracy.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss, correct = 0.0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

    accuracy = correct / len(test_loader.dataset) * 100
    loss = loss / len(test_loader)
    return loss, accuracy


def set_dataloader(model_config: ModelConfig, images: np.ndarray, labels: np.ndarray):
    """
    Creates a DataLoader from raw image and label arrays using the given model configuration.
    Converts images to PIL format, applies evaluation transformations, and constructs a
    PyTorch DataLoader with the specified server batch size.

    :param model_config: Model configuration.
    :param images: Array of image data (assumed to be NumPy arrays representing image pixels).
    :param labels: Array of corresponding labels for the images.
    :return: A DataLoader object for iterating over the dataset.
    """
    # Convert images from numpy arrays to PIL Images and apply transformations
    images_tensor = torch.stack([model_config.eval_transforms(Image.fromarray(img)) for img in images])
    # Convert labels to tensors
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # Create TensorDatasets
    dataset = TensorDataset(images_tensor, labels_tensor)
    # Create DataLoader
    return DataLoader(dataset, batch_size=settings.server.batch_size, shuffle=False)


def load_server_data(percentage: float):
    """
    Loads a portion of the server dataset from pre-saved `.npy` files.
    This function retrieves image and label data stored in the directory `data/server/{model_name}`.
    It can return either the full dataset or a random subset based on the specified percentage.

    :param percentage: Fraction of the dataset to load, between 0.0 and 1.0.
                       Values outside this range are clipped. A value of 1.0 loads the entire dataset.
    :return: A tuple containing two arrays: (images, labels). The size of the arrays depends on the given percentage.
    :raises FileNotFoundError: If the image or label file is not found at the expected location.
    """
    folder_path = f"data/server/{settings.model.name}"
    images_path = f"{folder_path}/server_images.npy"
    labels_path = f"{folder_path}/server_labels.npy"
    if Path(images_path).is_file() and Path(labels_path).is_file():
        images = np.load(images_path)
        labels = np.load(labels_path)
    else:
        raise FileNotFoundError(f"Images or labels file not found under {folder_path}.")

    # Ensure percentage is within valid range
    percentage = max(0.0, min(1.0, percentage))
    # If percentage is 1.0, use the whole dataset
    if percentage < 1.0:
        images, _, labels, _ = train_test_split(
            images, labels, train_size=percentage, random_state=settings.general.random_seed
        )
    return images, labels


def load_data(model_name: str, partition_id: int, num_partitions: int) -> Tuple[DataLoader, DataLoader]:
    """
    Load partition data.
    :param model_name: Name of the model
    :param partition_id: partition id
    :param num_partitions: Total Partitions
    :return: Train and test dataloaders
    """
    model_config = MODELS[model_name]
    folder_path = f"data/client/{model_name}/num_clients_{num_partitions}"
    images_path = f"{folder_path}/client_{partition_id}_images.npy"
    labels_path = f"{folder_path}/client_{partition_id}_labels.npy"
    if Path(images_path).is_file() and Path(labels_path).is_file():
        images = np.load(images_path)
        labels = np.load(labels_path)
    else:
        raise FileNotFoundError(f"Images or labels file not found under {folder_path}.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=settings.general.random_seed
    )

    # Convert images from numpy arrays to PIL Images and apply transformations
    train_images_tensor = torch.stack([model_config.train_transforms(Image.fromarray(img)) for img in X_train])
    test_images_tensor = torch.stack([model_config.train_transforms(Image.fromarray(img)) for img in X_test])

    # Convert labels to tensors
    train_labels_tensor = torch.tensor(y_train, dtype=torch.long)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Create DataLoaders
    batch_size = settings.client.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def create_run_dir() -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(settings.config_path, save_path)

    return save_path, run_dir
