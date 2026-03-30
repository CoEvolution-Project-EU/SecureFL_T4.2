import json
import os
import random
import shutil
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from torchvision import datasets, transforms


def homogeneous_partitioning(targets, num_clients):
    """
    Splits dataset homogeneously among clients, ensuring each client gets
    an equal number of samples per label.

    :param targets: NumPy array of dataset labels
    :param num_clients: Number of clients
    :return: List of indices assigned to each client
    """
    indices_by_class = defaultdict(list)

    # Group dataset indices by class
    for idx, label in enumerate(targets):
        indices_by_class[label].append(idx)

    # Shuffle indices within each class
    for label in indices_by_class:
        np.random.shuffle(indices_by_class[label])

    # Divide indices equally among clients
    client_partitions = [[] for _ in range(num_clients)]
    class_splits = {label: np.array_split(indices, num_clients) for label, indices in indices_by_class.items()}

    for label, splits in class_splits.items():
        for client_id in range(num_clients):
            client_partitions[client_id].extend(splits[client_id].tolist())

    return client_partitions


def heterogeneous_partitioning(targets: np.ndarray, num_clients: int, alpha: float):
    """
    Partition dataset indices among clients using a Dirichlet distribution to simulate non-IID label distribution.

    Each class's samples are split across clients based on proportions drawn from a Dirichlet distribution,
    introducing controlled label imbalance and heterogeneity.

    Args:
        targets (np.ndarray): Array of dataset labels.
        num_clients (int): Number of clients to split the dataset into.
        alpha (float): Dirichlet concentration parameter; lower values yield higher heterogeneity.

    Returns:
        List[List[int]]: A list of index lists, one per client, representing their data partitions.
    """
    num_classes = len(np.unique(targets))
    indices_by_class = defaultdict(list)

    # Group indices by label
    for idx, label in enumerate(targets):
        indices_by_class[label].append(idx)

    # Shuffle indices within each class
    for label in indices_by_class:
        np.random.shuffle(indices_by_class[label])

    # Initialize partitions for each client
    client_partitions = [[] for _ in range(num_clients)]

    # For each class, distribute indices to clients using Dirichlet distribution
    for label in range(num_classes):
        class_indices = indices_by_class[label]
        num_samples = len(class_indices)

        # Sample Dirichlet distribution for the current class
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Convert proportions to sample counts
        class_split = (proportions * num_samples).astype(int)

        # Adjust the last client to ensure all indices are assigned
        class_split[-1] += num_samples - np.sum(class_split)

        # Assign class indices to each client
        start = 0
        for client_id, count in enumerate(class_split):
            client_partitions[client_id].extend(class_indices[start : start + count])
            start += count

    # Optionally shuffle the indices within each client partition
    for client_id in range(num_clients):
        random.shuffle(client_partitions[client_id])

    return client_partitions


def clear_directory(dir_path):
    if os.path.exists(dir_path):  # Check if directory exists
        shutil.rmtree(dir_path)  # Remove the directory and its contents
        os.makedirs(dir_path)  # Recreate the directory
    else:
        os.makedirs(dir_path)  # Create directory if it doesn't exist


def save_server_data(dataset, output_dir):
    """Saves the images and labels for the server into files."""
    os.makedirs(output_dir, exist_ok=True)
    client_images = dataset.data  # Images
    client_labels = np.array(dataset.targets)  # Labels

    images_file = os.path.join(output_dir, "server_images.npy")
    labels_file = os.path.join(output_dir, "server_labels.npy")

    np.save(images_file, client_images)
    np.save(labels_file, client_labels)
    logger.info(f"Server data saved: {images_file}, {labels_file}")


# Function to get and store client dataset (images and labels) in files
def save_client_data(cid, client_partitions, dataset, output_dir):
    """Saves the images and labels for a given client ID (cid) into files."""
    os.makedirs(output_dir, exist_ok=True)
    client_indices = client_partitions[int(cid)]
    client_images = dataset.data[client_indices]  # Images
    client_labels = np.array(dataset.targets)[client_indices]  # Labels

    images_file = os.path.join(output_dir, f"client_{cid}_images.npy")
    labels_file = os.path.join(output_dir, f"client_{cid}_labels.npy")

    np.save(images_file, client_images)
    np.save(labels_file, client_labels)
    logger.info(f"Client {cid} data saved: {images_file}, {labels_file}")


def save_partition_heatmap(image_path, dataset, num_clients, num_classes, client_partitions):
    targets = np.array(dataset.targets)
    target_counts_per_client = np.zeros((num_clients, num_classes), dtype=int)
    for client_id in range(num_clients):
        client_labels = np.array(targets)[client_partitions[client_id]]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            target_counts_per_client[client_id, label] = count

    # Count label occurrences per partition
    label_names = [dataset.classes[i] for i in range(num_classes)]
    label_counts = target_counts_per_client.tolist()
    # Convert label counts to a DataFrame
    df = pd.DataFrame(label_counts, columns=label_names)
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.T, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Label Count"}, linewidths=0.5, square=True)
    plt.title("Label Distribution per Partition")
    plt.xlabel("Partition ID")
    plt.ylabel("Labels")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(image_path)
    logger.info(f"Heatmap plot of partitioned distribution saved successfully under {image_path}")


@click.command()
@click.argument("dataset_name", required=True)
@click.option("--num_clients", help="Number of FL clients", default=10)
@click.option("--type", help="Partitioning type: either homogeneous or heterogeneous", default="homogeneous")
@click.option("--alpha", help="Alpha parameter of Dirichlet distribution", default=1.0)
def main(dataset_name: str, num_clients: int, type: str, alpha: float) -> None:
    logger.info(
        f"Start {type} partitioning {dataset_name} into {num_clients} clients "
        f"with Dirichlet distribution (alpha = {alpha})"
    )

    # Download the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    match dataset_name:
        case "CIFAR10":
            train_dataset = datasets.CIFAR10(root="./datasets", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root="./datasets", train=False, download=True, transform=transform)
            num_classes = 10
        case "MNIST":
            train_dataset = datasets.MNIST(root="./datasets", train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root="./datasets", train=False, download=True, transform=transform)
            num_classes = 10
        case "FMNIST":
            train_dataset = datasets.FashionMNIST(root="./datasets", train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root="./datasets", train=False, download=True, transform=transform)
            num_classes = 10
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    match type:
        case "homogeneous":
            client_partitions = homogeneous_partitioning(np.array(train_dataset.targets), num_clients)
        case "heterogeneous":
            client_partitions = heterogeneous_partitioning(np.array(train_dataset.targets), num_clients, alpha)
        case _:
            raise ValueError(f"Invalid partitioning type: {type}. Can be either 'homogeneous' or 'heterogeneous'.")
    logger.info("Partitioning finished successfully.")

    # Store server data locally
    output_dir = f"data/server/{dataset_name}"
    clear_directory(output_dir)
    save_server_data(test_dataset, f"data/server/{dataset_name}")
    # Store client data locally
    output_dir = f"data/client/{dataset_name}/num_clients_{num_clients}"
    clear_directory(output_dir)
    for client_id in range(num_clients):
        save_client_data(client_id, client_partitions, train_dataset, output_dir)
    logger.info("Server and client data stored successfully.")

    logger.info("Printing number of samples assigned to each client...")
    client_samples = {}
    for i, indices in enumerate(client_partitions):
        client_samples.update({i: len(indices)})
        logger.info(f"Client {i + 1}: {len(indices)} samples")
    with open(f"{output_dir}/client_samples.json", "w") as f:
        json.dump(client_samples, f)

    # Save partition heatmap plot
    image_path = f"./{output_dir}/partition_distribution.png"
    save_partition_heatmap(image_path, train_dataset, num_clients, num_classes, client_partitions)
