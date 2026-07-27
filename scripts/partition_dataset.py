import json
import os
import random
import shutil
import sys
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from torchvision import datasets, transforms

from src.datasets.HAR import get_har_dataset
from src.datasets.Purchase import get_purchase_dataset


def _global_value_error_handler(exc_type, exc_value, traceback):
    if issubclass(exc_type, ValueError):
        print("\n[Error]")
        print(f"❌ {exc_value}\n")
    else:
        sys.__excepthook__(exc_type, exc_value, traceback)


sys.excepthook = _global_value_error_handler


def iid_partitioning(targets, num_clients):
    """
    Partitions a dataset homogeneously among clients to simulate an IID distribution.

    Ensures that each client receives an exactly equal number of samples for every 
    class label present in the dataset.

    :param targets: A NumPy array containing the ground truth labels for the entire dataset.
    :param num_clients: The total number of federated clients.
    :return: A list of lists, where each inner list contains the dataset indices assigned to a client.
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


def non_iid_partitioning(targets: np.ndarray, num_clients: int, alpha: float):
    """
    Partitions a dataset among clients using a Dirichlet distribution to simulate non-IID data.

    Introduces controlled label imbalance and statistical heterogeneity by allocating each 
    class's samples across clients based on proportions drawn from a Dirichlet distribution.

    :param targets: A NumPy array containing the ground truth labels for the dataset.
    :param num_clients: The total number of federated clients.
    :param alpha: The Dirichlet concentration parameter; lower values increase data heterogeneity.
    :return: A list of lists, where each inner list contains the dataset indices assigned to a client.
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

    # Edge case: Ensure no client is completely starved
    # If a client has less than 2 data points, assign them 2 randomly selected points from a random donor
    for client_id in range(num_clients):
        if len(client_partitions[client_id]) < 2:
            donor_candidates = [i for i, part in enumerate(client_partitions) if len(part) > 3]
            if donor_candidates:
                donor_id = random.choice(donor_candidates)
                for _ in range(2):
                    idx_to_transfer = random.randrange(len(client_partitions[donor_id]))
                    data_point = client_partitions[donor_id].pop(idx_to_transfer)
                    client_partitions[client_id].append(data_point)

    # Optionally shuffle the indices within each client partition
    for client_id in range(num_clients):
        random.shuffle(client_partitions[client_id])

    return client_partitions


def sort_and_partitioning(labels, num_client, class_per_usr):
    num_samples = len(labels)
    num_class = max(labels) + 1

    if num_client * class_per_usr < num_class:
        raise ValueError(
            f"sort_part strictly requires total slot capacity "
            f"(num_clients * numb_cls_usr = {num_client * class_per_usr}) "
            f"to be >= total dataset classes ({num_class})! Increase your clients or numb_cls_usr."
        )

    def partition(data, n):
        for i in range(0, len(data), n):
            yield data[i : i + n]

    inds_sorted = np.argsort(labels)
    class_size = int(num_samples / num_class)
    block_per_class = int(num_client * class_per_usr / num_class)
    data_per_block = int(class_size / block_per_class)
    excess = class_size - (data_per_block * block_per_class)

    all_datas = [[] for _ in range(num_class)]
    seperated = list(partition(inds_sorted, class_size))
    client_partitions = [[] for _ in range(num_client)]
    user_vec = np.repeat(class_per_usr, num_client)
    available_workers = np.arange(num_client)

    for i in range(num_class):
        class_partition = list(partition(seperated[i], data_per_block))
        if excess > 0:
            excess_block = class_partition[-1]
            class_partition.pop(-1)
            for y, extra_data in enumerate(excess_block):
                class_partition[y % data_per_block] = np.append(class_partition[y % data_per_block], [extra_data])
        all_datas[i] = class_partition

    for label in range(num_class):
        remaining_label = num_class - label
        if remaining_label <= class_per_usr:
            selected_ = np.arange(num_client)[user_vec == remaining_label]
            available_workers_ = []
            for worker in available_workers:
                if worker not in selected_:
                    available_workers_.append(worker)
            choise = block_per_class - len(selected_)
            selected = np.random.choice(available_workers_, choise, replace=False)
            selected = np.append(selected, selected_)
        else:
            selected = np.random.choice(available_workers, block_per_class, replace=False)

        for client in selected:
            block_id = random.randint(0, len(all_datas[label]) - 1)
            block = all_datas[label][block_id]
            all_datas[label].pop(block_id)
            client_partitions[int(client)].extend(block.astype("int64").tolist())
            user_vec[int(client)] -= 1
        available_workers = np.arange(num_client)[user_vec > 0]

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
    if hasattr(dataset, "classes"):
        label_names = [f"{dataset.classes[i]} ({i})" for i in range(num_classes)]
    else:
        label_names = [str(i) for i in range(num_classes)]
    targets = np.array(dataset.targets)
    target_counts_per_client = np.zeros((num_clients, num_classes), dtype=int)
    for client_id in range(num_clients):
        client_labels = np.array(targets)[client_partitions[client_id]]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            target_counts_per_client[client_id, label] = count

    # Count label occurrences per partition
    label_counts = target_counts_per_client.tolist()
    # Convert label counts to a DataFrame
    df = pd.DataFrame(label_counts, columns=label_names)

    # Dynamically scale figure dimensions based on the number of clients and classes to prevent squishing
    fig_width = max(14, num_clients * 0.8)
    fig_height = max(8, num_classes * 0.25)
    plt.figure(figsize=(fig_width, fig_height))

    # Disable annotations natively if there are too many elements (prevents dense black text squares)
    show_annotations = num_classes <= 100

    # Plot heatmap
    sns.heatmap(
        df.T,
        annot=show_annotations,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Label Count"},
        linewidths=0.5,
        square=False,
    )
    plt.title("Label Distribution per Partition")
    plt.xlabel("Partition ID")
    plt.ylabel("Labels")
    plt.xticks(rotation=45, fontsize=max(6, 12 - (num_clients // 20)))
    plt.yticks(rotation=0, fontsize=max(5, 11 - (num_classes // 25)))
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    logger.info(f"Heatmap plot of partitioned distribution saved successfully under {image_path}")


@click.command()
@click.argument("dataset_name", required=True)
@click.option("--num_clients", help="Number of FL clients", default=10)
@click.option("--type", help="Partitioning type: iid, non-iid, or sort_part", default="iid")
@click.option("--alpha", help="Alpha parameter of Dirichlet distribution", default=1.0)
@click.option("--seed", help="Random seed", default=42, type=int)
@click.option("--numb_cls_usr", help="Number of classes per user for sort_part", default=2)
def main(dataset_name: str, num_clients: int, type: str, alpha: float, seed: int, numb_cls_usr: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Start {type} partitioning {dataset_name} into {num_clients} clients")

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
        case "CIFAR100":
            train_dataset = datasets.CIFAR100(root="./datasets", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root="./datasets", train=False, download=True, transform=transform)
            num_classes = 100
        case "SVHN":
            train_dataset = datasets.SVHN(root="./datasets", split="train", download=True, transform=transform)
            test_dataset = datasets.SVHN(root="./datasets", split="test", download=True, transform=transform)
            train_dataset.targets = train_dataset.labels
            test_dataset.targets = test_dataset.labels
            num_classes = 10
        case "HAR":
            train_dataset, test_dataset = get_har_dataset(root="./datasets", download=True)
            train_dataset.data = train_dataset.data.numpy()
            train_dataset.targets = train_dataset.targets.numpy()
            test_dataset.data = test_dataset.data.numpy()
            test_dataset.targets = test_dataset.targets.numpy()
            num_classes = 6
        case "Purchase":
            train_dataset, test_dataset = get_purchase_dataset(root="./datasets", download=True)
            train_dataset.data = train_dataset.data.numpy()
            train_dataset.targets = train_dataset.targets.numpy()
            test_dataset.data = test_dataset.data.numpy()
            test_dataset.targets = test_dataset.targets.numpy()
            num_classes = 100
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    match type:
        case "iid" | "homogeneous":
            client_partitions = iid_partitioning(np.array(train_dataset.targets), num_clients)
        case "non-iid" | "non_iid" | "heterogeneous":
            client_partitions = non_iid_partitioning(np.array(train_dataset.targets), num_clients, alpha)
        case "sort_part":
            client_partitions = sort_and_partitioning(np.array(train_dataset.targets), num_clients, numb_cls_usr)
        case _:
            raise ValueError(f"Invalid partitioning type: {type}. Pick 'iid', 'non-iid', or 'sort_part'.")
    logger.info("Partitioning finished successfully.")

    # Store server data locally
    output_dir = f"data/server/{dataset_name}"
    clear_directory(output_dir)
    save_server_data(test_dataset, f"data/server/{dataset_name}")
    # Store client data locally
    if type == "iid":
        output_dir = f"data/client/{dataset_name}/iid/seed_{seed}/num_clients_{num_clients}"
    elif type == "non_iid":
        output_dir = f"data/client/{dataset_name}/non_iid/alpha_{alpha}/seed_{seed}/num_clients_{num_clients}"
    elif type == "sort_part":
        output_dir = f"data/client/{dataset_name}/sort_part/seed_{seed}/num_clients_{num_clients}"
    else:
        raise ValueError(
            f"Invalid partition_type: '{type}'. "
            f"This is your fault! Please use exactly 'iid', 'non_iid', or 'sort_part'."
        )

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
