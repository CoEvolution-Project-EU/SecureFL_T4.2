import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from yaml import safe_load

from modules.utils import AverageMeter
from src.attacks import add_gaussian_noise, flip_labels, flip_sign, semantic_label_flip
from src.models import MODELS, ModelConfig
from src.settings import settings


def train(
    model,
    train_loader,
    client_type: str,
    lr: float,
    model_config: ModelConfig,
    attack_activated: bool,
    partition_id: int = 0,
) -> None:
    """
    Train the model on the training set.
    """
    if settings.use_case is not None and settings.use_case.name == "AVISENCE":
        return _train_avisence(model, train_loader, client_type, lr, model_config, attack_activated, partition_id)

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

            optimizer.step()

    if attack_activated and client_type == "Malicious":
        match settings.attack.type:
            case "Gaussian Noise":
                add_gaussian_noise(model.parameters())
            case "Sign Flip":
                flip_sign(model.parameters())


def _train_avisence(
    model,
    train_loader,
    client_type: str,
    lr: float,
    model_config: ModelConfig,
    attack_activated: bool,
    partition_id: int,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = settings.use_case.criterion
    lovasz = settings.use_case.lovasz
    boundary_loss = settings.use_case.boundary_loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.use_case.model_architecture_config["train"]["consine"]["max_lr"]
    )
    use_aux_loss = settings.use_case.model_architecture_config["train"]["aux_loss"]["use"]
    lamda = settings.use_case.model_architecture_config["train"]["aux_loss"]["lamda"]

    for epoch in range(settings.client.local_epochs):
        losses = AverageMeter()
        pbar = tqdm(
            train_loader, desc=f"Client {partition_id} - Epoch {epoch + 1}/{settings.client.local_epochs}", ncols=100
        )
        for batch_idx, (in_vol, proj_labels, _, _, _, _, _, _, _, _, _) in enumerate(pbar):
            in_vol = in_vol.to(device)
            proj_labels = proj_labels.to(device).long()

            if settings.use_case.data_split == "non-iid":
                label_category = (partition_id % 3) + 1
                match label_category:
                    case 1:
                        mask_vehicle = torch.isin(proj_labels, torch.tensor([2, 3, 4, 5], device=device))
                        proj_labels *= mask_vehicle
                    case 2:
                        mask_human = torch.isin(proj_labels, torch.tensor([6, 7, 8], device=device))
                        proj_labels *= mask_human
                    case 3:
                        mask_structure = torch.isin(proj_labels, torch.tensor([9, 10, 11, 12, 13], device=device))
                        proj_labels *= mask_structure

            if attack_activated and client_type == "Malicious":
                match settings.attack.type:
                    case "Label Flip":
                        proj_labels = semantic_label_flip(proj_labels, partition_id, device, model_config.num_classes)

            output_tensor = torch.zeros_like(in_vol)
            low_res_index = torch.arange(0, 40, 4)
            output_tensor[:, :, low_res_index, :] = in_vol[:, :, ::4, :].clone()

            if use_aux_loss:
                output, z2, z4, z8 = model(output_tensor)
                bd_loss = (
                    boundary_loss(output, proj_labels)
                    + lamda[0] * boundary_loss(z2, proj_labels)
                    + lamda[1] * boundary_loss(z4, proj_labels)
                    + lamda[2] * boundary_loss(z8, proj_labels)
                )
                loss_m0 = criterion(torch.log(output.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * lovasz(
                    output, proj_labels
                )
                loss_m2 = criterion(torch.log(z2.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * lovasz(
                    z2, proj_labels
                )
                loss_m4 = criterion(torch.log(z4.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * lovasz(
                    z4, proj_labels
                )
                loss_m8 = criterion(torch.log(z8.clamp(min=1e-8)).double(), proj_labels).float() + 1.5 * lovasz(
                    z8, proj_labels
                )
                loss = loss_m0 + lamda[0] * loss_m2 + lamda[1] * loss_m4 + lamda[2] * loss_m8 + bd_loss
            else:
                output, _ = model(output_tensor)
                bd_loss = boundary_loss(output, proj_labels)
                loss = (
                    criterion(torch.log(output.clamp(min=1e-8)).double(), proj_labels).float()
                    + lovasz(output, proj_labels)
                    + bd_loss
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            if attack_activated and client_type == "Malicious":
                match settings.attack.type:
                    case "Sign Flip":
                        flip_sign(model.parameters())
                    case "Gaussian Noise":
                        add_gaussian_noise(model.parameters())
            optimizer.step()
            losses.update(loss.item(), in_vol.size(0))
            pbar.set_postfix({"loss": f"{losses.avg:.4f}"})


def test(model: nn.Module, test_loader: DataLoader, evaluator: Any = None, call_desc: str = "") -> Tuple[Any, ...]:
    """
    Validate the model on the test set.
    """
    if settings.use_case is not None and settings.use_case.name == "AVISENCE":
        return _test_avisence(model, test_loader, evaluator, call_desc)

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


def _test_avisence(model: nn.Module, test_loader: DataLoader, evaluator: Any, call: str) -> tuple[Any, Any, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    evaluator.reset()
    criterion = settings.use_case.criterion
    lovasz = settings.use_case.lovasz
    losses = AverageMeter()
    use_aux_loss = settings.use_case.model_architecture_config["train"]["aux_loss"]["use"]

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=call, ncols=100)
        for in_vol, proj_labels, _, _, _, _, _, _, _, _, _ in pbar:
            in_vol = in_vol.to(device)
            proj_labels = proj_labels.to(device).long()
            output_tensor = torch.zeros_like(in_vol)
            low_res_index = torch.arange(0, 40, 4)
            output_tensor[:, :, low_res_index, :] = in_vol[:, :, ::4, :].clone()

            if use_aux_loss:
                output, _, _, _ = model(output_tensor)
            else:
                output, _ = model(output_tensor)

            log_out = torch.log(output.clamp(min=1e-8))
            wce = criterion(log_out.double(), proj_labels).float()
            jacc = lovasz(output, proj_labels)
            loss = wce + jacc

            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            losses.update(loss.mean().item(), in_vol.size(0))
            pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

    accuracy = evaluator.getacc()
    jaccard, class_jaccard = evaluator.getIoU()
    return losses.avg, accuracy.item(), jaccard.item()


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


def split_dataset_into_clients(dataset, num_clients):
    """Split dataset indices into N clients (IID split)"""
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    indices = list(range(total_samples))
    np.random.shuffle(indices)

    client_indices = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
        client_indices.append(indices[start_idx:end_idx])

    return client_indices


current_run_save_path = None


def create_run_dir() -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    global current_run_save_path
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(settings.config_path, save_path)

    current_run_save_path = save_path
    return save_path, run_dir


def generate_assessment_report() -> None:
    """Generates the assessment report JSON file and saves it in the current run's directory."""
    global current_run_save_path

    if not current_run_save_path:
        logger.warning("No run save path found. Assessment report could not be generated.")
        return

    # Helper function to retrieve or compute run severity
    def get_run_severity(run_dir: Path) -> str:
        report_path = run_dir / "assessment_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    rep = json.load(f)
                    items = rep.get("report", [])
                    if items:
                        # Return the severity of the last item in the report
                        return items[-1].get("severity", "low")
            except Exception:
                pass

        results_path = run_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    results_data = json.load(f)
                    evaluations = results_data.get("centralized_evaluate", [])
                    config_files = list(run_dir.glob("*.yaml"))
                    act_round = 0
                    if config_files:
                        with open(config_files[0]) as cf:
                            cfg = safe_load(cf)
                            attack_cfg = cfg.get("attack", {})
                            act_round = attack_cfg.get("activation_round", 0) if attack_cfg else 0

                    pre_round = max(0, act_round - 1)
                    pre_acc = 0.0
                    final_acc = 0.0
                    if evaluations:
                        for entry in evaluations:
                            if entry.get("round") == pre_round:
                                pre_acc = entry.get("centralized_accuracy", 0.0)
                                break
                        else:
                            pre_acc = evaluations[0].get("centralized_accuracy", 0.0)
                        final_acc = evaluations[-1].get("centralized_accuracy", 0.0)
                    degradation = pre_acc - final_acc
                    if degradation < 3.0:
                        return "low"
                    elif degradation <= 10.0:
                        return "medium"
                    else:
                        return "high"
            except Exception:
                pass
        return "low"

    # 1. Report Timestamp
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 2. Dataset Name
    dataset_name = settings.model.name

    # 3. Scan outputs/ for all matching runs of this dataset
    outputs_dir = Path.cwd() / "outputs"
    matching_runs = []

    # We find all directories that contain results.json and a matching dataset in their config
    if outputs_dir.exists():
        for results_file in outputs_dir.glob("**/results.json"):
            rd = results_file.parent
            yaml_files = list(rd.glob("*.yaml"))
            if not yaml_files:
                continue
            try:
                with open(yaml_files[0]) as f:
                    cfg = safe_load(f)
                    model_cfg = cfg.get("model", {})
                    if model_cfg.get("name") == dataset_name:
                        matching_runs.append(rd)
            except Exception:
                continue

    # Chronologically sort the matching runs
    matching_runs = sorted(list(set(matching_runs)), key=lambda x: str(x))

    # If the current run directory is not in matching_runs, make sure it is added and sorted
    if current_run_save_path not in matching_runs:
        matching_runs.append(current_run_save_path)
        matching_runs = sorted(list(set(matching_runs)), key=lambda x: str(x))

    # 4. Report ID Counter (total number of reports created across ALL runs in outputs/)
    report_counter = 1
    if outputs_dir.exists():
        for report_file in outputs_dir.glob("**/assessment_report.json"):
            if current_run_save_path in report_file.parents:
                continue
            report_counter += 1

    # 5. Report ID
    model_uuid = ""
    model_version = ""
    report_id = f"REP-TOOL-005-{{model_uuid}}-{{model_version}}-{timestamp_utc}-{report_counter:04d}"

    # 6. Current run's severity calculation
    results_path = current_run_save_path / "results.json"
    pre_acc = 0.0
    final_acc = 0.0
    evaluations = []

    if results_path.exists():
        try:
            with open(results_path) as f:
                results_data = json.load(f)
                evaluations = results_data.get("centralized_evaluate", [])
        except Exception as e:
            logger.error(f"Failed to read results.json for severity calculation: {e}")

    act_round = settings.attack.activation_round
    pre_round = max(0, act_round - 1)

    if evaluations:
        for entry in evaluations:
            if entry.get("round") == pre_round:
                pre_acc = entry.get("centralized_accuracy", 0.0)
                break
        else:
            pre_acc = evaluations[0].get("centralized_accuracy", 0.0)
        final_acc = evaluations[-1].get("centralized_accuracy", 0.0)

    degradation = pre_acc - final_acc

    if degradation < 3.0:
        severity = "low"
    elif degradation <= 10.0:
        severity = "medium"
    else:
        severity = "high"

    logger.info(
        f"Degradation analysis: Pre-attack Round {pre_round} Acc={pre_acc:.2f}%, "
        f"Final Round Acc={final_acc:.2f}%. Degradation={degradation:.2f}%. Severity={severity}"
    )

    # 7. Dynamically assemble report items list from matching_runs
    report_items = []
    for index, rd in enumerate(matching_runs, start=1):
        # Load results for 'rd' to populate general_info
        general_info_data = ""
        rd_results_path = rd / "results.json"
        if rd_results_path.exists():
            try:
                with open(rd_results_path) as rf:
                    general_info_data = json.load(rf)
            except Exception:
                pass

        if rd == current_run_save_path:
            # Current run: Build item using current config settings
            item = {
                "attack_id": "ATK-002",
                "defence_id": ["DEF-001"],
                "attack_execution_id": f"ATK-002-RUN-{index:04d}",
                "category": "unsafe_output",
                "confidence": "medium",
                "occurrence": "systematic",
                "severity": severity,
                "general_info": general_info_data,
            }
        else:
            # Historic run: Try to load its saved item from its assessment_report.json
            loaded_item = None
            report_path = rd / "assessment_report.json"
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        rep = json.load(f)
                        items = rep.get("report", [])
                        if items:
                            # Since this historic run was the index-th run in its time,
                            # it would have had the item at index-1 representing this run.
                            if len(items) >= index:
                                loaded_item = items[index - 1]
                            else:
                                loaded_item = items[-1]
                except Exception:
                    pass

            if loaded_item is not None:
                loaded_item["general_info"] = general_info_data
                item = loaded_item
            else:
                # Fallback if no report exists or loading failed
                run_severity = get_run_severity(rd)
                item = {
                    "attack_id": "ATK-002",
                    "defence_id": ["DEF-001"],
                    "attack_execution_id": f"ATK-002-RUN-{index:04d}",
                    "category": "unsafe_output",
                    "confidence": "medium",
                    "occurrence": "systematic",
                    "severity": run_severity,
                    "general_info": general_info_data,
                }
        report_items.append(item)

    # 8. Assemble the complete report
    assessment_report = {
        "report_id": report_id,
        "report_type": "assessment_report",
        "timestamp_utc": timestamp_utc,
        "model_uuid": model_uuid,
        "model_version": model_version,
        "dataset": dataset_name,
        "tool_id": "TOOL-005",
        "has_defence": True,
        "report": report_items,
    }

    # 9. Save to outputs/YYYY-MM-DD/HH-MM-SS/assessment_report.json
    save_file_path = current_run_save_path / "assessment_report.json"
    try:
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(assessment_report, f, indent=4)
        logger.info(f"Assessment report successfully created at: {save_file_path}")
    except Exception as e:
        logger.error(f"Failed to save assessment report: {e}")
