import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yaml import safe_load

from src.attacks import add_gaussian_noise, semantic_label_flip, flip_sign
from src.models import ModelConfig
from src.settings import settings
import math

from modules.utils import AverageMeter
from src.plot_utils import initialize_video_writer, write_vision_frame_to_video, release_video_writer

from src.utils import (
    get_device,
    _build_optimizer,
    _apply_sensor_mask,
    _resolve_video_filename,
    _calc_combined_loss,
)

def train(
    model: nn.Module,
    train_loader: DataLoader,
    client_type: str,
    lr: float,
    model_config: ModelConfig,
    attack_activated: bool,
    partition_id: int = 0,
    sensor_config: dict = None,
    sequence_meta: dict = None,
) -> tuple[float, float, float]:
    """
    Executes the localized training loop for a client, encompassing data loading, optimization, and attacks.

    Iterates through the provided dataloader for a specified number of epochs, applying 
    sensor masking constraints and recording video frames. If the client is adversarial, 
    post-training attacks (e.g., Sign-Flip, Gaussian noise) are injected.

    :param model: The client's local PyTorch model to be trained.
    :param train_loader: The dataloader providing local point cloud batches.
    :param client_type: The alignment of the client ("Honest" or "Malicious").
    :param lr: The learning rate for the optimizer.
    :param model_config: The global model configuration containing loss modules.
    :param attack_activated: A boolean flag signaling if malicious clients should execute their attacks.
    :param partition_id: The client's unique identifier.
    :param sensor_config: Constraints defining the client's simulated sensor degradation.
    :param sequence_meta: Sequence mapping information for video generation and tracking.
    :return: A tuple containing the final training loss, accuracy, and jaccard index (all zeroes currently).
    """


    # Device & model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Snapshot weights now so Sign-Flip can reference the pre-training state.
    original_weights = None
    if attack_activated and client_type == "Malicious" and settings.attack.type == "Sign-Flip":
        original_weights = [p.data.clone() for p in model.parameters()]

    # Loss functions, optimizer, and auxiliary-loss config
    criterion   = settings.use_case.criterion
    lovasz      = settings.use_case.lovasz
    boundary_loss = settings.use_case.boundary_loss
    optimizer   = _build_optimizer(model, lr)
    use_aux_loss = settings.use_case.model_architecture_config["train"]["aux_loss"]["use"]
    lamda        = settings.use_case.model_architecture_config["train"]["aux_loss"]["lamda"]

    # Video recording setup
    save_path = os.environ.get("RUN_SAVE_PATH")
    video_writer = None
    if save_path:
        filename, client_sensor_type = _resolve_video_filename(partition_id, sensor_config, sequence_meta)
        video_file = os.path.join(save_path, "client_videos", filename)
        if not os.path.exists(video_file):
            video_writer = initialize_video_writer(save_path, partition_id, client_sensor_type, sequence_meta)

    # Training loop
    for epoch in range(settings.client.local_epochs):
        losses = AverageMeter()
        pbar   = tqdm(
            train_loader,
            desc=f"Client {partition_id} - Epoch {epoch + 1}/{settings.client.local_epochs}",
            ncols=100,
        )
        for batch_idx, (in_vol, proj_labels, _, _, _, _, proj_range, _, _, proj_xyz, _) in enumerate(pbar):
            # Move data to device.
            in_vol      = in_vol.to(device)
            proj_labels = proj_labels.to(device).long()
            proj_range  = proj_range.to(device)
            proj_xyz    = proj_xyz.to(device)

            original_proj_labels = proj_labels.clone()
            sensor_profile = "Full-View"

            # Apply sensor-profile masking (simulates degraded LiDAR conditions).
            if settings.use_case.data_split == "sensor" and sensor_config is not None:
                sensor_type = sensor_config.get("type", "Full-View")
                proj_labels, in_vol, sensor_profile = _apply_sensor_mask(
                    sensor_type, sensor_config, proj_labels, proj_xyz, device,
                    in_vol=in_vol,
                )

            # Record the first-epoch frames for the client video.
            if video_writer is not None and epoch == 0:
                write_vision_frame_to_video(
                    video_writer, original_proj_labels, proj_labels,
                    partition_id, batch_idx, sensor_profile,
                )

            # In-batch attack: applied to labels before the forward pass.
            if attack_activated and client_type == "Malicious":
                match settings.attack.type:
                    case "Semantic-Label-Flip":
                        proj_labels = semantic_label_flip(
                            proj_labels, partition_id, device, model_config.num_classes
                        )

            # Subsample rows to create a lower-resolution input tensor.
            output_tensor = torch.zeros_like(in_vol)
            output_tensor[:, :, torch.arange(0, 40, 4), :] = in_vol[:, :, ::4, :].clone()

            # Forward pass + loss computation.
            if use_aux_loss:
                output, z2, z4, z8 = model(output_tensor)
                bd_loss = (
                    boundary_loss(output, proj_labels)
                    + lamda[0] * boundary_loss(z2, proj_labels)
                    + lamda[1] * boundary_loss(z4, proj_labels)
                    + lamda[2] * boundary_loss(z8, proj_labels)
                )
                loss = (
                    _calc_combined_loss(criterion, lovasz, output, proj_labels, lovasz_weight=1.5)
                    + lamda[0] * _calc_combined_loss(criterion, lovasz, z2, proj_labels, lovasz_weight=1.5)
                    + lamda[1] * _calc_combined_loss(criterion, lovasz, z4, proj_labels, lovasz_weight=1.5)
                    + lamda[2] * _calc_combined_loss(criterion, lovasz, z8, proj_labels, lovasz_weight=1.5)
                    + bd_loss
                )
            else:
                output, _ = model(output_tensor)
                bd_loss   = boundary_loss(output, proj_labels)
                loss      = _calc_combined_loss(criterion, lovasz, output, proj_labels) + bd_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

            losses.update(loss.item(), in_vol.size(0))
            pbar.set_postfix({"Loss": losses.avg})

        # Release video writer after epoch 0 is fully processed.
        if video_writer is not None and epoch == 0:
            release_video_writer(video_writer)

    # Post-training attacks: applied once after all local epochs
    if attack_activated and client_type == "Malicious":
        match settings.attack.type:
            case "Gaussian":
                add_gaussian_noise(model.parameters())
            case "Sign-Flip":
                flip_sign(model.parameters(), original_weights, scale_factor=-0.5)


def test(model: nn.Module, test_loader: DataLoader, evaluator: Any = None, call_desc: str = "") -> Tuple[Any, ...]:
    """
    Evaluates the model's semantic segmentation performance on a validation dataset.

    Computes loss, overall accuracy, and class-wise intersection-over-union (IoU) 
    using the provided evaluator module.

    :param model: The PyTorch model to be evaluated.
    :param test_loader: The dataloader providing validation point cloud batches.
    :param evaluator: An IoU evaluation tracker instance.
    :param call_desc: A descriptive string for progress bar logging (e.g., "Server Evaluation").
    :return: A tuple of the computed loss, overall accuracy, and mean Jaccard index.
    """


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    if evaluator is not None:
        evaluator.reset()
    criterion = settings.use_case.criterion
    lovasz = settings.use_case.lovasz
    losses = AverageMeter()
    use_aux_loss = settings.use_case.model_architecture_config["train"]["aux_loss"]["use"]

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=call_desc, ncols=100)
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

            if evaluator is not None:
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
            losses.update(loss.mean().item(), in_vol.size(0))
            pbar.set_postfix({"loss": f"{losses.avg:.4f}"})


    final_loss = losses.avg
    if math.isnan(final_loss) or math.isinf(final_loss):
        final_loss = 1e6  # Large penalty for completely shattered models (e.g., pure Gaussian noise)

    if evaluator is not None:
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        return final_loss, accuracy.item(), jaccard.item()
    else:
        # Return 2 values so that `loss, _ = test(...)` in strategies unpacks correctly
        return final_loss, 0.0


def split_dataset_into_clients(dataset, num_clients):
    """
    Split dataset into N clients based on sequences.
    Assigns sequences to clients round-robin. If multiple clients are assigned the same sequence,
    the sequence's frames are split equally among them without overlapping.
    """


    if not hasattr(dataset, "scan_files"):
        # Fallback to standard IID split if scan_files is missing
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients
        indices = list(range(total_samples))
        np.random.RandomState(settings.general.random_seed).shuffle(indices)
        client_indices = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
            client_indices.append({
                "indices": indices[start_idx:end_idx],
                "sequence": "unknown",
                "part_idx": i + 1,
                "total_parts": num_clients
            })
        return client_indices

    # 1. Group dataset indices by sequence
    sequence_to_indices = {}
    for idx, scan_file in enumerate(dataset.scan_files):
        # scan_file path example: /path/to/sequences/00/velodyne/000000.bin
        seq = os.path.basename(os.path.dirname(os.path.dirname(scan_file)))
        if seq not in sequence_to_indices:
            sequence_to_indices[seq] = []
        sequence_to_indices[seq].append(idx)
        
    available_seqs = sorted(list(sequence_to_indices.keys()))
    num_seqs = len(available_seqs)

    if num_seqs == 0:
        raise ValueError("No sequences found in the dataset.")

    # 2. Assign clients to sequences round-robin
    seq_to_clients = {seq: [] for seq in available_seqs}
    for i in range(num_clients):
        seq = available_seqs[i % num_seqs]
        seq_to_clients[seq].append(i)
        
    # 3. Split each sequence's indices among its assigned clients
    client_indices = [[] for _ in range(num_clients)]
    
    for seq, clients_for_seq in seq_to_clients.items():
        if not clients_for_seq:
            continue
            
        indices = sequence_to_indices[seq]
        # Use a fixed random seed for strictly deterministic distributions across runs
        np.random.RandomState(settings.general.random_seed).shuffle(indices)
        
        num_clients_for_seq = len(clients_for_seq)
        samples_per_client = len(indices) // num_clients_for_seq
        
        for idx_c, client_id in enumerate(clients_for_seq):
            start_idx = idx_c * samples_per_client
            end_idx = (idx_c + 1) * samples_per_client if idx_c < num_clients_for_seq - 1 else len(indices)
            client_indices[client_id] = {
                "indices": indices[start_idx:end_idx],
                "sequence": seq,
                "part_idx": idx_c + 1,
                "total_parts": num_clients_for_seq
            }

    return client_indices


_run_dir_cache = None
current_run_save_path = None


def create_run_dir() -> tuple[Path, str]:
    """
    Creates and returns a uniquely timestamped directory to store artifacts from the current simulation.

    :return: A tuple containing the Path object to the created directory and its string equivalent.
    """
    global _run_dir_cache, current_run_save_path
    if _run_dir_cache is not None:
        return _run_dir_cache

    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(settings.config_path, save_path)

    current_run_save_path = save_path
    _run_dir_cache = (save_path, run_dir)
    return save_path, run_dir


def generate_assessment_report() -> None:
    """
    Compiles and exports a JSON assessment report summarizing the simulation's configuration and results.
    
    The report details the model, use case, optimizer hyperparameters, and exact client data distributions 
    in a standardized format suitable for backend parsing or auditing.
    """
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
    dataset_name = settings.dataset.name

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
                    dataset_cfg = cfg.get("dataset", {})
                    if dataset_cfg.get("name") == dataset_name:
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
                "attack_id": "ATK-012",
                "defence_id": ["DEF-005"],
                "attack_execution_id": f"ATK-012-RUN-{index:04d}",
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
                    "attack_id": "ATK-012",
                    "defence_id": ["DEF-005"],
                    "attack_execution_id": f"ATK-012-RUN-{index:04d}",
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
        "has_defence": settings.server.strategy not in ["Mean", "Median", "Trimmed-Mean", "Trimmed Mean"],
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

def compute_exact_client_distributions(num_clients, num_classes, client_sensor_configs=None):
    """
    Calculates the exact, empirical distribution of class labels held by each client.

    Accounts for both the sequential data splitting boundaries and the localized 
    sensor degradation profiles (which may arbitrarily blind clients to certain classes).

    :param num_clients: The total number of federated clients.
    :param num_classes: The total number of unique object classes in the dataset.
    :param client_sensor_configs: Optional dictionary mapping client IDs to their sensor constraints.
    :return: A tuple containing the distribution counts per client and the split indices.
    """


    dataset = settings.use_case.parser.train_dataset
    client_indices = split_dataset_into_clients(dataset, num_clients)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    client_distributions = {i: {c: 0 for c in range(num_classes)} for i in range(num_clients)}
    
    for partition_id in range(num_clients):
        client_split = client_indices[partition_id]
        indices = client_split["indices"]
        if not indices:
            continue
            
        client_dataset = Subset(dataset, indices)
        loader = DataLoader(client_dataset, batch_size=settings.client.batch_size, shuffle=False, drop_last=False)
        
        sensor_config = client_sensor_configs.get(partition_id, None) if client_sensor_configs else None

        pbar = tqdm(loader, desc=f"Computing True Distribution for Client {partition_id}", ncols=100)
        for batch_idx, (in_vol, proj_labels, _, _, _, _, proj_range, _, _, proj_xyz, _) in enumerate(pbar):
            proj_labels = proj_labels.to(device).long()
            proj_range = proj_range.to(device)
            proj_xyz = proj_xyz.to(device)
            
            if settings.use_case.data_split == "sensor" and sensor_config is not None:
                sensor_type = sensor_config.get("type", "Full-View")
                # Labels-only masking: in_vol is not needed for distribution counting.
                proj_labels, _, _ = _apply_sensor_mask(
                    sensor_type, sensor_config, proj_labels, proj_xyz, device
                )
            
            # Count labels
            unique, counts = torch.unique(proj_labels, return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                if 0 <= u < num_classes:
                    client_distributions[partition_id][u] += c

    return client_distributions, client_indices


def compute_exact_server_distributions(num_classes):
    """
    Calculates the exact, empirical distribution of class labels within the server's datasets.

    Counts instances across the designated centralized evaluation dataset and the independent 
    defense validation dataset (used by robust aggregation strategies).

    :param num_classes: The total number of unique object classes in the dataset.
    :return: A tuple containing label counts for the evaluation and defense datasets.
    """


    test_dataset = settings.use_case.parser.valid_dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices = torch.arange(len(test_dataset))
    split = int(settings.defence.defence_dataset_percentage * len(test_dataset))
    
    eval_loader = DataLoader(Subset(test_dataset, indices[split:]), batch_size=settings.client.batch_size, shuffle=False)
    
    if settings.server.strategy in ["FedGreed", "FedCluster", "FedTruncate"]:
        def_loader = DataLoader(Subset(test_dataset, indices[:split]), batch_size=settings.client.batch_size, shuffle=False)
    else:
        def_loader = None
        
    def count_loader(loader, desc):
        counts = np.zeros(num_classes, dtype=int)
        if loader is None:
            return counts
        pbar = tqdm(loader, desc=desc, ncols=100)
        for batch_idx, (in_vol, proj_labels, _, _, _, _, _, _, _, _, _) in enumerate(pbar):
            proj_labels = proj_labels.to(device).long()
            unique, batch_counts = torch.unique(proj_labels, return_counts=True)
            for u, c in zip(unique.tolist(), batch_counts.tolist()):
                if 0 <= u < num_classes:
                    counts[u] += c
        return counts

    eval_counts = count_loader(eval_loader, "Computing Server Eval Distribution")
    def_counts = count_loader(def_loader, "Computing Server Def Distribution")
    
    return def_counts, eval_counts
