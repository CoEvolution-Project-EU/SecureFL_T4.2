import os
import cv2
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import seaborn as sns

from flwr.common import parameters_to_ndarrays
from logging import INFO, WARNING
from flwr.common.logger import log

from src.settings import settings
def get_label_name(data_config, class_id):
    """
    Retrieves the human-readable string name for a given class ID.

    Uses the dataset configuration to map internal class indices back to their semantic 
    string representations (e.g., mapping class 1 to 'Person').

    :param data_config: The dataset configuration dictionary containing label mappings.
    :param class_id: The integer class ID to look up.
    :return: The string name of the class, or a fallback 'Class X' string if not found.
    """
    labels_dict = data_config.get("labels", {})
    inv_map = data_config.get("learning_map_inv", {})
    orig_id = inv_map.get(class_id, class_id)
    return labels_dict.get(orig_id, f"Class {class_id}")

def plot_metrics_scatter(losses, parameters_list, client_types, selected_status, save_path, server_round):
    """
    Generates a scatter plot visualizing client updates in terms of loss versus magnitude.

    This diagnostic plot helps identify malicious outliers by plotting each client's 
    validation loss against the L2 norm of their parameter update.

    :param losses: A list of validation losses for each client.
    :param parameters_list: A list of client parameter updates.
    :param client_types: A list of string labels ('Honest' or 'Malicious') for each client.
    :param selected_status: A list of boolean flags indicating if the client's update was accepted.
    :param save_path: The base directory where the plot will be saved.
    :param server_round: The current federated learning round index.
    """
    if not losses:
        return

    metrics_dir = save_path / "plots" / "loss_vs_magnitude"
    os.makedirs(metrics_dir, exist_ok=True)

    try:
        magnitudes = []
        for params in parameters_list:
            ndarrays = parameters_to_ndarrays(params)
            flattened = np.concatenate([arr.flatten() for arr in ndarrays])
            magnitude = np.linalg.norm(flattened)
            magnitudes.append(magnitude)
            
        losses = np.array(losses)
        magnitudes = np.array(magnitudes)
        client_types = np.array(client_types)
        selected_status = np.array(selected_status)
        
        mask_h_s = (client_types == 'Honest') & selected_status
        mask_h_r = (client_types == 'Honest') & ~selected_status
        mask_m_s = (client_types == 'Malicious') & selected_status
        mask_m_r = (client_types == 'Malicious') & ~selected_status

        plt.figure(figsize=(10, 8))
        
        # Plot empty scatters to ensure all labels appear in the legend
        n_hs, n_hr = np.sum(mask_h_s), np.sum(mask_h_r)
        n_ms, n_mr = np.sum(mask_m_s), np.sum(mask_m_r)
        plt.scatter([], [], c='#1f77b4', marker='o', s=60, alpha=0.3, label=f'Honest (Rejected) n={n_hr}')
        plt.scatter([], [], c='#1f77b4', marker='o', s=100, edgecolor='black', linewidth=1.5, alpha=0.9, label=f'Honest (Selected) n={n_hs}')
        plt.scatter([], [], c='#d62728', marker='x', s=60, alpha=0.5, label=f'Malicious (Rejected) n={n_mr}')
        plt.scatter([], [], c='#d62728', marker='X', s=120, edgecolor='black', linewidth=1.5, alpha=0.9, label=f'Malicious (Selected) n={n_ms}')

        if np.any(mask_h_r):
            plt.scatter(losses[mask_h_r], magnitudes[mask_h_r], c='#1f77b4', marker='o', s=60, alpha=0.3, zorder=2)
        if np.any(mask_h_s):
            plt.scatter(losses[mask_h_s], magnitudes[mask_h_s], c='#1f77b4', marker='o', s=100, edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
        if np.any(mask_m_r):
            plt.scatter(losses[mask_m_r], magnitudes[mask_m_r], c='#d62728', marker='x', s=60, alpha=0.5, zorder=2)
        if np.any(mask_m_s):
            plt.scatter(losses[mask_m_s], magnitudes[mask_m_s], c='#d62728', marker='X', s=120, edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)

        selected_losses = losses[selected_status]
        rejected_losses = losses[~selected_status]
        
        if len(selected_losses) > 0 and len(rejected_losses) > 0:
            boundary_loss = (np.max(selected_losses) + np.min(rejected_losses)) / 2.0
            plt.axvline(boundary_loss, color='black', linestyle='--', linewidth=2.5, label='Selection Boundary', zorder=1)
            
            # Shade the selected region
            xmin, xmax = plt.gca().get_xlim()
            plt.axvspan(xmin, boundary_loss, color='#e5e7eb', alpha=0.5, zorder=0)

        plt.xlabel('Evaluation Loss', fontsize=14)
        plt.ylabel('Update Magnitude (L2 Norm)', fontsize=14)
        plt.title('Client Updates: Loss vs Magnitude', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(metrics_dir / f"metrics_round_{server_round}.png", dpi=150)
        plt.close()
        
        log(INFO, f"Saved Loss vs Magnitude scatter plot to {metrics_dir} for round {server_round}")
    except Exception as e:
        log(WARNING, f"Failed to generate Metrics scatter plot: {e}")

def generate_split_plot(def_counts, eval_counts, num_classes, save_dir):
    """
    Generates and saves a heatmap comparing the defense and evaluation dataset distributions.

    Visualizes the exact number of class instances present in both the defense validation 
    set and the global evaluation set, aiding in the analysis of data splits and label imbalance.

    :param def_counts: An array containing the label counts for the defense dataset.
    :param eval_counts: An array containing the label counts for the evaluation dataset.
    :param num_classes: The total number of unique classes.
    :param save_dir: The directory where the resulting heatmap image will be saved.
    """

    label_names = []
    if hasattr(settings, "use_case") and getattr(settings.use_case, "data_config", None):
        data_config = settings.use_case.data_config
        for i in range(num_classes):
            name = get_label_name(data_config, i)
            label_names.append(name)
    else:
        label_names = [str(i) for i in range(num_classes)]

    # Create the data matrix
    data_matrix = np.vstack([def_counts, eval_counts])
    row_labels = ["Defense Dataset", "Evaluation Dataset"]
    df = pd.DataFrame(data_matrix, index=row_labels, columns=label_names)

    # Format annotations with K/M logic
    annot_data = np.empty_like(df.values, dtype=object)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.values[i, j]
            if val == 0:
                annot_data[i, j] = "0"
            elif val >= 1_000_000:
                annot_data[i, j] = f"{val / 1_000_000:.1f}M"
            elif val >= 1_000:
                annot_data[i, j] = f"{val / 1_000:.1f}K"
            else:
                annot_data[i, j] = str(int(val))

    fig_width = max(14, num_classes * 0.8)
    fig_height = 4
    plt.figure(figsize=(fig_width, fig_height))


    def human_format_tick(x, pos):
        if x == 0:
            return "0"
        elif x >= 1_000_000:
            return f"{x / 1_000_000:.1f}M".replace(".0M", "M")
        elif x >= 1_000:
            return f"{x / 1_000:.1f}K".replace(".0K", "K")
        return str(int(x))

    cbar_formatter = FuncFormatter(human_format_tick)

    sns.heatmap(
        df,
        annot=annot_data,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Label Count", "format": cbar_formatter},
        linewidths=0.5,
        square=False,
    )

    num_samples = np.sum(def_counts)
    pct = settings.defence.defence_dataset_percentage
    plt.title(f"Server Dataset Distribution", fontsize=14)

    plt.xlabel("Class Label", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    out_path = save_dir / "server_split_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_client_split_plot(client_distributions: dict, num_classes: int, save_dir, client_sensor_configs=None, client_indices=None, filename="client_split_distribution.png"):
    """
    Generates and saves a comprehensive heatmap detailing the class distribution across all clients.

    This visualization highlights non-IID data partitions by displaying exact label counts 
    per client, and annotates clients with their assigned sensor degradation profiles (e.g., 'Blind-Class').

    :param client_distributions: A dictionary mapping client IDs to their label count arrays.
    :param num_classes: The total number of unique classes.
    :param save_dir: The directory where the resulting heatmap image will be saved.
    :param client_sensor_configs: Optional mapping of client IDs to their applied sensor profiles.
    :param client_indices: Optional mapping detailing the specific data splits assigned to clients.
    :param filename: The target filename for the generated plot.
    """

    clients = sorted(list(client_distributions.keys()))
    num_clients = len(clients)
    if num_clients == 0:
        return

    target_counts_per_client = np.zeros((num_clients, num_classes), dtype=int)
    for i, c in enumerate(clients):
        dist = client_distributions[c]
        for cls_id, count in dist.items():
            if cls_id < num_classes:
                target_counts_per_client[i, cls_id] = count

    label_names = []
    if hasattr(settings, "use_case") and getattr(settings.use_case, "data_config", None):
        data_config = settings.use_case.data_config
        for i in range(num_classes):
            name = get_label_name(data_config, i)
            label_names.append(name)
    else:
        label_names = [f"Class {i}" for i in range(num_classes)]

    # Convert label counts to a DataFrame
    df = pd.DataFrame(target_counts_per_client.tolist(), columns=label_names)

    # Dynamically scale figure dimensions based on the number of clients and classes to prevent squishing
    fig_width = max(14, num_clients * 0.9)
    fig_height = max(8, num_classes * 0.25)
    plt.figure(figsize=(fig_width, fig_height))

    # Disable annotations natively if there are too many elements (prevents dense black text squares)
    show_annotations = num_classes <= 100

    annot_data = False
    if show_annotations:
        annot_data = np.empty_like(df.T.values, dtype=object)
        for i in range(df.T.shape[0]):
            for j in range(df.T.shape[1]):
                val = df.T.values[i, j]
                if val == 0:
                    annot_data[i, j] = "0"
                elif val >= 1_000_000:
                    annot_data[i, j] = f"{val / 1_000_000:.1f}M"
                elif val >= 1_000:
                    annot_data[i, j] = f"{val / 1_000:.1f}K"
                else:
                    annot_data[i, j] = str(val)


    def human_format_tick(x, pos):
        if x == 0:
            return "0"
        elif x >= 1_000_000:
            return f"{x / 1_000_000:.1f}M".replace(".0M", "M")
        elif x >= 1_000:
            return f"{x / 1_000:.1f}K".replace(".0K", "K")
        return str(int(x))

    cbar_formatter = FuncFormatter(human_format_tick)

    # Plot heatmap
    sns.heatmap(
        df.T,
        annot=annot_data,
        annot_kws={"size": 12, "weight": "bold"},
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Label Count", "format": cbar_formatter},
        linewidths=0.5,
        square=False,
    )
    plt.xlabel("Partition ID", fontsize=14)
    plt.ylabel("Labels", fontsize=14)

    x_labels = []
    for c in clients:
        label = str(c)
        if client_sensor_configs and c in client_sensor_configs:
            cfg = client_sensor_configs[c]
            sensor_type = cfg.get("type", "Full-View")
            if sensor_type == "Blind-Class":
                blinded = cfg.get("blinded_category", "?")
                label += f"\nNo-{blinded}"
            else:
                label += f"\n{sensor_type}"
            
        if client_indices and c < len(client_indices):
            seq = client_indices[c].get("sequence", "")
            part_idx = client_indices[c].get("part_idx", 1)
            total_parts = client_indices[c].get("total_parts", 1)
            
            if total_parts > 1:
                label += f"\nSeq {seq} (Part {part_idx}/{total_parts})"
            else:
                label += f"\nSeq {seq}"

        x_labels.append(label)

    plt.xticks(np.arange(num_clients) + 0.5, x_labels, rotation=90, ha='center', fontsize=max(9, 14 - (num_clients // 20)))
    plt.yticks(rotation=0, fontsize=max(8, 14 - (num_classes // 25)))
    plt.tight_layout()

    out_path = save_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def initialize_video_writer(save_dir, client_id, sensor_type="Full-View", sequence_meta=None, fps=1, width=1500, height=800):
    """
    Initializes an OpenCV VideoWriter for recording a client's localized training progress.

    :param save_dir: The directory where the video file will be saved.
    :param client_id: The ID of the client generating the video.
    :param sensor_type: The applied sensor profile, used for the filename.
    :param sequence_meta: Metadata linking the client to a specific dataset sequence.
    :param fps: The frames-per-second rate of the output video.
    :param width: The horizontal resolution of the video.
    :param height: The vertical resolution of the video.
    :return: An initialized cv2.VideoWriter object, or None if OpenCV is unavailable.
    """
    vid_dir = os.path.join(save_dir, "client_videos")
    os.makedirs(vid_dir, exist_ok=True)
    
    if sequence_meta and sequence_meta.get("sequence") != "unknown":
        seq = sequence_meta["sequence"]
        part_idx = sequence_meta["part_idx"]
        total_parts = sequence_meta["total_parts"]
        if total_parts > 1:
            filename = f"client_{client_id}_seq_{seq}_part_{part_idx}of{total_parts}_{sensor_type}.mp4"
        else:
            filename = f"client_{client_id}_seq_{seq}_{sensor_type}.mp4"
    else:
        filename = f"client_{client_id}_{sensor_type}.mp4"
        
    out_path = os.path.join(vid_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(out_path, fourcc, fps, (width, height))

def write_vision_frame_to_video(video_writer, original_proj_labels, masked_proj_labels, client_id, frame_idx, sensor_profile="Standard"):
    """
    Renders and appends a single comparative frame to the client's training visualization video.

    The generated frame places the original (unmasked) ground truth alongside the 
    masked version perceived by the client, annotating it with the active sensor profile.

    :param video_writer: The active cv2.VideoWriter instance.
    :param original_proj_labels: The unmasked, original projected labels tensor.
    :param masked_proj_labels: The sensor-masked projected labels tensor.
    :param client_id: The ID of the client producing the frame.
    :param frame_idx: The current batch/frame index.
    :param sensor_profile: The string identifier of the applied sensor mask.
    """

    orig_label_map = original_proj_labels.cpu().numpy()
    masked_label_map = masked_proj_labels.cpu().numpy()

    if orig_label_map.ndim == 3:
        orig_label_map = orig_label_map[0]
        masked_label_map = masked_label_map[0]

    num_classes = 14
    label_names = []
    colors = []
    if hasattr(settings, "use_case") and getattr(settings.use_case, "data_config", None):
        data_config = settings.use_case.data_config
        inv_map = data_config.get("learning_map_inv", {})
        color_map_dict = data_config.get("color_map", {})
        num_classes = len(inv_map)
        for i in range(num_classes):
            orig_id = inv_map.get(i, i)
            name = get_label_name(data_config, i)
            label_names.append(name)
            
            # Extract color (BGR) and convert to RGB normalized
            bgr = color_map_dict.get(orig_id, [128, 128, 128])
            rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
            colors.append(rgb)
    else:
        label_names = [f"Class {i}" for i in range(num_classes)]
        try:
            cmap_fallback = matplotlib.colormaps.get_cmap("tab20").resampled(num_classes)
        except AttributeError:
            cmap_fallback = plt.cm.get_cmap("tab20", num_classes)
        colors = [cmap_fallback(i) for i in range(num_classes)]

    cmap = ListedColormap(colors)

    # Set DPI and figsize so width x height matches video writer (1500x800)
    # 15 inches x 8 inches at 100 dpi = 1500 x 800
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), dpi=100)
    
    # Use vmin=-0.5, vmax=num_classes-0.5 so that label i maps exactly to colors[i]
    im0 = axes[0].imshow(orig_label_map, cmap=cmap, vmin=-0.5, vmax=num_classes - 0.5, interpolation="nearest")
    axes[0].set_title(f"Client {client_id} - Frame {frame_idx:04d} - Original Dataset View (Full Vision)")
    axes[0].axis("off")

    im1 = axes[1].imshow(masked_label_map, cmap=cmap, vmin=-0.5, vmax=num_classes - 0.5, interpolation="nearest")
    axes[1].set_title(f"Client {client_id} - Frame {frame_idx:04d} - Client Perspective ({sensor_profile})")
    axes[1].axis("off")
    
    patches = []
    # Start loop from 0 to explicitly include "unlabeled" in the legend
    for i in range(num_classes):
        patches.append(mpatches.Patch(color=colors[i], label=label_names[i]))
    axes[1].legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    
    # Draw figure to a numpy array
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    
    # Convert RGBA to BGR for OpenCV
    img_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    video_writer.write(img_bgr)
    
    plt.close(fig)

def release_video_writer(video_writer):
    """
    Safely finalizes and releases the OpenCV VideoWriter resources.

    :param video_writer: The cv2.VideoWriter instance to release.
    """
    if video_writer is not None:
        video_writer.release()
