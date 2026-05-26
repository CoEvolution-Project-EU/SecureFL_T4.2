# SecureFL

## 📝 Overview

This repository contains the implementation of the SecureFL federated learning framework. SecureFL is designed to enable collaborative model training across multiple clients while preserving data privacy and ensuring robust performance against adversarial attacks in federated environments.

## 🗂️ Project Structure

```bash
.
├── avisence_datasets/  # Custom dataloaders for SemanticPOSS and SemanticKITTI
├── common/             # Utilities and base classes for 3D point cloud parsing
├── config/             # YAML configurations for network architectures and labels
├── modules/            # Specialized AVISENCE models, losses, and custom trainers
├── src/                # Core Flower code and model logic
│   ├── strategies/     # Flower server strategies (FedGreed, Loss-based Clustering, Mean, Trimmed-Mean, Median, Krum, Multi-Krum) 
├── data/               # Preprocessed partitioned data for FL clients
├── scripts/            # Data partitioning, simulation and utility scripts
├── outputs/            # Timestamped outputs: logs, metrics, and best global model checkpoints per experiment
├── pyproject.toml      # Python project configuration and dependencies
└── README.md           # Project documentation and usage instructions
```

> **Note:** The `data` and `outputs` directories are created automatically upon executing the data partitioning and simulation scripts, respectively.

## 🚀 Quick Start

### Prerequisites
Ensure you have [Poetry](https://python-poetry.org/docs/) installed and Python 3.11+ before proceeding.

### 1. Install Dependencies
Navigate to the root of the repository and install dependencies using Poetry:

```sh
poetry install
```

### 2. Preprocess and Distribute Data
Once dependencies are installed, preprocess the dataset and distribute it to the appropriate clients using the following command:

```sh
poetry run partition-dataset [OPTIONS]
```

#### Available Arguments
- `dataset_name` (Required): Name of the dataset (e.g., CIFAR10, FMNIST, MNIST).
- `--num_clients` (Optional): Number of federated learning (FL) clients.
- `--type` (Optional): Partitioning type, either `iid` or `non-iid`. Default is `iid`.
- `--alpha` (Optional): Alpha parameter for the Dirichlet distribution.

> **Note:** To set up non-iid distributions, use the `--type=non-iid` flag and set an `--alpha` value to control the skewness.

### 3. Custom Simulation
If you wish to configure your own simulation, you can set the specific configuration YAML file (`config.yaml`), and run the simulation using Poetry:

```sh
poetry run simulation
```

#### AVISENCE Use Case (3D Point Cloud Segmentation)
SecureFL supports 3D point cloud training for the AVISENCE project. To use it, simply add the following block to your `config.yaml`:

```yaml
use_case:
  name: "AVISENCE"
  data_split: "iid" # non-iid or iid
  data_config_path: "config/labels/semantic-poss.yaml"
  model_architecture_config_path: "config/arch/LENet_poss.yaml"
  data_dir: "./avisence_datasets/poss/dataset/SemanticPOSS"
```
Adding this block automatically configures the framework to load 3D data and run the proper 3D network model for the SemanticPOSS dataset.

> **Important Data Setup:** You must download the dataset sequences from [here](http://www.poss.pku.edu.cn/OpenDataResource/SemanticPOSS/SemanticPOSS_dataset.zip) and place them directly inside your data directory (e.g., `./avisence_datasets/poss/dataset/SemanticPOSS/sequences/`).

#### Configuring Attacks
You can simulate Byzantine attacks within your federated learning network by modifying the `attack` section in `config.yaml`. The framework currently supports the following attacks:
- **Sign Flip**: Malicious clients invert the sign of their model updates before sending them to the server.
- **Label Flip**: Malicious clients flip the target labels of their local training dataset.
- **Gaussian Noise**: Malicious clients introduce Gaussian noise (configured via `mean` and `std`) to their model weights.

### 4. Simulation Outputs
At the end of a simulation, a new timestamped directory is created under the `outputs/` folder (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`). This directory contains the complete artifacts from that specific run, including:

- `assessment_report.json`: A dynamic chronological assessment report detailing the model's resilience against attacks across runs.
- `results.json`: Raw centralized and federated evaluation metrics (loss and accuracy) per round.
- `config.yaml`: A snapshot of the configuration used for this simulation to ensure reproducibility.
- `simulation.log`: The full log output of the Flower execution.
- `*.pth`: PyTorch state dictionary checkpoints of the best performing global models found during training.

#### Example: `assessment_report.json`
The assessment report tracks multiple simulation executions over time. An example report for a "Sign Flip" attack on MNIST might look like this:

```json

{
    "report_id": "REP-TOOL-005-{model_uuid}-{model_version}-2026-05-24T09:59:53Z-0010",
    "report_type": "assessment_report",
    "timestamp_utc": "2026-05-24T09:59:53Z",
    "model_uuid": "",
    "model_version": "",
    "dataset": "SemanticPOSS",
    "tool_id": "TOOL-005",
    "has_defence": true,
    "report": [
        {
            "attack_id": "ATT-012",
            "defence_id": [
                "DEF-005"
            ],
            "attack_execution_id": "ATT-012-RUN-0001",
            "category": "unsafe_output",
            "confidence": "medium",
            "occurrence": "systematic",
            "severity": "high",
            "general_info": {
                "centralized_evaluate": [
                    {
                        "round": 0,
                        "centralized_loss": 2.3145557072511904,
                        "centralized_accuracy": 6.15
                    },
                    {
                        "round": 1,
                        "centralized_loss": 0.7125890412528044,
                        "centralized_accuracy": 76.18
                    },
                    {
                        "round": 2,
                        "centralized_loss": 4563743.073248408,
                        "centralized_accuracy": 9.74
                    }
                ],
                "federated_evaluate": [
                    {
                        "round": 1,
                        "federated_evaluate_loss": 0.9398310179419281,
                        "federated_evaluate_accuracy": 67.27757414195268
                    },
                    {
                        "round": 2,
                        "federated_evaluate_loss": 4068974.184107491,
                        "federated_evaluate_accuracy": 9.805064978340553
                    }
                ]
            }
        }
    ]
}
```
