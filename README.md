# SecureFL

## 📝 Overview

This repository contains the implementation of the SecureFL federated learning framework. SecureFL is designed to enable collaborative model training across multiple clients while preserving data privacy and ensuring robust performance against adversarial attacks in federated environments.

## 🗂️ Project Structure

```bash
.
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
- `--type` (Optional): Partitioning type, either `homogeneous` or `heterogeneous`. Default is `homogeneous`.
- `--alpha` (Optional): Alpha parameter for the Dirichlet distribution.

> **Note:** To set up heterogeneous distributions, use the `--type=heterogeneous` flag and set an `--alpha` value to control the skewness.

### 3. Custom Simulation
If you wish to configure your own simulation, you can set the specific configuration YAML file (`config.yaml`), and run the simulation using Poetry:

```sh
poetry run simulation
```

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
    "report_id": "REP-TOOL-005-{model_uuid}-{model_version}-2026-05-20T10:35:00Z-0001",
    "report_type": "assessment_report",
    "timestamp_utc": "2026-05-20T10:35:00Z",
    "model_uuid": "",
    "model_version": "",
    "dataset": "MNIST",
    "tool_id": "TOOL-005",
    "has_defence": true,
    "report": [
        {
            "attack_id": "ATK-002",
            "defence_id": [
                "DEF-001"
            ],
            "attack_execution_id": "ATK-002-RUN-0001",
            "category": "unsafe_output",
            "confidence": "medium",
            "occurrence": "systematic",
            "severity": "high",
            "general_info": ""
        }
    ]
}
```
