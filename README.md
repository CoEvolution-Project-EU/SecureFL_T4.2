# SecureFL

## 📝 Overview

This repository contains the implementation of the SecureFL federated learning framework. SecureFL is designed to enable collaborative model training across multiple clients while preserving data privacy and ensuring robust performance against adversarial attacks in federated environments.

## 🗂️ Project Structure

```bash
.
├── src/                # Core Flower code and model logic
│   ├── strategies/     # Flower server strategies (FedGreed, Mean, Trimmed-Mean, Median, Krum, Multi-Krum) 
├── configs/            # Configuration files for running and customizing experiments
├── data/               # Preprocessed partitioned data for FL clients
├── scripts/            # Data partitioning, simulation and utility scripts
├── outputs/            # Timestamped outputs: logs, metrics, and best global model checkpoints per experiment
├── pyproject.toml      # Python project configuration and dependencies
├── run_experiments.sh  # Automated Bash script for running the complete experimental suite
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

### 3. Run Pre-configured Experiments
To execute the automated script for running experiments across multiple strategies and datasets under different attack configurations (Label-Flipping, Gaussian Noise):

```sh
sh run_experiments.sh 
```
The results of each experiment will be stored in timestamped directories within the `outputs` folder.

### 4. Custom Simulation
If you wish to configure your own simulation, you can set a specific configuration YAML file as an environment variable and run the simulation using Poetry:

```sh
export config_file_name=config_no_attack
poetry run simulation
```

The `configs` directory contains predefined YAML configuration files designed for simulating various attacks, such as `config_data_attack` for data specific attacks and `config_model_attack` for model specific attacks. To apply a specific configuration, simply update the corresponding environment variable with the desired YAML file name.
