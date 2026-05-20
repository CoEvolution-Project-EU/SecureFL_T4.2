#!/bin/bash

DATASETS="CIFAR10 FMNIST MNIST"
STRATEGIES=("FedGreed" "Loss-based Clustering" "Krum" "Multi-Krum" "Median" "Trimmed-Mean" "Mean")
MALICIOUS_USERS="8 5 3"

for m in $MALICIOUS_USERS
do
  for d in $DATASETS
  do
    for s in "${STRATEGIES[@]}"
    do

      echo "Handling no attack under strategy: $s for dataset: $d with $m malicious users"
      YAML_FILE="config.yaml"
      yq -i -y ".server.strategy = \"$s\" | .model.name = \"$d\" | .attack.type = null" "$YAML_FILE"
      poetry run simulation

      echo "Handling Sign Flip attack under strategy: $s for dataset: $d with $m malicious users"
      YAML_FILE="config.yaml"
      yq -i -y ".server.strategy = \"$s\" | .attack.num_malicious_clients = $m | .model.name = \"$d\" | .attack.type = \"Sign Flip\"" "$YAML_FILE"
      poetry run simulation

    done
  done
done
