#!/bin/bash
# setup_partitions.sh
# This script systematically generates all three dataset partitioning structures
# (iid, non-iid, and sort_part) for every supported dataset across 10 clients.

set -e # Exit immediately if a command exits with a non-zero status

# All datasets natively supported by SecureFL's partition_dataset.py
# DATASETS=("MNIST" "FMNIST" "CIFAR10" "CIFAR100" "SVHN" "HAR" "Purchase")
DATASETS=("MNIST" "FMNIST" "CIFAR10" "CIFAR100" "SVHN")
NUM_CLIENTS=${1:-100}

# Hyperparameters for Non-IID distributions
DIRICHLET_ALPHA=${2:-0.1}
SORT_PART_CLASSES=${3:-2}

if [ -n "$4" ]; then
    SEEDS=("$4")
else
    SEEDS=(42 123 456)
fi

echo "=========================================================="
echo "🚀 Initiating Mass Offline Partitioning for SecureFL"
echo "=========================================================="
echo "Clients: $NUM_CLIENTS"
echo "Alpha (Non-IID): $DIRICHLET_ALPHA"
echo "Classes per user (Sort-Part): $SORT_PART_CLASSES"
echo "Seeds: ${SEEDS[*]}"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------"
        echo "📦 Processing Dataset: $DATASET (Seed: $SEED)"
        echo "----------------------------------------------------------"
    
        # 1. IID (Homogeneous)
        echo "  -> Generating [IID] partition..."
        poetry run partition-dataset "$DATASET" --num_clients "$NUM_CLIENTS" --type iid --seed "$SEED"
        
        # 2. Non-IID (Dirichlet)
        echo "  -> Generating [Non-IID] partition..."
        poetry run partition-dataset "$DATASET" --num_clients "$NUM_CLIENTS" --type non_iid --alpha "$DIRICHLET_ALPHA" --seed "$SEED"
        
        # 3. Sort-and-Partition
        CURRENT_SORT_PART=$SORT_PART_CLASSES
        if [ "$DATASET" = "CIFAR100" ] || [ "$DATASET" = "Purchase" ]; then
            CURRENT_SORT_PART=10
            echo "  ℹ️ Notice: Automatically scaling Sort-Part classes to 10 to satisfy 100-class constraint."
        fi
        
        echo "  -> Generating [Sort-and-Partition] partition..."
        poetry run partition-dataset "$DATASET" --num_clients "$NUM_CLIENTS" --type sort_part --numb_cls_usr "$CURRENT_SORT_PART" --seed "$SEED"
        
    done
done

echo ""
echo "✅ All permutations have been successfully generated and cached in data/client/ !"
