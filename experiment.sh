#!/bin/bash
# run_correct_experiments.sh

echo "Running HSGSP experiment"
dataset=$1

if [ "$dataset" = "CIFAR-10" ]; then
    echo "Running CIFAR-10 hybrid baseline..."
    python main.py \
        --task cifar10 \
        --train \
        --prune \
        --eval \
        --gpu 0
elif [ "$dataset" = "CIFAR-100" ]; then
    echo "Running CIFAR-100 hybrid baseline..."
    python main.py \
        --task cifar100 \
        --train \
        --prune \
        --eval \
        --gpu 0
else
    echo "Dataset not available"
fi

echo "Experiments completed!"