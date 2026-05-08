#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# COBRA — DC (Dataset Condensation, gradient matching) backbone.
# Reproduces the DC rows of Table 1 (vanilla / FairDD / COBRA).
#
# Usage:
#   bash scripts/run_dc.sh <dataset> <ipc> [mode]
#     dataset  : CIFAR10_S_90 | Colored_MNIST_foreground | Colored_MNIST_background
#                Colored_FashionMNIST_foreground | Colored_FashionMNIST_background
#                UTKface | BFFHQ
#     ipc      : images-per-class (e.g. 10, 50)
#     mode     : vanilla | fairdd | cobra   (default: cobra)
#
# Examples:
#   bash scripts/run_dc.sh CIFAR10_S_90 10 cobra
#   bash scripts/run_dc.sh Colored_MNIST_foreground 50 fairdd
# ----------------------------------------------------------------------------
set -euo pipefail

DATASET=${1:-CIFAR10_S_90}
IPC=${2:-10}
MODE=${3:-cobra}

python train_dc.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    --ipc "${IPC}" \
    --model ConvNet \
    --eval_mode S \
    --num_exp 1 \
    --num_eval 5 \
    --data_path data \
    --save_path result
