#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# COBRA — CAFE (multi-layer feature alignment) backbone.
# Reproduces the CAFE rows of Table 1 (vanilla / COBRA).
#
# Usage:
#   bash scripts/run_cafe.sh <dataset> <ipc> [mode]
#     dataset  : CIFAR10_S_90 | Colored_MNIST_foreground | Colored_MNIST_background
#                Colored_FashionMNIST_foreground | Colored_FashionMNIST_background
#                UTKface | BFFHQ
#     ipc      : images-per-class (e.g. 10, 50)
#     mode     : vanilla | cobra            (default: cobra)
#
# Note: CAFE uses its own networks (forward returns (logits, [features]))
# located under the `cafe/` sub-package.
#
# Examples:
#   bash scripts/run_cafe.sh CIFAR10_S_90 10 cobra
#   bash scripts/run_cafe.sh UTKface 50 vanilla
# ----------------------------------------------------------------------------
set -euo pipefail

DATASET=${1:-CIFAR10_S_90}
IPC=${2:-10}
MODE=${3:-cobra}

python train_cafe.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    --ipc "${IPC}" \
    --model ConvNet \
    --eval_mode S \
    --num_exp 1 \
    --num_eval 2 \
    --Iteration 2000 \
    --lr_img 0.1 \
    --batch_real 256 \
    --data_path data \
    --save_path result
