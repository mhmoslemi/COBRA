#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# COBRA — IDC (Improved DC with multi-formation / decode_zoom) backbone.
# Reproduces the IDC rows of Table 1 (vanilla / FairDD / COBRA).
#
# Usage:
#   bash scripts/run_idc.sh <dataset> <ipc> [mode]
#     dataset  : CIFAR10_S_90 | Colored_MNIST_foreground | Colored_MNIST_background
#                Colored_FashionMNIST_foreground | Colored_FashionMNIST_background
#                UTKface | BFFHQ
#     ipc      : images-per-class (e.g. 10, 50)
#     mode     : vanilla | fairdd | cobra   (default: cobra)
#
# IDC's multi-formation factor defaults to:
#   3 if ipc>=100, 4 for face datasets (UTKface / BFFHQ), 2 otherwise.
# Override with --factor.
#
# Examples:
#   bash scripts/run_idc.sh BFFHQ 10 cobra
#   bash scripts/run_idc.sh CIFAR10_S_90 50 vanilla
# ----------------------------------------------------------------------------
set -euo pipefail

DATASET=${1:-CIFAR10_S_90}
IPC=${2:-10}
MODE=${3:-cobra}

python train_idc.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    --ipc "${IPC}" \
    --model ConvNet \
    --eval_mode S \
    --num_exp 1 \
    --num_eval 3 \
    --Iteration 1000 \
    --lr_img 0.1 \
    --batch_real 256 \
    --data_path data \
    --save_path result
