#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# COBRA — DM (Distribution Matching, feature-mean alignment) backbone.
# Reproduces the DM rows of Table 1 (vanilla / FairDD / COBRA).
#
# Usage:
#   bash scripts/run_dm.sh <dataset> <ipc> [mode]
#     dataset  : CIFAR10_S_90 | Colored_MNIST_foreground | Colored_MNIST_background
#                Colored_FashionMNIST_foreground | Colored_FashionMNIST_background
#                UTKface | BFFHQ
#     ipc      : images-per-class (e.g. 10, 50)
#     mode     : vanilla | fairdd | cobra   (default: cobra)
#
# COBRA warm-starts the embedding network on the synthetic set for
#   K = 50 if ipc==50, 100 if ipc==100, else 10 epochs (override --cobra_warmup_epochs).
#
# Examples:
#   bash scripts/run_dm.sh BFFHQ 10 cobra
#   bash scripts/run_dm.sh Colored_FashionMNIST_background 50 fairdd
# ----------------------------------------------------------------------------
set -euo pipefail

DATASET=${1:-CIFAR10_S_90}
IPC=${2:-10}
MODE=${3:-cobra}

python train_dm.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    --ipc "${IPC}" \
    --model ConvNet \
    --eval_mode S \
    --num_exp 1 \
    --num_eval 5 \
    --Iteration 2600 \
    --lr_img 1.0 \
    --batch_real 600 \
    --data_path data \
    --save_path result
