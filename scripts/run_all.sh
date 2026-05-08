#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Reproduce the COBRA main table (Table 1) end-to-end:
# four backbones (DC / DM / CAFE / IDC) × all eight datasets × ipc=10
# under three fairness modes (vanilla / fairdd / cobra).
#
# This is intentionally serial so it works on a single GPU. Comment out any
# rows you don't want, or split across machines / SLURM jobs as needed.
# CAFE only runs in vanilla / cobra (FairDD has no CAFE variant in the paper).
#
# Usage:
#   bash scripts/run_all.sh [ipc]
# ----------------------------------------------------------------------------
set -euo pipefail

IPC=${1:-10}

DATASETS=(
    "CIFAR10_S_90"
    "Colored_MNIST_foreground"
    "Colored_MNIST_background"
    "Colored_FashionMNIST_foreground"
    "Colored_FashionMNIST_background"
    "UTKface"
    "BFFHQ"
)

DCDM_MODES=(vanilla fairdd cobra)
CAFE_MODES=(vanilla cobra)

for ds in "${DATASETS[@]}"; do
    for m in "${DCDM_MODES[@]}"; do
        echo "=== DC | ${ds} | ipc=${IPC} | ${m} ==="
        bash scripts/run_dc.sh  "${ds}" "${IPC}" "${m}"

        echo "=== DM | ${ds} | ipc=${IPC} | ${m} ==="
        bash scripts/run_dm.sh  "${ds}" "${IPC}" "${m}"

        echo "=== IDC | ${ds} | ipc=${IPC} | ${m} ==="
        bash scripts/run_idc.sh "${ds}" "${IPC}" "${m}"
    done

    for m in "${CAFE_MODES[@]}"; do
        echo "=== CAFE | ${ds} | ipc=${IPC} | ${m} ==="
        bash scripts/run_cafe.sh "${ds}" "${IPC}" "${m}"
    done
done
