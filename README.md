# COBRA — Fair Dataset Distillation via Cross-Group Barycenter Alignment



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.00185-b31b1b.svg)](https://arxiv.org/abs/2605.00185)


Official PyTorch implementation of
**"[Fair Dataset Distillation via Cross-Group Barycenter Alignment](https://arxiv.org/pdf/2605.00185)"** (ICML 2026).


COBRA distills a small synthetic training set whose class-conditional
representation matches the **uniform Wasserstein-style barycenter of the
sensitive subgroups**, instead of the (biased) population mean used by
vanilla distillation. The resulting synthetic data trains downstream
classifiers that are simultaneously **accurate** and **fair under
Equalized Odds**, across four distillation backbones and seven biased
benchmarks.

---

## 1. Repository layout

```
.
├── train_dc.py          # DC  backbone — gradient matching
├── train_dm.py          # DM  backbone — feature-mean matching
├── train_idc.py         # IDC backbone — DC + multi-formation (decode_zoom)
├── train_cafe.py        # CAFE backbone — multi-layer feature alignment
│
├── networks.py          # ConvNet / AlexNet / VGG / ResNet (logits-only)
├── utils.py             # shared training, evaluation, fairness metrics
│
├── cafe/                # CAFE-specific sub-package
│   ├── __init__.py
│   ├── networks.py      # forward returns (logits, [layer_features])
│   └── utils.py         # CAFE-aware get_network / epoch / evaluate_synset
│
├── data_handler/        # dataset wrappers (CIFAR10-S, C-MNIST, C-FMNIST,
│                          UTKFace, BFFHQ, CelebA)
│
├── scripts/             # one-line reproduction commands
│   ├── run_dc.sh
│   ├── run_dm.sh
│   ├── run_idc.sh
│   ├── run_cafe.sh
│   └── run_all.sh
│
├── requirements.txt
├── LICENSE
└── README.md
```

Each `train_*.py` exposes the **same CLI surface**, so switching
backbones means switching the script name. The fairness behaviour is
controlled by a single flag, `--mode`:

| `--mode`  | Meaning                                                      | Available in       |
| --------- | ------------------------------------------------------------ | ------------------ |
| `vanilla` | Original distillation loss (population mean)                 | DC / DM / IDC / CAFE |
| `fairdd`  | FairDD — per-subgroup loss, summed independently             | DC / DM / IDC      |
| `cobra`   | **COBRA** — match synthetic mean to subgroup barycenter      | DC / DM / IDC / CAFE |

---

## 2. Installation

```bash
git clone https://github.com/<your-org>/cobra.git
cd cobra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9 / 3.10, PyTorch ≥ 1.13 (CUDA 11.8), and a single
NVIDIA A100. CPU runs are supported but slow.

---

## 3. Datasets

All datasets are loaded automatically from `--data_path` (default
`./data`). The first run will download what is downloadable; UTKFace and
BFFHQ require a manual one-time download (see `data_handler/utkface.py`
and `data_handler/bffhq.py` for the expected directory structure).

| Key                                  | Source             | Sensitive attribute |
| ------------------------------------ | ------------------ | ------------------- |
| `CIFAR10_S_90`                       | CIFAR-10-S         | colour skew (0.9)   |
| `Colored_MNIST_foreground`           | Colored MNIST (FG) | digit colour        |
| `Colored_MNIST_background`           | Colored MNIST (BG) | background colour   |
| `Colored_FashionMNIST_foreground`    | Colored F-MNIST    | foreground colour   |
| `Colored_FashionMNIST_background`    | Colored F-MNIST    | background colour   |
| `UTKface`                            | UTKFace            | gender              |
| `BFFHQ`                              | BFFHQ              | age                 |

---

## 4. Quick start

The shell scripts under `scripts/` accept three positional arguments:
`<dataset> <ipc> <mode>`.

```bash
# DM, COBRA, CIFAR-10-S, 10 images per class
bash scripts/run_dm.sh CIFAR10_S_90 10 cobra

# IDC, FairDD baseline, BFFHQ, 50 images per class
bash scripts/run_idc.sh BFFHQ 50 fairdd

# CAFE, vanilla baseline, UTKFace, 10 images per class
bash scripts/run_cafe.sh UTKface 10 vanilla
```

To reproduce the entire main table at `ipc=10` (serial, ~1 GPU-day):

```bash
bash scripts/run_all.sh 10
```

Each run writes a checkpoint and a visualisation grid under `--save_path`
(default `./result/`):

```
result/
├── res_DM_CIFAR10_S_90_ConvNet_10ipc_cobra.pt
└── vis_DM_CIFAR10_S_90_ConvNet_10ipc_exp0_iter2600_cobra.png
```

The checkpoint stores `(synthetic_images, synthetic_labels)` plus the
list of accuracies across evaluation seeds.

---

## 5. CLI reference

Common flags for every `train_*.py`:

| Flag                  | Default          | Notes                                    |
| --------------------- | ---------------- | ---------------------------------------- |
| `--mode`              | `vanilla`        | `vanilla` / `fairdd` / `cobra` (CAFE: `vanilla` / `cobra`) |
| `--dataset`           | `CIFAR10_S_90`   | See dataset table above                  |
| `--ipc`               | `10`             | Images per class                          |
| `--model`             | `ConvNet`        | `ConvNet` / `AlexNet` / `VGG11` / `ResNet18` |
| `--num_exp`           | `1`              | # of independent distillation runs       |
| `--num_eval`          | `5` (DC/DM)      | # of evaluation seeds per run             |
| `--Iteration`         | varies           | Outer distillation iterations             |
| `--lr_img`            | varies           | Synthetic-data learning rate              |
| `--lr_net`            | `0.01`           | Inner-network learning rate               |
| `--batch_real`        | varies           | Real-data batch size per class            |
| `--init`              | `real`           | `real` (init from real samples) or `noise` |
| `--data_path`         | `data`           | Dataset root                              |
| `--save_path`         | `result`         | Output root                               |
| `--seed`              | `42`             | Top-level RNG seed                        |

Backbone-specific flags:

* **DM (`train_dm.py`)**: `--cobra_warmup_epochs K` — number of epochs
  the embedding network is warm-started on the current synthetic data
  before computing the barycenter under `--mode cobra`. `K=-1` (default)
  picks `50` for `ipc=50`, `100` for `ipc=100`, else `10`.
* **IDC (`train_idc.py`)**: `--factor F` — multi-formation factor used
  by `decode_zoom`. `F=-1` (default) picks `3` for `ipc>=100`, `4` for
  face datasets (UTKFace / BFFHQ), else `2`.
* **CAFE (`train_cafe.py`)**: `--first_weight / --second_weight /
  --third_weight / --fourth_weight` (per-layer MSE weights),
  `--inner_weight`, and `--lambda_1 / --lambda_2` (early-stop
  thresholds). Defaults match the values reported in the paper.

Run any script with `--help` for the full list.

---

## 6. Method in one paragraph

Given a class `c` and a sensitive attribute taking values `g ∈ G`,
COBRA computes per-subgroup feature means
`μ_{c,g} = E[ φ(x) | y=c, s=g ]`
through the embedding network `φ`, then sets the **target** for the
synthetic batch of class `c` to the uniform barycenter
`b_c = (1/|G|) Σ_g μ_{c,g}`.
The distillation loss aligns the synthetic feature mean to `b_c` (DM /
CAFE) or aligns gradient signals computed against `b_c` (DC / IDC). For
DM, the embedding network is briefly warm-started on the current
synthetic set so the target reflects an informative representation.
Because the barycenter weights subgroups uniformly regardless of their
prior frequency, the resulting synthetic data is balanced in
representation space rather than in input space, which empirically
yields lower Equalized-Odds gaps without sacrificing accuracy.

---

## 7. Citation

```bibtex
@article{moslemi2026fair,
  title={Fair Dataset Distillation via Cross-Group Barycenter Alignment},
  author={Moslemi, Mohammad Hossein and Dashtbayaz, Nima Hosseini and Mei, Zhimin and Wang, Boyu and Ghaddar, Bissan},
  journal={arXiv preprint arXiv:2605.00185},
  year={2026}
}
```


---

## 8. Acknowledgements

This codebase builds on the public implementations of
[DC/DM](https://github.com/VICO-UoE/DatasetCondensation),
[CAFE](https://github.com/kaiwang960112/CAFE),
[IDC](https://github.com/snu-mllab/Efficient-Dataset-Condensation),
and [FairDD](https://github.com/dq-coder/FairDD). We thank the original
authors for releasing their code.
