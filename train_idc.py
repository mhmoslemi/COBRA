"""
Fair Dataset Distillation via Cross-Group Barycenter Alignment (COBRA)
Backbone: Improved Dataset Condensation (IDC) — multi-formation gradient
matching (Kim et al., 2022).

This single script supports three training modes:
    - vanilla : standard IDC gradient matching
    - fairdd  : FairDD (per-group gradient matching, equally weighted)
    - cobra   : COBRA — match the subgroup-barycentric real gradient

Select the mode with --mode {vanilla,fairdd,cobra}.

Example:
    python train_idc.py --mode cobra --dataset Colored_MNIST_foreground --ipc 10
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from utils import (
    DiffAugment,
    ParamDiffAug,
    TensorDataset,
    epoch,
    evaluate_synset,
    get_dataset,
    get_eval_pool,
    get_loops,
    get_network,
    get_time,
    match_loss,
)


# --------------------------------------------------------------------------- #
# IDC multi-formation: split each synthetic image into factor*factor patches,
# upsample each patch to the original size, and treat them as separate samples.
# --------------------------------------------------------------------------- #
def decode_zoom(img, target, factor, size=(32, 32)):
    h = img.shape[-1]
    s_crop = h // factor
    resize = nn.Upsample(size=size, mode="bilinear", align_corners=True)

    cropped = []
    for i in range(factor):
        for j in range(factor):
            cropped.append(
                img[:, :, i * s_crop:(i + 1) * s_crop, j * s_crop:(j + 1) * s_crop]
            )
    cropped = torch.cat(cropped, dim=0)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(factor ** 2)])
    return data_dec, target_dec


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Real-gradient computation
# --------------------------------------------------------------------------- #
def real_grad_vanilla(net, params, criterion, img_real, lab_real):
    output_real = net(img_real)
    loss_real = criterion(output_real, lab_real)
    gw = torch.autograd.grad(loss_real, params)
    return [g.detach().clone() for g in gw]


def real_grad_per_group(net, params, criterion, img_real, lab_real, color):
    output_real = net(img_real)
    grads = {}
    for g in torch.unique(color):
        mask = color == g
        if mask.sum() == 0:
            continue
        loss_g = criterion(output_real[mask], lab_real[mask])
        grad = torch.autograd.grad(loss_g, params, retain_graph=True)
        grads[g.item()] = [t.detach().clone() for t in grad]
    return grads


def real_grad_cobra(net, params, criterion, img_real, lab_real, color):
    grads = real_grad_per_group(net, params, criterion, img_real, lab_real, color)
    if len(grads) == 0:
        return real_grad_vanilla(net, params, criterion, img_real, lab_real)
    keys = list(grads.keys())
    n_layers = len(grads[keys[0]])
    barycenter = []
    for layer in range(n_layers):
        stacked = torch.stack([grads[k][layer] for k in keys], dim=0)
        barycenter.append(stacked.mean(dim=0))
    return barycenter


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="IDC distillation with optional fairness mode.")
    p.add_argument("--mode", type=str, default="vanilla",
                   choices=["vanilla", "fairdd", "cobra"])
    p.add_argument("--dataset", type=str, default="Colored_MNIST_foreground")
    p.add_argument("--model", type=str, default="ConvNet")
    p.add_argument("--ipc", type=int, default=10)
    p.add_argument("--eval_mode", type=str, default="S")

    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=3)
    p.add_argument("--epoch_eval_train", type=int, default=1000)

    p.add_argument("--Iteration", type=int, default=1000)
    p.add_argument("--lr_img", type=float, default=0.1)
    p.add_argument("--lr_net", type=float, default=0.01)
    p.add_argument("--batch_real", type=int, default=256)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--init", type=str, default="real", choices=["real", "noise"])
    p.add_argument("--dsa_strategy", type=str, default="None")
    p.add_argument("--data_path", type=str, default="data")
    p.add_argument("--save_path", type=str, default="result")
    p.add_argument("--dis_metric", type=str, default="ours")
    p.add_argument("--factor", type=int, default=-1,
                   help="Multi-formation factor. -1 picks 4 for face datasets and 2 otherwise; "
                        "if --ipc>=100 and --factor=-1, falls back to 3 for face datasets.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main(args):
    args.method = "DC"  # IDC reuses DC's distance metric
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa_param = ParamDiffAug()
    args.dsa = args.dsa_strategy not in ("none", "None")
    args.FairDD = args.mode in ("fairdd", "cobra")

    os.makedirs(args.save_path, exist_ok=True)
    eval_it_pool = [args.Iteration]

    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # Default factor for IDC follows the paper's recipe
    if args.factor == -1:
        if args.dataset in ("BFFHQ", "UTKface"):
            args.factor = 3 if args.ipc >= 100 else 4
        else:
            args.factor = 2

    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []
    suffix = args.mode

    print(f"\n=== IDC distillation | mode={args.mode} | dataset={args.dataset} | ipc={args.ipc} | "
          f"factor={args.factor} ===")

    for exp in range(args.num_exp):
        print(f"\n================== Exp {exp} ==================")

        images_all = torch.cat(
            [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))],
            dim=0,
        ).to(args.device)
        labels_all = torch.tensor(
            [dst_train[i][1] for i in range(len(dst_train))],
            dtype=torch.long, device=args.device,
        )
        color_all = torch.tensor(
            [dst_train[i][2] for i in range(len(dst_train))],
            dtype=torch.long, device=args.device,
        )

        args.num_classes = int(len(torch.unique(labels_all)))
        args.num_groups = int(len(torch.unique(color_all)))
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all.tolist()):
            indices_class[lab].append(i)

        def get_images(c, n):
            idx = np.random.permutation(indices_class[c])[:n]
            return images_all[idx], labels_all[idx], color_all[idx]

        # Initialise synthetic data from real samples
        image_syn = torch.randn(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float, requires_grad=True, device=args.device,
        )
        label_syn = torch.tensor(
            [np.ones(args.ipc) * i for i in range(num_classes)],
            dtype=torch.long, device=args.device,
        ).view(-1)
        for c in range(num_classes):
            imgs, _, _ = get_images(c, args.ipc)
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = imgs.detach().data

        if args.ipc in (50, 100):
            # Wider gradient sweep helps higher-IPC IDC runs
            args.outer_loop = 30

        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print(f"{get_time()} training begins")

        for it in range(args.Iteration + 1):
            # ---------------- Evaluation ---------------- #
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print(f"\n--- Eval at iter {it} | train={args.model} | eval={model_eval} ---")
                    args.dc_aug_param = None
                    accs, max_eo_list, mean_eo_list = [], [], []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        with torch.no_grad():
                            image_syn_eval, label_syn_eval = decode_zoom(
                                image_syn.detach(), label_syn.detach(), args.factor, size=im_size
                            )
                        _, _, acc_test, max_eo, mean_eo, _, _ = evaluate_synset(
                            it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args
                        )
                        accs.append(acc_test)
                        max_eo_list.append(max_eo)
                        mean_eo_list.append(mean_eo)

                    print(f"Acc mean={np.mean(accs):.4f} std={np.std(accs):.4f} | "
                          f"max_EO={np.mean(max_eo_list):.4f} mean_EO={np.mean(mean_eo_list):.4f}")
                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

            # ---------------- Distillation step ---------------- #
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            args.dc_aug_param = None

            loss_avg = 0.0
            for ol in range(args.outer_loop):
                loss = torch.tensor(0.0, device=args.device)
                optimizer_img.zero_grad()
                for c in range(num_classes):
                    img_real, _, color = get_images(c, args.batch_real)
                    lab_real = torch.full((img_real.shape[0],), c,
                                          dtype=torch.long, device=args.device)

                    raw_syn = image_syn[c * args.ipc:(c + 1) * args.ipc]
                    raw_lab = label_syn[c * args.ipc:(c + 1) * args.ipc]
                    img_syn_decoded, lab_syn_decoded = decode_zoom(
                        raw_syn, raw_lab, args.factor, size=im_size
                    )

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy,
                                               seed=seed, param=args.dsa_param)
                        img_syn_decoded = DiffAugment(img_syn_decoded, args.dsa_strategy,
                                                      seed=seed, param=args.dsa_param)

                    output_syn = net(img_syn_decoded)
                    loss_syn = criterion(output_syn, lab_syn_decoded)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    if args.mode == "vanilla":
                        gw_real = real_grad_vanilla(net, net_parameters, criterion, img_real, lab_real)
                        loss = loss + match_loss(gw_syn, gw_real, args)
                    elif args.mode == "fairdd":
                        per_group = real_grad_per_group(
                            net, net_parameters, criterion, img_real, lab_real, color
                        )
                        if len(per_group) == 0:
                            gw_real = real_grad_vanilla(net, net_parameters, criterion,
                                                        img_real, lab_real)
                            loss = loss + match_loss(gw_syn, gw_real, args)
                        else:
                            for gw_real in per_group.values():
                                loss = loss + match_loss(gw_syn, gw_real, args) / len(per_group)
                    else:  # cobra
                        gw_real = real_grad_cobra(net, net_parameters, criterion,
                                                  img_real, lab_real, color)
                        loss = loss + match_loss(gw_syn, gw_real, args)

                loss.backward()
                loss_avg += loss.item()
                optimizer_img.step()

            loss_avg /= num_classes * args.outer_loop
            if it % 10 == 0:
                print(f"{get_time()} iter {it:04d} | loss={loss_avg:.4f}")

            if it == args.Iteration:
                data_save = [copy.deepcopy(image_syn.detach().cpu()),
                             copy.deepcopy(label_syn.detach().cpu())]
                ckpt = os.path.join(
                    args.save_path,
                    f"res_IDC_{args.dataset}_{args.model}_{args.ipc}ipc_{suffix}.pt",
                )
                torch.save({"data": data_save, "accs_all_exps": accs_all_exps}, ckpt)

    print("\n==================== Final Results ====================")
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        if accs:
            print(f"Trained on {args.model}, evaluated on {key}: "
                  f"acc mean={np.mean(accs)*100:.2f}%  std={np.std(accs)*100:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
