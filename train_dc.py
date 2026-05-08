"""
Fair Dataset Distillation via Cross-Group Barycenter Alignment (COBRA)
Backbone: Dataset Condensation (DC) — gradient matching.

This single script supports three training modes:
    - vanilla : standard DC gradient matching (Zhao et al., 2020)
    - fairdd  : FairDD baseline (per-group loss accumulation, Zhou et al., 2025)
    - cobra   : COBRA — match a per-class subgroup-barycentric real gradient

Select the mode with --mode {vanilla,fairdd,cobra}.

Example:
    python train_dc.py --mode cobra --dataset Colored_MNIST_foreground --ipc 10
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
    get_daparam,
    get_dataset,
    get_eval_pool,
    get_loops,
    get_network,
    get_time,
    match_loss,
)


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
# Distillation losses
# --------------------------------------------------------------------------- #
def compute_real_grad_vanilla(net, net_parameters, criterion, img_real, lab_real):
    """Standard DC: a single real gradient on the full class-conditional batch."""
    output_real = net(img_real)
    loss_real = criterion(output_real, lab_real)
    gw_real = torch.autograd.grad(loss_real, net_parameters)
    return [g.detach().clone() for g in gw_real]


def compute_real_grad_fairdd(net, net_parameters, criterion, img_real, lab_real, color):
    """FairDD: per-group gradients summed (one match-loss per group, not used here);
    we approximate the FairDD intent by summing per-group gradients (weighted equally)
    so the synthetic gradient is matched against each group separately downstream."""
    grads_per_group = []
    for grp in torch.unique(color):
        mask = color == grp
        if mask.sum() == 0:
            continue
        loss_g = criterion(net(img_real[mask]), lab_real[mask])
        g = torch.autograd.grad(loss_g, net_parameters, retain_graph=True)
        grads_per_group.append([t.detach().clone() for t in g])
    return grads_per_group


def compute_real_grad_cobra(net, net_parameters, criterion, img_real, lab_real, color):
    """COBRA: build a subgroup-barycentric target by averaging per-group real
    gradients with uniform weights — independent of per-group sample size."""
    output_real = net(img_real)
    group_grads = {}
    for grp in torch.unique(color):
        mask = color == grp
        if mask.sum() == 0:
            continue
        loss_g = criterion(output_real[mask], lab_real[mask])
        g = torch.autograd.grad(loss_g, net_parameters, retain_graph=True)
        group_grads[grp.item()] = [t.detach().clone() for t in g]

    if len(group_grads) == 0:
        # Fallback: standard real gradient
        loss_real = criterion(output_real, lab_real)
        gw = torch.autograd.grad(loss_real, net_parameters, retain_graph=True)
        return [t.detach().clone() for t in gw]

    barycenter = []
    keys = list(group_grads.keys())
    n_layers = len(group_grads[keys[0]])
    for layer in range(n_layers):
        stacked = torch.stack([group_grads[k][layer] for k in keys], dim=0)
        barycenter.append(stacked.mean(dim=0))
    return barycenter


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="DC distillation with optional fairness mode.")
    p.add_argument("--mode", type=str, default="vanilla",
                   choices=["vanilla", "fairdd", "cobra"],
                   help="Training objective: vanilla DC, FairDD, or COBRA.")
    p.add_argument("--dataset", type=str, default="Colored_MNIST_foreground",
                   help="Dataset name (see data_handler).")
    p.add_argument("--model", type=str, default="ConvNet")
    p.add_argument("--ipc", type=int, default=10, help="Synthetic images per class.")
    p.add_argument("--eval_mode", type=str, default="S",
                   help="Eval mode: S (same arch), M (multi-arch), W/D/A/P/N (network ablation).")

    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=5,
                   help="# random eval models trained on the distilled set.")
    p.add_argument("--epoch_eval_train", type=int, default=1000)

    p.add_argument("--Iteration", type=int, default=1000,
                   help="Outer distillation iterations.")
    p.add_argument("--lr_img", type=float, default=0.1)
    p.add_argument("--lr_net", type=float, default=0.01)
    p.add_argument("--batch_real", type=int, default=256)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--init", type=str, default="real", choices=["real", "noise"])
    p.add_argument("--dsa_strategy", type=str, default="None")
    p.add_argument("--data_path", type=str, default="data")
    p.add_argument("--save_path", type=str, default="result")
    p.add_argument("--dis_metric", type=str, default="ours",
                   help="Gradient-matching distance metric (see utils.match_loss).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main(args):
    args.method = "DC"
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa_param = ParamDiffAug()
    args.dsa = args.dsa_strategy not in ("none", "None")
    # DC's FairDD/COBRA branches were originally toggled by `args.FairDD`; we
    # keep the flag for utils.evaluate_synset compatibility.
    args.FairDD = args.mode in ("fairdd", "cobra")

    os.makedirs(args.save_path, exist_ok=True)
    eval_it_pool = [args.Iteration]

    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []
    suffix = args.mode  # used in output filenames

    print(f"\n=== DC distillation | mode={args.mode} | dataset={args.dataset} | ipc={args.ipc} ===")

    for exp in range(args.num_exp):
        print(f"\n================== Exp {exp} ==================")
        print("Hyper-parameters:", args.__dict__)
        print("Evaluation model pool:", model_eval_pool)

        # Build flat tensors for fast random sampling
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

        for c in range(num_classes):
            print(f"class c={c}: {len(indices_class[c])} real images")

        def get_images(c, n):
            idx = np.random.permutation(indices_class[c])[:n]
            return images_all[idx], labels_all[idx], color_all[idx]

        # Initialise synthetic data
        image_syn = torch.randn(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float, requires_grad=True, device=args.device,
        )
        label_syn = torch.tensor(
            [np.ones(args.ipc) * i for i in range(num_classes)],
            dtype=torch.long, device=args.device,
        ).view(-1)
        if args.init == "real":
            print("Initialising synthetic data from random real images.")
            for c in range(num_classes):
                imgs, _, _ = get_images(c, args.ipc)
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = imgs.detach().data
        else:
            print("Initialising synthetic data from random noise.")

        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print(f"{get_time()} training begins")

        for it in range(args.Iteration + 1):
            # ---------------- Evaluation ---------------- #
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print(f"\n--- Eval at iter {it} | train={args.model} | eval={model_eval} ---")
                    if args.dsa:
                        args.epoch_eval_train = max(args.epoch_eval_train, 1000)
                        args.dc_aug_param = None
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)

                    accs, max_eo_list, mean_eo_list = [], [], []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        image_syn_eval = copy.deepcopy(image_syn.detach())
                        label_syn_eval = copy.deepcopy(label_syn.detach())
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

                # Save image grid
                vis_path = os.path.join(
                    args.save_path,
                    f"vis_DC_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}_{suffix}.png",
                )
                vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    vis[:, ch] = vis[:, ch] * std[ch] + mean[ch]
                vis.clamp_(0.0, 1.0)
                save_image(vis, vis_path, nrow=args.ipc)

            # ---------------- Distillation step ---------------- #
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            args.dc_aug_param = None

            loss_avg = 0.0
            for ol in range(args.outer_loop):
                # Freeze BatchNorm running stats with a small real-data pass
                if any("BatchNorm" in m._get_name() for m in net.modules()):
                    bn_imgs = torch.cat(
                        [get_images(c, 16)[0] for c in range(num_classes)], dim=0
                    )
                    net.train()
                    net(bn_imgs)
                    for m in net.modules():
                        if "BatchNorm" in m._get_name():
                            m.eval()

                loss = torch.tensor(0.0, device=args.device)
                for c in range(num_classes):
                    img_real, _, color = get_images(c, args.batch_real)
                    lab_real = torch.full((img_real.shape[0],), c,
                                          dtype=torch.long, device=args.device)
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1])
                    )
                    lab_syn = torch.full((args.ipc,), c, dtype=torch.long, device=args.device)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy,
                                               seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy,
                                              seed=seed, param=args.dsa_param)

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    if args.mode == "vanilla":
                        gw_real = compute_real_grad_vanilla(
                            net, net_parameters, criterion, img_real, lab_real
                        )
                        loss = loss + match_loss(gw_syn, gw_real, args)
                    elif args.mode == "fairdd":
                        per_group_grads = compute_real_grad_fairdd(
                            net, net_parameters, criterion, img_real, lab_real, color
                        )
                        if len(per_group_grads) == 0:
                            gw_real = compute_real_grad_vanilla(
                                net, net_parameters, criterion, img_real, lab_real
                            )
                            loss = loss + match_loss(gw_syn, gw_real, args)
                        else:
                            for gw_real in per_group_grads:
                                loss = loss + match_loss(gw_syn, gw_real, args) / len(per_group_grads)
                    else:  # cobra
                        gw_real = compute_real_grad_cobra(
                            net, net_parameters, criterion, img_real, lab_real, color
                        )
                        loss = loss + match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break

                # Update inner network on current synthetic set
                image_syn_train = copy.deepcopy(image_syn.detach())
                label_syn_train = copy.deepcopy(label_syn.detach())
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(
                    dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                )
                for _ in range(args.inner_loop):
                    epoch("train", trainloader, net, optimizer_net, criterion, args, aug=args.dsa)

            loss_avg /= num_classes * args.outer_loop
            if it % 10 == 0:
                print(f"{get_time()} iter {it:04d} | loss={loss_avg:.4f}")

            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(image_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu()),
                ])
                ckpt = os.path.join(
                    args.save_path,
                    f"res_DC_{args.dataset}_{args.model}_{args.ipc}ipc_{suffix}.pt",
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
