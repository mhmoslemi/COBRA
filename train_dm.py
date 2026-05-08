"""
Fair Dataset Distillation via Cross-Group Barycenter Alignment (COBRA)
Backbone: Distribution Matching (DM) — feature-mean alignment.

This single script supports three training modes:
    - vanilla : standard DM (mean class features matched, Zhao & Bilen, 2023)
    - fairdd  : FairDD (per-group feature means matched independently)
    - cobra   : COBRA — match the class-conditional subgroup barycentre
                of the embedding network. The embedding network is first
                trained on the current synthetic set (warm-start).

Select the mode with --mode {vanilla,fairdd,cobra}.

Example:
    python train_dm.py --mode cobra --dataset CIFAR10_S_90 --ipc 10
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
    epoch2,
    evaluate_synset,
    get_dataset,
    get_eval_pool,
    get_loops,
    get_network,
    get_time,
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
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="DM distillation with optional fairness mode.")
    p.add_argument("--mode", type=str, default="vanilla",
                   choices=["vanilla", "fairdd", "cobra"])
    p.add_argument("--dataset", type=str, default="CIFAR10_S_90")
    p.add_argument("--model", type=str, default="ConvNet")
    p.add_argument("--ipc", type=int, default=10)
    p.add_argument("--eval_mode", type=str, default="S")

    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=5)
    p.add_argument("--epoch_eval_train", type=int, default=1500)

    p.add_argument("--Iteration", type=int, default=2600)
    p.add_argument("--lr_img", type=float, default=1.0)
    p.add_argument("--lr_net", type=float, default=0.01)
    p.add_argument("--batch_real", type=int, default=600)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--init", type=str, default="real", choices=["real", "noise"])
    p.add_argument("--dsa_strategy", type=str, default="color_crop_cutout_flip_scale_rotate")
    p.add_argument("--data_path", type=str, default="data")
    p.add_argument("--save_path", type=str, default="result")
    p.add_argument("--dis_metric", type=str, default="ours")

    p.add_argument("--cobra_warmup_epochs", type=int, default=-1,
                   help="Epochs to warm-start the embedding net before computing the "
                        "barycenter (COBRA only). -1 picks 50 if ipc=50, 100 if ipc=100, "
                        "else 10.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main(args):
    args.method = "DM"
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

    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []
    suffix = args.mode

    print(f"\n=== DM distillation | mode={args.mode} | dataset={args.dataset} | ipc={args.ipc} ===")

    for exp in range(args.num_exp):
        print(f"\n================== Exp {exp} ==================")

        images_all = torch.cat(
            [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))],
            dim=0,
        ).to(args.device)
        labels_all = torch.tensor(
            [int(dst_train[i][1]) for i in range(len(dst_train))],
            dtype=torch.long, device=args.device,
        )
        color_all = torch.tensor(
            [int(dst_train[i][2]) for i in range(len(dst_train))],
            dtype=torch.long, device=args.device,
        )

        args.num_classes = int(len(torch.unique(labels_all)))
        args.num_groups = int(len(torch.unique(color_all)))
        indices_class = [[] for _ in range(args.num_classes)]
        for i, lab in enumerate(labels_all.tolist()):
            indices_class[lab].append(i)

        def get_images(c, n):
            idx = np.random.permutation(indices_class[c])[:n]
            return images_all[idx], labels_all[idx], color_all[idx]

        # Initialise synthetic data
        image_syn = torch.randn(
            size=(args.num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float, requires_grad=True, device=args.device,
        )
        label_syn = torch.tensor(
            [np.ones(args.ipc) * i for i in range(num_classes)],
            dtype=torch.long, device=args.device,
        ).view(-1)
        if args.init == "real":
            print("Initialising synthetic data from random real images.")
            for c in range(args.num_classes):
                imgs, _, _ = get_images(c, args.ipc)
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = imgs.detach().data
        else:
            print("Initialising synthetic data from random noise.")

        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        print(f"{get_time()} training begins")

        for it in range(args.Iteration + 1):
            # ---------------- Evaluation ---------------- #
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print(f"\n--- Eval at iter {it} | train={args.model} | eval={model_eval} ---")
                    accs, max_eo_list, mean_eo_list = [], [], []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, args.num_classes, im_size).to(args.device)
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

                vis_path = os.path.join(
                    args.save_path,
                    f"vis_DM_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}_{suffix}.png",
                )
                vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    vis[:, ch] = vis[:, ch] * std[ch] + mean[ch]
                vis.clamp_(0.0, 1.0)
                save_image(vis, vis_path, nrow=args.ipc)

            # ---------------- Distillation step ---------------- #
            net = get_network(args.model, channel, args.num_classes, im_size).to(args.device)
            net.train()
            criterion = nn.CrossEntropyLoss().to(args.device)

            # COBRA mode: warm-start the embedding network on current synthetic data
            if args.mode == "cobra":
                K = args.cobra_warmup_epochs
                if K < 0:
                    K = 50 if args.ipc == 50 else (100 if args.ipc == 100 else 10)
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
                image_syn_train = copy.deepcopy(image_syn.detach())
                label_syn_train = copy.deepcopy(label_syn.detach())
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(
                    dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                )
                for _ in range(K):
                    _, _, net = epoch2("train", trainloader, net, optimizer_net,
                                       criterion, args, aug=args.dsa)
                for p in net.parameters():
                    p.requires_grad = False
            else:
                for p in net.parameters():
                    p.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

            loss = torch.tensor(0.0, device=args.device)
            for c in range(args.num_classes):
                img_real, _, color = get_images(c, args.batch_real)
                img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                    (args.ipc, channel, im_size[0], im_size[1])
                )

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy,
                                           seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy,
                                          seed=seed, param=args.dsa_param)

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                if args.mode == "vanilla":
                    loss = loss + torch.sum(
                        (torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2
                    )
                elif args.mode == "fairdd":
                    for col in torch.unique(color):
                        mask = color == col
                        if mask.sum() == 0:
                            continue
                        loss = loss + torch.sum(
                            (torch.mean(output_real[mask], dim=0)
                             - torch.mean(output_syn, dim=0)) ** 2
                        )
                else:  # cobra
                    group_means = []
                    for g in torch.unique(color):
                        mask = color == g
                        if mask.sum() == 0:
                            continue
                        group_means.append(torch.mean(embed(img_real[mask]), dim=0))
                    if len(group_means) > 0:
                        barycenter = torch.mean(torch.stack(group_means, dim=0), dim=0)
                        diff = barycenter - torch.mean(output_syn, dim=0)
                        loss = loss + torch.sum(diff.abs())

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            if it % 250 == 0:
                print(f"{get_time()} iter {it:05d} | loss={loss.item() / args.num_classes:.4f}")

            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(image_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu()),
                ])
                ckpt = os.path.join(
                    args.save_path,
                    f"res_DM_{args.dataset}_{args.model}_{args.ipc}ipc_{suffix}.pt",
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
