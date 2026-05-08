"""
Fair Dataset Distillation via Cross-Group Barycenter Alignment (COBRA)
Backbone: CAFE — multi-layer feature alignment (Wang et al., 2022).

Modes:
    - vanilla : standard CAFE multi-layer feature MSE alignment.
    - cobra   : COBRA — average per-subgroup feature means (uniform weights),
                then align synthetic feature mean to that barycentre.

CAFE's network forward must return (logits, [feature_layers]).
Use --mode cobra for the fair variant.

Example:
    python train_cafe.py --mode cobra --dataset Colored_FashionMNIST_background --ipc 50
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

# CAFE needs its own networks/utils because its forward returns
# (logits, [features]) for multi-layer alignment. The cafe.* package
# provides drop-in replacements that match the CAFE forward contract.
from cafe.utils import (
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


def adjust_learning_rate(optimizer, step, init_lr):
    """Multiplicative LR decay schedule from CAFE."""
    lr = init_lr
    for milestone in (1200, 1600, 1800):
        if step >= milestone:
            lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def feature_mse_class_mean(real_feat, syn_feat, num_classes):
    """Mean-of-class MSE between real and syn feature batches (CAFE-style)."""
    mse = nn.MSELoss(reduction="sum")
    real = real_feat.view(num_classes, real_feat.shape[0] // num_classes, *real_feat.shape[1:])
    syn = syn_feat.view(num_classes, syn_feat.shape[0] // num_classes, *syn_feat.shape[1:])
    return mse(real.mean(dim=1), syn.mean(dim=1))


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="CAFE distillation with optional COBRA mode.")
    p.add_argument("--mode", type=str, default="vanilla",
                   choices=["vanilla", "cobra"])
    p.add_argument("--dataset", type=str, default="CIFAR10_S_90")
    p.add_argument("--model", type=str, default="ConvNet")
    p.add_argument("--ipc", type=int, default=10)
    p.add_argument("--eval_mode", type=str, default="S")

    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=2)
    p.add_argument("--epoch_eval_train", type=int, default=1000)

    p.add_argument("--Iteration", type=int, default=2000)
    p.add_argument("--lr_img", type=float, default=0.1)
    p.add_argument("--lr_net", type=float, default=0.01)
    p.add_argument("--batch_real", type=int, default=256)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--init", type=str, default="real", choices=["real", "noise"])
    p.add_argument("--dsa_strategy", type=str, default="None")
    p.add_argument("--data_path", type=str, default="data")
    p.add_argument("--save_path", type=str, default="result")
    p.add_argument("--dis_metric", type=str, default="ours")

    # Layer weights, COBRA defaults match the paper config; see CAFE Eq. (5)
    p.add_argument("--first_weight", type=float, default=1.0)
    p.add_argument("--second_weight", type=float, default=1.0)
    p.add_argument("--third_weight", type=float, default=1.0)
    p.add_argument("--fourth_weight", type=float, default=1.0)
    p.add_argument("--inner_weight", type=float, default=0.01)
    p.add_argument("--lambda_1", type=float, default=0.05,
                   help="Outer-loop early-stop threshold (CAFE).")
    p.add_argument("--lambda_2", type=float, default=0.05,
                   help="Inner-loop early-stop threshold (CAFE).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main(args):
    args.method = "CAFE"
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa_param = ParamDiffAug()
    args.dsa = args.dsa_strategy not in ("none", "None")
    args.FairDD = args.mode == "cobra"

    os.makedirs(args.save_path, exist_ok=True)
    eval_it_pool = [args.Iteration]

    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []
    suffix = args.mode

    print(f"\n=== CAFE distillation | mode={args.mode} | dataset={args.dataset} | ipc={args.ipc} ===")

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
        criterion_sum = nn.CrossEntropyLoss(reduction="sum").to(args.device)
        mse_loss = nn.MSELoss(reduction="sum").to(args.device)
        print(f"{get_time()} training begins")

        for it in range(args.Iteration + 1):
            adjust_learning_rate(optimizer_img, it, args.lr_img)

            # ---------------- Evaluation ---------------- #
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print(f"\n--- Eval at iter {it} | train={args.model} | eval={model_eval} ---")
                    if args.dsa:
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

                vis_path = os.path.join(
                    args.save_path,
                    f"vis_CAFE_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}_{suffix}.png",
                )
                vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    vis[:, ch] = vis[:, ch] * std[ch] + mean[ch]
                vis.clamp_(0.0, 1.0)
                save_image(vis, vis_path, nrow=args.ipc)

            # ---------------- Distillation step ---------------- #
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            args.dc_aug_param = None

            loss_avg = 0.0
            for ol in range(args.outer_loop):
                acc_watcher = []
                loss = torch.tensor(0.0, device=args.device)

                if args.mode == "cobra":
                    # COBRA: per-class subgroup-averaged feature MSE (uniform group weights).
                    img_real_gather = []
                    lab_real_gather = []
                    target_layers = [0, -1, -2, -3, -4]
                    for c in range(num_classes):
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

                        img_real_gather.append(img_real)
                        lab_real_gather.append(
                            torch.full((img_real.shape[0],), c,
                                       dtype=torch.long, device=args.device)
                        )

                        _, real_features = net(img_real)
                        _, syn_features = net(img_syn)

                        unique_groups = torch.unique(color)
                        if len(unique_groups) == 0:
                            continue
                        avg_real = {}
                        for idx in target_layers:
                            avg_real[idx] = torch.zeros(
                                real_features[idx].shape[1:], device=args.device
                            )
                        for col in unique_groups:
                            mask = color == col
                            for idx in target_layers:
                                avg_real[idx] = avg_real[idx] + torch.mean(
                                    real_features[idx][mask], dim=0
                                )
                        for idx in target_layers:
                            avg_real[idx] = avg_real[idx] / len(unique_groups)

                        # Layer weights: shallow layers attenuated (-3, -4) per CAFE
                        for idx in (0, -1, -2):
                            loss = loss + (
                                mse_loss(avg_real[idx], torch.mean(syn_features[idx], dim=0))
                                / num_classes
                            )
                        for idx in (-3, -4):
                            loss = loss + 0.1 * (
                                mse_loss(avg_real[idx], torch.mean(syn_features[idx], dim=0))
                                / num_classes
                            )
                else:
                    # Vanilla CAFE: full multi-layer feature MSE plus inner classification loss.
                    img_real_gather = []
                    lab_real_gather = []
                    img_syn_gather = []
                    for c in range(num_classes):
                        img_real, _, _ = get_images(c, args.batch_real)
                        lab_real = torch.full((img_real.shape[0],), c,
                                              dtype=torch.long, device=args.device)
                        img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                            (args.ipc, channel, im_size[0], im_size[1])
                        )
                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy,
                                                   seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy,
                                                  seed=seed, param=args.dsa_param)
                        img_real_gather.append(img_real)
                        lab_real_gather.append(lab_real)
                        img_syn_gather.append(img_syn)

                    side = im_size[0]
                    img_real_gather = torch.stack(img_real_gather, dim=0).reshape(
                        args.batch_real * num_classes, channel, side, side
                    )
                    img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(
                        args.ipc * num_classes, channel, side, side
                    )
                    lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(
                        args.batch_real * num_classes
                    )

                    output_real, real_features = net(img_real_gather)
                    output_syn, syn_features = net(img_syn_gather)

                    loss_middle = (
                        args.fourth_weight * feature_mse_class_mean(real_features[-1],
                                                                    syn_features[-1], num_classes)
                        + args.third_weight * feature_mse_class_mean(real_features[-2],
                                                                     syn_features[-2], num_classes)
                        + args.second_weight * feature_mse_class_mean(real_features[-3],
                                                                      syn_features[-3], num_classes)
                        + args.first_weight * feature_mse_class_mean(real_features[-4],
                                                                     syn_features[-4], num_classes)
                    )
                    loss = loss + loss_middle + criterion(output_real, lab_real_gather)

                    last_real = torch.mean(
                        real_features[0].view(num_classes, -1, real_features[0].shape[1]), dim=1
                    )
                    last_syn = torch.mean(
                        syn_features[0].view(num_classes, -1, syn_features[0].shape[1]), dim=1
                    )
                    out_align = torch.mm(real_features[0], last_syn.t())
                    loss = loss + feature_mse_class_mean(last_syn, last_real, num_classes)
                    loss = loss + args.inner_weight * criterion_sum(out_align, lab_real_gather)

                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()
                loss_avg += loss.item()

                # Outer-loop early stopping (CAFE)
                with torch.no_grad():
                    acc_test = 0.0
                    for c in range(num_classes):
                        img_real_test, _, _ = get_images(c, 256)
                        prob, _ = net(img_real_test)
                        acc_test = acc_test + (
                            torch.full((len(img_real_test),), c, device=args.device)
                            == prob.max(dim=1)[1]
                        ).float().mean()
                    acc_test = acc_test / num_classes
                    acc_watcher.append(acc_test.detach().cpu().item())
                if len(acc_watcher) >= 10 and (max(acc_watcher) - min(acc_watcher)) < args.lambda_1:
                    break

                # Update inner network on current synthetic set
                image_syn_train = copy.deepcopy(image_syn.detach())
                label_syn_train = copy.deepcopy(label_syn.detach())
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(
                    dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                )
                for _ in range(max(1, args.inner_loop)):
                    epoch("train", trainloader, net, optimizer_net, criterion, args, aug=args.dsa)

            loss_avg /= num_classes * max(1, args.outer_loop)
            if it % 10 == 0:
                print(f"{get_time()} iter {it:04d} | loss={loss_avg:.4f}")

            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(image_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu()),
                ])
                ckpt = os.path.join(
                    args.save_path,
                    f"res_CAFE_{args.dataset}_{args.model}_{args.ipc}ipc_{suffix}.pt",
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
