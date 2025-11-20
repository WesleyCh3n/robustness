"""
Track attack convergence over iterations
Generate Figure 1: Robust accuracy vs iterations

Author: WesleyCh3n
Date: 2025-11-17 17:43:08 UTC
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attacks.apgd_ce import APGDAttack
from attacks.pgd import PGDAttack
from models.resnet import ResNet18


class ConvergenceTracker:
    """Track robust accuracy over iterations"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def track_pgd_convergence(self, x, y, eps, alpha, max_iters):
        """Track PGD convergence iteration by iteration"""
        batch_size = x.shape[0]
        robust_accuracies = []

        # Initialize
        x_adv = x.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)

        # Get initial predictions
        with torch.no_grad():
            initial_pred = self.model(x).argmax(1)
            initially_correct = initial_pred == y

        # Track at iteration 0
        with torch.no_grad():
            pred = self.model(x_adv).argmax(1)
            still_correct = (pred == y) & initially_correct
            robust_acc = still_correct.float().sum() / initially_correct.float().sum()
            robust_accuracies.append(robust_acc.item() * 100)

        # Run PGD iterations
        for i in range(max_iters):
            x_adv.requires_grad = True

            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            # Update
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1)

            # Track robust accuracy
            with torch.no_grad():
                pred = self.model(x_adv).argmax(1)
                still_correct = (pred == y) & initially_correct
                robust_acc = (
                    still_correct.float().sum() / initially_correct.float().sum()
                )
                robust_accuracies.append(robust_acc.item() * 100)

        return robust_accuracies

    def track_apgd_convergence(self, x, y, eps, max_iters):
        """Track APGD-CE convergence iteration by iteration"""
        batch_size = x.shape[0]
        robust_accuracies = []

        # Initialize
        t = 2 * torch.rand_like(x) - 1
        x_adv = x + eps * t
        x_adv = torch.clamp(x_adv, 0, 1)
        x_best = x_adv.clone()

        # Get initial predictions
        with torch.no_grad():
            initial_pred = self.model(x).argmax(1)
            initially_correct = initial_pred == y

        # Setup for APGD
        alpha = 2.0
        step_size = alpha * eps * torch.ones([batch_size, 1, 1, 1], device=self.device)

        # Checkpoint parameters (from official implementation)
        n_iter_2 = max(int(0.22 * max_iters), 1)
        n_iter_min = max(int(0.06 * max_iters), 1)
        size_decr = max(int(0.03 * max_iters), 1)

        loss_steps = torch.zeros([max_iters, batch_size], device=self.device)
        loss_best = torch.ones(batch_size, device=self.device) * (-float("inf"))

        x_adv_old = x_adv.clone()

        k = n_iter_2
        counter3 = 0
        thr_decr = 0.75

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones(batch_size, device=self.device)

        # Initial gradient computation (before iteration 0)
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)

        with torch.enable_grad():
            logits = self.model(x_adv)
            loss_indiv = F.cross_entropy(logits, y, reduction="none")
            loss = loss_indiv.sum()

        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()

        # Initial accuracy and loss
        acc = logits.detach().max(1)[1] == y
        loss_best = loss_indiv.detach().clone()

        # Track at iteration 0
        with torch.no_grad():
            pred = self.model(x_adv).argmax(1)
            still_correct = (pred == y) & initially_correct
            robust_acc = still_correct.float().sum() / initially_correct.float().sum()
            robust_accuracies.append(robust_acc.item() * 100)

        # Run APGD iterations
        for i in range(max_iters):
            # Gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                # Standard PGD step
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(torch.max(x_adv_1, x - eps), x + eps), 0.0, 1.0
                )
                # Add momentum
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - eps
                        ),
                        x + eps,
                    ),
                    0.0,
                    1.0,
                )
                x_adv = x_adv_1 + 0.0

            # Compute gradient for next iteration
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = F.cross_entropy(logits, y, reduction="none")
                loss = loss_indiv.sum()

            grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            # Update best
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0

                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0

                counter3 += 1

                # Check for step size reduction at checkpoints
                if counter3 == k:
                    # Oscillation check
                    t_osc = torch.zeros(batch_size, device=self.device)
                    for counter in range(k):
                        t_osc += (
                            loss_steps[i - counter] > loss_steps[i - counter - 1]
                        ).float()

                    fl_oscillation = (t_osc <= k * thr_decr).float()

                    # Check if no improvement since last check
                    fl_reduce_no_impr = (1.0 - reduced_last_check) * (
                        loss_best_last_check >= loss_best
                    ).float()

                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0

                        # Reset to best found so far for oscillating examples
                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    # Decrease checkpoint interval
                    k = max(k - size_decr, n_iter_min)
                    counter3 = 0

            # Track robust accuracy
            with torch.no_grad():
                pred = self.model(x_adv).argmax(1)
                still_correct = (pred == y) & initially_correct
                robust_acc = (
                    still_correct.float().sum() / initially_correct.float().sum()
                )
                robust_accuracies.append(robust_acc.item() * 100)

        return robust_accuracies


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Date and Time (UTC): 2025-11-17 17:43:08")
    print(f"Current User: WesleyCh3n")
    print(f"Using device: {device}\n")

    # Load data
    print("==> Loading CIFAR-10 test data...")
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Sample subset for convergence tracking (faster)
    indices = torch.randperm(len(testset))[: args.n_samples]
    testset_subset = torch.utils.data.Subset(testset, indices)
    test_loader = DataLoader(testset_subset, batch_size=args.n_samples, shuffle=False)

    # Load model
    print("==> Loading model...")
    model = ResNet18(num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded from: {args.checkpoint}\n")

    # Get batch
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)

    # Check initial clean accuracy
    with torch.no_grad():
        clean_pred = model(x).argmax(1)
        clean_acc = (clean_pred == y).float().mean().item() * 100
    print(f"Clean accuracy on sample: {clean_acc:.2f}%\n")

    # Track convergence
    tracker = ConvergenceTracker(model, device)

    print("Tracking PGD (α=2/255) convergence...")
    pgd_alpha2 = tracker.track_pgd_convergence(
        x, y, args.eps, alpha=2 / 255, max_iters=args.max_iters
    )

    print("Tracking PGD (α=1/255) convergence...")
    pgd_alpha1 = tracker.track_pgd_convergence(
        x, y, args.eps, alpha=1 / 255, max_iters=args.max_iters
    )

    print("Tracking APGD-CE convergence...")
    apgd = tracker.track_apgd_convergence(x, y, args.eps, max_iters=args.max_iters)

    # Plot
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))

    iterations = list(range(len(apgd)))

    # Plot robust accuracy (higher is better for defense, lower is better for attack)
    plt.plot(iterations, apgd, "b-", linewidth=2.5, label="APGD-CE", alpha=0.9)
    plt.plot(
        iterations, pgd_alpha2, "r--", linewidth=2.5, label="PGD (α=2/255)", alpha=0.9
    )
    plt.plot(
        iterations, pgd_alpha1, "g:", linewidth=2.5, label="PGD (α=1/255)", alpha=0.9
    )

    plt.xlabel("Iterations", fontsize=14, fontweight="bold")
    plt.ylabel("Robust Accuracy (%)", fontsize=14, fontweight="bold")
    plt.title(
        "Robust Accuracy vs. Iterations on PGD-AT Model\n(Lower is stronger attack)",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlim(0, args.max_iters)
    plt.ylim(0, 100)

    # Add reference line at 50% robust accuracy
    plt.axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    plt.text(5, 51, "50% robust accuracy", fontsize=9, alpha=0.6)

    # Find when each attack reaches 50% robust accuracy (50% broken)
    apgd_50_idx = next((i for i, v in enumerate(apgd) if v <= 50), None)
    pgd2_50_idx = next((i for i, v in enumerate(pgd_alpha2) if v <= 50), None)
    pgd1_50_idx = next((i for i, v in enumerate(pgd_alpha1) if v <= 50), None)

    if apgd_50_idx:
        plt.plot(apgd_50_idx, 50, "bo", markersize=8)
        plt.annotate(
            f"APGD: {apgd_50_idx} iter",
            xy=(apgd_50_idx, 50),
            xytext=(apgd_50_idx + 5, 45),
            fontsize=10,
            color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.6, lw=1.5),
        )

    if pgd2_50_idx:
        plt.plot(pgd2_50_idx, 50, "ro", markersize=8)
        plt.annotate(
            f"PGD (α=2/255): {pgd2_50_idx} iter",
            xy=(pgd2_50_idx, 50),
            xytext=(pgd2_50_idx - 20, 55),
            fontsize=10,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.6, lw=1.5),
        )

    if pgd1_50_idx:
        plt.plot(pgd1_50_idx, 50, "go", markersize=8)

    plt.tight_layout()

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "convergence_robust_acc.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to: {save_path}")

    save_path_png = os.path.join(args.save_dir, "convergence_robust_acc.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    print(f"Saved to: {save_path_png}")

    # Also create attack success rate version
    plt.figure(figsize=(10, 6))

    attack_success_apgd = [100 - x for x in apgd]
    attack_success_pgd2 = [100 - x for x in pgd_alpha2]
    attack_success_pgd1 = [100 - x for x in pgd_alpha1]

    plt.plot(
        iterations, attack_success_apgd, "b-", linewidth=2.5, label="APGD-CE", alpha=0.9
    )
    plt.plot(
        iterations,
        attack_success_pgd2,
        "r--",
        linewidth=2.5,
        label="PGD (α=2/255)",
        alpha=0.9,
    )
    plt.plot(
        iterations,
        attack_success_pgd1,
        "g:",
        linewidth=2.5,
        label="PGD (α=1/255)",
        alpha=0.9,
    )

    plt.xlabel("Iterations", fontsize=14, fontweight="bold")
    plt.ylabel("Attack Success Rate (%)", fontsize=14, fontweight="bold")
    plt.title(
        "Attack Success Rate vs. Iterations on PGD-AT Model\n(Higher is stronger attack)",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlim(0, args.max_iters)
    plt.ylim(0, 100)

    plt.tight_layout()

    save_path_success = os.path.join(args.save_dir, "convergence_attack_success.pdf")
    plt.savefig(save_path_success, dpi=300, bbox_inches="tight")
    print(f"Saved attack success version to: {save_path_success}")

    # Save data
    import json

    data = {
        "iterations": iterations,
        "apgd_ce_robust_acc": apgd,
        "pgd_alpha2_robust_acc": pgd_alpha2,
        "pgd_alpha1_robust_acc": pgd_alpha1,
        "apgd_ce_attack_success": attack_success_apgd,
        "pgd_alpha2_attack_success": attack_success_pgd2,
        "pgd_alpha1_attack_success": attack_success_pgd1,
        "apgd_50_iter": apgd_50_idx,
        "pgd2_50_iter": pgd2_50_idx,
        "pgd1_50_iter": pgd1_50_idx,
        "eps": args.eps,
        "max_iters": args.max_iters,
        "n_samples": args.n_samples,
        "clean_accuracy": clean_acc,
    }
    data_path = os.path.join(args.save_dir, "convergence_data.json")
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to: {data_path}")

    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    print(f"\nFinal robust accuracy (lower = stronger attack):")
    print(f"  APGD-CE:         {apgd[-1]:.2f}%")
    print(f"  PGD (α=2/255):   {pgd_alpha2[-1]:.2f}%")
    print(f"  PGD (α=1/255):   {pgd_alpha1[-1]:.2f}%")
    print(f"\nFinal attack success rate (higher = stronger attack):")
    print(f"  APGD-CE:         {attack_success_apgd[-1]:.2f}%")
    print(f"  PGD (α=2/255):   {attack_success_pgd2[-1]:.2f}%")
    print(f"  PGD (α=1/255):   {attack_success_pgd1[-1]:.2f}%")
    print(f"\nIterations to reach 50% robust accuracy:")
    print(f"  APGD-CE:         {apgd_50_idx if apgd_50_idx else 'Not reached'}")
    print(f"  PGD (α=2/255):   {pgd2_50_idx if pgd2_50_idx else 'Not reached'}")
    print(f"  PGD (α=1/255):   {pgd1_50_idx if pgd1_50_idx else 'Not reached'}")

    if apgd_50_idx and pgd2_50_idx:
        speedup = (pgd2_50_idx - apgd_50_idx) / pgd2_50_idx * 100
        print(
            f"\nAPGD-CE speedup: {speedup:.1f}% fewer iterations to reach 50% robust acc"
        )

    # APGD advantage
    final_gap = pgd_alpha2[-1] - apgd[-1]
    print(f"\nAPGD-CE advantage over PGD-50:")
    print(f"  {final_gap:.2f} percentage points lower robust accuracy")
    print(f"  (finds {final_gap:.2f}% more vulnerable examples)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track attack convergence (robust accuracy)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to adversarially trained model",
    )
    parser.add_argument(
        "--eps", type=float, default=8 / 255, help="perturbation budget"
    )
    parser.add_argument(
        "--max-iters", type=int, default=100, help="maximum iterations to track"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="number of samples to use"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./figures", help="directory to save figures"
    )

    args = parser.parse_args()
    main(args)
