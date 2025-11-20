"""
Evaluate APGD-CE and baseline attacks on trained models
Focused reproduction study

Author: WesleyCh3n
Date: 2025-11-17
"""

import argparse
import json
import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.apgd_ce import APGDAttack
from attacks.pgd import FGSMAttack, PGDAttack
from models.resnet import ResNet18


def evaluate_attack(
    model, test_loader, attacker, device, attack_name, track_convergence=False
):
    """
    Evaluate model against an attack

    Args:
        model: Target model
        test_loader: DataLoader for test set
        attacker: Attack object
        device: Device to run on
        attack_name: Name for progress bar
        track_convergence: Whether to track iteration-wise success (only for APGD/PGD)

    Returns:
        robust_acc: Robust accuracy (%)
        convergence_data: Dict with iteration-wise data (if track_convergence=True)
    """
    model.eval()
    correct = 0
    total = 0

    convergence_data = None
    if track_convergence and hasattr(attacker, "n_iter"):
        convergence_data = {"iterations": [], "success_rate": []}

    start_time = time.time()

    for inputs, targets in tqdm(test_loader, desc=f"Evaluating {attack_name}"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples
        adv_inputs = attacker.attack(model, inputs, targets)

        # Evaluate
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    elapsed_time = time.time() - start_time
    robust_acc = 100.0 * correct / total

    print(f"{attack_name}: {robust_acc:.2f}% robust accuracy ({elapsed_time:.1f}s)")

    return robust_acc, convergence_data


def evaluate_clean(model, test_loader, device):
    """Evaluate clean accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating clean"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    return acc


def evaluate_per_class(model, test_loader, attackers_dict, device):
    """
    Evaluate per-class robust accuracy

    Args:
        model: Target model
        test_loader: Must have batch_size=1 or handle carefully
        attackers_dict: Dict of {name: attacker}
        device: Device

    Returns:
        per_class_results: Dict with per-class accuracies
    """
    model.eval()

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Initialize counters
    per_class_results = {
        "classes": class_names,
        "clean": {cls: {"correct": 0, "total": 0} for cls in class_names},
    }

    for attack_name in attackers_dict.keys():
        per_class_results[attack_name] = {
            cls: {"correct": 0, "total": 0} for cls in class_names
        }

    # Collect all data first
    all_data = []
    for inputs, targets in test_loader:
        all_data.append((inputs, targets))

    # Evaluate clean
    print("Evaluating per-class clean accuracy...")
    for inputs, targets in tqdm(all_data):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(len(targets)):
                cls_name = class_names[targets[i].item()]
                per_class_results["clean"][cls_name]["total"] += 1
                if predicted[i] == targets[i]:
                    per_class_results["clean"][cls_name]["correct"] += 1

    # Evaluate each attack
    for attack_name, attacker in attackers_dict.items():
        print(f"Evaluating per-class {attack_name}...")
        for inputs, targets in tqdm(all_data):
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate adversarial examples
            adv_inputs = attacker.attack(model, inputs, targets)

            with torch.no_grad():
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)

                for i in range(len(targets)):
                    cls_name = class_names[targets[i].item()]
                    per_class_results[attack_name][cls_name]["total"] += 1
                    if predicted[i] == targets[i]:
                        per_class_results[attack_name][cls_name]["correct"] += 1

    # Compute accuracies
    results_summary = {"classes": class_names}
    for attack_name in ["clean"] + list(attackers_dict.keys()):
        results_summary[attack_name] = {}
        for cls_name in class_names:
            total = per_class_results[attack_name][cls_name]["total"]
            correct = per_class_results[attack_name][cls_name]["correct"]
            acc = 100.0 * correct / total if total > 0 else 0
            results_summary[attack_name][cls_name] = acc

    return results_summary


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("\n==> Preparing data..")
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    print("==> Loading model..")
    model = ResNet18(num_classes=10).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Evaluate
    results = {}

    # Clean accuracy
    print("\n" + "=" * 70)
    print("Evaluating Clean Accuracy")
    print("=" * 70)
    clean_acc = evaluate_clean(model, test_loader, device)
    results["clean"] = clean_acc
    print(f"Clean accuracy: {clean_acc:.2f}%")

    # FGSM
    print("\n" + "=" * 70)
    print("Evaluating FGSM")
    print("=" * 70)
    fgsm = FGSMAttack(eps=args.eps)
    fgsm_acc, _ = evaluate_attack(model, test_loader, fgsm, device, "FGSM")
    results["fgsm"] = fgsm_acc

    # PGD-50 (standard baseline)
    print("\n" + "=" * 70)
    print("Evaluating PGD-50 (α=2/255)")
    print("=" * 70)
    pgd50 = PGDAttack(eps=args.eps, alpha=2 / 255, steps=50, random_start=False)
    pgd50_acc, _ = evaluate_attack(model, test_loader, pgd50, device, "PGD-50")
    results["pgd50"] = pgd50_acc

    # PGD-100 with different step sizes
    print("\n" + "=" * 70)
    print("Evaluating PGD-100 (α=2/255)")
    print("=" * 70)
    pgd100_std = PGDAttack(eps=args.eps, alpha=2 / 255, steps=100, random_start=False)
    pgd100_std_acc, _ = evaluate_attack(
        model, test_loader, pgd100_std, device, "PGD-100 (α=2/255)"
    )
    results["pgd100_alpha2"] = pgd100_std_acc

    print("\n" + "=" * 70)
    print("Evaluating PGD-100 (α=1/255)")
    print("=" * 70)
    pgd100_small = PGDAttack(eps=args.eps, alpha=1 / 255, steps=100, random_start=False)
    pgd100_small_acc, _ = evaluate_attack(
        model, test_loader, pgd100_small, device, "PGD-100 (α=1/255)"
    )
    results["pgd100_alpha1"] = pgd100_small_acc

    # APGD-CE (our main focus)
    print("\n" + "=" * 70)
    print("Evaluating APGD-CE (100 iterations)")
    print("=" * 70)
    apgd = APGDAttack(eps=args.eps, n_iter=100, verbose=args.verbose)
    apgd_acc, _ = evaluate_attack(model, test_loader, apgd, device, "APGD-CE")
    results["apgd_ce"] = apgd_acc

    # Compute gaps
    results["gap_pgd50_to_apgd"] = pgd50_acc - apgd_acc
    results["gap_pgd100_to_apgd"] = pgd100_std_acc - apgd_acc

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = os.path.join(args.save_dir, "eval_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"Clean Accuracy:              {clean_acc:.2f}%")
    print(f"FGSM:                        {fgsm_acc:.2f}%")
    print(f"PGD-50 (α=2/255):           {pgd50_acc:.2f}%")
    print(f"PGD-100 (α=2/255):          {pgd100_std_acc:.2f}%")
    print(f"PGD-100 (α=1/255):          {pgd100_small_acc:.2f}%")
    print(f"APGD-CE (100 iter):          {apgd_acc:.2f}%")
    print("-" * 70)
    print(f"Gap (PGD-50 → APGD-CE):      {results['gap_pgd50_to_apgd']:.2f} pp")
    print(f"Gap (PGD-100 → APGD-CE):     {results['gap_pgd100_to_apgd']:.2f} pp")
    if pgd50_acc > 0:
        print(
            f"Relative improvement:        {100 * results['gap_pgd50_to_apgd'] / pgd50_acc:.1f}%"
        )
        print("=" * 70)

    print(f"\nResults saved to: {result_file}")

    # Per-class evaluation if requested
    if args.per_class:
        print("\n" + "=" * 70)
        print("Evaluating Per-Class Robust Accuracy")
        print("=" * 70)
        attackers_dict = {"PGD-50": pgd50, "APGD-CE": apgd}
        per_class_results = evaluate_per_class(
            model, test_loader, attackers_dict, device
        )

        per_class_file = os.path.join(args.save_dir, "per_class_results.json")
        with open(per_class_file, "w") as f:
            json.dump(per_class_results, f, indent=4)

        print(f"Per-class results saved to: {per_class_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate APGD-CE and baselines")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="path to model checkpoint"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--eps", type=float, default=8 / 255, help="perturbation budget"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="verbose output from APGD-CE"
    )
    parser.add_argument(
        "--per-class", action="store_true", help="compute per-class robust accuracy"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results", help="directory to save results"
    )

    args = parser.parse_args()
    main(args)
