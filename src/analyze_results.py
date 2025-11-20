"""
Analyze evaluation results and generate tables
Focused APGD-CE reproduction study

Author: WesleyCh3n
Date: 2025-11-17
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(result_file):
    """Load results from JSON file"""
    with open(result_file, "r") as f:
        return json.load(f)


def generate_main_table(standard_results, adversarial_results):
    """Generate Table 1: Main results comparison"""
    print("\n" + "=" * 80)
    print("Table 1: Robust Accuracy Comparison on CIFAR-10 Test Set (ε = 8/255)")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Clean':<10} {'FGSM':<10} {'PGD-50':<10} {'PGD-100':<10} {'APGD-CE':<10}"
    )
    print(
        f"{'':<20} {'Acc':<10} {'(8/255)':<10} {'(2/255)':<10} {'(2/255)':<10} {'(100it)':<10}"
    )
    print("-" * 80)

    # Standard model
    print(
        f"{'Standard':<20} {standard_results['clean']:>8.1f}% {standard_results['fgsm']:>8.1f}% "
        f"{standard_results['pgd50']:>8.1f}% {standard_results['pgd100_alpha2']:>8.1f}% "
        f"{standard_results.get('apgd_ce', 0):>8.1f}%"
    )

    # Adversarially trained model
    print(
        f"{'PGD-AT':<20} {adversarial_results['clean']:>8.1f}% {adversarial_results['fgsm']:>8.1f}% "
        f"{adversarial_results['pgd50']:>8.1f}% {adversarial_results['pgd100_alpha2']:>8.1f}% "
        f"{adversarial_results.get('apgd_ce', 0):>8.1f}%"
    )
    print("=" * 80)

    # Calculate ensemble gap
    if "apgd_ce" in adversarial_results:
        gap_abs = adversarial_results["gap_pgd50_to_apgd"]
        gap_rel = 100 * gap_abs / adversarial_results["pgd50"]
        print(f"\nAPGD-CE Improvement over PGD-50:")
        print(f"  Absolute gap: {gap_abs:.2f} percentage points")
        print(f"  Relative improvement: {gap_rel:.1f}%")


def generate_step_size_comparison_table(adversarial_results):
    """Generate Table 2: PGD step size comparison"""
    print("\n" + "=" * 60)
    print("Table 2: Effect of PGD Step Size (PGD-AT Model, 100 iterations)")
    print("=" * 60)
    print(f"{'Step Size (α)':<20} {'Robust Accuracy':<20} {'Notes':<20}")
    print("-" * 60)

    print(
        f"{'1/255':<20} {adversarial_results['pgd100_alpha1']:>17.1f}% {'Too conservative':<20}"
    )
    print(
        f"{'2/255 (standard)':<20} {adversarial_results['pgd100_alpha2']:>17.1f}% {'Common baseline':<20}"
    )
    print(
        f"{'Adaptive (APGD-CE)':<20} {adversarial_results['apgd_ce']:>17.1f}% {'Parameter-free':<20}"
    )
    print("=" * 60)


def generate_per_class_table(per_class_file):
    """Generate per-class robust accuracy table"""
    with open(per_class_file, "r") as f:
        per_class_data = json.load(f)

    print("\n" + "=" * 70)
    print("Table 3: Per-Class Robust Accuracy (PGD-AT Model)")
    print("=" * 70)
    print(f"{'Class':<15} {'Clean Acc':<15} {'PGD-50':<15} {'APGD-CE':<15} {'Gap':<10}")
    print("-" * 70)

    for cls in per_class_data["classes"]:
        clean_acc = per_class_data["clean"][cls]
        pgd_acc = per_class_data["PGD-50"][cls]
        apgd_acc = per_class_data["APGD-CE"][cls]
        gap = pgd_acc - apgd_acc

        print(
            f"{cls:<15} {clean_acc:>13.1f}% {pgd_acc:>13.1f}% {apgd_acc:>13.1f}% {gap:>8.1f}pp"
        )

    # Average
    avg_clean = np.mean(
        [per_class_data["clean"][cls] for cls in per_class_data["classes"]]
    )
    avg_pgd = np.mean(
        [per_class_data["PGD-50"][cls] for cls in per_class_data["classes"]]
    )
    avg_apgd = np.mean(
        [per_class_data["APGD-CE"][cls] for cls in per_class_data["classes"]]
    )
    avg_gap = avg_pgd - avg_apgd

    print("-" * 70)
    print(
        f"{'Average':<15} {avg_clean:>13.1f}% {avg_pgd:>13.1f}% {avg_apgd:>13.1f}% {avg_gap:>8.1f}pp"
    )
    print("=" * 70)


def main(args):
    print("=" * 80)
    print("APGD-CE Reproduction Study - Results Analysis")
    print("Author: WesleyCh3n")
    print("Date: 2025-11-17")
    print("=" * 80)

    # Load results
    print("\nLoading results...")

    standard_results = None
    adversarial_results = None

    if args.standard_results:
        standard_results = load_results(args.standard_results)
        print(f"✓ Loaded standard model results from: {args.standard_results}")

    if args.adversarial_results:
        adversarial_results = load_results(args.adversarial_results)
        print(f"✓ Loaded adversarial model results from: {args.adversarial_results}")

    # Generate tables
    if standard_results and adversarial_results:
        generate_main_table(standard_results, adversarial_results)
        generate_step_size_comparison_table(adversarial_results)

    # Per-class analysis
    if args.per_class_results:
        generate_per_class_table(args.per_class_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze APGD-CE evaluation results")
    parser.add_argument(
        "--standard-results", type=str, help="path to standard model results JSON"
    )
    parser.add_argument(
        "--adversarial-results", type=str, help="path to adversarial model results JSON"
    )
    parser.add_argument(
        "--per-class-results", type=str, help="path to per-class results JSON"
    )

    args = parser.parse_args()
    main(args)
