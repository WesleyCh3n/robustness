"""
Analyze and generate tables/figures from evaluation results
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
    print("\n" + "=" * 70)
    print("Table 1: Robust Accuracy Comparison on CIFAR-10 Test Set")
    print("=" * 70)
    print(
        f"{'Model':<20} {'Clean Acc':<12} {'FGSM':<12} {'PGD-50':<12} {'PGD-100':<12} {'AutoAttack':<12}"
    )
    print("-" * 70)

    # Standard model
    print(
        f"{'Standard':<20} {standard_results['clean']:>10.1f}% {standard_results['fgsm']:>10.1f}% "
        f"{standard_results['pgd50']:>10.1f}% {standard_results['pgd100']:>10.1f}% "
        f"{standard_results.get('autoattack', 0):>10.1f}%"
    )

    # Adversarially trained model
    print(
        f"{'PGD-AT':<20} {adversarial_results['clean']:>10.1f}% {adversarial_results['fgsm']:>10.1f}% "
        f"{adversarial_results['pgd50']:>10.1f}% {adversarial_results['pgd100']:>10.1f}% "
        f"{adversarial_results.get('autoattack', 0):>10.1f}%"
    )
    print("=" * 70)

    # Calculate ensemble gap
    if "autoattack" in adversarial_results:
        gap_abs = adversarial_results["pgd50"] - adversarial_results["autoattack"]
        gap_rel = (gap_abs / adversarial_results["pgd50"]) * 100
        print(
            f"\nEnsemble Gap (PGD-50 to AutoAttack): {gap_abs:.1f}% absolute, {gap_rel:.1f}% relative"
        )


def generate_convergence_plot(save_path="./figures/convergence.png"):
    """Generate convergence plot (placeholder - would need actual data)"""
    # This is a placeholder - in practice you'd collect this data during evaluation
    iterations = np.arange(0, 101)

    # Simulated data (replace with actual data collection)
    apgd_success = 100 * (1 - np.exp(-iterations / 30))
    pgd_success = 100 * (1 - np.exp(-iterations / 40))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, apgd_success, label="APGD-CE", linewidth=2)
    plt.plot(
        iterations, pgd_success, label="PGD (α=2/255)", linewidth=2, linestyle="--"
    )
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Attack Success Rate (%)", fontsize=12)
    plt.title("Attack Success Rate vs. Iterations", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nConvergence plot saved to: {save_path}")
    plt.close()


def main(args):
    # Load results
    print("Loading results...")

    standard_results = None
    adversarial_results = None

    if args.standard_results:
        standard_results = load_results(args.standard_results)
        print(f"Loaded standard model results from: {args.standard_results}")

    if args.adversarial_results:
        adversarial_results = load_results(args.adversarial_results)
        print(f"Loaded adversarial model results from: {args.adversarial_results}")

    # Generate tables
    if standard_results and adversarial_results:
        generate_main_table(standard_results, adversarial_results)

    # Generate plots
    if args.generate_plots:
        print("\nGenerating plots...")
        generate_convergence_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument(
        "--standard-results", type=str, help="path to standard model results JSON"
    )
    parser.add_argument(
        "--adversarial-results", type=str, help="path to adversarial model results JSON"
    )
    parser.add_argument(
        "--generate-plots", action="store_true", help="generate convergence plots"
    )

    args = parser.parse_args()
    main(args)
