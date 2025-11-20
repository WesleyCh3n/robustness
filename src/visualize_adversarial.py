"""
Visualize adversarial examples to show imperceptibility
Generate Figure: Clean vs Adversarial vs Perturbation

Author: WesleyCh3n
Date: 2025-11-17 18:25:28 UTC
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attacks.apgd_ce import APGDAttack
from attacks.pgd import PGDAttack
from models.resnet import ResNet18


def denormalize_cifar10(img):
    """Convert from [0,1] tensor to numpy image for display"""
    img = img.cpu().detach().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img


def visualize_samples(model, test_loader, device, save_dir, n_samples=8):
    """
    Visualize clean images, adversarial examples, and perturbations

    Args:
        model: Target model
        test_loader: DataLoader
        device: Device
        save_dir: Directory to save figures
        n_samples: Number of samples to visualize
    """
    model.eval()

    # Get a batch of images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Select only correctly classified examples for the clean images
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(1)
        correct_mask = preds == labels

    # Get correctly classified examples (we'll filter for successful attacks later)
    correct_indices = correct_mask.nonzero().squeeze()
    if correct_indices.dim() == 0:
        correct_indices = correct_indices.unsqueeze(0)

    # Get more samples than needed to ensure we have enough after filtering
    max_samples = min(len(correct_indices), n_samples * 4)
    correct_indices = correct_indices[:max_samples]
    images = images[correct_indices]
    labels = labels[correct_indices]

    # Class names
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

    # Generate adversarial examples with PGD-50
    print("Generating PGD-50 adversarial examples...")
    pgd = PGDAttack(eps=8 / 255, alpha=2 / 255, steps=50, random_start=True)
    adv_images_pgd = pgd.attack(model, images, labels)

    # Generate adversarial examples with APGD-CE
    print("Generating APGD-CE adversarial examples...")
    apgd = APGDAttack(
        eps=8 / 255, n_iter=100, n_restarts=1, verbose=False, device=device
    )
    adv_images_apgd = apgd.attack(model, images, labels)

    # Get predictions
    with torch.no_grad():
        pred_clean = model(images).argmax(1)
        pred_pgd = model(adv_images_pgd).argmax(1)
        pred_apgd = model(adv_images_apgd).argmax(1)

    # Filter for samples where EITHER attack succeeded (misclassified)
    attack_success_mask = (pred_pgd != labels) | (pred_apgd != labels)
    success_indices = attack_success_mask.nonzero().squeeze()

    if success_indices.dim() == 0:
        success_indices = success_indices.unsqueeze(0)

    # Take only n_samples successful attacks
    n_success = min(len(success_indices), n_samples)
    if n_success == 0:
        print("Warning: No successful attacks found!")
        return

    success_indices = success_indices[:n_success]

    # Filter all data to only successful attacks
    images = images[success_indices]
    labels = labels[success_indices]
    adv_images_pgd = adv_images_pgd[success_indices]
    adv_images_apgd = adv_images_apgd[success_indices]
    pred_clean = pred_clean[success_indices]
    pred_pgd = pred_pgd[success_indices]
    pred_apgd = pred_apgd[success_indices]

    # Update n_samples to actual number
    n_samples = n_success

    # Compute perturbations
    pert_pgd = adv_images_pgd - images
    pert_apgd = adv_images_apgd - images

    # Create visualization for PGD
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        # Clean image
        axes[i, 0].imshow(denormalize_cifar10(images[i]))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"Clean: {class_names[pred_clean[i]]}\n(True: {class_names[labels[i]]})",
            fontsize=10,
            color="green",
        )

        # PGD adversarial
        axes[i, 1].imshow(denormalize_cifar10(adv_images_pgd[i]))
        axes[i, 1].axis("off")
        color = "red" if pred_pgd[i] != labels[i] else "green"
        axes[i, 1].set_title(
            f"PGD-50: {class_names[pred_pgd[i]]}\n(Adv)", fontsize=10, color=color
        )

        # PGD perturbation (amplified for visibility)
        pert_display = denormalize_cifar10(pert_pgd[i] * 10 + 0.5)
        axes[i, 2].imshow(pert_display)
        axes[i, 2].axis("off")
        max_pert = pert_pgd[i].abs().max().item()
        axes[i, 2].set_title(f"Perturbation ×10\n(max={max_pert:.4f})", fontsize=10)

        # Difference (enhanced)
        diff = (adv_images_pgd[i] - images[i]).abs()
        diff_display = diff.cpu().detach().numpy().transpose(1, 2, 0)
        diff_display = diff_display / diff_display.max()  # Normalize to [0,1]
        axes[i, 3].imshow(diff_display, cmap="hot")
        axes[i, 3].axis("off")
        axes[i, 3].set_title("Absolute Diff\n(normalized)", fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "adversarial_examples_pgd.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved PGD visualization to: {save_path}")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create visualization for APGD
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        # Clean image
        axes[i, 0].imshow(denormalize_cifar10(images[i]))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"Clean: {class_names[pred_clean[i]]}\n(True: {class_names[labels[i]]})",
            fontsize=10,
            color="green",
        )

        # APGD adversarial
        axes[i, 1].imshow(denormalize_cifar10(adv_images_apgd[i]))
        axes[i, 1].axis("off")
        color = "red" if pred_apgd[i] != labels[i] else "green"
        axes[i, 1].set_title(
            f"APGD-CE: {class_names[pred_apgd[i]]}\n(Adv)", fontsize=10, color=color
        )

        # APGD perturbation (amplified for visibility)
        pert_display = denormalize_cifar10(pert_apgd[i] * 10 + 0.5)
        axes[i, 2].imshow(pert_display)
        axes[i, 2].axis("off")
        max_pert = pert_apgd[i].abs().max().item()
        axes[i, 2].set_title(f"Perturbation ×10\n(max={max_pert:.4f})", fontsize=10)

        # Difference (enhanced)
        diff = (adv_images_apgd[i] - images[i]).abs()
        diff_display = diff.cpu().detach().numpy().transpose(1, 2, 0)
        diff_display = diff_display / diff_display.max()  # Normalize to [0,1]
        axes[i, 3].imshow(diff_display, cmap="hot")
        axes[i, 3].axis("off")
        axes[i, 3].set_title("Absolute Diff\n(normalized)", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "adversarial_examples_apgd.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved APGD visualization to: {save_path}")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create a compact comparison (min of 3 or n_samples examples side-by-side)
    n_compact = min(3, n_samples)
    fig, axes = plt.subplots(n_compact, 5, figsize=(15, 3 * n_compact))

    # Handle case where n_compact == 1 (axes won't be 2D)
    if n_compact == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_compact):
        # Clean
        axes[i, 0].imshow(denormalize_cifar10(images[i]))
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Clean Image", fontsize=12, fontweight="bold")
        axes[i, 0].text(
            0.5,
            -0.15,
            f"{class_names[labels[i]]}",
            transform=axes[i, 0].transAxes,
            ha="center",
            fontsize=10,
        )

        # PGD adversarial
        axes[i, 1].imshow(denormalize_cifar10(adv_images_pgd[i]))
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("PGD-50\nAdversarial", fontsize=12, fontweight="bold")
        color = "red" if pred_pgd[i] != labels[i] else "green"
        axes[i, 1].text(
            0.5,
            -0.15,
            f"{class_names[pred_pgd[i]]}",
            transform=axes[i, 1].transAxes,
            ha="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

        # PGD perturbation
        pert_display = denormalize_cifar10(pert_pgd[i] * 10 + 0.5)
        axes[i, 2].imshow(pert_display)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title(
                "PGD-50\nPerturbation ×10", fontsize=12, fontweight="bold"
            )

        # APGD adversarial
        axes[i, 3].imshow(denormalize_cifar10(adv_images_apgd[i]))
        axes[i, 3].axis("off")
        if i == 0:
            axes[i, 3].set_title("APGD-CE\nAdversarial", fontsize=12, fontweight="bold")
        color = "red" if pred_apgd[i] != labels[i] else "green"
        axes[i, 3].text(
            0.5,
            -0.15,
            f"{class_names[pred_apgd[i]]}",
            transform=axes[i, 3].transAxes,
            ha="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

        # APGD perturbation
        pert_display = denormalize_cifar10(pert_apgd[i] * 10 + 0.5)
        axes[i, 4].imshow(pert_display)
        axes[i, 4].axis("off")
        if i == 0:
            axes[i, 4].set_title(
                "APGD-CE\nPerturbation ×10", fontsize=12, fontweight="bold"
            )

    plt.suptitle(
        "Adversarial Examples with ε=8/255",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(save_dir, "adversarial_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison to: {save_path}")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Print statistics
    print("\n" + "=" * 70)
    print("PERTURBATION STATISTICS")
    print("=" * 70)
    print("PGD-50:")
    print(
        f"  Max L-inf norm: {pert_pgd.abs().max().item():.6f} (budget: {8 / 255:.6f})"
    )
    print(
        f"  Mean L-inf norm: {pert_pgd.abs().view(n_samples, -1).max(1)[0].mean().item():.6f}"
    )
    print(f"  Attack success: {(pred_pgd != labels).sum().item()}/{n_samples}")
    print("\nAPGD-CE:")
    print(
        f"  Max L-inf norm: {pert_apgd.abs().max().item():.6f} (budget: {8 / 255:.6f})"
    )
    print(
        f"  Mean L-inf norm: {pert_apgd.abs().view(n_samples, -1).max(1)[0].mean().item():.6f}"
    )
    print(f"  Attack success: {(pred_apgd != labels).sum().item()}/{n_samples}")
    print("=" * 70)
    print(
        "\nNote: Only showing examples where attacks successfully caused misclassification."
    )
    print("      Perturbations are amplified 10× in visualizations for visibility.")
    print("      At actual scale (ε=8/255≈0.031), they are imperceptible to humans.")


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Date and Time (UTC): 2025-11-17 18:25:28")
    print("Current User: WesleyCh3n")
    print(f"Using device: {device}\n")

    # Load data
    print("==> Loading CIFAR-10 test data...")
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Use larger batch to ensure we get enough correctly classified examples
    test_loader = DataLoader(testset, batch_size=128, shuffle=True)

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

    # Generate visualizations
    print("==> Generating adversarial example visualizations...")
    visualize_samples(
        model, test_loader, device, args.save_dir, n_samples=args.n_samples
    )

    print(f"\nVisualizations saved to: {args.save_dir}/")
    print("Generated files:")
    print("  - adversarial_examples_pgd.pdf/png (detailed PGD examples)")
    print("  - adversarial_examples_apgd.pdf/png (detailed APGD examples)")
    print("  - adversarial_comparison.pdf/png (compact comparison)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize adversarial examples")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to adversarially trained model",
    )
    parser.add_argument(
        "--n-samples", type=int, default=8, help="number of samples to visualize"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./figures", help="directory to save figures"
    )

    args = parser.parse_args()
    main(args)
