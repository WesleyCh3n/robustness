import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attack import pgd


def evaluate_pgd_attack(
    model_path="resnet18_cifar10.pth",
    epsilon=8 / 255,
    alpha=2 / 255,
    iters=10,
    num_samples=1000,
):
    """
    Evaluate PGD attack on trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torchvision.models.resnet18(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    # Evaluate clean and adversarial accuracy
    clean_correct = 0
    adv_correct = 0
    total = 0

    # Store misclassified examples for visualization
    misclassified_examples = []

    for images, labels in testloader:
        if total >= num_samples:
            break

        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(labels).sum().item()

        # Generate adversarial examples
        adv_images = pgd(model, images, labels, epsilon, alpha, iters)

        # Adversarial accuracy
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(labels).sum().item()

        # Collect misclassified examples for visualization
        # Find examples that were correctly classified before but misclassified after attack
        clean_correct_mask = clean_predicted.eq(labels)
        adv_incorrect_mask = ~adv_predicted.eq(labels)
        successfully_attacked_mask = clean_correct_mask & adv_incorrect_mask
        successfully_attacked_indices = torch.where(successfully_attacked_mask)[0]

        for idx in successfully_attacked_indices:
            if len(misclassified_examples) >= 8:
                break
            misclassified_examples.append(
                {
                    "clean_image": images[idx],
                    "adv_image": adv_images[idx],
                    "true_label": labels[idx].item(),
                    "clean_pred": clean_predicted[idx].item(),
                    "adv_pred": adv_predicted[idx].item(),
                }
            )

        total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total

    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy (PGD): {adv_acc:.2f}%")
    print(f"Attack Success Rate: {100 - adv_acc:.2f}%")

    return clean_acc, adv_acc, misclassified_examples


def visualize_pgd_attack_from_examples(misclassified_examples):
    """
    Visualize PGD attack examples from evaluation results
    """
    if len(misclassified_examples) == 0:
        print("No misclassified examples found to visualize!")
        return None

    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Denormalize for visualization

    # Plot up to 4 examples
    num_examples = min(4, len(misclassified_examples))
    fig, axes = plt.subplots(2, num_examples, figsize=(3 * num_examples, 6))

    # Handle case when we have only one example
    if num_examples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_examples):
        example = misclassified_examples[i]

        # Denormalize images
        clean_img_vis = (example["clean_image"]).cpu()
        adv_img_vis = (example["adv_image"]).cpu()

        # Clean image
        axes[0, i].imshow(clean_img_vis.permute(1, 2, 0).numpy())
        clean_pred_label = classes[example["clean_pred"]]
        true_label = classes[example["true_label"]]
        axes[0, i].set_title(f"Clean: {clean_pred_label}\n(True: {true_label})")
        axes[0, i].axis("off")

        # Adversarial image
        axes[1, i].imshow(adv_img_vis.permute(1, 2, 0).numpy())
        adv_pred_label = classes[example["adv_pred"]]
        axes[1, i].set_title(f"Adversarial: {adv_pred_label}\n(True: {true_label})")
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig("pgd_attack_misclassified_examples.png", dpi=150, bbox_inches="tight")
    print(
        f"Visualization of {num_examples} misclassified examples saved to pgd_attack_misclassified_examples.png"
    )

    return fig


if __name__ == "__main__":
    # Run evaluation and collect misclassified examples
    clean_acc, adv_acc, misclassified_examples = evaluate_pgd_attack(
        model_path="./weights/resnet18_cifar10.pth",
        epsilon=8 / 255,
        alpha=3 / 255,
        iters=3,
        num_samples=1000,
    )

    # Create visualization using the actual misclassified examples from evaluation
    visualize_pgd_attack_from_examples(misclassified_examples)
