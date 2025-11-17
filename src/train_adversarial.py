"""
Adversarial training (PGD-AT) script for CIFAR-10
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from attacks.pgd import PGDAttack
from models.resnet import ResNet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger


def train_epoch_adversarial(
    model, train_loader, optimizer, criterion, device, epoch, attacker
):
    """Train for one epoch with adversarial examples"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples
        model.eval()  # Set to eval mode for attack
        adv_inputs = attacker.attack(model, inputs, targets)
        model.train()  # Back to train mode

        optimizer.zero_grad()
        outputs = model(adv_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            {"loss": train_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
        )

    return train_loss / len(train_loader), 100.0 * correct / total


def test(model, test_loader, criterion, device):
    """Test the model on clean examples"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    return test_loss / len(test_loader), acc


def test_adversarial(model, test_loader, device, attacker):
    """Test the model on adversarial examples"""
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in tqdm(test_loader, desc="Testing PGD robustness"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples
        adv_inputs = attacker.attack(model, inputs, targets)

        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    return acc


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    print("==> Building model..")
    model = ResNet18(num_classes=10).to(device)

    # Adversarial attacker for training
    train_attacker = PGDAttack(
        eps=args.eps, alpha=args.alpha, steps=args.attack_steps, random_start=True
    )

    # Adversarial attacker for evaluation (stronger)
    eval_attacker = PGDAttack(
        eps=args.eps,
        alpha=args.alpha,
        steps=50,  # More steps for evaluation
        random_start=True,
    )

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logger
    logger = Logger(os.path.join(args.save_dir, "log.txt"))

    # Training
    print("==> Starting adversarial training..")
    best_clean_acc = 0
    best_robust_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_adversarial(
            model, train_loader, optimizer, criterion, device, epoch, train_attacker
        )
        test_loss, test_acc = test(model, test_loader, criterion, device)

        # Evaluate robustness every 10 epochs
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            robust_acc = test_adversarial(model, test_loader, device, eval_attacker)
            logger.log(
                f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | "
                f"Train Acc: {train_acc:.2f}% | Clean Acc: {test_acc:.2f}% | "
                f"Robust Acc: {robust_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.4f}"
            )

            # Save best model based on robust accuracy
            if robust_acc > best_robust_acc:
                print(f"Saving best robust model.. (robust acc: {robust_acc:.2f}%)")
                state = {
                    "model": model.state_dict(),
                    "clean_acc": test_acc,
                    "robust_acc": robust_acc,
                    "epoch": epoch,
                }
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(state, os.path.join(args.save_dir, "best_robust_model.pth"))
                best_robust_acc = robust_acc
                best_clean_acc = test_acc
        else:
            logger.log(
                f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | "
                f"Train Acc: {train_acc:.2f}% | Clean Acc: {test_acc:.2f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.4f}"
            )

        scheduler.step()

        # Save latest checkpoint
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, os.path.join(args.save_dir, "latest_model.pth"))

    print(f"Best clean accuracy: {best_clean_acc:.2f}%")
    print(f"Best robust accuracy: {best_robust_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Training on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--eps", type=float, default=8 / 255, help="perturbation budget"
    )
    parser.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size")
    parser.add_argument(
        "--attack-steps", type=int, default=10, help="PGD steps during training"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/adversarial",
        help="directory to save checkpoints",
    )

    args = parser.parse_args()
    main(args)
