import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attack import pgd


def train_standard(model, trainloader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 15], gamma=0.1
    )

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], "
                    f"Loss: {running_loss / 100:.4f}, Acc: {100.0 * correct / total:.2f}%"
                )
                running_loss = 0.0

        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{epochs}] completed. Accuracy: {100.0 * correct / total:.2f}%"
        )

    return model


def train_adversarial(model, trainloader, device, epochs=20, epsilon=8 / 255):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 15], gamma=0.1
    )

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total = 0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            model.eval()
            adv_images = pgd(
                model, images, labels, epsilon=epsilon, alpha=2 / 255, iters=10
            )
            model.train()

            optimizer.zero_grad()

            outputs_adv = model(adv_images)
            loss = criterion(outputs_adv, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                outputs_clean = model(images)
                _, predicted_clean = outputs_clean.max(1)
                _, predicted_adv = outputs_adv.max(1)

                total += labels.size(0)
                correct_clean += predicted_clean.eq(labels).sum().item()
                correct_adv += predicted_adv.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], "
                    f"Loss: {running_loss / 100:.4f}, "
                    f"Clean Acc: {100.0 * correct_clean / total:.2f}%, "
                    f"Adv Acc: {100.0 * correct_adv / total:.2f}%"
                )
                running_loss = 0.0

        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{epochs}] completed. "
            f"Clean Acc: {100.0 * correct_clean / total:.2f}%, "
            f"Adv Acc: {100.0 * correct_adv / total:.2f}%"
        )

    return model


def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    print("\n" + "=" * 60)
    print("Training STANDARD ResNet-18")
    print("=" * 60 + "\n")

    model_standard = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model_standard = model_standard.to(device)

    model_standard = train_standard(model_standard, trainloader, device, epochs=20)

    print("\nEvaluating standard model...")
    evaluate_model(model_standard, testloader, device)

    torch.save(model_standard.state_dict(), "weights/resnet18_cifar10_standard.pth")
    print("Saved standard model to resnet18_cifar10_standard.pth\n")

    print("\n" + "=" * 60)
    print("Training ROBUST (Adversarially Trained) ResNet-18")
    print("=" * 60 + "\n")

    model_robust = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model_robust = model_robust.to(device)

    model_robust = train_adversarial(
        model_robust, trainloader, device, epochs=20, epsilon=8 / 255
    )

    print("\nEvaluating robust model...")
    evaluate_model(model_robust, testloader, device)

    torch.save(model_robust.state_dict(), "weights/resnet18_cifar10_robust.pth")
    print("Saved robust model to resnet18_cifar10_robust.pth\n")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(
        "\nYou can now run main.py to evaluate both models with PGD and AutoPGD attacks."
    )


if __name__ == "__main__":
    main()
