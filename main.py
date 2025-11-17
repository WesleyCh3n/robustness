import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attack import autopgd, pgd
from evaluation import evaluate_attack


def load_cifar10_test(num_samples=1000):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    print(f"Loaded CIFAR-10 test set with {len(testset)} samples.")
    testset.data = testset.data[:num_samples]
    testset.targets = testset.targets[:num_samples]

    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return testloader


def load_model(model_path="./weights/resnet18_cifar10.pth"):
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {model_path}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading CIFAR-10 test dataset...")
    testloader = load_cifar10_test(num_samples=10000)

    for model_path in [
        "./weights/resnet18_cifar10_standard.pth",
        "./weights/resnet18_cifar10_robust.pth",
    ]:
        model = load_model(model_path).to(device)
        model.eval()

        print("Running PGD attack...")

        def pgd_wrapper(m, x, y):
            return pgd(m, x, y, epsilon=2 / 255, alpha=3 / 255, iters=10)

        pgd_results = evaluate_attack(
            model, testloader, pgd_wrapper, "PGD (10 iters)", device
        )

        print("Running AutoPGD attack...")

        def autopgd_wrapper(m, x, y):
            return autopgd(m, x, y, epsilon=2 / 255, iters=100, n_restarts=1)

        autopgd_results = evaluate_attack(
            model,
            testloader,
            autopgd_wrapper,
            "AutoPGD (100 iters, 5 restarts)",
            device,
        )

        print(f"\n{'=' * 60}")
        print(f"\n{pgd_results['attack']}:")
        print(f"  Clean Accuracy:        {pgd_results['clean_accuracy']:.2f}%")
        print(f"  Adversarial Accuracy:  {pgd_results['adversarial_accuracy']:.2f}%")
        print(f"  Attack Success Rate:   {pgd_results['attack_success_rate']:.2f}%")

        print(f"\n{autopgd_results['attack']}:")
        print(f"  Clean Accuracy:        {autopgd_results['clean_accuracy']:.2f}%")
        print(
            f"  Adversarial Accuracy:  {autopgd_results['adversarial_accuracy']:.2f}%"
        )
        print(f"  Attack Success Rate:   {autopgd_results['attack_success_rate']:.2f}%")

        print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    # from autoattack import AutoAttack
    # model = load_model("./resnet18_cifar10_robust.pth").to("cuda")
    # model.eval()
    # adv = AutoAttack(model, norm="Linf", eps=8 / 255, version="standard")
    # adv.attacks_to_run = ["apgd-ce"]
    # testset = load_cifar10_test(num_samples=1000)
    # for img, label in testset:
    #     img, label = img.to("cuda"), label.to("cuda")
    #     adv.run_standard_evaluation(img, label, bs=128)
    main()
