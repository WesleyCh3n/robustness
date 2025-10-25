import torch


def evaluate_attack(model, dataloader, attack_fn, attack_name, device):
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(labels).sum().item()

        adv_images = attack_fn(model, images, labels)

        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(labels).sum().item()

        total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    attack_success = (
        100.0 * (clean_correct - adv_correct) / clean_correct
        if clean_correct > 0
        else 0.0
    )

    return {
        "attack": attack_name,
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "attack_success_rate": attack_success,
    }
