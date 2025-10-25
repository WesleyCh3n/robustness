import torch
import torch.nn as nn


def pgd(model, images, labels, epsilon=8 / 255, alpha=2 / 255, iters=10):
    """
    PGD (Projected Gradient Descent) Attack

    Args:
        model: Target neural network
        images: Input images (batch)
        labels: True labels
        epsilon: Maximum perturbation (L-infinity norm)
        alpha: Step size for each iteration
        iters: Number of attack iterations

    Returns:
        Adversarial examples
    """
    images = images.clone().detach()
    x = torch.tensor(images, requires_grad=True)

    for _ in range(iters):
        # Forward pass
        x.requires_grad = True
        outputs = model(x)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        assert x.grad is not None, (
            "Gradient is None. Check if the model is in eval mode."
        )
        # Update perturbation with gradient ascent
        delta = alpha * x.grad.detach().sign()
        x = x.detach() + delta.detach()
        # Project back to the epsilon ball and valid pixel range
        x = torch.min(torch.max(x, images - epsilon), images + epsilon)
        x = torch.clamp(x, 0, 1)
    return x
