"""
Projected Gradient Descent (PGD) Attack
"""

import torch
import torch.nn.functional as F


class PGDAttack:
    def __init__(self, eps=8 / 255, alpha=2 / 255, steps=50, random_start=True):
        """
        PGD Attack

        Args:
            eps: Maximum perturbation (L-infinity norm)
            alpha: Step size
            steps: Number of attack iterations
            random_start: Whether to start from random point in epsilon ball
        """
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(self, model, x, y):
        """
        Perform PGD attack

        Args:
            model: Target model
            x: Clean images
            y: True labels

        Returns:
            x_adv: Adversarial examples
        """
        x_adv = x.clone().detach()

        # Random initialization
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(self.steps):
            x_adv.requires_grad = True

            with torch.enable_grad():
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            # Update adversarial example
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv.detach()


class FGSMAttack:
    def __init__(self, eps=8 / 255):
        """
        Fast Gradient Sign Method (FGSM)

        Args:
            eps: Maximum perturbation (L-infinity norm)
        """
        self.eps = eps

    def attack(self, model, x, y):
        """
        Perform FGSM attack

        Args:
            model: Target model
            x: Clean images
            y: True labels

        Returns:
            x_adv: Adversarial examples
        """
        x_adv = x.clone().detach()
        x_adv.requires_grad = True

        with torch.enable_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        # Single step update
        x_adv = x_adv.detach() + self.eps * grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.detach()
