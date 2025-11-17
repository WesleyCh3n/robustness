"""
Fast Adaptive Boundary (FAB) Attack - Simplified Implementation
"""

import torch
import torch.nn.functional as F


class FABAttack:
    def __init__(self, eps=8 / 255, n_iter=100, n_restarts=1):
        """
        Simplified FAB Attack

        Args:
            eps: Maximum perturbation (L-infinity norm)
            n_iter: Number of iterations
            n_restarts: Number of random restarts
        """
        self.eps = eps
        self.n_iter = n_iter
        self.n_restarts = n_restarts

    def attack(self, model, x, y):
        """
        Perform FAB attack (simplified version using projection-based approach)

        Args:
            model: Target model
            x: Clean images
            y: True labels

        Returns:
            x_adv: Adversarial examples
        """
        device = x.device
        batch_size = x.shape[0]

        # Track best adversarial examples
        x_best = x.clone()
        with torch.no_grad():
            logits = model(x)
            pred_correct = logits.argmax(1) == y

        for restart in range(self.n_restarts):
            # Random initialization
            x_adv = x.clone().detach()
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Initial step size
            alpha = self.eps / 5

            for i in range(self.n_iter):
                x_adv.requires_grad = True

                with torch.enable_grad():
                    logits = model(x_adv)

                    # Compute margin loss (distance to decision boundary)
                    y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
                    correct_logit = (logits * y_onehot).sum(1)
                    wrong_logit = (logits * (1 - y_onehot)).max(1)[0]
                    loss = (correct_logit - wrong_logit).sum()

                grad = torch.autograd.grad(loss, x_adv)[0]

                with torch.no_grad():
                    # Normalize gradient to follow boundary
                    grad_norm = (
                        grad.view(batch_size, -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                    )
                    grad_normalized = grad / (grad_norm + 1e-12)

                    # Update adversarial example
                    x_adv = x_adv.detach() - alpha * grad_normalized

                    # Project back to epsilon ball
                    delta = torch.clamp(x_adv - x, -self.eps, self.eps)
                    x_adv = torch.clamp(x + delta, 0, 1)

                    # Adaptive step size
                    if (i + 1) % (self.n_iter // 5) == 0:
                        alpha *= 0.8

            # Update best adversarial examples
            with torch.no_grad():
                logits = model(x_adv)
                pred_adv = logits.argmax(1)
                mask = (pred_adv != y) & pred_correct
                x_best[mask] = x_adv[mask].clone()

        return x_best.detach()


if __name__ == "__main__":
    from models.resnet import ResNet18

    model = ResNet18()
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    print("Testing FAB...")
    fab = FABAttack(eps=8 / 255, n_iter=100)
    x_adv = fab.attack(model, x, y)
    print(f"Max perturbation: {(x_adv - x).abs().max().item():.6f}")
