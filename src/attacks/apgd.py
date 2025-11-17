"""
Auto-PGD (APGD) Attack with adaptive step size
"""

import math

import torch
import torch.nn.functional as F


def dlr_loss(logits, y):
    """
    Difference of Logits Ratio (DLR) loss

    Args:
        logits: Model output logits
        y: True labels

    Returns:
        loss: DLR loss value
    """
    y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()

    # Correct class logit
    correct_logit = (logits * y_onehot).sum(1)

    # Best wrong class logit
    wrong_logit = (logits * (1 - y_onehot) - 1e10 * y_onehot).max(1)[0]

    # Top logit and third-highest logit for normalization
    top_logit = logits.max(1)[0]
    sorted_logits = torch.sort(logits, dim=1, descending=True)[0]
    third_logit = sorted_logits[:, 2]

    # DLR loss
    loss = -(correct_logit - wrong_logit) / (top_logit - third_logit + 1e-12)

    return loss.mean()


class APGDAttack:
    def __init__(self, eps=8 / 255, n_iter=100, loss_fn="ce", n_restarts=1):
        """
        Auto-PGD Attack

        Args:
            eps: Maximum perturbation (L-infinity norm)
            n_iter: Number of iterations
            loss_fn: Loss function ('ce' for cross-entropy, 'dlr' for DLR)
            n_restarts: Number of random restarts
        """
        self.eps = eps
        self.n_iter = n_iter
        self.loss_fn = loss_fn
        self.n_restarts = n_restarts
        self.thr_decr = 0.75  # Threshold for step size decrease

    def attack(self, model, x, y):
        """
        Perform APGD attack

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
            loss_best = self._compute_loss(logits, y).clone()

        for restart in range(self.n_restarts):
            # Random initialization
            x_adv = x.clone().detach()
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Initial step size
            alpha = 2 * self.eps / self.n_iter

            # Checkpoint for step size adaptation
            checkpoint_interval = max(self.n_iter // 5, 1)
            loss_steps = torch.zeros(self.n_iter, batch_size, device=device)

            for i in range(self.n_iter):
                x_adv.requires_grad = True

                with torch.enable_grad():
                    logits = model(x_adv)
                    loss = self._compute_loss(logits, y)

                grad = torch.autograd.grad(loss, x_adv)[0]

                with torch.no_grad():
                    # Store loss for adaptation
                    loss_steps[i] = self._compute_loss(logits, y, reduction="none")

                    # Update adversarial example
                    x_adv = x_adv.detach() + alpha * grad.sign()

                    # Project back to epsilon ball
                    delta = torch.clamp(x_adv - x, -self.eps, self.eps)
                    x_adv = torch.clamp(x + delta, 0, 1)

                    # Adaptive step size reduction
                    if (i + 1) % checkpoint_interval == 0:
                        # Check if loss improved
                        start_idx = max(0, i + 1 - checkpoint_interval)
                        fl = loss_steps[start_idx : i + 1].min(0)[0]

                        if start_idx > 0:
                            fl_prev = loss_steps[:start_idx].min(0)[0]
                            # Reduce step size if no sufficient improvement
                            mask = fl >= self.thr_decr * fl_prev
                            alpha = alpha * (1 - 0.5 * mask.float())

            # Update best adversarial examples
            with torch.no_grad():
                logits = model(x_adv)
                loss = self._compute_loss(logits, y, reduction="none")
                mask = loss > loss_best
                x_best[mask] = x_adv[mask].clone()
                loss_best[mask] = loss[mask]

        return x_best.detach()

    def _compute_loss(self, logits, y, reduction="mean"):
        """Compute loss based on loss_fn"""
        if self.loss_fn == "ce":
            if reduction == "none":
                loss = F.cross_entropy(logits, y, reduction="none")
            else:
                loss = F.cross_entropy(logits, y)
        elif self.loss_fn == "dlr":
            if reduction == "none":
                # DLR loss per sample
                y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
                correct_logit = (logits * y_onehot).sum(1)
                wrong_logit = (logits * (1 - y_onehot) - 1e10 * y_onehot).max(1)[0]
                top_logit = logits.max(1)[0]
                sorted_logits = torch.sort(logits, dim=1, descending=True)[0]
                third_logit = sorted_logits[:, 2]
                loss = -(correct_logit - wrong_logit) / (
                    top_logit - third_logit + 1e-12
                )
            else:
                loss = dlr_loss(logits, y)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        return loss


if __name__ == "__main__":
    # Test APGD
    from models.resnet import ResNet18

    model = ResNet18()
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    print("Testing APGD-CE...")
    apgd_ce = APGDAttack(eps=8 / 255, n_iter=100, loss_fn="ce")
    x_adv_ce = apgd_ce.attack(model, x, y)
    print(f"Max perturbation: {(x_adv_ce - x).abs().max().item():.6f}")

    print("\nTesting APGD-DLR...")
    apgd_dlr = APGDAttack(eps=8 / 255, n_iter=100, loss_fn="dlr")
    x_adv_dlr = apgd_dlr.attack(model, x, y)
    print(f"Max perturbation: {(x_adv_dlr - x).abs().max().item():.6f}")
