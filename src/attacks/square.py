"""
Square Attack - Black-box adversarial attack
"""

import math

import torch
import torch.nn.functional as F


class SquareAttack:
    def __init__(self, eps=8 / 255, n_queries=5000, p_init=0.8):
        """
        Square Attack

        Args:
            eps: Maximum perturbation (L-infinity norm)
            n_queries: Number of queries (iterations)
            p_init: Initial proportion of pixels to perturb
        """
        self.eps = eps
        self.n_queries = n_queries
        self.p_init = p_init

    def attack(self, model, x, y):
        """
        Perform Square Attack

        Args:
            model: Target model
            x: Clean images
            y: True labels

        Returns:
            x_adv: Adversarial examples
        """
        device = x.device
        batch_size, c, h, w = x.shape

        # Initialize with random perturbation
        x_adv = x.clone().detach()
        delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x + delta, 0, 1)

        # Get initial predictions and margins
        with torch.no_grad():
            logits = model(x_adv)
            margin_min = self._margin_loss(logits, y, reduction="none")

        n_features = c * h * w

        for i in range(self.n_queries):
            # Decrease square size over time
            p = self._get_square_prob(i)
            s = max(int(round(math.sqrt(p * n_features / c))), 1)
            s = min(s, h - 1, w - 1)

            # Random position for square
            center_h = torch.randint(0, h - s, (batch_size,))
            center_w = torch.randint(0, w - s, (batch_size,))

            # Create perturbation
            x_new = x_adv.clone()
            for j in range(batch_size):
                # Random perturbation value
                new_delta = torch.empty(c, s, s, device=device).uniform_(
                    -self.eps, self.eps
                )

                # Apply square perturbation
                x_new[
                    j, :, center_h[j] : center_h[j] + s, center_w[j] : center_w[j] + s
                ] = (
                    x[
                        j,
                        :,
                        center_h[j] : center_h[j] + s,
                        center_w[j] : center_w[j] + s,
                    ]
                    + new_delta
                )

                x_new[j] = torch.clamp(x_new[j], 0, 1)

            # Check if new perturbation is better
            with torch.no_grad():
                logits_new = model(x_new)
                margin_new = self._margin_loss(logits_new, y, reduction="none")

                # Update if margin decreased (closer to misclassification)
                idx_improved = margin_new < margin_min
                x_adv[idx_improved] = x_new[idx_improved].clone()
                margin_min[idx_improved] = margin_new[idx_improved]

        return x_adv.detach()

    def _get_square_prob(self, iteration):
        """Get probability of square size based on iteration"""
        return self.p_init * (1 - iteration / self.n_queries)

    def _margin_loss(self, logits, y, reduction="mean"):
        """Compute margin to decision boundary"""
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
        correct_logit = (logits * y_onehot).sum(1)
        wrong_logit = (logits * (1 - y_onehot)).max(1)[0]
        margin = correct_logit - wrong_logit

        if reduction == "mean":
            return margin.mean()
        else:
            return margin


if __name__ == "__main__":
    from models.resnet import ResNet18

    model = ResNet18()
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    print("Testing Square Attack...")
    square = SquareAttack(eps=8 / 255, n_queries=1000)
    x_adv = square.attack(model, x, y)
    print(f"Max perturbation: {(x_adv - x).abs().max().item():.6f}")
