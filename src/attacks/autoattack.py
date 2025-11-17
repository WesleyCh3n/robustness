"""
AutoAttack - Ensemble of parameter-free attacks
"""

import torch

from .apgd import APGDAttack
from .fab import FABAttack
from .square import SquareAttack


class AutoAttack:
    def __init__(self, eps=8 / 255, n_iter=100, n_restarts=1, verbose=True):
        """
        AutoAttack ensemble

        Args:
            eps: Maximum perturbation (L-infinity norm)
            n_iter: Number of iterations for white-box attacks
            n_restarts: Number of restarts for each attack
            verbose: Print progress
        """
        self.eps = eps
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.verbose = verbose

        # Initialize attack components
        self.apgd_ce = APGDAttack(
            eps=eps, n_iter=n_iter, loss_fn="ce", n_restarts=n_restarts
        )
        self.apgd_dlr = APGDAttack(
            eps=eps, n_iter=n_iter, loss_fn="dlr", n_restarts=n_restarts
        )
        self.fab = FABAttack(eps=eps, n_iter=n_iter, n_restarts=n_restarts)
        self.square = SquareAttack(eps=eps, n_queries=5000)

    def attack(self, model, x, y):
        """
        Perform AutoAttack ensemble

        Args:
            model: Target model
            x: Clean images
            y: True labels

        Returns:
            x_adv: Adversarial examples
            robust_flags: Boolean tensor indicating which examples remain correctly classified
        """
        device = x.device
        batch_size = x.shape[0]

        # Track adversarial examples and correctly classified indices
        x_adv = x.clone()

        with torch.no_grad():
            logits = model(x)
            correctly_classified = logits.argmax(1) == y

        if self.verbose:
            print(
                f"Initial correctly classified: {correctly_classified.sum().item()}/{batch_size}"
            )

        # Attack 1: APGD-CE
        if correctly_classified.any():
            if self.verbose:
                print("Running APGD-CE...")

            x_subset = x[correctly_classified]
            y_subset = y[correctly_classified]

            x_adv_subset = self.apgd_ce.attack(model, x_subset, y_subset)
            x_adv[correctly_classified] = x_adv_subset

            with torch.no_grad():
                logits = model(x_adv)
                correctly_classified = logits.argmax(1) == y

            if self.verbose:
                print(
                    f"After APGD-CE: {correctly_classified.sum().item()}/{batch_size}"
                )

        # Attack 2: APGD-DLR
        if correctly_classified.any():
            if self.verbose:
                print("Running APGD-DLR...")

            x_subset = x[correctly_classified]
            y_subset = y[correctly_classified]

            x_adv_subset = self.apgd_dlr.attack(model, x_subset, y_subset)
            x_adv[correctly_classified] = x_adv_subset

            with torch.no_grad():
                logits = model(x_adv)
                correctly_classified = logits.argmax(1) == y

            if self.verbose:
                print(
                    f"After APGD-DLR: {correctly_classified.sum().item()}/{batch_size}"
                )

        # Attack 3: FAB
        if correctly_classified.any():
            if self.verbose:
                print("Running FAB...")

            x_subset = x[correctly_classified]
            y_subset = y[correctly_classified]

            x_adv_subset = self.fab.attack(model, x_subset, y_subset)
            x_adv[correctly_classified] = x_adv_subset

            with torch.no_grad():
                logits = model(x_adv)
                correctly_classified = logits.argmax(1) == y

            if self.verbose:
                print(f"After FAB: {correctly_classified.sum().item()}/{batch_size}")

        # Attack 4: Square Attack
        if correctly_classified.any():
            if self.verbose:
                print("Running Square Attack...")

            x_subset = x[correctly_classified]
            y_subset = y[correctly_classified]

            x_adv_subset = self.square.attack(model, x_subset, y_subset)
            x_adv[correctly_classified] = x_adv_subset

            with torch.no_grad():
                logits = model(x_adv)
                correctly_classified = logits.argmax(1) == y

            if self.verbose:
                print(f"After Square: {correctly_classified.sum().item()}/{batch_size}")

        return x_adv, correctly_classified


if __name__ == "__main__":
    from models.resnet import ResNet18

    model = ResNet18()
    model.eval()

    x = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    print("Testing AutoAttack...")
    autoattack = AutoAttack(
        eps=8 / 255, n_iter=20, verbose=True
    )  # Reduced iterations for testing
    x_adv, robust_flags = autoattack.attack(model, x, y)
    print(f"\nMax perturbation: {(x_adv - x).abs().max().item():.6f}")
    print(f"Robust accuracy: {robust_flags.float().mean().item() * 100:.2f}%")
