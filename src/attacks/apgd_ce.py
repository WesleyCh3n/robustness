"""
Auto-PGD with Cross-Entropy Loss (APGD-CE)
Focused implementation for reproduction study

Author: WesleyCh3n
Date: 2025-11-17
FIXED VERSION - Per-example step sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class APGDAttack:
    def __init__(
        self,
        eps=8 / 255,
        n_iter=100,
        n_restarts=1,
        norm="Linf",
        loss="ce",
        eot_iter=1,
        rho=0.75,
        verbose=False,
        device=None,
    ):
        """
        Auto-PGD Attack (following official implementation)

        Args:
            eps: Maximum perturbation (L-infinity norm)
            n_iter: Number of iterations
            n_restarts: Number of random restarts
            norm: Norm for the attack ('Linf', 'L2', 'L1')
            loss: Loss function ('ce' or 'dlr')
            eot_iter: Expectation over Transformation iterations
            rho: Threshold for step size decrease (0.75 in paper)
            verbose: Print debug information
            device: Device to run on
        """
        self.eps = eps
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.norm = norm
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho  # Threshold for decreasing step size
        self.verbose = verbose
        self.device = device

        # Set checkpoint parameters (from official implementation)
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)  # Checkpoint interval
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)  # Minimum checkpoint interval
        self.size_decr = max(int(0.03 * self.n_iter), 1)  # Size decrease for checkpoint

        assert self.norm in ["Linf", "L2", "L1"]
        assert self.loss in ["ce", "dlr"]

    def check_oscillation(self, loss_steps, step_idx, k, loss_best, k3=0.75):
        """
        Check for oscillation in loss values

        Args:
            loss_steps: History of loss values [n_iter, batch_size]
            step_idx: Current step index
            k: Window size to check
            loss_best: Best loss so far
            k3: Threshold (0.75 default)

        Returns:
            Boolean tensor indicating oscillation per example
        """
        t = torch.zeros(loss_steps.shape[1]).to(self.device)
        for counter in range(k):
            t += (
                loss_steps[step_idx - counter] > loss_steps[step_idx - counter - 1]
            ).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def dlr_loss(self, x, y):
        """
        Difference of Logits Ratio loss

        Args:
            x: Logits
            y: True labels

        Returns:
            DLR loss value
        """
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0], device=x.device)

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1.0 - ind)) / (
            x_sorted[:, -1] - x_sorted[:, -3] + 1e-12
        )

    def attack_single_run(self, model, x, y, x_init=None):
        """
        Single run of APGD attack (one restart)

        Args:
            model: Target model
            x: Clean images [batch_size, C, H, W]
            y: True labels [batch_size]
            x_init: Initial perturbation (optional)

        Returns:
            x_best: Best adversarial examples (highest loss)
            acc: Accuracy on adversarial examples
            loss_best: Best loss values
            x_best_adv: Best adversarial examples (first successful)
        """
        # Initialize adversarial example
        if x_init is None:
            if self.norm == "Linf":
                # Random initialization in [-eps, eps]
                t = 2 * torch.rand_like(x) - 1
                x_adv = x + self.eps * t
            elif self.norm == "L2":
                # Random direction, scaled to eps
                t = torch.randn_like(x)
                t_norm = t.view(x.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                x_adv = x + self.eps * t / (t_norm + 1e-12)
            else:
                raise NotImplementedError(f"Norm {self.norm} not implemented")
        else:
            x_adv = x_init.clone()

        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()

        # Track loss and accuracy over iterations
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        # Setup loss function
        if self.loss == "ce":
            criterion_indiv = nn.CrossEntropyLoss(reduction="none")
        elif self.loss == "dlr":
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

        # Initial gradient computation
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)

        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        # Initial accuracy
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        # Step size (alpha = 2 for Linf/L2, 1 for L1)
        alpha = 2.0 if self.norm in ["Linf", "L2"] else 1.0
        step_size = (
            alpha * self.eps * torch.ones([x.shape[0], 1, 1, 1], device=self.device)
        )

        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0  # Checkpoint interval
        counter3 = 0  # Counter for checkpoint

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        # Main iteration loop
        for i in range(self.n_iter):
            ### Gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old  # Momentum term
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0  # Momentum coefficient

                if self.norm == "Linf":
                    # Standard PGD step
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )
                    # Add momentum
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    # Normalize gradient to unit norm
                    grad_norm = (
                        grad.view(x.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                    )
                    grad_normalized = grad / (grad_norm + 1e-12)

                    # PGD step
                    x_adv_1 = x_adv + step_size * grad_normalized

                    # Project to L2 ball
                    delta = x_adv_1 - x
                    delta_norm = (
                        delta.view(x.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                    )
                    delta = (
                        delta
                        * torch.min(self.eps * torch.ones_like(delta_norm), delta_norm)
                        / (delta_norm + 1e-12)
                    )
                    x_adv_1 = torch.clamp(x + delta, 0.0, 1.0)

                    # Add momentum
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)

                    # Project again
                    delta = x_adv_1 - x
                    delta_norm = (
                        delta.view(x.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                    )
                    delta = (
                        delta
                        * torch.min(self.eps * torch.ones_like(delta_norm), delta_norm)
                        / (delta_norm + 1e-12)
                    )
                    x_adv_1 = torch.clamp(x + delta, 0.0, 1.0)

                x_adv = x_adv_1 + 0.0

            ### Compute gradient for next iteration
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            # Update accuracy
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0

            # Update best adversarial examples (first successful attack)
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.0

            if self.verbose and (i % 20 == 0 or i == self.n_iter - 1):
                print(
                    f"[APGD] iteration: {i} - best loss: {loss_best.sum():.6f} "
                    f"- robust accuracy: {acc.float().mean():.2%} "
                    f"- step size: {step_size.mean():.6f}"
                )

            ### Check step size and update
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0

                # Update best loss
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                # Check for step size reduction at checkpoints
                if counter3 == k:
                    if self.norm in ["Linf", "L2"]:
                        # Check oscillation
                        fl_oscillation = self.check_oscillation(
                            loss_steps, i, k, loss_best, k3=self.thr_decr
                        )

                        # Check if no improvement since last check
                        fl_reduce_no_impr = (1.0 - reduced_last_check) * (
                            loss_best_last_check >= loss_best
                        ).float()

                        fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                        reduced_last_check = fl_oscillation.clone()
                        loss_best_last_check = loss_best.clone()

                        if fl_oscillation.sum() > 0:
                            ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                            step_size[ind_fl_osc] /= 2.0

                            if self.verbose:
                                print(
                                    f"  [Step size reduced] for {fl_oscillation.sum():.0f}/{x.shape[0]} examples"
                                )

                            # Reset to best found so far for oscillating examples
                            x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                            grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                        # Decrease checkpoint interval
                        k = max(k - self.size_decr, self.n_iter_min)

                    counter3 = 0

        return x_best, acc, loss_best, x_best_adv

    def attack(self, model, x, y):
        """
        Perform APGD attack with multiple restarts

        Args:
            model: Target model
            x: Clean images [batch_size, C, H, W]
            y: True labels [batch_size]

        Returns:
            adv: Adversarial examples [batch_size, C, H, W]
        """
        if self.device is None:
            self.device = x.device

        x = x.detach().clone().float().to(self.device)
        y = y.detach().clone().long().to(self.device)

        # Get initial predictions
        model.eval()
        with torch.no_grad():
            y_pred = model(x).max(1)[1]

        adv = x.clone()
        acc = y_pred == y

        if self.verbose:
            print("=" * 70)
            print(
                f"Running APGD-{self.loss.upper()} attack with epsilon {self.eps:.5f}"
            )
            print(f"Initial accuracy: {acc.float().mean():.2%}")
            print("=" * 70)

        # Run multiple restarts
        torch.manual_seed(0)  # For reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

        for counter in range(self.n_restarts):
            # Only attack correctly classified examples
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)

            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                # Run single attack
                best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(
                    model, x_to_fool, y_to_fool
                )

                # Update adversarial examples
                ind_curr = (acc_curr == 0).nonzero().squeeze()
                if ind_curr.numel() > 0:
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                if self.verbose:
                    print(
                        f"Restart {counter + 1}/{self.n_restarts} - "
                        f"Robust accuracy: {acc.float().mean():.2%}"
                    )

        return adv


if __name__ == "__main__":
    # Test APGD-CE implementation
    print("Testing APGD-CE implementation (Official Version)")
    print(f"Date: 2025-11-17 15:56:28 UTC")
    print(f"User: WesleyCh3n\n")

    import sys

    sys.path.append("..")
    from models.resnet import ResNet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = ResNet18()
    model = model.to(device)
    model.eval()

    # Create dummy batch
    x = torch.randn(8, 3, 32, 32).to(device)
    y = torch.randint(0, 10, (8,)).to(device)

    # Get clean predictions
    with torch.no_grad():
        clean_pred = model(x).argmax(1)
        clean_correct = (clean_pred == y).float().mean().item()

    print(f"Clean accuracy: {clean_correct * 100:.1f}%\n")

    # Test APGD-CE
    print("=" * 70)
    print("Test 1: APGD-CE (Linf, 1 restart)")
    print("=" * 70)
    apgd = APGDAttack(
        eps=8 / 255,
        n_iter=50,
        n_restarts=1,
        norm="Linf",
        loss="ce",
        verbose=True,
        device=device,
    )
    x_adv = apgd.attack(model, x, y)

    # Verify
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(1)
        adv_correct = (adv_pred == y).float().mean().item()

    max_pert = (x_adv - x).abs().max().item()
    print(f"\nResults:")
    print(f"  Adversarial accuracy: {adv_correct * 100:.1f}%")
    print(f"  Attack success rate: {(1 - adv_correct / clean_correct) * 100:.1f}%")
    print(f"  Max perturbation: {max_pert:.6f} (budget: {8 / 255:.6f})")
    print(f"  Within budget: {max_pert <= 8 / 255 + 1e-6}")

    # Test with 5 restarts
    print("\n" + "=" * 70)
    print("Test 2: APGD-CE (Linf, 5 restarts)")
    print("=" * 70)
    apgd5 = APGDAttack(
        eps=8 / 255,
        n_iter=50,
        n_restarts=5,
        norm="Linf",
        loss="ce",
        verbose=True,
        device=device,
    )
    x_adv5 = apgd5.attack(model, x, y)

    with torch.no_grad():
        adv_pred5 = model(x_adv5).argmax(1)
        adv_correct5 = (adv_pred5 == y).float().mean().item()

    print(f"\nResults:")
    print(f"  Adversarial accuracy: {adv_correct5 * 100:.1f}%")
    print(f"  Attack success rate: {(1 - adv_correct5 / clean_correct) * 100:.1f}%")
    print(f"  Improvement over 1 restart: {(adv_correct - adv_correct5) * 100:.1f} pp")
