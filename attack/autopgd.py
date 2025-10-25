import torch
import torch.nn as nn


def autopgd_step_size_schedule(iteration, total_iters, epsilon):
    if iteration < 0.25 * total_iters:
        return 2.0 * epsilon
    elif iteration < 0.5 * total_iters:
        return 1.5 * epsilon
    elif iteration < 0.75 * total_iters:
        return 1.0 * epsilon
    else:
        return 0.5 * epsilon


def autopgd(model, images, labels, epsilon=8 / 255, iters=100, n_restarts=1):
    max_loss = torch.zeros(images.shape[0]).to(images.device)
    max_delta = torch.zeros_like(images)

    for _ in range(n_restarts):
        delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        momentum = torch.zeros_like(images)

        for i in range(iters):
            adv_images = torch.clamp(images + delta, 0, 1)
            outputs = model(adv_images)
            loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels)

            grad = torch.autograd.grad(loss.sum(), delta)[0]
            momentum = 0.75 * momentum + grad / grad.view(grad.shape[0], -1).norm(
                1, dim=-1
            ).view(-1, 1, 1, 1)

            alpha = autopgd_step_size_schedule(i, iters, epsilon)
            delta.data = delta + alpha * momentum.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images

        final_loss = loss.detach()
        max_delta[final_loss > max_loss] = delta.detach()[final_loss > max_loss]
        max_loss = torch.max(max_loss, final_loss)

    return torch.clamp(images + max_delta, 0, 1)
