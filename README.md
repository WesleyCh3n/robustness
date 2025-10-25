# Robustness Evaluation

Implementation of PGD (Projected Gradient Descent) and AutoPGD adversarial attacks for evaluating neural network robustness on CIFAR-10.

## Features

- **PGD Attack**: Standard projected gradient descent attack with configurable epsilon, step size, and iterations
- **AutoPGD Attack**: Enhanced PGD with adaptive step size scheduling, momentum, and multiple restarts for stronger adversarial examples
- **Model Training**: Scripts to train both standard and adversarially robust ResNet-18 models
- **Evaluation Framework**: Compare attack success rates across different models

## Installation

```bash
uv sync
```

## Usage

### Train Models

Train both standard and adversarially robust ResNet-18 models on CIFAR-10:

```bash
python train_models.py
```

This will generate two model checkpoints:
- `resnet18_cifar10_standard.pth` - Standard trained model
- `resnet18_cifar10_robust.pth` - Adversarially trained model (PGD-based)

### Run Attacks

Evaluate both models against PGD and AutoPGD attacks:

```bash
python main.py
```

This will output:
- Clean accuracy
- Adversarial accuracy under attack
- Attack success rate

## Implementation Details

### PGD Attack (attack/pgd.py)

```python
def pgd(model, images, labels, epsilon=8/255, alpha=2/255, iters=10)
```

Standard PGD attack using gradient ascent with epsilon-ball projection.

### AutoPGD Attack (attack/autopgd.py)

```python
def autopgd(model, images, labels, epsilon=8/255, iters=100, n_restarts=1)
```

Enhanced attack with:
- Adaptive step size schedule (decreases over iterations)
- Momentum-based gradient updates
- Multiple random restarts
- Per-sample worst-case selection

## Project Structure

```
autopgd/
├── attack/
│   ├── autopgd.py      # AutoPGD implementation
│   └── pgd.py          # PGD implementation
├── evaluation.py       # Attack evaluation utilities
├── main.py             # Main evaluation script
└── train_models.py     # Model training script
```

## Requirements

- Python ≥3.10
- PyTorch ≥2.8.0
- torchvision ≥0.23.0
- matplotlib ≥3.10.6
- autoattack (for comparison)

## References

- [AutoAttack: improving adversarial robustness through a standardized adversarial evaluation](https://arxiv.org/abs/2003.01690)
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
