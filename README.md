# Adversarial Robustness Research

A PyTorch implementation for studying adversarial robustness on CIFAR-10, with a focus on comparing APGD-CE (Auto-PGD with Cross-Entropy loss) against standard PGD attacks.

## Overview

This project provides a complete pipeline for:
- Training standard and adversarially robust neural networks on CIFAR-10
- Evaluating model robustness against various adversarial attacks (FGSM, PGD, APGD-CE)
- Analyzing attack convergence and per-class robustness
- Visualizing adversarial examples

## Features

- **Training Scripts**
  - Standard training (clean images only)
  - Adversarial training using PGD-AT (Projected Gradient Descent Adversarial Training)

- **Attack Implementations**
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent) with configurable step sizes
  - APGD-CE (Auto-PGD with Cross-Entropy loss) - adaptive step size attack
  - Integration with AutoAttack library

- **Evaluation Tools**
  - Comprehensive attack evaluation with multiple baselines
  - Per-class robustness analysis
  - Convergence tracking
  - Adversarial example visualization

## Requirements

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Python 3.10+ is required.

### Dependencies

- PyTorch >= 2.8.0
- torchvision >= 0.23.0
- autoattack (from GitHub)
- numpy >= 2.2.6
- matplotlib >= 3.10.6
- tqdm >= 4.67.1

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd robustness

# Install dependencies using uv
uv sync
```

## Project Structure

```
robustness/
├── src/
│   ├── attacks/           # Attack implementations
│   │   ├── apgd_ce.py    # Auto-PGD with Cross-Entropy loss
│   │   ├── pgd.py        # PGD and FGSM attacks
│   │   ├── apgd.py       # Additional APGD variants
│   │   ├── autoattack.py # AutoAttack integration
│   │   ├── fab.py        # FAB attack
│   │   └── square.py     # Square attack
│   ├── models/
│   │   └── resnet.py     # ResNet-18 architecture
│   ├── utils/
│   │   └── logger.py     # Training logger
│   ├── train_standard.py     # Standard training script
│   ├── train_adversarial.py  # Adversarial training script
│   ├── eval_attacks.py       # Attack evaluation script
│   ├── track_convergence.py  # Convergence analysis
│   ├── analyze_results.py    # Results analysis
│   └── visualize_adversarial.py  # Visualization tools
├── attack/                # Legacy attack implementations
├── run_exp.sh            # Complete experimental pipeline
├── pyproject.toml        # Project configuration
└── README.md
```

## Usage

### Quick Start

Run the complete experimental pipeline:

```bash
bash run_exp.sh
```

This script will:
1. Train a standard model
2. Train an adversarially robust model (PGD-AT)
3. Evaluate both models against multiple attacks
4. Generate analysis and visualizations

### Individual Scripts

#### 1. Train a Standard Model

```bash
uv run src/train_standard.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --save-dir ./checkpoints/standard
```

#### 2. Train an Adversarially Robust Model

```bash
uv run src/train_adversarial.py \
    --epochs 100 \
    --batch-size 128 \
    --eps 0.03137 \      # 8/255
    --alpha 0.00784 \    # 2/255
    --attack-steps 10 \
    --save-dir ./checkpoints/adversarial
```

#### 3. Evaluate Model Robustness

```bash
uv run src/eval_attacks.py \
    --checkpoint ./checkpoints/adversarial/best_robust_model.pth \
    --batch-size 128 \
    --eps 0.03137 \
    --save-dir ./results/adversarial \
    --per-class          # Optional: compute per-class metrics
```

#### 4. Visualize Adversarial Examples

```bash
uv run src/visualize_adversarial.py \
    --checkpoint ./checkpoints/adversarial/best_robust_model.pth \
    --n-samples 4 \
    --save-dir ./figures
```

### Command-Line Arguments

#### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch-size` | Training batch size | 128 |
| `--epochs` | Number of training epochs | 100 |
| `--lr` | Initial learning rate | 0.1 |
| `--momentum` | SGD momentum | 0.9 |
| `--weight-decay` | Weight decay coefficient | 5e-4 |
| `--seed` | Random seed | 42 |
| `--save-dir` | Directory to save checkpoints | varies |

#### Adversarial Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--eps` | Perturbation budget (L∞) | 8/255 |
| `--alpha` | PGD step size | 2/255 |
| `--attack-steps` | PGD iterations during training | 10 |

#### Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to model checkpoint | required |
| `--eps` | Perturbation budget | 8/255 |
| `--verbose` | Verbose attack output | False |
| `--per-class` | Compute per-class metrics | False |

## Attack Implementations

### APGD-CE (Auto-PGD with Cross-Entropy)

An adaptive attack that automatically adjusts step sizes based on loss oscillation:
- Initial step size: α = 2 × ε
- Adaptive step size reduction when progress stalls
- Momentum-based updates for better convergence
- Checkpoint-based monitoring every ~22% of iterations

Key parameters:
- `eps`: Maximum perturbation (default: 8/255)
- `n_iter`: Number of iterations (default: 100)
- `rho`: Threshold for step reduction (default: 0.75)

### PGD (Projected Gradient Descent)

Standard iterative attack with fixed step size:
- Configurable number of iterations (50 or 100 typical)
- Fixed step size α (typically 1/255 or 2/255)
- Optional random initialization

### FGSM (Fast Gradient Sign Method)

Single-step attack for baseline comparison:
- Perturbation: δ = ε × sign(∇_x L(θ, x, y))
- Fast but less effective than iterative methods

## Experimental Results

The evaluation script compares:
- Clean accuracy (no attack)
- FGSM robustness
- PGD-50 robustness (α=2/255)
- PGD-100 robustness (α=2/255 and α=1/255)
- APGD-CE robustness (100 iterations)

Results are saved as JSON files with:
- Overall accuracy metrics
- Performance gaps between attacks
- Optional per-class breakdown

## Output Structure

```
checkpoints/
├── standard/
│   └── best_model.pth
└── adversarial/
    ├── best_robust_model.pth
    └── latest_model.pth

results/
├── standard/
│   └── eval_results.json
└── adversarial/
    ├── eval_results.json
    └── per_class_results.json

figures/
└── adversarial_examples.png

data/
└── cifar-10-batches-py/  # Downloaded automatically
```

## Model Architecture

ResNet-18 adapted for CIFAR-10:
- Input: 32×32×3 images
- Output: 10 classes
- Standard residual blocks with batch normalization
- Trained with SGD + Cosine Annealing LR schedule

## Training Details

### Standard Training
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Learning rate: 0.1 with cosine annealing
- Data augmentation: Random crop (padding=4), Random horizontal flip
- 100 epochs

### Adversarial Training (PGD-AT)
- Same optimizer and schedule as standard training
- Adversarial examples generated with PGD-10 (ε=8/255, α=2/255)
- Evaluation every 10 epochs with PGD-50
- Saves best model based on robust accuracy

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@inproceedings{croce2020reliable,
  title={Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
  author={Croce, Francesco and Hein, Matthias},
  booktitle={ICML},
  year={2020}
}

@inproceedings{madry2018towards,
  title={Towards deep learning models resistant to adversarial attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  booktitle={ICLR},
  year={2018}
}
```

## Acknowledgments

- AutoAttack implementation: https://github.com/fra31/auto-attack
- ResNet architecture based on PyTorch official implementation
- CIFAR-10 dataset from Krizhevsky et al.
