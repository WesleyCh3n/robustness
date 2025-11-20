#!/bin/bash

# AutoAttack Reproduction - Complete Experimental Pipeline
# Author: WesleyCh3n
# Date: 2025-11-17

echo "========================================="
echo "AutoAttack Reproduction Study"
echo "========================================="

# Configuration
EPOCHS=100
BATCH_SIZE=128
EPS=0.03137  # 8/255
ALPHA=0.00784  # 2/255

# Create directories
mkdir -p checkpoints/standard
mkdir -p checkpoints/adversarial
mkdir -p results
mkdir -p figures

echo ""
echo "Step 1: Training Standard Model..."
echo "========================================="
uv run src/train_standard.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --save-dir ./checkpoints/standard

echo ""
echo "Step 2: Training Adversarially Robust Model..."
echo "========================================="
uv run src/train_adversarial.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --eps $EPS \
    --alpha $ALPHA \
    --attack-steps 10 \
    --save-dir ./checkpoints/adversarial

echo ""
echo "========================================================================"
echo "Step 3: Evaluating Standard Model"
echo "========================================================================"
echo "This will test FGSM, PGD-50, PGD-100 (various α), and APGD-CE"
uv run src/eval_attacks.py \
    --checkpoint ./checkpoints/standard/best_model.pth \
    --batch-size $BATCH_SIZE \
    --eps $EPS \
    --save-dir ./results/standard

echo ""
echo "========================================================================"
echo "Step 4: Evaluating Adversarially Trained Model"
echo "========================================================================"
echo "This is the main experiment - comparing APGD-CE against PGD baselines"
uv run src/eval_attacks.py \
    --checkpoint ./checkpoints/adversarial/best_robust_model.pth \
    --batch-size $BATCH_SIZE \
    --eps $EPS \
    --save-dir ./results/adversarial \
    --per-class

echo ""
echo "========================================================================"
echo "Step 5: Analyzing Results and Generating Tables"
echo "========================================================================"
uv run src/analyze_results.py \
    --standard-results ./results/standard/eval_results.json \
    --adversarial-results ./results/adversarial/eval_results.json \
    --per-class-results ./results/adversarial/per_class_results.json


echo ""
echo "Step 3: Evaluating Standard Model..."
echo "========================================="
uv run eval_attacks.py \
    --checkpoint ./checkpoints/standard/best_model.pth \
    --batch-size $BATCH_SIZE \
    --eps $EPS \
    --alpha $ALPHA \
    --aa-iter 100 \
    --save-dir ./results/standard

echo ""
echo "Step 4: Evaluating Adversarially Trained Model..."
echo "========================================="
uv run eval_attacks.py \
    --checkpoint ./checkpoints/adversarial/best_robust_model.pth \
    --batch-size $BATCH_SIZE \
    --eps $EPS \
    --alpha $ALPHA \
    --aa-iter 100 \
    --save-dir ./results/adversarial

echo ""
echo "Step 5: Analyzing Results..."
echo "========================================="
uv run analyze_results.py \
    --standard-results ./results/standard/eval_results.json \
    --adversarial-results ./results/adversarial/eval_results.json \
    --generate-plots

echo ""
echo "========================================================================"
echo "Step 5: Generating Adversarial Example Visualizations"
echo "========================================================================"
uv run src/visualize_adversarial.py \
    --checkpoint ./checkpoints/adversarial/best_robust_model.pth \
    --n-samples 4 \
    --save-dir ./figures \
    2>&1 | tee logs/visualization.log

echo ""
echo "========================================="
echo "Experiments Complete!"
echo "========================================="
echo "Results saved in ./results/"
echo "Figures saved in ./figures/"
echo "Checkpoints saved in ./checkpoints/"
