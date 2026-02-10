#!/bin/bash
#
# TerrainFormer - Run Entire Project
# Training (Phase 1 & 2) + Evaluation
#

set -e

echo "=========================================================================================================="
echo "TERRAINFORMER - FULL PROJECT RUN"
echo "=========================================================================================================="
echo ""
echo "Pipeline:"
echo "  [1] World Model Training (100 epochs)"
echo "  [2] Decision Transformer Training (50 epochs)"
echo "  [3] Evaluation on Test Set"
echo "  [4] Predictive Evaluation"
echo ""
echo "=========================================================================================================="
echo ""

# Kill any existing training
echo "[0] Stopping any existing processes..."
pkill -9 -f "python scripts/train.py" 2>/dev/null || true
sleep 2
echo "Done"
echo ""

# Activate environment
source venv/bin/activate

# Clean old outputs
echo "[1] Cleaning old outputs..."
rm -rf outputs/world_model_pretrain
rm -rf outputs/decision_train
mkdir -p outputs/world_model_pretrain
mkdir -p outputs/decision_train
echo "Done"
echo ""

# Phase 1: World Model Training
echo "=========================================================================================================="
echo "PHASE 1: WORLD MODEL TRAINING"
echo "=========================================================================================================="
echo "Started at: $(date)"
echo ""

python scripts/train.py \
    --config configs/training/pretrain_world_model.yaml \
    --device cuda

echo ""
echo "Phase 1 Complete: $(date)"
echo ""

sleep 3

# Phase 2: Decision Transformer Training
echo "=========================================================================================================="
echo "PHASE 2: DECISION TRANSFORMER TRAINING"
echo "=========================================================================================================="
echo "Started at: $(date)"
echo ""

python scripts/train.py \
    --config configs/training/train_decision.yaml \
    --device cuda

echo ""
echo "Phase 2 Complete: $(date)"
echo ""

sleep 3

# Phase 3: Evaluation
echo "=========================================================================================================="
echo "PHASE 3: EVALUATION"
echo "=========================================================================================================="
echo ""

python scripts/evaluate_decision.py

sleep 2

# Phase 4: Predictive Evaluation
echo "=========================================================================================================="
echo "PHASE 4: PREDICTIVE EVALUATION"
echo "=========================================================================================================="
echo ""

python scripts/evaluate_predictive.py

# Final Summary
echo ""
echo "=========================================================================================================="
echo "PROJECT COMPLETE"
echo "=========================================================================================================="
echo ""
echo "Training Results:"
python << 'EOF'
import torch
wm = torch.load('outputs/world_model_pretrain/best_model.pt', map_location='cpu', weights_only=False)
dt = torch.load('outputs/decision_train/best_model.pt', map_location='cpu', weights_only=False)
print(f"  World Model:  Epoch {wm.get('epoch')}, Loss {wm.get('best_val_loss'):.4f}")
print(f"  Decision:     Epoch {dt.get('epoch')}, Accuracy {dt.get('best_accuracy'):.4f}")
EOF

echo ""
echo "Output Files:"
echo "  outputs/world_model_pretrain/best_model.pt"
echo "  outputs/decision_train/best_model.pt"
echo "  outputs/decision_train/evaluation_results.txt"
echo "  outputs/decision_train/predictive_results.txt"
echo "  outputs/decision_train/confusion_matrix.png"
echo "  outputs/decision_train/predictive_comparison.png"
echo ""
echo "Finished at: $(date)"
echo "=========================================================================================================="
