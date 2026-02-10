# TerrainFormer

Autonomous off-road navigation using World Models and Decision Transformers.

## Algorithm Overview

```
LiDAR Point Cloud → BEV Projection → World Model → Decision Transformer → Action
```

### Components

1. **BEV Projection**: Converts 3D point cloud to 2D bird's-eye-view (256×256×64)
2. **World Model**: Predicts future terrain states and extracts scene features
3. **Decision Transformer**: Predicts driving actions from world features

### Training Phases

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | World Model Pretraining | Future prediction, traversability |
| 2 | Decision Training | Action prediction (frozen world model) |
| 3 | Joint Fine-tuning | End-to-end optimization (optional) |

## Project Structure

```
terrainformer/
├── configs/
│   ├── model/                    # Model architectures
│   ├── training/                 # Training configs (phase 1, 2, 3)
│   └── data/                     # Dataset config
├── src/
│   ├── models/
│   │   ├── world_model/          # Future prediction
│   │   ├── decision/             # Action prediction
│   │   └── unified/              # Combined model
│   ├── training/trainers/        # Training loops
│   ├── data/datasets/            # Data loading
│   └── utils/                    # BEV projection, helpers
├── scripts/
│   ├── train.py                  # Main training
│   ├── evaluate_decision.py      # Decision accuracy
│   ├── evaluate_predictive.py    # Predictive evaluation
│   └── generate_action_labels.py # Generate actions from poses
├── outputs/
│   ├── world_model_pretrain/     # Phase 1 checkpoint
│   └── decision_train/           # Phase 2 checkpoint + results
└── run.sh                        # Run entire project
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

RELLIS-3D dataset structure:
```
../RELLIS/
├── bin/00000-00004/       # Point clouds (.bin)
├── calib/*/poses.txt      # Ground truth poses
├── label/                 # Semantic labels
└── actions/               # Generated action labels
```

Generate action labels from poses:
```bash
python scripts/generate_action_labels.py
```

## Training

### Run Entire Project (Recommended)
```bash
./run.sh
```
This runs: Training (Phase 1 + 2) → Evaluation → Results

### Manual Training
```bash
source venv/bin/activate

# Phase 1: World Model (100 epochs, ~2 hours)
python scripts/train.py --config configs/training/pretrain_world_model.yaml --device cuda

# Phase 2: Decision Transformer (50 epochs, ~1 hour)
python scripts/train.py --config configs/training/train_decision.yaml --device cuda
```

## Evaluation

```bash
source venv/bin/activate

# Check training results
python -c "
import torch
wm = torch.load('outputs/world_model_pretrain/best_model.pt', map_location='cpu', weights_only=False)
dt = torch.load('outputs/decision_train/best_model.pt', map_location='cpu', weights_only=False)
print(f'World Model: Epoch {wm[\"epoch\"]}, Loss {wm[\"best_val_loss\"]:.4f}')
print(f'Decision: Epoch {dt[\"epoch\"]}, Accuracy {dt[\"best_accuracy\"]:.4f}')
"

# Full evaluation on test set
python scripts/evaluate_decision.py

# Predictive evaluation (using predicted future)
python scripts/evaluate_predictive.py
```

## Results

| Model | Metric | Value |
|-------|--------|-------|
| World Model | Best Loss | 0.3478 |
| Decision (Validation) | Accuracy | 92.96% |
| Decision (Test) | Accuracy | 79.78% |
| Predictive (Test) | Accuracy | 79.73% |

## Action Space

| ID | Action | ID | Action |
|----|--------|----|---------|
| 0 | straight | 11 | maintain speed |
| 1-2 | left/right 5° | 12 | slow down |
| 3-4 | left/right 10° | 13 | speed up |
| 5-6 | left/right 20° | 14 | stop |
| 7-8 | left/right 30° | 15 | reverse straight |
| 9-10 | left/right 45° | 16-17 | reverse left/right |

## Key Files

- Training: `scripts/train.py`
- Evaluation: `scripts/evaluate_decision.py`, `scripts/evaluate_predictive.py`
- World Model: `src/models/world_model/world_model.py`
- Decision: `src/models/decision/decision_transformer.py`
- Fast BEV: `src/utils/bev_projection_fast.py`
