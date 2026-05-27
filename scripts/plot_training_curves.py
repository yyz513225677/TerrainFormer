#!/usr/bin/env python3
"""
Plot training curves for the paper's fig:loss_curves.

Reads per-epoch metrics from:
    outputs/world_model_pretrain/metrics.csv   (Phase 1)
    outputs/decision_train/metrics.csv         (Phase 2)

These are produced automatically by WorldModelTrainer and DecisionTrainer
(see src/training/trainers/*_trainer.py); each row contains at minimum
{epoch, train/loss, val/loss [, val/accuracy]}.

If the current run's metrics.csv is missing, we fall back to reconstructing
Phase 1 from per-epoch checkpoints in
    outputs/world_model_pretrain_OLD_*/checkpoint_epoch_*.pt
which store {epoch, best_val_loss}.

Outputs:
    paper/Terrainformer_MDPI/loss_curves.png   (two-panel figure)

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --output-dir <path> --phase1-dir <path> --phase2-dir <path>
"""
import argparse
import csv
import struct
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Pickle opcodes used by ``_extract_meta`` to skip memoize markers and read
# scalar values without unpickling. See CPython's pickle module for the spec.
PICKLE_OP_SHORT_BINPUT = b'q'   # followed by 1-byte memo id
PICKLE_OP_LONG_BINPUT  = b'r'   # followed by 4-byte memo id
PICKLE_OP_BININT1      = b'K'   # 1-byte unsigned int follows
PICKLE_OP_BININT2      = b'M'   # 2-byte little-endian unsigned int follows
PICKLE_OP_BININT       = b'J'   # 4-byte little-endian signed int follows
PICKLE_OP_BINFLOAT     = b'G'   # 8-byte big-endian IEEE-754 double follows


# ----------------------------------------------------------------------------
# CSV loader (preferred path -- works once the patched trainer has run once)
# ----------------------------------------------------------------------------
def load_metrics_csv(csv_path: Path) -> dict:
    """Return dict {col_name: numpy array}. Empty dict if file missing."""
    if not csv_path.exists():
        return {}
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        vals = []
        for r in rows:
            v = r.get(k, '')
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


# ----------------------------------------------------------------------------
# Fallback: scan per-epoch checkpoints in an OLD run dir
# ----------------------------------------------------------------------------
def _extract_meta(pt_path: Path) -> dict:
    """Read {epoch, val_loss, best_val_loss, best_accuracy} from a .pt file
    without loading torch (parses the pickle byte stream directly)."""
    with zipfile.ZipFile(pt_path) as z:
        name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
        data = z.read(name)
    out = {}
    for key in [b'epoch', b'val_loss', b'best_val_loss', b'best_accuracy']:
        i = data.find(key)
        if i < 0:
            continue
        j = i + len(key)
        # skip any memoize marker (1- or 4-byte memo id) that follows the key
        if data[j:j+1] == PICKLE_OP_SHORT_BINPUT:
            j += 2
        elif data[j:j+1] == PICKLE_OP_LONG_BINPUT:
            j += 5
        op = data[j:j+1]
        if op == PICKLE_OP_BININT1:
            out[key.decode()] = data[j+1]
        elif op == PICKLE_OP_BININT2:
            out[key.decode()] = struct.unpack('<H', data[j+1:j+3])[0]
        elif op == PICKLE_OP_BININT:
            out[key.decode()] = struct.unpack('<i', data[j+1:j+5])[0]
        elif op == PICKLE_OP_BINFLOAT:
            out[key.decode()] = struct.unpack('>d', data[j+1:j+9])[0]
    return out


def load_from_checkpoints(run_dir: Path) -> dict:
    """Walk a run dir, parse every checkpoint_epoch_*.pt for {epoch, val_loss},
    and return a dict-of-arrays compatible with load_metrics_csv()."""
    files = sorted(run_dir.glob('checkpoint_epoch_*.pt'))
    if not files:
        return {}
    epochs, val_losses = [], []
    for f in files:
        m = _extract_meta(f)
        if 'epoch' not in m or 'val_loss' not in m:
            continue
        epochs.append(m['epoch'])
        val_losses.append(m['val_loss'])
    if not epochs:
        return {}
    order = np.argsort(epochs)
    return {
        'epoch':    np.array(epochs)[order],
        'val/loss': np.array(val_losses)[order],
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _plot_phase1(ax, m: dict, source_label: str) -> None:
    epoch = m['epoch']
    if 'train/loss' in m:
        ax.plot(epoch, m['train/loss'], '-',  color='#1f77b4', label='train')
    if 'val/loss' in m:
        ax.plot(epoch, m['val/loss'],   '--', color='#1f77b4', label='validation')
    best_idx = int(np.nanargmin(m['val/loss']))
    ax.scatter([epoch[best_idx]], [m['val/loss'][best_idx]],
               marker='*', s=140, color='red', zorder=5,
               label=f'best (ep {int(epoch[best_idx])}, {m["val/loss"][best_idx]:.4f})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'Phase 1 — World Model ({source_label})')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')


def _plot_phase2(ax, m: dict, source_label: str) -> None:
    epoch = m['epoch']
    # left axis: focal loss
    color_loss = '#2ca02c'
    if 'train/loss' in m:
        ax.plot(epoch, m['train/loss'], '-',  color=color_loss, label='train loss')
    if 'val/loss' in m:
        ax.plot(epoch, m['val/loss'],   '--', color=color_loss, label='val loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('focal loss', color=color_loss)
    ax.tick_params(axis='y', labelcolor=color_loss)
    ax.grid(True, alpha=0.3, linestyle=':')

    # right axis: val accuracy
    if 'val/accuracy' in m:
        ax2 = ax.twinx()
        color_acc = '#d62728'
        ax2.plot(epoch, m['val/accuracy'], '-o', color=color_acc, markersize=3,
                 label='val accuracy')
        best = int(np.nanargmax(m['val/accuracy']))
        ax2.scatter([epoch[best]], [m['val/accuracy'][best]],
                    marker='*', s=140, color='black', zorder=5,
                    label=f'best (ep {int(epoch[best])}, {m["val/accuracy"][best]*100:.2f}%)')
        ax2.set_ylabel('validation accuracy', color=color_acc)
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.set_ylim(0, 1)

        # combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='lower right', framealpha=0.9, fontsize=9)
    else:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    ax.set_title(f'Phase 2 — Decision Transformer ({source_label})')


def main():
    p = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parent.parent
    p.add_argument('--phase1-dir', type=Path, default=repo / 'outputs' / 'world_model_pretrain')
    p.add_argument('--phase2-dir', type=Path, default=repo / 'outputs' / 'decision_train')
    p.add_argument('--phase1-fallback-dir', type=Path,
                   default=repo / 'outputs' / 'world_model_pretrain_OLD_20260122_222225',
                   help='Per-epoch checkpoint directory to use if Phase 1 CSV is missing')
    p.add_argument('--output-dir', type=Path,
                   default=repo / 'paper' / 'Terrainformer_MDPI')
    p.add_argument('--dpi',        type=int, default=200)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    m1 = load_metrics_csv(args.phase1_dir / 'metrics.csv')
    p1_src = 'this run'
    if not m1:
        m1 = load_from_checkpoints(args.phase1_fallback_dir)
        p1_src = f'reconstructed from {args.phase1_fallback_dir.name}'
    if not m1:
        print('!! No Phase 1 data available (no metrics.csv, no checkpoint fallback).')

    # Phase 2
    m2 = load_metrics_csv(args.phase2_dir / 'metrics.csv')
    p2_src = 'this run'
    if not m2:
        print('!! No Phase 2 data available (metrics.csv missing). '
              'Re-run training with the updated trainer to populate it.')

    # Plot
    n_panels = int(bool(m1)) + int(bool(m2))
    if n_panels == 0:
        raise SystemExit('No data to plot. Run training (with the updated trainers) first.')

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    axes = axes[0]
    i = 0
    if m1:
        _plot_phase1(axes[i], m1, p1_src); i += 1
    if m2:
        _plot_phase2(axes[i], m2, p2_src); i += 1

    fig.suptitle('Training curves for both training phases', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = args.output_dir / 'loss_curves.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
