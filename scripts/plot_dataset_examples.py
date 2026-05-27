#!/usr/bin/env python3
"""
Render one representative BEV LiDAR scan from each of the three datasets used
in the paper, and save them as PNGs ready for inclusion as fig:dataset_examples
in the manuscript.

Outputs (default):
    paper/Terrainformer_MDPI/dataset_examples_rellis3d.png
    paper/Terrainformer_MDPI/dataset_examples_lidardustx.png
    paper/Terrainformer_MDPI/dataset_examples_goose3d.png
    paper/Terrainformer_MDPI/dataset_examples.png   (combined 1x3 figure)

Usage:
    python scripts/plot_dataset_examples.py
    python scripts/plot_dataset_examples.py --rellis-dir <path> --output-dir <path>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Subsample BEV plots above this point count (visual identical, render much faster).
PLOT_SUBSAMPLE_LIMIT = 250_000
PLOT_SUBSAMPLE_SEED = 0

# ----------------------------------------------------------------------------
# Default dataset locations on this machine (override with CLI flags as needed)
# ----------------------------------------------------------------------------
DEFAULTS = {
    'rellis3d':   Path('/home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences'),
    'lidardustx': Path('/home/rickslab3/Documents/Datasets/LidarDustX'),
    'goose3d':    Path('/home/rickslab3/Documents/Datasets/goose_3d_train/lidar/train'),
}
DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'paper' / 'Terrainformer_MDPI'


# ----------------------------------------------------------------------------
# Per-dataset loaders. All return an (N, 4) numpy array with columns
# (x, y, z, intensity) in metres.
# ----------------------------------------------------------------------------
def _load_kitti_bin(path: Path) -> np.ndarray:
    """RELLIS-3D, LidarDustX, GOOSE-3D all use KITTI-format .bin: float32, (x, y, z, i)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def pick_first_bin(root: Path) -> Path:
    """Walk a directory tree and return the first .bin file found."""
    if not root.exists():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    for p in sorted(root.rglob('*.bin')):
        return p
    raise FileNotFoundError(f'No .bin files under {root}')


def load_rellis3d(root: Path, seq: str = '00000', frame: int = 100) -> tuple[np.ndarray, str]:
    """Pick frame `frame` from the chosen sequence (representative off-road scene)."""
    velo_dir = root / seq / 'velodyne'
    bins = sorted(velo_dir.glob('*.bin'))
    if not bins:
        raise FileNotFoundError(f'No .bin files in {velo_dir}')
    idx = min(frame, len(bins) - 1)
    return _load_kitti_bin(bins[idx]), f'RELLIS-3D (seq {seq} / frame {bins[idx].stem})'


def load_lidardustx(root: Path) -> tuple[np.ndarray, str]:
    """Pick the first .bin under any sensor sub-directory (representative dusty scan)."""
    p = pick_first_bin(root)
    sensor = p.parent.name
    return _load_kitti_bin(p), f'LidarDustX ({sensor} sensor)'


def load_goose3d(root: Path) -> tuple[np.ndarray, str]:
    """Pick the first vls128 .bin (representative mixed-outdoor scan)."""
    p = pick_first_bin(root)
    return _load_kitti_bin(p), 'GOOSE-3D (Velodyne VLS128)'


# ----------------------------------------------------------------------------
# BEV rendering
# ----------------------------------------------------------------------------
def _render_missing(ax, label: str) -> None:
    """Draw a 'not found' placeholder when a dataset failed to load."""
    ax.text(0.5, 0.5, label, ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def render_bev(ax, pts: np.ndarray, title: str, range_m: float = 50.0,
               point_size: float = 0.4, height_clip: tuple = (-2.5, 4.0)) -> None:
    """Draw a top-down BEV scatter plot coloured by height."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = (np.abs(x) < range_m) & (np.abs(y) < range_m)
    x, y, z = x[keep], y[keep], z[keep]

    if len(x) > PLOT_SUBSAMPLE_LIMIT:
        rng = np.random.default_rng(PLOT_SUBSAMPLE_SEED)
        idx = rng.choice(len(x), PLOT_SUBSAMPLE_LIMIT, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    sc = ax.scatter(x, y, c=z, cmap='viridis', s=point_size,
                    vmin=height_clip[0], vmax=height_clip[1],
                    rasterized=True, linewidths=0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2, linestyle=':')
    # Ego marker
    ax.scatter([0], [0], marker='^', s=70, c='red', edgecolors='black',
               linewidths=0.8, zorder=5, label='Ego')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    plt.colorbar(sc, ax=ax, label='height (m)', shrink=0.85)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rellis-dir',   type=Path, default=DEFAULTS['rellis3d'])
    p.add_argument('--lidardust-dir',type=Path, default=DEFAULTS['lidardustx'])
    p.add_argument('--goose-dir',    type=Path, default=DEFAULTS['goose3d'])
    p.add_argument('--output-dir',   type=Path, default=DEFAULT_OUT)
    p.add_argument('--rellis-seq',   type=str,  default='00000')
    p.add_argument('--rellis-frame', type=int,  default=100)
    p.add_argument('--range',        type=float,default=50.0, help='BEV half-extent in metres')
    p.add_argument('--dpi',          type=int,  default=200)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        ('rellis3d',   lambda: load_rellis3d(args.rellis_dir, args.rellis_seq, args.rellis_frame)),
        ('lidardustx', lambda: load_lidardustx(args.lidardust_dir)),
        ('goose3d',    lambda: load_goose3d(args.goose_dir)),
    ]

    samples = []
    for key, loader in loaders:
        try:
            pts, title = loader()
            print(f'[{key}] loaded {len(pts):>7d} points -- {title}')
            samples.append((key, pts, title))
        except Exception as e:
            print(f'[{key}] FAILED: {e}', file=sys.stderr)
            samples.append((key, None, f'{key}: NOT FOUND'))

    # ------- Individual figures (one PNG per dataset) -------
    for key, pts, title in samples:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        if pts is not None:
            render_bev(ax, pts, title, range_m=args.range)
        else:
            _render_missing(ax, title)
        out = args.output_dir / f'dataset_examples_{key}.png'
        fig.tight_layout()
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out}')

    # ------- Combined 1x3 figure (paper-ready) -------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (key, pts, title) in zip(axes, samples):
        if pts is not None:
            render_bev(ax, pts, title, range_m=args.range)
        else:
            _render_missing(ax, title)
    fig.suptitle('Representative LiDAR scans from each dataset (BEV, height-coloured)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.output_dir / 'dataset_examples.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
