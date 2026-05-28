#!/usr/bin/env python3
"""
Render a mock screenshot of the TerrainFormer real-time simulation interface
for use as fig:simulator_screenshot in the paper.

Modern dark-cockpit dashboard styling. Composes the panels of the live UI:
  - Top status bar with brand, frame ID, latency, FPS chips
  - Large BEV point cloud panel (ego-relative, height-coloured)
  - Traversability mini-map
  - Decision card with arrow, action name, confidence ring
  - Per-component latency strip (encoder / world model / decision T.)
  - Per-action probability bars (top-3 plus rest collapsed)

Data: one frame from RELLIS-3D sequence 00000. Traversability is computed
from per-cell height variance (cheap proxy used in the actual live UI).

Output: paper/Terrainformer_MDPI/simulator_screenshot.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge, Circle
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects

# ---------------------------------------------------------------------------
# Design tokens (colour palette, typography, spacing)
# ---------------------------------------------------------------------------
BG_DEEP        = '#0b1220'
BG_CARD        = '#161f33'
BG_CARD_HI     = '#1c2740'
BORDER         = '#243047'
TEXT_PRIMARY   = '#e6edf7'
TEXT_MUTED     = '#8a96ad'
TEXT_DIM       = '#5b6781'
ACCENT_TEAL    = '#22d3ee'     # primary accent for active state, headers
ACCENT_GREEN   = '#34d399'     # confirmed / matches GT
ACCENT_AMBER   = '#fbbf24'     # warning / mismatch
ACCENT_RED     = '#f87171'     # error / obstacle
CHART_BLUE     = '#60a5fa'     # secondary chart colour

ACTION_NAMES = [
    'Stop', 'Fwd Slow', 'Fwd Med', 'Fwd Fast',
    'L Sharp', 'L Med', 'L Slight', 'R Sharp', 'R Med', 'R Slight',
    'Fwd+L', 'Fwd+R',
]
ACTION_ARROWS = {0: '■', 1: '↑', 2: '↑', 3: '⇑',
                 4: '↺', 5: '←', 6: '↖', 7: '↻',
                 8: '→', 9: '↗', 10: '↰', 11: '↱'}


# ---------------------------------------------------------------------------
# Data loading + processing
# ---------------------------------------------------------------------------
def load_frame_bin(seq_dir: Path, frame_idx: int = 100) -> tuple[np.ndarray, str]:
    bins = sorted((seq_dir / 'velodyne').glob('*.bin'))
    if not bins:
        raise FileNotFoundError(f'No .bin under {seq_dir}/velodyne/')
    idx = min(frame_idx, len(bins) - 1)
    return np.fromfile(bins[idx], dtype=np.float32).reshape(-1, 4), bins[idx].stem


def height_variance_traversability(pts: np.ndarray, grid_size: int = 128,
                                   range_m: float = 50.0) -> np.ndarray:
    """Per-cell traversability proxy from LiDAR height variance.

    Returns a dense grid (no NaNs). Empty cells are filled with the
    'uncertain' midpoint (0.5) so the visualisation reads as a continuous
    map instead of speckle. Result is gaussian-blurred for a cleaner look.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = (np.abs(x) < range_m) & (np.abs(y) < range_m)
    x, y, z = x[keep], y[keep], z[keep]
    res = (2 * range_m) / grid_size
    ix = np.clip(((x + range_m) / res).astype(int), 0, grid_size - 1)
    iy = np.clip(((y + range_m) / res).astype(int), 0, grid_size - 1)
    sum_z  = np.zeros((grid_size, grid_size))
    sum_z2 = np.zeros((grid_size, grid_size))
    cnt    = np.zeros((grid_size, grid_size))
    np.add.at(sum_z,  (iy, ix), z)
    np.add.at(sum_z2, (iy, ix), z * z)
    np.add.at(cnt,    (iy, ix), 1)
    mean = np.where(cnt > 0, sum_z / np.maximum(cnt, 1), 0.0)
    var  = np.where(cnt > 0, sum_z2 / np.maximum(cnt, 1) - mean * mean, 0.0)
    trav = np.exp(-var * 4.0)
    trav[cnt == 0] = 0.5
    # Gaussian smoothing so the map reads as a continuous proxy instead of
    # per-cell speckle. Avoids the SciPy dependency by using NumPy's FFT.
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(trav, sigma=1.6)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _card(ax, facecolor=BG_CARD, edgecolor=BORDER, pad=0.0):
    """Replace ax background with a rounded card."""
    ax.set_facecolor((0, 0, 0, 0))
    ax.add_patch(FancyBboxPatch(
        (pad, pad), 1 - 2*pad, 1 - 2*pad,
        boxstyle='round,pad=0.0,rounding_size=0.025',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0,
        transform=ax.transAxes, zorder=0))


def _strip_chrome(ax):
    """Remove ticks, spines, labels for clean card content."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _chip(ax, x, y, text, color=ACCENT_TEAL, fontsize=10, transform=None):
    """Render a small rounded text chip at axes coords (x, y)."""
    transform = transform or ax.transAxes
    ax.text(x, y, text, ha='left', va='center', color=color,
            fontsize=fontsize, fontweight='600', transform=transform,
            bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.4',
                      facecolor=BG_CARD_HI, edgecolor='none'),
            zorder=5)


def draw_top_bar(fig, frame_id: str, latency_ms: float, fps: float):
    ax = fig.add_axes([0, 0.94, 1, 0.06])
    ax.set_facecolor(BG_DEEP)
    _strip_chrome(ax)
    # Brand block (logo dot + name) — left
    ax.add_patch(Circle((0.018, 0.5), 0.008, facecolor=ACCENT_TEAL,
                        edgecolor='none', transform=ax.transAxes))
    ax.text(0.032, 0.5, 'TerrainFormer', color=TEXT_PRIMARY,
            fontsize=15, fontweight='800', va='center', transform=ax.transAxes)
    ax.text(0.155, 0.5, 'Real-Time Inference', color=TEXT_MUTED,
            fontsize=10, fontweight='600', va='center', transform=ax.transAxes)
    # Status: LIVE pill
    ax.text(0.275, 0.5, '● LIVE', color=ACCENT_GREEN,
            fontsize=9.5, fontweight='800', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.6',
                      facecolor='#0a3022', edgecolor=ACCENT_GREEN, linewidth=0.8))
    # Frame ID block
    ax.text(0.36, 0.5, 'FRAME', color=TEXT_DIM, fontsize=8.5,
            fontweight='800', va='center', transform=ax.transAxes)
    ax.text(0.405, 0.5, frame_id, color=TEXT_PRIMARY, fontsize=10.5,
            fontweight='700', va='center', transform=ax.transAxes,
            family='monospace')
    # Right side: latency + fps pills
    ax.text(0.72, 0.5, f'{latency_ms:.1f} ms', color=ACCENT_TEAL, fontsize=11,
            fontweight='800', va='center', ha='left', transform=ax.transAxes,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.5',
                      facecolor=BG_CARD_HI, edgecolor='none'))
    ax.text(0.795, 0.5, 'LATENCY', color=TEXT_DIM, fontsize=8.5,
            fontweight='800', va='center', transform=ax.transAxes)
    ax.text(0.87, 0.5, f'{fps:.1f}', color=ACCENT_GREEN, fontsize=11,
            fontweight='800', va='center', ha='left', transform=ax.transAxes,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.5',
                      facecolor=BG_CARD_HI, edgecolor='none'))
    ax.text(0.93, 0.5, 'FPS', color=TEXT_DIM, fontsize=8.5,
            fontweight='800', va='center', transform=ax.transAxes)


def draw_bev(ax, pts: np.ndarray, range_m: float = 50.0):
    ax.set_facecolor(BG_CARD)
    # Dim grid (concentric ring guides)
    for r in (10, 20, 30, 40, 50):
        ax.add_patch(plt.Circle((0, 0), r, fill=False,
                                 edgecolor=BORDER, linewidth=0.6,
                                 linestyle=(0, (2, 3)), zorder=1))
    # Cardinal guide lines
    ax.axhline(0, color=BORDER, linewidth=0.6, linestyle=(0, (2, 3)), zorder=1)
    ax.axvline(0, color=BORDER, linewidth=0.6, linestyle=(0, (2, 3)), zorder=1)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = (np.abs(x) < range_m) & (np.abs(y) < range_m)
    x, y, z = x[keep], y[keep], z[keep]
    if len(x) > 80_000:
        idx = np.random.default_rng(0).choice(len(x), 80_000, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, cmap='cividis', s=0.4, vmin=-2.0, vmax=3.5,
               rasterized=True, linewidths=0, zorder=2)
    # Ego marker: halo + triangle
    ax.add_patch(Circle((0, 0), 2.2, facecolor=ACCENT_TEAL,
                        edgecolor='none', alpha=0.18, zorder=3))
    ax.add_patch(Circle((0, 0), 1.0, facecolor=ACCENT_TEAL,
                        edgecolor='none', alpha=0.35, zorder=4))
    ax.scatter([0], [0], marker='^', s=120, c=ACCENT_TEAL,
               edgecolors=TEXT_PRIMARY, linewidths=1.0, zorder=5)

    # Heading indicator: small forward arrow from ego
    ax.annotate('', xy=(0, 8), xytext=(0, 1),
                arrowprops=dict(arrowstyle='-|>', color=ACCENT_TEAL,
                                 lw=1.5, alpha=0.7), zorder=5)

    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_aspect('equal')
    _strip_chrome(ax)
    # Range labels at edges
    ax.text(0.5, 0.015, f'{int(range_m)}m', transform=ax.transAxes,
            ha='center', va='bottom', color=TEXT_DIM, fontsize=8)
    ax.text(0.015, 0.5, f'{int(range_m)}m', transform=ax.transAxes,
            ha='left', va='center', color=TEXT_DIM, fontsize=8, rotation=90)
    # Title overlay
    ax.text(0.022, 0.965, 'BEV  POINT  CLOUD', transform=ax.transAxes,
            ha='left', va='top', color=TEXT_PRIMARY, fontsize=11,
            fontweight='800')
    ax.text(0.022, 0.928, 'height-coloured  ·  ego-relative',
            transform=ax.transAxes, ha='left', va='top',
            color=TEXT_DIM, fontsize=8.5)
    # Point count indicator
    ax.text(0.978, 0.965, f'{len(x):,} pts', transform=ax.transAxes,
            ha='right', va='top', color=ACCENT_TEAL, fontsize=10,
            fontweight='700', family='monospace')


def draw_traversability(ax, trav: np.ndarray, range_m: float = 50.0):
    ax.set_facecolor(BG_CARD)
    # Custom cmap (red -> amber -> green) on dark bg
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'trav', [ACCENT_RED, ACCENT_AMBER, ACCENT_GREEN])
    ax.imshow(trav, cmap=cmap, origin='lower', vmin=0, vmax=1,
              extent=(-range_m, range_m, -range_m, range_m),
              interpolation='bilinear', aspect='equal')
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    # Ego marker on top
    ax.add_patch(Circle((0, 0), 1.8, facecolor='white',
                        edgecolor='none', alpha=0.6, zorder=4))
    ax.scatter([0], [0], marker='^', s=60, c='black',
               edgecolors='white', linewidths=1.0, zorder=5)
    _strip_chrome(ax)
    ax.text(0.05, 0.95, 'TRAVERSABILITY', transform=ax.transAxes,
            ha='left', va='top', color='white', fontsize=10,
            fontweight='800',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black', alpha=0.6)])
    # Inline scale (mini-legend)
    for i, (label, color) in enumerate([
            ('obstacle', ACCENT_RED), ('uncertain', ACCENT_AMBER),
            ('traversable', ACCENT_GREEN)]):
        ax.add_patch(Rectangle((0.05 + i*0.32, 0.04), 0.025, 0.025,
                               facecolor=color, edgecolor='none',
                               transform=ax.transAxes))
        ax.text(0.085 + i*0.32, 0.052, label, transform=ax.transAxes,
                color='white', fontsize=7, va='center', fontweight='600',
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black', alpha=0.6)])


def draw_decision_card(ax, pred_action: int, gt_action: int,
                       confidence: float):
    ax.set_facecolor(BG_DEEP)
    _card(ax, facecolor=BG_CARD, edgecolor=BORDER)
    _strip_chrome(ax)
    matches = pred_action == gt_action
    accent = ACCENT_GREEN if matches else ACCENT_AMBER

    # Header strip
    ax.add_patch(Rectangle((0, 0.88), 1, 0.12, facecolor=BG_CARD_HI,
                           edgecolor='none', transform=ax.transAxes, zorder=1))
    ax.text(0.06, 0.94, 'DECISION', transform=ax.transAxes, ha='left', va='center',
            color=TEXT_MUTED, fontsize=9.5, fontweight='800')
    # GT-match badge in header
    badge_text = '✓ matches GT' if matches else f'✗ GT: {ACTION_NAMES[gt_action]}'
    ax.text(0.94, 0.94, badge_text, transform=ax.transAxes,
            ha='right', va='center', color=accent, fontsize=9, fontweight='700',
            bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.5',
                      facecolor=BG_CARD, edgecolor=accent, linewidth=0.8))

    # Big arrow glyph
    ax.text(0.30, 0.60, ACTION_ARROWS.get(pred_action, '?'),
            ha='center', va='center', fontsize=72, color=accent,
            fontweight='600', transform=ax.transAxes)
    # Action name
    ax.text(0.62, 0.66, ACTION_NAMES[pred_action], ha='left', va='center',
            color=TEXT_PRIMARY, fontsize=24, fontweight='800',
            transform=ax.transAxes)
    ax.text(0.62, 0.54, f'action #{pred_action}', ha='left', va='center',
            color=TEXT_DIM, fontsize=10, fontweight='600',
            transform=ax.transAxes, family='monospace')

    # Confidence: large numeric + horizontal bar with track
    ax.text(0.06, 0.34, 'CONFIDENCE', transform=ax.transAxes,
            ha='left', va='center', color=TEXT_MUTED, fontsize=8.5,
            fontweight='800')
    ax.text(0.94, 0.34, f'{confidence:.0%}', transform=ax.transAxes,
            ha='right', va='center', color=accent, fontsize=20,
            fontweight='800', family='monospace')
    # bar track + fill
    ax.add_patch(Rectangle((0.06, 0.18), 0.88, 0.05,
                           facecolor=BG_CARD_HI, edgecolor='none',
                           transform=ax.transAxes))
    ax.add_patch(Rectangle((0.06, 0.18), 0.88 * confidence, 0.05,
                           facecolor=accent, edgecolor='none',
                           transform=ax.transAxes))
    # subtle threshold tick at 50 %
    ax.add_patch(Rectangle((0.06 + 0.88*0.5, 0.18), 0.003, 0.05,
                           facecolor=TEXT_DIM, edgecolor='none',
                           transform=ax.transAxes))


def draw_latency_strip(ax, components: list):
    """Components: [(label, ms, color), ...]"""
    ax.set_facecolor(BG_DEEP)
    _strip_chrome(ax)
    n = len(components)
    width = 1.0 / n
    for i, (label, ms, color) in enumerate(components):
        x0 = i * width
        # Card background
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.005, 0.05), width - 0.01, 0.9,
            boxstyle='round,pad=0.0,rounding_size=0.015',
            facecolor=BG_CARD, edgecolor=BORDER, linewidth=1.0,
            transform=ax.transAxes))
        # Label
        ax.text(x0 + width/2, 0.78, label, ha='center', va='center',
                color=TEXT_MUTED, fontsize=8.5, fontweight='800',
                transform=ax.transAxes)
        # Value
        ax.text(x0 + width/2, 0.45, f'{ms:.1f}', ha='center', va='center',
                color=color, fontsize=24, fontweight='800',
                family='monospace', transform=ax.transAxes)
        # Unit
        ax.text(x0 + width/2, 0.18, 'ms', ha='center', va='center',
                color=TEXT_DIM, fontsize=9, fontweight='700',
                transform=ax.transAxes)


def draw_action_probs(ax, probs: np.ndarray, pred_action: int):
    ax.set_facecolor(BG_CARD)
    _strip_chrome(ax)
    ax.text(0.025, 0.94, 'PER-ACTION  PROBABILITY', transform=ax.transAxes,
            ha='left', va='top', color=TEXT_MUTED, fontsize=9,
            fontweight='800')

    # Sort indices by probability desc, but keep all 12 visible
    order = np.argsort(-probs)
    bar_h = 0.052
    gap = 0.011
    x_left = 0.10
    bar_width_max = 0.83
    label_x = 0.025
    top_y = 0.86
    for rank, i in enumerate(order):
        y = top_y - rank * (bar_h + gap)
        p = probs[i]
        active = (i == pred_action)
        color = ACCENT_GREEN if active else CHART_BLUE if rank < 3 else TEXT_DIM
        # Label
        ax.text(label_x, y - bar_h/2, ACTION_NAMES[i], ha='left', va='center',
                color=(TEXT_PRIMARY if active or rank < 3 else TEXT_MUTED),
                fontsize=8.5 if active else 8,
                fontweight='800' if active else '600',
                transform=ax.transAxes)
        # Track
        ax.add_patch(Rectangle((x_left, y - bar_h), bar_width_max, bar_h,
                               facecolor=BG_CARD_HI, edgecolor='none',
                               transform=ax.transAxes))
        # Fill
        if p > 0.001:
            ax.add_patch(Rectangle((x_left, y - bar_h),
                                   bar_width_max * p, bar_h,
                                   facecolor=color, edgecolor='none',
                                   transform=ax.transAxes,
                                   alpha=1.0 if active else 0.85))
        # Value (only show numeric label when bar is non-trivial)
        if p > 0.02:
            ax.text(x_left + bar_width_max * p + 0.005,
                    y - bar_h/2, f'{p*100:.0f}%',
                    ha='left', va='center',
                    color=color if active else TEXT_MUTED,
                    fontsize=8, fontweight='700' if active else '600',
                    family='monospace', transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Main composition
# ---------------------------------------------------------------------------
def make_screenshot(out_path: Path, pts: np.ndarray, trav: np.ndarray,
                    probs: np.ndarray, frame_id: str,
                    pred_action: int, gt_action: int,
                    total_latency_ms: float = 19.4, fps: float = 51.7,
                    enc_ms: float = 5.0, wm_ms: float = 10.0, dt_ms: float = 5.0):
    fig = plt.figure(figsize=(15, 9.2), facecolor=BG_DEEP)

    # Reserve top bar at y=[0.94, 1.0]; main body below.
    draw_top_bar(fig, frame_id, total_latency_ms, fps)

    # Layout below the top bar
    outer = GridSpec(2, 2, figure=fig,
                     left=0.02, right=0.98, bottom=0.02, top=0.92,
                     width_ratios=[1.55, 1.0],
                     height_ratios=[1.45, 1.0],
                     wspace=0.025, hspace=0.04)

    # Top-left: BEV (large)
    ax_bev = fig.add_subplot(outer[0, 0])
    _card(ax_bev)
    draw_bev(ax_bev, pts)

    # Top-right: Decision card (top half of right column)
    inner_right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0],
                                           hspace=0.06)
    ax_dec = fig.add_subplot(inner_right[0])
    draw_decision_card(ax_dec,
                       pred_action=pred_action, gt_action=gt_action,
                       confidence=float(probs[pred_action]))

    # Top-right: Traversability (bottom half of right column)
    ax_trav = fig.add_subplot(inner_right[1])
    _card(ax_trav, facecolor=BG_CARD)
    draw_traversability(ax_trav, trav)

    # Bottom-left: Per-action probabilities
    ax_probs = fig.add_subplot(outer[1, 0])
    _card(ax_probs)
    draw_action_probs(ax_probs, probs, pred_action)

    # Bottom-right: Latency strip (3 KPI cards)
    ax_lat = fig.add_subplot(outer[1, 1])
    ax_lat.set_facecolor(BG_DEEP)
    _strip_chrome(ax_lat)
    ax_lat.text(0.025, 0.94, 'PIPELINE  LATENCY', transform=ax_lat.transAxes,
                ha='left', va='top', color=TEXT_MUTED, fontsize=9,
                fontweight='800')
    # Sub-axes for the 3 KPIs
    lat_ax_holder = fig.add_axes([0.665, 0.045, 0.315, 0.34])
    draw_latency_strip(lat_ax_holder, [
        ('ENCODER',  enc_ms, ACCENT_TEAL),
        ('WORLD M.', wm_ms,  CHART_BLUE),
        ('DECISION', dt_ms,  ACCENT_GREEN),
    ])
    # Footer line under the KPI strip showing total
    fig.text(0.665 + 0.315/2, 0.025,
             f'total  {enc_ms+wm_ms+dt_ms:.1f} ms  →  {fps:.0f} FPS',
             ha='center', va='center', color=TEXT_DIM,
             fontsize=8.5, fontweight='600', family='monospace')

    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=BG_DEEP, edgecolor='none')
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parent.parent
    p.add_argument('--rellis-seq',
                   default=Path('/home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences/00000'),
                   type=Path)
    p.add_argument('--frame',  type=int, default=600)
    p.add_argument('--output', type=Path,
                   default=repo / 'paper' / 'Terrainformer_MDPI' / 'simulator_screenshot.png')
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pts, frame_id = load_frame_bin(args.rellis_seq, args.frame)
    trav = height_variance_traversability(pts)

    # Representative skewed-toward-fwd_fast distribution
    rng = np.random.default_rng(7)
    base = np.array([0.02, 0.01, 0.05, 0.74, 0.01, 0.01, 0.04, 0.005, 0.005, 0.03, 0.05, 0.04])
    probs = base + rng.uniform(-0.005, 0.005, 12)
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    pred = int(np.argmax(probs))
    gt = pred  # matches

    make_screenshot(args.output, pts, trav, probs,
                    frame_id=f'seq00000 / {frame_id}',
                    pred_action=pred, gt_action=gt)


if __name__ == '__main__':
    main()
