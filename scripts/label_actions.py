#!/usr/bin/env python3
"""
TerrainFormer Action Labeler

Interactive UI for manually labeling LiDAR frames with navigation actions.
Loads a folder of .bin point-cloud files and an optional poses.txt file,
then lets you assign one of the 12 training actions to each frame.

Saved files (in the .bin folder):
  actions.npy           — (N,) int64 array, -1 for unlabeled frames
  actions.csv           — human-readable: frame, x, y, z, action, action_name
  actions_metadata.json — statistics / config

Keyboard shortcuts:
  ← / →     navigate frames
  0 – 9     label actions 0-9
  a / d     label Fwd+Left (10) / Fwd+Right (11)
  Space     skip to next unlabeled frame
  s         save
  Delete    clear label for current frame

Usage:
  python scripts/label_actions.py
  python scripts/label_actions.py --folder /path/to/bins --poses /path/to/poses.txt
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button


# ── Action definitions ────────────────────────────────────────────────────────
# Must match generate_action_labels.py

ACTION_LABELS = {
    0:  "Stop",
    1:  "Fwd Slow",
    2:  "Fwd Med",
    3:  "Fwd Fast",
    4:  "Left Sharp",
    5:  "Left Med",
    6:  "Left Slight",
    7:  "Right Slight",
    8:  "Right Med",
    9:  "Right Sharp",
    10: "Fwd+Left",
    11: "Fwd+Right",
}

# Category color per action
ACTION_COLORS = {
    0:  '#e74c3c',   # Stop
    1:  '#2ecc71',   # Fwd Slow
    2:  '#27ae60',   # Fwd Med
    3:  '#1a8040',   # Fwd Fast
    4:  '#2980b9',   # Left Sharp
    5:  '#5dade2',   # Left Med
    6:  '#85c1e9',   # Left Slight
    7:  '#f39c12',   # Right Slight
    8:  '#e67e22',   # Right Med
    9:  '#d35400',   # Right Sharp
    10: '#1abc9c',   # Fwd+Left
    11: '#16a085',   # Fwd+Right
}

ACTION_ARROWS = {
    0:  '■',
    1:  '↑',
    2:  '↑',
    3:  '⇑',
    4:  '↺',
    5:  '←',
    6:  '↖',
    7:  '↗',
    8:  '→',
    9:  '↻',
    10: '↰',
    11: '↱',
}

# Keyboard → action_id
KEY_SHORTCUTS = {
    '0': 0, '1': 1, '2': 2, '3': 3,
    '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9,
    'a': 10,   # Fwd+Left
    'd': 11,   # Fwd+Right
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_bin(file_path: str) -> np.ndarray:
    """Load KITTI/RELLIS .bin point cloud → (N, 4) float32: x y z intensity."""
    pts = np.fromfile(file_path, dtype=np.float32)
    if len(pts) % 4 == 0:
        return pts.reshape(-1, 4)
    pts = pts.reshape(-1, 3)
    return np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float32)])


def load_poses(poses_file: str) -> np.ndarray:
    """Load poses.txt: each line is 12 floats → 3×4 [R|t] matrix.
    Returns (N, 3, 4) float64 array.
    """
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 12:
                poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def pose_xyzrpy(pose: np.ndarray):
    """Extract (x, y, z, roll°, pitch°, yaw°) from a 3×4 pose matrix."""
    x, y, z = pose[:, 3]
    yaw   = float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))
    pitch = float(np.degrees(np.arcsin(np.clip(-pose[2, 0], -1, 1))))
    roll  = float(np.degrees(np.arctan2(pose[2, 1], pose[2, 2])))
    return float(x), float(y), float(z), roll, pitch, yaw


# ── Main UI ───────────────────────────────────────────────────────────────────

class ActionLabeler:
    """
    Interactive action labeling tool.

    Panels (left → right):
      ax_bev   — large BEV scatter plot (dark background)
      ax_info  — frame XYZ / pose info table
      ax_traj  — mini overhead trajectory with labeled-frame colors
      ax_prog  — labeling progress bar + per-action distribution
      12 action buttons in a 4×3 grid
      Navigation buttons at the bottom
    """

    def __init__(self):
        self.bin_folder: Optional[str] = None
        self.bin_files: list = []
        self.poses: Optional[np.ndarray] = None   # (N, 3, 4)
        self.labels: dict = {}                      # frame_idx → action_id (-1 = cleared)
        self.current_idx: int = 0
        self.auto_advance: bool = True
        self.batch_size: int = 20        # frames labeled per button click
        self.preview_playing: bool = False
        self._preview_offset: int = 0   # 0 … batch_size-1

        self._build_figure()
        self._build_action_buttons()
        self._build_nav_buttons()

        # Timer fires every 150 ms → ~6.7 fps preview loop
        self._preview_timer = self.fig.canvas.new_timer(interval=150)
        self._preview_timer.add_callback(self._preview_tick)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._refresh()

    # ── Figure skeleton ───────────────────────────────────────────────────────

    def _build_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(22, 11))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.fig.suptitle(
            'TerrainFormer  |  Action Labeler',
            fontsize=17, fontweight='bold', color='#1a1a2e'
        )

        # ── BEV (left, large dark panel) ─────────────────────────────────
        # x0=0.02  y0=0.10  w=0.47  h=0.85
        self.ax_bev = self.fig.add_axes([0.02, 0.10, 0.47, 0.85])
        self.ax_bev.set_facecolor('white')
        self.ax_bev.set_xlim(-50, 50)
        self.ax_bev.set_ylim(-50, 50)
        self.ax_bev.set_aspect('equal')
        self.ax_bev.grid(True, alpha=0.4, color='#dddddd')
        self.ax_bev.tick_params(colors='#333333', labelsize=10)
        self.ax_bev.set_xlabel('X (m)', fontsize=11, color='#333333')
        self.ax_bev.set_ylabel('Y (m)', fontsize=11, color='#333333')
        for sp in self.ax_bev.spines.values():
            sp.set_edgecolor('#bbbbbb')

        # ── Info table (top-right)  x0=0.52  y0=0.68  w=0.22  h=0.27 ───
        self.ax_info = self.fig.add_axes([0.52, 0.68, 0.22, 0.27])
        self.ax_info.set_facecolor('white')
        self.ax_info.axis('off')

        # ── Mini trajectory (mid-right)  x0=0.52  y0=0.40  w=0.22  h=0.26 ─
        self.ax_traj = self.fig.add_axes([0.52, 0.40, 0.22, 0.26])
        self.ax_traj.set_facecolor('#fafafa')
        self.ax_traj.set_title('Trajectory', fontsize=12, fontweight='bold', pad=3)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.tick_params(labelsize=9)
        self.ax_traj.grid(True, alpha=0.3)

        # ── Progress + distribution (bottom-right of left block)
        #    x0=0.52  y0=0.10  w=0.22  h=0.28
        self.ax_prog = self.fig.add_axes([0.52, 0.10, 0.22, 0.28])
        self.ax_prog.set_facecolor('white')
        self.ax_prog.axis('off')

        # Action-button area x0=0.76 y0=0.10 w=0.22 h=0.85 — built separately

    # ── Action buttons (4 rows × 3 cols) ─────────────────────────────────────

    def _build_action_buttons(self):
        # Grid occupies x=[0.76, 0.98], y=[0.10, 0.95]
        ax0, ay0 = 0.76, 0.10
        aw,  ah  = 0.22, 0.85
        ncols, nrows = 3, 4
        gap = 0.007
        bw = (aw - gap * (ncols + 1)) / ncols
        bh = (ah - gap * (nrows + 1)) / nrows

        self._btn_objs: list = []   # (Button, axes) pairs
        self._btn_axes: list = []

        for i in range(12):
            row = i // ncols
            col = i % ncols
            bx = ax0 + gap + col * (bw + gap)
            by = ay0 + ah - gap - (row + 1) * (bh + gap)

            ax_b = self.fig.add_axes([bx, by, bw, bh])
            color = ACTION_COLORS[i]

            # Light fill for the button; text is the label
            label = f"{ACTION_ARROWS[i]}  {ACTION_LABELS[i]}"
            btn = Button(ax_b, label,
                         color=color + 'aa',       # default: semi-transparent
                         hovercolor=color)
            btn.label.set_fontsize(12)
            btn.label.set_fontweight('bold')
            # White text for dark colors, dark text for light ones
            dark_actions = {0, 3, 4, 9}
            btn.label.set_color('white' if i in dark_actions else '#111111')

            def _click(event, aid=i):
                self._on_action_clicked(aid)
            btn.on_clicked(_click)

            self._btn_objs.append(btn)
            self._btn_axes.append(ax_b)

    # ── Navigation bar ────────────────────────────────────────────────────────

    def _build_nav_buttons(self):
        btn_y, btn_h = 0.01, 0.065
        defs = [
            # (label,        x,    w,     color,         hover)
            ('Load Folder',  0.02, 0.11, 'lightblue',   '#4a90d9'),
            ('Load Poses',   0.14, 0.10, 'lightblue',   '#4a90d9'),
            ('← Prev',       0.26, 0.07, '#dddddd',     'silver'),
            ('Auto ▶',       0.34, 0.09, 'lightgreen',  'limegreen'),
            ('Skip →',       0.44, 0.07, '#fff3cd',     '#ffc107'),
            ('⌫ Clear',      0.52, 0.07, '#ffcccc',     '#ff6b6b'),
            ('▶ Loop',       0.60, 0.07, '#d0a8e8',     '#b07fd4'),
            ('Save',         0.68, 0.06, '#f9c74f',     '#f9a825'),
        ]
        callbacks = [
            self._on_load_folder,
            self._on_load_poses,
            self._on_prev,
            self._on_auto_toggle,
            self._on_skip,
            self._on_clear,
            self._on_loop_toggle,
            self._on_save,
        ]
        self._nav_btns = {}
        for (label, x, w, color, hover), cb in zip(defs, callbacks):
            ax = self.fig.add_axes([x, btn_y, w, btn_h])
            btn = Button(ax, label, color=color, hovercolor=hover)
            btn.label.set_fontsize(12)
            btn.on_clicked(cb)
            self._nav_btns[label] = btn

        # ── Batch size control  [−] [ Batch: 20 ] [+] ─────────────────────
        bx = 0.752
        ax_dec = self.fig.add_axes([bx,        btn_y, 0.025, btn_h])
        ax_inc = self.fig.add_axes([bx + 0.09, btn_y, 0.025, btn_h])
        self._btn_dec = Button(ax_dec, '−', color='#eeeeee', hovercolor='#cccccc')
        self._btn_inc = Button(ax_inc, '+', color='#eeeeee', hovercolor='#cccccc')
        self._btn_dec.label.set_fontsize(15)
        self._btn_inc.label.set_fontsize(15)
        self._btn_dec.on_clicked(self._on_batch_dec)
        self._btn_inc.on_clicked(self._on_batch_inc)

        # Batch-size display label (text axes between − and +)
        self._ax_batch = self.fig.add_axes([bx + 0.027, btn_y, 0.061, btn_h])
        self._ax_batch.set_facecolor('#fff8e1')
        self._ax_batch.axis('off')
        self._batch_text = self._ax_batch.text(
            0.5, 0.5, f'Batch: {self.batch_size}',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#333333', transform=self._ax_batch.transAxes
        )

        # File-info text (bottom-right)
        self._ax_finfo = self.fig.add_axes([0.882, btn_y, 0.115, btn_h])
        self._ax_finfo.axis('off')
        self._finfo_text = self._ax_finfo.text(
            0.0, 0.5,
            'p=loop  s=save',
            ha='left', va='center', fontsize=11, color='#888888',
            transform=self._ax_finfo.transAxes
        )

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_load_folder(self, event=None):
        root = tk.Tk(); root.withdraw()
        folder = filedialog.askdirectory(title='Select folder containing .bin files')
        root.destroy()
        if not folder:
            return
        self.bin_folder = folder
        bins = sorted(Path(folder).glob('*.bin'))
        if not bins:
            messagebox.showwarning('No .bin files',
                                   f'No .bin files found in:\n{folder}')
            return
        self.bin_files = [str(p) for p in bins]
        self.current_idx = 0
        self.labels = {}

        # Auto-load existing labels if present
        existing = Path(folder) / 'actions.npy'
        if existing.exists():
            arr = np.load(str(existing))
            self.labels = {i: int(v) for i, v in enumerate(arr) if v >= 0}
            print(f"Resumed: loaded {len(self.labels)} existing labels from {existing}")

        print(f"Loaded {len(self.bin_files)} .bin files from {folder}")
        self._refresh()

    def _on_load_poses(self, event=None):
        root = tk.Tk(); root.withdraw()
        pose_file = filedialog.askopenfilename(
            title='Select poses file (poses.txt)',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        root.destroy()
        if not pose_file:
            return
        try:
            self.poses = load_poses(pose_file)
            print(f"Loaded {len(self.poses)} poses from {pose_file}")
        except Exception as e:
            messagebox.showerror('Load Error', f'Failed to load poses:\n{e}')
            return
        self._refresh()

    def _on_prev(self, event=None):
        self._stop_loop()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._refresh()

    def _on_skip(self, event=None):
        """Jump forward by batch_size to the next unlabeled batch start."""
        n = len(self.bin_files)
        # Find the next frame index that starts an unlabeled batch
        for i in range(self.current_idx + self.batch_size, n, self.batch_size):
            if i not in self.labels:
                self.current_idx = i
                self._refresh()
                return
        # Fallback: fine-grained search for any unlabeled frame
        for i in range(self.current_idx + 1, n):
            if i not in self.labels:
                self.current_idx = i
                self._refresh()
                return
        # All labeled — move one batch forward
        nxt = min(self.current_idx + self.batch_size, n - 1)
        if nxt != self.current_idx:
            self.current_idx = nxt
            self._refresh()

    def _on_batch_dec(self, event=None):
        self.batch_size = max(1, self.batch_size - 1)
        self._batch_text.set_text(f'Batch: {self.batch_size}')
        self.fig.canvas.draw_idle()

    def _on_batch_inc(self, event=None):
        self.batch_size = min(200, self.batch_size + 1)
        self._batch_text.set_text(f'Batch: {self.batch_size}')
        self.fig.canvas.draw_idle()

    def _on_clear(self, event=None):
        """Remove labels for the current batch window."""
        n = len(self.bin_files)
        end = min(self.current_idx + self.batch_size, n)
        for i in range(self.current_idx, end):
            self.labels.pop(i, None)
        self._refresh()

    def _on_auto_toggle(self, event=None):
        self.auto_advance = not self.auto_advance
        btn = self._nav_btns.get('Auto ▶')
        if btn:
            if self.auto_advance:
                btn.label.set_text('Auto ▶  ON')
                btn.ax.set_facecolor('limegreen')
            else:
                btn.label.set_text('Auto ▶ OFF')
                btn.ax.set_facecolor('#dddddd')
        self.fig.canvas.draw_idle()

    def _on_loop_toggle(self, event=None):
        """Start / stop the 20-frame preview loop."""
        self.preview_playing = not self.preview_playing
        btn = self._nav_btns.get('▶ Loop')
        if self.preview_playing:
            self._preview_offset = 0
            self._preview_timer.start()
            if btn:
                btn.label.set_text('⏹ Loop')
                btn.ax.set_facecolor('#9b59b6')
        else:
            self._preview_timer.stop()
            self._preview_offset = 0
            if btn:
                btn.label.set_text('▶ Loop')
                btn.ax.set_facecolor('#d0a8e8')
            self._refresh()
        self.fig.canvas.draw_idle()

    def _stop_loop(self):
        """Silently stop the preview loop (called before navigation/labeling)."""
        if self.preview_playing:
            self.preview_playing = False
            self._preview_timer.stop()
            self._preview_offset = 0
            btn = self._nav_btns.get('▶ Loop')
            if btn:
                btn.label.set_text('▶ Loop')
                btn.ax.set_facecolor('#d0a8e8')

    def _preview_tick(self):
        """Timer callback: advance one frame within the batch and redraw BEV."""
        if not self.preview_playing or not self.bin_files:
            return
        n = len(self.bin_files)
        self._preview_offset = (self._preview_offset + 1) % self.batch_size
        frame_idx = min(self.current_idx + self._preview_offset, n - 1)
        self._draw_bev(frame_idx)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_bev(self, frame_idx: int):
        """Render only the BEV panel for the given frame index."""
        n = len(self.bin_files)
        self.ax_bev.clear()
        self.ax_bev.set_facecolor('white')
        self.ax_bev.set_xlim(-50, 50)
        self.ax_bev.set_ylim(-50, 50)
        self.ax_bev.set_aspect('equal')
        self.ax_bev.grid(True, alpha=0.4, color='#dddddd')
        self.ax_bev.tick_params(colors='#333333', labelsize=10)
        self.ax_bev.set_xlabel('X (m)', fontsize=11, color='#333333')
        self.ax_bev.set_ylabel('Y (m)', fontsize=11, color='#333333')
        for sp in self.ax_bev.spines.values():
            sp.set_edgecolor('#bbbbbb')

        try:
            pts = load_bin(self.bin_files[frame_idx])
            if len(pts) > 20000:
                pts = pts[::len(pts) // 20000]
            z_norm = ((pts[:, 2] - pts[:, 2].min())
                      / (pts[:, 2].max() - pts[:, 2].min() + 1e-6))
            self.ax_bev.scatter(pts[:, 0], pts[:, 1],
                                s=0.5, c=z_norm, cmap='viridis', alpha=0.85)
        except Exception as e:
            self.ax_bev.text(0, 0, f'Load error:\n{e}',
                             ha='center', va='center', color='tomato', fontsize=13)

        # Ego marker
        self.ax_bev.plot(0, 0, '^', color='#e74c3c', markersize=12,
                         markeredgecolor='#333333', markeredgewidth=1.3, zorder=10)
        self.ax_bev.annotate('EGO', (0, 0),
                             textcoords='offset points', xytext=(8, 8),
                             fontsize=10, color='#e74c3c', fontweight='bold')

        # Frame counter badge (top-right)
        batch_end  = min(self.current_idx + self.batch_size, n)
        offset_str = f'{self._preview_offset + 1}/{self.batch_size}'
        frame_str  = f'frame {frame_idx + 1} / {n}   ({offset_str})'
        self.ax_bev.text(0.98, 0.975, frame_str,
                         transform=self.ax_bev.transAxes,
                         fontsize=11, ha='right', va='top', color='#555555',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#fffde7', edgecolor='#f39c12',
                                   linewidth=1.2))

        # Current-label badge (top-left)
        cur_label = self.labels.get(self.current_idx)
        if cur_label is not None:
            bc = ACTION_COLORS[cur_label]
            btxt = (f"{ACTION_ARROWS[cur_label]}  {ACTION_LABELS[cur_label]}"
                    f"  ×{self.batch_size}")
            self.ax_bev.text(0.02, 0.975, btxt,
                             transform=self.ax_bev.transAxes,
                             fontsize=16, fontweight='bold',
                             va='top', color='white',
                             bbox=dict(boxstyle='round,pad=0.35',
                                       facecolor=bc, edgecolor='white',
                                       linewidth=1.5))
        else:
            range_str = f'frames {self.current_idx + 1}–{batch_end}'
            self.ax_bev.text(0.02, 0.975, f'?  Unlabeled  ({range_str})',
                             transform=self.ax_bev.transAxes,
                             fontsize=14, va='top', color='#e67e22',
                             bbox=dict(boxstyle='round,pad=0.35',
                                       facecolor='#fff3cd', edgecolor='#f39c12',
                                       linewidth=1.5))

        fname = Path(self.bin_files[self.current_idx]).name
        loop_tag = '  ⟳ LOOP' if self.preview_playing else ''
        self.ax_bev.set_title(
            f'{fname}   [ {self.current_idx + 1}–{batch_end} / {n} ]'
            f'   batch={self.batch_size}{loop_tag}',
            fontsize=14, fontweight='bold', color='#1a1a2e', pad=6)

    def _on_key(self, event):
        k = event.key
        if k == 'right':
            self._stop_loop()
            self.current_idx = min(self.current_idx + self.batch_size,
                                   len(self.bin_files) - 1)
            self._refresh()
        elif k == 'left':
            self._stop_loop()
            self.current_idx = max(self.current_idx - self.batch_size, 0)
            self._refresh()
        elif k in KEY_SHORTCUTS:
            self._stop_loop()
            self._on_action_clicked(KEY_SHORTCUTS[k])
        elif k == 'p':
            self._on_loop_toggle()
        elif k == 's':
            self._on_save()
        elif k == ' ':
            self._stop_loop()
            self._on_skip()
        elif k in ('delete', 'backspace'):
            self._stop_loop()
            self._on_clear()
        elif k == ']':
            self._on_batch_inc()
        elif k == '[':
            self._on_batch_dec()

    def _on_action_clicked(self, action_id: int):
        if not self.bin_files:
            return
        n = len(self.bin_files)
        start = self.current_idx
        end   = min(start + self.batch_size, n)

        # Label every frame in the batch window
        for i in range(start, end):
            self.labels[i] = action_id

        x, y, z = self._get_xyz(start)
        n_labeled = sum(1 for i in self.labels if i < n)
        print(f"[{start + 1:04d}–{end:04d}/{n}]  "
              f"XYZ=({x:+.2f}, {y:+.2f}, {z:+.2f})  →  "
              f"{ACTION_LABELS[action_id]} × {end - start} frames  "
              f"({n_labeled}/{n} labeled)")

        # Advance to the next batch
        if self.auto_advance:
            next_start = end
            if next_start < n:
                self.current_idx = next_start
            self._refresh()
        else:
            self._refresh()

    def _on_save(self, event=None):
        if not self.bin_files:
            messagebox.showwarning('Nothing to save', 'Load a folder first.')
            return
        n = len(self.bin_files)

        # Dense array: -1 for unlabeled frames
        arr = np.full(n, -1, dtype=np.int64)
        for i, a in self.labels.items():
            if 0 <= i < n:
                arr[i] = a

        out_dir = Path(self.bin_folder)

        # actions.npy
        npy_path = out_dir / 'actions.npy'
        np.save(str(npy_path), arr)

        # actions.csv
        csv_path = out_dir / 'actions.csv'
        with open(str(csv_path), 'w') as f:
            f.write('frame,x,y,z,action,action_name\n')
            for i in range(n):
                if arr[i] >= 0:
                    x, y, z = self._get_xyz(i)
                    f.write(f"{i},{x:.4f},{y:.4f},{z:.4f},"
                            f"{arr[i]},{ACTION_LABELS[arr[i]]}\n")

        # actions_metadata.json
        labeled_vals = [v for v in arr if v >= 0]
        unique, counts = (np.unique(labeled_vals, return_counts=True)
                          if labeled_vals else ([], []))
        meta = {
            'num_frames':       n,
            'num_labeled':      int(len(labeled_vals)),
            'coverage_pct':     round(100 * len(labeled_vals) / n, 1) if n else 0,
            'action_counts':    {int(k): int(v) for k, v in zip(unique, counts)},
            'action_names':     ACTION_LABELS,
        }
        with open(str(out_dir / 'actions_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        msg = (f"Saved {len(labeled_vals)}/{n} labels\n\n"
               f"{npy_path}\n{csv_path}")
        print(f"\nSaved → {npy_path}  ({len(labeled_vals)}/{n} labeled)")
        messagebox.showinfo('Saved', msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_xyz(self, idx: int):
        if self.poses is not None and idx < len(self.poses):
            p = self.poses[idx]
            return float(p[0, 3]), float(p[1, 3]), float(p[2, 3])
        return 0.0, 0.0, 0.0

    # ── Display refresh ───────────────────────────────────────────────────────

    def _refresh(self):
        n = len(self.bin_files)
        idx = self.current_idx

        # ── BEV panel ─────────────────────────────────────────────────────
        if n > 0:
            self._draw_bev(idx)

        # ── Info table ────────────────────────────────────────────────────
        self.ax_info.clear()
        self.ax_info.set_facecolor('white')
        self.ax_info.axis('off')

        rows = []   # (label_str, value_str, value_color)
        if n > 0:
            batch_end = min(idx + self.batch_size, n)
            rows.append(('Frames', f'{idx + 1} – {batch_end}  / {n}', '#333333'))
            if self.poses is not None and idx < len(self.poses):
                x, y, z, roll, pitch, yaw = pose_xyzrpy(self.poses[idx])
                rows += [
                    ('X', f'{x:+.3f} m', '#2c3e50'),
                    ('Y', f'{y:+.3f} m', '#2c3e50'),
                    ('Z', f'{z:+.3f} m', '#2c3e50'),
                    ('Yaw',   f'{yaw:+.1f}°',   '#7f8c8d'),
                    ('Pitch', f'{pitch:+.1f}°', '#7f8c8d'),
                    ('Roll',  f'{roll:+.1f}°',  '#7f8c8d'),
                ]
            else:
                rows.append(('Pose', 'not loaded', '#aaaaaa'))

            n_labeled = sum(1 for i in self.labels if 0 <= i < n)
            rows.append(('Labeled', f'{n_labeled} / {n}', '#27ae60'))

            cur_label = self.labels.get(idx)
            if cur_label is not None:
                ac = ACTION_COLORS[cur_label]
                rows.append(('Action',
                              f"{ACTION_ARROWS[cur_label]}  {ACTION_LABELS[cur_label]}",
                              ac))
            else:
                rows.append(('Action', '— not set —', '#f39c12'))

        row_h = 1.0 / (max(len(rows), 1) + 0.5)
        for i, (lbl, val, vcol) in enumerate(rows):
            yp = 1.0 - (i + 0.8) * row_h
            self.ax_info.text(0.04, yp, lbl,
                              fontsize=12, ha='left', va='center',
                              color='#999999',
                              transform=self.ax_info.transAxes)
            self.ax_info.text(0.96, yp, val,
                              fontsize=13, ha='right', va='center',
                              color=vcol, fontweight='bold',
                              transform=self.ax_info.transAxes)
            if i < len(rows) - 1:
                self.ax_info.axhline(yp - row_h * 0.42,
                                     xmin=0.02, xmax=0.98,
                                     color='#eeeeee', linewidth=0.7)

        self.ax_info.set_title('Frame Info', fontsize=12, fontweight='bold', pad=4)

        # ── Trajectory panel ──────────────────────────────────────────────
        self.ax_traj.clear()
        self.ax_traj.set_facecolor('#fafafa')
        self.ax_traj.set_title('Trajectory', fontsize=12, fontweight='bold', pad=3)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.tick_params(labelsize=9)
        self.ax_traj.grid(True, alpha=0.3)

        if self.poses is not None and len(self.poses) > 0:
            xs = self.poses[:, 0, 3]
            ys = self.poses[:, 1, 3]
            # Full path (gray)
            self.ax_traj.plot(xs, ys, '-', color='#cccccc', linewidth=1.0, zorder=1)
            # Color each labeled frame dot by action
            for i in range(len(self.poses)):
                lbl = self.labels.get(i)
                if lbl is not None:
                    self.ax_traj.plot(xs[i], ys[i], 'o',
                                      color=ACTION_COLORS[lbl],
                                      markersize=3, zorder=2)
            # Highlight current batch window (yellow segment)
            batch_end = min(idx + self.batch_size, len(self.poses))
            if batch_end > idx:
                self.ax_traj.plot(xs[idx:batch_end], ys[idx:batch_end],
                                  '-', color='#f1c40f', linewidth=3,
                                  alpha=0.7, zorder=3)
            # Current position marker
            if idx < len(self.poses):
                self.ax_traj.plot(xs[idx], ys[idx], '^',
                                  color='#e74c3c', markersize=7,
                                  markeredgecolor='white',
                                  markeredgewidth=1, zorder=5)
            # Batch-end marker
            if batch_end < len(self.poses):
                self.ax_traj.plot(xs[batch_end - 1], ys[batch_end - 1], 'D',
                                  color='#f1c40f', markersize=4,
                                  markeredgecolor='#333333',
                                  markeredgewidth=0.5, zorder=4)

        # ── Progress + distribution ────────────────────────────────────────
        self.ax_prog.clear()
        self.ax_prog.set_facecolor('white')
        self.ax_prog.axis('off')

        if n > 0:
            n_labeled = sum(1 for i in self.labels if 0 <= i < n)
            frac = n_labeled / n

            # Progress bar background
            self.ax_prog.add_patch(mpatches.Rectangle(
                (0.05, 0.80), 0.90, 0.10,
                transform=self.ax_prog.transAxes,
                facecolor='#e0e0e0', zorder=1))
            if frac > 0:
                self.ax_prog.add_patch(mpatches.Rectangle(
                    (0.05, 0.80), 0.90 * frac, 0.10,
                    transform=self.ax_prog.transAxes,
                    facecolor='#27ae60', zorder=2))

            self.ax_prog.text(0.5, 0.94,
                              f'{n_labeled} / {n} labeled  ({frac:.0%})',
                              ha='center', va='center', fontsize=12,
                              fontweight='bold',
                              color='#27ae60' if frac > 0 else '#aaaaaa',
                              transform=self.ax_prog.transAxes)

            # Per-action mini-bar chart
            counts = [0] * 12
            for v in self.labels.values():
                if 0 <= v < 12:
                    counts[v] += 1
            max_c = max(counts) if max(counts) > 0 else 1

            self.ax_prog.text(0.5, 0.72, 'Action distribution',
                              ha='center', va='center', fontsize=11,
                              color='#888888',
                              transform=self.ax_prog.transAxes)

            bar_y0   = 0.05
            bar_h_max = 0.60
            bar_w    = 0.90 / 12
            for i, c in enumerate(counts):
                bx = 0.05 + i * bar_w
                bh = (c / max_c) * bar_h_max if max_c > 0 else 0
                self.ax_prog.add_patch(mpatches.Rectangle(
                    (bx, bar_y0), bar_w * 0.80, bh,
                    transform=self.ax_prog.transAxes,
                    facecolor=ACTION_COLORS[i], zorder=2))
                if c > 0:
                    self.ax_prog.text(bx + bar_w * 0.40,
                                      bar_y0 + bh + 0.03,
                                      str(c),
                                      ha='center', va='bottom',
                                      fontsize=8, color='#555555',
                                      transform=self.ax_prog.transAxes)

        self.ax_prog.set_title('Progress', fontsize=12, fontweight='bold', pad=3)

        # ── Action button highlighting ─────────────────────────────────────
        cur_label = self.labels.get(idx)
        for i, (btn, ax_b) in enumerate(zip(self._btn_objs, self._btn_axes)):
            base = ACTION_COLORS[i]
            if cur_label == i:
                # Selected for this frame: fully opaque + thick border
                ax_b.set_facecolor(base)
                for sp in ax_b.spines.values():
                    sp.set_edgecolor('white')
                    sp.set_linewidth(2.5)
                btn.label.set_color('white')
            else:
                ax_b.set_facecolor(base + 'aa')
                for sp in ax_b.spines.values():
                    sp.set_edgecolor('#cccccc')
                    sp.set_linewidth(0.8)
                dark_actions = {0, 3, 4, 9}
                btn.label.set_color('white' if i in dark_actions else '#222222')

        # ── File info strip ────────────────────────────────────────────────
        if n > 0:
            folder_name = Path(self.bin_folder).name
            fname = Path(self.bin_files[idx]).name
            self._finfo_text.set_text(
                f'📁 {folder_name} / {fname}   '
                f'  Keys: ←→ navigate  |  0-9 a d label  |  Space skip  |  s save'
            )
        else:
            self._finfo_text.set_text(
                'Load a folder of .bin files and a poses.txt to get started.'
            )

        self.fig.canvas.draw_idle()

    # ── Entry ─────────────────────────────────────────────────────────────────

    def load_folder(self, folder: str):
        """Programmatically load a bin folder (called from CLI args)."""
        self.bin_folder = folder
        bins = sorted(Path(folder).glob('*.bin'))
        self.bin_files = [str(p) for p in bins]
        self.current_idx = 0
        labels_file = Path(folder) / 'actions.npy'
        if labels_file.exists():
            arr = np.load(str(labels_file))
            self.labels = {i: int(v) for i, v in enumerate(arr) if v >= 0}
            print(f"Resumed: {len(self.labels)} existing labels loaded.")
        print(f"Loaded {len(self.bin_files)} .bin files from {folder}")

    def load_poses_file(self, path: str):
        """Programmatically load a poses file (called from CLI args)."""
        self.poses = load_poses(path)
        print(f"Loaded {len(self.poses)} poses from {path}")

    def run(self):
        print()
        print('TerrainFormer Action Labeler')
        print('────────────────────────────')
        print('Keyboard shortcuts:')
        print('  ← / →   navigate frames')
        print('  0-9     label actions 0-9')
        print('  a / d   Fwd+Left (10) / Fwd+Right (11)')
        print('  Space   skip to next unlabeled')
        print('  s       save')
        print('  Del     clear current label')
        print()
        plt.show(block=True)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='TerrainFormer Action Labeler — manually label LiDAR frames'
    )
    parser.add_argument('--folder', type=str, default=None,
                        help='Folder containing .bin point cloud files')
    parser.add_argument('--poses', type=str, default=None,
                        help='Path to poses.txt (3×4 transformation matrix per line)')
    args = parser.parse_args()

    app = ActionLabeler()

    if args.folder:
        app.load_folder(args.folder)
        app._refresh()

    if args.poses:
        try:
            app.load_poses_file(args.poses)
            app._refresh()
        except Exception as e:
            print(f'Warning: could not load poses: {e}')

    app.run()


if __name__ == '__main__':
    main()
