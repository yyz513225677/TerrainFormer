"""
TerrainFormer Architecture Flowchart Generator
Generates paper/terrainformer_flowchart.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
C = {
    'data':    '#E8F4FD',   # light blue   – data / inputs
    'enc':     '#FFF3E0',   # light orange – LiDAR encoder
    'world':   '#E8F5E9',   # light green  – world model
    'dec':     '#F3E5F5',   # light purple – decision transformer
    'loss':    '#FBE9E7',   # light red    – loss functions
    'out':     '#E0F7FA',   # light cyan   – outputs
    'phase1':  '#E3F2FD',   # phase 1 background
    'phase2':  '#FFF8E1',   # phase 2 background
    'arrow':   '#455A64',   # dark slate   – arrows
    'border1': '#1565C0',   # phase 1 border
    'border2': '#E65100',   # phase 2 border
    'frozen':  '#CFD8DC',   # frozen component
}

FIG_W, FIG_H = 28, 56
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('#FAFAFA')


# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, color, text, bold_first=False, fontsize=8.5,
        edge='#546E7A', lw=1.2, style='round,pad=0.1', alpha=1.0,
        text_color='#212121'):
    """Draw a rounded-rect box centred at (cx,cy) with size (w,h)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=style, linewidth=lw,
        edgecolor=edge, facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    lines = text.split('\n')
    if bold_first and len(lines) > 0:
        ax.text(cx, cy + (len(lines)-1)*fontsize*0.022,
                lines[0], ha='center', va='center',
                fontsize=fontsize+0.5, fontweight='bold', color=text_color,
                zorder=4, linespacing=1.4)
        body = '\n'.join(lines[1:])
        if body:
            ax.text(cx, cy - fontsize*0.020,
                    body, ha='center', va='center',
                    fontsize=fontsize-0.5, color=text_color,
                    zorder=4, linespacing=1.45, family='monospace')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, color=text_color,
                zorder=4, linespacing=1.45)
    return rect


def arr(ax, x1, y1, x2, y2, label='', color='#455A64', lw=1.5,
        style='->', label_side='right'):
    """Draw a straight arrow with optional label."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx = 0.18 if label_side == 'right' else -0.18
        ax.text(mx+dx, my, label, ha='left' if label_side == 'right' else 'right',
                va='center', fontsize=7, color='#37474F', style='italic',
                zorder=5)


def phase_bg(ax, x, y, w, h, color, edge, title, lw=2):
    """Phase background: shaded rect + left-side vertical banner label."""
    # shaded background
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.2', linewidth=lw,
        edgecolor=edge, facecolor=color, alpha=0.30, zorder=1)
    ax.add_patch(rect)
    # left-side solid banner (1.0 wide)
    banner = FancyBboxPatch(
        (x, y), 1.0, h,
        boxstyle='square,pad=0', linewidth=lw,
        edgecolor=edge, facecolor=edge, alpha=0.85, zorder=6)
    ax.add_patch(banner)
    # rotated white text inside the banner
    ax.text(x + 0.5, y + h/2, title,
            ha='center', va='center',
            fontsize=9, fontweight='bold', color='white',
            rotation=90, zorder=7, linespacing=1.3)


def section_label(ax, x, y, text, color='#37474F'):
    ax.text(x, y, text, ha='left', va='center',
            fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.8), zorder=6)


# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, FIG_H - 0.8,
        'TerrainFormer: Complete Architecture',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color='#1A237E', zorder=6)
ax.text(FIG_W/2, FIG_H - 1.5,
        'Off-Road Navigation via Cross-Dataset World Model + Decision Transformer',
        ha='center', va='center', fontsize=10, color='#455A64', zorder=6)

# ═══════════════════════════════════════════════════════════════════════════
# ── DATA SOURCES ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_DS = FIG_H - 3.5

# Phase 1 datasets
box(ax,  6.0, Y_DS, 5.5, 1.0, C['data'],
    'LidarDustX\n174 sequences · 7,562 frames\n6-sensor array', bold_first=True, fontsize=8)
box(ax, 13.0, Y_DS, 5.5, 1.0, C['data'],
    'GOOSE-3D\n23 scenes · 7,719 frames\n128-ch LiDAR', bold_first=True, fontsize=8)

# Phase 2 dataset
box(ax, 22.0, Y_DS, 5.5, 1.0, C['data'],
    'RELLIS-3D\n5 sequences · ~13,000 frames\n64-ch LiDAR · 70/15/15 split', bold_first=True, fontsize=8)

# Dataset labels
ax.text(9.5,  Y_DS + 0.75, 'Phase 1 Pretraining Datasets', ha='center', fontsize=8,
        style='italic', color=C['border1'])
ax.text(22.0, Y_DS + 0.75, 'Phase 2 Fine-tuning Dataset', ha='center', fontsize=8,
        style='italic', color=C['border2'])

# ═══════════════════════════════════════════════════════════════════════════
# ── LIDAR INPUT ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_IN = Y_DS - 2.5

# Phase 1 input
box(ax, 9.5, Y_IN, 7.0, 0.85, C['data'],
    'LiDAR Point Cloud  [Phase 1]\n(B, N, 4)  |  x, y, z, intensity  |  N ≤ 65,536',
    bold_first=True, fontsize=8.5)

# Phase 2 input
box(ax, 22.0, Y_IN, 6.5, 0.85, C['data'],
    'LiDAR Point Cloud  [Phase 2]\n(B, N, 4)  |  RELLIS-3D only',
    bold_first=True, fontsize=8.5)

arr(ax,  6.0, Y_DS-0.5, 9.5,  Y_IN+0.42,  label='', color=C['border1'])
arr(ax, 13.0, Y_DS-0.5, 9.5,  Y_IN+0.42,  label='', color=C['border1'])
arr(ax, 22.0, Y_DS-0.5, 22.0, Y_IN+0.42,  label='', color=C['border2'])

# ═══════════════════════════════════════════════════════════════════════════
# ── POINT PILLAR PROJECTION ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_PP = Y_IN - 3.2

box(ax, 14.0, Y_PP, 15.5, 2.5, C['enc'],
    'PointPillarProjection  (Shared LiDAR Encoder)',
    bold_first=True, fontsize=9.5, edge='#BF360C', lw=1.8)

# sub-boxes inside
box(ax, 8.5, Y_PP+0.4, 5.8, 0.85, '#FFF8F5',
    'Pillarization\nmax_pillars=12,000 · max_pts=32\nGrid: 256×256, res=0.39 m/px', fontsize=7.5)
box(ax, 14.5, Y_PP+0.4, 5.8, 0.85, '#FFF8F5',
    'PFN (PointNet Feature Net)\nLinear(9→64) → ReLU → Linear(64→64)\nMax-pool over pts → (B,64,H,W)', fontsize=7.5)
box(ax, 20.5, Y_PP+0.4, 4.5, 0.85, '#FFF8F5',
    '2D Backbone\n2 × Conv2d(64,3×3)+BN+ReLU\n~5 ms / frame', fontsize=7.5)

box(ax, 14.0, Y_PP-0.55, 6.5, 0.55, '#FFF3E0',
    'BEV Features  (B, 64, 256, 256)',
    fontsize=8.5, edge='#BF360C', lw=1.2)

arr(ax, 9.5, Y_IN-0.42, 9.5, Y_PP+0.82,   label='(B,N,4)', color=C['border1'])
arr(ax, 22.0, Y_IN-0.42, 22.0, Y_PP+0.82, label='(B,N,4)', color=C['border2'])
arr(ax, 8.5, Y_PP-0.02, 11.5, Y_PP-0.02, label='', color='#455A64', lw=1)
arr(ax, 14.5, Y_PP-0.02, 18.0, Y_PP-0.02, label='', color='#455A64', lw=1)
arr(ax, 14.0, Y_PP+0.1, 14.0, Y_PP-0.28, label='', color='#455A64')

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════
Y_P1_BOTTOM = Y_PP - 12.2
Y_P1_TOP    = Y_PP - 1.0
phase_bg(ax, 0.3, Y_P1_BOTTOM, 27.4, Y_P1_TOP - Y_P1_BOTTOM,
         C['phase1'], C['border1'],
         'PHASE 1\n—\nWorld Model\nPretraining\n\n47.70M params\n100 epochs\n~13 h')

# ═══════════════════════════════════════════════════════════════════════════
# ── WORLD MODEL ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_WM_TOP = Y_PP - 1.2

# ── TerrainTokenizer ──
Y_TK = Y_WM_TOP
box(ax, 14.0, Y_TK, 15.5, 2.2, C['world'],
    'TerrainTokenizer',
    bold_first=True, fontsize=9.5, edge='#2E7D32', lw=1.6)

box(ax, 8.8,  Y_TK+0.3, 4.8, 0.85, '#F1F8E9',
    'PatchEmbed\nConv2d(64→512, k=16, s=16)\n(B,64,256,256)→(B,256,512)', fontsize=7.5)
box(ax, 14.0, Y_TK+0.3, 4.8, 0.85, '#F1F8E9',
    'CLS Token  (1,1,512)\n+ LearnedPosEmbed (1,256,512)\n+ TemporalEmbed (1,T,1,512)', fontsize=7.5)
box(ax, 19.5, Y_TK+0.3, 4.0, 0.85, '#F1F8E9',
    'Dropout(0.1)\nOutput tokens:\n(B, 257, 512)', fontsize=7.5)
box(ax, 14.0, Y_TK-0.7, 6.0, 0.52, '#E8F5E9',
    '→  tokens  (B, 257, 512)', fontsize=8.5, edge='#2E7D32', lw=1.2)

arr(ax, 14.0, Y_PP-0.82, 14.0, Y_TK+1.1, label='', color=C['border1'])

# ── DynamicsTransformer ──
Y_DT = Y_TK - 2.8
box(ax, 14.0, Y_DT, 15.5, 2.6, C['world'],
    'DynamicsTransformer  (L=6, H=8, d=512)',
    bold_first=True, fontsize=9.5, edge='#2E7D32', lw=1.6)

box(ax, 8.8, Y_DT+0.5, 4.5, 1.3, '#F1F8E9',
    'TransformerBlock × 6\n─────────────────\nLN → MHA(H=8)\nhead_dim = 64\nscale = 1/√64', fontsize=7.5)
box(ax, 14.0, Y_DT+0.5, 4.5, 1.3, '#F1F8E9',
    'Feed-Forward × 6\n─────────────────\nLN → Linear(512→2048)\nGELU\nLinear(2048→512)', fontsize=7.5)
box(ax, 19.5, Y_DT+0.5, 4.0, 1.3, '#F1F8E9',
    'Residuals +\nDropout(0.1)\n─────────────\nFinal LN(512)\n~40M params', fontsize=7.5)
box(ax, 14.0, Y_DT-0.85, 6.5, 0.52, '#E8F5E9',
    '→  tokens  (B, 257, 512)  [enriched]', fontsize=8.5, edge='#2E7D32', lw=1.2)

arr(ax, 14.0, Y_TK-0.96, 14.0, Y_DT+1.3, label='(B,257,512)', color='#388E3C')

# ── LatentState ──
Y_LS = Y_DT - 3.4
box(ax, 14.0, Y_LS, 15.5, 2.8, C['world'],
    'LatentState  (Encoder–Decoder)',
    bold_first=True, fontsize=9.5, edge='#2E7D32', lw=1.6)

box(ax, 7.8,  Y_LS+0.55, 4.5, 1.4, '#F1F8E9',
    'Cross-Attn Encoder\n──────────────────\nLearnable queries\n(1, 64, 512)\nMHA(d=512, H=8)\n→ (B, 64, 512)', fontsize=7.5)
box(ax, 13.0, Y_LS+0.55, 4.5, 1.4, '#F1F8E9',
    'Cross-Attn Decoder\n──────────────────\nOutput queries\n(1, 256, 512)\nMHA(d=512, H=8)\n→ (B, 256, 512)', fontsize=7.5)
box(ax, 18.5, Y_LS+0.55, 4.5, 1.4, '#F1F8E9',
    'Global Pooling\n──────────────────\nMean over 64 tokens\nMLP(512→512)\n→ (B, 512)\nLatent shape:(B,64,512)', fontsize=7.5)
box(ax, 14.0, Y_LS-0.85, 8.5, 0.52, '#E8F5E9',
    'latent(B,64,512)  |  decoded(B,256,512)  |  global(B,512)',
    fontsize=8, edge='#2E7D32', lw=1.2)

arr(ax, 14.0, Y_DT-1.12, 14.0, Y_LS+1.4, label='(B,257,512)', color='#388E3C')

# ── PredictionHeads ──
Y_PH = Y_LS - 3.6
box(ax, 14.0, Y_PH, 22.0, 2.8, C['world'],
    'PredictionHeads  (Phase 1 auxiliary outputs)',
    bold_first=True, fontsize=9.5, edge='#2E7D32', lw=1.6)

# 4 heads
for i, (lbl, detail, cx) in enumerate([
    ('TraversabilityHead',
     '4×ConvTranspose2d\n(512→256→128→64→1)\n+ Sigmoid\n→ (B, 1, 256, 256)', 5.5),
    ('ElevationHead',
     '4×ConvTranspose2d\n(512→256→128→64→1)\nNo activation\n→ (B, 1, 256, 256)', 10.5),
    ('SemanticHead',
     '4×ConvTranspose2d\n(512→256→128→64→20)\nLogits\n→ (B, 20, 256, 256)', 15.5),
    ('FuturePredHead',
     'Conv2d + reshape\n→ (B, 10, 64, 16, 16)\nFuture BEV frames\nK=10 horizon', 21.5),
]):
    box(ax, cx, Y_PH+0.3, 4.2, 1.55, '#F1F8E9', f'{lbl}\n{detail}',
        bold_first=True, fontsize=7.5)

arr(ax, 14.0, Y_LS-1.12, 14.0, Y_PH+1.4, label='decoded→(B,512,16,16)', color='#388E3C')

# ── WorldModelLoss ──
Y_WL = Y_PH - 2.0
box(ax, 14.0, Y_WL, 16.0, 1.35, C['loss'],
    'WorldModelLoss  (Phase 1 Training Objective)',
    bold_first=True, fontsize=9.5, edge='#B71C1C', lw=1.6)
box(ax, 14.0, Y_WL-0.02, 15.0, 0.85, '#FBE9E7',
    'L = 0.5·BCE(trav) + 0.3·MSE(elev) + 0.5·CE(sem)     +  [future contrastive optional]',
    fontsize=8.5)

arr(ax, 14.0, Y_PH-1.4, 14.0, Y_WL+0.67, label='predictions', color='#B71C1C')

# ── Optimiser note ──
box(ax, 14.0, Y_WL-1.2, 12.0, 0.6, '#ECEFF1',
    'AdamW  (lr=1e-4, wd=0.01, β=[0.9,0.999])  |  CosineAnnealing  (warmup=5ep, min_lr=1e-6)  |  fp16  |  grad_clip=1.0',
    fontsize=7.5, edge='#607D8B')
arr(ax, 14.0, Y_WL-0.67, 14.0, Y_WL-0.9, color='#B71C1C')

# ═══════════════════════════════════════════════════════════════════════════
# FROZEN marker
# ═══════════════════════════════════════════════════════════════════════════
Y_FREEZE = Y_WL - 2.2
ax.text(FIG_W/2, Y_FREEZE+0.35,
        '❄  World Model + LiDAR Encoder frozen  →  Phase 2',
        ha='center', va='center', fontsize=10, color='#1565C0',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                  edgecolor='#1565C0', alpha=0.9), zorder=6)
arr(ax, 14.0, Y_WL-1.5, 14.0, Y_FREEZE+0.0, color='#1565C0', lw=2)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════
Y_P2_TOP    = Y_FREEZE - 0.1
Y_P2_BOTTOM = Y_FREEZE - 16.0
phase_bg(ax, 0.3, Y_P2_BOTTOM, 27.4, Y_P2_TOP - Y_P2_BOTTOM,
         C['phase2'], C['border2'],
         'PHASE 2\n—\nDecision\nTransformer\nTraining\n\n57.36M total\n9.66M trainable\n50 epochs\n~5 h')

# ═══════════════════════════════════════════════════════════════════════════
# ── ACTION TOKENIZER ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_AT = Y_FREEZE - 2.2

# Action vocabulary (left)
box(ax, 4.5, Y_AT, 5.5, 3.0, C['data'],
    'Action Vocabulary (12 classes)\n'
    '0: STOP        1: FWD_SLOW\n'
    '2: FWD_MED     3: FWD_FAST\n'
    '4: TURN_L_SHARP  5: TURN_L_MED\n'
    '6: TURN_L_SLT    7: TURN_R_SLT\n'
    '8: TURN_R_MED    9: TURN_R_SHARP\n'
    '10: FWD_LEFT    11: FWD_RIGHT',
    bold_first=True, fontsize=7.5, edge='#01579B')

# Vehicle state / goal (right)
box(ax, 22.5, Y_AT, 5.5, 2.0, C['data'],
    'Vehicle State & Goal\n'
    'state (B,6): [vx, vy, ω, pitch, roll, yaw]\n'
    'goal  (B,2): [Δx, Δy] to waypoint',
    bold_first=True, fontsize=8, edge='#01579B')

# ActionTokenizer (centre)
box(ax, 13.0, Y_AT, 8.5, 2.2, C['dec'],
    'ActionTokenizer',
    bold_first=True, fontsize=9.5, edge='#6A1B9A', lw=1.6)
box(ax, 13.0, Y_AT-0.05, 7.5, 1.5, '#F3E5F5',
    'Embedding(12+1→128, padding_idx=12)\n+ TemporalEmbedding(10→128)\nAction History (B,10) → (B,10,128)\n+ LayerNorm → aggregate → (B,128)', fontsize=8)

arr(ax, 4.5, Y_AT-1.5, 9.3, Y_AT-0.3, label='action_history (B,10)', color='#6A1B9A')

# ═══════════════════════════════════════════════════════════════════════════
# ── CONTEXT AGGREGATOR ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_CA = Y_AT - 3.8

box(ax, 14.0, Y_CA, 22.0, 3.2, C['dec'],
    'ContextAggregator',
    bold_first=True, fontsize=9.5, edge='#6A1B9A', lw=1.6)

# 4 encoder boxes
for lbl, detail, cx in [
    ('StateEncoder\n(6→128)',    'Linear(6→128)\nLN→ReLU→Linear(128→128)\n→ (B,128)',   6.0),
    ('GoalEncoder\n(2→128)',     'Linear(2→128)\nLN→ReLU→Linear(128→128)\n→ (B,128)',   10.5),
    ('ActionAgg\n(128)',         'mean over T\n→ (B,128)',                               14.8),
    ('WorldProj\n(512→384)',     'Linear(512→384)\n→ (B,384)',                           19.2),
]:
    box(ax, cx, Y_CA+0.55, 3.5, 1.25, '#EDE7F6', f'{lbl}\n{detail}',
        bold_first=True, fontsize=7.5)

# Fusion
box(ax, 14.0, Y_CA-0.85, 13.0, 0.72, '#EDE7F6',
    'Concat(384+128+128+128=768)  →  Linear(768→768)+LN+ReLU  →  Linear(768→384)+LN  →  context (B,384)',
    fontsize=8)

# CrossAttn
box(ax, 14.0, Y_CA-1.85, 13.5, 0.78, '#EDE7F6',
    'CrossAttn(H=8):  Q=context(B,1,384)  K,V=world_latent_proj(B,64,384)\n→ attended (B,384)  +  Residual + LN  →  final context (B,384)',
    fontsize=8)

# Arrows into ContextAggregator
arr(ax, 22.5, Y_AT-1.0, 19.2, Y_CA+1.17, label='world_global(B,512)', color='#6A1B9A')
arr(ax, 22.5, Y_AT-1.0, 14.8, Y_CA+1.17, label='action_agg(B,128)',   color='#6A1B9A')
arr(ax, 22.5, Y_AT-1.0, 10.5, Y_CA+1.17, label='goal(B,2)',           color='#6A1B9A')
arr(ax, 22.5, Y_AT-1.0,  6.0, Y_CA+1.17, label='state(B,6)',          color='#6A1B9A')
arr(ax, 13.0, Y_AT-1.1, 14.8, Y_CA+1.17, label='action_embed(B,128)', color='#6A1B9A')

# latent from World Model
arr(ax, 9.5, Y_WL-1.5, 9.5, Y_CA-1.35, label='latent(B,64,512)', color='#388E3C', lw=1.8)
arr(ax, 9.5, Y_CA-1.35, 14.0, Y_CA-1.45, color='#388E3C', lw=1.5)

# ═══════════════════════════════════════════════════════════════════════════
# ── DECISION TRANSFORMER ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_DTR = Y_CA - 4.8

box(ax, 14.0, Y_DTR, 22.5, 4.5, C['dec'],
    'DecisionTransformer  (L=4, H=6, d=384)',
    bold_first=True, fontsize=9.5, edge='#6A1B9A', lw=1.8)

# Token sequence diagram
Y_SEQ = Y_DTR + 1.4
ax.text(14.0, Y_SEQ+0.68, 'Token Sequence  (B, 80, 384)', ha='center',
        fontsize=8.5, fontweight='bold', color='#4A148C', zorder=6)

seq_items = [
    ('context\ntoken\n(B,1,384)',  2.0, '#D1C4E9'),
    ('world\ntokens\n(B,64,384)', 5.5, '#C8E6C9'),
    ('action\ntokens\n(B,10,384)', 9.8, '#BBE0F7'),
    ('chunk\nqueries\n(B,5,384)',  13.2, '#FFE0B2'),
]
for lbl, cx_off, fc in seq_items:
    rect = FancyBboxPatch((cx_off+2.5, Y_SEQ-0.5), 2.4 if '64' in lbl else 1.5, 0.95,
                          boxstyle='round,pad=0.05', linewidth=1,
                          edgecolor='#7B1FA2', facecolor=fc, alpha=0.9, zorder=4)
    ax.add_patch(rect)
    ax.text(cx_off+3.7 if '64' in lbl else cx_off+3.25, Y_SEQ-0.02,
            lbl, ha='center', va='center', fontsize=7, color='#1A237E', zorder=5)

ax.text(14.0, Y_SEQ-0.7,
        '1 + 64 + 10 + 5  =  80 total tokens\n'
        'Positional Embed: Parameter(1, 80, 384)  ·  Chunk queries: learnable Parameter(1, 5, 384)',
        ha='center', fontsize=7.5, color='#4A148C', zorder=5)

# 4 transformer blocks
Y_TBLK = Y_DTR - 0.5
for i, cx in enumerate([6.5, 10.5, 14.5, 18.5]):
    box(ax, cx, Y_TBLK, 3.5, 1.55, '#EDE7F6',
        f'Block {i+1}\n──────────────\nLN→MHA(H=6,d=384)\nhead_dim=64\nLN→MLP(384→1536→384)\n+Residuals',
        bold_first=True, fontsize=7.2)

arr(ax, 14.0, Y_CA-2.6, 14.0, Y_DTR+2.25, label='context(B,384)', color='#6A1B9A', lw=1.8)

# ═══════════════════════════════════════════════════════════════════════════
# ── OUTPUT EXTRACTION ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_OE = Y_DTR - 2.6

box(ax, 14.0, Y_OE, 22.0, 1.0, '#EDE7F6',
    'Output Extraction:   x[:,0] → (B,384)  [context token]      x[:,-5:] → (B,5,384)  [chunk queries]',
    fontsize=9, edge='#6A1B9A')

arr(ax, 14.0, Y_DTR-2.25, 14.0, Y_OE+0.5, label='(B,80,384)', color='#6A1B9A')

# ═══════════════════════════════════════════════════════════════════════════
# ── OUTPUT HEADS ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_OH = Y_OE - 2.8

box(ax, 14.0, Y_OH, 22.5, 2.4, C['out'],
    'OutputHeads',
    bold_first=True, fontsize=9.5, edge='#006064', lw=1.6)

# 5 heads
for lbl, detail, cx in [
    ('ActionHead',       '384→256→ReLU\n→256→12 logits\n→ (B,12)', 5.0),
    ('ChunkHead',        '(B,5,384) reshape\n→ActionHead(×5)\n→ (B,5,12)', 9.5),
    ('ConfidenceHead',   '384→128→ReLU\n→1→Sigmoid\n→ (B,1) ∈[0,1]', 14.0),
    ('TravHead (aux)',   '384→128→ReLU\n→1→Sigmoid\n→ (B,1)', 18.5),
    ('CollisionHead',    '384→128→ReLU\n→1→Sigmoid\n→ (B,1)', 23.2),
]:
    box(ax, cx, Y_OH-0.0, 3.8, 1.55, '#E0F7FA', f'{lbl}\n{detail}',
        bold_first=True, fontsize=7.5, edge='#006064')

arr(ax, 14.0, Y_OE-0.5, 14.0, Y_OH+1.2, label='(B,384) / (B,5,384)', color='#006064')

# ═══════════════════════════════════════════════════════════════════════════
# ── DECISION LOSS ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_DL = Y_OH - 2.4

box(ax, 14.0, Y_DL, 17.5, 1.6, C['loss'],
    'DecisionLoss  (Phase 2 Training Objective)',
    bold_first=True, fontsize=9.5, edge='#B71C1C', lw=1.6)
box(ax, 14.0, Y_DL-0.05, 16.5, 1.0, '#FBE9E7',
    'L = 1.0·FocalLoss(action, γ=2.0, ls=0.1)\n'
    '  + 0.2·BCE(traversability_aux)\n'
    '  + 0.3·BCE(collision_aux)\n'
    '  + 0.5·FocalLoss(action_chunk)  [K=5 future steps]',
    fontsize=8)

arr(ax, 14.0, Y_OH-1.2, 14.0, Y_DL+0.8, label='all outputs', color='#B71C1C')

# Optimiser note
box(ax, 14.0, Y_DL-1.55, 14.0, 0.6, '#ECEFF1',
    'AdamW  (lr=3e-4, wd=0.05, β=[0.9,0.95])  |  CosineAnnealing  (warmup=3ep, min_lr=1e-6)  |  fp16  |  BestCkpt on val_accuracy',
    fontsize=7.5, edge='#607D8B')
arr(ax, 14.0, Y_DL-0.8, 14.0, Y_DL-1.25, color='#B71C1C')

# ═══════════════════════════════════════════════════════════════════════════
# ── INFERENCE ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_INF = Y_DL - 3.2

box(ax, 10.0, Y_INF, 8.5, 2.0, C['out'],
    'Temporal Ensembling  (Inference)',
    bold_first=True, fontsize=9.5, edge='#006064', lw=1.6)
box(ax, 10.0, Y_INF-0.05, 7.5, 1.2, '#E0F7FA',
    'Buffer last K=5 chunk predictions\n'
    'Weighted average:  a* = argmax Σ λ^k · logits[k]\n'
    'Decay λ=0.9  →  smooth, jitter-free actions',
    fontsize=8)

box(ax, 21.5, Y_INF, 6.5, 1.8, C['out'],
    'Final Results  (RELLIS-3D Test)',
    bold_first=True, fontsize=9.5, edge='#006064', lw=1.8)
box(ax, 21.5, Y_INF-0.05, 5.5, 1.1, '#E0F7FA',
    'Accuracy: 87.31%  |  Macro F1: 0.7948\n'
    'Predictive drop: 0.79%\n'
    'Agreement w/ live: 98.82%',
    fontsize=8.5)

arr(ax, 14.0, Y_DL-2.1, 10.0, Y_INF+1.0, label='chunk_logits(B,5,12)', color='#006064', lw=1.8)
arr(ax, 10.0, Y_INF-1.0, 21.5, Y_INF-0.05, label='action*', color='#006064', lw=2.0)

# ═══════════════════════════════════════════════════════════════════════════
# ── LEGEND ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
Y_LEG = Y_INF - 2.5
legend_items = [
    (C['data'],  'Data / Input'),
    (C['enc'],   'LiDAR Encoder'),
    (C['world'], 'World Model'),
    (C['dec'],   'Decision Transformer'),
    (C['loss'],  'Loss Function'),
    (C['out'],   'Output / Inference'),
    (C['frozen'],'Frozen at Phase 2'),
]
lx = 1.5
for fc, label in legend_items:
    ax.add_patch(FancyBboxPatch((lx, Y_LEG-0.25), 1.2, 0.5,
                                boxstyle='round,pad=0.05',
                                facecolor=fc, edgecolor='#546E7A',
                                linewidth=1, zorder=5))
    ax.text(lx+1.4, Y_LEG, label, fontsize=7.5, va='center', color='#212121', zorder=5)
    lx += 3.0

ax.text(FIG_W/2, Y_LEG-0.7,
        'Shapes: B=batch · N=num_points(≤65,536) · T=temporal frames · K=chunk_size=5',
        ha='center', fontsize=7.5, style='italic', color='#546E7A', zorder=5)

# ── save ─────────────────────────────────────────────────────────────────────
out = 'paper/terrainformer_flowchart.png'
plt.tight_layout(pad=0)
plt.savefig(out, dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f'Saved {out}')
