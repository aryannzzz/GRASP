"""
Generate three professional architecture diagrams for a robotics research paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, label, bg_color, text_color='white',
                fontsize=10, bold=False, sublabel=None, alpha=1.0,
                border_color=None, lw=1.5):
    """Draw a rounded rectangle with centred label."""
    bc = border_color if border_color else bg_color
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.04",
                         facecolor=bg_color, edgecolor=bc,
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x, y - 0.022, sublabel,
                ha='center', va='center', fontsize=fontsize - 2,
                color=text_color, alpha=0.80, style='italic', zorder=4)

def arrow(ax, x0, y0, x1, y1, color='#a0aec0', lw=1.5, arrowsize=12,
          style='->', connectionstyle='arc3,rad=0.0'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f'->, head_width=0.4, head_length=0.4',
                                color=color, lw=lw,
                                connectionstyle=connectionstyle),
                zorder=5)

def section_label(ax, x, y, text, color='#e2e8f0', fontsize=11):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b',
                      edgecolor=color, linewidth=1, alpha=0.9), zorder=6)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 – ACT Architecture: Standard vs Modified
# ═════════════════════════════════════════════════════════════════════════════

def make_act_diagram(out_path):
    BG = '#0d1117'
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(7, 8.6, 'ACT Architecture: Standard vs Modified',
            ha='center', va='center', fontsize=16, color='white',
            fontweight='bold')
    ax.text(7, 8.2, 'Action Chunking with Transformers',
            ha='center', va='center', fontsize=11, color='#94a3b8')

    # ── Divider ────────────────────────────────────────────────────────────
    ax.plot([7, 7], [0.4, 8.0], color='#30363d', lw=1.5, ls='--', zorder=2)

    # ── Column headers ─────────────────────────────────────────────────────
    ax.text(3.5, 7.85, 'Standard ACT', ha='center', va='center',
            fontsize=13, color='#60a5fa', fontweight='bold')
    ax.text(10.5, 7.85, 'Modified ACT (Ours)', ha='center', va='center',
            fontsize=13, color='#34d399', fontweight='bold')

    # ── Color palette ──────────────────────────────────────────────────────
    C_IMG   = '#1d4ed8'    # blue  – image path
    C_ENC   = '#7c3aed'    # purple – encoder
    C_DEC   = '#0f766e'    # teal  – decoder
    C_ACT   = '#b45309'    # amber – action output
    C_CVAE  = '#be185d'    # pink  – CVAE standard
    C_CVAE2 = '#dc2626'    # red   – CVAE modified (key change)
    C_Z     = '#475569'    # slate – latent
    C_JS    = '#064e3b'    # dark green – joint state

    BW, BH   = 2.2, 0.46   # box width / height
    BW2, BH2 = 2.6, 0.46   # wider for CVAE labels

    # ══════════════════════════════════════════════════════════════════════
    # LEFT COLUMN – Standard ACT
    # ══════════════════════════════════════════════════════════════════════
    LX = 3.5          # centre-x of left column

    # Main visual stream (top → bottom)
    boxes_L = [
        (LX, 7.35, 'Image Observation',     C_IMG, BW, BH),
        (LX, 6.60, 'ResNet18 Encoder',       C_ENC, BW, BH),
        (LX, 5.50, 'Transformer Decoder',    C_DEC, BW, BH),
        (LX, 4.60, 'Action Chunk Output',    C_ACT, BW, BH),
    ]
    for (cx, cy, lbl, col, w, h) in boxes_L:
        rounded_box(ax, cx, cy, w, h, lbl, col, fontsize=10)

    # Arrows main stream
    for (_, y0, _, _, _, h0), (_, y1, _, _, _, h1) in zip(boxes_L, boxes_L[1:]):
        arrow(ax, LX, y0 - h0/2, LX, y1 + h1/2)

    # CVAE branch (left side)
    CVAE_X = 1.5

    # CLS + Joint State + Action Chunk input
    rounded_box(ax, CVAE_X, 7.35, 2.0, 0.48,
                '[CLS] Joint State', C_JS, fontsize=9,
                sublabel='+ Action Chunk')
    # μ, σ → z
    rounded_box(ax, CVAE_X, 6.28, 1.9, 0.44, 'CVAE Encoder', C_CVAE, fontsize=9)
    rounded_box(ax, CVAE_X, 5.50, 1.6, 0.40, 'μ, σ  →  z', C_Z, fontsize=9)

    # arrows: input → CVAE encoder → μσ → decoder
    arrow(ax, CVAE_X, 7.10, CVAE_X, 6.50)
    arrow(ax, CVAE_X, 6.06, CVAE_X, 5.70)
    # z feeds into decoder
    arrow(ax, CVAE_X + 0.8, 5.50, LX - BW/2, 5.50,
          connectionstyle='arc3,rad=0.0')

    # Inference note
    ax.text(CVAE_X, 5.00, 'Inference: z = 0',
            ha='center', va='center', fontsize=8.5,
            color='#fbbf24', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#27272a',
                      edgecolor='#fbbf24', linewidth=1))

    # ══════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN – Modified ACT
    # ══════════════════════════════════════════════════════════════════════
    RX = 10.5

    boxes_R = [
        (RX, 7.35, 'Image Observation',    C_IMG, BW, BH),
        (RX, 6.60, 'ResNet18 Encoder',      C_ENC, BW, BH),
        (RX, 5.50, 'Transformer Decoder',   C_DEC, BW, BH),
        (RX, 4.60, 'Action Chunk Output',   C_ACT, BW, BH),
    ]
    for (cx, cy, lbl, col, w, h) in boxes_R:
        rounded_box(ax, cx, cy, w, h, lbl, col, fontsize=10)

    for (_, y0, _, _, _, h0), (_, y1, _, _, _, h1) in zip(boxes_R, boxes_R[1:]):
        arrow(ax, RX, y0 - h0/2, RX, y1 + h1/2)

    # Modified CVAE branch
    CVAE_RX = 8.7

    rounded_box(ax, CVAE_RX, 7.35, 2.0, 0.48,
                '[CLS] Joint State', C_JS, fontsize=9,
                sublabel='+ Action Chunk')
    # KEY CHANGE box – wider, brighter red
    rounded_box(ax, CVAE_RX, 6.28, 2.2, 0.48,
                'CVAE Encoder (Modified)', C_CVAE2, fontsize=9,
                bold=True, border_color='#ff6b6b', lw=2.5)
    # Image features input arrow into modified CVAE
    rounded_box(ax, CVAE_RX, 5.50, 1.6, 0.40, 'μ, σ  →  z', C_Z, fontsize=9)

    arrow(ax, CVAE_RX, 7.10, CVAE_RX, 6.52)
    # Image features also fed to CVAE
    arrow(ax, RX - BW/2, 6.60, CVAE_RX + 1.1, 6.28,
          color='#f87171', connectionstyle='arc3,rad=-0.25')
    arrow(ax, CVAE_RX, 6.04, CVAE_RX, 5.70)
    arrow(ax, CVAE_RX + 0.8, 5.50, RX - BW/2, 5.50,
          connectionstyle='arc3,rad=0.0')

    # Inference note
    ax.text(CVAE_RX, 5.00, 'Inference: z = 0',
            ha='center', va='center', fontsize=8.5,
            color='#fbbf24', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#27272a',
                      edgecolor='#fbbf24', linewidth=1))

    # ── KEY CHANGE annotation ──────────────────────────────────────────────
    ax.annotate('KEY CHANGE:\nImages added to\nCVAE encoder',
                xy=(CVAE_RX + 0.2, 6.28),
                xytext=(12.6, 6.10),
                fontsize=8.5, color='#ff6b6b', fontweight='bold',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.8,
                                connectionstyle='arc3,rad=-0.15'),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#27272a',
                          edgecolor='#ff6b6b', linewidth=1.5),
                zorder=7)

    # ── Image features label on arrow ────────────────────────────────────
    ax.text(9.55, 6.7, 'Image\nFeatures', ha='center', va='bottom',
            fontsize=7.5, color='#f87171', style='italic', zorder=6)

    # ── Footer ─────────────────────────────────────────────────────────────
    footer_bg = FancyBboxPatch((1.5, 0.55), 11, 0.55,
                               boxstyle="round,pad=0.05",
                               facecolor='#1e1b4b', edgecolor='#4f46e5',
                               linewidth=1.5, zorder=3)
    ax.add_patch(footer_bg)
    ax.text(7, 0.82, '27.8% lower validation loss with Modified ACT',
            ha='center', va='center', fontsize=11,
            color='#a5b4fc', fontweight='bold', zorder=4)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'[OK] Saved: {out_path}')


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 – VLA-GradCAM Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def make_gradcam_diagram(out_path):
    BG = '#0d0d1a'
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(7, 8.65, 'VLA-GradCAM Pipeline: Language-Conditioned Attribution',
            ha='center', va='center', fontsize=15, color='white',
            fontweight='bold')
    ax.text(7, 8.25, 'Instruction-specific visual saliency for Vision-Language-Action models',
            ha='center', va='center', fontsize=10, color='#94a3b8')

    # ── Color palette ─────────────────────────────────────────────────────
    C_INPUT  = '#1e3a5f'
    C_VISION = '#312e81'
    C_TEXT   = '#1e3a5f'
    C_CROSS  = '#5b21b6'
    C_ACT    = '#065f46'
    C_BACK   = '#7c2d12'
    C_HEAT   = '#831843'
    C_PROB   = '#78350f'
    C_SOL    = '#134e4a'

    def rb(cx, cy, w, h, lbl, col, fs=9.5, sub=None, bold=False, bcol=None):
        rounded_box(ax, cx, cy, w, h, lbl, col, fontsize=fs,
                    sublabel=sub, bold=bold, border_color=bcol, lw=1.8)

    def ar(x0, y0, x1, y1, col='#818cf8', cs='arc3,rad=0.0', lw=1.6):
        arrow(ax, x0, y0, x1, y1, color=col, lw=lw, connectionstyle=cs)

    # ─── INFERENCE SPINE (top to bottom, centred at x=5) ──────────────────
    IX = 5.0
    BW, BH = 3.2, 0.50

    # Input row: camera + language side by side
    rb(3.0, 7.55,  2.4, 0.50, 'Camera Image',          C_INPUT, fs=10)
    rb(7.0, 7.55,  2.8, 0.50, 'Language Instruction',  C_TEXT, fs=10)

    # CLIP encoders
    rb(3.0, 6.60,  2.6, 0.50, 'CLIP Vision Encoder',   C_VISION,
       fs=9.5, sub='[196 patches]')
    rb(7.0, 6.60,  2.8, 0.50, 'CLIP Text Encoder',     C_VISION, fs=9.5)

    # Cross-attention pooling
    rb(5.0, 5.65,  3.6, 0.52, 'Cross-Attention Pooling',     C_CROSS,
       fs=10, sub='text attends to vision patches', bold=True)

    # Action head
    rb(5.0, 4.68,  3.0, 0.50, 'Action Head  (MLP)',    C_ACT, fs=10)

    # Robot action output
    rb(5.0, 3.72,  4.2, 0.52,
       'Robot Action', C_ACT, fs=10,
       sub='dx, dy, dz | roll, pitch, yaw | gripper', bold=True)

    # Arrows inference spine
    ar(3.0, 7.30, 3.0, 6.85)                        # camera → vision enc
    ar(7.0, 7.30, 7.0, 6.85)                        # lang → text enc
    ar(3.0, 6.35, 4.0, 5.91, col='#a78bfa',         # vision → cross-attn
       cs='arc3,rad=0.25')
    ar(7.0, 6.35, 6.0, 5.91, col='#a78bfa',         # text → cross-attn
       cs='arc3,rad=-0.25')
    ar(5.0, 5.39, 5.0, 4.93)                        # cross → action head
    ar(5.0, 4.43, 5.0, 3.98)                        # head → robot action

    # ─── ATTRIBUTION LOOP (right side) ────────────────────────────────────
    RX = 11.2
    AW, AH = 2.8, 0.50

    rb(RX, 4.68, AW, AH, 'Backpropagate',         C_BACK, fs=9.5,
       sub='∂action / ∂features')
    rb(RX, 3.72, AW, AH, 'Saliency Weights (αk)',  C_BACK, fs=9.5)
    rb(RX, 2.75, AW, AH, 'Weighted Sum of Features', C_HEAT, fs=9.5)
    rb(RX, 1.80, AW, AH, 'GradCAM Heatmap',        C_HEAT, fs=10,
       sub='overlaid on image', bold=True, bcol='#f472b6')

    # trigger from action head to backprop
    ar(5.0 + 3.0/2, 4.68, RX - AW/2, 4.68, col='#fb923c',
       cs='arc3,rad=0.0')
    ar(RX, 4.43, RX, 3.97)
    ar(RX, 3.47, RX, 3.00)
    ar(RX, 2.50, RX, 2.05)

    ax.text(RX, 5.10, 'ATTRIBUTION\nPATH', ha='center', va='center',
            fontsize=8, color='#fb923c', fontweight='bold')

    # ─── PROBLEM vs SOLUTION panel ────────────────────────────────────────
    # Background panels
    prob_bg = FancyBboxPatch((0.3, 0.45), 6.2, 1.85,
                             boxstyle='round,pad=0.05',
                             facecolor='#1c1917', edgecolor='#ef4444',
                             linewidth=1.5, zorder=2)
    sol_bg  = FancyBboxPatch((7.5, 0.45), 6.2, 1.85,
                             boxstyle='round,pad=0.05',
                             facecolor='#0a1a14', edgecolor='#10b981',
                             linewidth=1.5, zorder=2)
    ax.add_patch(prob_bg)
    ax.add_patch(sol_bg)

    ax.text(3.4, 2.18, 'PROBLEM: Mean Pooling', ha='center',
            fontsize=9, color='#ef4444', fontweight='bold', zorder=5)
    ax.text(3.4, 1.85, 'Uniform gradients — every patch\ncontributes equally regardless of instruction',
            ha='center', va='center', fontsize=8.5, color='#fca5a5', zorder=5)
    ax.text(3.4, 1.38, r'$\partial \mathcal{L} / \partial x_i = 1/N \;\forall\; i$',
            ha='center', va='center', fontsize=10, color='#fca5a5', zorder=5)

    ax.text(10.6, 2.18, 'SOLUTION: Attention Pooling', ha='center',
            fontsize=9, color='#10b981', fontweight='bold', zorder=5)
    ax.text(10.6, 1.85, 'Instruction-specific weights let gradients\nreflect language-guided visual attention',
            ha='center', va='center', fontsize=8.5, color='#6ee7b7', zorder=5)
    ax.text(10.6, 1.38, r'$\alpha_k = \text{softmax}(q_\text{text} \cdot k_\text{vis}^k)$',
            ha='center', va='center', fontsize=10, color='#6ee7b7', zorder=5)

    # 85% annotation
    ax.annotate('85% reduction in\nspurious correlation',
                xy=(RX, 1.80), xytext=(RX - 1.6, 1.05),
                fontsize=8.5, color='#f472b6', fontweight='bold',
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#f472b6', lw=1.6),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#27272a',
                          edgecolor='#f472b6', linewidth=1.3), zorder=8)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'[OK] Saved: {out_path}')


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 – Classical Perception-to-Action Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def make_classical_pipeline(out_path):
    BG = '#0a0f0d'
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(7, 8.65, 'Classical Perception-to-Action Pipeline',
            ha='center', va='center', fontsize=16, color='white',
            fontweight='bold')
    ax.text(7, 8.25, 'Modular, interpretable robot control without demonstration data',
            ha='center', va='center', fontsize=10, color='#94a3b8')

    # ── Stage definitions ─────────────────────────────────────────────────
    # 6 stages evenly spaced across x=[1 … 13], y=5.5 (main row)
    stages = [
        dict(x=1.1,  label='RGB-D\nCamera',
             sub='Depth-enabled\nperception',
             col='#065f46', icon='[📷]'),
        dict(x=3.5,  label='Open-Vocabulary\nDetection',
             sub='Bounding boxes +\ntext query',
             col='#064e3b',  icon=''),
        dict(x=5.9,  label='3D\nLocalization',
             sub='Pixel→world via\nintrinsics + depth',
             col='#065f46', icon=''),
        dict(x=8.3,  label='Grasp\nPlanning',
             sub='Top-down approach\ntrajectory',
             col='#14532d', icon=''),
        dict(x=10.7, label='IK\nSolver',
             sub='Joint angle\ncomputation',
             col='#166534', icon=''),
        dict(x=13.1, label='Robot Arm\nExecution',
             sub='Real-time joint\ncontrol',
             col='#15803d', icon=''),
    ]

    BW_S = 1.85   # stage box width
    BH_S = 0.80   # stage box height (taller for 2-line labels)
    MAIN_Y = 5.8

    # Draw boxes and arrows
    for i, s in enumerate(stages):
        cx = s['x']
        # Gradient-like colour brightening for later stages
        col = s['col']
        rounded_box(ax, cx, MAIN_Y, BW_S, BH_S,
                    s['label'], col,
                    fontsize=10, bold=True,
                    border_color='#22c55e', lw=1.8)

        # Stage number badge
        badge = plt.Circle((cx - BW_S/2 + 0.18, MAIN_Y + BH_S/2 - 0.18),
                            0.165, color='#22c55e', zorder=6)
        ax.add_patch(badge)
        ax.text(cx - BW_S/2 + 0.18, MAIN_Y + BH_S/2 - 0.18, str(i+1),
                ha='center', va='center', fontsize=8,
                color='black', fontweight='bold', zorder=7)

        # Sub-note below box
        ax.text(cx, MAIN_Y - BH_S/2 - 0.28, s['sub'],
                ha='center', va='top', fontsize=8,
                color='#86efac', style='italic', zorder=4)

        # Arrows between stages
        if i < len(stages) - 1:
            x_right = cx + BW_S/2
            x_left  = stages[i+1]['x'] - BW_S/2
            arrow(ax, x_right, MAIN_Y, x_left, MAIN_Y,
                  color='#4ade80', lw=2.2)

    # ─── Detail callout boxes ─────────────────────────────────────────────
    # Box 1: detection example
    det_bg = FancyBboxPatch((2.55, 6.85), 2.0, 0.90,
                            boxstyle='round,pad=0.05',
                            facecolor='#052e16', edgecolor='#22c55e',
                            linewidth=1.3, alpha=0.92, zorder=3)
    ax.add_patch(det_bg)
    ax.text(3.55, 7.52, 'Query: "red block"', ha='center', fontsize=8,
            color='#bbf7d0', fontweight='bold', zorder=5)
    # Simulate bounding box drawing
    det_rect = FancyBboxPatch((2.70, 6.93), 0.65, 0.40,
                              boxstyle='round,pad=0.02',
                              facecolor='#7f1d1d', edgecolor='#ef4444',
                              linewidth=1.5, zorder=4)
    ax.add_patch(det_rect)
    ax.text(3.03, 7.13, 'bbox', ha='center', va='center',
            fontsize=7, color='#fca5a5', zorder=5)
    ax.plot([3.5, 3.55, 4.45, 4.50], [6.93, 6.93, 6.93, 6.93],
            color='#22c55e', lw=0.5, ls=':', alpha=0)
    # confidence score label
    ax.text(4.0, 7.05, '0.94', ha='center', va='center',
            fontsize=8, color='#4ade80', zorder=5)
    ax.annotate('', xy=(3.5, 7.10), xytext=(3.5, 6.42),
                arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.2),
                zorder=5)

    # Box 2: 3D projection formula
    proj_bg = FancyBboxPatch((4.90, 6.85), 2.1, 0.90,
                             boxstyle='round,pad=0.05',
                             facecolor='#052e16', edgecolor='#22c55e',
                             linewidth=1.3, alpha=0.92, zorder=3)
    ax.add_patch(proj_bg)
    ax.text(5.95, 7.58, '3D Projection', ha='center', fontsize=8,
            color='#bbf7d0', fontweight='bold', zorder=5)
    ax.text(5.95, 7.25, r'$X = (u - c_x) \cdot d / f_x$',
            ha='center', va='center', fontsize=8.5, color='#86efac', zorder=5)
    ax.text(5.95, 6.98, r'$Y = (v - c_y) \cdot d / f_y$',
            ha='center', va='center', fontsize=8.5, color='#86efac', zorder=5)
    ax.annotate('', xy=(5.9, 7.10), xytext=(5.9, 6.42),
                arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.2),
                zorder=5)

    # Box 3: grasp trajectory
    grasp_bg = FancyBboxPatch((7.35, 6.85), 2.0, 0.90,
                              boxstyle='round,pad=0.05',
                              facecolor='#052e16', edgecolor='#22c55e',
                              linewidth=1.3, alpha=0.92, zorder=3)
    ax.add_patch(grasp_bg)
    ax.text(8.35, 7.58, 'Approach Path', ha='center', fontsize=8,
            color='#bbf7d0', fontweight='bold', zorder=5)
    # Simple trajectory sketch
    traj_x = np.linspace(7.55, 9.25, 10)
    traj_y = 7.12 + 0.25 * np.exp(-((traj_x - 9.25)**2) / 0.8)
    ax.plot(traj_x, traj_y, color='#4ade80', lw=2, zorder=5)
    ax.plot(9.25, 7.12, 'o', color='#22c55e', ms=6, zorder=6)
    ax.text(9.25, 6.95, 'target', ha='center', fontsize=7,
            color='#86efac', zorder=5)
    ax.annotate('', xy=(8.3, 7.10), xytext=(8.3, 6.42),
                arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.2),
                zorder=5)

    # ─── Data flow bus at bottom ───────────────────────────────────────────
    bus_bg = FancyBboxPatch((0.5, 3.55), 13.0, 1.25,
                            boxstyle='round,pad=0.05',
                            facecolor='#052e16', edgecolor='#166534',
                            linewidth=1.5, alpha=0.85, zorder=2)
    ax.add_patch(bus_bg)

    data_items = [
        (1.5,  'Raw\nRGB-D\nFrames',      '#4ade80'),
        (3.7,  'Object\nClass + BBox\n(u,v)',  '#86efac'),
        (5.9,  'World\nCoordinates\n(X,Y,Z)',  '#4ade80'),
        (8.1,  'End-Effector\nTrajectory\nPlan', '#86efac'),
        (10.3, 'Joint\nAngles\nθ₁..θ₇',   '#4ade80'),
        (12.5, 'Motor\nTorque\nCommands',   '#86efac'),
    ]
    for x, lbl, col in data_items:
        ax.text(x, 4.17, lbl, ha='center', va='center',
                fontsize=7.5, color=col, zorder=4)
        # downward arrow from main boxes to data bus
        arrow(ax, x, MAIN_Y - BH_S/2 - 0.65, x, 4.78,
              color=col, lw=1.0)

    ax.text(7.0, 3.70, 'DATA FLOW',
            ha='center', va='center', fontsize=7.5,
            color='#166534', fontweight='bold', zorder=5)

    # ─── Annotation: no demonstrations ────────────────────────────────────
    ax.annotate('No demonstrations required\nfor new objects',
                xy=(13.1, MAIN_Y + BH_S/2),
                xytext=(11.6, 8.40),
                fontsize=9, color='#fde047', fontweight='bold',
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#fde047', lw=1.8,
                                connectionstyle='arc3,rad=0.3'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1c1917',
                          edgecolor='#fde047', linewidth=1.5),
                zorder=8)

    # ─── Right-side legend ────────────────────────────────────────────────
    leg_bg = FancyBboxPatch((0.2, 0.40), 13.6, 0.75,
                            boxstyle='round,pad=0.05',
                            facecolor='#052e16', edgecolor='#22c55e',
                            linewidth=1.5, zorder=3)
    ax.add_patch(leg_bg)
    ax.text(7.0, 0.78,
            'Advantages: Interpretable modules  •  No training data needed  '
            '•  Generalises to new objects via language queries  '
            '•  Easy failure diagnosis',
            ha='center', va='center', fontsize=8.5,
            color='#86efac', zorder=4)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'[OK] Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Run all three
# ─────────────────────────────────────────────────────────────────────────────

BASE = '/home/aryannzzz/GRASP/grasp-research-showcase/figures/architecture'

make_act_diagram      (f'{BASE}/act_architecture.png')
make_gradcam_diagram  (f'{BASE}/gradcam_vla_pipeline.png')
make_classical_pipeline(f'{BASE}/classical_pipeline.png')

print('\nAll diagrams generated successfully.')
