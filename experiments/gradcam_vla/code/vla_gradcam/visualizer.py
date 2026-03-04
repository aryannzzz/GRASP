"""
VLA-GradCAM Visualizer: Publication-quality figures and video generation.

Creates visualizations showing:
  - Per-action-dimension saliency overlays
  - Instruction comparison panels
  - Action contrast maps
  - Animated videos of saliency across instructions/action dims
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict, Tuple, Sequence
from pathlib import Path
from dataclasses import dataclass

from .gradcam_engine import VLAGradCAMResult


def overlay_heatmap(
    image: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay saliency heatmap on image. Returns [H,W,3] uint8 RGB."""
    img = image.copy()
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Ensure saliency is [0, 1]
    sal = saliency.copy().astype(np.float32)
    if sal.max() > sal.min():
        sal = (sal - sal.min()) / (sal.max() - sal.min())

    # Resize saliency to image dims
    sal_resized = cv2.resize(sal, (img.shape[1], img.shape[0]))

    # Create colored heatmap
    heatmap = cv2.applyColorMap((sal_resized * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    out = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return out


def figure_to_array(fig: plt.Figure, width: int = 1500, height: int = 500) -> np.ndarray:
    """Convert matplotlib figure to numpy RGB array with fixed dimensions."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.3)
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to fixed dimensions for consistent video frames
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


class VLAGradCAMVisualizer:
    """Visualization toolkit for VLA-GradCAM results."""

    def __init__(self, figsize_scale: float = 1.0):
        self.scale = figsize_scale

    # ------------------------------------------------------------------
    # Static image visualizations
    # ------------------------------------------------------------------

    def plot_per_action_saliency(
        self,
        result: VLAGradCAMResult,
        save_path: Optional[str] = None,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """
        Grid showing saliency for each action dimension.
        First cell is the original image, remaining are overlays.
        """
        maps = result.saliency_maps
        n = len(maps) + 1  # +1 for original
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols * self.scale, 4 * nrows * self.scale),
        )
        axes = np.array(axes).flatten()

        # Original image
        axes[0].imshow(result.image)
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis('off')

        for i, (name, sal) in enumerate(maps.items()):
            ax = axes[i + 1]
            overlay = overlay_heatmap(result.image, sal, alpha=alpha)
            ax.imshow(overlay)
            dim_idx = result.action_names.index(name) if name in result.action_names else i
            val = result.predicted_action[dim_idx] if dim_idx < len(result.predicted_action) else 0
            ax.set_title(f"{name}\na={val:.3f}", fontsize=10)
            ax.axis('off')

        for j in range(n, len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f'VLA-GradCAM: "{result.instruction}"\nPer-Action-Dimension Saliency',
            fontsize=13 * self.scale,
            fontweight='bold',
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_instruction_comparison(
        self,
        results: List[VLAGradCAMResult],
        action_dim_name: str = 'delta_x',
        save_path: Optional[str] = None,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """
        Compare saliency across different instructions on the same image.
        Shows how language changes visual attention.
        """
        n = len(results) + 1
        fig, axes = plt.subplots(1, n, figsize=(4 * n * self.scale, 4 * self.scale))

        # Original image
        axes[0].imshow(results[0].image)
        axes[0].set_title("Scene", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        for i, res in enumerate(results):
            ax = axes[i + 1]
            sal = res.saliency_maps.get(action_dim_name, res.combined_saliency)
            overlay = overlay_heatmap(res.image, sal, alpha=alpha)
            ax.imshow(overlay)
            ax.set_title(f'"{res.instruction}"', fontsize=10)
            ax.axis('off')

        fig.suptitle(
            f"Language-Conditioned Attention ({action_dim_name})\n"
            "Different instructions -> Different visual focus",
            fontsize=12 * self.scale,
            fontweight='bold',
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_combined_overview(
        self,
        result: VLAGradCAMResult,
        save_path: Optional[str] = None,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """
        Combined overview: original + combined saliency + action bar chart.
        """
        fig = plt.figure(figsize=(16 * self.scale, 5 * self.scale))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

        # Original
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(result.image)
        ax0.set_title("Input Image", fontsize=12, fontweight='bold')
        ax0.axis('off')

        # Combined saliency
        ax1 = fig.add_subplot(gs[1])
        overlay = overlay_heatmap(result.image, result.combined_saliency, alpha=alpha)
        ax1.imshow(overlay)
        ax1.set_title("Combined Saliency", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Action values bar chart
        ax2 = fig.add_subplot(gs[2])
        n_dims = len(result.predicted_action)
        names = result.action_names[:n_dims]
        vals = result.predicted_action[:n_dims]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_dims))
        bars = ax2.barh(names, vals, color=colors)
        ax2.set_xlabel("Action Value")
        ax2.set_title("Predicted Action", fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(
                v + 0.02 if v >= 0 else v - 0.06,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                va='center',
                fontsize=9,
            )

        fig.suptitle(
            f'VLA-GradCAM Overview | Instruction: "{result.instruction}"',
            fontsize=13 * self.scale,
            fontweight='bold',
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_action_contrast(
        self,
        image: np.ndarray,
        contrast: np.ndarray,
        label_a: str,
        label_b: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize action contrast map (diverging colormap)."""
        fig, axes = plt.subplots(1, 3, figsize=(14 * self.scale, 4.5 * self.scale))

        img = image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        axes[0].imshow(img)
        axes[0].set_title("Image", fontsize=12)
        axes[0].axis('off')

        contrast_resized = cv2.resize(contrast, (img.shape[1], img.shape[0]))
        im = axes[1].imshow(contrast_resized, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title(f"Contrast: {label_a} vs {label_b}", fontsize=11)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        pos_mask = np.clip(contrast_resized, 0, 1)
        neg_mask = np.clip(-contrast_resized, 0, 1)
        overlay_rgb = np.zeros_like(img, dtype=np.float32)
        overlay_rgb[:, :, 0] = pos_mask * 255  # Red = favors A
        overlay_rgb[:, :, 2] = neg_mask * 255   # Blue = favors B
        overlay = cv2.addWeighted(img, 0.6, overlay_rgb.astype(np.uint8), 0.4, 0)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (Red=A, Blue=B)", fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def make_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 2,
        codec: str = "mp4v",
    ):
        """
        Write a list of RGB numpy frames to an MP4 video.

        Args:
            frames: list of [H, W, 3] uint8 arrays (all same size)
            output_path: path ending in .mp4
            fps: frames per second
            codec: fourcc codec string
        """
        if not frames:
            return
        # Normalize all frames to the same size as the first frame
        h, w = frames[0].shape[:2]
        normalized = []
        for f in frames:
            if f.shape[0] != h or f.shape[1] != w:
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            normalized.append(f)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (w, h),
        )
        for frame in normalized:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()
        print(f"Video saved: {output_path} ({len(normalized)} frames, {fps} fps)")

    def generate_instruction_video(
        self,
        results: List[VLAGradCAMResult],
        output_path: str,
        action_dim_name: str = "delta_x",
        fps: int = 1,
        hold_frames: int = 3,
        alpha: float = 0.5,
    ):
        """
        Generate video cycling through different instructions.

        Each instruction is shown for `hold_frames` frames.
        Produces a compelling visualization of language-conditioned attention.
        """
        frames = []
        for res in results:
            sal = res.saliency_maps.get(action_dim_name, res.combined_saliency)
            # Build a combined frame: original | saliency overlay | info
            fig = self._make_video_frame(res, sal, alpha=alpha)
            arr = figure_to_array(fig)
            plt.close(fig)
            for _ in range(hold_frames):
                frames.append(arr)
        self.make_video(frames, output_path, fps=fps)

    def generate_action_dim_video(
        self,
        result: VLAGradCAMResult,
        output_path: str,
        fps: int = 1,
        hold_frames: int = 3,
        alpha: float = 0.5,
    ):
        """
        Generate video cycling through action dimensions for one instruction.
        Shows how different action components attend to different regions.
        """
        frames = []
        for dim_name, sal in result.saliency_maps.items():
            fig = self._make_video_frame(result, sal, dim_name=dim_name, alpha=alpha)
            arr = figure_to_array(fig)
            plt.close(fig)
            for _ in range(hold_frames):
                frames.append(arr)
        self.make_video(frames, output_path, fps=fps)

    def generate_full_demo_video(
        self,
        results_by_instruction: Dict[str, VLAGradCAMResult],
        output_path: str,
        fps: int = 2,
        hold_frames: int = 4,
        alpha: float = 0.5,
    ):
        """
        Full demo video:
          1. Show original scene
          2. For each instruction: cycle through action dims
          3. Show instruction comparison
        """
        frames = []
        all_results = list(results_by_instruction.values())
        first = all_results[0]

        # Section 1: Original scene
        fig = self._make_title_frame(first.image, "VLA-GradCAM Demo")
        arr = figure_to_array(fig)
        plt.close(fig)
        for _ in range(hold_frames * 2):
            frames.append(arr)

        # Section 2: Per-instruction saliency
        for instruction, result in results_by_instruction.items():
            # Title for this instruction
            fig = self._make_title_frame(
                result.image,
                f'Instruction: "{instruction}"',
            )
            arr = figure_to_array(fig)
            plt.close(fig)
            for _ in range(hold_frames):
                frames.append(arr)

            # Combined saliency
            fig = self._make_video_frame(
                result, result.combined_saliency, dim_name="combined", alpha=alpha,
            )
            arr = figure_to_array(fig)
            plt.close(fig)
            for _ in range(hold_frames):
                frames.append(arr)

            # Per-action-dim saliency
            for dim_name, sal in result.saliency_maps.items():
                fig = self._make_video_frame(result, sal, dim_name=dim_name, alpha=alpha)
                arr = figure_to_array(fig)
                plt.close(fig)
                for _ in range(hold_frames):
                    frames.append(arr)

        # Section 3: Instruction comparison
        fig = self._make_comparison_frame(all_results, alpha=alpha)
        arr = figure_to_array(fig)
        plt.close(fig)
        for _ in range(hold_frames * 3):
            frames.append(arr)

        self.make_video(frames, output_path, fps=fps)

    # ------------------------------------------------------------------
    # Internal helpers for video frames
    # ------------------------------------------------------------------

    def _make_video_frame(
        self,
        result: VLAGradCAMResult,
        saliency: np.ndarray,
        dim_name: Optional[str] = None,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """Create a single video frame: image | overlay | action info."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        img = result.image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Original
        axes[0].imshow(img)
        axes[0].set_title("Input Image", fontsize=12)
        axes[0].axis('off')

        # Overlay
        overlay = overlay_heatmap(img, saliency, alpha=alpha)
        axes[1].imshow(overlay)
        title = dim_name if dim_name else "saliency"
        axes[1].set_title(f"Saliency: {title}", fontsize=12)
        axes[1].axis('off')

        # Action info
        n_dims = min(len(result.action_names), len(result.predicted_action))
        names = result.action_names[:n_dims]
        vals = result.predicted_action[:n_dims]
        colors = ['#e74c3c' if n == dim_name else '#3498db' for n in names]
        axes[2].barh(names, vals, color=colors)
        axes[2].set_xlabel("Action Value")
        axes[2].set_title("Predicted Action", fontsize=12)
        axes[2].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

        fig.suptitle(
            f'VLA-GradCAM | "{result.instruction}"',
            fontsize=13,
            fontweight='bold',
        )
        plt.tight_layout()
        return fig

    def _make_title_frame(
        self,
        image: np.ndarray,
        title: str,
    ) -> plt.Figure:
        """Create a title frame with centered image and text."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        img = image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')
        plt.tight_layout()
        return fig

    def _make_comparison_frame(
        self,
        results: List[VLAGradCAMResult],
        alpha: float = 0.5,
    ) -> plt.Figure:
        """Create comparison frame showing all instructions side by side."""
        n = len(results) + 1
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

        img = results[0].image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        axes[0].imshow(img)
        axes[0].set_title("Scene", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        for i, res in enumerate(results):
            overlay = overlay_heatmap(img, res.combined_saliency, alpha=alpha)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f'"{res.instruction}"', fontsize=10)
            axes[i + 1].axis('off')

        fig.suptitle(
            "Instruction Comparison: Same image, different instructions\n"
            "-> Different visual attention patterns",
            fontsize=13, fontweight='bold',
        )
        plt.tight_layout()
        return fig
