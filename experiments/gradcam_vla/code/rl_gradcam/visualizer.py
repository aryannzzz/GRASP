"""
Visualization tools for RL-GradCAM

Creates interpretable visualizations showing:
1. Per-action attention heatmaps
2. Action comparison panels
3. Decision explanations
4. Q-value overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2
from typing import Dict, List, Optional, Tuple, Any
import io


class RLGradCAMVisualizer:
    """
    Comprehensive visualization for RL-GradCAM outputs.
    
    Creates publication-quality figures showing:
    - Per-action attention maps
    - Side-by-side action comparisons
    - Decision explanation panels
    - Q-value annotations
    """
    
    # Color scheme for consistent visualizations
    COLORS = {
        'chosen': '#2ecc71',      # Green for chosen action
        'best': '#3498db',        # Blue for best Q-value
        'highlight': '#e74c3c',   # Red for highlighting
        'neutral': '#95a5a6',     # Gray for neutral
    }
    
    def __init__(
        self,
        action_names: List[str] = ['Left', 'Down', 'Right', 'Up'],
        figsize_scale: float = 1.0
    ):
        self.action_names = action_names
        self.figsize_scale = figsize_scale
        
    def plot_single_action_cam(
        self,
        observation: np.ndarray,
        cam: np.ndarray,
        action_name: str,
        q_value: Optional[float] = None,
        is_chosen: bool = False,
        is_best: bool = False,
        ax: Optional[plt.Axes] = None,
        show_colorbar: bool = True
    ) -> plt.Axes:
        """
        Plot GradCAM for a single action.
        
        Args:
            observation: (C, H, W) or (H, W, C) visual observation
            cam: (h, w) GradCAM heatmap
            action_name: Name of the action
            q_value: Q-value for this action
            is_chosen: Whether this action was taken
            is_best: Whether this has highest Q-value
            ax: Matplotlib axes (creates new if None)
            show_colorbar: Add colorbar
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4 * self.figsize_scale, 4 * self.figsize_scale))
        
        # Convert observation to displayable format
        if observation.ndim == 3 and observation.shape[0] == 3:
            obs_display = observation.transpose(1, 2, 0)
        else:
            obs_display = observation
        
        if obs_display.max() <= 1.0:
            obs_display = (obs_display * 255).astype(np.uint8)
        
        # Resize CAM to match observation
        cam_resized = cv2.resize(cam, (obs_display.shape[1], obs_display.shape[0]))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(obs_display, 0.6, heatmap, 0.4, 0)
        
        # Display
        im = ax.imshow(overlay)
        
        # Title with Q-value
        title = action_name
        if q_value is not None:
            title += f'\nQ={q_value:.3f}'
        
        # Color-code title based on status
        title_color = 'black'
        if is_chosen:
            title_color = self.COLORS['chosen']
            title += ' ✓'
        elif is_best:
            title_color = self.COLORS['best']
            title += ' ★'
        
        ax.set_title(title, fontsize=12 * self.figsize_scale, color=title_color, fontweight='bold')
        ax.axis('off')
        
        # Add border for chosen/best action
        if is_chosen or is_best:
            color = self.COLORS['chosen'] if is_chosen else self.COLORS['best']
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
        
        return ax
    
    def plot_all_actions_comparison(
        self,
        observation: np.ndarray,
        cams: Dict[str, np.ndarray],
        q_values: np.ndarray,
        chosen_action: Optional[int] = None,
        title: str = "Action Attention Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comparison panel showing CAM for all actions.
        
        This is the key visualization for understanding action selection:
        see at a glance what visual features influence each possible action.
        """
        n_actions = len(self.action_names)
        
        # Determine grid layout
        n_cols = min(4, n_actions)
        n_rows = (n_actions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols * self.figsize_scale, 4 * n_rows * self.figsize_scale)
        )
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Find best action
        best_action = np.argmax(q_values)
        
        # Plot each action
        for i, action_name in enumerate(self.action_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            cam = cams.get(action_name, np.zeros_like(list(cams.values())[0]))
            
            self.plot_single_action_cam(
                observation=observation,
                cam=cam,
                action_name=action_name,
                q_value=q_values[i],
                is_chosen=(chosen_action == i),
                is_best=(i == best_action),
                ax=ax,
                show_colorbar=False
            )
        
        # Hide unused subplots
        for i in range(n_actions, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        # Add overall title
        fig.suptitle(title, fontsize=14 * self.figsize_scale, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_action_contrast(
        self,
        observation: np.ndarray,
        contrast_cam: np.ndarray,
        action_a: str,
        action_b: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize contrast between two actions.
        
        Shows what visual features differentiate action A from action B.
        Positive (red) = favors A, Negative (blue) = favors B
        """
        fig, axes = plt.subplots(1, 3, figsize=(12 * self.figsize_scale, 4 * self.figsize_scale))
        
        # Convert observation
        if observation.ndim == 3 and observation.shape[0] == 3:
            obs_display = observation.transpose(1, 2, 0)
        else:
            obs_display = observation
        
        if obs_display.max() <= 1.0:
            obs_display = (obs_display * 255).astype(np.uint8)
        
        # Original observation
        axes[0].imshow(obs_display)
        axes[0].set_title('Observation', fontsize=12)
        axes[0].axis('off')
        
        # Contrast map
        contrast_resized = cv2.resize(contrast_cam, (obs_display.shape[1], obs_display.shape[0]))
        
        # Use diverging colormap for contrast
        im = axes[1].imshow(contrast_resized, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title(f'Contrast: {action_a} vs {action_b}', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, label=f'← {action_b} | {action_a} →')
        
        # Overlay
        # Convert contrast to color (red = positive, blue = negative)
        contrast_rgb = np.zeros((*contrast_resized.shape, 3), dtype=np.uint8)
        contrast_rgb[contrast_resized > 0, 0] = (contrast_resized[contrast_resized > 0] * 255).astype(np.uint8)
        contrast_rgb[contrast_resized < 0, 2] = (-contrast_resized[contrast_resized < 0] * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(obs_display, 0.6, contrast_rgb, 0.4, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Contrast Overlay', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_decision_explanation(
        self,
        observation: np.ndarray,
        cams: Dict[str, np.ndarray],
        q_values: np.ndarray,
        chosen_action: int,
        state_info: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive decision explanation panel.
        
        Shows:
        - Original observation with grid info
        - Attention for chosen action
        - Q-values bar chart
        - Why chosen action (highest Q highlighted)
        """
        fig = plt.figure(figsize=(16 * self.figsize_scale, 8 * self.figsize_scale))
        gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1])
        
        # Convert observation
        if observation.ndim == 3 and observation.shape[0] == 3:
            obs_display = observation.transpose(1, 2, 0)
        else:
            obs_display = observation
        
        if obs_display.max() <= 1.0:
            obs_display = (obs_display * 255).astype(np.uint8)
        
        # 1. Original observation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(obs_display)
        ax1.set_title('Observation', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Chosen action attention
        ax2 = fig.add_subplot(gs[0, 1])
        chosen_name = self.action_names[chosen_action]
        chosen_cam = cams[chosen_name]
        
        cam_resized = cv2.resize(chosen_cam, (obs_display.shape[1], obs_display.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(obs_display, 0.5, heatmap, 0.5, 0)
        
        ax2.imshow(overlay)
        ax2.set_title(f'Attention: {chosen_name}', fontsize=12, fontweight='bold', color='green')
        ax2.axis('off')
        
        # 3. Best action attention (if different from chosen)
        ax3 = fig.add_subplot(gs[0, 2])
        best_action = np.argmax(q_values)
        best_name = self.action_names[best_action]
        best_cam = cams[best_name]
        
        cam_resized = cv2.resize(best_cam, (obs_display.shape[1], obs_display.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(obs_display, 0.5, heatmap, 0.5, 0)
        
        ax3.imshow(overlay)
        title_color = 'green' if best_action == chosen_action else 'blue'
        ax3.set_title(f'Best Q: {best_name}', fontsize=12, fontweight='bold', color=title_color)
        ax3.axis('off')
        
        # 4. Q-values bar chart
        ax4 = fig.add_subplot(gs[0, 3])
        colors = ['green' if i == chosen_action else ('blue' if i == best_action else 'gray') 
                  for i in range(len(self.action_names))]
        bars = ax4.bar(self.action_names, q_values, color=colors)
        ax4.set_ylabel('Q-value')
        ax4.set_title('Action Q-values', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Annotate bars
        for bar, qv in zip(bars, q_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{qv:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 5-8. All action CAMs (bottom row)
        for i, action_name in enumerate(self.action_names):
            ax = fig.add_subplot(gs[1, i])
            cam = cams[action_name]
            
            cam_resized = cv2.resize(cam, (obs_display.shape[1], obs_display.shape[0]))
            heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(obs_display, 0.6, heatmap, 0.4, 0)
            
            ax.imshow(overlay)
            
            # Mark chosen and best
            suffix = ''
            color = 'black'
            if i == chosen_action:
                suffix = ' ✓'
                color = 'green'
            if i == best_action:
                suffix += ' ★'
                if i != chosen_action:
                    color = 'blue'
            
            ax.set_title(f'{action_name}{suffix}\nQ={q_values[i]:.3f}', fontsize=10, color=color)
            ax.axis('off')
        
        plt.suptitle('Decision Explanation', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def figure_to_array(self, fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array (useful for video creation)."""
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


# Quick test
if __name__ == "__main__":
    print("Testing RLGradCAMVisualizer...")
    
    # Create dummy data
    obs = np.random.rand(3, 64, 64).astype(np.float32)
    cams = {
        'Left': np.random.rand(8, 8).astype(np.float32),
        'Down': np.random.rand(8, 8).astype(np.float32),
        'Right': np.random.rand(8, 8).astype(np.float32),
        'Up': np.random.rand(8, 8).astype(np.float32),
    }
    q_values = np.array([0.1, 0.8, 0.3, 0.2])
    
    # Create visualizer
    viz = RLGradCAMVisualizer()
    
    # Test comparison plot
    fig = viz.plot_all_actions_comparison(
        observation=obs,
        cams=cams,
        q_values=q_values,
        chosen_action=1,
        title="Test Comparison"
    )
    plt.close(fig)
    print("✓ All actions comparison plot works")
    
    # Test decision explanation
    fig = viz.plot_decision_explanation(
        observation=obs,
        cams=cams,
        q_values=q_values,
        chosen_action=1
    )
    plt.close(fig)
    print("✓ Decision explanation plot works")
    
    print("\n✓ RLGradCAMVisualizer test passed!")
