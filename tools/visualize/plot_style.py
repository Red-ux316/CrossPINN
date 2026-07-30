"""
Unified Plotting & Styling Module for PINN Subsystem.
Provides standard layout, theme parameters, model color palettes, and helper functions
to ensure consistent, high-quality publication-ready plots across evaluate, visualize,
plot_comparison, and analyze_study scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Model Palette ---
MODEL_COLORS = {
    'MLP': '#1f77b4',        # Steel Blue
    'mlp': '#1f77b4',
    'RISN': '#ff7f0e',       # Vibrant Orange
    'risn': '#ff7f0e',
    'TLP': '#2ca02c',        # Forest Green
    'tlp': '#2ca02c',
    'TransPINN': '#d62728',   # Crimson Red
    'transpinn': '#d62728',
    'Exact': '#000000',       # Pure Black (dashed)
    'exact': '#000000',
}

DEFAULT_STYLE = {
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
}

def apply_global_plot_style():
    """Configures global Matplotlib rcParams for publication-quality plots."""
    plt.rcParams.update(DEFAULT_STYLE)

def create_figure(figsize=(9, 5), is_2d=False, **kwargs):
    """
    Creates and returns a new figure and axes configured with standard PINN layout settings.
    """
    apply_global_plot_style()
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax

def format_plot(ax, title=None, xlabel='x', ylabel='u(x)', yscale=None, xscale=None, grid=True, legend=True):
    """
    Applies standard titles, labels, scaling, grid, and legend formatting to an axes instance.
    """
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    if xscale:
        ax.set_xscale(xscale)
    if grid:
        if yscale == 'log' or xscale == 'log':
            ax.grid(True, which="both", alpha=0.3, linestyle='--')
        else:
            ax.grid(True, alpha=0.3, linestyle='--')
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, framealpha=0.9)

def save_and_close_figure(fig, path, dpi=300):
    """
    Applies tight_layout, saves the figure to the specified path, and closes the figure to free memory.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_obj, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
