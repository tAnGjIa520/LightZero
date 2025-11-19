"""
2D loss landscape visualization.

Supports contour plots, heatmaps, and 3D surface plots.
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map for loss landscape.

    ⭐ NEW: Auto-detects and plots multiple metrics if available.

    Args:
        surf_file: Path to HDF5 surface file
        surf_name: Name of surface to plot (default: 'train_loss')
                  If 'auto', will detect and plot all available metrics
        vmin, vmax: Value range for contour levels
        vlevel: Spacing between contour levels
        show: Whether to display plot
    """
    print('-' * 60)
    print('Plotting 2D contour')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    # ⭐ NEW: Auto-detect multiple metrics
    if surf_name == 'auto':
        # Find all train_loss_* keys
        metric_keys = [k for k in f.keys() if k.startswith('train_loss_')]
        if not metric_keys:
            # Fall back to standard 'train_loss' key
            metric_keys = ['train_loss'] if 'train_loss' in f.keys() else []
    else:
        metric_keys = [surf_name]

    if not metric_keys:
        raise KeyError(f"No metrics found in {surf_file}")

    print(f"Loading: {surf_file}")
    print(f"X range: {len(x)}, Y range: {len(y)}")
    print(f"Found {len(metric_keys)} metric(s): {metric_keys}")

    if len(x) <= 1 or len(y) <= 1:
        print("Insufficient coordinates for plotting contours")
        f.close()
        return

    # ⭐ NEW: Plot each metric
    for key in metric_keys:
        if key not in f.keys():
            print(f"Warning: '{key}' not found, skipping")
            continue

        Z = np.array(f[key][:])
        print(f"\nPlotting {key}: min={np.min(Z):.4f}, max={np.max(Z):.4f}")

        # Extract metric name (remove 'train_loss_' prefix if present)
        metric_label = key.replace('train_loss_', '')

        # Plot contour lines
        fig = plt.figure(figsize=(12, 10))
        CS = plt.contour(X, Y, Z, cmap='summer',
                         levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        plt.xlabel('X Direction', fontsize='x-large')
        plt.ylabel('Y Direction', fontsize='x-large')
        plt.title(f'{metric_label} - Contour Plot', fontsize='x-large')
        save_path = f"{surf_file}_{key}_2dcontour.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")

        # Plot filled contours
        fig = plt.figure(figsize=(12, 10))
        CS = plt.contourf(X, Y, Z, cmap='summer',
                          levels=np.arange(vmin, vmax, vlevel))
        plt.colorbar(CS, label=metric_label)
        plt.xlabel('X Direction', fontsize='x-large')
        plt.ylabel('Y Direction', fontsize='x-large')
        plt.title(f'{metric_label} - Filled Contour', fontsize='x-large')
        save_path = f"{surf_file}_{key}_2dcontourf.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")

        # Plot heatmap
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                    xticklabels=False, yticklabels=False)
        plt.title(f'{metric_label} - Heatmap', fontsize='x-large')
        save_path = f"{surf_file}_{key}_2dheat.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")

    f.close()
    if show:
        plt.show()


def plot_2d_surface(surf_file, surf_name='train_loss', show=False):
    """Plot 3D surface for loss landscape.

    ⭐ NEW: Auto-detects and plots multiple metrics if available.

    Args:
        surf_file: Path to HDF5 surface file
        surf_name: Name of surface to plot (default: 'train_loss')
                  If 'auto', will detect and plot all available metrics
        show: Whether to display plot
    """
    print('-' * 60)
    print('Plotting 3D surface')
    print('-' * 60)

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    # ⭐ NEW: Auto-detect multiple metrics
    if surf_name == 'auto':
        # Find all train_loss_* keys
        metric_keys = [k for k in f.keys() if k.startswith('train_loss_')]
        if not metric_keys:
            # Fall back to standard 'train_loss' key
            metric_keys = ['train_loss'] if 'train_loss' in f.keys() else []
    else:
        metric_keys = [surf_name]

    if not metric_keys:
        raise KeyError(f"No metrics found in {surf_file}")

    print(f"Loading: {surf_file}")
    print(f"X range: {len(x)}, Y range: {len(y)}")
    print(f"Found {len(metric_keys)} metric(s): {metric_keys}")

    # ⭐ NEW: Plot each metric
    for key in metric_keys:
        if key not in f.keys():
            print(f"Warning: '{key}' not found, skipping")
            continue

        Z = np.array(f[key][:])
        print(f"\nPlotting {key}: min={np.min(Z):.4f}, max={np.max(Z):.4f}")

        # Extract metric name (remove 'train_loss_' prefix if present)
        metric_label = key.replace('train_loss_', '')

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, shrink=0.5, aspect=5, label=metric_label)

        ax.set_xlabel('X Direction', fontsize='x-large')
        ax.set_ylabel('Y Direction', fontsize='x-large')
        ax.set_zlabel(metric_label, fontsize='x-large')
        ax.set_title(f'{metric_label} - 3D Surface', fontsize='x-large')

        save_path = f"{surf_file}_{key}_3dsurface.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  Saved: {save_path}")

    f.close()
    if show:
        plt.show()


def plot_2d(losses, x_coords, y_coords, surf_name='loss', show=False):
    """Simple 2D plot from numpy arrays (without HDF5).

    Args:
        losses: 2D numpy array of loss values
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates
        surf_name: Label for the surface
        show: Whether to display plot
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    # Contour plot
    fig = plt.figure(figsize=(10, 8))
    CS = plt.contour(X, Y, losses, cmap='summer', levels=15)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.colorbar(CS, label=surf_name)
    plt.xlabel('X Direction', fontsize='x-large')
    plt.ylabel('Y Direction', fontsize='x-large')
    plt.title(f'{surf_name} - Contour', fontsize='x-large')

    if show:
        plt.show()

    # 3D surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, losses, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=surf_name)
    ax.set_xlabel('X', fontsize='x-large')
    ax.set_ylabel('Y', fontsize='x-large')
    ax.set_zlabel(surf_name, fontsize='x-large')
    ax.set_title(f'{surf_name} - 3D Surface', fontsize='x-large')

    if show:
        plt.show()

    return fig
