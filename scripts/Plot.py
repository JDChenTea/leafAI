import os
import tempfile
from pathlib import Path

# Some sandboxed macOS/Linux environments have an unwritable home cache.
_matplotlib_cache = Path(tempfile.gettempdir()) / "leafai-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def hsv_pair_plot(x, y, z, xlabel='h', ylabel='s', zlabel='v',
                  path='hsv_distribution.png'):
    data = pd.DataFrame({xlabel: np.asarray(x).reshape(-1),
                         ylabel: np.asarray(y).reshape(-1),
                         zlabel: np.asarray(z).reshape(-1)})
    if data.empty:
        raise ValueError("cannot plot an empty HSV data set")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Histograms are more robust than KDE for constant/small image samples.
    fig = sns.pairplot(data=data, vars=[xlabel, ylabel, zlabel],
                       diag_kind='hist', markers='+')
    fig.savefig(target, dpi=300, bbox_inches='tight')
    plt.close(fig.fig)


def hsv_surface_data(h, s, v, h_bins=36, s_bins=32):
    """Aggregate HSV pixels into an H/S grid whose height is mean V."""
    h = np.asarray(h, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    if not (h.size and h.size == s.size == v.size):
        raise ValueError("H, S and V must be non-empty arrays of equal length")
    counts, h_edges, s_edges = np.histogram2d(
        h, s, bins=(h_bins, s_bins), range=((0, 180), (0, 256))
    )
    value_sums, _, _ = np.histogram2d(
        h, s, bins=(h_bins, s_bins), range=((0, 180), (0, 256)), weights=v
    )
    mean_v = np.full(counts.shape, np.nan, dtype=float)
    np.divide(value_sums, counts, out=mean_v, where=counts > 0)
    h_centers = (h_edges[:-1] + h_edges[1:]) / 2
    s_centers = (s_edges[:-1] + s_edges[1:]) / 2
    return np.meshgrid(h_centers, s_centers, indexing="ij"), mean_v, counts


def hsv_surface_plot(h, s, v, path, h_bins=36, s_bins=32):
    """Plot mean V as a three-dimensional surface over the H/S plane."""
    (h_grid, s_grid), mean_v, counts = hsv_surface_data(
        h, s, v, h_bins=h_bins, s_bins=s_bins
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        h_grid, s_grid, np.ma.masked_invalid(mean_v), cmap="viridis",
        linewidth=0.25, edgecolor="none", antialiased=True,
    )
    occupied = counts > 0
    axis.scatter(h_grid[occupied], s_grid[occupied], mean_v[occupied],
                 c=mean_v[occupied], cmap="viridis", s=7, alpha=0.45)
    axis.set(xlabel="H (0–179)", ylabel="S (0–255)", zlabel="Mean V (0–255)",
             xlim=(0, 180), ylim=(0, 256), zlim=(0, 255),
             title="HSV surface: mean V over H–S bins")
    axis.view_init(elev=28, azim=-135)
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.1, label="Mean V")
    figure.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(figure)

def histogram(data, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    ax = sns.histplot(data, kde=True)
    fig = ax.get_figure()
    fig.savefig(target, dpi=300, bbox_inches='tight')
    plt.close(fig)

def linear_regression(data, path, x, y):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig = sns.jointplot(x=x, y=y, data=data,
                  kind="reg", truncate=False,
                  color="m", height=7)
    fig.savefig(target, dpi=300, bbox_inches='tight')
    plt.close(fig.fig)

