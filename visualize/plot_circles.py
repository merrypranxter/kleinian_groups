import numpy as np
from typing import Optional, List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_apollonian(
    gasket: list,
    ax=None,
    max_radius: Optional[float] = None,
    color_map: str = 'viridis'
) -> None:
    """Plot Apollonian gasket circles."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    import matplotlib.cm as cm
    cmap = cm.get_cmap(color_map)

    radii = [r for _, r in gasket if np.isfinite(r)]
    if not radii:
        return
    max_r = max(radii)

    for center, radius in gasket:
        if not np.isfinite(radius) or radius <= 0:
            continue
        if max_radius is not None and radius > max_radius:
            continue
        t = 1.0 - radius / max_r
        color = cmap(t)
        circle = patches.Circle(
            (center.real, center.imag), radius,
            fill=False, edgecolor=color, linewidth=0.5
        )
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale()


def plot_ford(ford_circles_list: list, ax=None) -> None:
    """Plot Ford circles."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    import matplotlib.patches as patches

    for center, radius in ford_circles_list:
        circle = patches.Circle(
            (center.real, center.imag), radius,
            fill=False, edgecolor='blue', linewidth=0.5, alpha=0.7
        )
        ax.add_patch(circle)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 0.6)
    ax.set_aspect('equal')
    ax.set_title('Ford Circles')


def plot_isometric_circles(
    circles: list,
    ax=None,
    color_map: str = 'plasma'
) -> None:
    """Plot isometric circles with depth-based coloring."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    import matplotlib.cm as cm
    import matplotlib.patches as patches
    cmap = cm.get_cmap(color_map)

    if not circles:
        return

    max_depth = max(c.get('depth', 1) for c in circles)

    for c in circles:
        center = c['center']
        radius = c['radius']
        depth = c.get('depth', 1)
        t = (depth - 1) / max(max_depth - 1, 1)
        color = cmap(t)
        circle = patches.Circle(
            (center.real, center.imag), radius,
            fill=False, edgecolor=color, linewidth=max(0.3, 1.5 - 0.3 * depth)
        )
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_title('Isometric Circles')
