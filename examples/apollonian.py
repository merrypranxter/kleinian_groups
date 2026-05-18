"""Apollonian gasket via Möbius inversions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kleinian import Mobius
from kleinian.circle_packing import apollonian_gasket, descartes_step


def apollonian_via_mobius():
    """Express Apollonian packing steps as Möbius inversions."""
    k1, k2, k3 = -1.0, 2.0, 2.0
    k4 = descartes_step(k1, k2, k3)
    print(f"Starting curvatures: {k1}, {k2}, {k3} → new curvature: {k4:.4f}")

    lhs = (k1 + k2 + k3 + k4)**2
    rhs = 2 * (k1**2 + k2**2 + k3**2 + k4**2)
    print(f"Descartes check: LHS={lhs:.6f}, RHS={rhs:.6f}, diff={abs(lhs-rhs):.2e}")

    return k1, k2, k3, k4


if __name__ == '__main__':
    print("=" * 60)
    print("Apollonian Gasket Example")
    print("=" * 60)

    k1, k2, k3, k4 = apollonian_via_mobius()

    print(f"\nGenerating Apollonian gasket with curvatures ({k1},{k2},{k3},{k4})...")
    gasket = apollonian_gasket(k1, k2, k3, k4, max_iter=4)
    print(f"Generated {len(gasket)} circles")

    print("\nPlotting Apollonian gasket...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')

    radii = [r for _, r in gasket if np.isfinite(r) and r > 0]
    if radii:
        max_r = max(radii)
        for center, radius in gasket:
            if not np.isfinite(radius) or radius <= 0 or radius > max_r * 2:
                continue
            t = 1.0 - min(radius / max_r, 1.0)
            color = cmap(t)
            circle = mpatches.Circle(
                (center.real, center.imag), radius,
                fill=False, edgecolor=color, linewidth=0.7
            )
            ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_title('Apollonian Gasket (curvatures: -1, 2, 2, 3)')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.title.set_color('white')

    plt.tight_layout()
    plt.savefig('apollonian_gasket.png', dpi=150, bbox_inches='tight',
                facecolor='black')
    print("Saved apollonian_gasket.png")
    plt.close()
    print("Done.")
