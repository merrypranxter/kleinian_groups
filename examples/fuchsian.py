"""PSL(2,Z) modular group / Farey tessellation example."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kleinian import Mobius, KleinianGroup
from kleinian.circle_packing import ford_circles


def build_modular_group():
    """PSL(2,Z) with generators S: z->-1/z and T: z->z+1."""
    S = Mobius(0, -1, 1, 0)
    T = Mobius(1, 1, 0, 1)
    return S, T


if __name__ == '__main__':
    print("=" * 60)
    print("Fuchsian Group (PSL(2,Z)) Example")
    print("=" * 60)

    S, T = build_modular_group()
    print(f"\nS (z -> -1/z): {S}")
    print(f"T (z -> z+1):  {T}")

    S2 = S * S
    ST = S * T
    ST3 = ST * ST * ST
    I = Mobius.identity()
    print(f"\nS² = I? {S2 == I}")
    print(f"(ST)³ = I? {ST3 == I}")

    fps = S.fixed_points()
    print(f"\nFixed points of S: {fps}")
    print(f"S(i) = {S(1j):.6f} (should be i)")

    group = KleinianGroup([S, T], names=['s', 't'])
    print("\nEnumerating words up to depth 6...")
    words = list(group.enumerate_words(max_depth=6))
    print(f"Total words: {len(words)}")

    orbit_i = group.orbit(1j, max_depth=4)
    print(f"\nOrbit of i (depth 4): {len(orbit_i)} points")

    print("\nGenerating Ford circles (max_q=15)...")
    fc = ford_circles(max_q=15)
    print(f"Generated {len(fc)} Ford circles")

    print("\nPlotting Ford circles / Farey tessellation...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    import matplotlib.patches as mpatches
    for center, radius in fc:
        circle = mpatches.Circle(
            (center.real, center.imag), radius,
            fill=True, facecolor=(0.2, 0.4, 0.8, 0.3),
            edgecolor='blue', linewidth=0.5
        )
        ax1.add_patch(circle)
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.02, 0.35)
    ax1.set_aspect('equal')
    ax1.set_title('Ford Circles (Farey Tessellation)')
    ax1.set_xlabel('Re(z)')

    ax2 = axes[1]
    orbit_re = np.real(orbit_i)
    orbit_im = np.imag(orbit_i)

    ax2.scatter(orbit_re, orbit_im, s=10, c='blue', alpha=0.6, zorder=5)
    ax2.axhline(y=0, color='black', linewidth=1)

    theta = np.linspace(0, np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='|z|=1')
    ax2.axvline(x=-0.5, color='g', linestyle='--', alpha=0.5, label='Re(z)=-1/2')
    ax2.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='Re(z)=1/2')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(0, 3)
    ax2.set_aspect('equal')
    ax2.set_title(f'Orbit of i under PSL(2,Z) (depth 4, {len(orbit_i)} points)')
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('fuchsian_modular.png', dpi=150, bbox_inches='tight')
    print("Saved fuchsian_modular.png")
    plt.close()
    print("Done.")
