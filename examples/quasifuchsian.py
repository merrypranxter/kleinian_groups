"""Quasifuchsian deformation of a Fuchsian group."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kleinian import Mobius, KleinianGroup, compute_limit_set


def genus2_fuchsian():
    """Construct generators for a deformable Fuchsian-like group."""
    lam = 2.0
    A = Mobius(lam, 0, 0, 1 / lam)

    phi = Mobius.rotation(2 * np.pi / 3)
    B = A.conjugate_by(phi)

    phi2 = Mobius.rotation(4 * np.pi / 3)
    C = A.conjugate_by(phi2)

    D = (A * B * C).inverse()

    return [A, B, C, D]


def bend_fuchsian(generators: list, bend_angle: float) -> list:
    """Apply a bending deformation to a Fuchsian group."""
    bent = []
    twist = np.exp(1j * bend_angle)
    for g in generators:
        t14 = twist ** 0.25
        conj = Mobius(t14, 0, 0, 1 / t14)
        bent.append(g.conjugate_by(conj))
    return bent


if __name__ == '__main__':
    print("=" * 60)
    print("Quasifuchsian Group Example")
    print("=" * 60)

    gens_fuchsian = genus2_fuchsian()
    print(f"\nFuchsian group: {len(gens_fuchsian)} generators")
    for i, g in enumerate(gens_fuchsian):
        print(f"  g{i}: trace={g.trace():.4f}")

    group_fuchsian = KleinianGroup(gens_fuchsian[:2], names=['a', 'b'])

    print("\nComputing Fuchsian limit set (depth=6)...")
    pts_fuchsian = compute_limit_set(group_fuchsian, max_depth=6, tol=1e-5)
    print(f"Fuchsian limit set: {len(pts_fuchsian)} points")
    if len(pts_fuchsian) > 0:
        print(f"  Im(z) range: [{np.imag(pts_fuchsian).min():.6f}, {np.imag(pts_fuchsian).max():.6f}]")

    bend_angles = [0.0, 0.15, 0.4, 0.8]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, angle in zip(axes, bend_angles):
        bent_gens = bend_fuchsian(gens_fuchsian[:2], angle)
        group = KleinianGroup(bent_gens, names=['a', 'b'])
        pts = compute_limit_set(group, max_depth=6, tol=1e-5)

        if len(pts) > 0:
            ax.scatter(np.real(pts), np.imag(pts), s=1, c='cyan', alpha=0.6)

        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(y=0, color='red', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='red', linewidth=0.5, alpha=0.5)
        ax.set_title(f'Bend angle θ = {angle:.2f} rad\n({len(pts)} points)')
        ax.set_facecolor('black')

        if angle == 0.0:
            ax.set_title(f'Fuchsian (θ=0): limit set on ℝ\n({len(pts)} points)')

    fig.suptitle('Quasifuchsian Deformation: Fuchsian → Quasifuchsian',
                  fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('quasifuchsian_deformation.png', dpi=150, bbox_inches='tight',
                facecolor='black')
    print("\nSaved quasifuchsian_deformation.png")
    plt.close()
    print("Done.")
