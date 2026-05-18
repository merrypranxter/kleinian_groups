"""Classic 2-generator Schottky group example."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from kleinian import Mobius, KleinianGroup, compute_limit_set, hausdorff_dim_estimate
from kleinian.discrete import jorgensen_test


def build_schottky_group():
    """Build a 2-generator Schottky group using hyperbolic generators."""
    t = 0.8
    cosh_t = np.cosh(t)
    sinh_t = np.sinh(t)

    A = Mobius(cosh_t, sinh_t, sinh_t, cosh_t)

    phi = Mobius.rotation(np.pi / 2)
    B = A.conjugate_by(phi)

    return A, B


if __name__ == '__main__':
    print("=" * 60)
    print("Schottky Group Example")
    print("=" * 60)

    A, B = build_schottky_group()
    print(f"\nGenerator A: {A}")
    print(f"Generator B: {B}")

    group = KleinianGroup([A, B], names=['a', 'b'])
    result = jorgensen_test(A, B)
    print(f"\nJørgensen test: {result['message']}")

    print(f"\nTrace(A) = {A.trace():.6f}, trace²(A) = {A.trace()**2:.6f}")
    print(f"Type of A: ", end="")
    if A.is_hyperbolic(): print("hyperbolic")
    elif A.is_parabolic(): print("parabolic")
    elif A.is_elliptic(): print("elliptic")
    else: print("loxodromic")

    print(f"Type of B: ", end="")
    if B.is_hyperbolic(): print("hyperbolic")
    elif B.is_parabolic(): print("parabolic")
    elif B.is_elliptic(): print("elliptic")
    else: print("loxodromic")

    print("\nComputing limit set (depth=8)...")
    pts = compute_limit_set(group, max_depth=8, tol=1e-6)
    print(f"Number of limit set points: {len(pts)}")

    if len(pts) > 0:
        print(f"Bounding box: Re ∈ [{np.real(pts).min():.4f}, {np.real(pts).max():.4f}], "
              f"Im ∈ [{np.imag(pts).min():.4f}, {np.imag(pts).max():.4f}]")

        if len(pts) >= 10:
            hdim = hausdorff_dim_estimate(pts)
            print(f"Estimated Hausdorff dimension: {hdim:.4f}")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from visualize.render import render_limit_set
        print("\nRendering limit set to schottky_limit_set.png...")
        img = render_limit_set(pts, width=800, height=800, color_by='depth',
                                output_path='schottky_limit_set.png')
        print("Saved schottky_limit_set.png")
    except Exception as e:
        print(f"Rendering failed: {e}")

    print("\nDone.")
