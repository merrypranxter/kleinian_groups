import numpy as np
from typing import TYPE_CHECKING
from .mobius import Mobius

if TYPE_CHECKING:
    from .groups import KleinianGroup


def jorgensen_test(A: Mobius, B: Mobius) -> dict:
    """Jørgensen inequality test for a 2-generator group."""
    commutator = A * B * A.inverse() * B.inverse()
    value = abs(A.trace() ** 2 - 4) + abs(commutator.trace() - 2)
    passes = value >= 1.0 - 1e-10
    if passes:
        msg = f"Jørgensen test PASSED (value={value:.6f} ≥ 1): group may be discrete."
    else:
        msg = f"Jørgensen test FAILED (value={value:.6f} < 1): group is NOT discrete."
    return {'passes': passes, 'value': float(value), 'message': msg}


def is_elementary(group) -> bool:
    """True if limit set has ≤ 2 points (elementary group)."""
    from .limit_set import compute_limit_set
    pts = compute_limit_set(group, max_depth=5, tol=1e-4)
    if len(pts) == 0:
        return True
    if len(pts) <= 2:
        return True
    from scipy.spatial.distance import cdist
    coords = np.column_stack([np.real(pts), np.imag(pts)])
    sample = coords[:min(50, len(coords))]
    dists = cdist(sample, sample)
    max_dist = dists.max()
    return max_dist < 1e-3


def fundamental_domain_circles(group) -> list:
    """Returns list of isometric circles forming the Ford/Schottky fundamental domain."""
    from .limit_set import isometric_circle_limit_set
    return isometric_circle_limit_set(group, max_depth=1)


def ping_pong_test(
    A: Mobius, B: Mobius,
    D_A: tuple, D_B: tuple
) -> bool:
    """Ping-pong lemma test.
    D_A = (center_a, radius_a): disk preserved by A and A^-1
    D_B = (center_b, radius_b): disk preserved by B and B^-1
    Returns True if A^±1(D_B) ⊂ D_A and B^±1(D_A) ⊂ D_B.
    """
    center_a, r_a = D_A
    center_b, r_b = D_B

    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    test_pts_B = center_b + r_b * np.exp(1j * angles)

    A_inv = A.inverse()
    B_inv = B.inverse()

    for z in test_pts_B:
        Az = A(z)
        Ainvz = A_inv(z)
        if not (abs(Az - center_a) < r_a + 1e-10):
            return False
        if not (abs(Ainvz - center_a) < r_a + 1e-10):
            return False

    test_pts_A = center_a + r_a * np.exp(1j * angles)
    for z in test_pts_A:
        Bz = B(z)
        Binvz = B_inv(z)
        if not (abs(Bz - center_b) < r_b + 1e-10):
            return False
        if not (abs(Binvz - center_b) < r_b + 1e-10):
            return False

    return True
