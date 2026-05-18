import numpy as np
from math import gcd
from typing import Optional
from .groups import KleinianGroup


def descartes_step(c1: float, c2: float, c3: float, c4_sign: float = 1.0) -> float:
    """Descartes circle theorem: given three mutually tangent circles with curvatures
    c1, c2, c3, return the curvature of the fourth tangent circle."""
    s = c1 + c2 + c3
    p = c1 * c2 + c2 * c3 + c1 * c3
    disc = max(p, 0.0)
    return s + c4_sign * 2.0 * np.sqrt(disc)


def _apollonian_center(k1: float, k2: float, k3: float, k4: float,
                        z1: complex, z2: complex, z3: complex):
    """Compute center of fourth Apollonian circle using complex Descartes theorem."""
    numerator_base = k1 * z1 + k2 * z2 + k3 * z3
    under_root = k1 * k2 * z1 * z2 + k2 * k3 * z2 * z3 + k1 * k3 * z1 * z3
    sqrt_val = np.sqrt(complex(under_root))
    z4a = (numerator_base + 2 * sqrt_val) / k4
    z4b = (numerator_base - 2 * sqrt_val) / k4
    return z4a, z4b


def apollonian_gasket(
    k1: float, k2: float, k3: float, k4: float,
    max_iter: int = 5
) -> list:
    """Iterative Apollonian packing. Initial 4 circles must be mutually tangent.
    Returns list of (center, radius) pairs."""
    from collections import deque

    def compute_initial_centers(ka, kb, kc, kd):
        ra = 1 / abs(ka) if ka != 0 else np.inf
        rb = 1 / abs(kb) if kb != 0 else np.inf
        rc = 1 / abs(kc) if kc != 0 else np.inf
        za = 0 + 0j
        if ka < 0:
            zb = complex(ra - rb, 0)
            dist_ac = ra - rc
            dist_bc = rb + rc
            x_b = ra - rb
            if abs(x_b) < 1e-12:
                x = 0.0
            else:
                x = (dist_ac ** 2 - dist_bc ** 2 + x_b ** 2) / (2 * x_b)
            y_sq = dist_ac ** 2 - x ** 2
            y = np.sqrt(max(y_sq, 0))
            zc = complex(x, y)
        else:
            zb = complex(ra + rb, 0)
            dist_ac = ra + rc
            dist_bc = rb + rc
            x_b = ra + rb
            if abs(x_b) < 1e-12:
                x = 0.0
            else:
                x = (dist_ac ** 2 - dist_bc ** 2 + x_b ** 2) / (2 * x_b)
            y_sq = dist_ac ** 2 - x ** 2
            y = np.sqrt(max(y_sq, 0))
            zc = complex(x, y)
        return za, zb, zc

    za, zb, zc = compute_initial_centers(k1, k2, k3, k4)
    z4a, z4b = _apollonian_center(k1, k2, k3, k4, za, zb, zc)
    zd = z4a

    all_circles = [(k1, za), (k2, zb), (k3, zc), (k4, zd)]

    queue = deque()
    queue.append((0, 1, 2, 3))

    visited = set()
    visited.add(frozenset([0, 1, 2, 3]))

    iter_count = 0
    while queue and iter_count < max_iter * 200:
        i, j, k_idx, l = queue.popleft()
        iter_count += 1

        for triple in [(j, k_idx, l), (i, k_idx, l), (i, j, l), (i, j, k_idx)]:
            a_idx, b_idx, c_idx = triple
            ka, za_ = all_circles[a_idx]
            kb, zb_ = all_circles[b_idx]
            kc, zc_ = all_circles[c_idx]

            other_idx = {i, j, k_idx, l} - set(triple)
            other_idx = other_idx.pop() if other_idx else None

            k_new = descartes_step(ka, kb, kc)

            if abs(k_new) > 1e6 or abs(k_new) < 1e-10:
                continue

            z_new_a, z_new_b = _apollonian_center(ka, kb, kc, k_new, za_, zb_, zc_)

            if other_idx is not None:
                _, z_other = all_circles[other_idx]
                if abs(z_new_a - z_other) > abs(z_new_b - z_other):
                    z_new = z_new_a
                else:
                    z_new = z_new_b
            else:
                z_new = z_new_a

            new_idx = len(all_circles)
            all_circles.append((k_new, z_new))

            new_quad = frozenset([a_idx, b_idx, c_idx, new_idx])
            if new_quad not in visited and len(all_circles) < max_iter * 50:
                visited.add(new_quad)
                queue.append((a_idx, b_idx, c_idx, new_idx))

    result = []
    for k_val, z_val in all_circles:
        r = 1 / abs(k_val) if abs(k_val) > 1e-10 else np.inf
        result.append((z_val, r))
    return result


def ford_circles(max_q: int = 20) -> list:
    """Ford circles for p/q with gcd(p,q)=1, 0<=p<=q<=max_q.
    center = p/q + i/(2q^2), radius = 1/(2q^2)."""
    circles = []
    for q in range(1, max_q + 1):
        for p in range(0, q + 1):
            if gcd(p, q) == 1:
                r = 1.0 / (2 * q * q)
                center = complex(p / q, r)
                circles.append((center, r))
    return circles


def circle_inversion(center: complex, radius: float, point: complex) -> complex:
    """Invert point in circle with given center and radius."""
    dz = point - center
    if abs(dz) < 1e-300:
        return complex(np.inf)
    return center + radius ** 2 / np.conj(dz)


def packing_from_schottky(group: KleinianGroup, depth: int = 4) -> list:
    """Generate circle-packing from isometric circles of a Schottky group."""
    from .limit_set import isometric_circle_limit_set
    raw_circles = isometric_circle_limit_set(group, max_depth=depth)
    return raw_circles
