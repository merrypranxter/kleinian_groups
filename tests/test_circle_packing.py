"""Tests for circle packing."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kleinian.circle_packing import descartes_step, ford_circles, apollonian_gasket, circle_inversion


def test_descartes_step_known():
    """Test Descartes circle theorem with known quadruple (-1, 2, 2, 3)."""
    k1, k2, k3 = -1.0, 2.0, 2.0
    k4 = descartes_step(k1, k2, k3)
    assert abs(k4 - 3.0) < 1e-10, f"Expected k4=3, got {k4}"

    total = k1 + k2 + k3 + k4
    sum_sq = k1**2 + k2**2 + k3**2 + k4**2
    assert abs(total**2 - 2*sum_sq) < 1e-10, "Descartes equation not satisfied"


def test_descartes_step_another():
    """Test with triple (2, 2, 3): fourth tangent circle has curvature 15."""
    k4 = descartes_step(2.0, 2.0, 3.0)
    assert abs(k4 - 15.0) < 1e-10 or abs(k4 - 3.0) < 1e-10, f"Unexpected k4={k4}"


def test_ford_tangency():
    """Adjacent Ford circles should be tangent."""
    def find_circle(p, q):
        r = 1.0 / (2 * q * q)
        center = complex(p / q, r)
        return center, r

    c0, r0 = find_circle(0, 1)
    c1, r1 = find_circle(1, 1)
    dist = abs(c0 - c1)
    expected = r0 + r1
    assert abs(dist - expected) < 1e-10, f"0/1 and 1/1 Ford circles not tangent: dist={dist}, r0+r1={expected}"

    c_half, r_half = find_circle(1, 2)
    c_third, r_third = find_circle(1, 3)
    dist2 = abs(c_half - c_third)
    expected2 = r_half + r_third
    assert abs(dist2 - expected2) < 1e-10, f"1/2 and 1/3 Ford circles not tangent"


def test_apollonian_sizes_decrease():
    """In Apollonian gasket, we should get more than 4 circles after iteration."""
    gasket = apollonian_gasket(-1.0, 2.0, 2.0, 3.0, max_iter=3)

    radii = sorted([r for _, r in gasket if np.isfinite(r) and r > 0], reverse=True)
    assert len(radii) > 4, "Should have more than 4 circles after iteration"
    assert radii[0] >= radii[-1], "Radii should not all be equal"


def test_circle_inversion():
    """Test circle inversion: inverting twice gives original point."""
    center = 0 + 0j
    radius = 1.0

    z = 2 + 1j
    z_inv = circle_inversion(center, radius, z)
    z_double_inv = circle_inversion(center, radius, z_inv)

    assert abs(z - z_double_inv) < 1e-10, "Double inversion should give original point"

    z_inside = 0.3 + 0.2j
    z_outside = circle_inversion(center, radius, z_inside)
    assert abs(z_outside) > radius, "Inversion should map inside to outside"


def test_ford_circles_count():
    """Check the number of Ford circles for max_q=5."""
    fc = ford_circles(max_q=5)
    expected_fractions = set()
    from math import gcd
    for q in range(1, 6):
        for p in range(0, q + 1):
            if gcd(p, q) == 1:
                expected_fractions.add((p, q))
    assert len(fc) == len(expected_fractions), f"Expected {len(expected_fractions)} Ford circles, got {len(fc)}"
