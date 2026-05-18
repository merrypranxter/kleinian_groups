"""Tests for limit set computation."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kleinian.mobius import Mobius
from kleinian.groups import KleinianGroup
from kleinian.limit_set import compute_limit_set, hausdorff_dim_estimate, isometric_circle_limit_set


def make_schottky():
    t = 0.8
    A = Mobius(np.cosh(t), np.sinh(t), np.sinh(t), np.cosh(t))
    phi = Mobius.rotation(np.pi / 2)
    B = A.conjugate_by(phi)
    return KleinianGroup([A, B], names=['a', 'b'])


def test_limit_set_nonempty():
    """Schottky group should produce a non-empty limit set."""
    group = make_schottky()
    pts = compute_limit_set(group, max_depth=5, tol=1e-4)
    assert len(pts) > 0, "Limit set should be non-empty for Schottky group"


def test_limit_set_complex_array():
    """Limit set should be a complex128 numpy array."""
    group = make_schottky()
    pts = compute_limit_set(group, max_depth=4, tol=1e-4)
    assert isinstance(pts, np.ndarray)
    assert pts.dtype == np.complex128


def test_hausdorff_in_range():
    """Hausdorff dimension estimate should be in (0, 2)."""
    group = make_schottky()
    pts = compute_limit_set(group, max_depth=6, tol=1e-6)
    if len(pts) >= 10:
        hdim = hausdorff_dim_estimate(pts)
        assert 0 < hdim < 2, f"Hausdorff dim {hdim} out of range (0, 2)"


def test_isometric_circles_nonempty():
    """Schottky group should produce isometric circles."""
    group = make_schottky()
    circles = isometric_circle_limit_set(group, max_depth=3)
    assert len(circles) > 0, "Should have isometric circles"
    for c in circles[:5]:
        assert 'center' in c
        assert 'radius' in c
        assert 'depth' in c
        assert 'word' in c
        assert c['radius'] > 0


def test_hausdorff_trivial_set():
    """Function should not crash on near-trivial inputs."""
    pts = np.array([1+0j, 1+0j, 1+0j])
    hdim = hausdorff_dim_estimate(pts)
    assert hdim >= 0
