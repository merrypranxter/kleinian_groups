"""Tests for Möbius transformations."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kleinian.mobius import Mobius


def test_identity():
    I = Mobius.identity()
    z = 1.5 + 2.3j
    assert abs(I(z) - z) < 1e-12

    A = Mobius(1 + 1j, 2, -0.5j, 3)
    assert (A * I) == A
    assert (I * A) == A


def test_inverse():
    A = Mobius(2, 1, 1, 1)
    A_inv = A.inverse()
    composed = A * A_inv
    I = Mobius.identity()
    assert composed == I, f"A * A^-1 should be identity, got {composed}"

    composed2 = A_inv * A
    assert composed2 == I


def test_composition():
    A = Mobius(2, 1j, -1j, 3)
    B = Mobius(1, 1, 0, 1)

    z = 0.5 + 0.5j
    direct = (A * B)(z)
    composed = A(B(z))
    assert abs(direct - composed) < 1e-10, f"Composition failed: {direct} vs {composed}"


def test_fixed_points():
    H = Mobius(2, 0, 0, 1)
    fps = H.fixed_points()
    for fp in fps:
        if np.isfinite(abs(fp)):
            assert abs(H(fp) - fp) < 1e-10, f"Fixed point {fp} not fixed"

    P = Mobius.parabolic(1)
    fps_p = P.fixed_points()
    assert len(fps_p) >= 1


def test_isometric_circle():
    """Points on the isometric circle satisfy |cz+d| = 1 (for det=1)."""
    A = Mobius(2, 1, 1, 1)
    center, radius = A.isometric_circle()

    thetas = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    for theta in thetas:
        z = center + radius * np.exp(1j * theta)
        val = abs(A.c * z + A.d)
        assert abs(val - 1.0) < 1e-10, f"Not on isometric circle: |cz+d|={val}"


def test_parabolic_trace():
    P = Mobius.parabolic(2.0 + 1j)
    t2 = P.trace() ** 2
    assert abs(t2 - 4) < 1e-10, f"Parabolic should have trace²=4, got {t2}"


def test_projective_equality():
    A = Mobius(1, 2, 3, 4)
    neg_A = Mobius(-1, -2, -3, -4)
    assert A == neg_A, "M and -M should be projectively equal"

    other = Mobius(1, 2, 3, 5)
    assert A != other


def test_mobius_types():
    H = Mobius.hyperbolic(2.0)
    assert H.is_hyperbolic()
    assert not H.is_parabolic()
    assert not H.is_elliptic()

    P = Mobius.parabolic(1.0)
    assert P.is_parabolic()
    assert not P.is_hyperbolic()

    E = Mobius.rotation(np.pi / 3)
    assert E.is_elliptic()


def test_call_infinity():
    """Test evaluation at infinity and poles."""
    A = Mobius(2, 1, 1, 0)
    result = A(complex(np.inf))
    assert abs(result - 2.0) < 1e-10 or np.isinf(abs(result))


def test_conjugate_by():
    A = Mobius(2, 0, 0, 0.5)
    phi = Mobius.rotation(np.pi / 4)
    B = A.conjugate_by(phi)

    z = 1.0 + 0.5j
    phi_inv = phi.inverse()
    manual = phi(A(phi_inv(z)))
    auto = B(z)
    assert abs(manual - auto) < 1e-9
