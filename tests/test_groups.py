"""Tests for KleinianGroup."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kleinian.mobius import Mobius
from kleinian.groups import KleinianGroup


def make_schottky():
    t = 0.8
    A = Mobius(np.cosh(t), np.sinh(t), np.sinh(t), np.cosh(t))
    phi = Mobius.rotation(np.pi / 2)
    B = A.conjugate_by(phi)
    return KleinianGroup([A, B], names=['a', 'b'])


def test_word_to_mobius_cancellation():
    """'aA' should give identity (a followed by a-inverse)."""
    group = make_schottky()
    aA = group.word_to_mobius('aA')
    I = Mobius.identity()
    assert aA == I, f"aA should be identity, got {aA}"

    bB = group.word_to_mobius('bB')
    assert bB == I


def test_word_to_mobius_composition():
    """word_to_mobius should correctly compose generators."""
    group = make_schottky()
    A = group.generators[0]
    B = group.generators[1]

    ab = group.word_to_mobius('ab')
    manual = A * B
    assert ab == manual


def test_no_cancellation_in_enumerate():
    """enumerate_words should not yield words with consecutive cancellations."""
    group = make_schottky()
    for word, _ in group.enumerate_words(max_depth=4):
        for i in range(len(word) - 1):
            pair = word[i:i+2]
            assert pair not in ('aA', 'Aa', 'bB', 'Bb'), \
                f"Cancellation found in word '{word}': '{pair}'"


def test_orbit_size_grows_with_depth():
    """Orbit size should increase with depth."""
    group = make_schottky()
    z0 = 0.1 + 0.1j

    orbit3 = group.orbit(z0, max_depth=3)
    orbit5 = group.orbit(z0, max_depth=5)

    assert len(orbit5) >= len(orbit3), "Deeper orbit should have at least as many points"
    assert len(orbit3) > 0, "Orbit should be non-empty"


def test_jorgensen_number_discrete():
    """A known discrete group should have non-negative Jørgensen number."""
    group = make_schottky()
    A = group.generators[0]
    B = group.generators[1]

    j_num = group.jorgensen_number(A, B)
    assert j_num >= 0, "Jørgensen number should be non-negative"


def test_identity_word():
    """Empty word should give identity."""
    group = make_schottky()
    words = list(group.enumerate_words(max_depth=3))
    first_word, first_transform = words[0]
    assert first_word == ""
    I = Mobius.identity()
    assert first_transform == I
