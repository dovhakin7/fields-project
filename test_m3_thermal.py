"""Milestone 3 acceptance tests — coil resistance, dissipation and force."""

import numpy as np
import pytest

from emlab import thermal as th
from emlab.geometry import CoilPair


def test_ulaby_exercise_4_14():
    assert th.wire_resistance(50.0, np.pi * 0.02 ** 2) == pytest.approx(6.9e-4, rel=0.02)


def test_historical_ampere_definition():
    assert th.force_per_length_parallel_wires(1.0, 1.0, 1.0) == pytest.approx(
        2e-7, rel=1e-6)


def test_dissipation_uses_the_per_turn_current():
    coil = CoilPair()
    R = th.coil_resistance(coil)
    P = th.coil_dissipation(coil)
    assert P == pytest.approx(coil.current ** 2 * R, rel=1e-9)
    assert P == pytest.approx(1.45e3, rel=0.05)
    assert P < 1e-3 * coil.ampere_turns ** 2 * R


def test_dissipation_scales_as_current_squared():
    assert (th.coil_dissipation(CoilPair(current=200.0))
            / th.coil_dissipation(CoilPair(current=100.0))) == pytest.approx(4.0, rel=1e-9)


def test_force_between_the_coils():
    F = th.force_between_coils(CoilPair(), n_seg=720)
    assert F == pytest.approx(690.0, rel=0.05)
    assert F > 0.0


def test_force_scales_and_temperature_rise_is_reported():
    f1 = th.force_between_coils(CoilPair(current=100.0), n_seg=360)
    f2 = th.force_between_coils(CoilPair(current=200.0), n_seg=360)
    assert f2 / f1 == pytest.approx(4.0, rel=1e-6)
    coil = CoilPair()
    assert th.current_density(coil) == pytest.approx(coil.current / coil.wire_area)
    assert th.temperature_rise(coil) > 100.0
