"""Milestone 4 acceptance tests — the finite-volume solve in tissue."""

import numpy as np
import pytest

from emlab import conduction as cd
from emlab.geometry import AblationSettings


def _coarse(**kw):
    kw.setdefault("cells_per_tip_radius", 6)
    return AblationSettings(**kw)


def test_harmonic_mean_of_face_conductivity():
    assert cd.harmonic_mean(2.0, 2.0) == pytest.approx(2.0)
    assert cd.harmonic_mean(1.0, 3.0) == pytest.approx(1.5)
    assert cd.harmonic_mean(0.54, 0.70) < 0.5 * (0.54 + 0.70)


def test_sphere_in_a_single_medium():
    s = _coarse(depth=0.0, sigma_myocardium=0.70)
    out = cd.solve(s)
    assert out["impedance"] == pytest.approx(
        cd.analytic_resistance(s, straddling=False), rel=0.20)


def test_straddling_sphere_matches_the_closed_form():
    s = _coarse(depth=2.0e-3)
    out = cd.solve(s)
    closed = cd.analytic_resistance(s, straddling=True)
    assert out["impedance"] == pytest.approx(closed, rel=0.20)
    assert out["impedance"] > closed


def test_power_splits_in_proportion_to_conductivity():
    s = _coarse(depth=2.0e-3)
    out = cd.solve(s)
    expected = s.sigma_myocardium / (s.sigma_myocardium + s.sigma_blood)
    assert out["fraction_myocardium"] == pytest.approx(expected, rel=0.01)


def test_tangential_field_is_continuous_across_the_interface():
    s = _coarse(depth=2.0e-3)
    grid = cd.build_grid(s)
    sigma = cd.conductivity_map(grid, s)
    v = cd.solve_potential(grid, sigma, s)
    j = int(np.argmin(np.abs(grid["z"])))
    far = grid["r"] > 3.0 * s.electrode_radius
    et_above = np.gradient(v[:, j], grid["r"])[far]
    et_below = np.gradient(v[:, j - 1], grid["r"])[far]
    assert np.allclose(et_above, et_below, rtol=0.02, atol=1e-3)


def test_staircase_error_is_first_order():
    closed = cd.analytic_resistance(AblationSettings(depth=2.0e-3), straddling=True)
    errors = []
    for n in (6, 12):
        out = cd.solve(AblationSettings(depth=2.0e-3, cells_per_tip_radius=n))
        errors.append(abs(out["impedance"] / closed - 1.0))
    assert errors[0] > errors[1]
    assert 1.6 < errors[0] / errors[1] < 2.6


def test_impedance_rises_as_the_tip_seats():
    shallow = cd.solve(_coarse(depth=0.0))
    deep = cd.solve(_coarse(depth=4.0e-3))
    assert deep["impedance"] > shallow["impedance"]
    assert deep["fraction_myocardium"] > shallow["fraction_myocardium"] + 0.3
    assert deep["lesion_volume"] > 1.5 * shallow["lesion_volume"]
