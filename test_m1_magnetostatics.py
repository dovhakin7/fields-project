"""Milestone 1 acceptance tests — the navigation field."""

import numpy as np
import pytest

from emlab import constants as C
from emlab import magnetostatics as ms
from emlab.geometry import CoilPair


def test_loop_centre_equals_mu0_i_over_2a():
    a, I = 0.25, 10.0
    mid, vec = ms.loop_segments(a, 0.0, 600)
    B = ms.biot_savart([[0.0, 0.0, 0.0]], mid, vec, I)[0]
    assert B[2] == pytest.approx(C.MU0 * I / (2 * a), rel=1e-4)


def test_on_axis_matches_closed_form():
    a, I = 0.25, 10.0
    mid, vec = ms.loop_segments(a, 0.0, 800)
    z = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    numeric = ms.biot_savart(pts, mid, vec, I)[:, 2]
    assert np.allclose(numeric, ms.on_axis_loop_bz(z, a, I), rtol=2e-4)


def test_transverse_components_vanish_on_axis():
    mid, vec = ms.loop_segments(0.3, 0.0, 720)
    z = np.linspace(-0.2, 0.2, 9)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    B = ms.biot_savart(pts, mid, vec, 5.0)
    assert np.max(np.abs(B[:, :2])) < 1e-9 * np.max(np.abs(B[:, 2]))


def test_helmholtz_centre_value():
    coil = CoilPair()
    B = ms.coil_pair_field([[0.0, 0.0, 0.0]], coil, n_seg=720)[0]
    expected = (4.0 / 5.0) ** 1.5 * C.MU0 * coil.ampere_turns / coil.radius
    assert np.linalg.norm(B) == pytest.approx(expected, rel=1e-3)
    assert abs(B[0]) < 1e-9 and abs(B[1]) < 1e-9


def test_uniformity_over_working_volume():
    coil = CoilPair()
    z = np.linspace(-0.05, 0.05, 41)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    mag = np.linalg.norm(ms.coil_pair_field(pts, coil, n_seg=600), axis=1)
    centre = mag[len(mag) // 2]
    assert 1e-5 < ms.uniformity(mag, centre) < 1e-3
    assert ms.working_volume_points(0.05, 5).shape == (125, 3)


def test_segment_count_converges():
    a, I = 0.25, 10.0
    target = ms.on_axis_loop_bz(0.1, a, I)
    errors = []
    for n in (24, 96, 384):
        mid, vec = ms.loop_segments(a, 0.0, n)
        B = ms.biot_savart([[0.0, 0.0, 0.1]], mid, vec, I)[0, 2]
        errors.append(abs(B / target - 1.0))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-4
