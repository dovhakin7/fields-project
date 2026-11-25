"""Milestone 2 acceptance tests — force and torque on the tip magnet."""

import numpy as np
import pytest

from emlab import steering as st
from emlab.geometry import TipMagnet, Catheter


def test_moment_magnitude_is_13_milli_amp_metre_squared():
    m = st.moment_vector(TipMagnet())
    assert np.linalg.norm(m) == pytest.approx(0.0130, rel=0.02)
    assert m[0] == pytest.approx(0.0) and m[1] == pytest.approx(0.0)


def test_equivalent_loop_carries_the_same_moment():
    magnet = TipMagnet()
    _, _, current = st.equivalent_loop(magnet)
    assert current * np.pi * magnet.radius ** 2 == pytest.approx(
        magnet.moment_magnitude, rel=1e-9)


def test_net_force_vanishes_in_a_uniform_field():
    mid, vec, current = st.equivalent_loop(TipMagnet(), tilt=0.6)
    B = np.array([0.0, 0.0, 0.08])
    F = st.force_on_loop(mid, vec, current, B)
    scale = current * np.abs(vec).max() * np.linalg.norm(B)
    assert np.linalg.norm(F) < 1e-9 * max(scale, 1e-12)


def test_torque_integral_reproduces_m_cross_b():
    magnet, tilt = TipMagnet(), 0.5
    mid, vec, current = st.equivalent_loop(magnet, tilt=tilt, n_seg=720)
    B = np.array([0.0, 0.0, 0.08])
    T_loop = st.torque_on_loop(mid, vec, current, B)
    m = st.moment_vector(magnet, direction=(np.sin(tilt), 0.0, np.cos(tilt)))
    assert np.allclose(T_loop, np.cross(m, B), rtol=3e-3, atol=1e-12)


def test_torque_vanishes_when_the_magnet_is_aligned():
    mid, vec, current = st.equivalent_loop(TipMagnet(), tilt=0.0, n_seg=360)
    T = st.torque_on_loop(mid, vec, current, np.array([0.0, 0.0, 0.08]))
    assert np.linalg.norm(T) < 1e-9


def test_tip_undershoots_a_60_degree_command():
    cat = Catheter()
    reached = np.degrees(st.equilibrium_tilt(np.radians(60.0), 0.0796, cat))
    assert reached == pytest.approx(30.0, abs=2.0)
    angles = [st.equilibrium_tilt(np.radians(60.0), b, cat)
              for b in (0.02, 0.04, 0.08, 0.16)]
    assert np.all(np.diff(angles) > 0) and angles[-1] < np.radians(60.0)
    assert st.angle_error(np.radians(60.0), 0.08, cat) == pytest.approx(
        np.radians(60.0) - angles[2], rel=1e-6)
