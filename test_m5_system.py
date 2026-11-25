"""Milestone 5 acceptance tests — the four links chained together."""

import numpy as np

from emlab import conduction as cd
from emlab import steering as st
from emlab.geometry import AblationSettings, Catheter, seating_depth


def chain(commanded_deg, b_magnitude, cells=6):
    commanded = np.radians(commanded_deg)
    reached = st.equilibrium_tilt(commanded, b_magnitude, Catheter())
    error = commanded - reached
    depth = float(seating_depth(error))
    out = cd.solve(AblationSettings(depth=depth, cells_per_tip_radius=cells))
    return {"commanded": commanded, "reached": reached, "error": error,
            "depth": depth, "impedance": out["impedance"],
            "fraction_myocardium": out["fraction_myocardium"],
            "lesion_volume": out["lesion_volume"]}


def test_chain_runs_end_to_end():
    r = chain(45.0, 0.080)
    for key in ("reached", "error", "depth", "impedance", "lesion_volume"):
        assert key in r and np.isfinite(r[key])
    assert 0.0 <= r["depth"] <= 4.0e-3
    assert r["reached"] < r["commanded"]


def test_more_field_means_less_angle_error():
    errors = [chain(60.0, b)["error"] for b in (0.040, 0.080, 0.160)]
    assert errors[0] > errors[1] > errors[2]


def test_deeper_seating_grows_the_lesion():
    shallow = cd.solve(AblationSettings(depth=0.5e-3, cells_per_tip_radius=6))
    deep = cd.solve(AblationSettings(depth=3.5e-3, cells_per_tip_radius=6))
    assert deep["lesion_volume"] > shallow["lesion_volume"]
    assert deep["fraction_myocardium"] > shallow["fraction_myocardium"]


def test_field_buys_lesion_at_the_steepest_command():
    weak, strong = chain(75.0, 0.040), chain(75.0, 0.160)
    assert strong["depth"] > weak["depth"]
    assert strong["lesion_volume"] > weak["lesion_volume"]
