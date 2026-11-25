#!/usr/bin/env python3
"""Check your work.  PROVIDED — do not edit.

    python check.py          every milestone
    python check.py 1        just Milestone 1
    python check.py 1 2      Milestones 1 and 2

For each check you get one of three results:

    [ ok ]   it works
    [ todo ] you have not written that function yet
    [FAIL]   you wrote it, but the physics is not right yet — with a hint

Work on the first [FAIL] you see, not the last.  Later checks often fail only
because an earlier function they depend on is still wrong.
"""

import contextlib
import io
import sys
import re
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:
    sys.exit("pytest is missing.  Run:  pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent


def locate(label):
    """Turn 'Milestone 2: equilibrium_tilt' into 'emlab/steering.py line 121'."""
    func = label.split(":")[-1].strip()
    for path in sorted((ROOT / "emlab").glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.startswith(f"def {func}("):
                return f"emlab/{path.name}  line {n}:  def {func}(...)"
    return label

MILESTONES = {
    "1": ("tests/test_m1_magnetostatics.py", "Milestone 1 - the navigation field"),
    "2": ("tests/test_m2_steering.py", "Milestone 2 - force and torque on the tip"),
    "3": ("tests/test_m3_thermal.py", "Milestone 3 - what the field costs"),
    "4": ("tests/test_m4_conduction.py", "Milestone 4 - current in tissue"),
    "5": ("tests/test_m5_system.py", "Milestone 5 - the whole chain"),
}

TITLES = {
    "test_loop_centre_equals_mu0_i_over_2a":
        ("field at the centre of a loop equals mu0 I / 2a",
         "a wrong SIGN means np.cross(dl, R) is the other way round; a factor of 2 "
         "means a diameter where a radius belongs; a factor of 4 pi means the "
         "mu0/4pi in front of the sum is missing"),
    "test_on_axis_matches_closed_form":
        ("field along the axis matches Eq. (5.34)",
         "your numerical sum and your closed form disagree - one of the two is "
         "wrong, and the closed form is the easier one to check by hand"),
    "test_transverse_components_vanish_on_axis":
        ("on the axis, Bx and By vanish by symmetry",
         "if these are not tiny, the loop is not closed or the segment midpoints "
         "and vectors do not line up - print seg_vec.sum(axis=0), it should be ~0"),
    "test_helmholtz_centre_value":
        ("the coil pair gives 79.6 mT at the centre",
         "did you pass coil.ampere_turns (N*I) rather than coil.current, and add "
         "BOTH coils?"),
    "test_uniformity_over_working_volume":
        ("the field is flat to about 0.05 % over +-50 mm",
         "uniformity must be a fraction, (max-min)/reference, not a percentage"),
    "test_segment_count_converges":
        ("more segments means less error",
         "if the error does not fall, the segment count is not reaching "
         "loop_segments, or you built the nodes with the wrong number of points"),
    "test_moment_magnitude_is_13_milli_amp_metre_squared":
        ("the tip magnet has a moment of 13 mA*m^2",
         "m = M*V, and magnet.moment_magnitude already computes it - normalise the "
         "direction before scaling"),
    "test_equivalent_loop_carries_the_same_moment":
        ("the equivalent loop has the same moment as the slug",
         "I_eq = M * L, using magnet.magnetisation and magnet.length"),
    "test_net_force_vanishes_in_a_uniform_field":
        ("a closed loop in a uniform field feels no net force (Eq. 5.11)",
         "sum over the segments with axis=0, and check that your loop closes"),
    "test_torque_integral_reproduces_m_cross_b":
        ("the torque integral agrees with m x B (Eq. 5.20)",
         "check the order inside both cross products: dl x B, then r x dF"),
    "test_torque_vanishes_when_the_magnet_is_aligned":
        ("no torque when the magnet already points along B",
         "with tilt = 0 the moment is parallel to B, so m x B must be zero"),
    "test_tip_undershoots_a_60_degree_command":
        ("a 60 degree command reaches about 30 degrees",
         "solve k*theta = m|B| sin(commanded - theta) with brentq between 0 and "
         "commanded - do not solve for sin(theta)"),
    "test_ulaby_exercise_4_14":
        ("50 m of 2 cm copper is 6.9e-4 ohm (Exercise 4-14)",
         "R = l / (sigma A), and A is the area, pi r^2, not the radius"),
    "test_historical_ampere_definition":
        ("1 A at 1 m gives 2e-7 N/m (Eq. 5.41)",
         "F' = mu0 I1 I2 / (2 pi d)"),
    "test_dissipation_uses_the_per_turn_current":
        ("coil dissipation is about 1.45 kW",
         "THE TRAP: P = I^2 R uses coil.current, the per-turn current, not "
         "coil.ampere_turns"),
    "test_dissipation_scales_as_current_squared":
        ("doubling the current quadruples the heat",
         "P must depend on the current squared"),
    "test_force_between_the_coils":
        ("the coils pull together with about 690 N",
         "pass ampere_turns for BOTH coils, and return the attraction as a "
         "positive number"),
    "test_force_scales_and_temperature_rise_is_reported":
        ("force follows I^2, and the temperature rise is reported",
         "same ampere-turns question as above; dT = P / (h A)"),
    "test_harmonic_mean_of_face_conductivity":
        ("the harmonic mean is right",
         "2 sa sb / (sa + sb) - not (sa + sb)/2"),
    "test_sphere_in_a_single_medium":
        ("with one conductivity, Z approaches the closed form",
         "check conductivity_map, the masks, and that the far boundary is set to 0 V"),
    "test_straddling_sphere_matches_the_closed_form":
        ("with the sphere on the interface, Z approaches the closed form",
         "the closed form is (1/a - 1/R_out) / [2 pi (sigma_m + sigma_b)]"),
    "test_power_splits_in_proportion_to_conductivity":
        ("the power splits as sigma_myo : sigma_blood",
         "if the split is 50:50 your conductivity map is uniform; if it is far off, "
         "check that power_density counts each face half to each neighbour"),
    "test_tangential_field_is_continuous_across_the_interface":
        ("tangential E carries across the interface (Section 4-8)",
         "this one breaks if you used the arithmetic mean instead of the harmonic "
         "mean for the face conductivity"),
    "test_staircase_error_is_first_order":
        ("halving the cell size halves the error",
         "if the error does not fall, cells_per_tip_radius is not reaching "
         "build_grid; if it falls much faster, your electrode mask is not a sphere"),
    "test_impedance_rises_as_the_tip_seats":
        ("pushing the tip into tissue raises the impedance",
         "myocardium conducts worse than blood, so burying the electrode must "
         "raise Z and send more power into tissue"),
    "test_chain_runs_end_to_end":
        ("the whole chain runs and returns sensible numbers",
         "this needs all four modules working"),
    "test_more_field_means_less_angle_error":
        ("more field means a smaller aiming error",
         "check equilibrium_tilt and angle_error"),
    "test_deeper_seating_grows_the_lesion":
        ("deeper contact makes a bigger lesion",
         "check lesion_volume: count only cells with Z < 0 and p above the threshold"),
    "test_field_buys_lesion_at_the_steepest_command":
        ("at the steepest command, more field buys more lesion",
         "this is the headline result of the whole project"),
}

GREEN, RED, GREY, YELLOW, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = GREY = YELLOW = RESET = ""


class Collector:
    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        if report.when not in ("call", "setup"):
            return
        if report.when == "setup" and not report.failed:
            return
        name = report.nodeid.split("::")[-1]
        text = str(report.longrepr) if report.longrepr else ""
        todo = "NotImplementedError" in text
        missing = re.search(r"NotImplementedError: (.+)", text)
        self.results.append({
            "file": report.nodeid.split("::")[0],
            "name": name,
            "passed": report.passed,
            "todo": todo,
            "missing": missing.group(1).strip() if missing else "",
        })


def main():
    wanted = [a for a in sys.argv[1:] if a in MILESTONES] or list(MILESTONES)
    files = [MILESTONES[m][0] for m in wanted]
    collector = Collector()
    # Swallow pytest's own output.  Its tracebacks are forty lines of machinery
    # per failure, and the report below says the same thing in one line.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        pytest.main(["-q", "--no-header", "--tb=no", "-p", "no:cacheprovider",
                     *files], plugins=[collector])

    print("=" * 74)
    n_ok = n_todo = n_fail = 0
    for m in wanted:
        path, heading = MILESTONES[m]
        rows = [r for r in collector.results if r["file"] == path]
        if not rows:
            continue
        print(f"\n{heading}")
        for i, r in enumerate(rows, 1):
            title, hint = TITLES.get(r["name"], (r["name"], ""))
            if r["passed"]:
                n_ok += 1
                print(f"  {GREEN}[ ok ]{RESET} {i}. {title}")
            elif r["todo"]:
                n_todo += 1
                print(f"  {GREY}[todo]{RESET} {i}. {title}")
                if r["missing"]:
                    print(f"         write it in {locate(r['missing'])}")
            else:
                n_fail += 1
                print(f"  {RED}[FAIL]{RESET} {i}. {title}")
                if hint:
                    print(f"         {YELLOW}hint:{RESET} {hint}")
    total = n_ok + n_todo + n_fail
    print("\n" + "-" * 74)
    print(f"  {n_ok} of {total} checks pass"
          f"   |   {n_todo} not written yet   |   {n_fail} written but wrong")
    if n_fail:
        print("  Fix the first [FAIL] above; the ones after it often depend on it.")
    elif n_todo:
        print("  Keep going: open the file named in the first [todo] line.")
    else:
        print(f"  {GREEN}All checks pass.  Run the milestone scripts and write the report.{RESET}")
    print("=" * 74)
    return 0 if n_fail == 0 and n_todo == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
