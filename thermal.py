"""MILESTONE 3 — what the field costs.  YOUR WORK.

Short functions, one trap.  The field is set by the ampere-turns N*I, but each
turn carries only I, and heating goes as the square of what flows in the
conductor.  Putting N*I into P = I^2 R overstates the heat by N^2 — a factor
of 24,000 for the baseline coil.  The checker will catch it, but think about
which current belongs where before you write it.

    python check.py 3
"""

import numpy as np

from . import constants as C
from .geometry import CoilPair
from .magnetostatics import loop_segments, biot_savart

def wire_resistance(length, area, sigma=C.SIGMA_COPPER):
    """JOB  resistance of a straight conductor of given length and cross-section.

    Ulaby Eq. (4.80):  R = l / (sigma A), ohms.  One line.

    CHECK  50 m of copper with a 2 cm radius gives 6.9e-4 ohm (Exercise 4-14).
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    return length / (sigma * area)


def coil_resistance(coil: CoilPair = None, sigma=C.SIGMA_COPPER):
    """Resistance of ONE coil.

    The conductor length is already worked out for you: coil.wire_length is
    N * 2 pi a.  Pass it and coil.wire_area to wire_resistance.  Two lines.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    return wire_resistance(coil.wire_length, coil.wire_area, sigma)


def coil_dissipation(coil: CoilPair = None, sigma=C.SIGMA_COPPER):
    """JOB  heat dissipated in one coil's winding at the operating current.

    Ulaby Eq. (4.92):  P = I^2 R, watts.  One line — but WHICH current?

    CHECK  the baseline coil should come out near 1.45 kW.  If you get tens of
    megawatts you used coil.ampere_turns where coil.current belongs.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    return coil.current ** 2 * coil_resistance(coil, sigma)


def current_density(coil: CoilPair = None):
    """A/m^2 in the conductor: the per-turn current over the wire area.

    One line.  Compare the answer with 4 A/mm^2, about what a winding can take
    with forced cooling.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    return coil.current / coil.wire_area


def force_between_coils(coil: CoilPair = None, n_seg=720):
    """Axial force one coil exerts on the other, N.  Positive = attraction.

    Do this the long way, with your Milestone 1 and 2 code:
        1. za, zb = coil.centres
        2. build both loops with loop_segments(coil.radius, za, n_seg) and
           the same at zb
        3. evaluate the field of coil 1 ON the conductor of coil 2:
           B1 = biot_savart(mid2, mid1, vec1, coil.ampere_turns)
        4. F = coil.ampere_turns * np.sum(np.cross(vec2, B1), axis=0)
        5. the answer is the z component.  Return it with the sign flipped,
           float(-F[2]), so that attraction comes out positive.

    CHECK  about 690 N at the baseline, and it must scale as I^2.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    za, zb = coil.centres
    mid1, vec1 = loop_segments(coil.radius, za, n_seg)
    mid2, vec2 = loop_segments(coil.radius, zb, n_seg)
    B1 = biot_savart(mid2, mid1, vec1, coil.ampere_turns)
    F = coil.ampere_turns * np.sum(np.cross(vec2, B1), axis=0)
    return float(-F[2])


def force_per_length_parallel_wires(current_1, current_2, separation):
    """JOB  force per metre between two long parallel wires, as a textbook check.

    Ulaby Eq. (5.41):  F' = mu0 I1 I2 / (2 pi d), N/m.  One line.

    CHECK  1 A in each wire 1 m apart gives 2e-7 N/m — the definition the
    ampere carried until 2019.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    return C.MU0 * current_1 * current_2 / (2.0 * np.pi * separation)


def convective_area(coil: CoilPair = None):
    """Outer surface of the winding bundle, m^2.

    Treat the winding as a square-section torus: the bundle side is
    np.sqrt(coil.turns * coil.wire_area), and the surface is the circumference
    2 pi a times the bundle perimeter 4 * side.  Two lines.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    side = np.sqrt(coil.turns * coil.wire_area)
    return 2.0 * np.pi * coil.radius * 4.0 * side


def temperature_rise(coil: CoilPair = None, h=C.H_FREE_CONVECTION):
    """Lumped free-convection estimate, dT = P / (h A), in kelvin.  One line.

    Crude on purpose: no radiation, no conduction into the former, uniform
    surface temperature.  You get about 260 K, which is the number that tells
    the mechanical designer this coil cannot run uncooled.  Report the caveats
    with it.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    return coil_dissipation(coil) / (h * convective_area(coil))
