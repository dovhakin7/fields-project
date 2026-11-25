"""MILESTONE 2 — force and torque on the tip magnet.  YOUR WORK.

The magnet is treated as an equivalent current loop so that force and torque
come from the two integrals taught in Section 5-1 rather than from a quoted
dipole formula.  Everything here is short; the longest function is six lines.

    python check.py 2
"""

import numpy as np

from . import constants as C
from .geometry import TipMagnet, Catheter
from scipy.optimize import brentq
from .magnetostatics import loop_segments


def moment_vector(magnet: TipMagnet = None, direction=(0.0, 0.0, 1.0)):
    """JOB  turn the tip magnet's strength into a vector pointing where it points.

    m = M V n_hat, in A*m^2.

    RECIPE (4 lines)
        1. if magnet is None: magnet = TipMagnet()
        2. d = np.asarray(direction, dtype=float)
        3. normalise it:  d = d / np.linalg.norm(d)
        4. return magnet.moment_magnitude * d

    magnet.moment_magnitude already does M*V for you — see emlab/geometry.py.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if magnet is None:
        magnet = TipMagnet()
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return magnet.moment_magnitude * d


def equivalent_loop(magnet: TipMagnet = None, tilt=0.0, n_seg=180):
    """Represent the slug as ONE current loop.  Returns (seg_mid, seg_vec, current).

    THE PHYSICS  a uniformly magnetised cylinder carries bound surface current
    K = M x n_hat around its side: a solenoid of total current M*L.  Collapse
    that to a single loop of the magnet's radius carrying

        I_eq = M * L            (magnet.magnetisation * magnet.length)

    which has the right moment, since I_eq * pi r^2 = M * L * pi r^2 = M * V.

    RECIPE (about 7 lines)
        1. current = magnet.magnetisation * magnet.length
        2. build the loop in the x-y plane with your Milestone 1 function:
           mid, vec = loop_segments(magnet.radius, 0.0, n_seg)
           (import it at the top:  from .magnetostatics import loop_segments)
        3. rotate both arrays about the y axis by `tilt` radians.  The rotation
           matrix is
               c, s = np.cos(tilt), np.sin(tilt)
               rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
           and you apply it to a table of row vectors with  mid @ rot.T
           [MATLAB] the @ operator is matrix multiply, like * in MATLAB.
        4. return mid_rotated, vec_rotated, current

    CHECK  current * pi * magnet.radius**2 must equal magnet.moment_magnitude.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if magnet is None:
        magnet = TipMagnet()
    current = magnet.magnetisation * magnet.length
    mid, vec = loop_segments(magnet.radius, 0.0, n_seg)
    c, s = np.cos(tilt), np.sin(tilt)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return mid @ rot.T, vec @ rot.T, current


def force_on_loop(seg_mid, seg_vec, current, b_field):
    """JOB  add up the force the field exerts on every piece of the loop.

    F = I * sum of (dl x B), newtons.  Ulaby Eq. (5.10).

    b_field is EITHER a function you can call with the (S,3) midpoints and that
    returns (S,3), OR a single (3,) vector for a uniform field.  Handle both:

        if callable(b_field):
            B = b_field(seg_mid)
        else:
            B = np.broadcast_to(np.asarray(b_field, dtype=float), seg_mid.shape)

    then one line:  return current * np.sum(np.cross(seg_vec, B), axis=0)

    CHECK  in a uniform field the answer must be zero to round-off — that is
    Eq. (5.11), and the test insists on it.  If you get something like 1e-3 N,
    your loop is not closing: check loop_segments.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    seg_mid = np.asarray(seg_mid, dtype=float)
    seg_vec = np.asarray(seg_vec, dtype=float)
    if callable(b_field):
        B = b_field(seg_mid)
    else:
        B = np.broadcast_to(np.asarray(b_field, dtype=float), seg_mid.shape)
    return current * np.sum(np.cross(seg_vec, B), axis=0)


def torque_on_loop(seg_mid, seg_vec, current, b_field, origin=(0.0, 0.0, 0.0)):
    """JOB  add up the turning moment of those same per-piece forces about a point.

    T = sum of r x (I dl x B), N*m.  Must reproduce m x B, Eq. (5.20).

    RECIPE  same first step as force_on_loop to get B, then
        1. dF = current * np.cross(seg_vec, B)          # force on each piece
        2. r = seg_mid - np.asarray(origin, dtype=float)
        3. return np.sum(np.cross(r, dF), axis=0)

    Note the order inside each cross product.  r x dF, not dF x r.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    seg_mid = np.asarray(seg_mid, dtype=float)
    seg_vec = np.asarray(seg_vec, dtype=float)
    if callable(b_field):
        B = b_field(seg_mid)
    else:
        B = np.broadcast_to(np.asarray(b_field, dtype=float), seg_mid.shape)
    dF = current * np.cross(seg_vec, B)
    r = seg_mid - np.asarray(origin, dtype=float)
    return np.sum(np.cross(r, dF), axis=0)


def equilibrium_tilt(commanded, b_magnitude, catheter: Catheter = None):
    """The angle the tip actually reaches, in RADIANS.

    THE PHYSICS  the magnetic torque turns the tip toward the commanded
    direction, the shaft resists in proportion to how far it is bent, and the
    tip stops where they balance:

        k * theta  =  m |B| sin(commanded - theta)

    There is no formula for theta; you solve the equation numerically.

    RECIPE (about 8 lines)
        1. m = catheter.magnet.moment_magnitude ; k = catheter.shaft_stiffness
        2. define the residual (the left side minus the right side):
               def residual(theta):
                   return k*theta - m*b_magnitude*np.sin(commanded - theta)
           [MATLAB] this is a function handle, @(theta) ...
        3. if commanded <= 0: return 0.0
        4. the residual is negative at theta = 0 and positive at
           theta = commanded, so a root lies between.  Find it with
               from scipy.optimize import brentq
               return float(brentq(residual, 0.0, commanded, xtol=1e-12))
           [MATLAB] brentq is fzero with a bracket.

    CHECK  at 79.6 mT a 60 degree command should reach about 30 degrees.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if catheter is None:
        catheter = Catheter()
    m = catheter.magnet.moment_magnitude
    k = catheter.shaft_stiffness

    def residual(theta):
        return k * theta - m * b_magnitude * np.sin(commanded - theta)

    if commanded <= 0:
        return 0.0
    return float(brentq(residual, 0.0, commanded, xtol=1e-12))


def angle_error(commanded, b_magnitude, catheter: Catheter = None):
    """commanded - achieved, radians.  One line, using equilibrium_tilt."""
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    return commanded - equilibrium_tilt(commanded, b_magnitude, catheter)
