"""MILESTONE 1 — the navigation field.  YOUR WORK.

Work through emlab/warmup.py and scripts/m0_warmup.py before starting here.
Two of the functions below are the warm-up example again, with a circle in
place of a straight wire, so you can copy its structure.

How to work:  write one function, then run

    python check.py 1

which tells you, in English, which of the six Milestone 1 checks pass.
"""

import numpy as np

from . import constants as C
from .geometry import CoilPair


def loop_segments(radius, centre_z=0.0, n_seg=400):
    """Chop a circular loop into straight segments.

    This is wire_nodes() from the warm-up, with the wire bent into a circle.

    INPUTS
        radius    loop radius, m
        centre_z  the loop lies in the plane z = centre_z
        n_seg     how many straight pieces

    RETURNS
        seg_mid   (n_seg, 3) array, midpoint of each piece
        seg_vec   (n_seg, 3) array, vector along each piece

    RECIPE (about 5 lines)
        1. phi = np.linspace(0, 2*np.pi, n_seg + 1)
        2. the node positions are x = radius*cos(phi), y = radius*sin(phi),
           z = centre_z everywhere.  Glue them into an (n_seg+1, 3) table with
           np.stack([x, y, z], axis=1).  np.full_like(phi, centre_z) gives the
           z column.
        3. seg_vec = nodes[1:] - nodes[:-1]
        4. seg_mid = 0.5 * (nodes[1:] + nodes[:-1])
        5. return seg_mid, seg_vec

    CHECK  the returned arrays must have n_seg rows, and the segment vectors
    must sum to (almost) zero because the loop closes on itself.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    phi = np.linspace(0.0, 2.0 * np.pi, n_seg + 1)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = np.full_like(phi, centre_z)
    nodes = np.stack([x, y, z], axis=1)
    seg_vec = nodes[1:] - nodes[:-1]
    seg_mid = 0.5 * (nodes[1:] + nodes[:-1])
    return seg_mid, seg_vec


def biot_savart(points, seg_mid, seg_vec, current):
    """Magnetic flux density at a set of points, in tesla.

    THE PHYSICS — Ulaby Eq. (5.21), summed over straight segments:

        B(r) = (mu0 I / 4 pi) * sum over s of  dl_s x R_s / |R_s|^3

    with R_s = r - r_s pointing FROM the segment TO the field point.

    INPUTS
        points    (M, 3) where you want the field
        seg_mid   (S, 3) segment midpoints
        seg_vec   (S, 3) segment vectors
        current   one number, A.  For an N-turn coil pass the ampere-turns.
    RETURNS
        (M, 3) — one B vector per point

    RECIPE  this is wire_field() in emlab/warmup.py, unchanged.  Read it, then
    write the same five steps here.  Do not import it; type it out, because you
    need to understand it for the rest of the project.

    COMMON BUGS
        * np.cross(R, dl) instead of np.cross(dl, R) — every sign flips.
        * forgetting dist[:, :, None], which gives a broadcasting error about
          shapes (M, S, 3) and (M, S).
        * dividing by |R|^2 instead of |R|^3.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    points = np.atleast_2d(np.asarray(points, dtype=float))
    seg_mid = np.asarray(seg_mid, dtype=float)
    seg_vec = np.asarray(seg_vec, dtype=float)
    out = np.zeros((points.shape[0], 3))
    block = max(1, int(2_000_000 // max(1, seg_mid.shape[0])))
    for i0 in range(0, points.shape[0], block):
        pts = points[i0:i0 + block]
        R = pts[:, None, :] - seg_mid[None, :, :]
        dist = np.linalg.norm(R, axis=2)
        dist = np.where(dist < 1e-12, np.inf, dist)
        cross = np.cross(seg_vec[None, :, :], R)
        out[i0:i0 + block] = np.sum(cross / dist[:, :, None] ** 3, axis=1)
    return (C.MU0 * current / (4.0 * np.pi)) * out


def coil_pair_field(points, coil: CoilPair = None, n_seg=400):
    """Field of BOTH navigation coils, driven in series.  Returns (M, 3).

    RECIPE (about 6 lines)
        1. if coil is None: coil = CoilPair()
        2. start with an all-zeros answer:
           total = np.zeros((np.atleast_2d(points).shape[0], 3))
        3. coil.centres gives the two z positions of the coil planes.
           Loop over them with:  for zc in coil.centres:
        4. inside the loop, build that coil's segments with loop_segments(
           coil.radius, zc, n_seg) and add its biot_savart contribution to
           total.  Pass coil.ampere_turns as the current, not coil.current:
           all N turns sit on top of each other.
        5. return total

    CHECK  at the origin you should get (4/5)^1.5 * mu0 * N I / a = 79.6 mT for
    the baseline coil, pointing along +z.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    total = np.zeros((pts.shape[0], 3))
    for zc in coil.centres:
        mid, vec = loop_segments(coil.radius, zc, n_seg)
        total += biot_savart(pts, mid, vec, coil.ampere_turns)
    return total


def on_axis_loop_bz(z, radius, current, turns=1):
    """Closed form on the axis of one loop — Ulaby Eq. (5.34):

        Bz = mu0 N I a^2 / [ 2 (a^2 + z^2)^{3/2} ]

    One line.  z may be a single number or an array; write it with numpy
    operations (**, +, *) and it will work for both.

    This is the reference your numerical field gets checked against, so it must
    NOT call biot_savart.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    z = np.asarray(z, dtype=float)
    return C.MU0 * turns * current * radius ** 2 / (2.0 * (radius ** 2 + z ** 2) ** 1.5)


def on_axis_pair_bz(z, coil: CoilPair = None):
    """Closed form for the pair: superpose two loops.

    RECIPE  za, zb = coil.centres, then add on_axis_loop_bz(z - za, ...) to
    on_axis_loop_bz(z - zb, ...), passing coil.radius, coil.current and
    coil.turns.  Superposition is legal because Maxwell's equations are linear.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    if coil is None:
        coil = CoilPair()
    za, zb = coil.centres
    return (on_axis_loop_bz(np.asarray(z, dtype=float) - za, coil.radius, coil.current, coil.turns)
            + on_axis_loop_bz(np.asarray(z, dtype=float) - zb, coil.radius, coil.current, coil.turns))


def uniformity(field_magnitudes, reference=None):
    """How flat a set of |B| values is, as a FRACTION.

    uniformity = (largest - smallest) / reference

    so 5e-4 means "flat to 0.05 %".  If reference is None use the mean of the
    values.  Three lines: np.asarray, then .max(), .min(), and a divide.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    v = np.asarray(field_magnitudes, dtype=float)
    if reference is None:
        reference = v.mean()
    return float((v.max() - v.min()) / reference)


def working_volume_points(half_width=0.050, n=7):
    """An n x n x n grid of points filling the +-50 mm working volume.

    RETURNS (n**3, 3)

    RECIPE (about 4 lines)
        1. s = np.linspace(-half_width, half_width, n)
        2. X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
           [MATLAB] meshgrid, but note indexing="ij" keeps the axis order sane.
        3. flatten each with .ravel() and np.stack them with axis=1.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = np.linspace(-half_width, half_width, n)
    X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
