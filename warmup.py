"""MILESTONE 0 — a worked example.  PROVIDED, COMPLETE, AND COMMENTED.

Nothing here is graded.  This file exists so that you can see a finished
function of exactly the kind you are about to write, in the style the rest of
the kit expects.  Read it, run `python scripts/m0_warmup.py`, and keep it open
beside you while you work on Milestone 1 — the structure of `wire_field` below
is the structure of `biot_savart`.

The problem: a straight wire of length L on the z axis carries current I.
Find B at a set of points, and check it against the textbook result for an
infinitely long wire,  B = mu0 I / (2 pi r).

If you come from MATLAB, the comments marked [MATLAB] give the equivalent.
"""

import numpy as np                      # [MATLAB] numpy is the array library.
                                        # np.array <-> a MATLAB matrix.
from . import constants as C


def wire_nodes(length, n_seg):
    """Cut a straight wire of this length into n_seg pieces.

    Returns
    -------
    seg_mid : (n_seg, 3) array — the midpoint of each piece
    seg_vec : (n_seg, 3) array — the vector along each piece

    An (n_seg, 3) array is a table with n_seg rows and 3 columns: one row per
    piece, the columns being x, y, z.
    """
    # 1. n_seg+1 evenly spaced z values from -L/2 to +L/2.
    #    [MATLAB] z = linspace(-length/2, length/2, n_seg+1);
    z = np.linspace(-0.5 * length, 0.5 * length, n_seg + 1)

    # 2. Make the full (n_seg+1, 3) table of node positions.  x and y are zero
    #    because the wire lies on the z axis.
    #    np.zeros_like(z) is an array of zeros the same shape as z.
    #    np.stack([...], axis=1) glues the three columns side by side.
    #    [MATLAB] nodes = [zeros(size(z)), zeros(size(z)), z];
    nodes = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)

    # 3. The vector along piece s is (node s+1) - (node s).
    #    nodes[1:] means "every row from index 1 onwards"   [MATLAB] nodes(2:end,:)
    #    nodes[:-1] means "every row except the last"       [MATLAB] nodes(1:end-1,:)
    #    NOTE: Python indices start at 0, not 1.
    seg_vec = nodes[1:] - nodes[:-1]

    # 4. The midpoint of piece s is the average of its two end nodes.
    seg_mid = 0.5 * (nodes[1:] + nodes[:-1])

    return seg_mid, seg_vec


def wire_field(points, seg_mid, seg_vec, current):
    """B at every point, from every segment.  Biot-Savart, Ulaby Eq. (5.21):

        B(r) = (mu0 I / 4 pi) * sum over segments of  dl x R / |R|^3

    where R = r - r_segment points FROM the segment TO the field point.

    points   (M, 3)  where you want the field
    seg_mid  (S, 3)  segment midpoints      (from wire_nodes)
    seg_vec  (S, 3)  segment vectors        (from wire_nodes)
    current  a single number, amperes
    returns  (M, 3)  one B vector per point

    The trick below is "broadcasting": instead of looping over M points and S
    segments, we build one (M, S, 3) block holding every point-segment pair and
    let numpy do the arithmetic in one go.  It is the same answer a double for
    loop would give, perhaps a hundred times faster.
    """
    # Make sure points is a 2-D table even if the caller passed one point.
    points = np.atleast_2d(np.asarray(points, dtype=float))

    # 1. R for every (point, segment) pair.
    #    points[:, None, :] has shape (M, 1, 3); seg_mid[None, :, :] is (1, S, 3).
    #    Subtracting them gives (M, S, 3): numpy repeats along the size-1 axes.
    R = points[:, None, :] - seg_mid[None, :, :]

    # 2. |R| for every pair -> (M, S).  axis=2 means "collapse the x,y,z axis".
    #    [MATLAB] the closest thing is vecnorm(R, 2, 3)
    dist = np.linalg.norm(R, axis=2)

    # 3. Protect against a point sitting exactly on the wire (|R| = 0).
    #    np.where(condition, a, b) is an element-by-element if-else.
    dist = np.where(dist < 1e-12, np.inf, dist)

    # 4. dl x R for every pair -> (M, S, 3).  np.cross does the cross product
    #    on the last axis, and broadcasts the rest.
    cross = np.cross(seg_vec[None, :, :], R)

    # 5. Divide by |R|^3 and add up over the segments (axis=1), then scale.
    #    dist[:, :, None] turns (M, S) into (M, S, 1) so it divides each of the
    #    three components.
    return (C.MU0 * current / (4.0 * np.pi)) * np.sum(
        cross / dist[:, :, None] ** 3, axis=1)


def long_wire_closed_form(r, current):
    """The textbook answer for an infinite wire: B = mu0 I / (2 pi r)."""
    return C.MU0 * current / (2.0 * np.pi * np.asarray(r, dtype=float))
