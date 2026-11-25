"""MILESTONE 4 — current in tissue.  YOUR WORK, with the linear algebra provided.

We solve  div(sigma grad V) = 0  on an (r, z) grid, which is legal because
everything is symmetric about the catheter axis: nothing depends on the angle
around it, so a 2-D grid describes a 3-D problem.

Geometry: the electrode is a sphere of radius a centred at z = a - depth, the
tissue interface is the plane z = 0, myocardium fills z < 0 and blood z > 0.
Cells at or beyond ``domain_radius`` from the origin are held at V = 0.

TWO FUNCTIONS ARE GIVEN TO YOU: build_grid, which is bookkeeping, and
assemble_and_solve, which builds the matrix and solves it.  Writing a sparse
solver is software work, not electromagnetics.  Everything else is yours, and
all of it is physics.

    python check.py 4
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import constants as C
from .geometry import AblationSettings


# ===========================================================================
#  PROVIDED — read these two, do not edit them
# ===========================================================================

def build_grid(settings: AblationSettings = None):
    """PROVIDED.  Cell-centred axisymmetric grid.

    Returns a dict with:
        h        cell size, m
        nr, nz   number of cells across and up
        r, z     1-D arrays of cell-centre coordinates
        R, Z     (nr, nz) tables of those coordinates, one entry per cell
        volume   (nr, nz) cell volumes.  A cell at radius r is a ring, so its
                 volume is 2 pi r h^2, not h^3.
    """
    s = settings or AblationSettings()
    h = s.cell_size
    nr = int(round(s.domain_radius / h))
    nz = int(round(2.0 * s.domain_half_height / h))
    r = (np.arange(nr) + 0.5) * h
    z = -s.domain_half_height + (np.arange(nz) + 0.5) * h
    R, Z = np.meshgrid(r, z, indexing="ij")
    return {"h": h, "nr": nr, "nz": nz, "r": r, "z": z, "R": R, "Z": Z,
            "volume": 2.0 * np.pi * R * h * h}


def assemble_and_solve(grid, g_r, g_z, fixed, fixed_values):
    """PROVIDED.  Solve the network of conductances for the cell voltages.

    Think of every cell as a node and every face as a resistor.  Kirchhoff's
    current law at each free node is one equation; this builds that system and
    solves it.

    g_r          (nr-1, nz) conductance of each radial face
    g_z          (nr, nz-1) conductance of each axial face
    fixed        (nr, nz) boolean, True where the voltage is known
    fixed_values (nr, nz) the known voltages (anything where fixed is False is
                 ignored)
    returns      (nr, nz) the solved potential
    """
    nr, nz = grid["nr"], grid["nz"]
    idx = np.arange(nr * nz).reshape(nr, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(nr * nz)

    def add_face(ia, ja, ib, jb, g):
        a, b = idx[ia, ja].ravel(), idx[ib, jb].ravel()
        gg = g.ravel()
        rows.extend([a, b]); cols.extend([b, a]); vals.extend([-gg, -gg])
        np.add.at(diag, a, gg); np.add.at(diag, b, gg)

    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nz), indexing="ij")
    add_face(ii, jj, ii + 1, jj, g_r)
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nz - 1), indexing="ij")
    add_face(ii, jj, ii, jj + 1, g_z)

    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nr * nz, nr * nz)).tocsr() + sp.diags(diag)
    f = fixed.ravel()
    b = -A[:, f] @ fixed_values.ravel()[f]
    v = np.zeros(nr * nz)
    v[f] = fixed_values.ravel()[f]
    v[~f] = spla.spsolve(A[~f][:, ~f].tocsc(), b[~f])
    return v.reshape(nr, nz)


# ===========================================================================
#  YOUR WORK
# ===========================================================================

def conductivity_map(grid, settings: AblationSettings = None):
    """(nr, nz) table of sigma: myocardium below the interface, blood above.

    One line with np.where(condition, value_if_true, value_if_false):
    the condition is grid["Z"] < 0.0.
    [MATLAB] np.where is the vectorised ternary, like (cond).*a + (~cond).*b
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    return np.where(grid["Z"] < 0.0, s.sigma_myocardium, s.sigma_blood)


def electrode_mask(grid, settings: AblationSettings = None):
    """(nr, nz) boolean: True in cells inside the spherical electrode.

    RECIPE (2 lines)
        1. distance of each cell centre from the electrode centre:
           d = np.hypot(grid["R"], grid["Z"] - settings.electrode_centre_z)
        2. return d <= settings.electrode_radius

    This blocky approximation of a smooth sphere is called a staircase, and it
    is the biggest error in the whole milestone.  The refinement study measures
    it: the error falls only in proportion to the cell size, not its square.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    d = np.hypot(grid["R"], grid["Z"] - s.electrode_centre_z)
    return d <= s.electrode_radius


def far_boundary_mask(grid, settings: AblationSettings = None):
    """(nr, nz) boolean: True in cells at or beyond the grounded shell.

    One line: np.hypot(grid["R"], grid["Z"]) >= settings.domain_radius
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    return np.hypot(grid["R"], grid["Z"]) >= s.domain_radius


def harmonic_mean(sigma_a, sigma_b):
    """JOB  the effective conductivity of a face between two cells of different sigma.

    2 sa sb / (sa + sb).  One line.

    WHY THIS AND NOT THE AVERAGE  two half-cells sit in series across a face,
    and resistances in series add.  The harmonic mean is what keeps the normal
    component of J continuous across the blood/myocardium jump, which is the
    boundary condition from Section 4-8.  Run the whole milestone again later
    with 0.5*(sa+sb) instead and report what changes — it is a two-character
    edit and it makes the point better than any argument.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    return 2.0 * sigma_a * sigma_b / (sigma_a + sigma_b)


def face_conductances(sigma, grid):
    """Conductance of every face.  Returns (g_r, g_z).

    THE PHYSICS  a face of area A and thickness h between two cells conducts
    G = sigma_face * A / h, with sigma_face the harmonic mean of its two cells.

    The face areas come from the ring geometry:
        radial face between cell i and i+1 :  A = 2 pi r_face h, where
            r_face = 0.5 * (R[:-1, :] + R[1:, :])
        axial face between cell j and j+1  :  A = 2 pi r h, using R[:, :-1]

    RECIPE (about 6 lines)
        1. h = grid["h"] ; R = grid["R"]
        2. r_face as above
        3. g_r = harmonic_mean(sigma[:-1, :], sigma[1:, :]) * (2*np.pi*r_face*h) / h
        4. g_z = harmonic_mean(sigma[:, :-1], sigma[:, 1:]) * (2*np.pi*R[:, :-1]*h) / h
        5. return g_r, g_z

    SHAPES  g_r is (nr-1, nz) and g_z is (nr, nz-1).  One fewer face than cells
    in the direction you are crossing.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    h = grid["h"]
    R = grid["R"]
    r_face = 0.5 * (R[:-1, :] + R[1:, :])
    g_r = harmonic_mean(sigma[:-1, :], sigma[1:, :]) * (2.0 * np.pi * r_face * h) / h
    g_z = harmonic_mean(sigma[:, :-1], sigma[:, 1:]) * (2.0 * np.pi * R[:, :-1] * h) / h
    return g_r, g_z


def solve_potential(grid, sigma, settings: AblationSettings = None):
    """(nr, nz) potential, volts.  This one just wires your pieces together.

    RECIPE (about 6 lines)
        1. elec = electrode_mask(grid, settings)
        2. far = far_boundary_mask(grid, settings)
        3. fixed = elec | far            ( | is element-by-element OR )
        4. fixed_values = np.where(elec, settings.voltage, 0.0)
        5. g_r, g_z = face_conductances(sigma, grid)
        6. return assemble_and_solve(grid, g_r, g_z, fixed, fixed_values)
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    elec = electrode_mask(grid, s)
    far = far_boundary_mask(grid, s)
    fixed = elec | far
    fixed_values = np.where(elec, s.voltage, 0.0)
    g_r, g_z = face_conductances(sigma, grid)
    return assemble_and_solve(grid, g_r, g_z, fixed, fixed_values)


def face_powers(v, sigma, grid):
    """Power dissipated in each face, G * (dV)^2.  Returns (p_r, p_z).

    This is Joule's law written for the network: each face is a resistor with
    a voltage across it.  Summing these gives the exact total, which is why we
    do not square a cell-centred gradient instead.

    RECIPE (4 lines)
        1. g_r, g_z = face_conductances(sigma, grid)
        2. p_r = g_r * (v[:-1, :] - v[1:, :])**2
        3. p_z = g_z * (v[:, :-1] - v[:, 1:])**2
        4. return p_r, p_z
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    g_r, g_z = face_conductances(sigma, grid)
    p_r = g_r * (v[:-1, :] - v[1:, :]) ** 2
    p_z = g_z * (v[:, :-1] - v[:, 1:]) ** 2
    return p_r, p_z

def _cell_power(v, sigma, grid):
    p_r, p_z = face_powers(v, sigma, grid)
    cell = np.zeros_like(v)
    cell[:-1, :] += 0.5 * p_r
    cell[1:, :] += 0.5 * p_r
    cell[:, :-1] += 0.5 * p_z
    cell[:, 1:] += 0.5 * p_z
    return cell

def power_density(v, sigma, grid, settings: AblationSettings = None):
    """(nr, nz) table of p = sigma |E|^2 in W/m^3.

    Each face power belongs half to the cell on each side of it.

    RECIPE (about 8 lines)
        1. p_r, p_z = face_powers(v, sigma, grid)
        2. cell = np.zeros_like(v)
        3. cell[:-1, :] += 0.5 * p_r ;  cell[1:, :] += 0.5 * p_r
        4. cell[:, :-1] += 0.5 * p_z ;  cell[:, 1:] += 0.5 * p_z
        5. out = cell / grid["volume"]          # watts per cubic metre
        6. zero it inside the metal electrode:
           out[electrode_mask(grid, settings)] = 0.0
        7. return out
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    cell = _cell_power(v, sigma, grid)
    out = cell / grid["volume"]
    out[electrode_mask(grid, s)] = 0.0
    return out


def total_power(v, sigma, grid, region=None):
    """Total dissipated power, W.

    Multiply power_density by grid["volume"] and sum.  If region (a boolean
    table) is given, sum only those cells:  np.sum(p[region] * vol[region]).
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    cell = _cell_power(v, sigma, grid)
    if region is None:
        return float(np.sum(cell))
    return float(np.sum(cell[region]))


def electrode_current(v, sigma, grid, settings: AblationSettings = None):
    """Current delivered by the electrode, A.

    Joule's law gives it without adding up fluxes over a staircase surface:

        P = integral(sigma |E|^2 dv) = V0 * I      so      I = P / V0

    Two lines, guarding against V0 = 0.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    if s.voltage == 0.0:
        return 0.0
    return total_power(v, sigma, grid) / s.voltage


def impedance(v, sigma, grid, settings: AblationSettings = None):
    """Z = V0 / I, ohms.  One line.  This is the number electrophysiologists
    watch on the generator display."""
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    i = electrode_current(v, sigma, grid, s)
    return s.voltage / i if i > 0.0 else float("inf")


def power_fraction_myocardium(v, sigma, grid):
    """Share of the delivered power landing in tissue, between 0 and 1.

    RECIPE  power_density, then sum over cells with grid["Z"] < 0 and divide by
    the sum over all cells.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    total = total_power(v, sigma, grid)
    if total <= 0.0:
        return 0.0
    return total_power(v, sigma, grid, region=grid["Z"] < 0.0) / total


def lesion_volume(v, sigma, grid, settings: AblationSettings = None):
    """Volume of myocardium above the power-density threshold, m^3.

    RECIPE (3 lines)
        1. p = power_density(v, sigma, grid, settings)
        2. hot = (p > settings.lesion_threshold) & (grid["Z"] < 0.0) \\
                 & (~electrode_mask(grid, settings))
        3. return float(np.sum(grid["volume"][hot]))

    A contour of power density is a PROXY for a lesion, not a prediction of
    one.  A real answer needs the bioheat equation, perfusion and a thermal
    dose integral.  Say so in the report; marks are given for it.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    p = power_density(v, sigma, grid, s)
    hot = (p > s.lesion_threshold) & (grid["Z"] < 0.0) & (~electrode_mask(grid, s))
    return float(np.sum(grid["volume"][hot]))


def analytic_resistance(settings: AblationSettings = None, straddling=True):
    """The closed form your grid answer is checked against, ohms.

    THE PHYSICS  put the sphere exactly on the interface.  Its field lines run
    radially outward and lie in the interface plane, so V = A/R + B solves
    Laplace's equation in both half spaces at once, and each side draws current
    in proportion to its own conductivity.  The two halves are resistors in
    parallel:

        R = (1/a - 1/R_out) / [ 2 pi (sigma_myo + sigma_blood) ]

    With straddling=False the sphere sits wholly in blood and the 2 pi becomes
    4 pi with a single sigma.  Three lines.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    a = s.electrode_radius
    r_out = s.domain_radius
    geom = 1.0 / a - 1.0 / r_out
    if straddling:
        return geom / (2.0 * np.pi * (s.sigma_myocardium + s.sigma_blood))
    return geom / (4.0 * np.pi * s.sigma_blood)


def solve(settings: AblationSettings = None):
    """Run the whole chain once and return a dict.

    RECIPE  build the grid, make sigma, solve for v, then return
        {"grid": grid, "sigma": sigma, "v": v,
         "impedance": ..., "current": ..., "power": ...,
         "fraction_myocardium": ..., "lesion_volume": ...}
    using the functions above.  About 8 lines, no new physics.
    """
    # >>>>>>>>>>>>>>>>>>>>  WRITE YOUR CODE HERE  <<<<<<<<<<<<<<<<<<<<
    # Delete the one line below, then type the body in its place,
    # indented exactly this far.  Follow the RECIPE above.  End with  return
    s = settings or AblationSettings()
    grid = build_grid(s)
    sigma = conductivity_map(grid, s)
    v = solve_potential(grid, sigma, s)
    return {"grid": grid, "sigma": sigma, "v": v,
            "impedance": impedance(v, sigma, grid, s),
            "current": electrode_current(v, sigma, grid, s),
            "power": total_power(v, sigma, grid),
            "fraction_myocardium": power_fraction_myocardium(v, sigma, grid),
            "lesion_volume": lesion_volume(v, sigma, grid, s)}
