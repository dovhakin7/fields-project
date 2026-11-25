"""BONUS EXTENSIONS - Authors: [YOUR NAME(S) HERE]  -- fill in before submitting

Four optional extensions from Section 9 of the handout.  Nothing here is needed
by the 29 acceptance tests; it builds on the four graded modules.

  1  three-axis coil set and steering authority over the working volume
  2  irrigated electrode: Pennes bioheat equation beside the conduction solve
  3  inductance of a navigation coil and the supply voltage to slew the field
  4  catheter shaft as a flexible beam, and the tip trajectory
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe

from . import constants as C
from . import conduction as cd
from . import magnetostatics as ms
from . import steering as st
from .geometry import CoilPair, Catheter, AblationSettings


# ===========================================================================
#  EXTENSION 1 - three orthogonal coil pairs
# ===========================================================================

# (e1, e2, e3): a right-handed frame whose third axis is the pair's own axis.
_FRAMES = {
    "x": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
    "y": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "z": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}


def pair_field_per_amp(points, axis, coil: CoilPair, n_seg=240):
    """B (T) per ampere of per-turn current for a Helmholtz pair whose axis is x, y or z."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    e1, e2, e3 = _FRAMES[axis]
    local = np.stack([pts @ e1, pts @ e2, pts @ e3], axis=1)
    unit = CoilPair(radius=coil.radius, separation=coil.separation, turns=coil.turns,
                    current=1.0, wire_area=coil.wire_area)
    b = ms.coil_pair_field(local, unit, n_seg=n_seg)
    return b[:, 0:1] * e1 + b[:, 1:2] * e2 + b[:, 2:3] * e3


def centre_tesla_per_amp(coil: CoilPair):
    return (4.0 / 5.0) ** 1.5 * C.MU0 * coil.turns / coil.radius


def triad_basis(points, coils, n_seg=240):
    """(3, M, 3): the field of each pair per ampere, evaluated at the points."""
    return np.stack([pair_field_per_amp(points, ax, c, n_seg) for ax, c in zip("xyz", coils)])


def currents_for_field(b_vector, coils):
    """Per-turn currents (Ix, Iy, Iz) that put b_vector (T) at the centre."""
    b = np.asarray(b_vector, dtype=float)
    return np.array([b[i] / centre_tesla_per_amp(coils[i]) for i in range(3)])


def max_field_along(direction, coils, i_max):
    """Largest |B| at the centre along a unit direction before ANY axis hits i_max."""
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    per = np.array([centre_tesla_per_amp(c) * i_max for c in coils])
    with np.errstate(divide="ignore"):
        lim = np.where(np.abs(n) > 1e-12, per / np.abs(n), np.inf)
    return float(lim.min())


def tip_direction(b_vectors, catheter: Catheter, rest_axis=(0.0, 0.0, 1.0)):
    """Where the tip points for each local field vector.  Returns (M, 3) unit vectors and (M,) tilt, rad.

    The tip starts along rest_axis.  The field pulls it toward b_hat in the plane
    of the two; the shaft resists with k*theta; equilibrium is the same scalar
    equation as Milestone 2, solved with the LOCAL field magnitude and direction.
    """
    z = np.asarray(rest_axis, dtype=float)
    z = z / np.linalg.norm(z)
    out = np.zeros((len(b_vectors), 3))
    tilt = np.zeros(len(b_vectors))
    for i, b in enumerate(b_vectors):
        mag = np.linalg.norm(b)
        bh = b / mag
        phi = float(np.arccos(np.clip(bh @ z, -1.0, 1.0)))
        th = st.equilibrium_tilt(phi, mag, catheter) if phi > 1e-12 else 0.0
        perp = bh - (bh @ z) * z
        pn = np.linalg.norm(perp)
        perp = perp / pn if pn > 1e-12 else np.zeros(3)
        out[i] = np.cos(th) * z + np.sin(th) * perp
        tilt[i] = th
    return out, tilt


# ===========================================================================
#  EXTENSION 2 - Pennes bioheat beside the conduction solve
# ===========================================================================

K_MYO = 0.56          # W/(m K)   thermal conductivity, myocardium (literature, nominal)
K_BLOOD = 0.52        # W/(m K)   whole blood (literature, nominal)
K_METAL = 70.0        # W/(m K)   platinum-class tip, effectively isothermal
W_MYO = 4.0e4         # W/(m^3 K) rho_b c_b omega_b, perfused myocardium (order of magnitude; vary it)
T_BODY = 37.0         # C


def cell_power_electrode_aware(v, sigma, grid, elec):
    """Watts per cell, with every face touching the electrode credited wholly to the
    non-electrode cell.  Conserves the total (equals the exact network power), and
    puts no heat inside the metal."""
    p_r, p_z = cd.face_powers(v, sigma, grid)
    cell = np.zeros_like(v)
    a, b = elec[:-1, :], elec[1:, :]
    w = np.where(a & ~b, 0.0, np.where(b & ~a, 1.0, 0.5))
    cell[:-1, :] += w * p_r
    cell[1:, :] += (1.0 - w) * p_r
    a, b = elec[:, :-1], elec[:, 1:]
    w = np.where(a & ~b, 0.0, np.where(b & ~a, 1.0, 0.5))
    cell[:, :-1] += w * p_z
    cell[:, 1:] += (1.0 - w) * p_z
    return cell


def solve_pennes(grid, k_map, g_off, w_map, source_watts, fixed, fixed_values, t_art=T_BODY):
    """Steady Pennes bioheat on the (r, z) grid:

        div(k grad T) + q - w (T - T_art) = 0

    as a network: every face a thermal conductance k_face A / h, every cell a sink
    w V to T_art and a source of q V watts.  g_off = (off_r, off_z) boolean tables of
    faces whose conductance is forced to zero (used to cut the electrode from blood).
    """
    nr, nz = grid["nr"], grid["nz"]
    g_r, g_z = cd.face_conductances(k_map, grid)
    g_r = np.where(g_off[0], 0.0, g_r)
    g_z = np.where(g_off[1], 0.0, g_z)
    idx = np.arange(nr * nz).reshape(nr, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(nr * nz)

    def add(ia, ja, ib, jb, g):
        a, b = idx[ia, ja].ravel(), idx[ib, jb].ravel()
        gg = g.ravel()
        rows.extend([a, b]); cols.extend([b, a]); vals.extend([-gg, -gg])
        np.add.at(diag, a, gg); np.add.at(diag, b, gg)

    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nz), indexing="ij")
    add(ii, jj, ii + 1, jj, g_r)
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nz - 1), indexing="ij")
    add(ii, jj, ii, jj + 1, g_z)
    sink = (w_map * grid["volume"]).ravel()
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nr * nz, nr * nz)).tocsr() + sp.diags(diag + sink)
    rhs = source_watts.ravel() + sink * t_art
    f = fixed.ravel()
    fv = fixed_values.ravel()
    rhs = rhs - A[:, f] @ fv[f]
    T = np.zeros(nr * nz)
    T[f] = fv[f]
    T[~f] = spla.spsolve(A[~f][:, ~f].tocsc(), rhs[~f])
    return T.reshape(nr, nz), (g_r, g_z)


def heat_balance_fluxes(T, g_r, g_z, elec):
    """Net heat (W) conducted from the non-electrode cells INTO the electrode."""
    q = 0.0
    dr = T[:-1, :] - T[1:, :]
    a, b = elec[:-1, :], elec[1:, :]
    q += np.sum(np.where(~a & b, g_r * dr, 0.0)) + np.sum(np.where(a & ~b, -g_r * dr, 0.0))
    dz = T[:, :-1] - T[:, 1:]
    a, b = elec[:, :-1], elec[:, 1:]
    q += np.sum(np.where(~a & b, g_z * dz, 0.0)) + np.sum(np.where(a & ~b, -g_z * dz, 0.0))
    return float(q)


def ablation_thermal(settings: AblationSettings, irrigated, t_tip=T_BODY, h_tip=2000.0,
                     w_myo=W_MYO, voltage=None):
    """Temperature rise field for one electrode and one RF voltage.

    irrigated=True   the tip surface is held at t_tip (cooled by saline).
    irrigated=False  the tip is one isothermal lump that loses heat to the blood by
                     convection h_tip over its blood-exposed area; its temperature
                     is found from the heat balance.
    Blood cells are held at T_BODY (ideal convective washout - an upper bound on
    surface cooling).  Returns a dict.
    """
    s = settings
    grid = cd.build_grid(s)
    sigma = cd.conductivity_map(grid, s)
    v = cd.solve_potential(grid, sigma, s)
    elec = cd.electrode_mask(grid, s)
    far = cd.far_boundary_mask(grid, s)
    blood = (grid["Z"] >= 0.0) & ~elec & ~far
    myo = (grid["Z"] < 0.0) & ~elec & ~far
    power = cell_power_electrode_aware(v, sigma, grid, elec)

    k_map = np.where(elec, K_METAL, np.where(grid["Z"] < 0.0, K_MYO, K_BLOOD))
    w_map = np.where(myo, w_myo, 0.0)
    # cut every face that joins the electrode to a blood cell (replaced by convection)
    off_r = (elec[:-1, :] & blood[1:, :]) | (blood[:-1, :] & elec[1:, :])
    off_z = (elec[:, :-1] & blood[:, 1:]) | (blood[:, :-1] & elec[:, 1:])
    src = np.where(blood | elec, 0.0, power)               # heat deposited in blood is washed away
    src_blood_power = float(np.sum(power[blood]))

    def run(t_e):
        fixed = elec | far | blood
        vals = np.where(elec, t_e, T_BODY)
        return solve_pennes(grid, k_map, (off_r, off_z), w_map, src, fixed, vals)

    if irrigated:
        T, (g_r, g_z) = run(t_tip)
        t_e = t_tip
    else:
        T0, (g_r, g_z) = run(T_BODY)
        T1, _ = run(T_BODY + 50.0)
        q0 = heat_balance_fluxes(T0, g_r, g_z, elec)
        q1 = heat_balance_fluxes(T1, g_r, g_z, elec)
        gt = (q0 - q1) / 50.0                              # W/K, conduction back into tissue
        zc = s.electrode_centre_z
        height = float(np.clip(zc + s.electrode_radius, 0.0, 2.0 * s.electrode_radius))
        area_blood = 2.0 * np.pi * s.electrode_radius * height
        t_e = T_BODY + q0 / (gt + h_tip * area_blood)
        T, (g_r, g_z) = run(t_e)
    dT = np.where(elec, np.nan, T - T_BODY)
    tissue_dT = np.where(myo, T - T_BODY, -np.inf)
    return {"grid": grid, "T": T, "myo": myo, "elec": elec, "t_tip": t_e, "power": float(power.sum()),
            "peak_dT": float(tissue_dT.max()), "power_in_blood": src_blood_power, "dT": dT}


def iso_volume(res, dt_threshold):
    """Volume (m^3) of myocardium whose temperature rise is at least dt_threshold."""
    hot = res["myo"] & ((res["T"] - T_BODY) >= dt_threshold)
    return float(np.sum(res["grid"]["volume"][hot]))


# ===========================================================================
#  EXTENSION 3 - inductance of a navigation coil
# ===========================================================================

def mutual_filaments(a, b, d):
    """Maxwell's formula: mutual inductance of two coaxial circular filaments, H."""
    k2 = 4.0 * a * b / ((a + b) ** 2 + d ** 2)
    k = np.sqrt(k2)
    return C.MU0 * np.sqrt(a * b) * ((2.0 / k - k) * ellipk(k2) - (2.0 / k) * ellipe(k2))


def self_filament(a, gmd):
    """Self-inductance of a thin ring of radius a whose cross-section has self-GMD gmd, H."""
    return C.MU0 * a * (np.log(8.0 * a / gmd) - 2.0)


def coil_inductances(coil: CoilPair, m=8):
    """(L_one_coil, M_between_coils), henry, for N-turn bundles of square section.

    The bundle (side s = sqrt(N * wire_area)) is cut into m x m filament cells, each
    carrying N/m^2 turns.  L = (N/m^2)^2 * sum_ij M_ij over all cell pairs, using
    Maxwell's elliptic formula for i != j and the square's self-GMD (0.44705 c) for
    each cell with itself.
    """
    s = np.sqrt(coil.turns * coil.wire_area)
    c = s / m
    off = (np.arange(m) - 0.5 * (m - 1)) * c
    R, Z = np.meshgrid(coil.radius + off, off, indexing="ij")
    R, Z = R.ravel(), Z.ravel()
    n = R.size
    scale = (coil.turns / n) ** 2
    L = 0.0
    M = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                L += self_filament(R[i], 0.44705 * c)
            else:
                L += mutual_filaments(R[i], R[j], Z[i] - Z[j])
            M += mutual_filaments(R[i], R[j], Z[i] - Z[j] + coil.separation)
    return scale * L, scale * M


# ===========================================================================
#  EXTENSION 4 - flexible shaft
# ===========================================================================

def beam_equilibrium(phi, b_magnitude, ei_of_s, length, magnet=None, n=2001):
    """Quasi-static shape of a cantilever shaft with the tip magnet's torque at its end.

    In a UNIFORM field the net force on the magnet is zero (Eq. 5.11), so the only
    load is the torque T = m B sin(phi - theta_tip) at the free end.  A pure end
    moment gives bending moment T at every section, so curvature is T/EI(s):

        theta(s) = T * integral_0^s ds'/EI,    tip angle = T / k_eff,
        k_eff = 1 / integral_0^L ds/EI.

    The tip angle then obeys k_eff*theta = m B sin(phi - theta), the same scalar
    equation as Milestone 2.  Returns s, theta(s), x(s), z(s) (x lateral, z along the
    undeflected shaft) and the tip angle.
    """
    magnet = magnet or Catheter().magnet
    m = magnet.moment_magnitude
    s = np.linspace(0.0, length, n)
    inv = 1.0 / ei_of_s(s)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))])
    k_eff = 1.0 / cum[-1]
    if phi <= 0:
        th_tip = 0.0
    else:
        th_tip = brentq(lambda t: k_eff * t - m * b_magnitude * np.sin(phi - t), 0.0, phi, xtol=1e-14)
    torque = m * b_magnitude * np.sin(phi - th_tip)
    theta = torque * cum
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * np.diff(s))])
    z = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * np.diff(s))])
    return {"s": s, "theta": theta, "x": x, "z": z, "tip_angle": float(theta[-1]),
            "k_eff": k_eff, "torque": float(torque)}
