"""Bonus extensions 

    python scripts/m7_extensions.py

1  three-axis coil set: steering authority and field quality over the working volume
2  irrigated electrode: Pennes bioheat beside the conduction solve
3  coil inductance and the supply voltage to slew the field in 100 ms
4  flexible catheter shaft: tip trajectory, not just the final angle
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import constants as C
from emlab import conduction as cd
from emlab import extensions as ex
from emlab import steering as st
from emlab import thermal as th
from emlab import plotting as P
from emlab.geometry import AblationSettings, Catheter, CoilPair


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


cat = Catheter()
coil = CoilPair()
I_MAX = coil.current                      # per-axis limit: the baseline 200 A per turn
B_AX = ex.centre_tesla_per_amp(coil) * I_MAX

# =========================================================================
hdr("1. THREE-AXIS COIL SET")
# =========================================================================
coils_same = (coil, coil, coil)
print(f"  one pair, 200 A: {B_AX*1e3:.2f} mT along its own axis (the baseline).")
print("  Three IDENTICAL pairs on x, y, z (idealised: real coils of equal radius would collide).")
print("  Largest field the set can put at the centre along a direction, before any axis passes 200 A:")
for name, n in (("along an axis", (0, 0, 1)), ("face diagonal", (1, 1, 0)), ("cube diagonal", (1, 1, 1))):
    print(f"    {name:14s}: {ex.max_field_along(n, coils_same, I_MAX)*1e3:6.1f} mT")

print("\n  Steering authority at the centre: tip reached for a 60 deg command, at the largest field available")
phi = np.radians(60.0)
psis = np.radians(np.arange(0, 91, 15))
auth = []
for psi in psis:
    n = np.array([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])
    bmax = ex.max_field_along(n, coils_same, I_MAX)
    reached = np.degrees(st.equilibrium_tilt(phi, bmax, cat))
    auth.append((np.degrees(psi), bmax, reached))
    print(f"    azimuth {np.degrees(psi):3.0f} deg:  B_max {bmax*1e3:6.1f} mT  ->  tip reaches {reached:5.1f} deg")
print("  (the baseline single pair cannot steer sideways at all; it only sets |B| along z)")


def volume_map(coils, half, n, command, b_mag):
    xs = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    basis = ex.triad_basis(np.vstack([[0.0, 0.0, 0.0], pts]), coils, n_seg=240)    # (3, M+1, 3)
    cur = ex.currents_for_field(b_mag * np.asarray(command) / np.linalg.norm(command), coils)
    B = np.einsum("i,imk->mk", cur, basis)
    Bc, Bp = B[0], B[1:]
    mag = np.linalg.norm(Bp, axis=1)
    cosang = np.clip((Bp @ Bc) / (mag * np.linalg.norm(Bc)), -1, 1)
    bdir_err = np.degrees(np.arccos(cosang))
    tdir, tilt = ex.tip_direction(np.vstack([Bc, Bp]), cat)
    t0, tilt0 = tdir[0], tilt[0]
    tp, tiltp = tdir[1:], tilt[1:]
    tip_err = np.degrees(np.arccos(np.clip(tp @ t0, -1, 1)))
    return {"pts": pts, "cur": cur, "mag": mag, "Bc": Bc, "bdir_err": bdir_err, "tilt": np.degrees(tiltp),
            "tilt0": np.degrees(tilt0), "tip_err": tip_err}


cmd = np.array([np.sin(phi) * np.cos(np.radians(45)), np.sin(phi) * np.sin(np.radians(45)), np.cos(phi)])
print("\n  Field quality over the volume, command = 60 deg tilt at azimuth 45 deg, 79.6 mT at the centre:")
nested = (CoilPair(radius=0.53, separation=0.53), CoilPair(radius=0.44, separation=0.44), CoilPair(radius=0.35, separation=0.35))
results = {}
for label, cs in (("three identical pairs (a = 350 mm)", coils_same), ("nested pairs (a = 530/440/350 mm for x/y/z)", nested)):
    print(f"   {label}")
    for half, n in ((0.05, 7), (0.10, 9)):
        r = volume_map(cs, half, n, cmd, 0.0796)
        results[(label, half)] = r
        print(f"     +-{half*1e3:3.0f} mm cube: |B| spread {100*(r['mag'].max()-r['mag'].min())/np.linalg.norm(r['Bc']):.3f} %, "
              f"field direction error max {r['bdir_err'].max():.3f} deg, "
              f"tip tilt {r['tilt'].min():.2f}-{r['tilt'].max():.2f} deg (centre {r['tilt0']:.2f}), "
              f"tip pointing error max {r['tip_err'].max():.3f} deg")
    print(f"     currents for this command: Ix = {r['cur'][0]:.0f} A, Iy = {r['cur'][1]:.0f} A, Iz = {r['cur'][2]:.0f} A")

lab0 = "three identical pairs (a = 350 mm)"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
fig.subplots_adjust(wspace=0.6)
axes[0].plot([a[0] for a in auth], [a[2] for a in auth], "o-", color=P.BLUE, lw=2, label="tip reached")
axes[0].axhline(60, color=P.MUTED, ls="--", lw=1.2, label="commanded 60$^\\circ$")
ax2 = axes[0].twinx()
ax2.plot([a[0] for a in auth], [a[1] * 1e3 for a in auth], "s--", color=P.ORANGE, lw=1.6)
ax2.set_ylabel("largest field available  (mT)", color=P.ORANGE, fontsize=10)
P.style(axes[0], "azimuth of the commanded direction  (deg)", "tip angle reached  (deg)",
        "(a)  steering authority depends on direction")
axes[0].legend(frameon=False, fontsize=9, loc="center left")
r = results[(lab0, 0.10)]
sel = np.isclose(r["pts"][:, 2], 0.0)
xs = np.unique(r["pts"][:, 0])
Zerr = r["tip_err"][sel].reshape(len(xs), len(xs))
im = axes[1].pcolormesh(xs * 1e3, xs * 1e3, Zerr.T, cmap="Blues", shading="auto")
fig.colorbar(im, ax=axes[1], label="tip pointing error  (deg)")
axes[1].add_patch(plt.Rectangle((-50, -50), 100, 100, fill=False, ec=P.ORANGE, lw=1.6, ls="--"))
axes[1].set_aspect("equal")
P.style(axes[1], "x  (mm)", "y  (mm)", "(b)  pointing error in the z = 0 plane")
P.save(fig, FIGDIR / "ext1_three_axis.png")

# =========================================================================
hdr("2. IRRIGATED ELECTRODE (Pennes bioheat beside the conduction solve)")
# =========================================================================
print("  Steady Pennes: div(k grad T) + q - w (T - 37) = 0 on the same (r, z) grid; q from the conduction solve.")
print("  Blood held at 37 C (ideal washout); myocardium w = 4e4 W/m^3K; k_myo 0.56, k_blood 0.52 W/mK (nominal, from memory).")
print("  Non-irrigated tip: one isothermal lump, convection h = 2000 W/m^2K to blood over its blood-exposed area.")
print("  Irrigated tip: surface held at 37 C.  Both solved at 25 V, then scaled (the problem is linear in V^2).")
T_LIM, T_LES = 100.0, 50.0               # peak tissue temperature limit (steam pops), lesion isotherm
print(f"  Comparison rule: raise the RF voltage until the hottest tissue cell reaches {T_LIM:.0f} C, then count tissue >= {T_LES:.0f} C.\n")
print("   depth  tip       Z(ohm)  P@25V(W)  tip T@25V  peak dT@25V   V_max(V)  P_max(W)  lesion@limit(mm^3)  proxy lesion@25V")
rows = {}
for d in (1.0e-3, 1.66e-3, 2.0e-3):
    s = AblationSettings(depth=d, cells_per_tip_radius=12)
    base = cd.solve(s)
    for name, irr in (("bare", False), ("irrigated", True)):
        r = ex.ablation_thermal(s, irr)
        scale = (T_LIM - ex.T_BODY) / r["peak_dT"]
        vmax = s.voltage * np.sqrt(scale)
        pmax = r["power"] * scale
        les = ex.iso_volume(r, (T_LES - ex.T_BODY) / scale)
        rows[(d, name)] = (vmax, pmax, les * 1e9, r["peak_dT"], r["t_tip"])
        print(f"   {d*1e3:4.2f}  {name:9s} {base['impedance']:6.1f}  {r['power']:7.2f}   {r['t_tip']:7.1f}    {r['peak_dT']:8.0f}     "
              f"{vmax:6.1f}   {pmax:6.1f}      {les*1e9:7.1f}           {base['lesion_volume']*1e9:6.1f}")
print("  -> at the same 100 C peak-temperature limit the irrigated tip carries more power and heats more tissue above 50 C.")
for irr, nm in ((False, "bare"), (True, "irrigated")):
    s_h = AblationSettings(depth=2.0e-3, cells_per_tip_radius=12)
    r_ = ex.ablation_thermal(s_h, irr)
    g_ = r_["grid"]
    Tm = np.where(r_["myo"], r_["T"], -np.inf)
    i_, j_ = np.unravel_index(np.argmax(Tm), Tm.shape)
    gap = np.hypot(g_["r"][i_], g_["z"][j_] - s_h.electrode_centre_z) - s_h.electrode_radius
    print(f"     {nm:9s} hottest tissue cell at r = {g_['r'][i_]*1e3:.2f} mm, z = {g_['z'][j_]*1e3:.2f} mm, "
          f"{gap*1e3:.2f} mm outside the tip surface")
print("     (The hot spot is subsurface, not at the tip: holding the tip at 37 C lowers that peak for a given power, but I have")
print("      not separated this from the other model choices, so treat the size of the gain as model-dependent.)")

print("\n  Grid dependence (hemisphere contact, 12 vs 24 cells per radius):")
for n in (12, 24):
    s = AblationSettings(depth=2.0e-3, cells_per_tip_radius=n)
    out = []
    for irr in (False, True):
        r = ex.ablation_thermal(s, irr)
        scale = (T_LIM - ex.T_BODY) / r["peak_dT"]
        out.append((r["peak_dT"], ex.iso_volume(r, (T_LES - ex.T_BODY) / scale) * 1e9, r["power"] * scale))
    print(f"    {n:2d} cells: bare peak dT {out[0][0]:6.0f} K, lesion {out[0][1]:6.1f} mm^3 at {out[0][2]:5.1f} W;  "
          f"irrigated peak dT {out[1][0]:6.0f} K, lesion {out[1][1]:6.1f} mm^3 at {out[1][2]:5.1f} W")
print("  Perfusion sensitivity (hemisphere, irrigated, 12 cells; lesion at the 100 C limit):")
s = AblationSettings(depth=2.0e-3, cells_per_tip_radius=12)
for f in (0.25, 1.0, 4.0):
    r = ex.ablation_thermal(s, True, w_myo=ex.W_MYO * f)
    scale = (T_LIM - ex.T_BODY) / r["peak_dT"]
    print(f"    perfusion x{f:4.2f}: lesion {ex.iso_volume(r, (T_LES - ex.T_BODY) / scale)*1e9:6.1f} mm^3, V_max {s.voltage*np.sqrt(scale):5.1f} V")

# figure: temperature fields, hemisphere contact, scaled to the 100 C limit
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
fig.subplots_adjust(wspace=0.45)
for ax, irr, ttl in ((axes[0], False, "(a)  bare tip"), (axes[1], True, "(b)  irrigated tip (37 C)")):
    r = ex.ablation_thermal(s, irr)
    scale = (T_LIM - ex.T_BODY) / r["peak_dT"]
    g = r["grid"]
    nr = int(0.010 / g["h"]); nz0 = g["nz"] // 2
    T = (ex.T_BODY + scale * (r["T"] - ex.T_BODY))
    Tw = T[:nr, nz0 - nr:nz0 + nr]
    rr = g["r"][:nr] * 1e3
    zz = g["z"][nz0 - nr:nz0 + nr] * 1e3
    full = np.concatenate([Tw[::-1], Tw])
    im = ax.pcolormesh(np.concatenate([-rr[::-1], rr]), zz, full.T, cmap="Blues", vmin=37, vmax=100, shading="auto")
    ax.contour(np.concatenate([-rr[::-1], rr]), zz, full.T, levels=[50.0], colors=[P.ORANGE], linewidths=1.6)
    ax.axhline(0, color=P.NAVY, lw=1.0)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="T  (C)")
    P.style(ax, "r  (mm)", "z  (mm)", ttl)
P.save(fig, FIGDIR / "ext2_irrigated.png")

# =========================================================================
hdr("3. COIL INDUCTANCE AND SLEW VOLTAGE")
# =========================================================================
L, M = ex.coil_inductances(coil, m=8)
L12, _ = ex.coil_inductances(coil, m=12)
side = np.sqrt(coil.turns * coil.wire_area)
thin = C.MU0 * coil.turns ** 2 * coil.radius * (np.log(8 * coil.radius / (0.44705 * side)) - 2)
Rc = th.coil_resistance(coil)
print(f"  bundle {side*1e3:.0f} mm square, a = {coil.radius*1e3:.0f} mm, N = {coil.turns}")
print(f"  self-inductance of one coil: {L*1e3:.2f} mH (8x8 filament cells; 12x12 gives {L12*1e3:.2f} mH)")
print(f"    closed-form GMD estimate  : {thin*1e3:.2f} mH ({100*(thin/L-1):+.1f} %: the thin-ring formula, bundle is not thin: side/a = {side/coil.radius:.2f})")
print(f"  mutual inductance between the coils at Helmholtz spacing: {M*1e3:.2f} mH, coupling k = {M/L:.3f}")
Ltot = 2 * (L + M)
Rtot = 2 * Rc
print(f"  pair in series, field-aiding: L_total = 2(L + M) = {Ltot*1e3:.1f} mH, R_total = {Rtot*1e3:.1f} mohm, L/R = {Ltot/Rtot:.2f} s")
for label, cc, di in (("baseline 79.6 mT, 0 -> full", coil, 1.0), ("baseline 79.6 mT, reverse (-I -> +I)", coil, 2.0),
                      ("recommended 100 mT, 0 -> full", CoilPair(current=0.100 / (ex.centre_tesla_per_amp(coil))), 1.0),
                      ("recommended 100 mT, reverse", CoilPair(current=0.100 / (ex.centre_tesla_per_amp(coil))), 2.0)):
    dIdt = di * cc.current / 0.100
    vL = Ltot * dIdt
    vR = Rtot * cc.current
    print(f"  {label:38s}: dI/dt = {dIdt:6.0f} A/s, V_L = {vL:6.1f} V, V_R(end) = {vR:5.1f} V, supply >= {vL+vR:6.1f} V, "
          f"peak power {(vL+vR)*cc.current/1e3:5.1f} kW")
Wm = 0.5 * Ltot * coil.current ** 2
print(f"  stored energy at 200 A: {Wm:.0f} J (a 100 ms slew moves that, plus the 2.9 kW ohmic loss, through the supply)")
print(f"  time constant L/R is {Ltot/Rtot:.2f} s, so a resistive-limited ramp is far slower than 100 ms; the slew needs a voltage")
print(f"  of order {Ltot*coil.current/0.1:.0f} V (inductive) - about {Ltot*coil.current/0.1/(Rtot*coil.current):.0f}x the resistive drop.")
t = np.linspace(0, 0.1, 200)
Ir = coil.current * t / 0.1
fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.plot(t * 1e3, Ltot * coil.current / 0.1 * np.ones_like(t), color=P.BLUE, lw=2, label="inductive  L dI/dt")
ax.plot(t * 1e3, Rtot * Ir, color=P.ORANGE, lw=2, label="resistive  R I")
ax.plot(t * 1e3, Ltot * coil.current / 0.1 + Rtot * Ir, color=P.NAVY, lw=2.2, ls="--", label="supply voltage")
P.style(ax, "time  (ms)", "voltage  (V)", "linear 0 to 200 A ramp in 100 ms")
ax.legend(frameon=False, fontsize=9)
P.save(fig, FIGDIR / "ext3_slew.png")

# =========================================================================
hdr("4. FLEXIBLE SHAFT")
# =========================================================================
Ls = 0.040                                        # flexible length, m
EI_u = cat.shaft_stiffness * Ls                   # uniform: k = EI/L = 1 mN.m/rad
# soft distal 10 mm (EI_d), proximal 30 mm ten times stiffer:  int ds/EI = 0.010/EI_d + 0.030/(10 EI_d) = 0.013/EI_d = 1/k
EI_d = 0.013 * cat.shaft_stiffness
uniform = lambda s: np.full_like(s, EI_u)
softtip = lambda s: np.where(s > Ls - 0.010, EI_d, 10.0 * EI_d)
print(f"  shaft length {Ls*1e3:.0f} mm.  Uniform EI = {EI_u:.2e} N.m^2  (k_eff = EI/L = {EI_u/Ls*1e3:.2f} mN.m/rad)")
print(f"  soft-distal shaft: last 10 mm EI = {EI_d:.2e}, proximal 30 mm ten times stiffer; chosen so k_eff is also 1.00 mN.m/rad")
print("  Finding: in a uniform field the only load is the tip torque (Eq. 5.11: no net force), so the bending moment is")
print("  constant along the shaft and k_eff = 1/integral(ds/EI).  The Milestone 2 torsional spring is EXACT for tip angle;")
print("  what it cannot give is where the electrode ends up, and that depends on how EI is distributed.\n")
for name, fn in (("uniform", uniform), ("soft distal", softtip)):
    res = ex.beam_equilibrium(np.radians(60), 0.0796, fn, Ls)
    print(f"  {name:11s} 60 deg command, 79.6 mT: tip angle {np.degrees(res['tip_angle']):5.2f} deg (k_eff {res['k_eff']*1e3:.3f} mN.m/rad), "
          f"tip at x = {res['x'][-1]*1e3:5.2f} mm, z = {res['z'][-1]*1e3:5.2f} mm (straight: x = 0, z = {Ls*1e3:.0f})")
print("  Trajectory (uniform shaft) as the commanded angle is ramped 0 -> 90 deg, tip position (x, z) in mm:")
cmds = np.radians([15, 30, 45, 60, 75, 90])
print("   field(mT)  " + "   ".join(f"{np.degrees(c):4.0f} deg      " for c in cmds))
for B in (0.040, 0.080, 0.160):
    pos = [ex.beam_equilibrium(c, B, uniform, Ls) for c in cmds]
    print(f"    {B*1e3:5.0f}    " + "  ".join(f"({p['x'][-1]*1e3:5.2f},{p['z'][-1]*1e3:5.2f})" for p in pos))
dtip = ex.beam_equilibrium(np.radians(60), 0.0796, uniform, Ls)["x"][-1] - ex.beam_equilibrium(np.radians(60), 0.0796, softtip, Ls)["x"][-1]
print(f"  same tip angle, electrode lateral position differs by {abs(dtip)*1e3:.2f} mm between the two shafts")
# size of the neglected gradient force over the working volume, from the actual coil field
from emlab import magnetostatics as ms
vol = ms.working_volume_points(0.050, 5)
dh = 1.0e-3
gmax = 0.0
for ax in range(3):
    e = np.zeros(3); e[ax] = dh
    bp = np.linalg.norm(ms.coil_pair_field(vol + e, coil, n_seg=300), axis=1)
    bm = np.linalg.norm(ms.coil_pair_field(vol - e, coil, n_seg=300), axis=1)
    gmax = max(gmax, float(np.max(np.abs(bp - bm) / (2 * dh))))
f_grad = cat.magnet.moment_magnitude * gmax
f_ref = cat.shaft_stiffness * np.radians(30) / Ls          # tip torque / shaft length, N
print(f"  Limitation: uniform-field assumption. Largest |grad|B| in the +-50 mm cube is {gmax*1e3:.3f} mT/m, so the neglected force")
print(f"  m*grad|B| <= {f_grad:.1e} N, against tip torque / shaft length = {f_ref:.1e} N ({100*f_grad/f_ref:.3f} % of it).")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
fig.subplots_adjust(wspace=0.35)
cm = np.radians(np.linspace(0, 90, 46))
for B, col in zip((0.040, 0.080, 0.160), P.SERIES):
    tx = [ex.beam_equilibrium(c, B, uniform, Ls)["x"][-1] * 1e3 for c in cm]
    tz = [ex.beam_equilibrium(c, B, uniform, Ls)["z"][-1] * 1e3 for c in cm]
    axes[0].plot(tx, tz, lw=2.1, color=col, alpha=0.5)
    r60 = ex.beam_equilibrium(np.radians(60), B, uniform, Ls)
    axes[0].plot([r60["x"][-1] * 1e3], [r60["z"][-1] * 1e3], "o", ms=9, color=col,
                 label=f"{B*1e3:.0f} mT, 60$^\\circ$ command")
axes[0].plot([0], [Ls * 1e3], "k.", ms=7)
P.style(axes[0], "lateral tip position  x  (mm)", "axial tip position  z  (mm)", "(a)  one path for every field (axes not to scale)")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
for name, fn, col in (("uniform EI", uniform, P.BLUE), ("soft distal 10 mm", softtip, P.ORANGE)):
    r = ex.beam_equilibrium(np.radians(60), 0.0796, fn, Ls)
    axes[1].plot(r["x"] * 1e3, r["z"] * 1e3, lw=2.4, color=col, label=f"{name}  (tip {np.degrees(r['tip_angle']):.1f}$^\\circ$)")
axes[1].set_aspect("equal")
P.style(axes[1], "x  (mm)", "z  (mm)", "(b)  same tip angle, different shape (60$^\\circ$, 79.6 mT)")
axes[1].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
P.save(fig, FIGDIR / "ext4_flexible_shaft.png")
