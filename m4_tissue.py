"""Milestone 4 — impedance, power split, lesion, and the refinement study."""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import conduction as cd
from emlab import plotting as P
from emlab.geometry import AblationSettings

base = AblationSettings(depth=2.0e-3)
closed = cd.analytic_resistance(base, straddling=True)
print(f"closed form, sphere on the interface : {closed:.2f} ohm")
print("grid refinement (staircase error is first order):")
errs = {}
for n in (6, 12, 24):
    s = AblationSettings(depth=2.0e-3, cells_per_tip_radius=n)
    out = cd.solve(s)
    errs[n] = out["impedance"] / closed - 1.0
    print(f"   {n:2d} cells/radius  h = {s.cell_size*1e3:.4f} mm   "
          f"Z = {out['impedance']:6.2f} ohm   error {errs[n]*100:5.2f} %")
rich = errs[24] - (errs[12] - errs[24])          # Richardson, first order
print(f"   Richardson extrapolation -> residual error about {abs(rich)*100:.2f} %")
print(f"   power split at hemisphere contact: {cd.solve(base)['fraction_myocardium']*100:.1f} % "
      f"to myocardium (sigma ratio {base.sigma_myocardium/(base.sigma_myocardium+base.sigma_blood)*100:.1f} %)")

depths = np.linspace(0.0, 4.0e-3, 17)
Z, frac, les = [], [], []
for d in depths:
    o = cd.solve(AblationSettings(depth=float(d), cells_per_tip_radius=12))
    Z.append(o["impedance"]); frac.append(o["fraction_myocardium"])
    les.append(o["lesion_volume"] * 1e9)
print(f"\nimpedance {Z[0]:.0f} -> {Z[-1]:.0f} ohm,  tissue share "
      f"{frac[0]*100:.0f} -> {frac[-1]*100:.0f} %,  lesion {les[0]:.0f} -> {les[-1]:.0f} mm^3")

s = AblationSettings(depth=2.0e-3, cells_per_tip_radius=12)
g = cd.build_grid(s); sig = cd.conductivity_map(g, s)
v = cd.solve_potential(g, sig, s); p = cd.power_density(v, sig, g, s)

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
win = (g["R"] < 0.008) & (np.abs(g["Z"]) < 0.008)
nr = int(0.008 / g["h"]); nz0 = g["nz"] // 2
rr = g["r"][:nr] * 1e3
zz = g["z"][nz0 - nr:nz0 + nr] * 1e3
im = axes[0].pcolormesh(np.concatenate([-rr[::-1], rr]), zz,
                        np.log10(np.maximum(np.concatenate(
                            [p[:nr, nz0-nr:nz0+nr][::-1], p[:nr, nz0-nr:nz0+nr]]).T, 1e2)),
                        cmap="Blues", shading="auto")
axes[0].axhline(0, color=P.ORANGE, lw=1.4)
fig.colorbar(im, ax=axes[0], label="log$_{10}$ p  (W/m$^3$)")
axes[0].set_aspect("equal")
P.style(axes[0], "r  (mm)", "z  (mm)", "(a)  power density, hemisphere contact")
axes[1].plot(depths * 1e3, Z, lw=2.2, color=P.BLUE)
P.style(axes[1], "contact depth  (mm)", "impedance  (ohm)", "(b)  Z rises on contact")
axes[2].plot(depths * 1e3, 100 * np.array(frac), lw=2.2, color=P.ORANGE)
P.style(axes[2], "contact depth  (mm)", "power into myocardium  (%)", "(c)  where the power goes")
axes[3].plot(depths * 1e3, les, lw=2.2, color=P.GREEN)
P.style(axes[3], "contact depth  (mm)", "lesion volume  (mm$^3$)", "(d)  lesion against seating")
P.save(fig, FIGDIR / "m4_tissue.png")
