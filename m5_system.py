"""Milestone 5 — chain the four links and sweep the design."""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import conduction as cd
from emlab import steering as st
from emlab import thermal as th
from emlab import plotting as P
from emlab.geometry import AblationSettings, Catheter, CoilPair, seating_depth

CELLS = 12
cat = Catheter()
commands = np.radians(np.array([15.0, 30.0, 45.0, 60.0, 75.0]))
fields = [0.040, 0.080, 0.160]


def chain(commanded, b_magnitude):
    reached = st.equilibrium_tilt(commanded, b_magnitude, cat)
    error = commanded - reached
    depth = float(seating_depth(error))
    out = cd.solve(AblationSettings(depth=depth, cells_per_tip_radius=CELLS))
    return error, depth, out


results = {}
for b in fields:
    rows = [chain(c, b) for c in commands]
    results[b] = rows
    print(f"\nB = {b*1e3:.0f} mT")
    print("  cmd(deg)  error(deg)  depth(mm)   Z(ohm)   tissue(%)   lesion(mm^3)")
    for c, (err, depth, out) in zip(commands, rows):
        print(f"   {np.degrees(c):5.0f}     {np.degrees(err):6.1f}      {depth*1e3:5.2f}"
              f"     {out['impedance']:6.1f}     {out['fraction_myocardium']*100:5.1f}"
              f"        {out['lesion_volume']*1e9:6.1f}")
    les = [o["lesion_volume"] * 1e9 for _, _, o in rows]
    print(f"  lesion {les[0]:.0f} -> {les[-1]:.0f} mm^3 across the sweep "
          f"(worst case {min(les):.0f})")

print("\nWhat the field costs at each level (per coil):")
for b in fields:
    scale = b / 0.0796
    coil = CoilPair(current=CoilPair().current * scale)
    print(f"  {b*1e3:5.0f} mT  ->  I = {coil.current:5.1f} A   "
          f"P = {th.coil_dissipation(coil)/1e3:5.2f} kW   "
          f"F = {th.force_between_coils(coil, 240):6.0f} N")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for b, col in zip(fields, P.SERIES):
    rows = results[b]
    axes[0].plot(np.degrees(commands), [d * 1e3 for _, d, _ in rows], "o-",
                 lw=2.1, color=col, label=f"{b*1e3:.0f} mT")
    axes[1].plot(np.degrees(commands), [o["lesion_volume"] * 1e9 for _, _, o in rows],
                 "o-", lw=2.1, color=col, label=f"{b*1e3:.0f} mT")
P.style(axes[0], "commanded angle  (deg)", "seating depth  (mm)",
        "(a)  aiming error costs contact")
axes[0].legend(frameon=False, fontsize=9)
P.style(axes[1], "commanded angle  (deg)", "lesion volume  (mm$^3$)",
        "(b)  and contact is what makes the lesion")
axes[1].legend(frameon=False, fontsize=9)
P.save(fig, FIGDIR / "m5_system.png")
