"""Milestone 2 — how far the field can actually bend the tip."""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import steering as st
from emlab import plotting as P
from emlab.geometry import TipMagnet, Catheter

magnet, cat = TipMagnet(), Catheter()
print(f"tip magnet: r = {magnet.radius*1e3:.1f} mm, L = {magnet.length*1e3:.1f} mm, "
      f"m = {magnet.moment_magnitude*1e3:.2f} mA.m^2")
print(f"shaft stiffness k = {cat.shaft_stiffness*1e3:.2f} mN.m/rad")

# --- the two physics checks, printed so they appear in the report ------------
mid, vec, cur = st.equivalent_loop(magnet, tilt=0.5, n_seg=720)
B = np.array([0.0, 0.0, 0.0796])
F = st.force_on_loop(mid, vec, cur, B)
T_loop = st.torque_on_loop(mid, vec, cur, B)
T_dip = np.cross(st.moment_vector(magnet, (np.sin(0.5), 0, np.cos(0.5))), B)
print(f"net force in a uniform field : {np.linalg.norm(F):.3e} N  (must be ~0)")
print(f"torque, loop integral        : {T_loop[1]*1e3:+.4f} mN.m")
print(f"torque, m x B                : {T_dip[1]*1e3:+.4f} mN.m  "
      f"(agree to {abs(T_loop[1]/T_dip[1]-1)*1e3:.2f} parts in 10^3)")
print(f"m|B| at the baseline field   : {magnet.moment_magnitude*0.0796*1e3:.2f} mN.m")

# --- commanded versus achieved ----------------------------------------------
cmd = np.radians(np.linspace(0, 90, 91))
fields = [0.040, 0.080, 0.160]
print("\ncommanded -> achieved (deg):")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for B0, col in zip(fields, P.SERIES):
    reached = np.array([st.equilibrium_tilt(c, B0, cat) for c in cmd])
    axes[0].plot(np.degrees(cmd), np.degrees(reached), lw=2.1, color=col,
                 label=f"{B0*1e3:.0f} mT")
    print(f"  {B0*1e3:5.0f} mT :  30 -> {np.degrees(st.equilibrium_tilt(np.radians(30),B0,cat)):5.1f}"
          f"   60 -> {np.degrees(st.equilibrium_tilt(np.radians(60),B0,cat)):5.1f}"
          f"   75 -> {np.degrees(st.equilibrium_tilt(np.radians(75),B0,cat)):5.1f}")
axes[0].plot(np.degrees(cmd), np.degrees(cmd), lw=1.2, ls="--", color=P.MUTED,
             label="ideal")
P.style(axes[0], "commanded angle  (deg)", "achieved angle  (deg)",
        "(a)  the tip undershoots, and the shortfall grows")
axes[0].legend(frameon=False, fontsize=9)

theta = np.radians(np.linspace(0, 60, 200))
for B0, col in zip(fields, P.SERIES):
    axes[1].plot(np.degrees(theta),
                 magnet.moment_magnitude * B0 * np.sin(np.radians(60) - theta) * 1e3,
                 lw=2.0, color=col, label=f"magnetic, {B0*1e3:.0f} mT")
axes[1].plot(np.degrees(theta), cat.shaft_stiffness * theta * 1e3, lw=2.0,
             color=P.MUTED, ls="--", label="elastic  k$\\theta$")
P.style(axes[1], "tip angle  (deg)", "torque  (mN.m)",
        "(b)  equilibrium is where the curves cross (60$^\\circ$ command)")
axes[1].legend(frameon=False, fontsize=9)
P.save(fig, FIGDIR / "m2_steering.png")
