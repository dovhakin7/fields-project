"""Milestone 3 — price the field: dissipation, current density, coil force."""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import thermal as th
from emlab import plotting as P
from emlab.geometry import CoilPair

coil = CoilPair()
print(f"conductor length per coil : {coil.wire_length:.1f} m")
print(f"resistance per coil       : {th.coil_resistance(coil)*1e3:.2f} mohm")
print(f"dissipation per coil      : {th.coil_dissipation(coil)/1e3:.3f} kW   "
      f"(both coils {2*th.coil_dissipation(coil)/1e3:.2f} kW)")
print(f"current density           : {th.current_density(coil)/1e6:.2f} A/mm^2")
print(f"force between the coils   : {th.force_between_coils(coil):.1f} N (attractive)")
print(f"lumped free-convection dT : {th.temperature_rise(coil):.0f} K  -> cannot run uncooled")
print(f"Ulaby 4-14 check          : {th.wire_resistance(50, np.pi*0.02**2)*1e4:.2f}e-4 ohm")
print(f"ampere definition check   : {th.force_per_length_parallel_wires(1,1,1):.2e} N/m")

areas = np.logspace(np.log10(2e-5), np.log10(1e-3), 60)
power = np.array([th.coil_dissipation(CoilPair(wire_area=a)) for a in areas])
dens = coil.current / areas
currents = np.linspace(50, 400, 40)
force = np.array([th.force_between_coils(CoilPair(current=i), n_seg=240) for i in currents])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
axes[0].loglog(areas * 1e6, power / 1e3, lw=2.2, color=P.BLUE)
axes[0].plot(coil.wire_area * 1e6, th.coil_dissipation(coil) / 1e3, "o", color=P.NAVY)
P.style(axes[0], "conductor cross-section  (mm$^2$)", "dissipation per coil  (kW)",
        "(a)  heat against copper")
axes[1].loglog(areas * 1e6, dens / 1e6, lw=2.2, color=P.ORANGE)
axes[1].axhline(4.0, color=P.MUTED, ls="--", lw=1.4)
axes[1].text(30, 4.4, "4 A/mm$^2$ practical ceiling", fontsize=8.5, color=P.MUTED)
axes[1].plot(coil.wire_area * 1e6, th.current_density(coil) / 1e6, "o", color=P.NAVY)
P.style(axes[1], "conductor cross-section  (mm$^2$)", "current density  (A/mm$^2$)",
        "(b)  manufacturability")
axes[2].plot(currents, force, lw=2.2, color=P.GREEN)
axes[2].plot(coil.current, th.force_between_coils(coil, 240), "o", color=P.NAVY)
P.style(axes[2], "per-turn current  (A)", "attraction between coils  (N)",
        "(c)  force follows I$^2$, Eq. (5.41)")
P.save(fig, FIGDIR / "m3_coil_budget.png")
