"""Milestone 1 — map the navigation field and prove the routine is right."""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import constants as C
from emlab import magnetostatics as ms
from emlab import plotting as P
from emlab.geometry import CoilPair

coil = CoilPair()
print(f"coil: a = {coil.radius*1e3:.0f} mm, d = {coil.separation*1e3:.0f} mm, "
      f"N = {coil.turns}, I = {coil.current:.0f} A  ->  NI = {coil.ampere_turns:.0f} A-turns")

B0 = np.linalg.norm(ms.coil_pair_field([[0, 0, 0]], coil, n_seg=720)[0])
closed = (4 / 5) ** 1.5 * C.MU0 * coil.ampere_turns / coil.radius
print(f"centre field      : {B0*1e3:.3f} mT   closed form {closed*1e3:.3f} mT   "
      f"error {abs(B0/closed-1)*100:.3f} %")

print("segment convergence at z = 100 mm on the axis of one loop:")
target = ms.on_axis_loop_bz(0.1, coil.radius, coil.current, coil.turns)
for n in (24, 48, 96, 192, 384, 768):
    mid, vec = ms.loop_segments(coil.radius, 0.0, n)
    b = ms.biot_savart([[0, 0, 0.1]], mid, vec, coil.ampere_turns)[0, 2]
    print(f"   {n:4d} segments   error {abs(b/target-1)*100:9.5f} %")

z = np.linspace(-0.12, 0.12, 241)
pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
pair = np.linalg.norm(ms.coil_pair_field(pts, coil, n_seg=600), axis=1)
single = np.abs(ms.on_axis_loop_bz(z, coil.radius, coil.current, coil.turns))
win = np.abs(z) <= 0.050
print(f"uniformity +-50 mm: {ms.uniformity(pair[win], B0)*100:.4f} %  (single coil "
      f"{ms.uniformity(single[win], single[len(z)//2])*100:.2f} %)")
vol = ms.working_volume_points(0.050, 7)
print(f"uniformity in the +-50 mm cube: "
      f"{ms.uniformity(np.linalg.norm(ms.coil_pair_field(vol, coil, n_seg=400), axis=1), B0)*100:.3f} %")

x = np.linspace(-0.45, 0.45, 121)
X, Z = np.meshgrid(x, x, indexing="ij")
pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
mag = np.linalg.norm(ms.coil_pair_field(pts, coil, n_seg=240), axis=1).reshape(X.shape)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
im = axes[0].pcolormesh(X * 1e3, Z * 1e3, np.clip(mag * 1e3, 0, 160),
                        cmap="Blues", shading="auto")
axes[0].add_patch(plt.Rectangle((-50, -50), 100, 100, fill=False, ec=P.ORANGE,
                                lw=1.6, ls="--"))
for zc in coil.centres:
    axes[0].plot([-coil.radius*1e3, coil.radius*1e3], [zc*1e3, zc*1e3], "o",
                 color=P.NAVY, ms=5)
fig.colorbar(im, ax=axes[0], label="|B|  (mT, clipped at 160)")
axes[0].set_aspect("equal")
P.style(axes[0], "x  (mm)", "z  (mm)", "(a)  |B| in the x-z plane")
axes[1].plot(z * 1e3, pair * 1e3, lw=2.2, color=P.BLUE, label="Helmholtz pair")
axes[1].plot(z * 1e3, single * 1e3, lw=2.0, ls="--", color=P.ORANGE,
             label="single coil, same NI")
axes[1].axvspan(-50, 50, color=P.GREEN, alpha=0.08)
P.style(axes[1], "z on axis  (mm)", "|B|  (mT)",
        "(b)  on-axis profile, flat to 0.05 % over +-50 mm")
axes[1].legend(frameon=False, fontsize=9)
P.save(fig, FIGDIR / "m1_navigation_field.png")
