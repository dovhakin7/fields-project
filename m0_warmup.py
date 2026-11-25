"""MILESTONE 0 — run this first.  Nothing here is graded.

    python scripts/m0_warmup.py

It uses the finished example in emlab/warmup.py to show you (a) that your
Python installation works, (b) what the Biot-Savart sum actually does, and
(c) how a numerical answer is checked against a closed form.

There are three TODOs at the bottom.  They take two minutes each and they are
the whole point of the exercise.
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np
import matplotlib.pyplot as plt

from emlab import plotting as P
from emlab.warmup import wire_nodes, wire_field, long_wire_closed_form

CURRENT = 10.0        # A
LENGTH = 2.0          # m, the wire runs from z = -1 m to z = +1 m
N_SEG = 200           # how many straight pieces we chop it into

# Build the wire, then ask for the field at four distances from it.
seg_mid, seg_vec = wire_nodes(LENGTH, N_SEG)
print(f"the wire was cut into {seg_mid.shape[0]} pieces, "
      f"each described by {seg_mid.shape[1]} numbers (x, y, z)")

radii = np.array([0.02, 0.05, 0.10, 0.20])            # metres from the wire
points = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
B = wire_field(points, seg_mid, seg_vec, CURRENT)

print("\n  r (mm)   |B| computed (uT)   mu0 I/2pi r (uT)   difference")
for r, b in zip(radii, B):
    numeric = np.linalg.norm(b) * 1e6
    closed = long_wire_closed_form(r, CURRENT) * 1e6
    print(f"   {r*1e3:5.0f}        {numeric:8.3f}          {closed:8.3f}"
          f"        {100*(numeric/closed-1):+6.2f} %")

print("\nThe field points along +y at a point on the +x axis, as the right-hand")
print(f"rule says: B at r = 20 mm is ({B[0,0]:+.2e}, {B[0,1]:+.2e}, {B[0,2]:+.2e}) T")

# A picture of the same thing.
r_plot = np.linspace(0.01, 0.30, 60)
pts = np.stack([r_plot, np.zeros_like(r_plot), np.zeros_like(r_plot)], axis=1)
mag = np.linalg.norm(wire_field(pts, seg_mid, seg_vec, CURRENT), axis=1)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(r_plot * 1e3, mag * 1e6, lw=2.2, color=P.BLUE, label="2 m wire, computed")
ax.plot(r_plot * 1e3, long_wire_closed_form(r_plot, CURRENT) * 1e6, lw=2.0,
        ls="--", color=P.ORANGE, label="infinite wire, mu0 I / 2 pi r")
P.style(ax, "distance from the wire  (mm)", "|B|  (uT)",
        "Milestone 0: a finite wire against the textbook result")
ax.legend(frameon=False, fontsize=9)
P.save(fig, FIGDIR / "m0_warmup.png")

print("""
TODO 1.  Change N_SEG at the top of this file from 200 to 4 and run it again.
         The wire is now a crude 4-piece zig-zag.  How big does the error get?
         Put it back to 200 afterwards.

TODO 2.  Change LENGTH from 2.0 to 0.2 m and run again.  The closed form is for
         an INFINITE wire, so the computed field should now fall short of it.
         At which radius does the disagreement pass 10 %?

TODO 3.  Open emlab/warmup.py and find step 4 of wire_field.  Swap the two
         arguments of np.cross so it reads np.cross(R, seg_vec[None, :, :]).
         Run again: every sign flips, because A x B = -(B x A).  Put it back.
         This is the single most common bug in Milestone 1.
""")
