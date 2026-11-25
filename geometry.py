"""Device and tissue descriptions.  PROVIDED — do not edit.

These are "dataclasses": containers that hold the numbers describing a device.
If MATLAB is your background, think of them as structs that come with default
values already filled in.

    from emlab.geometry import CoilPair
    coil = CoilPair()              # the baseline design
    print(coil.radius)             # 0.35
    strong = CoilPair(current=400) # same coil, twice the current
"""

from dataclasses import dataclass
import numpy as np

from . import constants as C


@dataclass
class CoilPair:
    """Two coaxial multi-turn coils on the z axis, centred on the origin."""
    radius: float = C.COIL_RADIUS
    separation: float = C.COIL_SEPARATION
    turns: int = C.COIL_TURNS
    current: float = C.COIL_CURRENT
    wire_area: float = C.COIL_WIRE_AREA

    @property
    def ampere_turns(self) -> float:
        """N*I — what sets the field."""
        return self.turns * self.current

    @property
    def wire_length(self) -> float:
        """Conductor length in ONE coil, m."""
        return self.turns * 2.0 * np.pi * self.radius

    @property
    def centres(self):
        """z coordinates of the two coil planes."""
        return -0.5 * self.separation, +0.5 * self.separation


@dataclass
class TipMagnet:
    """Cylindrical NdFeB slug in the catheter tip, magnetised along its axis."""
    radius: float = C.TIP_MAGNET_RADIUS
    length: float = C.TIP_MAGNET_LENGTH
    remanence: float = C.BR_NDFEB

    @property
    def volume(self) -> float:
        return np.pi * self.radius ** 2 * self.length

    @property
    def magnetisation(self) -> float:
        """M = Br/mu0, A/m."""
        return self.remanence / C.MU0

    @property
    def moment_magnitude(self) -> float:
        """|m| = M * V, A*m^2  (about 13 mA*m^2 for the baseline slug)."""
        return self.magnetisation * self.volume


@dataclass
class Catheter:
    """Tip magnet plus the shaft that resists being bent."""
    magnet: TipMagnet = None
    shaft_stiffness: float = C.SHAFT_STIFFNESS

    def __post_init__(self):
        if self.magnet is None:
            self.magnet = TipMagnet()


@dataclass
class AblationSettings:
    """Spherical electrode straddling a plane blood/myocardium interface.

    ``depth`` is how far the centre of the electrode has been pushed past the
    tissue surface:  depth = 0 leaves the sphere just touching and entirely in
    blood, depth = electrode_radius is hemispherical contact, and
    depth = 2*electrode_radius buries it completely in myocardium.
    """
    electrode_radius: float = C.ELECTRODE_RADIUS
    depth: float = C.ELECTRODE_RADIUS
    voltage: float = C.ABLATION_VOLTAGE
    sigma_myocardium: float = C.SIGMA_MYOCARDIUM
    sigma_blood: float = C.SIGMA_BLOOD
    domain_radius: float = C.DOMAIN_RADIUS
    domain_half_height: float = C.DOMAIN_HALF_HEIGHT
    cells_per_tip_radius: int = C.CELLS_PER_TIP_RADIUS
    lesion_threshold: float = C.LESION_THRESHOLD

    @property
    def cell_size(self) -> float:
        return self.electrode_radius / self.cells_per_tip_radius

    @property
    def electrode_centre_z(self) -> float:
        """Tissue fills z < 0; the interface is the plane z = 0."""
        return self.electrode_radius - self.depth


def seating_depth(angle_error, electrode_radius=C.ELECTRODE_RADIUS,
                  stroke=C.INSERTION_STROKE):
    """How deeply the tip seats when it meets the wall off-normal.  PROVIDED.

    The operator advances the catheter by a fixed stroke along the shaft axis.
    Only the component normal to the wall, stroke*cos(dtheta), pushes the tip
    in, and a tilted sphere must first close the gap it opens by rolling on the
    wall, a*(1 - cos(dtheta)):

        depth = stroke*cos(dtheta) - a*(1 - cos(dtheta)),  clamped to [0, 2a]

    This is a geometric assumption, not a result.  Milestone 5 asks you to
    critique it.
    """
    c = np.cos(np.asarray(angle_error, dtype=float))
    depth = stroke * c - electrode_radius * (1.0 - c)
    return np.clip(depth, 0.0, 2.0 * electrode_radius)
