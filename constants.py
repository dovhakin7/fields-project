"""Physical constants and baseline device parameters.  PROVIDED — do not edit.

Every number here is either a physical constant or a baseline design value
quoted in the project handout.  Import from this module rather than typing
numbers into your own code.
"""

import numpy as np

# ----------------------------------------------------------------- constants
MU0 = 4.0e-7 * np.pi          # H/m   permeability of free space
EPS0 = 8.8541878128e-12       # F/m   permittivity of free space

SIGMA_COPPER = 5.8e7          # S/m   annealed copper at 20 C
SIGMA_MYOCARDIUM = 0.54       # S/m   heart muscle, ~500 kHz
SIGMA_BLOOD = 0.70            # S/m   whole blood, ~500 kHz

BR_NDFEB = 1.30               # T     remanence of sintered NdFeB, grade N42

# ------------------------------------------------- baseline navigation coils
COIL_RADIUS = 0.350           # m     a
COIL_SEPARATION = 0.350       # m     Helmholtz spacing, d = a
COIL_TURNS = 155              # per coil
COIL_CURRENT = 200.0          # A     per turn
COIL_WIRE_AREA = 1.62e-4      # m^2   conductor cross-section (~14 mm square)

# ------------------------------------------------------------ catheter tip
TIP_MAGNET_RADIUS = 1.0e-3    # m
TIP_MAGNET_LENGTH = 4.0e-3    # m
SHAFT_STIFFNESS = 1.0e-3      # N*m/rad  restoring torque of the catheter shaft

# --------------------------------------------------------- ablation electrode
ELECTRODE_RADIUS = 2.0e-3     # m     spherical tip electrode
INSERTION_STROKE = 1.66e-3    # m     how far the operator advances past first touch
ABLATION_VOLTAGE = 25.0       # V     RF drive amplitude used for the dc proxy
LESION_THRESHOLD = 1.2e6      # W/m^3 power-density contour taken as the lesion

# ------------------------------------------------------------ solver defaults
DOMAIN_RADIUS = 0.040         # m     outer radius of the tissue domain
DOMAIN_HALF_HEIGHT = 0.040    # m
CELLS_PER_TIP_RADIUS = 12     # baseline grid resolution

# ----------------------------------------------------- lumped cooling model
H_FREE_CONVECTION = 4.0       # W/(m^2*K)  still air, bare winding (range 2-25)
