"""Point probes: sparse linear operators from nodal fields to sensor readings.

Every supported quantity is a linear functional of either the nodal
temperature or the nodal displacement, so a set of virtual sensors reduces
to two sparse matrices.  Only these readings are kept in memory, never the
full fields.

Quantities and output units
---------------------------
temperature                     °C
displacement_x / _y / _z        mm
inclination                     mm/m  (= ∂u_y/∂z · 1e3, the MKP inclinometer
                                       convention; same sign as the model)
strain_xx / _yy / _zz           µm/m  (normal strains)
strain_xy / _yz / _xz           µm/m  (tensor shear components, ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i))

Gradient quantities are the exact point value of the displacement gradient
in the containing cell: piecewise constant for P1, linear for P2.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .elements import Space
from .mesh import Mesh

THERMAL = {"temperature"}
_DISP = {"displacement_x": 0, "displacement_y": 1, "displacement_z": 2}
_GRAD = {  # name -> list of (i, j, weight) with value = Σ w ∂u_i/∂x_j
    "inclination": [(1, 2, 1.0)],
    "strain_xx": [(0, 0, 1.0)],
    "strain_yy": [(1, 1, 1.0)],
    "strain_zz": [(2, 2, 1.0)],
    "strain_xy": [(0, 1, 0.5), (1, 0, 0.5)],
    "strain_yz": [(1, 2, 0.5), (2, 1, 0.5)],
    "strain_xz": [(0, 2, 0.5), (2, 0, 0.5)],
}
UNITS = {"temperature": "°C", "inclination": "mm/m"}
UNITS.update({k: "mm" for k in _DISP})
UNITS.update({k: "µm/m" for k in _GRAD if k.startswith("strain")})
_SCALE = {"temperature": 1.0, "inclination": 1e3}
_SCALE.update({k: 1e3 for k in _DISP})
_SCALE.update({k: 1e6 for k in _GRAD if k.startswith("strain")})

QUANTITIES = sorted(UNITS)


def build_probes(mesh: Mesh, quantities: list[str], positions: np.ndarray, space: Space | int = 1):
    """Return (P_T, P_u, offset_T, is_thermal).

    reading = P_T @ T + offset_T   for thermal rows (T in K, reading in °C)
    reading = P_u @ u              for mechanical rows (u in m)

    ``space`` is the finite-element :class:`Space` of the fields (or its degree).
    """
    unknown = sorted(set(quantities) - set(QUANTITIES))
    if unknown:
        raise ValueError(f"unsupported quantity {unknown}; supported: {QUANTITIES}")
    space = space if isinstance(space, Space) else Space(mesh, int(space))
    cells, bary = mesh.locate(positions)
    phi, dphi = space.point_operator_rows(cells, bary)       # (S, nloc), (S, nloc, 3)
    n = len(quantities)
    PT = sp.lil_matrix((n, space.n))
    Pu = sp.lil_matrix((n, 3 * space.n))
    offset = np.zeros(n)
    is_thermal = np.zeros(n, dtype=bool)
    for r, q in enumerate(quantities):
        nodes = space.cell_dofs[cells[r]]
        s = _SCALE[q]
        if q in THERMAL:
            is_thermal[r] = True
            for a, d in enumerate(nodes):
                PT[r, d] += phi[r, a]
            offset[r] = -273.15
        elif q in _DISP:
            i = _DISP[q]
            for a, d in enumerate(nodes):
                Pu[r, 3 * d + i] += s * phi[r, a]
        else:
            for i, j, w in _GRAD[q]:
                for a, d in enumerate(nodes):
                    Pu[r, 3 * d + i] += s * w * dphi[r, a, j]
    return PT.tocsr(), Pu.tocsr(), offset, is_thermal
