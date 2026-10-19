"""Bridge reference frame: IFC world coordinates <-> section coordinates.

The MKP IFC models are in ETRS89 / UTM zone 32 with the zone prefix in the
easting (E ≈ 32 455 100 m) and NHN heights; none of them carries an
``IfcMapConversion``, so placement coordinates are map coordinates directly.

The frame is derived from the as-planned superstructure model ("STB"):

* plan axis ``e_s`` = principal direction of all superstructure vertices
  (the river bridge is straight in plan, |t| ≤ 6.73 m along its full length),
  oriented so that the instrumented cantilever lies at positive stations;
* station origin = centre of the pier head at pier axis E
  ("Pfeilerkopf E"), so ``s`` reproduces the station of MKP channel names,
  e.g. ``E+233`` -> s = 23.3 m;
* ``t`` = horizontal offset from the axis, positive to the left of ``e_s``,
  i.e. towards the north.

**Station -> model z.**  The model's ``z = 0`` section is the calibrated 2-D
pilot section (5.87 m deep), i.e. the instrumented section of the MKP
sensors ``E+040``, not the pier axis.  Ray-cast depth profiles along the
cantilever confirm this: ``z = s - 4.0`` matches the IFC girder depth with
an RMS error of 0.11 m (cavity height 0.15 m), whereas ``z = s`` gives
0.31 m.  Lengths are kept (no scaling), so the clamp sits at E+4.0 and the
real hinge (E+44.1) at z = 40.1; the model girder continues to z = 47.6,
which does not affect anything closer to the clamp (statically determinate
cantilever under thermal load).  ``model_z0_station`` holds this anchor.
Both it and ``mirror_x`` are set in ``data/ifc_config.json`` (``frame``).

**Handedness.**  The simulation model puts the southern box girder at
``x > 0`` while ``z`` runs from pier E towards the river (≈ ENE) and ``y``
is up.  With ``x = +t`` that frame would be right-handed but the southern
box would sit at ``x < 0``; the model is therefore the *mirror image* of the
real bridge and ``x = -t``.  (The model is symmetric in x, so this does not
change any prediction, but positions must be mirrored, not rotated.)  A
rigid-body fit of IFC to model coordinates would silently miss this.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BridgeFrame:
    origin_e: float            # easting of the axis point at station 0 [m]
    origin_n: float            # northing [m]
    axis_e: float              # unit plan axis, easting component
    axis_n: float              # unit plan axis, northing component
    mirror_x: float            # model x = mirror_x * t
    model_z0_station: float    # station of the model section z = 0 [m]
    source: str = ""

    @property
    def axis(self) -> np.ndarray:
        return np.array([self.axis_e, self.axis_n])

    @property
    def left(self) -> np.ndarray:
        return np.array([-self.axis_n, self.axis_e])

    def to_station(self, xyz) -> np.ndarray:
        """World (E, N, H) -> (s, t, H)."""
        p = np.atleast_2d(np.asarray(xyz, float))
        d = p[:, :2] - [self.origin_e, self.origin_n]
        return np.column_stack([d @ self.axis, d @ self.left, p[:, 2]])

    def model_z(self, station):
        return np.asarray(station, float) - self.model_z0_station

    def to_section(self, xyz) -> np.ndarray:
        """World (E, N, H) -> IFC section coordinates (a, v, s) = (mirror·t, H, station)."""
        stn = self.to_station(xyz)
        return np.column_stack([self.mirror_x * stn[:, 1], stn[:, 2], stn[:, 0]])

    def save(self, path: Path) -> None:
        """Write the derived geometry; ``mirror_x`` and ``model_z0_station`` live in ``ifc_config.json``."""
        d = asdict(self)
        d.pop("mirror_x"), d.pop("model_z0_station")
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, frame_cfg: dict | None = None) -> "BridgeFrame":
        """Derived geometry from ``path``; ``mirror_x`` and ``model_z0_station`` from the IFC config."""
        if frame_cfg is None:
            from ..transform import load_ifc_config

            frame_cfg = load_ifc_config()["frame"]
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{**d, "mirror_x": float(frame_cfg["mirror_x"]),
                      "model_z0_station": float(frame_cfg["model_z0_station"])})


def frame_from_superstructure(verts_by_name: dict[str, np.ndarray],
                              pier_head: str, cantilever: str, frame_cfg: dict,
                              source: str = "") -> BridgeFrame:
    """Build the frame from superstructure vertices grouped by element name."""
    allv = np.vstack(list(verts_by_name.values()))
    ref = allv[:, :2].mean(axis=0)
    _, _, vt = np.linalg.svd(allv[:, :2] - ref, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    pier = verts_by_name[pier_head][:, :2]
    cant = verts_by_name[cantilever][:, :2]
    if (cant - ref).mean(0) @ axis < (pier - ref).mean(0) @ axis:
        axis = -axis                     # cantilever at positive stations
    left = np.array([-axis[1], axis[0]])
    s_pier = (pier - ref) @ axis
    s0 = 0.5 * (s_pier.min() + s_pier.max())
    t_all = (allv[:, :2] - ref) @ left
    t0 = 0.5 * (t_all.min() + t_all.max())   # centre line of the deck
    origin = ref + s0 * axis + t0 * left
    return BridgeFrame(float(origin[0]), float(origin[1]), float(axis[0]), float(axis[1]),
                       mirror_x=float(frame_cfg["mirror_x"]),
                       model_z0_station=float(frame_cfg["model_z0_station"]), source=source)
