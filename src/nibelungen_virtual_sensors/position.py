"""Sensor-position uncertainty: samples of the true position of a temperature sensor.

The registered position of a sensor is not exactly where the probe sits (the
IFC marker cube and the property text disagree on the depth, installation
tolerance).  The delivered predictive distribution therefore averages over the
position (law of total variance, see :func:`.uq.total_variance_moments`).  This
module draws the position samples; the model is then evaluated at all of them
with one probe matrix per quadrature node.

Prior (member-relative, decided by the authors; parameters in
``uncertainty.position`` of ``data/model_config.json``, placement margin in
``mapping.margin_m`` of ``data/ifc_config.json``):

* **depth** ``d`` behind the cavity face, along the member normal, uniform on
  ``[min(d_text, d_ifc) - tol, max(d_text, d_ifc) + tol]``; ``d_text`` is the
  depth of the position text, ``d_ifc`` the depth measured at the IFC marker
  centre;
* **in the face plane** (across the member and along ``z``): independent
  ``N(0, sigma_plane^2)``;
* every sample is **clipped into the member** with the placement margin: depth
  to ``[margin, thickness - margin]``, the in-plane coordinate inside the cell
  and ``z`` inside the model, so that no sample lies in the cavity or outside
  the concrete.

The samples are a fixed-seed scrambled Sobol sequence (``d = 4``: depth, two
in-plane deviations and one spare dimension, as in the study
``code/studies/position_uncertainty``), so the same input always gives the
same output.

A sensor definition selects its prior (all keys optional, temperature only):

* IFC-derived sensors carry a ``mapping`` block (member, ``d_text`` =
  ``depth_behind_cavity_face_m``, ``d_ifc`` = ``depth_in_ifc_geometry_m``): the
  prior above is built automatically;
* ``"member"`` + ``"depth_range": [lo, hi]`` [m] sets the depth range directly
  (``tolerance`` is then not added); ``"position_std"`` [m] replaces the in-plane
  standard deviation;
* ``"position_std"`` alone (no member): isotropic ``N(0, std^2 I)`` around the
  given position in model coordinates, truncated to the concrete (samples that
  fall outside the mesh are rejected).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .fem.mesh import Mesh
from .ifc.mapping import MEMBERS
from .ifc.section import SectionProbe

#: Sobol dimensions: depth, in-plane (across), in-plane (z), spare.
_DIM = 4


def _sobol(n: int, seed: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # scipy warns for n that is not a power of 2
        return qmc.Sobol(d=_DIM, scramble=True, seed=seed).random(n)


def section_probe(mesh: Mesh) -> SectionProbe:
    """Boundary of the simulation mesh as a section probe ``(a, v, z) = (x, y, z)``."""
    return SectionProbe(mesh.points[mesh.facets])


def _place(probe: SectionProbe, cell, z0: float, z_end: float, member: str, depth: float,
           e_face: float, e_z: float, a0: float, v0: float, margin: float) -> tuple[np.ndarray, float]:
    """Model point of one member-relative sample; returns (xyz, depth used).

    ``a0`` / ``v0`` are the nominal x and y of the sensor (the in-plane coordinate
    is x for deck and bottom slab, y for the webs); the cell fixes the clipping range.
    """
    z = float(np.clip(z0 + e_z, margin, z_end - margin))
    a_mid, v_mid = 0.5 * (cell.a0 + cell.a1), 0.5 * (cell.floor + cell.ceiling)
    if member in ("deck", "bottom_slab"):
        x = float(np.clip(a0 + e_face, cell.a0 + margin, cell.a1 - margin))
        inner, outer = probe.face_above(x, z, v_mid) if member == "deck" else probe.face_below(x, z, v_mid)
        d = float(np.clip(depth, margin, abs(outer - inner) - margin))
        return np.array([x, inner + d if member == "deck" else inner - d, z]), d
    y = float(np.clip(v0 + e_face, cell.floor + margin, cell.ceiling - margin))
    side = +1 if member == "web_south" else -1
    inner, outer = probe.web(y, z, a_mid, side)
    d = float(np.clip(depth, margin, abs(outer - inner) - margin))
    return np.array([inner + side * d, y, z]), d


def _member_samples(sensor: dict, pos_cfg: dict, margin: float, probe: SectionProbe,
                    z_end: float) -> np.ndarray | None:
    mapping = sensor.get("mapping") or {}
    member = sensor.get("member", mapping.get("member"))
    if member is None:
        return None
    if member not in MEMBERS:
        raise ValueError(f"sensor {sensor['name']!r}: unknown member {member!r}; one of {MEMBERS}")
    tol = float(pos_cfg["tolerance_m"])
    if sensor.get("depth_range") is not None:
        lo, hi = (float(v) for v in sensor["depth_range"])
        if hi < lo:
            raise ValueError(f"sensor {sensor['name']!r}: depth_range must be [lo, hi] with lo <= hi")
    elif "depth_behind_cavity_face_m" in mapping and "depth_in_ifc_geometry_m" in mapping:
        d_text, d_ifc = float(mapping["depth_behind_cavity_face_m"]), float(mapping["depth_in_ifc_geometry_m"])
        lo, hi = min(d_text, d_ifc) - tol, max(d_text, d_ifc) + tol
    else:
        return None
    sigma = float(sensor.get("position_std", pos_cfg["in_plane_std_m"]))
    n = int(pos_cfg["n_samples"])
    seed = int(pos_cfg["seed"]) + MEMBERS.index(member)

    x0, y0, z0 = (float(c) for c in sensor["position"])
    cell = probe.cavity_near(x0, y0, z0)
    u = _sobol(n, seed)
    d_raw = lo + (hi - lo) * u[:, 0]
    e = ndtri(u[:, 1:4]) * sigma
    out = np.empty((n, 3))
    for k in range(n):
        out[k], _ = _place(probe, cell, z0, z_end, member, d_raw[k], e[k, 0], e[k, 1], x0, y0, margin)
    return out


def _isotropic_samples(sensor: dict, pos_cfg: dict, mesh: Mesh) -> np.ndarray:
    sigma = float(sensor["position_std"])
    n = int(pos_cfg["n_samples"])
    seed = int(pos_cfg["seed"]) + len(MEMBERS)
    x0 = np.asarray(sensor["position"], dtype=np.float64)
    kept: list[np.ndarray] = []
    for attempt in range(8):                 # truncate the Gaussian to the concrete by rejection
        u = _sobol(8 * n, seed + 100 * attempt)
        cand = x0 + ndtri(np.clip(u[:, :3], 1e-12, 1 - 1e-12)) * sigma
        kept.extend(c for c, ok in zip(cand, mesh.contains(cand)) if ok)
        if len(kept) >= n:
            return np.array(kept[:n])
    raise ValueError(f"sensor {sensor['name']!r}: position_std={sigma} m leaves fewer than {n} samples "
                     "inside the concrete; is the position close to a surface?")


def position_samples(sensor: dict, pos_cfg: dict, margin: float, mesh: Mesh,
                     probe: SectionProbe | None = None) -> np.ndarray | None:
    """Position samples ``(n_samples, 3)`` in model coordinates, or ``None`` for no position term.

    Only temperature sensors get a term (tilt, displacement and strain are
    insensitive to the position at the scale of the tolerance).
    """
    if sensor["quantity"] != "temperature":
        return None
    probe = probe or section_probe(mesh)
    z_end = float(mesh.points[:, 2].max())
    s = _member_samples(sensor, pos_cfg, margin, probe, z_end)
    if s is not None:
        return s
    if float(sensor.get("position_std", 0.0)) > 0.0:
        return _isotropic_samples(sensor, pos_cfg, mesh)
    return None
