"""Map IFC sensors into the simulation model by structural member.

The as-planned IFC girder and the simulation model are close but not equal:
at station E+23 the IFC deck is 0.28 m thick (model 0.25 m), the webs of
the southern box are 1.45 m apart (model 1.60 m), and the deck has a
longitudinal gradient and cross-fall that the model (deck top at y = 0)
does not have.  The IFC sensor solids are 25 cm marker cubes.  A rigid
transformation of IFC coordinates would therefore move a sensor that is
"20 cm deep behind the cavity face of the deck" into the cavity, the air,
or the wrong member.

Instead each sensor is described relative to the box-girder cell
(cavity) it belongs to, and rebuilt in the model from the same description:

1. **station**: from the MKP measuring axis (``NiE+233`` -> 23.3 m),
   checked against the station of the IFC geometry, then converted to model
   ``z`` with :meth:`.frame.BridgeFrame.model_z` (z = station - 4.0 m);
2. **member**: deck, bottom slab, southern or northern web of the cell, from
   the ``ePset_Position`` text or else from the position relative to the
   cell in the IFC section;
3. **depth behind the inner (cavity) face**: from the text
   ("20 cm tief von der Innenseite") or else measured in the IFC section;
4. **position along the face**: "zentriert"/"mittig" -> cell centre, else
   the relative position measured in the IFC section.

Both sections are probed with the same ray-casting code
(:mod:`.section`), in the section frame ``(a, v, z)`` of :mod:`.frame`.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .frame import BridgeFrame
from .section import SectionProbe

log = logging.getLogger(__name__)

MEMBERS = ("deck", "bottom_slab", "web_south", "web_north")
# Conventions (text patterns, tolerances, pier axis) are in ``data/ifc_config.json``.


def _cfg(cfg: dict | None) -> dict:
    if cfg is None:
        from ..transform import load_ifc_config

        cfg = load_ifc_config()
    return cfg


def parse_station(axis_text: str, pier: str | None = None, cfg: dict | None = None) -> float | None:
    """'NiE+233' -> 23.3 (decimetres after the pier axis letter)."""
    cfg = _cfg(cfg)
    pier = pier or cfg["structure"]["pier_axis"]
    m = re.search(cfg["mapping"]["measuring_axis_pattern"].format(pier=pier), axis_text or "")
    if not m:
        return None
    return (1 if m.group(1) == "+" else -1) * int(m.group(2)) / 10.0


def parse_position_text(text: str, cfg: dict | None = None) -> dict:
    """Member, depth and centring from the German ``ePset_Position`` text."""
    pt = _cfg(cfg)["position_text"]
    t = (text or "").lower()
    out: dict = {}
    web = re.search(pt["web"], t)
    if web:
        out["member"] = "web_south" if web.group(1) == pt["web_south_prefix"] else "web_north"
    elif re.match(pt["deck"], t):
        out["member"] = "deck"
    elif re.match(pt["bottom_slab"], t):
        out["member"] = "bottom_slab"
    d = re.search(pt["depth_cm"], t)
    if d:
        out["depth"] = float(d.group(1).replace(",", ".")) / 100.0
    if re.search(pt["centred"], t):
        out["fraction"] = 0.5
    return out


def _classify(cell, a, v) -> tuple[str, float]:
    """Member of point (a, v) relative to a cell and its signed depth behind the face."""
    beyond = {
        "web_north": cell.a0 - a,
        "web_south": a - cell.a1,
        "bottom_slab": cell.floor - v,
        "deck": v - cell.ceiling,
    }
    member = max(beyond, key=beyond.get)
    return member, beyond[member]


def _measure_ifc(probe: SectionProbe, a: float, v: float, z: float) -> dict:
    cell = probe.cavity_near(a, v, z)
    member, _ = _classify(cell, a, v)
    v_mid = 0.5 * (cell.floor + cell.ceiling)
    a_mid = 0.5 * (cell.a0 + cell.a1)
    if member == "deck":
        depth = v - probe.face_above(a, z, v_mid)[0]
    elif member == "bottom_slab":
        depth = probe.face_below(a, z, v_mid)[0] - v
    elif member == "web_south":
        depth = a - probe.web(v, z, a_mid, +1)[0]
    else:
        depth = probe.web(v, z, a_mid, -1)[0] - a
    if member in ("deck", "bottom_slab"):
        fraction = (a - cell.a0) / (cell.a1 - cell.a0)
    else:
        fraction = (v - cell.floor) / (cell.ceiling - cell.floor)
    return {"member": member, "depth": float(depth), "fraction": float(fraction), "cell": cell}


def _place_in_model(probe: SectionProbe, a_guess: float, z: float,
                    member: str, depth: float, fraction: float,
                    margin: float) -> tuple[np.ndarray, float]:
    """Model point at ``depth`` behind the cavity face of ``member``; returns (xyz, used depth)."""
    cell = probe.cavity_near(a_guess, 0.0, z)
    v_mid = 0.5 * (cell.floor + cell.ceiling)
    a_mid = 0.5 * (cell.a0 + cell.a1)
    if member in ("deck", "bottom_slab"):
        x = cell.a0 + fraction * (cell.a1 - cell.a0)
        inner, outer = probe.face_above(x, z, v_mid) if member == "deck" else probe.face_below(x, z, v_mid)
        d = float(np.clip(depth, margin, abs(outer - inner) - margin))
        y = inner + d if member == "deck" else inner - d
    else:
        y = cell.floor + fraction * (cell.ceiling - cell.floor)
        side = +1 if member == "web_south" else -1
        inner, outer = probe.web(y, z, a_mid, side)
        d = float(np.clip(depth, margin, abs(outer - inner) - margin))
        x = inner + side * d
    return np.array([x, y, z]), d


def map_sensor(sensor: dict, frame: BridgeFrame, ifc_probe: SectionProbe,
               model_probe: SectionProbe, cfg: dict | None = None) -> dict:
    """Model position and an audit trail for one sensor from :func:`.reader.read_sensors`.

    ``cfg`` is the content of ``data/ifc_config.json`` (default: the packaged one).
    """
    cfg = _cfg(cfg)
    margin = float(cfg["mapping"]["margin_m"])
    station_tol = float(cfg["mapping"]["station_tolerance_m"])
    a, v, s_geo = frame.to_section(sensor["centroid"])[0]
    notes: list[str] = []

    s_axis = parse_station(sensor.get("measuring_axis", ""), cfg=cfg)
    station = s_axis if s_axis is not None else s_geo
    if s_axis is not None and abs(s_axis - s_geo) > station_tol:
        notes.append(f"measuring axis station {s_axis} m vs IFC geometry {s_geo:.2f} m")
    z = float(frame.model_z(station))
    if z < 0:
        raise ValueError(f"{sensor.get('object_name')}: station {station} m lies before the "
                         f"model's clamped section (E+{frame.model_z0_station} m)")

    geo = _measure_ifc(ifc_probe, a, v, s_geo)
    text = parse_position_text(sensor.get("position_text", ""), cfg)
    member = text.get("member", geo["member"])
    if "member" in text and text["member"] != geo["member"]:
        notes.append(f"text says {text['member']}, IFC geometry says {geo['member']}; using the text")
    depth = text.get("depth", geo["depth"])
    fraction = text.get("fraction", geo["fraction"])
    if depth <= 0:
        notes.append(f"IFC point lies {-depth:.3f} m inside the cavity; placed {margin} m into the {member}")

    xyz, used_depth = _place_in_model(model_probe, a, z, member, depth, fraction, margin)
    if abs(used_depth - depth) > 1e-9 and depth > 0:
        notes.append(f"depth {depth:.3f} m clipped to {used_depth:.3f} m (model member thickness)")
    for n in notes:
        log.warning("%s: %s", sensor.get("object_name"), n)
    return {
        "position": [round(float(c), 4) for c in xyz],
        "mapping": {
            "station_m": round(float(station), 4),
            "model_z_m": round(z, 4),
            "station_source": "measuring_axis" if s_axis is not None else "ifc_geometry",
            "station_ifc_geometry_m": round(float(s_geo), 4),
            "member": member,
            "member_source": "position_text" if "member" in text else "ifc_geometry",
            "depth_behind_cavity_face_m": round(used_depth, 4),
            "depth_source": "position_text" if "depth" in text else "ifc_geometry",
            "depth_in_ifc_geometry_m": round(geo["depth"], 4),
            "fraction_along_face": round(float(fraction), 4),
            "fraction_source": "position_text" if "fraction" in text else "ifc_geometry",
            "notes": notes,
        },
    }
