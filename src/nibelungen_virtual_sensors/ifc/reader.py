"""IFC access (requires the optional dependency ``ifcopenshell``)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _ifc():
    try:
        import ifcopenshell
        import ifcopenshell.geom  # noqa: F401
        import ifcopenshell.util.element  # noqa: F401
        import ifcopenshell.util.placement  # noqa: F401
        import ifcopenshell.util.unit  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("IFC support needs: pip install 'nibelungen-virtual-sensors[ifc]'") from exc
    return ifcopenshell


def _map_conversion(model):
    """Return f(local xyz in metres) -> map coordinates, honouring IfcMapConversion."""
    convs = model.by_type("IfcMapConversion") if model.schema != "IFC2X3" else []
    if not convs:
        return lambda p: p
    c = convs[0]
    scale = c.Scale or 1.0
    a, o = c.XAxisAbscissa or 1.0, c.XAxisOrdinate or 0.0
    n = np.hypot(a, o)
    a, o = a / n, o / n

    def f(p):
        p = np.atleast_2d(p)
        e = c.Eastings + scale * (a * p[:, 0] - o * p[:, 1])
        nn = c.Northings + scale * (o * p[:, 0] + a * p[:, 1])
        return np.column_stack([e, nn, c.OrthogonalHeight + scale * p[:, 2]])

    log.info("applying IfcMapConversion of %s", model.schema)
    return f


def _world_settings():
    import ifcopenshell.geom

    settings = ifcopenshell.geom.settings()
    try:
        settings.set("use-world-coords", True)          # ifcopenshell >= 0.8
    except Exception:  # pragma: no cover
        settings.set(settings.USE_WORLD_COORDS, True)
    return settings


def _shape(model, element, settings, to_map):
    import ifcopenshell.geom

    g = ifcopenshell.geom.create_shape(settings, element).geometry
    v = np.asarray(g.verts, float).reshape(-1, 3)
    f = np.asarray(g.faces, int).reshape(-1, 3)
    # the geometry iterator returns metres (SI); placements need the unit scale
    return to_map(v), f


def read_superstructure(path: str | Path, cfg: dict):
    """Triangles (world coordinates) of the box-girder elements, by element name.

    ``cfg`` is the ``structure`` section of ``data/ifc_config.json``.
    """
    ifcopenshell = _ifc()
    import ifcopenshell.util.element as el

    model = ifcopenshell.open(str(path))
    settings = _world_settings()
    to_map = _map_conversion(model)
    out: dict[str, list[np.ndarray]] = {}
    for e in model.by_type("IfcProduct"):
        if not e.Representation:
            continue
        oid = el.get_psets(e).get(cfg["element_id_pset"], {}).get(cfg["element_id_property"], "") or ""
        if cfg["element_id_filter"] not in oid:
            continue
        v, f = _shape(model, e, settings, to_map)
        out.setdefault(e.Name, []).append(v[f])
    return {k: np.concatenate(v) for k, v in out.items()}


def read_sensors(path: str | Path, cfg: dict) -> list[dict]:
    """Sensor proxies with their MKP property sets and geometry.

    The proxies of the MKP "DZ-Sim" models are 25 cm marker cubes whose
    placement origin is a cube corner; ``centroid`` is the cube centre.
    ``cfg`` is the ``sensor_ifc`` section of ``data/ifc_config.json``.
    """
    ifcopenshell = _ifc()
    import ifcopenshell.util.element as el
    import ifcopenshell.util.placement as pl
    import ifcopenshell.util.unit as un

    model = ifcopenshell.open(str(path))
    settings = _world_settings()
    unit = un.calculate_unit_scale(model)
    to_map = _map_conversion(model)
    sensors = []
    ident, pos, comp = cfg["identity_pset"], cfg["position_pset"], cfg["component_pset"]
    for p in model.by_type(cfg["ifc_class"]):
        ps = el.get_psets(p)
        if ps.get(ident, {}).get(cfg["sensor_flag_property"]) != cfg["sensor_flag_value"]:
            continue
        channels = [
            {k: v for k, v in ps[k].items() if k != "id"}
            for k in sorted(ps) if k.startswith(cfg["channel_pset_prefix"])
        ]
        origin = to_map(pl.get_local_placement(p.ObjectPlacement)[:3, 3] * unit)[0]
        centroid = origin
        if p.Representation:
            v, _ = _shape(model, p, settings, to_map)
            centroid = 0.5 * (v.min(0) + v.max(0))
        units = {c.get(cfg["channel_unit_property"]) for c in channels}
        quantity = next((cfg["quantity_by_unit"][u] for u in units if u in cfg["quantity_by_unit"]), None)
        sensors.append({
            "global_id": p.GlobalId,
            "object_name": ps[ident].get(cfg["name_property"]),
            "channels": channels,
            "quantity": quantity,
            "position_text": ps.get(pos, {}).get(cfg["position_text_property"], ""),
            "orientation_text": ps.get(pos, {}).get(cfg["orientation_property"], ""),
            "measuring_axis": ps.get(pos, {}).get(cfg["measuring_axis_property"], ""),
            "component_id": ps.get(comp, {}).get(cfg["component_id_property"], ""),
            "origin": origin.tolist(),
            "centroid": centroid.tolist(),
        })
    log.info("%d sensor proxies read from %s", len(sensors), Path(path).name)
    return sensors
