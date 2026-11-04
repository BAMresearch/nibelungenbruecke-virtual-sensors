"""Regression check of the package FEM against the dolfinx research pipeline.

Development tool: needs the pipeline outputs under ``code/results`` and the
processed forcing under ``code/data`` (plus xarray and PyTables), so run it
in the ``snakemake_bridge`` conda env from ``code/``::

    python virtual_sensors/tools/validate_against_pipeline.py

It replays, per PCE node, the forcing of

* ``in_domain`` (June 2024, 30 min) -> ``results/simulations/thermal_3d/in_domain_no_hr_ensemble.nc``
* ``hindcast`` (June 2025, 1 h)     -> ``results/simulations/thermomechanical/hindcast_no_hr_ensemble.nc``

with the pipeline's spin-up (100 initial-condition steps + 700 burn-in rows)
and prints the largest deviation per sensor. Sensor positions are read from the pipeline's own files
(`config/pipeline_config.yaml`, `sensors/`), the element degree is the package default (P2), as in the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from nibelungen_virtual_sensors.fem.elements import Space  # noqa: E402
from nibelungen_virtual_sensors.fem.mechanics import ThermoElasticModel  # noqa: E402
from nibelungen_virtual_sensors.fem.mesh import Mesh  # noqa: E402
from nibelungen_virtual_sensors.fem.probes import build_probes  # noqa: E402
from nibelungen_virtual_sensors.fem.thermal import ThermalModel  # noqa: E402
from nibelungen_virtual_sensors.transform import _data_path, load_model_config  # noqa: E402
from nibelungen_virtual_sensors.uq import convection_rule  # noqa: E402

CODE = HERE.parents[1]
FORCING = ["air_temperature", "inner_temperature", "shortwave_irradiation", "wind_speed"]
IC_STEPS, BURN_IN = 100, 700


def _positions() -> tuple[dict, dict]:
    """Sensor positions of the pipeline runs, read from the pipeline's own files.

    * pilot sensors (E+4.0, z = 0): ``sensor_positions`` of ``config/pipeline_config.yaml``
      (calibrated positions) and the y of the webs from ``sensors/sensors_temperature_3d.json``;
    * validation sensors (E+41.3, z = 37.3): ``sensors/registration.json``;
    * mechanical: ``sensors/sensors_mechanical.json`` (inclinometer at z = 38.3,
      free-end displacement at z = 40.09, the hinge).
    """
    import yaml

    cfg = yaml.safe_load((CODE / "config/pipeline_config.yaml").read_text())
    p = cfg["sensor_positions"]
    js = {s["id"]: s["where"] for s in json.loads((CODE / "sensors/sensors_temperature_3d.json").read_text())["sensors"]}
    reg = json.loads((CODE / "sensors/registration.json").read_text())["validation"]["sensors"]
    thermal = {
        "bridge_temperature_u": [2.85, p["bottom"], 0.0],
        "bridge_temperature_o": [2.85, p["top"], 0.0],
        "bridge_temperature_s": [p["south"], js["Sensor_s"][1], 0.0],
        "bridge_temperature_n": [p["north"], js["Sensor_n"][1], 0.0],
    }
    for key, short in (("bottom", "u"), ("top", "o"), ("south", "s"), ("north", "n")):
        thermal[f"bridge_temperature_{short}_val"] = reg[key]["position"]
    mech = json.loads((CODE / "sensors/sensors_mechanical.json").read_text())["sensors"]
    mech = {s["id"]: s["where"] for s in mech}
    # sensors_mechanical.json: inclination (rad) and free-end displacement u_y (m)
    mech_sensors = {"angle_z_free": ("inclination", mech["inclination"], 1e3),
                    "uy_z_free": ("displacement_y", mech["disp_free_end"], 1e3)}
    return thermal, mech_sensors


def _forcing(h5: str, start: str, end: str) -> pd.DataFrame:
    return pd.read_hdf(CODE / "data/processed" / h5, "data").loc[start:end]


def _reference(nc: str, var: str, node: int, times, sensors) -> np.ndarray:
    import xarray as xr

    ds = xr.open_dataset(CODE / "results/simulations" / nc)
    da = ds[var].isel(node=node).assign_coords(time=pd.to_datetime(ds.time.values, utc=True))
    return da.sel(time=times, sensor=list(sensors)).values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--degree", type=int, default=None,
                    help="element degree (default: the package default, which is the pipeline's, P2)")
    a = ap.parse_args()
    cfg = load_model_config()
    if a.degree:
        cfg["thermal"]["degree"] = cfg["mechanical"]["degree"] = a.degree
    degree = int(cfg["thermal"]["degree"])
    THERMAL_SENSORS, MECH_SENSORS = _positions()
    mesh = Mesh.load(_data_path(cfg["mesh_file"]))
    print(f"mesh {mesh.n_nodes} nodes, z = {mesh.points[:, 2].min():.2f} ... {mesh.points[:, 2].max():.2f} m, P{degree}")
    space = Space(mesh, degree)
    thermal = ThermalModel(mesh, cfg["thermal"], space)
    nodes, _ = convection_rule(cfg["uncertainty"])   # (n_nodes, 2): c_top, c_ext, in the order of the pipeline PCE

    names = list(THERMAL_SENSORS)
    PT, _, off, _ = build_probes(mesh, ["temperature"] * len(names), np.array(list(THERMAL_SENSORS.values())), space)
    df = _forcing("validation_june_2024.h5", "2024-05-15T01:00:00Z", "2024-06-30T23:00:00Z")
    for j, c in enumerate(nodes):
        out = {}
        thermal.run(df[FORCING].to_numpy(), 1800.0, tuple(c), IC_STEPS, BURN_IN,
                    lambda k, T: out.__setitem__(k, PT @ T + off))
        ks = sorted(out)
        ref = _reference("thermal_3d/in_domain_no_hr_ensemble.nc", "temperature", j, df.index[ks], names) - 273.15
        dev = np.abs(np.array([out[k] for k in ks]) - ref).max(0)
        print(f"thermal  node {j} (c_top={c[0]:.4f}, c_ext={c[1]:.4f}), {len(ks)} steps, max |dT| [K]:",
              {k.replace("bridge_temperature_", ""): float(v) for k, v in zip(names, dev)})

    # mechanical readings as the package computes them: y = W (T - T_ref), one adjoint solve per sensor
    mech = ThermoElasticModel(mesh, cfg["mechanical"], space)
    mnames = list(MECH_SENSORS)
    _, Pu, _, _ = build_probes(mesh, [v[0] for v in MECH_SENSORS.values()],
                               np.array([v[1] for v in MECH_SENSORS.values()]), space)
    W = mech.sensor_weights(Pu, cfg["mechanical"].get("solver", "ilu"))
    scale = np.array([v[2] for v in MECH_SENSORS.values()])
    df = _forcing("hindcast_june_2025.h5", "2025-05-15T01:00:00Z", "2025-06-30T23:00:00Z")
    for j, c in enumerate(nodes):
        out = {}
        thermal.run(df[FORCING].to_numpy(), 3600.0, tuple(c), IC_STEPS, BURN_IN,
                    lambda k, T: out.__setitem__(k, W @ (T - mech.T_ref)))
        ks = sorted(out)
        ref = _reference("thermomechanical/hindcast_no_hr_ensemble.nc", "displacement", j, df.index[ks], mnames) * scale
        rel = np.abs(np.array([out[k] for k in ks]) - ref).max(0) / np.abs(ref).max(0)
        print(f"mechanic node {j} (c_top={c[0]:.4f}, c_ext={c[1]:.4f}), {len(ks)} steps, max relative deviation:",
              dict(zip(mnames, (float(r) for r in rel))))


if __name__ == "__main__":
    main()
