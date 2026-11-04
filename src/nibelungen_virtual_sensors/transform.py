"""Entry point for the MKP platform: ``transform(df, ...) -> pd.DataFrame``.

The function is stateless: every call rebuilds the thermal state from the
history contained in ``df`` (spin-up), so it can run in a container that
keeps nothing between the daily executions.  Only the predictions at the
requested virtual sensors are held in memory and returned; full fields are
never stored.
"""

from __future__ import annotations

import copy
import json
import logging
from importlib import resources
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .fem.elements import Space
from .fem.mechanics import ThermoElasticModel
from .fem.mesh import Mesh
from .fem.probes import UNITS, build_probes
from .fem.thermal import ThermalModel
from .position import position_samples, section_probe
from .uq import convection_rule, total_variance_moments, weighted_moments

log = logging.getLogger(__name__)

_ORDER = ["air_temperature", "cavity_temperature", "shortwave_irradiation", "wind_speed"]

#: Suffixes of the output columns of each virtual sensor ``<name>``.
OUTPUT_SUFFIXES = {"mean": "", "std": "_std", "lower": "_lower", "upper": "_upper"}


def _data_path(name: str) -> Path:
    return Path(str(resources.files("nibelungen_virtual_sensors") / "data" / name))


def load_model_config() -> dict:
    with open(_data_path("model_config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_ifc_config() -> dict:
    """IFC conventions and model/IFC alignment (``data/ifc_config.json``)."""
    with open(_data_path("ifc_config.json"), encoding="utf-8") as fh:
        return json.load(fh)


#: Model input -> default MKP channel name (``input_channels`` of
#: ``data/model_config.json``, as in the ``sensors.mkp`` section of
#: ``code/config/pipeline_config.yaml``).  Wind speed is not measured by MKP;
#: the column is attached beforehand by :func:`.weather.add_wind_speed`
#: (Open-Meteo ``wind_speed_10m`` or a constant).
DEFAULT_INPUT_CHANNELS: dict[str, str] = {
    k: v for k, v in load_model_config()["input_channels"].items() if not k.startswith("_")
}


def load_default_sensors() -> list[dict]:
    """Virtual sensors shipped with the package (see ``data/virtual_sensors.json``)."""
    path = _data_path("virtual_sensors.json")
    if not path.exists():
        raise ValueError("no virtual_sensors given and the package ships no default set")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["sensors"]


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _noise_std(sensor: dict, cfg: dict) -> float:
    if sensor.get("noise_std") is not None:
        return float(sensor["noise_std"])
    family = sensor["quantity"].split("_")[0]
    return float(cfg["uncertainty"]["default_noise_std"].get(family, 0.0))


def _prepare_forcing(df: pd.DataFrame, channels: dict, timestep: str) -> pd.DataFrame:
    if "Timestamp" in df.columns:
        data = df.set_index("Timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        data = df
    else:
        raise ValueError("df needs a 'Timestamp' column (datetime64[ns])")
    data = data[~data.index.duplicated(keep="last")].sort_index()

    out = pd.DataFrame(index=data.index)
    for role in _ORDER:
        col = channels[role]
        if col in data.columns and data[col].notna().any():
            out[role] = pd.to_numeric(data[col], errors="coerce").astype("float64")
        else:
            hint = " (attach it with add_wind_speed)" if role == "wind_speed" else ""
            raise ValueError(f"input channel '{col}' for {role} missing or empty{hint}")

    # Same preprocessing as the data generation of the research pipeline:
    # fill, resample to the model time step (bin mean), interpolate.
    out = out.ffill().bfill().resample(timestep).mean().interpolate().ffill().bfill()
    if out["wind_speed"].lt(0).any() or out["shortwave_irradiation"].lt(-5).any():
        log.warning("negative wind speed / irradiation in the input; clipped to 0")
    out["wind_speed"] = out["wind_speed"].clip(lower=0.0)
    out["shortwave_irradiation"] = out["shortwave_irradiation"].clip(lower=0.0)
    return out


def run_transform(simulate, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """``transform`` with another finite-element backend (see :func:`simulate_numpy`)."""
    return transform(df, *args, _simulate=simulate, **kwargs)


def transform(
    df: pd.DataFrame,
    virtual_sensors: list | None = None,
    timestep: str = "30min",
    input_channels: dict | None = None,
    sensor_ifc: str | None = None,
    confidence_level: float = 0.95,
    initial_condition_steps: int = 100,
    min_history_hours: float = 336.0,
    model_overrides: dict | None = None,
    position_uncertainty: bool = True,
    *,
    _simulate=None,
) -> pd.DataFrame:
    """Predict virtual sensor readings of the Nibelungen bridge with credible intervals.

    Parameters
    ----------
    df
        Measured inputs, schema ``Timestamp: datetime64[ns], <channel>: float64, ...``
        (see :data:`DEFAULT_INPUT_CHANNELS`, including the wind speed column, see
        :func:`.weather.add_wind_speed`).  It must cover the report window
        **plus** the spin-up history before it: the thermal state is rebuilt
        from scratch on every call.  14 days of history (``min_history_hours``)
        reproduce the calibrated set-up; less is accepted with a warning.  The
        report window is read from ``df.attrs["report_start"]`` and
        ``df.attrs["report_end"]`` (ISO-8601 strings or timestamps, both
        inclusive); either one that is missing defaults to the end of ``df``
        and 24 h before it, respectively.
    virtual_sensors
        List of dicts ``{"name": str, "quantity": str, "position": [x, y, z],
        "noise_std": float (optional)}``.  ``position`` is in simulation model
        coordinates [m] (see README); ``quantity`` one of ``temperature``,
        ``displacement_{x,y,z}``, ``inclination``, ``strain_{xx,yy,zz,xy,yz,xz}``.
        ``noise_std`` is a residual error in the output unit, added to the
        parametric (PCE) variance; only for sensors where one was calibrated
        (the measured pilot sensors).  Virtual sensors carry only the
        propagated embedded MFU (and the position term), default 0.  Optional, for the
        position term of temperature sensors: ``position_std`` (isotropic
        N(0, std²) in m around ``position``) or ``member`` + ``depth_range``
        ``[lo, hi]`` (depth behind the cavity face, m; ``position_std`` is then
        the in-plane std); see :mod:`.position`.  Default: the sensors of
        ``sensor_ifc`` if given, else those shipped in
        ``data/virtual_sensors.json`` (mapped from MKP's sensor IFC model).
    timestep
        Model time step (pandas offset alias); inputs are bin-averaged to it.
    input_channels
        Overrides of :data:`DEFAULT_INPUT_CHANNELS` (role -> column name).
    sensor_ifc
        Path to an IFC file whose sensor proxies define the virtual sensors;
        they are mapped to model coordinates on the fly (needs the ``ifc``
        extra).  Ignored if ``virtual_sensors`` is given.
    confidence_level
        Two-sided level of the ``_lower`` / ``_upper`` columns.
    initial_condition_steps
        Steps repeating the first input row before the history is run.
    min_history_hours
        History length below which a warning is logged.
    model_overrides
        Nested dict merged into ``data/model_config.json`` (expert use).
    position_uncertainty
        Include the sensor-position uncertainty in the predictive distribution of
        the **temperature** sensors that have a position prior (law of total
        variance: mean ``E_c E_x Y``, variance ``Var_c[E_x Y] + E_c[Var_x Y]``
        + ``noise_std``², 0 unless given).  IFC-derived sensors have it automatically; explicit
        sensors only with ``position_std`` or ``member`` + ``depth_range``.
        Mechanical sensors never have it.  ``False`` returns the result of the
        nominal position (identical to the previous versions).

    Returns
    -------
    pd.DataFrame
        ``Timestamp`` plus, per virtual sensor ``<name>``, the columns
        ``<name>`` (predictive mean), ``<name>_std`` (predictive standard
        deviation), ``<name>_lower`` and ``<name>_upper`` (Gaussian credible
        interval at ``confidence_level``), all float64.  Units are in
        ``result.attrs["units"]``.
    """
    simulate = _simulate or simulate_numpy
    cfg = load_model_config()
    if model_overrides:
        _deep_update(cfg, copy.deepcopy(model_overrides))
    channels = {**DEFAULT_INPUT_CHANNELS, **(input_channels or {})}
    if virtual_sensors is not None:
        sensors = list(virtual_sensors)
    elif sensor_ifc is not None:
        from .ifc import virtual_sensors_from_ifc

        sensors = virtual_sensors_from_ifc(sensor_ifc)
    else:
        sensors = load_default_sensors()
    if not sensors:
        raise ValueError("virtual_sensors is empty")
    names = [str(s["name"]) for s in sensors]
    if len(set(names)) != len(names):
        raise ValueError("virtual sensor names must be unique")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    forcing = _prepare_forcing(df, channels, timestep)
    ts = forcing.index
    report_start, report_end = df.attrs.get("report_start"), df.attrs.get("report_end")
    t_end = pd.Timestamp(report_end) if report_end else ts[-1]
    t_start = pd.Timestamp(report_start) if report_start else t_end - pd.Timedelta(hours=24) + pd.Timedelta(timestep)
    if ts.tz is not None:
        t_start = t_start.tz_localize(ts.tz) if t_start.tz is None else t_start.tz_convert(ts.tz)
        t_end = t_end.tz_localize(ts.tz) if t_end.tz is None else t_end.tz_convert(ts.tz)
    else:
        t_start, t_end = t_start.tz_localize(None), t_end.tz_localize(None)
    report = np.flatnonzero((ts >= t_start) & (ts <= t_end))
    if report.size == 0:
        raise ValueError(f"no input rows inside the report window [{t_start}, {t_end}]")
    first = int(report[0])
    history_h = (ts[first] - ts[0]).total_seconds() / 3600.0
    if history_h < min_history_hours:
        log.warning(
            "only %.0f h of spin-up history before %s (recommended >= %.0f h); "
            "the first predictions depend on the assumed initial state",
            history_h, ts[first], min_history_hours,
        )
    dt = pd.Timedelta(timestep).total_seconds()

    quantities = [str(s["quantity"]) for s in sensors]
    positions = np.array([s["position"] for s in sensors], dtype=np.float64)
    if positions.shape != (len(sensors), 3):
        raise ValueError("each virtual sensor needs a 3-D 'position'")
    unknown = sorted(set(quantities) - set(UNITS))
    if unknown:
        raise ValueError(f"unsupported quantity {unknown}; supported: {sorted(UNITS)}")
    mesh = Mesh.load(_data_path(cfg["mesh_file"]))
    mesh.locate(positions)  # same error for every backend

    unc = cfg["uncertainty"]
    nodes, weights = convection_rule(unc)      # (n_nodes, 2): c_top, c_ext

    F = forcing[_ORDER].to_numpy()[: int(report[-1]) + 1]

    # Position term: every temperature sensor with a prior is read at all its position
    # samples in the same model run (one probe matrix per node); the backends see only a
    # longer list of point sensors.
    samples: dict[int, np.ndarray] = {}
    if position_uncertainty:
        margin = float(load_ifc_config()["mapping"]["margin_m"])
        probe = section_probe(mesh)
        for i, sensor in enumerate(sensors):
            sm = position_samples({**sensor, "name": names[i]}, unc["position"], margin, mesh, probe)
            if sm is not None:
                samples[i] = sm
    if samples:
        ext_q, ext_p, slots = [], [], []
        for i, (q, p) in enumerate(zip(quantities, positions)):
            pts = samples.get(i, p[None, :])
            slots.append(slice(len(ext_q), len(ext_q) + len(pts)))
            ext_q += [q] * len(pts)
            ext_p.append(pts)
        mesh.locate(np.vstack(ext_p))                   # samples are inside the concrete by construction
        Y_ext = simulate(cfg, F, dt, nodes, int(initial_condition_steps), first, ext_q, np.vstack(ext_p))
        Y = np.stack([Y_ext[:, :, sl][..., 0] if i not in samples else Y_ext[:, :, sl].mean(-1)
                      for i, sl in enumerate(slots)], axis=2)
        mean, var_pce = weighted_moments(Y, weights)    # mechanical / no-prior sensors: as before
        for i in samples:
            mean[:, i], var_c, var_x = total_variance_moments(Y_ext[:, :, slots[i]], weights)
            var_pce[:, i] = var_c + var_x
    else:
        Y = simulate(cfg, F, dt, nodes, int(initial_condition_steps), first, quantities, positions)
        mean, var_pce = weighted_moments(Y, weights)
    noise = np.array([_noise_std(s, cfg) for s in sensors])
    std = np.sqrt(var_pce + noise**2)
    zq = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)

    out = {"Timestamp": ts[report]}
    for i, n in enumerate(names):
        out[n + OUTPUT_SUFFIXES["mean"]] = mean[:, i]
        out[n + OUTPUT_SUFFIXES["std"]] = std[:, i]
        out[n + OUTPUT_SUFFIXES["lower"]] = mean[:, i] - zq * std[:, i]
        out[n + OUTPUT_SUFFIXES["upper"]] = mean[:, i] + zq * std[:, i]
    result = pd.DataFrame(out)
    if result["Timestamp"].dt.tz is None:
        result["Timestamp"] = result["Timestamp"].astype("datetime64[ns]")
    result.attrs.update({
        "units": {n: UNITS[q] for n, q in zip(names, quantities)},
        "confidence_level": confidence_level,
        "spinup_history_hours": history_h,
        "quadrature_nodes": nodes.tolist(),
        "quadrature_weights": weights.tolist(),
        "backend": getattr(simulate, "backend_name", simulate.__name__),
        "element_degree": int(cfg["thermal"].get("degree", 1)),
        "position_uncertainty": {
            "enabled": bool(position_uncertainty),
            "sensors": [names[i] for i in sorted(samples)],
            "n_samples": int(unc["position"]["n_samples"]),
        },
    })
    return result


def simulate_numpy(cfg: dict, forcing: np.ndarray, dt: float, nodes: np.ndarray,
                   n_initial_steps: int, first: int, quantities: list[str],
                   positions: np.ndarray) -> np.ndarray:
    """NumPy/SciPy finite-element backend.

    Runs the thermal (and, if needed, mechanical) model once per quadrature
    node ``(c_top, c_ext)`` over all ``forcing`` rows and returns the readings of the
    rows ``first ... len(forcing)-1``, shape ``(n_nodes, n_rows, n_sensors)``.
    Every backend implements this signature.
    """
    mesh = Mesh.load(_data_path(cfg["mesh_file"]))
    degree = int(cfg["thermal"].get("degree", 1))
    if int(cfg["mechanical"].get("degree", degree)) != degree:
        raise ValueError("thermal and mechanical element degree must be equal (shared dof numbering)")
    space = Space(mesh, degree)
    PT, Pu, offset, is_thermal = build_probes(mesh, quantities, positions, space)
    thermal = ThermalModel(mesh, cfg["thermal"], space)

    # The displacement is linear in the temperature, so a mechanical reading is
    # y = W (T - T_ref).  One adjoint solve per mechanical sensor gives W; the
    # time loop then needs no mechanical solve and no factorisation of K.
    W = T_ref = None
    mech_rows = np.flatnonzero(~is_thermal)
    if mech_rows.size:
        mech = ThermoElasticModel(mesh, cfg["mechanical"], space)
        W = mech.sensor_weights(Pu[mech_rows], cfg["mechanical"].get("solver", "ilu"))
        T_ref = mech.T_ref
        del mech

    Y = np.empty((len(nodes), len(forcing) - first, len(quantities)))
    for j, c in enumerate(nodes):
        log.info("quadrature node %d/%d: c_top = %.4f, c_ext = %.4f", j + 1, len(nodes), *c)

        def collect(k: int, T: np.ndarray, j=j) -> None:
            y = PT @ T + offset
            if W is not None:
                y[mech_rows] = W @ (T - T_ref)
            Y[j, k - first] = y

        thermal.run(forcing, dt, tuple(float(x) for x in c), n_initial_steps, first, collect)
    return Y


simulate_numpy.backend_name = "numpy"
