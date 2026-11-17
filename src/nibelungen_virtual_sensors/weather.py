"""Weather inputs from Open-Meteo: the wind speed, and the irradiation after the sensor fault.

* **Wind speed.**  MKP does not measure it.  The calibration used hourly
  ``wind_speed_10m`` [m/s] of the Open-Meteo archive API at the bridge
  (49.63 N, 8.38 E, ``sensors.open_meteo.location`` of the research pipeline),
  kept here so the calibrated convection factors stay consistent.
* **Shortwave irradiation.**  The bridge pyranometer (``F_plus_000S_KaS-o-``)
  has been faulty since September 2025.  From :data:`IRRADIATION_FAULT_START`
  on, the measured column is replaced by Open-Meteo ``shortwave_radiation``
  [W/m²] (global horizontal irradiance) at the same grid point.

The archive lags real time by about five days, so recent days are taken from
the forecast API (``past_days``), i.e. from a different model chain than the
calibration data.

Standard library only (``urllib``); network access is needed at call time,
and only when something has to be fetched.

:func:`add_weather` is a processing step of its own: it takes the MKP frame,
attaches the wind speed column, replaces the irradiation where the sensor is
faulty and returns the frame that :func:`.transform` expects.  It does not
need the model, so it can run in a separate container.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .transform import DEFAULT_INPUT_CHANNELS

log = logging.getLogger(__name__)

LOCATION = (49.63, 8.38)
#: First time stamp (UTC) at which the bridge irradiation sensor is faulty.
IRRADIATION_FAULT_START = pd.Timestamp("2025-09-01")
_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_LAG_DAYS = 6
_WIND, _IRR = "wind_speed_10m", "shortwave_radiation"


def _get(url: str, params: dict, timeout: float) -> dict:
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{q}", timeout=timeout) as r:  # noqa: S310 (fixed https host)
        return json.loads(r.read().decode())


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t


def fetch_open_meteo(start, end, variables=(_WIND,), location=LOCATION,
                     timeout: float = 30.0) -> pd.DataFrame:
    """Hourly Open-Meteo ``variables`` in UTC covering [start, end] (tz-naive index).

    Wind speed in m/s; ``shortwave_radiation`` is the mean of the preceding hour [W/m²].
    """
    t0, t1 = (_utc_naive(t) for t in (start, end))
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    base = {
        "latitude": location[0],
        "longitude": location[1],
        "hourly": ",".join(variables),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    parts = []
    split = today - pd.Timedelta(days=_ARCHIVE_LAG_DAYS)
    try:
        if t0 < split:
            a_end = min(t1, split - pd.Timedelta(days=1))
            parts.append(_get(_ARCHIVE, {**base, "start_date": f"{t0:%Y-%m-%d}",
                                         "end_date": f"{a_end:%Y-%m-%d}"}, timeout))
        if t1 >= split:
            f_start = max(t0, split)
            parts.append(_get(_FORECAST, {**base, "start_date": f"{f_start:%Y-%m-%d}",
                                          "end_date": f"{t1:%Y-%m-%d}"}, timeout))
    except OSError as exc:
        raise RuntimeError(
            f"could not fetch {', '.join(variables)} from Open-Meteo; for the wind use a constant "
            "wind_speed=<m/s> in add_weather, or supply the columns yourself"
        ) from exc
    frames = [pd.DataFrame({v: p["hourly"][v] for v in variables},
                           index=pd.to_datetime(p["hourly"]["time"]), dtype="float64") for p in parts]
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def fetch_wind_speed(start, end, location=LOCATION, timeout: float = 30.0) -> pd.Series:
    """Hourly 10 m wind speed [m/s] in UTC covering [start, end] (tz-naive index)."""
    return fetch_open_meteo(start, end, (_WIND,), location, timeout)[_WIND].dropna()


def _interp(s: pd.Series, at: pd.DatetimeIndex) -> np.ndarray:
    s = s.dropna()
    return s.reindex(s.index.union(at)).interpolate("time").ffill().bfill().reindex(at).to_numpy()


def add_weather(df: pd.DataFrame, wind_speed: float | str = "open-meteo",
                irradiation: str = "auto", input_channels: dict | None = None,
                fault_start=IRRADIATION_FAULT_START) -> pd.DataFrame:
    """Return a copy of ``df`` with the wind speed attached and the faulty irradiation replaced.

    ``df`` has the schema of :func:`.transform`, without the wind column and,
    after the irradiation fault, possibly without the irradiation column.  Its
    ``attrs`` (e.g. the report window) are kept.  An existing wind column is
    replaced.  Time stamps without a time zone are taken as UTC.

    Parameters
    ----------
    wind_speed
        ``"open-meteo"`` fetches ``wind_speed_10m`` at the bridge (needs
        network) and interpolates it to the time stamps of ``df``; a number is
        a constant wind speed [m/s].
    irradiation
        ``"auto"`` keeps the measured irradiation before ``fault_start`` and
        takes Open-Meteo ``shortwave_radiation`` from then on (all rows if the
        column is missing); ``"open-meteo"`` takes it for all rows;
        ``"measured"`` leaves the column untouched.  Open-Meteo gives the mean
        of the preceding hour; it is placed at the centre of that hour and
        interpolated to the centre of each row's interval, as the measured
        bin means.
    input_channels
        Overrides of :data:`.DEFAULT_INPUT_CHANNELS` (role -> column name).
    fault_start
        First time stamp (UTC) at which the irradiation sensor is faulty.
    """
    channels = {**DEFAULT_INPUT_CHANNELS, **(input_channels or {})}
    wind_col, irr_col = channels["wind_speed"], channels["shortwave_irradiation"]
    if irradiation not in ("auto", "open-meteo", "measured"):
        raise ValueError(f"irradiation must be 'auto', 'open-meteo' or 'measured', not {irradiation!r}")
    if "Timestamp" in df.columns:
        index = pd.DatetimeIndex(df["Timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    else:
        raise ValueError("df needs a 'Timestamp' column (datetime64[ns])")
    if len(index) == 0:
        raise ValueError("df is empty")
    utc = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index

    if irradiation == "measured":
        replace = np.zeros(len(utc), bool)
    elif irradiation == "open-meteo" or irr_col not in df.columns:
        replace = np.ones(len(utc), bool)
    else:
        replace = np.asarray(utc >= _utc_naive(fault_start))

    variables = ([_WIND] if wind_speed == "open-meteo" else []) + ([_IRR] if replace.any() else [])
    if variables:
        om = fetch_open_meteo(utc.min() - pd.Timedelta(hours=1), utc.max() + pd.Timedelta(hours=2), variables)

    out = df.copy()
    if wind_speed == "open-meteo":
        out[wind_col] = _interp(om[_WIND], utc)
        log.info("wind speed: Open-Meteo wind_speed_10m, %d values", om[_WIND].notna().sum())
    else:
        log.info("wind speed: constant %.2f m/s", float(wind_speed))
        out[wind_col] = float(wind_speed)

    if replace.any():
        g = om[_IRR].copy()
        g.index = g.index - pd.Timedelta("30min")       # mean of the preceding hour -> centred at h - 30 min
        steps = pd.Series(utc).diff().dropna()
        half = (steps.median() if len(steps) else pd.Timedelta("30min")) / 2
        g_rows = np.clip(_interp(g, utc + half), 0.0, None)
        measured = (pd.to_numeric(out[irr_col], errors="coerce").to_numpy("float64")
                    if irr_col in out.columns else np.full(len(utc), np.nan))
        out[irr_col] = np.where(replace, g_rows, measured)
        log.info("shortwave irradiation: Open-Meteo shortwave_radiation for %d of %d rows "
                 "(sensor faulty since %s)", replace.sum(), len(utc), f"{_utc_naive(fault_start):%Y-%m-%d}")
    out.attrs = dict(df.attrs)
    return out


def add_wind_speed(df: pd.DataFrame, wind_speed: float | str = "open-meteo",
                   input_channels: dict | None = None, irradiation: str = "auto") -> pd.DataFrame:
    """Former name of :func:`add_weather`; it also replaces the faulty irradiation."""
    return add_weather(df, wind_speed, irradiation, input_channels)
