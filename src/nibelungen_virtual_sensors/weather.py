"""Wind speed from Open-Meteo, the one model input MKP does not measure.

The calibration used hourly ``wind_speed_10m`` [m/s] of the Open-Meteo
archive API at (49.9 N, 8.35 E) — the grid point configured in the research
pipeline (``sensors.open_meteo.location``), kept here so the calibrated c_c
stays consistent.  The archive lags real time by about five days, so recent
days are taken from the forecast API (``past_days``), i.e. from a different
model chain than the calibration data.

Standard library only (``urllib``); network access is needed at call time.

:func:`add_wind_speed` is a processing step of its own: it takes the MKP frame,
attaches the wind speed column and returns the frame that :func:`.transform`
expects.  It does not need the model, so it can run in a separate container.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

import pandas as pd

from .transform import DEFAULT_INPUT_CHANNELS

log = logging.getLogger(__name__)

LOCATION = (49.9, 8.38)
_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_LAG_DAYS = 6


def _get(url: str, params: dict, timeout: float) -> dict:
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{q}", timeout=timeout) as r:  # noqa: S310 (fixed https host)
        return json.loads(r.read().decode())


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t


def fetch_wind_speed(start, end, location=LOCATION, timeout: float = 30.0) -> pd.Series:
    """Hourly 10 m wind speed [m/s] in UTC covering [start, end] (tz-naive index)."""
    t0, t1 = (_utc_naive(t) for t in (start, end))
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    base = {
        "latitude": location[0],
        "longitude": location[1],
        "hourly": "wind_speed_10m",
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
            "could not fetch wind speed from Open-Meteo; use a constant wind_speed=<m/s> "
            "in add_wind_speed or supply the wind column yourself"
        ) from exc
    series = [pd.Series(p["hourly"]["wind_speed_10m"], index=pd.to_datetime(p["hourly"]["time"]),
                        dtype="float64") for p in parts]
    s = pd.concat(series)
    s = s[~s.index.duplicated(keep="first")].sort_index().dropna()
    return s.rename("wind_speed_10m")


def add_wind_speed(df: pd.DataFrame, wind_speed: float | str = "open-meteo",
                   input_channels: dict | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with the wind speed column [m/s] attached.

    ``df`` has the schema of :func:`.transform` without the wind column.  Its
    ``attrs`` (e.g. the report window) are kept.  An existing wind column is
    replaced.

    Parameters
    ----------
    wind_speed
        ``"open-meteo"`` fetches ``wind_speed_10m`` at the calibration grid
        point (needs network) and interpolates it to the time stamps of ``df``;
        a number is a constant wind speed [m/s].
    input_channels
        Overrides of :data:`.DEFAULT_INPUT_CHANNELS` (role -> column name).
    """
    col = {**DEFAULT_INPUT_CHANNELS, **(input_channels or {})}["wind_speed"]
    if "Timestamp" in df.columns:
        index = pd.DatetimeIndex(df["Timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    else:
        raise ValueError("df needs a 'Timestamp' column (datetime64[ns])")
    if len(index) == 0:
        raise ValueError("df is empty")
    if wind_speed == "open-meteo":
        utc = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
        w = fetch_wind_speed(utc.min() - pd.Timedelta(hours=1), utc.max() + pd.Timedelta(hours=1))
        w = w.reindex(w.index.union(utc)).interpolate("time").ffill().bfill().reindex(utc)
        log.info("wind speed: Open-Meteo wind_speed_10m, %d values", w.notna().sum())
        values = w.to_numpy()
    else:
        log.info("wind speed: constant %.2f m/s", float(wind_speed))
        values = float(wind_speed)
    out = df.copy()
    out[col] = values
    out.attrs = dict(df.attrs)
    return out
