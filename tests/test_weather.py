"""Checks ``add_weather`` offline: Open-Meteo is replaced by a stub."""

import numpy as np
import pandas as pd
import pytest

from nibelungen_virtual_sensors import DEFAULT_INPUT_CHANNELS, add_weather, add_wind_speed
from nibelungen_virtual_sensors import weather

IRR, WIND = DEFAULT_INPUT_CHANNELS["shortwave_irradiation"], DEFAULT_INPUT_CHANNELS["wind_speed"]


def _hours(params):
    return pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta("23h"), freq="h")


@pytest.fixture
def requests(monkeypatch):
    """Stub of the HTTP call: wind 3 m/s; shortwave_radiation = the hour of its label times 10 W/m²."""
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        t = _hours(params)
        hourly = {"time": [f"{x:%Y-%m-%dT%H:%M}" for x in t]}
        if "wind_speed_10m" in params["hourly"]:
            hourly["wind_speed_10m"] = [3.0] * len(t)
        if "shortwave_radiation" in params["hourly"]:
            hourly["shortwave_radiation"] = list(10.0 * t.hour)
        return {"hourly": hourly}

    monkeypatch.setattr(weather, "_get", fake_get)
    return calls


def _frame(start, periods=96):
    t = pd.date_range(start, periods=periods, freq="30min")
    return pd.DataFrame({"Timestamp": t, IRR: 500.0})


def test_irradiation_replaced_from_the_fault_on(requests):
    df = _frame("2025-08-31 12:00")
    df.attrs["report_start"] = "2025-09-01"
    out = add_weather(df)
    assert len(requests) == 1 and requests[0]["hourly"] == "wind_speed_10m,shortwave_radiation"
    before = out["Timestamp"] < weather.IRRADIATION_FAULT_START
    assert (out.loc[before, IRR] == 500.0).all()
    assert (out[WIND] == 3.0).all()
    assert out.attrs == df.attrs
    # Open-Meteo labels the mean of the preceding hour: the 10:00 value (100) is the mean of 09:00-10:00,
    # so the bin 09:00-09:30 (centre 09:15) gets 100 - 2.5 and the bin 09:30-10:00 gets 100 + 2.5
    g = out.set_index("Timestamp")[IRR]
    assert g["2025-09-01 09:00"] == pytest.approx(97.5)
    assert g["2025-09-01 09:30"] == pytest.approx(102.5)


def test_before_the_fault_with_constant_wind_needs_no_network(requests):
    out = add_weather(_frame("2024-06-01"), wind_speed=2.7)
    assert requests == []
    assert (out[IRR] == 500.0).all() and (out[WIND] == 2.7).all()


def test_missing_irradiation_column_is_fetched(requests):
    out = add_wind_speed(_frame("2026-11-01").drop(columns=IRR), 2.7)
    assert requests[0]["hourly"] == "shortwave_radiation"
    assert out[IRR].notna().all() and (out[IRR] >= 0).all()


def test_irradiation_modes(requests):
    df = _frame("2026-11-01")
    assert (add_weather(df, 2.7, irradiation="measured")[IRR] == 500.0).all()
    assert requests == []
    assert (add_weather(_frame("2024-06-01"), 2.7, irradiation="open-meteo")[IRR] != 500.0).any()
    with pytest.raises(ValueError, match="irradiation"):
        add_weather(df, irradiation="sensor")
