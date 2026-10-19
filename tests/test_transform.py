"""Demonstrates and checks the platform entry point ``transform``."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nibelungen_virtual_sensors import DEFAULT_INPUT_CHANNELS, add_wind_speed, transform

SAMPLE = Path(__file__).parents[1] / "examples" / "mkp_sample_2024-06-01_15.csv"
# The unit tests run on a small mesh of the same geometry (3k nodes, P2 by default);
# the slow test below uses the packaged production mesh.
COARSE = Path(__file__).parent / "data" / "coarse_hinge_mesh.npz"
OV = {"mesh_file": str(COARSE)}

SENSORS = [
    {"name": "T_bottom", "quantity": "temperature", "position": [2.85, -5.264, 0.0], "noise_std": 0.1468},
    {"name": "T_top", "quantity": "temperature", "position": [2.85, -0.171, 0.0]},
    {"name": "tilt", "quantity": "inclination", "position": [-2.25, -0.2, 38.3]},
    {"name": "uy_tip", "quantity": "displacement_y", "position": [0.0, -0.1, 39.5]},
]


@pytest.fixture(scope="module")
def inputs() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE, parse_dates=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
    return df[["Timestamp", *DEFAULT_INPUT_CHANNELS.values()]]


@pytest.fixture(scope="module")
def short_run(inputs):
    # 2 days: 1 day spin-up + 1 report day (production: >= 14 days spin-up)
    return transform(inputs.tail(96), SENSORS, model_overrides=OV)


def test_output_schema(short_run):
    out = short_run
    assert out.columns[0] == "Timestamp"
    assert out["Timestamp"].dtype == "datetime64[ns]"
    expected = [f"{s['name']}{suf}" for s in SENSORS for suf in ("", "_std", "_lower", "_upper")]
    assert list(out.columns[1:]) == expected
    assert all(out[c].dtype == "float64" for c in expected)
    assert len(out) == 48 and out["Timestamp"].is_monotonic_increasing
    assert out["Timestamp"].iloc[-1] == pd.Timestamp("2024-06-15 23:30")
    assert out.notna().all().all()
    assert out.attrs["units"] == {"T_bottom": "°C", "T_top": "°C", "tilt": "mm/m", "uy_tip": "mm"}
    assert out.attrs["element_degree"] == 2


def test_intervals_bracket_mean(short_run):
    for s in SENSORS:
        n = s["name"]
        assert (short_run[f"{n}_lower"] < short_run[n]).all()
        assert (short_run[n] < short_run[f"{n}_upper"]).all()
    # calibrated noise is a floor of the predictive std
    assert (short_run["T_bottom_std"] >= 0.1468).all()


def test_report_window_and_constant_wind(inputs):
    df = inputs.drop(columns=["wind_speed_10m"]).tail(60)
    df.attrs.update(report_start="2024-06-15T18:00", report_end="2024-06-15T20:00")
    out = transform(add_wind_speed(df, 2.7), SENSORS[:1], model_overrides=OV)
    assert list(out["Timestamp"].dt.strftime("%H:%M")) == ["18:00", "18:30", "19:00", "19:30", "20:00"]


def test_missing_channel_is_an_error(inputs):
    with pytest.raises(ValueError, match="F_plus_000TA_KaS-o-_Avg1"):
        transform(inputs.drop(columns=["F_plus_000TA_KaS-o-_Avg1"]).tail(10), SENSORS[:1], model_overrides=OV)
    with pytest.raises(ValueError, match="wind_speed"):
        transform(inputs.drop(columns=["wind_speed_10m"]).tail(10), SENSORS[:1], model_overrides=OV)


def test_sensor_outside_concrete_is_an_error(inputs):
    bad = [{"name": "air", "quantity": "temperature", "position": [2.85, -2.5, 5.0]}]  # cavity
    with pytest.raises(ValueError, match="outside the bridge mesh"):
        transform(inputs.tail(10), bad, model_overrides=OV)


@pytest.mark.slow
def test_full_daily_run_tracks_measurements():
    """14 days spin-up + 1 report day on the production mesh (~4 min, 1.3 GB), compared with the
    real pilot sensors.  Run with ``pytest -m slow``."""
    raw = pd.read_csv(SAMPLE, parse_dates=["Timestamp"])
    # free end at the hinge (z = 40.09 as in sensors/sensors_mechanical.json)
    out = transform(raw, [*SENSORS[:3], {**SENSORS[3], "position": [0.0, -0.001, 40.09]}])
    meas = raw.set_index("Timestamp").loc[out["Timestamp"]]
    for name, col in [("T_bottom", "E_plus_040TU_HS--u-_Avg1"), ("T_top", "E_plus_040TU_HS--o-_Avg1")]:
        y = meas[col].to_numpy()
        inside = (y >= out[f"{name}_lower"].to_numpy()) & (y <= out[f"{name}_upper"].to_numpy())
        assert inside.mean() > 0.85
    r = np.corrcoef(meas["E_plus_423NU_HSN-o-_Avg1"], out["tilt"])[0, 1]
    assert r > 0.9


def test_virtual_sensors_carry_no_residual_error():
    """Only the embedded MFU (and the position) is propagated; sigma_k stays with the pilot sensors."""
    from nibelungen_virtual_sensors.transform import load_default_sensors, load_model_config
    assert all("noise_std" not in s for s in load_default_sensors())
    defaults = load_model_config()["uncertainty"]["default_noise_std"]
    assert all(v == 0.0 for k, v in defaults.items() if not k.startswith("_"))
