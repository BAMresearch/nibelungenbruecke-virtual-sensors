"""Second layer: IFC sensors -> model coordinates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nibelungen_virtual_sensors.fem.mesh import Mesh
from nibelungen_virtual_sensors.ifc import model_probe
from nibelungen_virtual_sensors.ifc.frame import BridgeFrame
from nibelungen_virtual_sensors.ifc.mapping import parse_position_text, parse_station
from nibelungen_virtual_sensors.weather import add_wind_speed
from nibelungen_virtual_sensors.transform import _data_path, load_default_sensors, load_ifc_config, transform

TUS_IFC = Path(__file__).parents[1] / "examples" / "20260916_70021_02_DZ-Sim_Nibelungbrücke-TUS.ifc"


def test_parse_station():
    assert parse_station("NiE+233") == pytest.approx(23.3)
    assert parse_station("NiE+040") == pytest.approx(4.0)
    assert parse_station("NiF-105") is None          # other pier axis


@pytest.mark.parametrize("text,expected", [
    ("Oben, zentriert im südlichen Hohlkasten; 20 cm tief von der Innenseite",
     {"member": "deck", "depth": 0.20, "fraction": 0.5}),
    ("Unten, zentriert im südlichen Hohlkasten; 10 cm tief von der Innenseite",
     {"member": "bottom_slab", "depth": 0.10, "fraction": 0.5}),
    ("Mittig im südlichen Steg des südlichen Hohlkastens; 20 cm tief von der Innenseite",
     {"member": "web_south", "depth": 0.20, "fraction": 0.5}),
    ("Mittig im nördlichen Steg des südlichen Hohlkastens; 20 cm tief von der Innenseite",
     {"member": "web_north", "depth": 0.20, "fraction": 0.5}),
])
def test_parse_position_text(text, expected):
    assert parse_position_text(text) == pytest.approx(expected)


def test_model_cell_at_pilot_section():
    cell = model_probe().cavity_near(2.85, -2.6, 0.5)
    # pilot cross-section (model_parameters_2d_pilot.json / README)
    assert (cell.a0, cell.a1) == pytest.approx((2.25, 3.45), abs=0.02)
    assert (cell.ceiling, cell.top) == pytest.approx((-0.25, 0.0), abs=0.02)


def test_frame_is_a_mirror():
    """South of the axis must map to model x > 0 (the instrumented box)."""
    f = BridgeFrame.load(_data_path("bridge_frame.json"))
    south = np.array([f.origin_e, f.origin_n, 100.0]) + 3.0 * np.r_[f.axis_n, -f.axis_e, 0.0]
    a, v, z = f.to_section(south)[0]
    assert a == pytest.approx(3.0, abs=1e-6) and z == pytest.approx(0.0, abs=1e-6)
    assert f.mirror_x == -1.0
    assert f.mirror_x == load_ifc_config()["frame"]["mirror_x"]
    assert f.model_z(23.3) == pytest.approx(19.3)
    assert np.degrees(np.arctan2(f.axis_e, f.axis_n)) == pytest.approx(73.9, abs=0.2)


def test_packaged_sensors_sit_in_their_members():
    mesh = Mesh.load(_data_path("mesh_3d.npz"))
    sensors = {s["name"]: s for s in load_default_sensors()}
    assert set(sensors) == {"E+233TUS_HS--o-", "E+233TUS_HS--u-", "E+233TUS_HSS-m-", "E+233TUS_HSN-m-"}
    mesh.locate([s["position"] for s in sensors.values()])  # all inside the concrete
    for s in sensors.values():
        assert s["mapping"]["station_m"] == pytest.approx(23.3)                  # NiE+233
        assert s["mapping"]["station_ifc_geometry_m"] == pytest.approx(23.3, abs=0.05)
        assert s["position"][2] == pytest.approx(23.3 - 4.0)                    # z = 0 at E+4.0
    assert sensors["E+233TUS_HS--o-"]["position"][1] == pytest.approx(-0.05)      # 20 cm above deck underside
    assert sensors["E+233TUS_HSS-m-"]["position"][0] == pytest.approx(3.65)       # 20 cm into the south web
    assert sensors["E+233TUS_HSN-m-"]["position"][0] == pytest.approx(2.05)       # 20 cm into the north web
    cell = model_probe().cavity_near(2.85, -1.5, 19.3)
    assert sensors["E+233TUS_HS--u-"]["position"][1] == pytest.approx(cell.floor - 0.10, abs=1e-3)


def test_ifc_reproduces_packaged_sensors():
    pytest.importorskip("ifcopenshell")
    from nibelungen_virtual_sensors.ifc import virtual_sensors_from_ifc

    fresh = virtual_sensors_from_ifc(str(TUS_IFC))
    packaged = load_default_sensors()
    assert [(s["name"], s["position"]) for s in fresh] == [(s["name"], s["position"]) for s in packaged]


def test_transform_from_ifc_with_open_meteo_wind():
    """Fully automatic: sensors from the IFC file, wind from Open-Meteo."""
    pytest.importorskip("ifcopenshell")
    sample = Path(__file__).parents[1] / "examples" / "mkp_sample_2024-06-01_15.csv"
    df = pd.read_csv(sample, parse_dates=["Timestamp"]).drop(columns=["wind_speed_10m"]).tail(96)
    try:
        # P1 on the packaged mesh keeps this test short; the IFC mapping does not depend on the degree
        out = transform(add_wind_speed(df), sensor_ifc=str(TUS_IFC),
                        model_overrides={"thermal": {"degree": 1}, "mechanical": {"degree": 1}})
    except RuntimeError as exc:
        if "Open-Meteo" in str(exc):
            pytest.skip("no network access to Open-Meteo")
        raise
    assert "E+233TUS_HS--o-_upper" in out.columns and len(out) == 48


def test_frame_anchor_comes_from_config():
    cfg = load_ifc_config()["frame"]
    f = BridgeFrame.load(_data_path("bridge_frame.json"), {**cfg, "model_z0_station": 10.0})
    assert f.model_z(23.3) == pytest.approx(13.3)
    assert "model_z0_station" not in json.loads(_data_path("bridge_frame.json").read_text())
