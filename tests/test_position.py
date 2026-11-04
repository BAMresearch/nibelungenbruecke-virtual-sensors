"""Sensor-position uncertainty in ``transform`` (law of total variance).

Runs on the small mesh of the hinge-end geometry; the E+23.3 sensors are placed on it with the
package's own placement rule.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nibelungen_virtual_sensors import DEFAULT_INPUT_CHANNELS, transform
from nibelungen_virtual_sensors.fem.mesh import Mesh
from nibelungen_virtual_sensors.ifc.mapping import _place_in_model
from nibelungen_virtual_sensors.position import _place, position_samples, section_probe
from nibelungen_virtual_sensors.transform import load_ifc_config, load_model_config
from nibelungen_virtual_sensors.uq import convection_rule, lognormal_hermite_rule, total_variance_moments, weighted_moments

SAMPLE = Path(__file__).parents[1] / "examples" / "mkp_sample_2024-06-01_15.csv"
COARSE = Path(__file__).parent / "data" / "coarse_hinge_mesh.npz"
OV = {"mesh_file": str(COARSE)}
MARGIN = load_ifc_config()["mapping"]["margin_m"]

# (member, d_text, d_ifc) of the E+23.3 sensors (data/virtual_sensors.json)
IFC_LIKE = {"deck": (0.20, 0.075), "bottom_slab": (0.10, -0.0255), "web_south": (0.20, 0.075),
            "web_north": (0.20, 0.075)}


@pytest.fixture(scope="module")
def mesh():
    return Mesh.load(COARSE)


@pytest.fixture(scope="module")
def probe(mesh):
    return section_probe(mesh)


def _ifc_like(probe, member, d_text, d_ifc, z=19.3, name=None):
    xyz, d = _place_in_model(probe, 2.85, z, member, d_text, 0.5, MARGIN)
    return {"name": name or f"T_{member}", "quantity": "temperature", "position": [float(c) for c in xyz],
            "mapping": {"member": member, "depth_behind_cavity_face_m": d,
                        "depth_in_ifc_geometry_m": d_ifc}}


@pytest.fixture(scope="module")
def sensors(probe):
    return [_ifc_like(probe, m, dt, di) for m, (dt, di) in IFC_LIKE.items()]


@pytest.fixture(scope="module")
def inputs() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE, parse_dates=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
    return df[["Timestamp", *DEFAULT_INPUT_CHANNELS.values()]].tail(96)


@pytest.fixture(scope="module")
def tilt():
    return {"name": "tilt", "quantity": "inclination", "position": [-2.25, -0.2, 38.3]}


@pytest.fixture(scope="module")
def runs(inputs, sensors, tilt):
    allv = [*sensors, tilt]
    return {"on": transform(inputs, allv, model_overrides=OV),
            "off": transform(inputs, allv, model_overrides=OV, position_uncertainty=False),
            "plain": transform(inputs, [{k: v for k, v in s.items() if k != "mapping"} for s in allv],
                               model_overrides=OV)}


# ── (i) switch off == previous result ─────────────────────────────────────────

def test_switch_off_is_the_previous_result(runs):
    # without a position prior (no mapping block) the previous code path is taken: bit-for-bit
    pd.testing.assert_frame_equal(runs["off"], runs["plain"], check_exact=True)
    assert runs["off"].attrs["position_uncertainty"]["enabled"] is False
    assert runs["off"].attrs["position_uncertainty"]["sensors"] == []


def test_switch_off_matches_the_nominal_probe_run(inputs, sensors, runs):
    """The old transform body, written out: nominal position, PCE moments, + noise."""
    from nibelungen_virtual_sensors.transform import _prepare_forcing, DEFAULT_INPUT_CHANNELS as CH, simulate_numpy

    cfg = load_model_config()
    cfg["mesh_file"] = str(COARSE)
    unc = cfg["uncertainty"]
    nodes, w = convection_rule(unc)
    forcing = _prepare_forcing(inputs, CH, "30min")
    F = forcing[["air_temperature", "cavity_temperature", "shortwave_irradiation", "wind_speed"]].to_numpy()
    pos = np.array([s["position"] for s in sensors])
    Y = simulate_numpy(cfg, F, 1800.0, nodes, 100, len(F) - 48, ["temperature"] * len(sensors), pos)
    mean, var = weighted_moments(Y, w)
    off = runs["off"]
    for i, s in enumerate(sensors):
        np.testing.assert_array_equal(off[s["name"]].to_numpy(), mean[:, i])
    # no noise_std is given, so the generic temperature value applies
    std0 = np.sqrt(var + unc["default_noise_std"]["temperature"] ** 2)
    for i, s in enumerate(sensors):
        np.testing.assert_array_equal(off[s["name"] + "_std"].to_numpy(), std0[:, i])


def test_schema_unchanged_and_mechanics_untouched(runs, sensors):
    on, off = runs["on"], runs["off"]
    assert list(on.columns) == list(off.columns)
    assert len(on.columns) == 1 + 4 * (len(sensors) + 1)
    for suf in ("", "_std", "_lower", "_upper"):
        np.testing.assert_array_equal(on["tilt" + suf].to_numpy(), off["tilt" + suf].to_numpy())
    assert on.attrs["position_uncertainty"]["sensors"] == [s["name"] for s in sensors]


def test_position_term_changes_the_temperature_distribution(runs, sensors):
    on, off = runs["on"], runs["off"]
    for s in sensors:
        n = s["name"]
        assert (on[n + "_lower"] < on[n]).all() and (on[n] < on[n + "_upper"]).all()
        assert np.abs(on[n] - off[n]).max() > 1e-3                 # the mean moves
    # the deck depth 5.5-22 cm behind the cavity face has a clear effect on the daily cycle
    assert (on["T_deck_std"] - off["T_deck_std"]).abs().max() > 0.01


# ── (ii) zero tolerance and zero sigma: the term vanishes ─────────────────────

def test_no_position_spread_no_position_term(inputs, probe, runs):
    flat = [_ifc_like(probe, "deck", 0.15, 0.15, name="flat_deck"),
            _ifc_like(probe, "web_south", 0.15, 0.15, name="flat_web")]
    ov = {**OV, "uncertainty": {"position": {"tolerance_m": 0.0, "in_plane_std_m": 0.0}}}
    on = transform(inputs, flat, model_overrides=ov)
    off = transform(inputs, flat, model_overrides=ov, position_uncertainty=False)
    assert on.attrs["position_uncertainty"]["sensors"] == ["flat_deck", "flat_web"]
    for c in on.columns[1:]:
        np.testing.assert_allclose(on[c], off[c], rtol=0, atol=1e-10, err_msg=c)


def test_isotropic_zero_std_has_no_term(probe, mesh):
    pos_cfg = load_model_config()["uncertainty"]["position"]
    s = {"name": "x", "quantity": "temperature", "position": [2.85, -0.1, 10.0], "position_std": 0.0}
    assert position_samples(s, pos_cfg, MARGIN, mesh, probe) is None
    assert position_samples({k: v for k, v in s.items() if k != "position_std"}, pos_cfg, MARGIN, mesh, probe) is None


# ── (iii) law of total variance against brute force ───────────────────────────

def test_total_variance_equals_pooled_ensemble():
    rng = np.random.default_rng(3)
    n_c, n_t, n_x = 3, 5, 7
    Y = rng.normal(size=(n_c, n_t, n_x)) * np.array([1.0, 2.0, 0.5])[:, None, None] + 10.0
    w = np.array([0.8, 0.19, 0.01])
    mean, var_c, var_x = total_variance_moments(Y, w)
    # brute force: pooled ensemble, node weights w_i, uniform weights 1/n_x over the samples
    for t in range(n_t):
        pts = Y[:, t, :]
        wt = np.repeat(w[:, None] / n_x, n_x, axis=1)
        m = (wt * pts).sum()
        v = (wt * (pts - m) ** 2).sum()
        assert mean[t] == pytest.approx(m, rel=1e-13)
        assert var_c[t] + var_x[t] == pytest.approx(v, rel=1e-12)
        # and each term separately
        mc = pts.mean(1)
        assert var_c[t] == pytest.approx((w * (mc - m) ** 2).sum(), rel=1e-12)
        assert var_x[t] == pytest.approx((w * pts.var(1)).sum(), rel=1e-12)


def test_total_variance_linear_case():
    """Y = a(c) + b x with x uniform: Var_c[E_x Y] = Var(a), E_c Var_x Y = b² Var(x)."""
    nodes, w = lognormal_hermite_rule(1.28, 2.052, 5)
    c_mean = (w * nodes).sum(); c_var = (w * (nodes - c_mean) ** 2).sum()   # moments of c under the rule
    n = 4096
    x = (np.arange(n) + 0.5) / n                                # uniform on (0, 1)
    b = 2.5
    Y = nodes[:, None, None] * 3.0 + b * x[None, None, :]
    mean, var_c, var_x = total_variance_moments(Y, w)
    assert var_c[0] == pytest.approx(9 * c_var, rel=1e-9)
    assert var_x[0] == pytest.approx(b**2 / 12.0, rel=1e-6)
    assert mean[0] == pytest.approx(3 * c_mean + b / 2, rel=1e-9)


# ── (iv) samples lie inside the member ────────────────────────────────────────

@pytest.mark.parametrize("member", list(IFC_LIKE))
def test_samples_inside_the_member(member, probe, mesh):
    pos_cfg = load_model_config()["uncertainty"]["position"]
    s = _ifc_like(probe, member, *IFC_LIKE[member])
    pts = position_samples({**s, "name": member}, pos_cfg, MARGIN, mesh, probe)
    assert pts.shape == (pos_cfg["n_samples"], 3)
    assert mesh.contains(pts).all()
    again = position_samples({**s, "name": member}, pos_cfg, MARGIN, mesh, probe)
    np.testing.assert_array_equal(pts, again)                     # fixed seed
    cell = probe.cavity_near(*s["position"])
    m = MARGIN - 1e-9
    v_mid, a_mid = 0.5 * (cell.floor + cell.ceiling), 0.5 * (cell.a0 + cell.a1)
    for x, y, z in pts:                       # depth behind the LOCAL cavity face (the girder tapers along z)
        if member == "deck":
            inner, outer = probe.face_above(x, z, v_mid)
            depth = y - inner
        elif member == "bottom_slab":
            inner, outer = probe.face_below(x, z, v_mid)
            depth = inner - y
        else:
            side = +1 if member == "web_south" else -1
            inner, outer = probe.web(y, z, a_mid, side)
            depth = side * (x - inner)
        assert m <= depth <= abs(outer - inner) - m
    if member in ("deck", "bottom_slab"):     # in-plane coordinate inside the cell
        assert (pts[:, 0] >= cell.a0 + m).all() and (pts[:, 0] <= cell.a1 - m).all()
    else:
        assert (pts[:, 1] >= cell.floor + m).all() and (pts[:, 1] <= cell.ceiling - m).all()
    assert (pts[:, 2] >= m).all() and (pts[:, 2] <= mesh.points[:, 2].max() - m).all()
    assert pts.std(0).min() > 1e-4                                  # all three coordinates vary
    # the nominal point is the zero-deviation sample at the text depth
    nominal, _ = _place(probe, cell, s["position"][2], mesh.points[:, 2].max(), member,
                        s["mapping"]["depth_behind_cavity_face_m"], 0.0, 0.0,
                        s["position"][0], s["position"][1], MARGIN)
    np.testing.assert_allclose(nominal, s["position"], atol=1e-9)


def test_explicit_priors_and_isotropic_samples(probe, mesh):
    pos_cfg = load_model_config()["uncertainty"]["position"]
    base = _ifc_like(probe, "deck", *IFC_LIKE["deck"])
    explicit = {"name": "d", "quantity": "temperature", "position": base["position"], "member": "deck",
                "depth_range": [0.05, 0.10], "position_std": 0.0}
    pts = position_samples(explicit, pos_cfg, MARGIN, mesh, probe)
    depth = pts[:, 1] - probe.cavity_near(*base["position"]).ceiling
    assert depth.min() >= 0.05 - 1e-9 and depth.max() <= 0.10 + 1e-9 and depth.std() > 0.01
    np.testing.assert_allclose(pts[:, [0, 2]], np.tile(np.array(base["position"])[[0, 2]], (len(pts), 1)), atol=1e-12)
    iso = {"name": "i", "quantity": "temperature", "position": [2.85, -0.12, 10.0], "position_std": 0.03}
    pts = position_samples(iso, pos_cfg, MARGIN, mesh, probe)
    assert pts.shape == (pos_cfg["n_samples"], 3) and mesh.contains(pts).all()
    assert abs(pts.mean(0) - iso["position"]).max() < 0.03
    # mechanical sensors never get a position term
    mech = {"name": "t", "quantity": "inclination", "position": [-2.25, -0.2, 38.3], "position_std": 0.05}
    assert position_samples(mech, pos_cfg, MARGIN, mesh, probe) is None
