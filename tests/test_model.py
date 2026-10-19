"""Unit checks of the quadrature, the P1/P2 elements and the finite-element physics.

All model tests run on a small mesh of the hinge-end geometry (3k nodes).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from nibelungen_virtual_sensors.fem.elements import Space, tet_rule, tri_rule
from nibelungen_virtual_sensors.fem.mechanics import ThermoElasticModel
from nibelungen_virtual_sensors.fem.mesh import Mesh
from nibelungen_virtual_sensors.fem.probes import build_probes
from nibelungen_virtual_sensors.fem.thermal import ThermalModel
from nibelungen_virtual_sensors.transform import load_model_config
from nibelungen_virtual_sensors.uq import lognormal_gauss_rule, weighted_moments

COARSE = Path(__file__).parent / "data" / "coarse_hinge_mesh.npz"


@pytest.fixture(scope="module")
def cfg():
    return load_model_config()


@pytest.fixture(scope="module")
def mesh():
    return Mesh.load(COARSE)


def test_quadrature_matches_pipeline_pce():
    # results/simulations/pce_no_hr.pkl of the research pipeline (chaospy)
    nodes, weights = lognormal_gauss_rule(1.0502, 0.7763, 3)
    np.testing.assert_allclose(nodes, [0.68525864, 2.51141708, 9.20413900], rtol=1e-7)
    np.testing.assert_allclose(weights, [8.04618962e-01, 1.94164099e-01, 1.21693852e-03], rtol=1e-7)
    # 3-point Gauss rule is exact for the LogNormal mean and variance
    mean, var = weighted_moments(nodes[:, None], weights)
    assert mean[0] == pytest.approx(1.0502, rel=1e-10)
    assert np.sqrt(var[0]) == pytest.approx(0.7763, rel=1e-10)


def test_default_degree_is_p2(cfg):
    assert cfg["thermal"]["degree"] == cfg["mechanical"]["degree"] == 2


def test_mesh_frame(mesh):
    lo, hi = mesh.points.min(0), mesh.points.max(0)
    np.testing.assert_allclose(lo, [-7.32, -5.87, 0.0], atol=1e-6)
    np.testing.assert_allclose(hi, [7.32, 0.0, 40.1], atol=1e-6)
    assert mesh.volumes.min() > 0


def test_locate_rejects_points_outside_concrete(mesh):
    with pytest.raises(ValueError, match="outside the bridge mesh"):
        mesh.locate([2.85, -2.6, 0.5])        # inside the box-girder cavity
    cells, bary = mesh.locate([2.85, -0.1, 10.0])  # in the deck
    assert bary.sum() == pytest.approx(1.0)


# ── elements ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("degree", [2, 3, 4, 6])
def test_quadrature_is_exact(degree):
    lam, w = tet_rule(degree)
    x, y, z = lam[:, 1], lam[:, 2], lam[:, 3]
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(degree + 1 - a - b):
                exact = math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3)
                assert (w * x**a * y**b * z**c).sum() / 6 == pytest.approx(exact, rel=1e-12)
    lt, wt = tri_rule(degree)
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            exact = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
            assert (wt * lt[:, 1]**a * lt[:, 2]**b).sum() / 2 == pytest.approx(exact, rel=1e-12)


def test_p2_space_reproduces_quadratics_and_gradients():
    P = np.array([[0, 0, 0], [1.2, 0.1, 0], [0.2, 0.9, 0.1], [0.1, 0.2, 1.1]])
    one = Mesh(points=P, tets=np.array([[0, 1, 2, 3]]), facets=np.array([[0, 1, 2]]), facet_tags=np.array([9]))
    sp_ = Space(one, 2)
    assert sp_.n == 10
    f = lambda X: 1 + 2*X[..., 0] - X[..., 1] + 3*X[..., 2] + X[..., 0]*X[..., 1] - 2*X[..., 2]**2 + 0.5*X[..., 1]*X[..., 2]
    grad = lambda X: np.stack([2 + X[..., 1], -1 + X[..., 0] + 0.5*X[..., 2], 3 - 4*X[..., 2] + 0.5*X[..., 1]], -1)
    lam = np.random.default_rng(0).dirichlet(np.ones(4), size=6)
    X = lam @ P
    np.testing.assert_allclose(sp_.basis(lam).sum(1), 1.0, atol=1e-13)
    np.testing.assert_allclose(sp_.basis(lam) @ f(sp_.coords), f(X), atol=1e-12)
    G = np.stack([sp_.grad_basis(l, one.grads[0]) for l in lam])
    np.testing.assert_allclose(np.einsum("a,qak->qk", f(sp_.coords), G), grad(X), atol=1e-12)


def test_p2_dofs_are_vertices_plus_edges(mesh):
    sp_ = Space(mesh, 2)
    assert sp_.n == mesh.n_nodes + len(sp_.edge_keys)
    assert sp_.cell_dofs.shape == (len(mesh.tets), 10)
    assert Space(mesh, 1).n == mesh.n_nodes


# ── physics ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("degree", [1, 2])
def test_thermal_relaxes_to_ambient(mesh, cfg, degree):
    th = ThermalModel(mesh, {**cfg["thermal"], "degree": degree})
    forcing = np.tile([5.0, 5.0, 0.0, 3.0], (400, 1))  # 5 °C everywhere, no sun
    PT, _, off, _ = build_probes(mesh, ["temperature"] * 2, np.array([[2.85, -0.1, 5.0], [3.65, -2.0, 20.0]]), degree)
    last = {}
    th.run(forcing, 3 * 86400.0, 1.0, 0, 399, lambda k, T: last.update(T=T))
    np.testing.assert_allclose(PT @ last["T"] + off, 5.0, atol=0.05)


@pytest.mark.parametrize("degree", [1, 2])
def test_uniform_heating_is_free_expansion(mesh, cfg, degree):
    sp_ = Space(mesh, degree)
    me = ThermoElasticModel(mesh, {**cfg["mechanical"], "degree": degree}, sp_)
    dT = 10.0
    u = me.solve(np.full(sp_.n, cfg["mechanical"]["reference_temperature"] + dT), "lu")
    _, Pu, _, _ = build_probes(mesh, ["displacement_z", "inclination", "strain_zz"],
                               np.array([[2.85, -0.1, 39.0], [2.85, -0.1, 30.0], [2.85, -0.1, 25.0]]), sp_)
    uz_mm, incl, ezz = Pu @ u
    alpha = cfg["mechanical"]["thermal_expansion"]
    assert uz_mm == pytest.approx(alpha * dT * 39.0 * 1e3, rel=0.03)    # ≈ α ΔT z
    assert ezz == pytest.approx(alpha * dT * 1e6, rel=0.03)             # ≈ α ΔT, stress-free
    assert abs(incl) < 0.01 * alpha * dT * 1e3 * 40                      # no bending


@pytest.mark.parametrize("degree", [1, 2])
def test_adjoint_sensor_weights_equal_full_solve(mesh, cfg, degree):
    """y(t) = W (T - T_ref) with one adjoint solve per sensor reproduces the full solve."""
    sp_ = Space(mesh, degree)
    me = ThermoElasticModel(mesh, {**cfg["mechanical"], "degree": degree}, sp_)
    q = ["inclination", "displacement_y", "strain_zz", "strain_yz"]
    pos = np.array([[-2.25, -0.1, 30.0], [0.0, -0.1, 39.5], [2.85, -0.1, 10.0], [3.65, -1.5, 10.0]])
    _, Pu, _, _ = build_probes(mesh, q, pos, sp_)
    X = sp_.coords
    T = 293.15 + 5 * np.sin(0.4 * X[:, 0] + 0.15 * X[:, 2]) + 3 * np.cos(1.1 * X[:, 1])
    y_direct = Pu @ me.solve(T, "lu")
    for method in ("lu", "ilu"):
        W = me.sensor_weights(Pu, method)
        np.testing.assert_allclose(W @ (T - me.T_ref), y_direct, rtol=1e-6, atol=1e-8 * np.abs(y_direct).max(),
                                   err_msg=method)
