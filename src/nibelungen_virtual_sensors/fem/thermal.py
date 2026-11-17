"""Transient heat conduction in the box girder (P1 or P2, implicit Euler).

Re-implements, without dolfinx, the thermal model of the research pipeline
(``thermal_model/transient_thermal_material_problem.py`` with the
``simulation.thermal_3d`` settings of ``config/pipeline_config.yaml``):

    ρ c_p (T - T_old)/dt - ∇·(k ∇T) = 0                    in Ω
    k ∂T/∂n = c_top h_wind(v) (T_air - T) + c_r a I        on top (tag 9)
    k ∂T/∂n = c_ext h_wind(v) (T_air - T)                  on outer faces (tag 8)
    k ∂T/∂n = h_in (T_in - T)                              on the cavity (tag 42)
    k ∂T/∂n = 0                                            on the end faces

with the flat-plate correlation h_wind = 0.037 Re^0.8 Pr^(1/3) k_air / L,
Re = v L / ν_air.  ``p["degree"]`` selects P1 or P2.  As in the pipeline, the
ambient temperature is a nodal function: every dof on a cavity facet carries
the cavity air temperature, every other dof the outside air temperature.
All boundary terms are polynomial, so exact quadrature reproduces dolfinx.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elements import Space, tet_rule, tri_rule
from .mesh import Mesh

KELVIN = 273.15


def _tri_area(p: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def boundary_mass(space: Space, tags) -> sp.csr_matrix:
    mesh = space.mesh
    f = mesh.facets_with_tags(tags)
    if len(f) == 0:
        return sp.csr_matrix((space.n, space.n))
    fd = space.facet_dofs(f)
    lam, w = tri_rule(2 * space.degree)
    phi = space.facet_basis(lam)                                   # (nq, nfloc)
    ref = np.einsum("q,qi,qj->ij", w, phi, phi)
    vals = _tri_area(mesh.points[f])[:, None, None] * ref
    rows = np.repeat(fd, space.nfloc, axis=1).ravel()
    cols = np.tile(fd, (1, space.nfloc)).ravel()
    return sp.csr_matrix((vals.ravel(), (rows, cols)), shape=(space.n,) * 2)


def boundary_load(space: Space, tags) -> np.ndarray:
    mesh = space.mesh
    f = mesh.facets_with_tags(tags)
    b = np.zeros(space.n)
    if len(f):
        fd = space.facet_dofs(f)
        lam, w = tri_rule(space.degree)
        ref = np.einsum("q,qi->i", w, space.facet_basis(lam))
        np.add.at(b, fd.ravel(), (_tri_area(mesh.points[f])[:, None] * ref).ravel())
    return b


def _cg(A, b, x0, M):
    """Preconditioned CG; scipy renamed ``tol`` to ``rtol`` in 1.12."""
    try:
        return spla.cg(A, b, x0=x0, rtol=1e-12, atol=0.0, M=M, maxiter=2000)
    except TypeError:
        return spla.cg(A, b, x0=x0, tol=1e-12, atol=0.0, M=M, maxiter=2000)


def cell_matrix(space: Space, local: np.ndarray) -> sp.csr_matrix:
    """Assemble per-cell matrices (M, nloc, nloc) into a CSR matrix."""
    t = space.cell_dofs
    rows = np.repeat(t, space.nloc, axis=1).ravel()
    cols = np.tile(t, (1, space.nloc)).ravel()
    return sp.csr_matrix((local.ravel(), (rows, cols)), shape=(space.n,) * 2)


class ThermalModel:
    """Assembled thermal operators; :meth:`run` time-steps one parameter set."""

    def __init__(self, mesh: Mesh, p: dict, space: Space | None = None) -> None:
        self.mesh = mesh
        self.p = p
        self.space = space or Space(mesh, int(p.get("degree", 1)))
        sp_ = self.space
        v = mesh.volumes

        lam, w = tet_rule(2 * sp_.degree)
        phi = sp_.basis(lam)
        mass_ref = np.einsum("q,qi,qj->ij", w, phi, phi)
        M = cell_matrix(sp_, v[:, None, None] * mass_ref)

        lam, w = tet_rule(2 * (sp_.degree - 1))
        Kloc = np.zeros((len(v), sp_.nloc, sp_.nloc))
        for lq, wq in zip(lam, w):
            G = sp_.grad_basis(lq, mesh.grads)
            Kloc += wq * np.einsum("mik,mjk->mij", G, G)
        K = cell_matrix(sp_, v[:, None, None] * Kloc)

        tags = p["boundary_tags"]
        # per-boundary convection factors, as in the calibration: deck top and all other external faces
        self.Mb_top = boundary_mass(sp_, [tags["top"]])
        self.Mb_ext = boundary_mass(sp_, [tags["bottom"], tags["external"]])
        self.Mb_hole = boundary_mass(sp_, [tags["hole"]])
        self.b_top = boundary_load(sp_, [tags["top"]])
        hole_f = mesh.facets_with_tags([tags["hole"]])
        self.hole_nodes = np.unique(sp_.facet_dofs(hole_f))

        self.M = M
        self.K = float(p["conductivity"]) * K
        self.A_hole = float(p["cavity_convection_coefficient"]) * self.Mb_hole

    def h_wind(self, wind_speed: float) -> float:
        p = self.p
        L = p["characteristic_length"]
        re = wind_speed * L / p["air_viscosity"]
        nu = 0.037 * re ** 0.8 * p["air_prandtl_number"] ** (1.0 / 3.0)
        return nu * p["air_thermal_conductivity"] / L

    def run(
        self,
        forcing: np.ndarray,
        dt: float,
        wind_constant: float | tuple[float, float],
        n_initial_steps: int,
        collect_from: int,
        on_step,
    ) -> None:
        """Integrate over the forcing rows.

        ``wind_constant`` is ``(c_top, c_ext)``, or one value for both boundaries.
        ``forcing`` columns: air temperature [°C], cavity air temperature
        [°C], shortwave irradiation [W/m²], wind speed [m/s].  The first row
        is repeated ``n_initial_steps`` times to reach a periodic-ish initial
        state (pipeline: ``initial_condition_steps``).  ``on_step(k, T)`` is
        called with the nodal temperature [K] for every row ``k >=
        collect_from``.
        """
        p = self.p
        mc = float(p["density"]) * float(p["heat_capacity"]) / dt
        A_static = (mc * self.M + self.K + self.A_hole).tocsr()
        Mmc = (mc * self.M).tocsr()
        absorbed = float(p["shortwave_radiation_constant"]) * float(p["absorptivity"])
        c_top, c_ext = (wind_constant, wind_constant) if np.ndim(wind_constant) == 0 else wind_constant
        Mb_c = (float(c_top) * self.Mb_top + float(c_ext) * self.Mb_ext).tocsr()

        T = np.full(self.space.n, float(p["initial_temperature"]))
        T_amb = np.empty_like(T)
        rows = [0] * n_initial_steps + list(range(len(forcing)))
        for i, k in enumerate(rows):
            t_air, t_in, irr, wind = forcing[k]
            T_amb[:] = t_air + KELVIN
            T_amb[self.hole_nodes] = t_in + KELVIN
            hw = self.h_wind(wind)
            A = A_static + hw * Mb_c
            rhs = Mmc @ T + hw * (Mb_c @ T_amb) + self.A_hole @ T_amb + absorbed * irr * self.b_top
            # SPD system; Jacobi-CG warm-started from the previous step is
            # ~10x faster than a direct solve and agrees to ~1e-9 K.
            jac = 1.0 / A.diagonal()
            pre = spla.LinearOperator(A.shape, matvec=lambda v: jac * v)
            T_new, info = _cg(A, rhs, T, pre)
            T = T_new if info == 0 else spla.spsolve(A.tocsc(), rhs)
            if i >= n_initial_steps and k >= collect_from:
                on_step(k, T)
