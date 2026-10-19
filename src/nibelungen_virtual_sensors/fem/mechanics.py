"""Linear thermo-elasticity of the cantilever (P1 or P2, small strains).

Re-implements ``LinearThermoElasticity`` of the research pipeline
(``thermal_model/thermomechanical_problem.py``) with the cantilever BCs of
``src/simulation/bridge_bcs.py`` (``extent: face``):

    σ = λ tr(ε) I + 2 μ ε - κ (T - T_ref) I,   κ = α_T (3λ + 2μ)
    ∇·σ = 0,   u = 0 on the clamped face z = z_min,   traction-free elsewhere

Thermal load only (no self-weight), so the response is linear in T:
``u_free = K⁻¹ F (T - T_ref)``.  The temperature has the same degree and dof
numbering as the displacement (``Space``), as in the pipeline.  Vector dofs
are interleaved: ``3*node + c``.

Virtual sensors are linear functionals ``y = P u`` of the displacement, so
``y(t) = W (T(t) - T_ref)`` with ``W = Pᵀ-adjoint``: :meth:`sensor_weights`
solves one adjoint problem per mechanical sensor, **once**, and every time
step (and quadrature node) then costs a single matrix-vector product — no
factorisation of K and no solve per step.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elements import Space, tet_rule
from .mesh import Mesh

log = logging.getLogger(__name__)
_CHUNK = 2000  # cells per assembly block (bounds the transient memory)


def _strain_operator(G: np.ndarray) -> np.ndarray:
    """B (M, 6, 3*nloc) from basis gradients G (M, nloc, 3); Voigt order xx, yy, zz, 2xy, 2yz, 2xz."""
    M, nloc, _ = G.shape
    B = np.zeros((M, 6, 3 * nloc))
    gx, gy, gz = G[:, :, 0], G[:, :, 1], G[:, :, 2]
    B[:, 0, 0::3] = gx
    B[:, 1, 1::3] = gy
    B[:, 2, 2::3] = gz
    B[:, 3, 0::3] = gy
    B[:, 3, 1::3] = gx
    B[:, 4, 1::3] = gz
    B[:, 4, 2::3] = gy
    B[:, 5, 0::3] = gz
    B[:, 5, 2::3] = gx
    return B


class ThermoElasticModel:
    def __init__(self, mesh: Mesh, p: dict, space: Space | None = None) -> None:
        self.mesh = mesh
        self.space = sp_ = space or Space(mesh, int(p.get("degree", 1)))
        self.p = p
        E, nu = float(p["E"]), float(p["nu"])
        lam_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        kappa = float(p["thermal_expansion"]) * (3 * lam_ + 2 * mu)
        self.T_ref = float(p["reference_temperature"])

        D = np.zeros((6, 6))
        D[:3, :3] = lam_
        D[np.arange(3), np.arange(3)] += 2 * mu
        D[np.arange(3, 6), np.arange(3, 6)] = mu

        nl, n = sp_.nloc, 3 * sp_.n
        rule_k = tet_rule(2 * (sp_.degree - 1))      # grad·grad
        rule_f = tet_rule(2 * sp_.degree - 1)        # φ · grad
        K = sp.csr_matrix((n, n))
        F = sp.csr_matrix((n, sp_.n))
        for lo in range(0, len(mesh.volumes), _CHUNK):
            sl = slice(lo, lo + _CHUNK)
            vol, g = mesh.volumes[sl], mesh.grads[sl]
            dofs = (3 * sp_.cell_dofs[sl][:, :, None] + np.arange(3)).reshape(-1, 3 * nl)
            Ke = np.zeros((len(vol), 3 * nl, 3 * nl))
            for lq, wq in zip(*rule_k):
                B = _strain_operator(sp_.grad_basis(lq, g))
                Ke += wq * np.matmul(B.transpose(0, 2, 1), np.matmul(D, B))
            Ke *= vol[:, None, None]
            K = K + sp.csr_matrix(
                (Ke.ravel(), (np.repeat(dofs, 3 * nl, axis=1).ravel(), np.tile(dofs, (1, 3 * nl)).ravel())),
                shape=(n, n))
            del Ke

            # Thermal load  f = F (T - T_ref),  F_{(a,c), b} = κ ∫ φ_b ∂φ_a/∂x_c
            Fe = np.zeros((len(vol), 3 * nl, nl))
            for lq, wq in zip(*rule_f):
                G = sp_.grad_basis(lq, g)                       # (M, nl, 3)
                phi = sp_.basis(lq)                             # (nl,)
                Fe += wq * (G.reshape(len(vol), 3 * nl, 1) * phi[None, None, :])
            Fe *= (kappa * vol)[:, None, None]
            F = F + sp.csr_matrix(
                (Fe.ravel(), (np.repeat(dofs, nl, axis=1).ravel(), np.tile(sp_.cell_dofs[sl], (1, 3 * nl)).ravel())),
                shape=(n, sp_.n))
            del Fe

        z = sp_.coords[:, 2]
        bc = p["clamp"]
        z_fix = z.min() if bc.get("fixed_end", "z_min") == "z_min" else z.max()
        atol = (z.max() - z.min()) * float(bc.get("face_atol_frac", 1e-3))
        clamped = np.flatnonzero(np.isclose(z, z_fix, atol=atol))
        if clamped.size == 0:
            raise RuntimeError("no nodes on the clamped face")
        fixed = (3 * clamped[:, None] + np.arange(3)).ravel()
        self.free = np.setdiff1d(np.arange(n), fixed)
        self.n_dofs = n
        self.K = K[self.free][:, self.free].tocsr()
        self.F = F[self.free].tocsr()
        self._lu = None

    # -- linear solves -----------------------------------------------------------
    def _factor(self):
        if self._lu is None:
            self._lu = spla.splu(self.K.tocsc(), permc_spec="MMD_AT_PLUS_A")
        return self._lu

    def solve_spd(self, rhs: np.ndarray, method: str = "ilu", rtol: float = 1e-10) -> np.ndarray:
        """Solve K x = rhs for the free dofs.  ``method``: ``"ilu"`` (GMRES with an
        incomplete-LU preconditioner, small memory) or ``"lu"`` (sparse direct)."""
        if method == "lu":
            return self._factor().solve(rhs)
        if method != "ilu":
            raise ValueError(f"unknown mechanics solver {method!r}")
        if getattr(self, "_ilu", None) is None:
            ilu = spla.spilu(self.K.tocsc(), drop_tol=1e-4, fill_factor=10.0, permc_spec="MMD_AT_PLUS_A")
            self._ilu = spla.LinearOperator(self.K.shape, matvec=ilu.solve)
        its = [0]
        try:
            x, info = spla.gmres(self.K, rhs, M=self._ilu, rtol=rtol, atol=0.0, restart=60, maxiter=40,
                                 callback=lambda _: its.__setitem__(0, its[0] + 1), callback_type="pr_norm")
        except TypeError:  # scipy < 1.12 uses `tol`
            x, info = spla.gmres(self.K, rhs, M=self._ilu, tol=rtol, atol=0.0, restart=60, maxiter=40,
                                 callback=lambda _: its.__setitem__(0, its[0] + 1), callback_type="pr_norm")
        if info != 0:
            raise RuntimeError(f"mechanics ILU-GMRES did not converge (info={info})")
        log.info("adjoint solve: %d GMRES iterations", its[0])
        return x

    # -- full displacement (used by tests and the direct path) --------------------
    def solve(self, T: np.ndarray, method: str = "lu") -> np.ndarray:
        """Nodal displacement (3 * n_dofs_T,) for nodal temperature T [K]."""
        u = np.zeros(self.n_dofs)
        u[self.free] = self.solve_spd(self.F @ (T - self.T_ref), method)
        return u

    # -- adjoint: readings as a linear map of the temperature ---------------------
    def sensor_weights(self, Pu: sp.csr_matrix, method: str = "ilu") -> np.ndarray:
        """W (n_sensors, n_T) with  y(t) = W @ (T(t) - T_ref)  for the rows of ``Pu``.

        K is symmetric, so  y = Pu K⁻¹ F ΔT = (K⁻¹ Puᵀ)ᵀ F ΔT : one solve per row.
        """
        Pf = Pu[:, self.free].tocsr()
        W = np.empty((Pf.shape[0], self.F.shape[1]))
        for r in range(Pf.shape[0]):
            z = self.solve_spd(Pf[r].toarray().ravel(), method)
            W[r] = self.F.T @ z
        return W
