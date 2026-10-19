"""Lagrange P1 / P2 elements on linear tetrahedra, without dolfinx.

``Space`` numbers the degrees of freedom of the continuous Lagrange space of
degree 1 or 2 on a :class:`~.mesh.Mesh`:

* P1: one dof per vertex (the vertex numbering of the mesh);
* P2: the vertices first, then one dof per mesh edge, at the edge midpoint.

The numbering differs from DOLFINx's, which is irrelevant: the finite-element
*space* is the same, so assembled quantities agree to solver tolerance.

Element integrals use exact quadrature: collapsed (Duffy) Gauss–Legendre
rules whose order is chosen from the polynomial degree of the integrand.
Cells are straight-sided, so Jacobians are constant per cell.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .mesh import Mesh

TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (0, 2), (1, 2))
_TET_EDGES = np.array(TET_EDGES)
_TRI_EDGES = np.array(TRI_EDGES)


@lru_cache(maxsize=None)
def tet_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric points (nq, 4) and weights (nq,, sum 1) exact for polynomials of ``degree``.

    Duffy map x = u, y = v(1-u), z = w(1-u)(1-v) of the unit cube; the
    transformed integrand has degree ``degree + 2`` in u, so Gauss–Legendre
    with n points (exact to 2n-1) needs n >= (degree + 3) / 2.
    """
    n = max(1, math.ceil((degree + 3) / 2))
    g, wg = np.polynomial.legendre.leggauss(n)
    g, wg = 0.5 * (g + 1.0), 0.5 * wg
    lam, w = [], []
    for u, wu in zip(g, wg):
        for v, wv in zip(g, wg):
            for t, wt in zip(g, wg):
                x, y, z = u, v * (1 - u), t * (1 - u) * (1 - v)
                lam.append((1 - x - y - z, x, y, z))
                w.append(6.0 * wu * wv * wt * (1 - u) ** 2 * (1 - v))
    return np.array(lam), np.array(w)


@lru_cache(maxsize=None)
def tri_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric points (nq, 3) and weights (sum 1) exact for polynomials of ``degree``."""
    n = max(1, math.ceil((degree + 2) / 2))
    g, wg = np.polynomial.legendre.leggauss(n)
    g, wg = 0.5 * (g + 1.0), 0.5 * wg
    lam, w = [], []
    for u, wu in zip(g, wg):
        for v, wv in zip(g, wg):
            x, y = u, v * (1 - u)
            lam.append((1 - x - y, x, y))
            w.append(2.0 * wu * wv * (1 - u))
    return np.array(lam), np.array(w)


class Space:
    """Continuous Lagrange space of degree 1 or 2 on the tetrahedral ``mesh``."""

    def __init__(self, mesh: Mesh, degree: int) -> None:
        if degree not in (1, 2):
            raise ValueError(f"element degree must be 1 or 2, got {degree}")
        self.mesh, self.degree = mesh, degree
        nv = mesh.n_nodes
        if degree == 1:
            self.n, self.nloc = nv, 4
            self.cell_dofs = mesh.tets
            self.coords = mesh.points
            self.edge_keys = None
        else:
            edges = mesh.tets[:, _TET_EDGES]                               # (M, 6, 2)
            keys = edges.min(-1).astype(np.int64) * nv + edges.max(-1)
            uniq, inv = np.unique(keys.ravel(), return_inverse=True)
            self.edge_keys = uniq
            self.cell_dofs = np.hstack([mesh.tets, nv + inv.reshape(-1, 6)])
            ends = np.column_stack([uniq // nv, uniq % nv])
            self.coords = np.vstack([mesh.points, 0.5 * (mesh.points[ends[:, 0]] + mesh.points[ends[:, 1]])])
            self.n, self.nloc = nv + len(uniq), 10

    @property
    def nfloc(self) -> int:
        """Dofs per boundary triangle."""
        return 3 if self.degree == 1 else 6

    # -- boundary --------------------------------------------------------------
    def facet_dofs(self, facets: np.ndarray) -> np.ndarray:
        """Dofs (K, nfloc) of boundary triangles given by vertex triples (K, 3)."""
        if self.degree == 1:
            return facets
        nv = self.mesh.n_nodes
        fe = facets[:, _TRI_EDGES]
        keys = fe.min(-1).astype(np.int64) * nv + fe.max(-1)
        pos = np.searchsorted(self.edge_keys, keys)
        if not np.array_equal(self.edge_keys[np.minimum(pos, len(self.edge_keys) - 1)], keys):
            raise ValueError("a boundary triangle has an edge that belongs to no cell")
        return np.hstack([facets, nv + pos])

    def facet_basis(self, mu: np.ndarray) -> np.ndarray:
        """Triangle basis functions at barycentric points mu (..., 3) -> (..., nfloc)."""
        if self.degree == 1:
            return mu
        edge = [4.0 * mu[..., i] * mu[..., j] for i, j in TRI_EDGES]
        return np.concatenate([mu * (2.0 * mu - 1.0), np.stack(edge, axis=-1)], axis=-1)

    # -- cells -----------------------------------------------------------------
    def basis(self, lam: np.ndarray) -> np.ndarray:
        """Basis functions at barycentric points lam (..., 4) -> (..., nloc)."""
        if self.degree == 1:
            return lam
        edge = [4.0 * lam[..., i] * lam[..., j] for i, j in TET_EDGES]
        return np.concatenate([lam * (2.0 * lam - 1.0), np.stack(edge, axis=-1)], axis=-1)

    def grad_basis(self, lam: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Gradients of the basis at barycentric ``lam`` (..., 4) for cells with
        barycentric-coordinate gradients ``g`` (..., 4, 3) -> (..., nloc, 3)."""
        if self.degree == 1:
            return g
        vert = (4.0 * lam[..., :, None] - 1.0) * g
        edge = [4.0 * (lam[..., i, None] * g[..., j, :] + lam[..., j, None] * g[..., i, :])
                for i, j in TET_EDGES]
        return np.concatenate([vert, np.stack(edge, axis=-2)], axis=-2)

    def point_operator_rows(self, cells: np.ndarray, lam: np.ndarray):
        """Values (S, nloc) and gradients (S, nloc, 3) of the basis at points inside ``cells``."""
        return self.basis(lam), self.grad_basis(lam, self.mesh.grads[cells])
