"""Linear tetrahedral mesh of the bridge cantilever and point location.

Coordinate frame of the simulation model (metres):

* ``x`` transverse; the instrumented (southern) box girder lies at ``x > 0``
* ``y`` vertical, up; ``y = 0`` is the deck top surface
* ``z`` along the span, from the clamped section at pier axis E (``z = 0``)
  towards the cantilever tip (``z ≈ 47.6``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    points: np.ndarray       # (n_nodes, 3) float64
    tets: np.ndarray         # (n_cells, 4) int
    facets: np.ndarray       # (n_facets, 3) int, tagged boundary triangles
    facet_tags: np.ndarray   # (n_facets,) int, gmsh physical group per facet
    volumes: np.ndarray = field(init=False)
    grads: np.ndarray = field(init=False)  # (n_cells, 4, 3) P1 shape-function gradients

    def __post_init__(self) -> None:
        p = self.points[self.tets]                     # (M, 4, 3)
        J = (p[:, 1:, :] - p[:, :1, :]).transpose(0, 2, 1)   # columns = edges
        det = np.linalg.det(J)
        self.volumes = np.abs(det) / 6.0
        Jinv = np.linalg.inv(J)                        # rows = grads of λ1..λ3
        g = np.empty((len(self.tets), 4, 3))
        g[:, 1:, :] = Jinv
        g[:, 0, :] = -Jinv.sum(axis=1)
        self.grads = g
        self._Jinv = Jinv

    @classmethod
    def load(cls, path: Path) -> "Mesh":
        with np.load(path) as d:
            return cls(
                points=d["points"].astype(np.float64),
                tets=d["tets"].astype(np.int64),
                facets=d["facets"].astype(np.int64),
                facet_tags=d["facet_tags"].astype(np.int64),
            )

    @property
    def n_nodes(self) -> int:
        return self.points.shape[0]

    def facets_with_tags(self, tags) -> np.ndarray:
        return self.facets[np.isin(self.facet_tags, list(tags))]

    def contains(self, xyz, tol: float = 1e-9) -> np.ndarray:
        """Boolean mask: is each point inside the mesh (same test as :meth:`locate`)."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        p0 = self.points[self.tets[:, 0]]
        ok = np.empty(len(xyz), dtype=bool)
        for i, x in enumerate(xyz):
            lam = np.einsum("mij,mj->mi", self._Jinv, x - p0)
            l0 = 1.0 - lam.sum(axis=1)
            ok[i] = bool(np.any((l0 >= -tol) & np.all(lam >= -tol, axis=1)))
        return ok

    def locate(self, xyz, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        """Containing cell and barycentric coordinates of each point.

        Raises ``ValueError`` for points outside the mesh: a virtual sensor
        outside the concrete is a coordinate-mapping error, never something
        to silently snap.
        """
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        cells = np.empty(len(xyz), dtype=np.int64)
        bary = np.empty((len(xyz), 4))
        p0 = self.points[self.tets[:, 0]]
        for i, x in enumerate(xyz):
            lam = np.einsum("mij,mj->mi", self._Jinv, x - p0)       # λ1..λ3
            l0 = 1.0 - lam.sum(axis=1)
            inside = (l0 >= -tol) & np.all(lam >= -tol, axis=1)
            hits = np.flatnonzero(inside)
            if hits.size == 0:
                d = np.linalg.norm(self.points - x, axis=1).min()
                raise ValueError(
                    f"point {x.tolist()} lies outside the bridge mesh "
                    f"(nearest mesh node {d:.3f} m away); check the sensor "
                    "position / coordinate mapping"
                )
            c = hits[0]
            cells[i] = c
            bary[i] = (l0[c], *lam[c])
        return cells, bary
