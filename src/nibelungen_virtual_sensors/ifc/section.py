"""Cross-section probing of a closed triangle surface by ray casting.

Used identically on the IFC girder solids and on the boundary of the
simulation mesh, in a common *section frame* ``(a, v, z)``:

* ``a`` transverse, oriented like model ``x`` (positive towards the south)
* ``v`` vertical, up
* ``z`` along the span (station)

so that both geometries are measured with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Small, irrational offsets keep rays off mesh edges and vertices, where a
# ray would otherwise hit two triangles at the same parameter.
_JITTER = np.array([1.37e-6, 2.71e-6, 3.14e-6])


def ray_hits(tris: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Sorted ray parameters of all triangle intersections (Möller–Trumbore)."""
    o = np.asarray(origin, float) + _JITTER
    d = np.asarray(direction, float)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    w = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    m = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
    return np.sort(t[m])


def solid_intervals(tris: np.ndarray, origin, direction, tol: float = 1e-6) -> list[tuple[float, float]]:
    """Intervals of the (closed) solid along an infinite line ``origin + t·direction``."""
    o = np.asarray(origin, float) - 1e4 * np.asarray(direction, float)
    t = ray_hits(tris, o, direction) - 1e4
    if len(t) == 0:
        return []
    t = t[np.concatenate([[True], np.diff(t) > tol])]  # merge coincident hits
    if len(t) % 2:
        raise ValueError("ray crossed the surface an odd number of times; is the solid closed?")
    return [(float(t[i]), float(t[i + 1])) for i in range(0, len(t), 2)]


@dataclass
class Cavity:
    """Box-girder cell around a point, in section coordinates."""

    a0: float        # face of the northern web (smaller a)
    a1: float        # face of the southern web
    floor: float     # top of the bottom slab
    ceiling: float   # underside of the deck
    top: float       # deck top surface above the cell centre
    bottom: float    # soffit below the cell centre


class SectionProbe:
    """Ray queries on a triangle surface given in section coordinates (a, v, z)."""

    def __init__(self, tris: np.ndarray) -> None:
        self.tris = np.asarray(tris, float)

    def vertical(self, a: float, z: float):
        return solid_intervals(self.tris, (a, 0.0, z), (0.0, 1.0, 0.0))

    def transverse(self, v: float, z: float):
        return solid_intervals(self.tris, (0.0, v, z), (1.0, 0.0, 0.0))

    def cavity_near(self, a: float, v: float, z: float) -> Cavity:
        """Closed cell of the box girder that contains or is nearest to (a, v)."""
        col = self.vertical(a, z)
        if not col:
            raise ValueError(f"no girder at a={a:.3f}, z={z:.3f}")
        v_mid = 0.5 * (col[0][0] + col[-1][1])
        row = self.transverse(v_mid, z)
        cells = []
        for (_, g0), (g1, _) in zip(row[:-1], row[1:]):
            c = self.vertical(0.5 * (g0 + g1), z)
            below = [iv for iv in c if iv[1] <= v_mid]
            above = [iv for iv in c if iv[0] >= v_mid]
            if below and above:  # closed cell (deck above, bottom slab below)
                cells.append(Cavity(g0, g1, below[-1][1], above[0][0], above[-1][1], below[0][0]))
        if not cells:
            raise ValueError(f"no closed box-girder cell found at z={z:.3f}")
        return min(cells, key=lambda c: max(c.a0 - a, a - c.a1, 0.0))

    def face_above(self, a: float, z: float, v_ref: float) -> tuple[float, float]:
        """(underside, top) of the first solid above v_ref at transverse a."""
        for lo, hi in self.vertical(a, z):
            if hi > v_ref and lo >= v_ref - 1e-9:
                return lo, hi
        raise ValueError("no solid above the reference level")

    def face_below(self, a: float, z: float, v_ref: float) -> tuple[float, float]:
        """(top face, bottom face) of the first solid below v_ref at transverse a."""
        for lo, hi in reversed(self.vertical(a, z)):
            if lo < v_ref and hi <= v_ref + 1e-9:
                return hi, lo
        raise ValueError("no solid below the reference level")

    def web(self, v: float, z: float, a_ref: float, side: int) -> tuple[float, float]:
        """(inner face, outer face) of the first solid from a_ref towards ``side`` (±1)."""
        row = self.transverse(v, z)
        if side > 0:
            for lo, hi in row:
                if lo >= a_ref - 1e-9:
                    return lo, hi
        else:
            for lo, hi in reversed(row):
                if hi <= a_ref + 1e-9:
                    return hi, lo
        raise ValueError("no web found")
