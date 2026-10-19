"""Export the 3-D bridge mesh to the compact ``.npz`` used at runtime.

Development tool only — it needs dolfinx, the runtime package does not.

The mesh is read with ``dolfinx.io.gmshio`` exactly as the research pipeline
(``code/src/simulation/thermomechanical_ensemble.py``) reads it, so the facet
markers are resolved the same way: in ``mesh_3d_thermal.msh`` a surface can
belong to several physical groups (e.g. ``(8, 100)``) and gmshio keeps one
of them per facet.  Re-implementing that rule in a hand-written ``.msh``
parser would be a silent source of differences in which faces receive
convection / radiation, so the dolfinx result is frozen into the ``.npz``.

Usage (from ``code/``, in the ``snakemake_bridge`` conda env)::

    python virtual_sensors/tools/export_mesh.py \
        models/mesh_3d_thermal.msh \
        virtual_sensors/src/nibelungen_virtual_sensors/data/mesh_3d.npz
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def export(msh_path: Path, out_path: Path) -> None:
    import dolfinx as df
    from mpi4py import MPI

    mesh, _, facet_tags = df.io.gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, 0, 3)
    tdim = mesh.topology.dim

    # P1 geometry: one geometry node per vertex, so the geometry dofmap is
    # the P1 cell connectivity in geometry-node numbering.
    points = mesh.geometry.x[:, :3].astype(np.float64)
    tets = mesh.geometry.dofmap.array.reshape(-1, 4).astype(np.int32)

    mesh.topology.create_connectivity(tdim - 1, 0)
    f2v = mesh.topology.connectivity(tdim - 1, 0)
    # vertex index -> geometry node index
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    v2g = np.empty(mesh.topology.index_map(0).size_local, dtype=np.int64)
    for c in range(tets.shape[0]):
        v2g[c2v.links(c)] = tets[c]

    facets = np.array([v2g[f2v.links(f)] for f in facet_tags.indices], dtype=np.int32)
    tags = facet_tags.values.astype(np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=points,
        tets=tets,
        facets=facets,
        facet_tags=tags,
        source_sha256=np.array(hashlib.sha256(msh_path.read_bytes()).hexdigest()),
    )
    uniq, cnt = np.unique(tags, return_counts=True)
    print(f"{points.shape[0]} nodes, {tets.shape[0]} tets, facet tags {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("msh", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    export(a.msh, a.out)
