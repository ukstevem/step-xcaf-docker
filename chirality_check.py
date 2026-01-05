#!/usr/bin/env python3
from __future__ import annotations

import itertools
import sys
import numpy as np
import pyvista as pv
import hashlib


def mesh_hash_pca(mesh: pv.DataSet, tol_mm: float = 0.5, allow_reflection: bool = True, max_pts: int = 4000) -> str:
    pts = np.asarray(mesh.points, dtype=np.float64)
    if pts.shape[0] == 0:
        return "0" * 12

    # deterministic downsample
    if pts.shape[0] > max_pts:
        step = max(1, pts.shape[0] // max_pts)
        pts = pts[::step]

    # center
    pts = pts - pts.mean(axis=0, keepdims=True)

    # PCA basis
    C = np.cov(pts.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    B = V[:, order]

    # enforce right-handed basis
    if np.linalg.det(B) < 0:
        B[:, 2] *= -1

    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    signs = list(itertools.product((-1, 1), repeat=3))

    best = None
    for p in perms:
        P = B[:, p]
        for s in signs:
            R = P * np.array(s, dtype=np.float64)[None, :]
            det = float(np.linalg.det(R))
            if (not allow_reflection) and det < 0:
                continue

            q = pts @ R
            q = np.round(q / tol_mm) * tol_mm
            q = q[np.lexsort((q[:, 2], q[:, 1], q[:, 0]))]

            h = hashlib.sha1(q.tobytes()).hexdigest()
            if best is None or h < best:
                best = h

    return (best[:12] if best is not None else ("0" * 12))


def mirror_x(mesh: pv.DataSet) -> pv.DataSet:
    m = mesh.copy(deep=True)
    pts = np.asarray(m.points, dtype=np.float64).copy()
    pts[:, 0] *= -1.0
    m.points = pts
    return m


def bbox_info(mesh: pv.DataSet) -> str:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    dx, dy, dz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    return f"bounds dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}"


def main(stl_path: str, tol_mm: float = 0.5, stl_path_2: str | None = None) -> int:
    mesh = pv.read(stl_path)
    if getattr(mesh, "n_points", 0) == 0:
        print("ERROR: empty mesh")
        return 2

    if stl_path_2:
        mir = pv.read(stl_path_2)
    else:
        mir = mirror_x(mesh)

    free_a = mesh_hash_pca(mesh, tol_mm=tol_mm, allow_reflection=True)
    free_b = mesh_hash_pca(mir,  tol_mm=tol_mm, allow_reflection=True)

    ch_a = mesh_hash_pca(mesh, tol_mm=tol_mm, allow_reflection=False)
    ch_b = mesh_hash_pca(mir,  tol_mm=tol_mm, allow_reflection=False)

    print("A:", stl_path)
    print("B:", stl_path_2 or "(mirrored in script)")
    print()
    print("FREE   (reflection allowed):", free_a, free_b, "match?", free_a == free_b)
    print("CHIRAL (reflection NOT allowed):", ch_a, ch_b, "differ?", ch_a != ch_b)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chirality_check.py <a.stl> [tol_mm] [b.stl]")
        raise SystemExit(2)

    a = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    b = sys.argv[3] if len(sys.argv) > 3 else None
    raise SystemExit(main(a, tol, b))

