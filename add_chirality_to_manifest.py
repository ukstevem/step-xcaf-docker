
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyvista as pv


def _quantize(v: float, tol: float) -> int:
    tol = max(float(tol), 1e-6)
    return int(round(float(v) / tol))


def _sig_from_points(
    pts: np.ndarray,
    centroid: Tuple[float, float, float],
    dims: Tuple[float, float, float],
    vol: float,
    tol: float,
    reflect: Tuple[int, int, int],
    max_pts: int = 1200,
) -> str:
    n = int(pts.shape[0])
    if n <= 0:
        return "0" * 12

    stride = max(1, n // max_pts)
    cx, cy, cz = centroid
    sx, sy, sz = reflect
    dx, dy, dz = dims

    header = (
        _quantize(dx, tol),
        _quantize(dy, tol),
        _quantize(dz, tol),
        _quantize(vol, max(tol**3, 1.0)),
    )

    triples: List[Tuple[int, int, int]] = []
    count = 0
    for i in range(0, n, stride):
        x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
        triples.append(
            (
                sx * _quantize(x - cx, tol),
                sy * _quantize(y - cy, tol),
                sz * _quantize(z - cz, tol),
            )
        )
        count += 1
        if count >= max_pts:
            break

    triples.sort()
    h = hashlib.sha1()
    for val in header:
        h.update(int(val).to_bytes(8, "little", signed=True))
    for a, b, c in triples:
        h.update(int(a).to_bytes(4, "little", signed=True))
        h.update(int(b).to_bytes(4, "little", signed=True))
        h.update(int(c).to_bytes(4, "little", signed=True))
    return h.hexdigest()[:12]


def _metrics(mesh: pv.PolyData) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    b = mesh.bounds  # xmin,xmax,ymin,ymax,zmin,zmax
    dx = float(b[1] - b[0])
    dy = float(b[3] - b[2])
    dz = float(b[5] - b[4])
    dims = (dx, dy, dz)

    try:
        com = mesh.center_of_mass()
        centroid = (float(com[0]), float(com[1]), float(com[2]))
    except Exception:
        pts = mesh.points
        centroid = (float(pts[:, 0].mean()), float(pts[:, 1].mean()), float(pts[:, 2].mean()))

    vol = float(abs(getattr(mesh, "volume", 0.0)))
    return centroid, dims, vol


def add_chirality(out_dir: str, tol: float) -> int:
    outp = Path(out_dir)
    manifest_path = outp / "stl_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 2

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        items = data
        wrap = False
    elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
        wrap = True
    else:
        print("ERROR: stl_manifest.json format not recognized", file=sys.stderr)
        return 2

    updated = 0
    skipped = 0

    for it in items:
        stl_rel = it.get("stl_path", "") or ""
        if not stl_rel:
            skipped += 1
            continue

        stl_path = (outp / stl_rel).resolve()
        if not stl_path.exists():
            stl_path = (outp / stl_rel.lstrip("/")).resolve()
            if not stl_path.exists():
                skipped += 1
                continue

        mesh = pv.read(str(stl_path))
        if getattr(mesh, "n_points", 0) <= 0:
            skipped += 1
            continue

        pts = np.asarray(mesh.points, dtype=np.float64)
        centroid, dims, vol = _metrics(mesh)

        sig_chiral = _sig_from_points(pts, centroid, dims, vol, tol, (1, 1, 1))

        cands = []
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    cands.append(_sig_from_points(pts, centroid, dims, vol, tol, (sx, sy, sz)))
        sig_free = min(cands) if cands else sig_chiral

        it["sig_chiral"] = sig_chiral
        it["sig_free"] = sig_free
        updated += 1

    if wrap:
        data["items"] = items
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    print(f"Updated: {manifest_path}")
    print(f"Chirality signatures added: {updated} (skipped: {skipped})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: add_chirality_to_manifest.py /out [tol_mm]", file=sys.stderr)
        raise SystemExit(2)
    out_dir = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    raise SystemExit(add_chirality(out_dir, tol))
