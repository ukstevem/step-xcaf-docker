<<<<<<< HEAD
﻿#!/usr/bin/env python3
"""
Step 3: add_chirality_to_manifest.py

Reads assets_manifest.json and adds chirality signatures for each item that has stl_path.

Outputs:
  - updates /out/assets_manifest.json in-place

Fields added per item:
  - chirality_sig       : mirror-sensitive hash
  - chirality_sig_free  : mirror-invariant hash (min over axis-flip variants)
  - chirality_algo      : algorithm id string
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CHIRALITY_ALGO = "stl_features_v1_sha256"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _quant(v: float, tol: float) -> int:
    return int(round(float(v) / tol))


def _tri_area(ax, ay, az, bx, by, bz, cx, cy, cz) -> float:
    abx, aby, abz = (bx - ax), (by - ay), (bz - az)
    acx, acy, acz = (cx - ax), (cy - ay), (cz - az)
    cxp = aby * acz - abz * acy
    cyp = abz * acx - abx * acz
    czp = abx * acy - aby * acx
    return 0.5 * math.sqrt(cxp * cxp + cyp * cyp + czp * czp)


def _is_binary_stl(buf: bytes) -> bool:
    if len(buf) < 84:
        return False
    # binary STL: 80 header + uint32 tri count + 50 bytes per tri
    tri_count = struct.unpack("<I", buf[80:84])[0]
    expected = 84 + (tri_count * 50)
    return expected == len(buf)


def _iter_triangles_from_binary_stl(buf: bytes) -> Iterable[Tuple[float, ...]]:
    tri_count = struct.unpack("<I", buf[80:84])[0]
    off = 84
    for _ in range(tri_count):
        # normal (3f) + 3 vertices (9f) + attr (uint16)
        rec = buf[off : off + 50]
        off += 50
        vals = struct.unpack("<12fH", rec)
        # ignore normal vals[0:3], use vertices
        ax, ay, az = vals[3], vals[4], vals[5]
        bx, by, bz = vals[6], vals[7], vals[8]
        cx, cy, cz = vals[9], vals[10], vals[11]
        yield (ax, ay, az, bx, by, bz, cx, cy, cz)


def _iter_triangles_from_ascii_stl(text: str) -> Iterable[Tuple[float, ...]]:
    # Basic ASCII STL parser: look for "vertex x y z"
    verts: List[Tuple[float, float, float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.lower().startswith("vertex "):
            continue
        parts = s.split()
        if len(parts) != 4:
            continue
        try:
            x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
        except Exception:
            continue
        verts.append((x, y, z))
        if len(verts) == 3:
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = verts
            verts = []
            yield (ax, ay, az, bx, by, bz, cx, cy, cz)


def _load_triangles(stl_path: Path) -> List[Tuple[float, ...]]:
    buf = stl_path.read_bytes()
    if _is_binary_stl(buf):
        return list(_iter_triangles_from_binary_stl(buf))
    # fallback ascii
    try:
        text = buf.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return list(_iter_triangles_from_ascii_stl(text))


def _bbox_of_tris(tris: List[Tuple[float, ...]]) -> Tuple[float, float, float, float, float, float]:
    mnx = mny = mnz = float("inf")
    mxx = mxy = mxz = float("-inf")
    for t in tris:
        ax, ay, az, bx, by, bz, cx, cy, cz = t
        for x, y, z in ((ax, ay, az), (bx, by, bz), (cx, cy, cz)):
            mnx = min(mnx, x); mny = min(mny, y); mnz = min(mnz, z)
            mxx = max(mxx, x); mxy = max(mxy, y); mxz = max(mxz, z)
    if not math.isfinite(mnx):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (mnx, mny, mnz, mxx, mxy, mxz)


def _features_for_tris(tris: List[Tuple[float, ...]], tol_mm: float) -> Dict[str, Any]:
    mnx, mny, mnz, mxx, mxy, mxz = _bbox_of_tris(tris)
    dx, dy, dz = (mxx - mnx), (mxy - mny), (mxz - mnz)

    # area histogram (quantized)
    areas_q: List[int] = []
    for t in tris:
        a = _tri_area(*t)
        areas_q.append(_quant(a, tol_mm * tol_mm))
    areas_q.sort()

    # vertex cloud signature (order-independent)
    vq: List[Tuple[int, int, int]] = []
    for t in tris:
        ax, ay, az, bx, by, bz, cx, cy, cz = t
        vq.append((_quant(ax, tol_mm), _quant(ay, tol_mm), _quant(az, tol_mm)))
        vq.append((_quant(bx, tol_mm), _quant(by, tol_mm), _quant(bz, tol_mm)))
        vq.append((_quant(cx, tol_mm), _quant(cy, tol_mm), _quant(cz, tol_mm)))
    vq.sort()

    # compress vertex list deterministically (bounded)
    # keep first/last N to avoid huge payloads while remaining stable
    N = 2000
    if len(vq) > (2 * N):
        vq_keep = vq[:N] + vq[-N:]
    else:
        vq_keep = vq

    return {
        "tri_count": len(tris),
        "bbox_q": [
            _quant(mnx, tol_mm), _quant(mny, tol_mm), _quant(mnz, tol_mm),
            _quant(mxx, tol_mm), _quant(mxy, tol_mm), _quant(mxz, tol_mm),
        ],
        "ext_q": [_quant(dx, tol_mm), _quant(dy, tol_mm), _quant(dz, tol_mm)],
        "areas_q_head": areas_q[:500],   # bounded
        "areas_q_tail": areas_q[-500:] if len(areas_q) > 500 else [],
        "verts_q_keep": vq_keep,
    }


def _hash_features(feat: Dict[str, Any]) -> str:
    b = json.dumps(feat, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _sha256_hex(b)


def _apply_axis_flip(tris: List[Tuple[float, ...]], flipx: bool, flipy: bool, flipz: bool) -> List[Tuple[float, ...]]:
    out: List[Tuple[float, ...]] = []
    sx = -1.0 if flipx else 1.0
    sy = -1.0 if flipy else 1.0
    sz = -1.0 if flipz else 1.0
    for t in tris:
        ax, ay, az, bx, by, bz, cx, cy, cz = t
        out.append((
            ax * sx, ay * sy, az * sz,
            bx * sx, by * sy, bz * sz,
            cx * sx, cy * sy, cz * sz,
        ))
    return out


def compute_chirality_sigs(stl_path: Path, tol_mm: float) -> Tuple[str, str]:
    tris = _load_triangles(stl_path)
    if not tris:
        # empty -> stable but unique-ish
        empty = _sha256_hex(b"empty-stl")
        return empty, empty

    # mirror-sensitive
    feat = _features_for_tris(tris, tol_mm)
    sig = _hash_features(feat)

    # mirror-invariant across axis flips: pick minimum hash of variants
    # (covers mirrored duplicates along any primary axis)
    free_sigs: List[str] = []
    for flipx, flipy, flipz in (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ):
        tris2 = _apply_axis_flip(tris, flipx, flipy, flipz)
        feat2 = _features_for_tris(tris2, tol_mm)
        free_sigs.append(_hash_features(feat2))
    sig_free = min(free_sigs)

    return sig, sig_free


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/out")
    ap.add_argument("--manifest", default="/out/assets_manifest.json")
    ns = ap.parse_args()

    out_dir = Path(ns.out_dir)
    manifest_path = Path(ns.manifest)

    m = _read_json(manifest_path)
    items = m.get("items")
    if not isinstance(items, list):
        raise RuntimeError("assets_manifest.json missing items[]")

    tol_mm = _env_float("CHIRALITY_TOL_MM", 0.05)

    updated = 0
    scanned = 0
    for idx, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        stl_rel = it.get("stl_path")
        if not (isinstance(stl_rel, str) and stl_rel.strip()):
            continue

        # already done?
        if isinstance(it.get("chirality_sig"), str) and it.get("chirality_sig"):
            continue

        stl_path = out_dir / stl_rel
        if not stl_path.exists():
            it["chirality_sig"] = None
            it["chirality_sig_free"] = None
            it["chirality_algo"] = CHIRALITY_ALGO
            it["chirality_error"] = f"missing_stl:{stl_rel}"
            updated += 1
            continue

        sig, sig_free = compute_chirality_sigs(stl_path, tol_mm)
        it["chirality_sig"] = sig
        it["chirality_sig_free"] = sig_free
        it["chirality_algo"] = CHIRALITY_ALGO
        updated += 1
        scanned += 1

        if (idx % 50) == 0:
            print(f"[add_chirality_to_manifest] progress: {idx}/{len(items)} items scanned...")

    meta = m.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        m["meta"] = meta
    meta["chirality"] = {
        "algo": CHIRALITY_ALGO,
        "tol_mm": float(tol_mm),
        "updated_items": int(updated),
        "scanned_stls": int(scanned),
    }

    _write_json(manifest_path, m)
    print(f"[add_chirality_to_manifest] updated items: {updated}")
    print(f"[add_chirality_to_manifest] wrote: {manifest_path}")
=======
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        triples.append((
            sx * _quantize(x - cx, tol),
            sy * _quantize(y - cy, tol),
            sz * _quantize(z - cz, tol),
        ))
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

    # centroid: prefer COM if available
    try:
        com = mesh.center_of_mass()
        centroid = (float(com[0]), float(com[1]), float(com[2]))
    except Exception:
        pts = mesh.points
        centroid = (float(pts[:, 0].mean()), float(pts[:, 1].mean()), float(pts[:, 2].mean()))

    # volume may be 0 if not watertight; still fine
    vol = float(abs(getattr(mesh, "volume", 0.0)))
    return centroid, dims, vol


def add_chirality(out_dir: str, tol: float) -> int:
    outp = Path(out_dir)
    manifest_path = outp / "stl_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 2

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Support either list manifest or dict-with-items
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
            # Sometimes paths are stored without leading dirs; try relative
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

        # FREE: reflection allowed => take minimum over 8 reflections
        cands = []
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    cands.append(_sig_from_points(pts, centroid, dims, vol, tol, (sx, sy, sz)))
        sig_free = min(cands) if cands else sig_chiral

        it["sig_chiral"] = sig_chiral
        it["sig_free"] = sig_free
        updated += 1

    # write back
    if wrap:
        data["items"] = items
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    print(f"Updated: {manifest_path}")
    print(f"Chirality signatures added: {updated} (skipped: {skipped})")
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b
    return 0


if __name__ == "__main__":
<<<<<<< HEAD
    raise SystemExit(main())
=======
    if len(sys.argv) < 2:
        print("Usage: add_chirality_to_manifest.py /out [tol_mm]", file=sys.stderr)
        raise SystemExit(2)
    out_dir = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    raise SystemExit(add_chirality(out_dir, tol))
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b
