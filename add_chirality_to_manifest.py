#!/usr/bin/env python3
"""
add_chirality_to_manifest.py (Step 3)

Reads:
  - /out/stl_manifest.json
  - /out/stl/*.stl

Writes:
  - updates /out/stl_manifest.json in-place:
      adds sig_chiral, sig_free (and signature_meta)

Signature approach (mesh-based, deterministic):
  1) Read STL vertices
  2) Deterministic subsample if huge
  3) Center at centroid
  4) PCA -> eigenvectors sorted by eigenvalues desc
  5) Make frame right-handed (det>0)
  6) Fix axis signs deterministically using third moment along each axis
  7) Transform vertices to PCA frame, quantize to tol (default 0.1mm), sort points, hash bytes
  8) Mirror X in PCA frame and hash again => sig_free = min(sig, sig_mirror)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

# numpy is required here (explicit)
import numpy as np


# ---------------------------
# Constants / guards
# ---------------------------

SIG_TOL_MM = 0.1
SIG_SCALE = int(round(1.0 / SIG_TOL_MM))  # 10
MAX_VERTICES_USED = 200_000  # bounded for performance/determinism


# ---------------------------
# JSON helpers
# ---------------------------

def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


# ---------------------------
# STL parsing (binary + ASCII)
# ---------------------------

def _is_probably_binary_stl(buf: bytes) -> bool:
    # Heuristic: binary STL has 84+ bytes and an uint32 triangle count consistent with length
    if len(buf) < 84:
        return False
    tri = struct.unpack_from("<I", buf, 80)[0]
    expected = 84 + tri * 50
    return expected == len(buf)


def _read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if _is_probably_binary_stl(data):
        tri = struct.unpack_from("<I", data, 80)[0]
        off = 84
        verts: List[Tuple[float, float, float]] = []
        # Each tri record: normal(12) + v1(12) + v2(12) + v3(12) + attr(2)
        for _ in range(tri):
            # skip normal
            off += 12
            v1 = struct.unpack_from("<fff", data, off); off += 12
            v2 = struct.unpack_from("<fff", data, off); off += 12
            v3 = struct.unpack_from("<fff", data, off); off += 12
            off += 2  # attribute
            verts.append(v1); verts.append(v2); verts.append(v3)
        return np.asarray(verts, dtype=np.float64)

    # ASCII fallback
    txt = data.decode("utf-8", errors="ignore").splitlines()
    verts2: List[Tuple[float, float, float]] = []
    for line in txt:
        s = line.strip()
        if not s.lower().startswith("vertex "):
            continue
        parts = s.split()
        if len(parts) != 4:
            continue
        try:
            verts2.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except Exception:
            continue
    if not verts2:
        raise RuntimeError(f"Failed to parse STL (no vertices): {path}")
    return np.asarray(verts2, dtype=np.float64)


# ---------------------------
# Deterministic signature
# ---------------------------

def _subsample_vertices(V: np.ndarray, max_n: int) -> np.ndarray:
    n = int(V.shape[0])
    if n <= max_n:
        return V
    stride = (n // max_n) + 1
    return V[::stride].copy()


def _pca_frame(X: np.ndarray) -> np.ndarray:
    # X: Nx3 centered
    # covariance
    n = float(X.shape[0])
    C = (X.T @ X) / max(n, 1.0)
    w, Q = np.linalg.eigh(C)  # eigenvalues asc
    idx = np.argsort(w)[::-1]
    Q = Q[:, idx]  # columns are principal axes

    # enforce right-handed
    if np.linalg.det(Q) < 0.0:
        Q[:, 2] *= -1.0

    # deterministic sign fixing using 3rd moment along each axis
    # (helps remove sign ambiguity and makes mirror detectable)
    for i in range(3):
        a = Q[:, i]
        proj = X @ a
        m3 = float(np.mean(proj ** 3))
        if m3 < 0.0:
            Q[:, i] *= -1.0

    # re-enforce right-handed after sign flips
    if np.linalg.det(Q) < 0.0:
        Q[:, 2] *= -1.0

    return Q


def _quantize_int(X: np.ndarray) -> np.ndarray:
    # round to tol and scale to int
    Qi = np.rint(X * SIG_SCALE).astype(np.int32)
    return Qi


def _hash_points_int(Qi: np.ndarray) -> str:
    # sort lexicographically for deterministic order
    # (bounded by MAX_VERTICES_USED, so OK)
    order = np.lexsort((Qi[:, 2], Qi[:, 1], Qi[:, 0]))
    S = Qi[order]

    h = hashlib.sha256()
    # pack as little-endian int32 triples
    for i in range(S.shape[0]):
        h.update(struct.pack("<iii", int(S[i, 0]), int(S[i, 1]), int(S[i, 2])))
    return h.hexdigest()


def compute_sig_chiral_and_free(vertices_mm: np.ndarray) -> Tuple[str, str, Dict[str, Any]]:
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise RuntimeError("vertices must be Nx3")

    V = _subsample_vertices(vertices_mm, MAX_VERTICES_USED)

    # center
    centroid = np.mean(V, axis=0)
    X = V - centroid

    # PCA frame
    Q = _pca_frame(X)  # 3x3

    # transform to PCA frame
    Y = X @ Q  # Nx3

    # chiral hash
    Yi = _quantize_int(Y)
    sig = _hash_points_int(Yi)

    # mirror X in PCA frame
    Y_m = Y.copy()
    Y_m[:, 0] *= -1.0
    Ymi = _quantize_int(Y_m)
    sig_m = _hash_points_int(Ymi)

    sig_free = min(sig, sig_m)

    meta = {
        "sig_tol_mm": SIG_TOL_MM,
        "sig_scale": SIG_SCALE,
        "vertices_in_file": int(vertices_mm.shape[0]),
        "vertices_used": int(V.shape[0]),
        "centroid_mm": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
        "pca_det": float(np.linalg.det(Q)),
    }
    return sig, sig_free, meta


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/out", help="Output dir (default: /out)")
    ap.add_argument("--manifest", default="/out/stl_manifest.json", help="Manifest path")
    ns = ap.parse_args()

    out_dir = Path(ns.out_dir)
    manifest_path = Path(ns.manifest)

    mani = _read_json(manifest_path)
    items = mani.get("items", [])
    if not isinstance(items, list):
        raise RuntimeError("manifest missing 'items' list")

    updated = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        rel = item.get("stl_path")
        if not rel:
            continue

        stl_path = out_dir / str(rel)
        if not stl_path.exists():
            raise FileNotFoundError(f"Missing STL referenced by manifest: {stl_path}")

        V = _read_stl_vertices(stl_path)
        sig_chiral, sig_free, meta = compute_sig_chiral_and_free(V)

        item["sig_chiral"] = sig_chiral
        item["sig_free"] = sig_free
        item["signature_meta"] = meta
        updated += 1

    # update manifest meta
    meta0 = mani.get("meta", {})
    if not isinstance(meta0, dict):
        meta0 = {}
    meta0["chirality_updated_utc"] = _utc_now_iso()
    meta0["chirality_counts"] = {"items_updated": updated}
    mani["meta"] = meta0

    _write_json(manifest_path, mani)
    print(f"[add_chirality_to_manifest] updated items: {updated}")
    print(f"[add_chirality_to_manifest] wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
