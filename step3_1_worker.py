#!/usr/bin/env python3
"""
Step 3.1 Worker: Parts Index + Chirality Grouping (common components layer)

Goal
----
Build a deterministic "parts index" that:
  - groups geometrically identical parts (even if mirrored) into a common family
  - detects chirality variants (LH/RH) within a family
  - produces a canonical "common_key" that downstream steps (categorisation, costing, outputs)
    can use to treat repeated parts consistently

Why this exists
---------------
BOM rows can repeat the same part signature in multiple contexts (different parents, names, etc.).
Also, mirrored parts are common in fabrication (LH/RH). If we categorise per-row, we get:
  - inconsistent category results (depends which row had bbox, etc.)
  - inability to identify mirrored mates

This worker runs AFTER explode (so we have solid STLs) and BEFORE categorisation,
so categorisation can be keyed on "common_key" and remain stable.

Inputs
------
Reads active BOM (first that exists):
  - /app/ui_runs/<run_id>/bom_global_exploded.json   (preferred)
  - /app/ui_runs/<run_id>/bom_global.json

Uses STL paths referenced by BOM rows:
  - row["stl_url"] points under /runs/<run_id>/...

Outputs
-------
Writes:
  - /app/ui_runs/<run_id>/parts_index.json

Key output fields
-----------------
For each BOM signature key (sig_key), we compute:
  - free_sig: hash of geometry with reflection allowed (groups mirrored together)
  - chiral_sig: hash of geometry with reflection disallowed (distinguishes LH/RH)
  - group_id: the family id (free_sig)
  - common_key: canonical key for the family (stable)
  - is_mirrored: True if this instance is not canonical in its family
  - mirror_of: canonical member sig_key (if mirrored)
  - rep_stl_url / bbox_mm (for UI / debugging)

Determinism / Performance
-------------------------
- bounded downsample of mesh points
- deterministic ordering and tie-breaking
- caches per STL path to avoid recompute
- env-configurable tolerances

Env knobs (optional)
--------------------
STEP3_1_TOL_MM=0.5
STEP3_1_ALLOW_REFLECTION=1
STEP3_1_MAX_PTS=4000
STEP3_1_MAX_ROWS=200000
STEP3_1_DEBUG=0
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# pyvista is already in your chirality script; keep it here for STL reads
import pyvista as pv


# -----------------------------
# Explicit constants / limits
# -----------------------------

SCHEMA_ID = "parts_index_v1"
RULESET_ID = "step3_1_parts_index_v1"

DEFAULT_TOL_MM = 0.5
DEFAULT_MAX_PTS = 4000
DEFAULT_MAX_ROWS = 200_000

PARTS_INDEX_NAME = "parts_index.json"

# Key fallback is intentionally last resort; keep deterministic.
_SIG_KEY_FALLBACK_PREFIX = "name:"


# -----------------------------
# Small helpers (Power-of-10)
# -----------------------------

def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        v = float(raw)
    except Exception:
        return float(default)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        v = int(raw)
    except Exception:
        return int(default)
    if v < lo:
        return int(lo)
    if v > hi:
        return int(hi)
    return int(v)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


CFG: Dict[str, Any] = {
    "STEP3_1_TOL_MM": _env_float("STEP3_1_TOL_MM", DEFAULT_TOL_MM, 0.05, 5.0),
    "STEP3_1_ALLOW_REFLECTION": _env_bool("STEP3_1_ALLOW_REFLECTION", True),
    "STEP3_1_MAX_PTS": _env_int("STEP3_1_MAX_PTS", DEFAULT_MAX_PTS, 256, 50_000),
    "STEP3_1_MAX_ROWS": _env_int("STEP3_1_MAX_ROWS", DEFAULT_MAX_ROWS, 1, 1_000_000),
    "STEP3_1_DEBUG": _env_bool("STEP3_1_DEBUG", False),
}


def _config_hash(cfg: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in sorted(cfg.keys()):
        parts.append(f"{k}={cfg[k]}")
    s = "|".join(parts).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


CONFIG_HASH = _config_hash(CFG)


def _read_json_obj(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON not object: {path}")
    return obj


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _norm_name(s: str) -> str:
    t = (s or "").strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def _active_bom_path(run_dir: Path) -> Path:
    p = run_dir / "bom_global_exploded.json"
    if p.is_file():
        return p
    return run_dir / "bom_global.json"


def _sig_key(row: Dict[str, Any]) -> str:
    """
    Must match UI + categoriser keying.
    """
    sig = row.get("ref_def_sig")
    if isinstance(sig, str) and sig.strip():
        return sig.strip()

    key = row.get("key")
    if isinstance(key, str) and key.startswith("sig:"):
        return key[4:].strip()

    ref_id = row.get("ref_def_id")
    if isinstance(ref_id, str) and ref_id.strip():
        return "def:" + ref_id.strip()

    return _SIG_KEY_FALLBACK_PREFIX + _norm_name(str(row.get("def_name") or ""))


def _stl_url_to_path(run_dir: Path, stl_url: str) -> Optional[Path]:
    """
    Convert /runs/<run_id>/... URL to local filesystem path under run_dir.
    """
    if not isinstance(stl_url, str) or not stl_url.strip():
        return None
    u = stl_url.strip()
    marker = "/runs/"
    i = u.find(marker)
    if i < 0:
        return None
    # /runs/<run_id>/rest/of/path
    rest = u[i + len(marker):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        return None
    rel = parts[1]
    # join to run_dir
    p = run_dir / rel
    return p


def _bbox_size_sorted(row: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    bbox = row.get("bbox_mm")
    if not isinstance(bbox, dict):
        return None
    size = bbox.get("size")
    if not (isinstance(size, list) and len(size) == 3):
        return None
    try:
        x, y, z = float(size[0]), float(size[1]), float(size[2])
    except Exception:
        return None
    dims = sorted([abs(x), abs(y), abs(z)], reverse=True)
    if dims[2] <= 1e-9:
        return None
    return (dims[0], dims[1], dims[2])


# -----------------------------
# Chirality hashing (from your script)
# -----------------------------

def _mesh_hash_pca(
    mesh: pv.DataSet,
    tol_mm: float,
    allow_reflection: bool,
    max_pts: int,
) -> str:
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
    if float(np.linalg.det(B)) < 0.0:
        B[:, 2] *= -1.0

    perms = (
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    )
    signs = list(itertools.product((-1, 1), repeat=3))

    best = None
    for p in perms:
        P = B[:, p]
        for s in signs:
            R = P * np.array(s, dtype=np.float64)[None, :]
            det = float(np.linalg.det(R))
            if (not allow_reflection) and det < 0.0:
                continue

            q = pts @ R
            q = np.round(q / tol_mm) * tol_mm
            q = q[np.lexsort((q[:, 2], q[:, 1], q[:, 0]))]

            h = hashlib.sha1(q.tobytes()).hexdigest()
            if best is None or h < best:
                best = h

    return best[:12] if best is not None else ("0" * 12)


@dataclass(frozen=True)
class HashPair:
    free_sig: str
    chiral_sig: str
    n_points: int


def _hash_stl_cached(
    stl_path: Path,
    cache: Dict[str, HashPair],
    tol_mm: float,
    max_pts: int,
) -> Optional[HashPair]:
    key = str(stl_path)
    got = cache.get(key)
    if got is not None:
        return got

    if not stl_path.is_file():
        cache[key] = HashPair("0" * 12, "0" * 12, 0)
        return cache[key]

    try:
        mesh = pv.read(str(stl_path))
    except Exception:
        cache[key] = HashPair("0" * 12, "0" * 12, 0)
        return cache[key]

    npts = int(getattr(mesh, "n_points", 0) or 0)
    if npts <= 0:
        cache[key] = HashPair("0" * 12, "0" * 12, 0)
        return cache[key]

    free_sig = _mesh_hash_pca(mesh, tol_mm=tol_mm, allow_reflection=True, max_pts=max_pts)
    ch_sig = _mesh_hash_pca(mesh, tol_mm=tol_mm, allow_reflection=False, max_pts=max_pts)
    hp = HashPair(free_sig=free_sig, chiral_sig=ch_sig, n_points=npts)
    cache[key] = hp
    return hp


# -----------------------------
# Main worker
# -----------------------------

def run_step3_1(run_dir: Path) -> Path:
    bom_path = _active_bom_path(run_dir)
    if not bom_path.is_file():
        raise FileNotFoundError(f"Missing BOM: {bom_path}")

    bom = _read_json_obj(bom_path)
    rows = bom.get("items")
    if not isinstance(rows, list):
        raise ValueError(f"BOM has no items[] list: {bom_path}")

    max_rows = int(CFG["STEP3_1_MAX_ROWS"])
    if len(rows) > max_rows:
        raise RuntimeError(f"Too many BOM rows ({len(rows)} > {max_rows}). Increase STEP3_1_MAX_ROWS if intended.")

    tol_mm = float(CFG["STEP3_1_TOL_MM"])
    max_pts = int(CFG["STEP3_1_MAX_PTS"])
    debug = bool(CFG["STEP3_1_DEBUG"])

    # 1) Compute hash pairs per unique STL path (cached)
    cache: Dict[str, HashPair] = {}
    per_sig: Dict[str, Dict[str, Any]] = {}

    # Also build family groups by free_sig
    groups: Dict[str, List[str]] = {}  # free_sig -> list of sig_keys

    # deterministic pass
    for row in rows:
        if not isinstance(row, dict):
            continue

        sigk = _sig_key(row)

        stl_url = row.get("stl_url")
        stl_path = _stl_url_to_path(run_dir, str(stl_url or "")) if stl_url else None

        bbox_sorted = _bbox_size_sorted(row)
        bbox_mm = row.get("bbox_mm") if isinstance(row.get("bbox_mm"), dict) else None

        # choose a "best representative" row per sig:
        # prefer ones with an STL path and bbox info
        rec = per_sig.get(sigk)
        if rec is None:
            rec = {
                "sig_key": sigk,
                "rep_stl_url": stl_url if isinstance(stl_url, str) else None,
                "rep_stl_path": str(stl_path) if stl_path else None,
                "bbox_mm": bbox_mm,
                "bbox_sorted": list(bbox_sorted) if bbox_sorted else None,
                "examples": 0,
                "where_used": [],  # compact
            }
            per_sig[sigk] = rec

        rec["examples"] = int(rec.get("examples") or 0) + 1

        # where_used: keep bounded and useful
        if len(rec["where_used"]) < 10:
            w = {
                "def_name": row.get("def_name"),
                "occ_label_sample": row.get("occ_label_sample"),
                "from_parent_def_sig": row.get("from_parent_def_sig"),
                "from_parent_def_name": row.get("from_parent_def_name"),
            }
            rec["where_used"].append(w)

        # upgrade representative if missing STL and we have one now
        if (not rec.get("rep_stl_path")) and stl_path is not None and stl_path.is_file():
            rec["rep_stl_url"] = stl_url
            rec["rep_stl_path"] = str(stl_path)

        # upgrade bbox if missing
        if rec.get("bbox_mm") is None and bbox_mm is not None:
            rec["bbox_mm"] = bbox_mm
            rec["bbox_sorted"] = list(bbox_sorted) if bbox_sorted else None

    # 2) Hash each representative mesh
    for sigk in sorted(per_sig.keys()):
        rec = per_sig[sigk]
        p = rec.get("rep_stl_path")
        if isinstance(p, str) and p:
            hp = _hash_stl_cached(Path(p), cache, tol_mm=tol_mm, max_pts=max_pts)
        else:
            hp = None

        if hp is None or hp.n_points <= 0:
            rec["free_sig"] = None
            rec["chiral_sig"] = None
            rec["n_points"] = 0
            rec["group_id"] = None
        else:
            rec["free_sig"] = hp.free_sig
            rec["chiral_sig"] = hp.chiral_sig
            rec["n_points"] = int(hp.n_points)
            rec["group_id"] = hp.free_sig

            groups.setdefault(hp.free_sig, []).append(sigk)

    # 3) For each group (free_sig), decide canonical + mirror mapping by chiral_sig
    # canonical_chiral = lexicographically smallest chiral_sig present
    group_meta: Dict[str, Any] = {}
    for free_sig in sorted(groups.keys()):
        members = sorted(groups[free_sig])

        chiral_sigs: List[str] = []
        for m in members:
            cs = per_sig[m].get("chiral_sig")
            if isinstance(cs, str) and cs:
                chiral_sigs.append(cs)

        chiral_sigs = sorted(set(chiral_sigs))
        canonical_chiral = chiral_sigs[0] if chiral_sigs else None

        # choose canonical member: lowest sig_key among those matching canonical chiral
        canonical_member = None
        if canonical_chiral is not None:
            for m in members:
                if per_sig[m].get("chiral_sig") == canonical_chiral:
                    canonical_member = m
                    break

        common_key = None
        if canonical_chiral is not None:
            # stable family key (keep short but deterministic)
            common_key = f"common:{free_sig}:{canonical_chiral}"
        else:
            # fallback (no STL -> no hash)
            common_key = f"common:none:{free_sig}"

        group_meta[free_sig] = {
            "group_id": free_sig,
            "member_count": len(members),
            "variant_count": len(chiral_sigs),
            "canonical_chiral_sig": canonical_chiral,
            "canonical_member_sig_key": canonical_member,
            "common_key": common_key,
        }

        for m in members:
            cs = per_sig[m].get("chiral_sig")
            per_sig[m]["common_key"] = common_key
            per_sig[m]["canonical_member_sig_key"] = canonical_member
            if canonical_member is not None and m != canonical_member and isinstance(cs, str) and canonical_chiral is not None:
                per_sig[m]["is_mirrored"] = (cs != canonical_chiral)
                per_sig[m]["mirror_of"] = canonical_member if per_sig[m]["is_mirrored"] else None
            else:
                per_sig[m]["is_mirrored"] = False
                per_sig[m]["mirror_of"] = None

    # 4) Write output
    out = {
        "schema": SCHEMA_ID,
        "ruleset": RULESET_ID,
        "generated_at": _utc_iso_z(),
        "source_bom": bom_path.name,
        "config": dict(CFG),
        "config_hash": CONFIG_HASH,
        "summary": {
            "rows_in_bom": len(rows),
            "unique_sig_keys": len(per_sig),
            "groups_with_hash": len(groups),
            "groups_total": len(group_meta),
        },
        "groups": group_meta,     # keyed by free_sig
        "items": {},              # keyed by sig_key
    }

    items_out: Dict[str, Any] = {}
    for sigk in sorted(per_sig.keys()):
        rec = per_sig[sigk]
        items_out[sigk] = {
            "sig_key": sigk,
            "common_key": rec.get("common_key"),
            "group_id": rec.get("group_id"),
            "free_sig": rec.get("free_sig"),
            "chiral_sig": rec.get("chiral_sig"),
            "is_mirrored": bool(rec.get("is_mirrored") is True),
            "mirror_of": rec.get("mirror_of"),
            "canonical_member_sig_key": rec.get("canonical_member_sig_key"),
            "rep_stl_url": rec.get("rep_stl_url"),
            "bbox_mm": rec.get("bbox_mm"),
            "bbox_sorted": rec.get("bbox_sorted"),
            "n_points": int(rec.get("n_points") or 0),
            "examples": int(rec.get("examples") or 0),
            # bounded where_used for UI debug
            "where_used": rec.get("where_used") if isinstance(rec.get("where_used"), list) else [],
        }

    out["items"] = items_out

    out_path = run_dir / PARTS_INDEX_NAME
    _write_json_atomic(out_path, out)

    if debug:
        print(f"[step3_1] wrote {out_path} groups={len(groups)} items={len(items_out)} tol={tol_mm} max_pts={max_pts}")

    return out_path


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python step3_1_worker.py <run_dir>")
        return 2
    run_dir = Path(argv[1])
    out = run_step3_1(run_dir)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(list(os.sys.argv)))
