"""
Step 4 - Multi-body Review (Stage 1–3 only)

Reads:  xcaf_instances.json (Step 1 output)
Writes:
  out/review/multibody_review.json
  out/review/multibody_decisions.json   (seed/update only; user edits)

No exploding happens here.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from review_rules import bucket_candidate


# ----------------------------
# Constants (stable output)
# ----------------------------

REVIEW_SUBDIR = Path("review")
REVIEW_JSON_NAME = "multibody_review.json"
DECISIONS_JSON_NAME = "multibody_decisions.json"

DECISION_DEFAULT = "defer"
DECISION_ENUM = ("keep_as_one", "explode", "defer")

BUCKET_ORDER = {"likely_explode": 0, "review": 1, "auto_keep": 2}


# ----------------------------
# Small helpers
# ----------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _best_name(defn: Dict[str, Any]) -> str:
    # Try a few common fields; fall back to empty string
    for k in ("name", "def_name", "label", "product_name", "ref_name"):
        v = defn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _format_bbox(defn: Dict[str, Any]) -> str:
    """
    Keep bbox as one column: "X×Y×Z" (mm, 1dp).
    We try common keys; if missing, return "" (and reason remains elsewhere).
    """
    # Common patterns you may have:
    # - bbox_mm: [x, y, z]
    # - bbox: {"dx":..,"dy":..,"dz":..}
    # - bbox_dims_mm: {"x":..,"y":..,"z":..}
    dims = None

    v = defn.get("bbox_mm")
    if isinstance(v, (list, tuple)) and len(v) == 3:
        dims = v

    if dims is None and isinstance(defn.get("bbox"), dict):
        b = defn["bbox"]
        if all(k in b for k in ("dx", "dy", "dz")):
            dims = [b["dx"], b["dy"], b["dz"]]

    if dims is None and isinstance(defn.get("bbox_dims_mm"), dict):
        b = defn["bbox_dims_mm"]
        if all(k in b for k in ("x", "y", "z")):
            dims = [b["x"], b["y"], b["z"]]

    if dims is None:
        return ""

    try:
        x, y, z = float(dims[0]), float(dims[1]), float(dims[2])
        return f"{x:.1f}×{y:.1f}×{z:.1f}"
    except Exception:
        return ""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_qty_by_def_id(occurrences: Dict[str, Any]) -> Dict[str, int]:
    """
    Single pass count of occurrences per ref_def (definition id).
    """
    out: Dict[str, int] = {}
    for _occ_id, occ in occurrences.items():
        if not isinstance(occ, dict):
            continue
        ref_def = occ.get("ref_def")
        if not isinstance(ref_def, str) or not ref_def:
            continue
        out[ref_def] = out.get(ref_def, 0) + 1
    return out

def _read_existing_decisions(path: Path) -> Dict[str, Tuple[str, str, str]]:
    """
    Returns: def_sig -> (decision, note, updated_utc)
    Supports:
      - JSON v1 format: {"schema": "...", "decisions": { "<sig>": {"decision":..,"note":..,"updated_utc":..}}}
    """
    out: Dict[str, Tuple[str, str, str]] = {}
    if not path.exists():
        return out

    try:
        obj = _load_json(path)
    except Exception:
        # Hard fail: we do not want to "quietly" lose user decisions
        raise ValueError(f"Could not parse decisions JSON: {path}")

    if not isinstance(obj, dict):
        raise ValueError(f"Decisions JSON must be an object: {path}")

    decisions = obj.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError(f"Decisions JSON missing 'decisions' object: {path}")

    for sig, rec in decisions.items():
        if not isinstance(sig, str) or not sig.strip():
            continue
        if not isinstance(rec, dict):
            continue

        decision = (rec.get("decision") or "").strip() if isinstance(rec.get("decision"), str) else ""
        note = (rec.get("note") or "").strip() if isinstance(rec.get("note"), str) else ""
        updated_utc = (rec.get("updated_utc") or "").strip() if isinstance(rec.get("updated_utc"), str) else ""

        if decision not in DECISION_ENUM:
            decision = DECISION_DEFAULT

        out[sig.strip()] = (decision, note, updated_utc)

    return out


def _write_decisions(path: Path, decisions: Dict[str, Dict[str, str]]) -> None:
    """
    Writes JSON:
      {
        "schema": "multibody_decisions_v1",
        "updated_utc": "...",
        "decisions": { "<def_sig>": {"decision":"...", "note":"...", "updated_utc":"..."} }
      }
    """
    obj = {
        "schema": "multibody_decisions_v1",
        "updated_utc": _now_utc_iso(),
        "decisions": decisions,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_review(path: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """
    Writes JSON:
      {
        "schema": "multibody_review_v1",
        "created_utc": "...",
        "meta": {...},
        "items": [...]
      }
    """
    obj = {
        "schema": "multibody_review_v1",
        "created_utc": _now_utc_iso(),
        "meta": meta,
        "items": rows,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


# ----------------------------
# Main build
# ----------------------------

def build_multibody_review(xcaf_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    data = _load_json(xcaf_path)

    definitions = data.get("definitions") or {}
    occurrences = data.get("occurrences") or {}

    if not isinstance(definitions, dict):
        raise ValueError("xcaf_instances.json: expected 'definitions' to be an object/dict")
    if not isinstance(occurrences, dict):
        raise ValueError("xcaf_instances.json: expected 'occurrences' to be an object/dict")

    qty_by_def_id = _count_qty_by_def_id(occurrences)

    review_dir = out_dir / REVIEW_SUBDIR
    _ensure_dir(review_dir)

    review_path = review_dir / REVIEW_JSON_NAME
    decisions_path = review_dir / DECISIONS_JSON_NAME

    review_rows: List[Dict[str, str]] = []
    candidates: List[Dict[str, Any]] = []

    # Stage 1: detect candidates (has_shape and solid_count>1)
    for def_id, defn in definitions.items():
        if not isinstance(def_id, str) or not def_id:
            continue
        if not isinstance(defn, dict):
            continue

        has_shape = bool(defn.get("has_shape"))
        if not has_shape:
            continue

        solid_count = _safe_int(defn.get("solid_count"), default=0)
        if solid_count <= 1:
            continue

        def_sig = defn.get("def_sig")
        if not isinstance(def_sig, str) or not def_sig:
            # Skip: cannot be stable/deterministic without def_sig
            continue

        candidates.append({"def_id": def_id, "defn": defn, "solid_count": solid_count, "def_sig": def_sig})

    # Stage 2/3: bucket + write review rows
    for c in candidates:
        def_id = c["def_id"]
        defn = c["defn"]
        solid_count = int(c["solid_count"])
        def_sig = c["def_sig"]

        name = _best_name(defn)
        qty_total = _safe_int(defn.get("qty_total"), default=qty_by_def_id.get(def_id, 0))
        bbox = _format_bbox(defn)

        bucket_res = bucket_candidate(name=name, qty_total=qty_total, solid_count=solid_count)

        review_rows.append(
            {
                "def_sig": def_sig,
                "def_sig_free": str(defn.get("def_sig_free") or ""),
                "def_id": def_id,
                "name": name,
                "qty_total": qty_total,
                "solid_count": solid_count,
                "bbox": bbox,
                "bucket": bucket_res.bucket,
                "reason": bucket_res.reason,
            }
        )

    # Deterministic ordering (stable review list)
    def _sort_key(r: Dict[str, Any]) -> Tuple[int, int, int, str, str]:
        bucket = r.get("bucket", "review")
        b_ord = BUCKET_ORDER.get(bucket, 9)
        sc = _safe_int(r.get("solid_count"), 0)
        qt = _safe_int(r.get("qty_total"), 0)
        nm = (r.get("name") or "").lower()
        sig = r.get("def_sig") or ""
        return (b_ord, -sc, -qt, nm, sig)


    review_rows.sort(key=_sort_key)

    meta = {
        "input_xcaf": str(xcaf_path.name),
        "note": "Stage 1-3 only (no exploding). Deterministic ordering by bucket/solid_count/qty/name/def_sig.",
    }
    _write_review(review_path, review_rows, meta=meta)

    # Decisions: seed/update JSON with existing preserved (keyed by def_sig)
    existing = _read_existing_decisions(decisions_path)  # sig -> (decision, note, updated_utc)
    now = _now_utc_iso()

    decisions_obj: Dict[str, Dict[str, str]] = {}
    seen: set[str] = set()

    for r in review_rows:
        sig = (r.get("def_sig") or "").strip()
        if not sig or sig in seen:
            continue
        seen.add(sig)

        if sig in existing:
            decision, note, updated_utc = existing[sig]
            # Preserve previous updated_utc if present; otherwise set now
            if not updated_utc:
                updated_utc = now
        else:
            decision, note, updated_utc = (DECISION_DEFAULT, "", now)

        decisions_obj[sig] = {
            "decision": decision,
            "note": note,
            "updated_utc": updated_utc,
        }

    _write_decisions(decisions_path, decisions_obj)


    print(f"[ok] {len(review_rows)} multi-body candidates")
    print(f"[write] {review_path}")
    print(f"[write] {decisions_path}")
    print(f"[ts]   {_now_utc_iso()}")

    return review_path, decisions_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="xcaf_path", required=True, help="Path to xcaf_instances.json")
    ap.add_argument("--outdir", dest="out_dir", default="out", help="Output directory (default: out)")
    ns = ap.parse_args()

    xcaf_path = Path(ns.xcaf_path)
    out_dir = Path(ns.out_dir)

    if not xcaf_path.exists():
        raise SystemExit(f"Input not found: {xcaf_path}")

    build_multibody_review(xcaf_path=xcaf_path, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
