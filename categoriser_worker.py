#!/usr/bin/env python3
"""
categoriser_worker.py  (Step 3.2 - Categorisation)

Purpose
- Categorise BOM items into: hardware / plate / section / pipe / fabrication / bought_out / unknown
- In exploded runs, infer fabrication vs bought_out using explode context
- NEW: If parts_index.json exists (Step 3.1), categorise by *common group* using canonical representative,
  then apply to all members of that group (prevents inconsistent categories for identical components).

Inputs (run_dir)
- bom_global_exploded.json  (preferred if exists)
- bom_global.json           (fallback)
- parts_index.json          (optional; produced by step3_1_worker)

Output (run_dir)
- categories.json
  - preserves manual overrides
  - stores config_hash to force regen on env changes

Notes
- Deterministic behaviour: stable ordering of keys, stable config hash
- Bounded loops:
  - MAX_ROWS prevents pathological BOMs
"""

from __future__ import annotations

import json
import re
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Explicit constants / limits
# -----------------------------

SCHEMA_ID = "categories_v1"
RULESET_ID = "categoriser_v3_grouped"

CATEGORIES = ("hardware", "plate", "section", "pipe", "fabrication", "bought_out", "unknown")

CATEGORIES_JSON = "categories.json"
PARTS_INDEX_JSON = "parts_index.json"

MAX_ROWS = 200000  # hard safety cap for very large models (you said ~12k typical)

# -----------------------------
# Env helpers (bounded)
# -----------------------------

def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        v = float(default)
    else:
        try:
            v = float(raw)
        except Exception:
            v = float(default)
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return float(v)

def _config_hash(cfg: Dict[str, float]) -> str:
    parts = [f"{k}={cfg[k]:.6f}" for k in sorted(cfg.keys())]
    s = "|".join(parts).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]

CFG: Dict[str, float] = {
    "CAT_PLATE_THK_MAX_MM": _env_float("CAT_PLATE_THK_MAX_MM", 30.0, 1.0, 300.0),
    "CAT_PLATE_A_OVER_C_MIN": _env_float("CAT_PLATE_A_OVER_C_MIN", 10.0, 1.0, 200.0),
    "CAT_PLATE_B_OVER_C_MIN": _env_float("CAT_PLATE_B_OVER_C_MIN", 6.0, 1.0, 200.0),

    "CAT_PLATE_THK_MAX_MM_HEAVY": _env_float("CAT_PLATE_THK_MAX_MM_HEAVY", 75.0, 1.0, 500.0),
    "CAT_PLATE_A_OVER_C_MIN_HEAVY": _env_float("CAT_PLATE_A_OVER_C_MIN_HEAVY", 10.0, 1.0, 200.0),
    "CAT_PLATE_B_OVER_C_MIN_HEAVY": _env_float("CAT_PLATE_B_OVER_C_MIN_HEAVY", 10.0, 1.0, 200.0),

    "CAT_SECTION_A_OVER_B_MIN": _env_float("CAT_SECTION_A_OVER_B_MIN", 6.0, 1.0, 200.0),
    "CAT_SECTION_A_OVER_C_MIN": _env_float("CAT_SECTION_A_OVER_C_MIN", 6.0, 1.0, 200.0),
}

CONFIG_HASH = _config_hash(CFG)

PLATE_THK_MAX_MM = CFG["CAT_PLATE_THK_MAX_MM"]
PLATE_A_OVER_C_MIN = CFG["CAT_PLATE_A_OVER_C_MIN"]
PLATE_B_OVER_C_MIN = CFG["CAT_PLATE_B_OVER_C_MIN"]

PLATE_THK_MAX_MM_HEAVY = CFG["CAT_PLATE_THK_MAX_MM_HEAVY"]
PLATE_A_OVER_C_MIN_HEAVY = CFG["CAT_PLATE_A_OVER_C_MIN_HEAVY"]
PLATE_B_OVER_C_MIN_HEAVY = CFG["CAT_PLATE_B_OVER_C_MIN_HEAVY"]

SECTION_A_OVER_B_MIN = CFG["CAT_SECTION_A_OVER_B_MIN"]
SECTION_A_OVER_C_MIN = CFG["CAT_SECTION_A_OVER_C_MIN"]

# Keyword sets (tight)
HW_KEYWORDS = (
    "bolt", "nut", "washer", "screw", "setscrew", "set screw", "stud", "pin", "rivet", "fastener", "anchor", "u-bolt",
)
BOUGHT_OUT_KEYWORDS = (
    "motor", "gearbox", "bearing", "valve", "pump", "sensor", "actuator", "assembly", "assy", "unit",
)
PIPE_KEYWORDS = ("pipe", "tube", "chs")

M_THREAD_RE = re.compile(r"\bm\d+\b", re.IGNORECASE)

# -----------------------------
# IO helpers
# -----------------------------

def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj

def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)

def _load_categories_or_init(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": SCHEMA_ID,
            "ruleset": RULESET_ID,
            "generated_at": _utc_iso_z(),
            "config_hash": CONFIG_HASH,
            "items": {},
        }
    obj = _read_json(path)
    if not isinstance(obj.get("items"), dict):
        obj["items"] = {}
    obj.setdefault("schema", SCHEMA_ID)
    obj.setdefault("ruleset", RULESET_ID)
    if not isinstance(obj.get("generated_at"), str):
        obj["generated_at"] = _utc_iso_z()
    if not isinstance(obj.get("config_hash"), str):
        obj["config_hash"] = ""
    return obj

# -----------------------------
# Keying + bbox
# -----------------------------

def _norm_name(s: str) -> str:
    t = (s or "").strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t

def _sig_key(row: Dict[str, Any]) -> str:
    """
    Deterministic key matching your BOM rows:
      1) ref_def_sig (preferred)
      2) key starting with 'sig:'
      3) def:ref_def_id
      4) name:normalized def_name (last resort)
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

    return "name:" + _norm_name(str(row.get("def_name") or ""))

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
    dims = sorted([abs(x), abs(y), abs(z)], reverse=True)  # a>=b>=c
    if dims[2] <= 1e-9:
        return None
    return (dims[0], dims[1], dims[2])

def _contains_any(name_norm: str, keywords: Tuple[str, ...]) -> bool:
    for kw in keywords:
        if kw in name_norm:
            return True
    return False

def _is_hardware(name_norm: str) -> bool:
    if _contains_any(name_norm, HW_KEYWORDS):
        return True
    return False

def _score_bbox_plate(a: float, b: float, c: float) -> bool:
    # guard
    if c <= 1e-9:
        return False

    # small tolerance to avoid float edge failures (e.g. 150/15 becomes 9.99999988)
    eps = 1e-6

    # precompute ratios once
    a_over_c = a / c
    b_over_c = b / c

    # normal plates
    if c <= PLATE_THK_MAX_MM:
        if (a_over_c + eps) >= PLATE_A_OVER_C_MIN and (b_over_c + eps) >= PLATE_B_OVER_C_MIN:
            return True

    # heavy plates (must still be very "sheet-like")
    if c <= PLATE_THK_MAX_MM_HEAVY:
        if (a_over_c + eps) >= PLATE_A_OVER_C_MIN_HEAVY and (b_over_c + eps) >= PLATE_B_OVER_C_MIN_HEAVY:
            return True

    return False


def _score_bbox_section(a: float, b: float, c: float) -> bool:
    if (a / b) < SECTION_A_OVER_B_MIN:
        return False
    if (a / c) < SECTION_A_OVER_C_MIN:
        return False
    return True

# -----------------------------
# Categorisation core
# -----------------------------

@dataclass(frozen=True)
class CatResult:
    category: str
    confidence: float
    reasons: List[str]

def _categorise_row_auto(
    row: Dict[str, Any],
    exploded_mode: bool,
    exploded_parent_sigs: Optional[set],
    pidx_items: Optional[Dict[str, Any]] = None,   # NEW
) -> CatResult:
    name = str(row.get("def_name") or "")
    n = _norm_name(name)
    solid_count = int(row.get("solid_count") or 0)
    shape_kind = str(row.get("shape_kind") or "")
    is_subpart = bool(row.get("is_exploded_subpart") is True)

    sig = _sig_key(row)

    # Prefer canonical bbox_sorted from parts_index (Step 3.1) to avoid per-row noise
    bbox: Optional[Tuple[float, float, float]] = None
    if isinstance(pidx_items, dict):
        rec = pidx_items.get(sig)
        if isinstance(rec, dict):
            bb = rec.get("bbox_sorted")
            if isinstance(bb, list) and len(bb) == 3:
                try:
                    a = float(bb[0])
                    b = float(bb[1])
                    c = float(bb[2])
                    if c > 1e-9:
                        bbox = (a, b, c)
                except Exception:
                    bbox = None

    # Fallback to row bbox if no canonical bbox available
    if bbox is None:
        bbox = _bbox_size_sorted(row)

    # 1) Hardware by name
    if _is_hardware(n):
        reasons = ["name_hw_keyword"]
        if M_THREAD_RE.search(n):
            reasons.append("thread_marker")
        return CatResult("hardware", 0.92, reasons)

    # exploded-only inference sets
    is_exploded_parent = False
    if exploded_mode and exploded_parent_sigs is not None:
        is_exploded_parent = (sig in exploded_parent_sigs)

    # 2) Exploded parent => fabrication (exploded mode only)
    if exploded_mode and is_exploded_parent:
        reasons = ["exploded_parent:true"]
        if solid_count >= 2:
            reasons.append("multibody:true")
        return CatResult("fabrication", 0.85, reasons)

    # 3) bbox-driven plate/section
    if bbox is not None:
        a, b, c = bbox
        if _score_bbox_plate(a, b, c):
            return CatResult("plate", 0.88, ["bbox_plate"])
        if _contains_any(n, PIPE_KEYWORDS) and _score_bbox_section(a, b, c):
            return CatResult("pipe", 0.75, ["name_pipe_keyword", "bbox_section_like"])
        if _score_bbox_section(a, b, c):
            return CatResult("section", 0.80, ["bbox_section"])

    # 4) Exploded mode: multibody + not exploded + not subpart => bought_out
    if exploded_mode and (not is_subpart) and (solid_count >= 2) and (not is_exploded_parent):
        return CatResult("bought_out", 0.75, ["multibody_not_exploded:true"])

    # 5) Bought-out keyword hint
    if _contains_any(n, BOUGHT_OUT_KEYWORDS):
        return CatResult("bought_out", 0.70, ["name_bought_out_keyword"])

    # 6) Base mode: multibody => fabrication (can't assume bought_out)
    if (not exploded_mode) and (solid_count >= 2):
        return CatResult("fabrication", 0.60, ["multibody:true", "base_mode"])

    # 7) Empty/group rows
    if shape_kind == "EMPTY":
        if _contains_any(n, ("bolt", "nut", "washer", "screw")):
            return CatResult("hardware", 0.70, ["empty_shape", "name_hw_keyword"])
        return CatResult("unknown", 0.40, ["empty_shape"])

    return CatResult("unknown", 0.45, ["no_rule_match"])


# -----------------------------
# parts_index helpers (Step 3.1)
# -----------------------------

def _load_parts_index(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / PARTS_INDEX_JSON
    if not p.exists():
        return None
    try:
        obj = _read_json(p)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

def _build_group_membership_from_bom(
    items: List[Dict[str, Any]],
    pidx: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - sig_key -> group_id
      - group_id -> group_rec (from parts_index['groups'][group_id]) if available

    Preferred mapping:
      parts_index.json["items"][sig_key]["group_id"]

    Fallback:
      group_id = sig_key (no grouping)
    """
    # Collect unique sig keys from BOM (bounded + deterministic)
    sigs: List[str] = []
    for row in items:
        if isinstance(row, dict):
            sigs.append(_sig_key(row))
    sigs = sorted(set(sigs))

    sig_to_group: Dict[str, str] = {}
    group_recs: Dict[str, Dict[str, Any]] = {}

    if not isinstance(pidx, dict):
        for s in sigs:
            sig_to_group[s] = s
        return sig_to_group, group_recs

    groups = pidx.get("groups")
    groups = groups if isinstance(groups, dict) else {}

    pidx_items = pidx.get("items")
    if not isinstance(pidx_items, dict):
        # No usable membership map
        for s in sigs:
            sig_to_group[s] = s
        return sig_to_group, group_recs

    # Map each sig -> group_id from parts_index["items"]
    for s in sigs:
        rec = pidx_items.get(s)
        if isinstance(rec, dict):
            gid = rec.get("group_id")
            if isinstance(gid, str) and gid.strip():
                gid = gid.strip()
                sig_to_group[s] = gid
                grec = groups.get(gid)
                if isinstance(grec, dict):
                    group_recs[gid] = grec
                continue

        # fallback: treat as its own group
        sig_to_group[s] = s

    return sig_to_group, group_recs


# -----------------------------
# Main entry
# -----------------------------

def run_categoriser(run_dir: Path) -> Path:
    bom_exploded = run_dir / "bom_global_exploded.json"
    bom_base = run_dir / "bom_global.json"
    bom_path = bom_exploded if bom_exploded.exists() else bom_base
    if not bom_path.exists():
        raise FileNotFoundError(f"Missing BOM: {bom_path}")

    bom_obj = _read_json(bom_path)
    items = bom_obj.get("items")
    if not isinstance(items, list):
        raise ValueError(f"BOM has no 'items' list: {bom_path}")

    if len(items) > MAX_ROWS:
        raise RuntimeError(f"BOM too large: {len(items)} > MAX_ROWS={MAX_ROWS}")

    # exploded mode detection + parent set
    exploded_mode = False
    exploded_parent_sigs: set = set()
    for row in items:
        if not isinstance(row, dict):
            continue
        if row.get("is_exploded_subpart") is True:
            exploded_mode = True
            p = row.get("from_parent_def_sig")
            if isinstance(p, str) and p.strip():
                exploded_parent_sigs.add(p.strip())

    # load existing categories (preserve manual)
    out_path = run_dir / CATEGORIES_JSON
    cats = _load_categories_or_init(out_path)

    # regenerate metadata
    cats["schema"] = SCHEMA_ID
    cats["ruleset"] = RULESET_ID
    cats["generated_at"] = _utc_iso_z()
    cats["config_hash"] = CONFIG_HASH
    cats["source_bom"] = bom_path.name

    items_out: Dict[str, Any] = cats.get("items", {})
    if not isinstance(items_out, dict):
        items_out = {}

    # Build sig -> representative row mapping (first occurrence is fine, deterministic by scan order)
    sig_to_row: Dict[str, Dict[str, Any]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        s = _sig_key(row)
        if s not in sig_to_row:
            sig_to_row[s] = row

    # Group membership (from parts_index.json if present)
    pidx = _load_parts_index(run_dir)
    sig_to_group, group_recs = _build_group_membership_from_bom(items, pidx)

    # Build group -> members (deterministic)
    group_to_members: Dict[str, List[str]] = {}
    for s, gid in sig_to_group.items():
        group_to_members.setdefault(gid, []).append(s)
    for gid in group_to_members:
        group_to_members[gid] = sorted(group_to_members[gid])

    # Categorise per group using canonical representative if available
    for gid in sorted(group_to_members.keys()):
        members = group_to_members[gid]
        grec = group_recs.get(gid, {})
        rep_sig = None

        cm = grec.get("canonical_member_sig_key") if isinstance(grec, dict) else None
        if isinstance(cm, str) and cm in members:
            rep_sig = cm
        else:
            rep_sig = members[0]

        rep_row = sig_to_row.get(rep_sig)
        if not isinstance(rep_row, dict):
            # should not happen, but keep safe
            auto = CatResult("unknown", 0.40, ["missing_rep_row"])
        else:
            pidx_items = pidx.get("items") if isinstance(pidx, dict) else None
            auto = _categorise_row_auto(rep_row, exploded_mode, exploded_parent_sigs, pidx_items)

            # annotate reasons so you can see grouping effect
            auto = CatResult(auto.category, auto.confidence, list(auto.reasons) + ["grouped:true"])

        auto_obj = {
            "category": auto.category,
            "confidence": float(auto.confidence),
            "reasons": list(auto.reasons),
            "group_id": gid,
            "rep_sig_key": rep_sig,
        }

        # Apply to all members (manual overrides still win per-member)
        for s in members:
            prev = items_out.get(s)
            prev = prev if isinstance(prev, dict) else {}
            manual = prev.get("manual", None)

            if isinstance(manual, dict) and isinstance(manual.get("category"), str) and manual["category"]:
                eff = {"category": manual["category"], "source": "manual"}
            else:
                eff = {"category": auto.category, "source": "auto"}

            items_out[s] = {
                "auto": auto_obj,
                "manual": manual if manual else None,
                "effective": eff,
            }

    cats["items"] = items_out
    _write_json_atomic(out_path, cats)
    return out_path
