#!/usr/bin/env python3
"""
Step 9 (in-place): persistent classification + rollup tables in out/assets_manifest_active.json

Writes:
  derived.step9 = {
    created_utc,
    rules_v1,
    summary,
    tables,
    warnings
  }

Also persists per-record category:
  - assets_manifest.items[*].category
  - derived.subpart_definitions[subpart_sig].category

Stable keys only. Deterministic ordering. No unstable joins (no ref_def).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_ACTIVE = "out/assets_manifest_active.json"
BACKUP_SUFFIX = ".bak_step9"

CATEGORIES = ("hardware", "plate", "section", "grating", "assembly", "fabrication", "unknown")

RULES_V1 = {
    # geometry thresholds (mm)
    "hardware_max_L_mm": 80.0,

    "plate_max_t_mm": 25.0,
    "plate_min_m_mm": 80.0,
    "plate_min_L_mm": 120.0,

    "section_min_L_mm": 800.0,
    "section_min_m_mm": 40.0,

    # grating heuristic: thin-ish, big-ish, but low fill ratio
    "grating_max_t_mm": 50.0,
    "grating_min_m_mm": 150.0,
    "grating_min_L_mm": 200.0,
    "grating_max_fill_ratio": 0.35,  # volume / bbox_volume

    # assembly name tokens
    "assembly_name_tokens": ["assembly", "assy", "weldment", "frame", "subframe", "module"],
    "grating_name_tokens": ["grating", "grate", "mesh", "floorgrating", "floor-grating", "grid"],
    "hardware_name_tokens": ["bolt", "bolts", "nut", "nuts", "washer", "washers", "stud", "studs", "anchor", "anchors", "screw", "screws"],
    "plate_name_tokens": ["plate", "pl"],
    "section_name_tokens": ["ub", "uc", "pfc", "rhs", "shs", "chs", "ipe", "hea", "heb", "angle", "ea", "ua", "channel", "beam", "column"],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def safe_floats3(v: Any) -> Tuple[float, float, float]:
    if isinstance(v, list) and len(v) == 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            return (0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def bbox_sorted(b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    xs = sorted((abs(b[0]), abs(b[1]), abs(b[2])))
    return (xs[0], xs[1], xs[2])  # t, m, L


def fill_ratio(volume_mm3: float, t: float, m: float, L: float) -> float:
    denom = t * m * L
    if denom <= 0.0:
        return 1.0
    r = volume_mm3 / denom
    # clamp to sane range for stability
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


def tokenize(*texts: str) -> List[str]:
    toks: List[str] = []
    for t in texts:
        if not t:
            continue
        toks.extend(_TOKEN_RE.findall(t.lower()))
    return toks


def has_any_token(tokens: List[str], needles: List[str]) -> bool:
    s = set(tokens)
    for n in needles:
        if n.lower() in s:
            return True
    return False


def get_name_hint_from_subpart(rec: Dict[str, Any]) -> str:
    # Try a few likely fields
    for k in ("display_name", "name", "subpart_name", "label"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Or mine where-used parents for a hint (if present)
    wups = rec.get("where_used_parents", [])
    if isinstance(wups, list):
        for w in wups:
            if isinstance(w, dict):
                for k in ("parent_name", "name"):
                    v = w.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    return ""


def get_name_hint_from_base(item: Dict[str, Any]) -> str:
    for k in ("display_name", "name", "ref_name", "part_name", "product_name"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def classify_from_name_and_geom(
    name_hint: str,
    bbox_mm: Tuple[float, float, float],
    volume_mm3: float,
    kind: str,
) -> Tuple[str, str]:
    """
    kind = "subpart" or "base"
    Returns (category, reason)
    """
    tokens = tokenize(name_hint)
    t, m, L = bbox_sorted(bbox_mm)
    fr = fill_ratio(volume_mm3, t, m, L)

    # 1) Name-driven overrides (most reliable for special cases)
    if has_any_token(tokens, RULES_V1["hardware_name_tokens"]):
        return ("hardware", "name_token_hardware")

    if has_any_token(tokens, RULES_V1["grating_name_tokens"]):
        return ("grating", "name_token_grating")

    if has_any_token(tokens, RULES_V1["assembly_name_tokens"]) and kind == "base":
        return ("assembly", "name_token_assembly")

    # 2) Geometry-driven rules (deterministic)
    if L > 0.0 and L <= RULES_V1["hardware_max_L_mm"]:
        return ("hardware", "geom_small_envelope")

    # grating: low fill ratio in a big, thin-ish bbox
    if (
        t > 0.0 and t <= RULES_V1["grating_max_t_mm"]
        and m >= RULES_V1["grating_min_m_mm"]
        and L >= RULES_V1["grating_min_L_mm"]
        and fr <= RULES_V1["grating_max_fill_ratio"]
    ):
        return ("grating", "geom_low_fill_ratio")

    # plate: thin, reasonably large, but not too “open”
    if (
        t > 0.0 and t <= RULES_V1["plate_max_t_mm"]
        and m >= RULES_V1["plate_min_m_mm"]
        and L >= RULES_V1["plate_min_L_mm"]
    ):
        return ("plate", "geom_plate")

    # section: long with meaningful depth
    if (
        L >= RULES_V1["section_min_L_mm"]
        and m >= RULES_V1["section_min_m_mm"]
    ):
        return ("section", "geom_section")

    # base-only: if it isn't plate/section/hardware/grating, treat as fabrication unless name says assembly
    if L > 0.0:
        return ("fabrication", "default_fabrication")

    return ("unknown", "missing_bbox")


def ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = parent.get(key)
    if isinstance(v, dict):
        return v
    parent[key] = {}
    return parent[key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--active", default="")
    ap.add_argument("--backup", action="store_true")
    ns = ap.parse_args()

    repo = Path(ns.repo).resolve()
    active = Path(ns.active).resolve() if ns.active else (repo / DEFAULT_ACTIVE)
    if not active.is_file():
        raise SystemExit(f"Missing: {active}")

    if ns.backup:
        bak = Path(str(active) + BACKUP_SUFFIX)
        bak.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[step9] backup: {bak}")

    doc = read_json(active)
    derived = doc.get("derived", {})
    if not isinstance(derived, dict):
        raise SystemExit("active missing derived{}")

    # ---- classify base items (non-exploded) ----
    am = doc.get("assets_manifest", {})
    items = am.get("items", [])
    if not isinstance(items, list):
        items = []

    base_counts: Dict[str, int] = {c: 0 for c in CATEGORIES}

    for it in items:
        if not isinstance(it, dict):
            continue

        # Preserve existing if already set
        existing = safe_str(it.get("category")).strip().lower()
        if existing in CATEGORIES:
            base_counts[existing] += 1
            continue

        name_hint = get_name_hint_from_base(it)
        bbox = safe_floats3(it.get("bbox_mm"))  # may be absent for base items
        vol = safe_float(it.get("volume_mm3"), 0.0)

        cat, reason = classify_from_name_and_geom(name_hint, bbox, vol, kind="base")
        it["category"] = cat
        it["category_source"] = "rules_v1"
        it["category_reason"] = reason
        base_counts[cat] += 1

    # ---- classify exploded subparts (authoritative table) ----
    sdefs = derived.get("subpart_definitions", {})
    if not isinstance(sdefs, dict):
        raise SystemExit("active missing derived.subpart_definitions{}")

    sub_counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
    warnings: List[str] = []

    for subpart_sig in sorted(sdefs.keys()):
        rec = sdefs[subpart_sig]
        if not isinstance(rec, dict):
            continue

        existing = safe_str(rec.get("category")).strip().lower()
        if existing in CATEGORIES:
            sub_counts[existing] += 1
            continue

        name_hint = get_name_hint_from_subpart(rec)
        bbox = safe_floats3(rec.get("bbox_mm"))
        vol = safe_float(rec.get("volume_mm3"), 0.0)

        cat, reason = classify_from_name_and_geom(name_hint, bbox, vol, kind="subpart")
        rec["category"] = cat
        rec["category_source"] = "rules_v1"
        rec["category_reason"] = reason
        sub_counts[cat] += 1

        if cat == "unknown":
            warnings.append(f"unknown subpart_sig={subpart_sig}")

    # ---- rollup UI/procurement tables using Step8 mirror-safe quantities ----
    step8 = derived.get("step8", {})
    if not isinstance(step8, dict):
        raise SystemExit("run step8 first (missing derived.step8)")

    t8 = step8.get("tables", {})
    if not isinstance(t8, dict):
        raise SystemExit("missing derived.step8.tables")

    variants = t8.get("subpart_variants", [])
    where_used = t8.get("subpart_where_used", [])
    if not isinstance(variants, list) or not isinstance(where_used, list):
        raise SystemExit("missing step8 subpart tables")

    # where-used map
    wu_map: Dict[str, List[Dict[str, Any]]] = {}
    for w in where_used:
        if not isinstance(w, dict):
            continue
        sp = safe_str(w.get("subpart_sig"))
        if not sp:
            continue
        wu_map.setdefault(sp, []).append(w)

    for sp in wu_map.keys():
        wu_map[sp].sort(key=lambda r: (safe_str(r.get("parent_def_sig")), safe_int(r.get("qty_total"), 0)))

    tables: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}

    for v in sorted(variants, key=lambda r: (safe_str(r.get("group_key")), safe_str(r.get("variant_key")), safe_str(r.get("subpart_sig")))):
        if not isinstance(v, dict):
            continue
        sp = safe_str(v.get("subpart_sig"))
        if not sp:
            continue

        srec = sdefs.get(sp)
        if not isinstance(srec, dict):
            continue

        cat = safe_str(srec.get("category")).strip().lower()
        if cat not in CATEGORIES:
            cat = "unknown"

        wrows = wu_map.get(sp, [])
        parent_names: List[str] = []
        seen: set[str] = set()
        for w in wrows:
            if not isinstance(w, dict):
                continue
            pn = safe_str(w.get("parent_name")).strip()
            if pn and pn not in seen:
                seen.add(pn)
                parent_names.append(pn)
        parent_names.sort(key=lambda s: s.lower())
        if len(parent_names) <= 3:
            where_used_summary = " | ".join(parent_names)
        else:
            where_used_summary = " | ".join(parent_names[:3]) + f" (+{len(parent_names) - 3})"

        row = {
            "category": cat,
            "group_key": safe_str(v.get("group_key")),
            "variant_key": safe_str(v.get("variant_key")),
            "subpart_sig": sp,
            "qty_total": safe_int(v.get("qty_total"), 0),
            "bbox_mm": srec.get("bbox_mm", []),
            "volume_mm3": srec.get("volume_mm3", 0.0),
            "t_mm": bbox_sorted(safe_floats3(srec.get("bbox_mm")))[0],
            "m_mm": bbox_sorted(safe_floats3(srec.get("bbox_mm")))[1],
            "L_mm": bbox_sorted(safe_floats3(srec.get("bbox_mm")))[2],
            "representative_stl_path": safe_str(
                srec.get("representative_stl_path") or srec.get("rep_stl") or srec.get("stl_path") or v.get("representative_stl_path")
            ),
            "where_used_count": len(wu_map.get(sp, [])),
            "category_source": safe_str(srec.get("category_source")),
            "category_reason": safe_str(srec.get("category_reason")),
            "display_name": safe_str(srec.get("display_name") or v.get("display_name")),
            "where_used_count": len(wrows),
            "where_used_summary": where_used_summary,
        }
        tables[cat].append(row)

    # deterministic sort inside each category
    for c in CATEGORIES:
        tables[c].sort(key=lambda r: (r["group_key"], r["variant_key"], -r["qty_total"], r["subpart_sig"]))

    summary = {
        "base_items": len(items),
        "base_by_category": base_counts,
        "subparts": len(sdefs),
        "subparts_by_category": sub_counts,
        "subpart_variants_total": len(variants),
        "ruleset": "rules_v1",
    }

    derived["step9"] = {
        "created_utc": utc_now_iso(),
        "input": str(active.relative_to(repo).as_posix()),
        "rules_v1": RULES_V1,
        "summary": summary,
        "tables": tables,
        "warnings": warnings[:200],
    }
    doc["derived"] = derived
    doc["assets_manifest"] = am

    write_json(active, doc)

    print(f"[step9] enriched active: {active}")
    print("[step9] summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
