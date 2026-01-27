#!/usr/bin/env python3
"""
Step 9.2 (in-place): enrich base items in out/assets_manifest_active.json
with bbox/volume from out/xcaf_instances_active.json, then classify.

XCAF active layout observed:
  out/xcaf_instances_active.json
    xcaf_instances.definitions[def_id] = {
      def_sig,
      def_sig_free,
      name,
      bbox.size = [x,y,z] (mm)
      massprops.volume (mm^3)
    }

Stable join:
  assets_manifest.items[*].def_sig_used -> definitions[*].def_sig

Writes back into SAME file:
  - assets_manifest.items[*].bbox_mm / volume_mm3
  - assets_manifest.items[*].category/category_source/category_reason
  - derived.step9.summary.base_by_category + base_geom_hits/misses updated
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


AM_ACTIVE = Path("out/assets_manifest_active.json")
XCAF_ACTIVE = Path("out/xcaf_instances_active.json")

CATEGORIES = ("hardware", "plate", "section", "grating", "assembly", "fabrication", "unknown")

RULES_V1 = {
    # geometry thresholds (mm)
    "hardware_max_L_mm": 80.0,

    "plate_max_t_mm": 25.0,
    "plate_min_m_mm": 80.0,
    "plate_min_L_mm": 120.0,

    "section_min_L_mm": 800.0,
    "section_min_m_mm": 40.0,

    "grating_max_t_mm": 50.0,
    "grating_min_m_mm": 150.0,
    "grating_min_L_mm": 200.0,
    "grating_max_fill_ratio": 0.35,

    # name tokens
    "assembly_name_tokens": ["assembly", "assy", "weldment", "frame", "subframe", "module"],
    "grating_name_tokens": ["grating", "grate", "mesh", "floorgrating", "floor-grating", "grid"],
    "hardware_name_tokens": ["bolt", "bolts", "nut", "nuts", "washer", "washers", "stud", "studs", "anchor", "anchors", "screw", "screws"],
    "plate_name_tokens": ["plate", "pl"],
    "section_name_tokens": ["ub", "uc", "pfc", "rhs", "shs", "chs", "ipe", "hea", "heb", "angle", "ea", "ua", "channel", "beam", "column"],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _safe_floats3(v: Any) -> Tuple[float, float, float]:
    if isinstance(v, list) and len(v) == 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            return (0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def _bbox_sorted(b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    xs = sorted((abs(b[0]), abs(b[1]), abs(b[2])))
    return (xs[0], xs[1], xs[2])  # t, m, L


def _fill_ratio(volume_mm3: float, t: float, m: float, L: float) -> float:
    denom = t * m * L
    if denom <= 0.0:
        return 1.0
    r = volume_mm3 / denom
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _has_any_token(tokens: List[str], needles: List[str]) -> bool:
    s = set(tokens)
    for n in needles:
        if n.lower() in s:
            return True
    return False


def _best_name_from_item(it: Dict[str, Any]) -> str:
    for k in ("display_name", "name", "ref_name", "part_name", "product_name"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _classify_name_geom(name_hint: str, bbox_mm: Tuple[float, float, float], volume_mm3: float) -> Tuple[str, str]:
    tokens = _tokenize(name_hint)
    t, m, L = _bbox_sorted(bbox_mm)
    fr = _fill_ratio(volume_mm3, t, m, L)

    # name-driven
    if _has_any_token(tokens, RULES_V1["hardware_name_tokens"]):
        return ("hardware", "name_token_hardware")
    if _has_any_token(tokens, RULES_V1["grating_name_tokens"]):
        return ("grating", "name_token_grating")
    if _has_any_token(tokens, RULES_V1["assembly_name_tokens"]):
        return ("assembly", "name_token_assembly")

    # geometry-driven
    if L > 0.0 and L <= RULES_V1["hardware_max_L_mm"]:
        return ("hardware", "geom_small_envelope")

    if (
        t > 0.0 and t <= RULES_V1["grating_max_t_mm"]
        and m >= RULES_V1["grating_min_m_mm"]
        and L >= RULES_V1["grating_min_L_mm"]
        and fr <= RULES_V1["grating_max_fill_ratio"]
    ):
        return ("grating", "geom_low_fill_ratio")

    if (
        t > 0.0 and t <= RULES_V1["plate_max_t_mm"]
        and m >= RULES_V1["plate_min_m_mm"]
        and L >= RULES_V1["plate_min_L_mm"]
    ):
        return ("plate", "geom_plate")

    if L >= RULES_V1["section_min_L_mm"] and m >= RULES_V1["section_min_m_mm"]:
        return ("section", "geom_section")

    if L > 0.0:
        return ("fabrication", "default_fabrication")

    return ("unknown", "missing_bbox")


def _build_by_sig_from_xcaf_active(xcaf_active: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    xi = xcaf_active.get("xcaf_instances", {})
    if not isinstance(xi, dict):
        raise SystemExit("xcaf_instances_active.json missing xcaf_instances{}")
    defs = xi.get("definitions", {})
    if not isinstance(defs, dict):
        raise SystemExit("xcaf_instances_active.json missing xcaf_instances.definitions{}")

    by_sig: Dict[str, Dict[str, Any]] = {}
    for def_id in sorted(defs.keys()):
        d = defs[def_id]
        if not isinstance(d, dict):
            continue
        sig = d.get("def_sig")
        if not isinstance(sig, str) or not sig:
            continue

        bbox = (0.0, 0.0, 0.0)
        b = d.get("bbox")
        if isinstance(b, dict):
            bbox = _safe_floats3(b.get("size"))

        mp = d.get("massprops")
        vol = 0.0
        if isinstance(mp, dict):
            vol = _safe_float(mp.get("volume"), 0.0)

        nm = _safe_str(d.get("name"))
        by_sig[sig] = {"bbox_mm": [bbox[0], bbox[1], bbox[2]], "volume_mm3": vol, "name": nm}

    return by_sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", action="store_true")
    ns = ap.parse_args()

    if not AM_ACTIVE.is_file():
        raise SystemExit(f"Missing: {AM_ACTIVE}")
    if not XCAF_ACTIVE.is_file():
        raise SystemExit(f"Missing: {XCAF_ACTIVE}")

    if ns.backup:
        bak = Path(str(AM_ACTIVE) + ".bak_step9_2")
        bak.write_text(AM_ACTIVE.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[step9.2] backup: {bak}")

    doc = _read_json(AM_ACTIVE)
    am = doc.get("assets_manifest")
    if not isinstance(am, dict):
        raise SystemExit("assets_manifest_active missing assets_manifest{}")
    items = am.get("items")
    if not isinstance(items, list):
        raise SystemExit("assets_manifest_active missing assets_manifest.items[]")

    derived = doc.get("derived")
    if not isinstance(derived, dict):
        derived = {}
        doc["derived"] = derived

    xcaf_active = _read_json(XCAF_ACTIVE)
    by_sig = _build_by_sig_from_xcaf_active(xcaf_active)

    counts = {c: 0 for c in CATEGORIES}
    hits = 0
    misses = 0

    # deterministic pass over list order (we do not reorder items[])
    for it in items:
        if not isinstance(it, dict):
            continue

        sig_used = _safe_str(it.get("def_sig_used") or it.get("def_sig") or it.get("def_sig_free")).strip()
        if not sig_used:
            it["category"] = "unknown"
            it["category_source"] = "rules_v1+xcaf_active"
            it["category_reason"] = "missing_def_sig_used"
            counts["unknown"] += 1
            misses += 1
            continue

        info = by_sig.get(sig_used)
        if info is None:
            it["category"] = "fabrication"
            it["category_source"] = "rules_v1+xcaf_active"
            it["category_reason"] = "def_sig_not_found_default_fabrication"
            counts["fabrication"] += 1
            misses += 1
            continue

        bbox = _safe_floats3(info.get("bbox_mm"))
        vol = _safe_float(info.get("volume_mm3"), 0.0)
        it["bbox_mm"] = [bbox[0], bbox[1], bbox[2]]
        it["volume_mm3"] = vol
        # Prefer existing stable UI label; fall back to XCAF definition name.
        if not isinstance(it.get("display_name"), str) or not it.get("display_name").strip():
            nm = _safe_str(info.get("name"))
            if nm:
                it["display_name"] = nm


        name_hint = _best_name_from_item(it) or _safe_str(info.get("name"))
        cat, reason = _classify_name_geom(name_hint, bbox, vol)

        it["category"] = cat
        it["category_source"] = "rules_v1+xcaf_active"
        it["category_reason"] = reason
        counts[cat] += 1
        hits += 1

    # Ensure derived.step9 exists and update summary
    step9 = derived.get("step9")
    if not isinstance(step9, dict):
        step9 = {}
        derived["step9"] = step9
    summary = step9.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        step9["summary"] = summary

    summary["base_by_category"] = counts
    summary["base_geom_source"] = str(XCAF_ACTIVE.as_posix())
    summary["base_geom_hits"] = hits
    summary["base_geom_misses"] = misses
    summary["base_geom_updated_utc"] = utc_now_iso()

    _write_json(AM_ACTIVE, doc)

    print(f"[step9.2] wrote: {AM_ACTIVE}")
    print(f"[step9.2] hits={hits} misses={misses} base_by_category={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
