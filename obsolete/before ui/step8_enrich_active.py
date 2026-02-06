#!/usr/bin/env python3
"""
Step 8 (in-place): Enrich out/assets_manifest_active.json with grouping + UI tables.

- Base parts: assets_manifest.items[]
  group: chirality_sig_free
  split: chirality_sig
  keep: def_sig_used (stable link to Step 1)

- Exploded subparts: derived.subpart_definitions{} keyed by subpart_sig
  group: chirality_sig_free
  split: chirality_sig
  stable id: subpart_sig (and/or subpart_sig_free)
  provenance: where_used_parents[]

Writes back into the SAME active JSON by adding:
  derived.step8 = { base_groups, subpart_groups, tables, created_utc, input }

Deterministic: stable sorting, stable JSON output, no randomness.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_ACTIVE = "out/assets_manifest_active.json"
BACKUP_SUFFIX = ".bak_step8"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    # Stable JSON for deterministic diffs
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def safe_lower(v: Any) -> str:
    return safe_str(v).strip().lower()


def sig8(sig: str) -> str:
    s = safe_str(sig).strip()
    return s[:8] if len(s) >= 8 else s


def best_display_name_base(it: Dict[str, Any], fallback_sig: str) -> str:
    """Prefer stable UI label; fall back to legacy fields; final fallback is Part <sig8>."""
    for k in ("display_name", "name", "ref_name", "part_name", "product_name", "label"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"Part {sig8(fallback_sig) or 'unknown'}"


def best_display_name_subpart(rec: Dict[str, Any], fallback_sig: str) -> str:
    """Prefer stable UI label; fall back; final fallback is Subpart <sig8>."""
    for k in ("display_name", "name", "subpart_name", "label"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"Subpart {sig8(fallback_sig) or 'unknown'}"


def norm_group_variant(ch_free: str, ch: str, fallback: str) -> Tuple[str, str]:
    """
    Mirror-safe rule:
      primary group: chirality_sig_free
      split: chirality_sig
    Fallback (still stable):
      group = fallback
      variant = fallback
    """
    if ch_free and ch:
        return (ch_free, ch)
    return (fallback, fallback)


def ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = parent.get(key)
    if isinstance(v, dict):
        return v
    parent[key] = {}
    return parent[key]


def build_base_rollups(
    doc: Dict[str, Any],
    annotate: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    am = doc.get("assets_manifest", {})
    items = am.get("items", [])
    if not isinstance(items, list):
        items = []

    # group_key -> variant_key -> accumulator
    acc: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for it in items:
        if not isinstance(it, dict):
            continue

        ch_free = safe_str(it.get("chirality_sig_free"))
        ch = safe_str(it.get("chirality_sig"))
        def_sig_used = safe_str(it.get("def_sig_used") or it.get("def_sig") or it.get("def_sig_free"))
        fallback = def_sig_used or safe_str(it.get("part_id")) or "UNKNOWN"

        disp = best_display_name_base(it, fallback_sig=(def_sig_used or fallback))
        cat = safe_lower(it.get("category"))

        bbox_mm = it.get("bbox_mm")
        if not (isinstance(bbox_mm, list) and len(bbox_mm) == 3):
            bbox_mm = []

        vol_mm3 = it.get("volume_mm3")
        if not isinstance(vol_mm3, (int, float)):
            vol_mm3 = 0.0

        gk, vk = norm_group_variant(ch_free, ch, fallback)

        if annotate:
            it["group_key"] = gk
            it["variant_key"] = vk

        if gk not in acc:
            acc[gk] = {}
        if vk not in acc[gk]:
            acc[gk][vk] = {
                "def_sig_used": def_sig_used,
                "display_name": "",
                "category": "",
                "bbox_mm": [],
                "volume_mm3": 0.0,
                "count": 0,
                "stl_examples": [],
            }

        # Keep first non-empty representative fields (deterministic given stable traversal order)
        if not acc[gk][vk].get("display_name"):
            acc[gk][vk]["display_name"] = disp
        if not acc[gk][vk].get("category"):
            acc[gk][vk]["category"] = cat
        if not acc[gk][vk].get("bbox_mm") and bbox_mm:
            acc[gk][vk]["bbox_mm"] = bbox_mm
        if not acc[gk][vk].get("volume_mm3") and vol_mm3:
            acc[gk][vk]["volume_mm3"] = float(vol_mm3)

        acc[gk][vk]["count"] += 1

        stl = safe_str(it.get("stl_path"))
        if stl and len(acc[gk][vk]["stl_examples"]) < 3:
            acc[gk][vk]["stl_examples"].append(stl)

    # Emit deterministic rollups + UI tables
    rollup: Dict[str, Any] = {"by_group_key": {}}
    tbl_groups: List[Dict[str, Any]] = []
    tbl_vars: List[Dict[str, Any]] = []

    for gk in sorted(acc.keys()):
        rollup["by_group_key"][gk] = {"variants": {}}
        total = 0
        nvars = 0

        for vk in sorted(acc[gk].keys()):
            rec = acc[gk][vk]
            total += safe_int(rec["count"], 0)
            nvars += 1

            rollup["by_group_key"][gk]["variants"][vk] = {
                "def_sig_used": rec["def_sig_used"],
                "display_name": safe_str(rec.get("display_name")),
                "category": safe_str(rec.get("category")),
                "bbox_mm": rec.get("bbox_mm", []),
                "volume_mm3": rec.get("volume_mm3", 0.0),
                "count": rec["count"],
                "stl_examples": sorted(rec["stl_examples"]),
            }

            tbl_vars.append(
                {
                    "group_key": gk,
                    "variant_key": vk,
                    "def_sig_used": rec["def_sig_used"],
                    "display_name": safe_str(rec.get("display_name")),
                    "category": safe_str(rec.get("category")),
                    "bbox_mm": rec.get("bbox_mm", []),
                    "volume_mm3": rec.get("volume_mm3", 0.0),
                    "count": rec["count"],
                    "stl_example": (sorted(rec["stl_examples"])[0] if rec["stl_examples"] else ""),
                }
            )

        # Representative fields from first variant (deterministic: sorted variant keys)
        rep_vk = sorted(acc[gk].keys())[0] if acc[gk] else ""
        rep = acc[gk].get(rep_vk, {}) if rep_vk else {}

        tbl_groups.append(
            {
                "group_key": gk,
                "display_name": safe_str(rep.get("display_name")),
                "category": safe_str(rep.get("category")),
                "count_total": total,
                "n_variants": nvars,
            }
        )

    return rollup, tbl_groups, tbl_vars


def build_subpart_rollups(
    doc: Dict[str, Any],
    annotate: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    derived = doc.get("derived", {})
    sdefs = derived.get("subpart_definitions", {})
    if not isinstance(sdefs, dict):
        sdefs = {}

    # group_key -> variant_key -> list of subparts (usually 1, but keep list for safety)
    acc: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    tbl_groups: List[Dict[str, Any]] = []
    tbl_vars: List[Dict[str, Any]] = []
    tbl_where: List[Dict[str, Any]] = []

    for subpart_sig in sorted(sdefs.keys()):
        rec = sdefs[subpart_sig]
        if not isinstance(rec, dict):
            continue

        ch_free = safe_str(rec.get("chirality_sig_free"))
        ch = safe_str(rec.get("chirality_sig"))
        sub_sig_free = safe_str(rec.get("subpart_sig_free"))
        fallback = sub_sig_free or subpart_sig

        disp = best_display_name_subpart(rec, fallback_sig=subpart_sig)
        cat = safe_lower(rec.get("category"))

        bbox_mm = rec.get("bbox_mm")
        if not (isinstance(bbox_mm, list) and len(bbox_mm) == 3):
            bbox_mm = []

        vol_mm3 = rec.get("volume_mm3")
        if not isinstance(vol_mm3, (int, float)):
            vol_mm3 = 0.0

        gk, vk = norm_group_variant(ch_free, ch, fallback)

        if annotate:
            rec["group_key"] = gk
            rec["variant_key"] = vk

        qty_total = safe_int(rec.get("qty_total"), 0)
        rep_stl = safe_str(rec.get("representative_stl_path") or rec.get("rep_stl") or rec.get("stl_path"))

        where_used = rec.get("where_used_parents", [])
        if not isinstance(where_used, list):
            where_used = []

        if gk not in acc:
            acc[gk] = {}
        if vk not in acc[gk]:
            acc[gk][vk] = []

        acc[gk][vk].append(
            {
                "subpart_sig": subpart_sig,
                "display_name": disp,
                "category": cat,
                "bbox_mm": bbox_mm,
                "volume_mm3": float(vol_mm3),
                "qty_total": qty_total,
                "representative_stl_path": rep_stl,
                "where_used_parents": where_used,
                "chirality_algo": safe_str(rec.get("chirality_algo")),
            }
        )

        for w in where_used:
            if not isinstance(w, dict):
                continue
            tbl_where.append(
                {
                    "group_key": gk,
                    "variant_key": vk,
                    "subpart_sig": subpart_sig,
                    "parent_def_sig": safe_str(w.get("parent_def_sig")),
                    "parent_name": safe_str(w.get("parent_name") or ""),
                    "parent_qty_total": safe_int(w.get("parent_qty_total"), 0),
                    "qty_per_parent": safe_int(w.get("qty_per_parent"), 0),
                    "qty_total": safe_int(w.get("qty_total"), 0),
                }
            )

    rollup: Dict[str, Any] = {"by_group_key": {}}

    for gk in sorted(acc.keys()):
        rollup["by_group_key"][gk] = {"variants": {}}
        group_qty = 0
        nvars = 0

        for vk in sorted(acc[gk].keys()):
            subparts = sorted(
                acc[gk][vk],
                key=lambda r: (safe_str(r.get("subpart_sig")), safe_str(r.get("representative_stl_path"))),
            )
            v_qty = sum(safe_int(s.get("qty_total"), 0) for s in subparts)
            group_qty += v_qty
            nvars += 1

            rep = subparts[0] if subparts else {}

            rollup["by_group_key"][gk]["variants"][vk] = {
                "subpart_sig": safe_str(rep.get("subpart_sig")),
                "display_name": safe_str(rep.get("display_name")),
                "category": safe_str(rep.get("category")),
                "bbox_mm": rep.get("bbox_mm", []),
                "volume_mm3": rep.get("volume_mm3", 0.0),
                "qty_total": v_qty,
                "representative_stl_path": safe_str(rep.get("representative_stl_path")),
                "chirality_algo": safe_str(rep.get("chirality_algo")),
                "where_used_parents": rep.get("where_used_parents", []),
            }

            tbl_vars.append(
                {
                    "group_key": gk,
                    "variant_key": vk,
                    "subpart_sig": safe_str(rep.get("subpart_sig")),
                    "display_name": safe_str(rep.get("display_name")),
                    "category": safe_str(rep.get("category")),
                    "bbox_mm": rep.get("bbox_mm", []),
                    "volume_mm3": rep.get("volume_mm3", 0.0),
                    "qty_total": v_qty,
                    "representative_stl_path": safe_str(rep.get("representative_stl_path")),
                }
            )

        rep_vk = sorted(acc[gk].keys())[0] if acc[gk] else ""
        rep = (
            sorted(acc[gk].get(rep_vk, []), key=lambda r: safe_str(r.get("subpart_sig")))[0]
            if rep_vk and acc[gk].get(rep_vk)
            else {}
        )

        tbl_groups.append(
            {
                "group_key": gk,
                "display_name": safe_str(rep.get("display_name")),
                "category": safe_str(rep.get("category")),
                "qty_total": group_qty,
                "n_variants": nvars,
            }
        )

    tbl_where.sort(key=lambda r: (r["group_key"], r["variant_key"], r["subpart_sig"], r["parent_def_sig"]))
    return rollup, tbl_groups, tbl_vars, tbl_where


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Repo root")
    ap.add_argument("--active", default="", help="Active manifest path (default out/assets_manifest_active.json)")
    ap.add_argument("--inplace", action="store_true", help="Write changes back into active JSON (default: true)")
    ap.add_argument("--backup", action="store_true", help="Write a .bak_step8 backup first")
    ap.add_argument("--annotate", action="store_true", help="Annotate items/subparts with group_key/variant_key")
    ns = ap.parse_args()

    repo = Path(ns.repo).resolve()
    active_path = Path(ns.active).resolve() if ns.active else (repo / DEFAULT_ACTIVE)

    if not active_path.is_file():
        raise SystemExit(f"Active JSON not found: {active_path}")

    doc = read_json(active_path)

    base_rollup, base_tbl_groups, base_tbl_vars = build_base_rollups(doc, annotate=bool(ns.annotate))
    sub_rollup, sub_tbl_groups, sub_tbl_vars, sub_tbl_where = build_subpart_rollups(doc, annotate=bool(ns.annotate))

    derived = ensure_dict(doc, "derived")
    derived["step8"] = {
        "created_utc": utc_now_iso(),
        "input": str(active_path.relative_to(repo).as_posix()),
        "base_groups": base_rollup,
        "subpart_groups": sub_rollup,
        "tables": {
            "base_groups": base_tbl_groups,
            "base_variants": base_tbl_vars,
            "subpart_groups": sub_tbl_groups,
            "subpart_variants": sub_tbl_vars,
            "subpart_where_used": sub_tbl_where,
        },
    }

    if ns.backup:
        backup_path = Path(str(active_path) + BACKUP_SUFFIX)
        backup_path.write_text(active_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[step8] backup: {backup_path}")

    write_json(active_path, doc)
    print(f"[step8] enriched active: {active_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
