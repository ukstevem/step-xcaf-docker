#!/usr/bin/env python3
"""
Step 8: Grouping / binning tables using assets_manifest_active.json.

Base parts:
  assets_manifest.items[]
  group: chirality_sig_free
  split: chirality_sig
  stable backref: def_sig_used

Exploded subparts:
  derived.subpart_definitions{} (keyed by subpart_sig)
  group: chirality_sig_free
  split: chirality_sig
  stable id: subpart_sig (and/or subpart_sig_free)
  provenance: where_used_parents[]

Never join by unstable ref_def/def_id.
Deterministic ordering everywhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# constants
# -----------------------------

SCHEMA_BASE = "step8_base_groups@1"
SCHEMA_SUBPART = "step8_subpart_groups@1"
SCHEMA_TABLES = "step8_tables@1"

DEFAULT_ACTIVE = "out/assets_manifest_active.json"

# Fallback order if "active" not present
FALLBACK_MANIFESTS = (
    "out/assets_manifest_derived_chiral.json",
    "out/assets_manifest_derived.json",
    "out/assets_manifest.json",
)


# -----------------------------
# helpers
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def pick_first_existing(repo: Path, rels: Tuple[str, ...]) -> Optional[Path]:
    for r in rels:
        p = repo / r
        if p.is_file():
            return p
    return None


def norm_group_variant(ch_free: str, ch: str, fallback_stable: str) -> Tuple[str, str]:
    """
    Enforce mirror rule when chirality present:
      group_key = chirality_sig_free
      variant_key = chirality_sig

    If chirality missing, fall back to stable:
      group_key = fallback_stable (prefer *_free if caller passes it)
      variant_key = fallback_stable
    """
    if ch_free and ch:
        return (ch_free, ch)
    # degrade safely but still stable
    return (fallback_stable, fallback_stable)


# -----------------------------
# builders
# -----------------------------

def build_base_groups(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      json_out,
      table_groups rows,
      table_variants rows
    """
    items = doc.get("assets_manifest", {}).get("items", [])
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

        # stable fallback if chirality absent
        fallback = def_sig_used or safe_str(it.get("part_id")) or "UNKNOWN"

        gk, vk = norm_group_variant(ch_free, ch, fallback)

        if gk not in acc:
            acc[gk] = {}
        if vk not in acc[gk]:
            acc[gk][vk] = {
                "def_sig_used": def_sig_used,
                "count": 0,
                "stl_examples": [],
            }

        acc[gk][vk]["count"] += 1

        stl = safe_str(it.get("stl_path"))
        if stl and len(acc[gk][vk]["stl_examples"]) < 3:
            acc[gk][vk]["stl_examples"].append(stl)

    # deterministic emit
    groups_out: List[Dict[str, Any]] = []
    tbl_groups: List[Dict[str, Any]] = []
    tbl_vars: List[Dict[str, Any]] = []

    for gk in sorted(acc.keys()):
        variants = []
        total = 0
        for vk in sorted(acc[gk].keys()):
            rec = acc[gk][vk]
            total += safe_int(rec["count"], 0)
            variants.append({
                "variant_key": vk,
                "def_sig_used": rec["def_sig_used"],
                "count": rec["count"],
                "stl_examples": sorted(rec["stl_examples"]),
            })
            tbl_vars.append({
                "group_key": gk,
                "variant_key": vk,
                "def_sig_used": rec["def_sig_used"],
                "count": rec["count"],
                "stl_example": (sorted(rec["stl_examples"])[0] if rec["stl_examples"] else ""),
            })

        groups_out.append({
            "group_key": gk,
            "variants": variants
        })

        tbl_groups.append({
            "group_key": gk,
            "count_total": total,
            "n_variants": len(variants),
        })

    out = {
        "schema": SCHEMA_BASE,
        "created_utc": utc_now_iso(),
        "groups": groups_out
    }
    return out, tbl_groups, tbl_vars


def build_subpart_groups(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Uses:
      derived.subpart_definitions{} keyed by subpart_sig
    Returns:
      json_out,
      table_groups rows,
      table_variants rows,
      table_where_used rows
    """
    derived = doc.get("derived", {})
    sdefs = derived.get("subpart_definitions", {})
    if not isinstance(sdefs, dict):
        sdefs = {}

    # group_key -> variant_key -> list of subparts
    acc: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # where-used table rows
    tbl_where: List[Dict[str, Any]] = []

    for subpart_sig in sorted(sdefs.keys()):
        rec = sdefs[subpart_sig]
        if not isinstance(rec, dict):
            continue

        ch_free = safe_str(rec.get("chirality_sig_free"))
        ch = safe_str(rec.get("chirality_sig"))
        sub_sig_free = safe_str(rec.get("subpart_sig_free"))
        rep_stl = safe_str(rec.get("representative_stl_path") or rec.get("rep_stl") or rec.get("stl_path"))

        # stable fallback if chirality absent
        fallback = sub_sig_free or subpart_sig

        gk, vk = norm_group_variant(ch_free, ch, fallback)

        if gk not in acc:
            acc[gk] = {}
        if vk not in acc[gk]:
            acc[gk][vk] = []

        qty_total = safe_int(rec.get("qty_total"), 0)
        where_used = rec.get("where_used_parents", [])
        if not isinstance(where_used, list):
            where_used = []

        acc[gk][vk].append({
            "subpart_sig": subpart_sig,
            "qty_total": qty_total,
            "representative_stl_path": rep_stl,
            "where_used_parents": where_used,
            "chirality_algo": safe_str(rec.get("chirality_algo")),
        })

        # flatten where-used (deterministic order)
        for w in where_used:
            if not isinstance(w, dict):
                continue
            tbl_where.append({
                "group_key": gk,
                "variant_key": vk,
                "subpart_sig": subpart_sig,
                "parent_def_sig": safe_str(w.get("parent_def_sig")),
                "parent_qty_total": safe_int(w.get("parent_qty_total"), 0),
                "qty_per_parent": safe_int(w.get("qty_per_parent"), 0),
                "qty_total": safe_int(w.get("qty_total"), 0),
            })

    # deterministic emit
    groups_out: List[Dict[str, Any]] = []
    tbl_groups: List[Dict[str, Any]] = []
    tbl_vars: List[Dict[str, Any]] = []

    for gk in sorted(acc.keys()):
        variants_out: List[Dict[str, Any]] = []
        group_qty = 0

        for vk in sorted(acc[gk].keys()):
            # there should usually be 1 subpart_sig per (gk,vk), but keep list to be safe
            subparts = sorted(
                acc[gk][vk],
                key=lambda r: (safe_str(r.get("subpart_sig")), safe_str(r.get("representative_stl_path")))
            )

            # aggregate variant qty deterministically
            v_qty = sum(safe_int(s.get("qty_total"), 0) for s in subparts)
            group_qty += v_qty

            # pick representative record (first)
            rep = subparts[0] if subparts else {}
            variants_out.append({
                "variant_key": vk,
                "subpart_sig": safe_str(rep.get("subpart_sig")),
                "qty_total": v_qty,
                "representative_stl_path": safe_str(rep.get("representative_stl_path")),
                "chirality_algo": safe_str(rep.get("chirality_algo")),
                "where_used_parents": rep.get("where_used_parents", []),
            })

            tbl_vars.append({
                "group_key": gk,
                "variant_key": vk,
                "subpart_sig": safe_str(rep.get("subpart_sig")),
                "qty_total": v_qty,
                "representative_stl_path": safe_str(rep.get("representative_stl_path")),
            })

        groups_out.append({
            "group_key": gk,
            "variants": variants_out
        })

        tbl_groups.append({
            "group_key": gk,
            "qty_total": group_qty,
            "n_variants": len(variants_out),
        })

    # stable sort where-used rows
    tbl_where.sort(key=lambda r: (r["group_key"], r["variant_key"], r["subpart_sig"], r["parent_def_sig"]))

    out = {
        "schema": SCHEMA_SUBPART,
        "created_utc": utc_now_iso(),
        "groups": groups_out
    }
    return out, tbl_groups, tbl_vars, tbl_where


# -----------------------------
# main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Repo root containing out/")
    ap.add_argument("--active", default="", help="Explicit active manifest path (default out/assets_manifest_active.json)")
    ap.add_argument("--outdir", default="out/step8", help="Output directory under repo")
    ns = ap.parse_args()

    repo = Path(ns.repo).resolve()
    outdir = (repo / ns.outdir).resolve()

    active_path = Path(ns.active).resolve() if ns.active else (repo / DEFAULT_ACTIVE)
    if not active_path.is_file():
        fb = pick_first_existing(repo, FALLBACK_MANIFESTS)
        if not fb:
            raise SystemExit("No active manifest found, and no fallbacks exist.")
        active_path = fb

    doc = read_json(active_path)

    base_json, base_tbl_groups, base_tbl_vars = build_base_groups(doc)
    sub_json, sub_tbl_groups, sub_tbl_vars, sub_tbl_where = build_subpart_groups(doc)

    # embed input path
    rel_in = str(active_path.relative_to(repo).as_posix())
    base_json["input"] = rel_in
    sub_json["input"] = rel_in

    tables = {
        "schema": SCHEMA_TABLES,
        "created_utc": utc_now_iso(),
        "input": rel_in,
        "tables": {
            "base_groups": base_tbl_groups,
            "base_variants": base_tbl_vars,
            "subpart_groups": sub_tbl_groups,
            "subpart_variants": sub_tbl_vars,
            "subpart_where_used": sub_tbl_where,
        }
    }

    write_json(outdir / "step8_base_groups.json", base_json)
    write_json(outdir / "step8_subpart_groups.json", sub_json)
    write_json(outdir / "step8_tables.json", tables)

    print("[step8] input :", rel_in)
    print("[step8] wrote : out/step8/step8_base_groups.json")
    print("[step8] wrote : out/step8/step8_subpart_groups.json")
    print("[step8] wrote : out/step8/step8_tables.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
