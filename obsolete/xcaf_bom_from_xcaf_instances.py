#!/usr/bin/env python3
"""
xcaf_bom_from_xcaf_instances.py

Creates BOM CSVs from xcaf_instances.json (per pss xcaf_instances.schema.json).

Outputs:
  outdir/bom_by_name.csv
    Grouped by definition.name (optionally merged by name).
  outdir/bom_where_used.csv
    Leaf-occurrence counts grouped by (child_def_name, parent_def_name).
    Also includes top_parent_name (first-level assembly under root).
  outdir/bom_leaf_occurrences.csv
    Flat list of leaf occurrences with names/parents (debuggable).

Run (Windows host example):
  python .\\xcaf_bom_from_xcaf_instances.py --in .\\out\\xcaf_instances.json --outdir .\\out\\bom
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# IO
# ----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------
# Schema helpers
# ----------------------------

def defn(definitions: Dict[str, Any], def_id: str) -> Optional[Dict[str, Any]]:
    d = definitions.get(def_id)
    return d if isinstance(d, dict) else None


def def_name(definitions: Dict[str, Any], def_id: str) -> str:
    d = defn(definitions, def_id)
    if d:
        nm = d.get("name")
        if isinstance(nm, str) and nm.strip():
            return nm.strip()
    return def_id


def def_has_shape(definitions: Dict[str, Any], def_id: str) -> bool:
    d = defn(definitions, def_id)
    return bool(d.get("has_shape")) if d else False


def build_parent_occ_map(children_by_parent_occ: Dict[str, Any]) -> Dict[str, str]:
    """
    indexes.children_by_parent_occ: { parent_occ_id: [child_occ_id, ...] }
    => invert to { child_occ_id: parent_occ_id }
    """
    out: Dict[str, str] = {}
    if not isinstance(children_by_parent_occ, dict):
        return out
    for parent_occ, kids in children_by_parent_occ.items():
        if not isinstance(kids, list):
            continue
        for kid in kids:
            kid_id = str(kid)
            if kid_id and kid_id not in out:
                out[kid_id] = str(parent_occ)
    return out


def find_top_parent_occ_id(
    root_def: str,
    occ_id: str,
    occurrences: Dict[str, Any],
    parent_occ_of: Dict[str, str],
) -> str:
    """
    Walk up occurrence parents until parent_def == root_def (i.e. we are directly under root).
    Return that occurrence id (the "top parent" assembly for this leaf).
    If we can't walk, return "".
    """
    cur = occ_id
    seen = 0
    while cur and seen < 10000:
        seen += 1
        occ = occurrences.get(cur)
        if not isinstance(occ, dict):
            return ""
        pdef = str(occ.get("parent_def", ""))
        if pdef == root_def:
            return cur
        cur = parent_occ_of.get(cur, "") or str(occ.get("parent_occ") or "")
    return ""


# ----------------------------
# BOM builders
# ----------------------------

def bom_by_name(definitions: Dict[str, Any], merge_same_names: bool) -> List[Dict[str, Any]]:
    """
    If merge_same_names=True:
      group all definitions that share the same name into a single line (sum qty_total).
    If False:
      output one line per def_id (name collisions appear multiple times).
    """
    if not merge_same_names:
        rows: List[Dict[str, Any]] = []
        for def_id, d in definitions.items():
            if not isinstance(d, dict):
                continue
            rows.append(
                {
                    "name": (d.get("name", "") or "").strip() or def_id,
                    "qty_total": int(d.get("qty_total", 0)),
                    "def_id": def_id,
                    "has_shape": bool(d.get("has_shape", False)),
                    "shape_kind": d.get("shape_kind", ""),
                    "solid_count": int(d.get("solid_count", 0)),
                    "def_sig": d.get("def_sig", "") or "",
                }
            )
        rows.sort(key=lambda r: (-int(r["qty_total"]), str(r["name"]).lower(), str(r["def_id"])))
        return rows

    # Merge by name
    agg: Dict[str, Dict[str, Any]] = {}
    for def_id, d in definitions.items():
        if not isinstance(d, dict):
            continue
        nm = (d.get("name", "") or "").strip() or def_id
        qty = int(d.get("qty_total", 0))

        if nm not in agg:
            agg[nm] = {
                "name": nm,
                "qty_total": 0,
                "def_ids": [],
                "has_shape_any": False,
            }
        agg[nm]["qty_total"] += qty
        agg[nm]["def_ids"].append(def_id)
        agg[nm]["has_shape_any"] = bool(agg[nm]["has_shape_any"] or bool(d.get("has_shape", False)))

    rows = []
    for nm, a in agg.items():
        rows.append(
            {
                "name": a["name"],
                "qty_total": a["qty_total"],
                "def_ids": ";".join(a["def_ids"]),
                "has_shape_any": 1 if a["has_shape_any"] else 0,
            }
        )
    rows.sort(key=lambda r: (-int(r["qty_total"]), str(r["name"]).lower()))
    return rows


def leaf_occurrences_flat(
    root_def: str,
    definitions: Dict[str, Any],
    occurrences: Dict[str, Any],
    indexes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    leaf_ids = indexes.get("leaf_occ_ids", [])
    if not isinstance(leaf_ids, list):
        leaf_ids = []

    parent_occ_of = build_parent_occ_map(indexes.get("children_by_parent_occ", {}))

    out: List[Dict[str, Any]] = []
    for occ_id in leaf_ids:
        occ = occurrences.get(occ_id)
        if not isinstance(occ, dict):
            continue

        child_def = str(occ.get("ref_def", ""))
        parent_def = str(occ.get("parent_def", ""))
        occ_name = str(occ.get("name", ""))
        depth = int(occ.get("depth", 0))

        top_parent_occ = find_top_parent_occ_id(root_def, str(occ_id), occurrences, parent_occ_of)
        top_parent_name = ""
        if top_parent_occ:
            top_occ = occurrences.get(top_parent_occ)
            if isinstance(top_occ, dict):
                top_parent_name = str(top_occ.get("name", ""))

        out.append(
            {
                "occ_id": str(occ_id),
                "child_def_id": child_def,
                "child_def_name": def_name(definitions, child_def),
                "parent_def_id": parent_def,
                "parent_def_name": def_name(definitions, parent_def),
                "top_parent_occ_id": top_parent_occ,
                "top_parent_name": top_parent_name,
                "occ_name": occ_name,
                "depth": depth,
            }
        )

    out.sort(key=lambda r: (str(r["top_parent_name"]).lower(), str(r["parent_def_name"]).lower(), str(r["child_def_name"]).lower(), str(r["occ_id"])))
    return out


def bom_where_used(leaf_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by (child_def_id, parent_def_id, top_parent_occ_id).
    qty_leaf counts how many leaf occurrences exist under that parent usage.
    """
    counts: Dict[Tuple[str, str, str], int] = {}
    names: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}

    for r in leaf_rows:
        cd = str(r["child_def_id"])
        pd = str(r["parent_def_id"])
        tp = str(r.get("top_parent_occ_id", ""))
        key = (cd, pd, tp)
        counts[key] = counts.get(key, 0) + 1
        names[key] = (str(r["child_def_name"]), str(r["parent_def_name"]), str(r.get("top_parent_name", "")))

    out: List[Dict[str, Any]] = []
    for (cd, pd, tp), qty in counts.items():
        cn, pn, tpn = names[(cd, pd, tp)]
        out.append(
            {
                "child_name": cn,
                "parent_name": pn,
                "top_parent_name": tpn,
                "qty_leaf": qty,
                "child_def_id": cd,
                "parent_def_id": pd,
                "top_parent_occ_id": tp,
            }
        )

    out.sort(key=lambda r: (-int(r["qty_leaf"]), str(r["child_name"]).lower(), str(r["top_parent_name"]).lower(), str(r["parent_name"]).lower()))
    return out


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Create BOM + where-used from xcaf_instances.json")
    ap.add_argument("--in", dest="in_json", required=True, help="Path to xcaf_instances.json")
    ap.add_argument("--outdir", dest="outdir", required=True, help="Output folder")
    ap.add_argument("--merge-same-names", action="store_true", help="Merge defs with identical name into one BOM line")
    ns = ap.parse_args()

    top = load_json(Path(ns.in_json))
    outdir = Path(ns.outdir)

    root_def = str(top["root_def"])
    definitions = top["definitions"]
    occurrences = top["occurrences"]
    indexes = top["indexes"]

    if not isinstance(definitions, dict) or not isinstance(occurrences, dict) or not isinstance(indexes, dict):
        raise TypeError("Unexpected JSON structure for definitions/occurrences/indexes")

    # BOM by name
    rows_bom = bom_by_name(definitions, merge_same_names=bool(ns.merge_same_names))
    if ns.merge_same_names:
        write_csv(outdir / "bom_by_name.csv", ["name", "qty_total", "def_ids", "has_shape_any"], rows_bom)
    else:
        write_csv(outdir / "bom_by_name.csv", ["name", "qty_total", "def_id", "has_shape", "shape_kind", "solid_count", "def_sig"], rows_bom)

    # Leaf flat + where-used
    leaf_flat = leaf_occurrences_flat(root_def, definitions, occurrences, indexes)
    where_used = bom_where_used(leaf_flat)

    write_csv(
        outdir / "bom_leaf_occurrences.csv",
        ["occ_id", "child_def_id", "child_def_name", "parent_def_id", "parent_def_name", "top_parent_occ_id", "top_parent_name", "occ_name", "depth"],
        leaf_flat,
    )

    write_csv(
        outdir / "bom_where_used.csv",
        ["child_name", "parent_name", "top_parent_name", "qty_leaf", "child_def_id", "parent_def_id", "top_parent_occ_id"],
        where_used,
    )

    print(f"[ok] wrote {outdir / 'bom_by_name.csv'} rows={len(rows_bom)}")
    print(f"[ok] wrote {outdir / 'bom_where_used.csv'} rows={len(where_used)}")
    print(f"[ok] wrote {outdir / 'bom_leaf_occurrences.csv'} rows={len(leaf_flat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
