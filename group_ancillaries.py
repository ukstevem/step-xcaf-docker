#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def try_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_name(ref_name: str, fallback: str) -> str:
    a = (ref_name or "").strip()
    if a:
        return a
    b = (fallback or "").strip()
    return b if b else ""


def build_assets_map(outp: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns ref_def -> {"ref_name":..., "stl_path":..., "png_path":...}
    from stl_manifest.json if present.
    """
    m: Dict[str, Dict[str, str]] = {}
    man = outp / "stl_manifest.json"
    j = try_read_json(man)
    if not j:
        return m

    # stl_manifest.json is typically a list of dicts
    if isinstance(j, list):
        for r in j:
            if not isinstance(r, dict):
                continue
            ref_def = str(r.get("ref_def") or r.get("ref") or "").strip()
            if not ref_def:
                continue
            m[ref_def] = {
                "ref_name": str(r.get("ref_name") or r.get("name") or "").strip(),
                "stl_path": str(r.get("stl_path") or "").strip(),
                "png_path": str(r.get("png_path") or "").strip(),
            }
    return m


def build_parent_qty_map(outp: Path) -> Dict[str, int]:
    """
    Reads bom_from_xcaf_leaf.csv: ref_def,item_name,qty
    """
    m: Dict[str, int] = {}
    p = outp / "bom_from_xcaf_leaf.csv"
    if not p.exists():
        return m
    for r in read_csv(p):
        ref_def = (r.get("ref_def") or "").strip()
        if not ref_def:
            continue
        try:
            qty = int(float(r.get("qty") or "0"))
        except Exception:
            qty = 0
        m[ref_def] = qty
    return m


def build_def_name_map(outp: Path) -> Dict[str, str]:
    """
    Reads bom_from_xcaf_leaf.csv: ref_def,item_name,qty
    """
    m: Dict[str, str] = {}
    p = outp / "bom_from_xcaf_leaf.csv"
    if not p.exists():
        return m
    for r in read_csv(p):
        ref_def = (r.get("ref_def") or "").strip()
        name = (r.get("item_name") or "").strip()
        if ref_def and name:
            m[ref_def] = name
    return m


def build_ancillary_group_map(outp: Path) -> Dict[str, Dict[str, Any]]:
    """
    Builds sig_id -> group info from ancillary_groups.csv (preferred) or bom_ancillary_grouped.csv.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    p1 = outp / "ancillary_groups.csv"
    if p1.exists():
        for r in read_csv(p1):
            sig_id = (r.get("sig_id") or "").strip()
            if not sig_id:
                continue
            groups[sig_id] = {
                "sig_id": sig_id,
                "sig_free": (r.get("sig_free") or "").strip(),
                "name": (r.get("rep_child_name") or "").strip(),
                "stl_path": (r.get("rep_stl_path") or "").strip(),
                "png_path": (r.get("rep_png_path") or "").strip(),
                "total_qty": _to_int(r.get("total_qty")),
                "dim_a_mm": _to_float(r.get("dim_a_mm")),
                "dim_b_mm": _to_float(r.get("dim_b_mm")),
                "dim_c_mm": _to_float(r.get("dim_c_mm")),
                "fill": _to_float(r.get("fill")),
                "vol_mm3": _to_float(r.get("vol_mm3")),
            }
        return groups

    p2 = outp / "bom_ancillary_grouped.csv"
    if p2.exists():
        for r in read_csv(p2):
            sig_id = (r.get("sig_id") or "").strip()
            if not sig_id:
                continue
            groups[sig_id] = {
                "sig_id": sig_id,
                "sig_free": (r.get("sig_free") or "").strip(),
                "name": (r.get("name") or "").strip(),
                "stl_path": (r.get("stl_path") or "").strip(),
                "png_path": (r.get("png_path") or "").strip(),
                "total_qty": _to_int(r.get("total_qty")),
                "dim_a_mm": _to_float(r.get("dim_a_mm")),
                "dim_b_mm": _to_float(r.get("dim_b_mm")),
                "dim_c_mm": _to_float(r.get("dim_c_mm")),
                "fill": _to_float(r.get("fill")),
                "vol_mm3": _to_float(r.get("vol_mm3")),
            }
    return groups


def _to_int(x: Any) -> int:
    try:
        return int(float(str(x or "0")))
    except Exception:
        return 0


def _to_float(x: Any) -> float:
    try:
        return float(str(x or "0"))
    except Exception:
        return 0.0


def build_ancillary_links(
    outp: Path,
    parent_qty: Dict[str, int],
    group_map: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns parent_ref_def -> [{sig_id, sig_free, qty_total, qty_per_parent, name, stl_path, png_path}]
    """
    links: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    p = outp / "ancillary_parent_map.csv"
    if not p.exists():
        return links

    for r in read_csv(p):
        parent_ref_def = (r.get("parent_ref_def") or "").strip()
        if not parent_ref_def:
            continue

        sig_id = (r.get("sig_id") or "").strip()
        sig_free = (r.get("sig_free") or "").strip()
        qty_total = _to_int(r.get("total_qty"))

        # decorate from group map if possible
        g = group_map.get(sig_id, {})
        name = (r.get("child_name") or "").strip() or str(g.get("name") or "").strip()
        stl_path = (r.get("child_stl_path") or "").strip() or str(g.get("stl_path") or "").strip()
        png_path = (r.get("child_png_path") or "").strip() or str(g.get("png_path") or "").strip()
        classif = g.get("classification", {})

        pq = int(parent_qty.get(parent_ref_def, 0))
        qty_per_parent: Optional[float]
        if pq > 0:
            qty_per_parent = float(qty_total) / float(pq)
        else:
            qty_per_parent = None

        links[parent_ref_def].append(
            {
                "sig_id": sig_id,
                "sig_free": sig_free or str(g.get("sig_free") or "").strip(),
                "qty_total": qty_total,
                "qty_per_parent": qty_per_parent,
                "name": name,
                "classification": classif,
                "stl_path": stl_path,
                "png_path": png_path,
            }
        )

    # sort each list by qty desc
    for k in list(links.keys()):
        links[k].sort(key=lambda x: int(x.get("qty_total", 0)), reverse=True)
    return links

def build_main_class_map(outp: Path) -> Dict[str, Dict[str, Any]]:
    p = outp / "bom_from_xcaf_leaf_classified.csv"
    if not p.exists():
        return {}
    m: Dict[str, Dict[str, Any]] = {}
    for r in read_csv(p):
        ref_def = (r.get("ref_def") or "").strip()
        if not ref_def:
            continue
        m[ref_def] = {
            "class_final": (r.get("class_final") or "").strip(),
            "class_name": (r.get("class_name") or "").strip(),
            "class_geom": (r.get("class_geom") or "").strip(),
            "bbox_x": _to_float(r.get("bbox_x")),
            "bbox_y": _to_float(r.get("bbox_y")),
            "bbox_z": _to_float(r.get("bbox_z")),
        }
    return m


def build_anc_group_class_map(outp: Path) -> Dict[str, Dict[str, Any]]:
    p = outp / "bom_ancillary_grouped_classified.csv"
    if not p.exists():
        return {}
    m: Dict[str, Dict[str, Any]] = {}
    for r in read_csv(p):
        sig_id = (r.get("sig_id") or "").strip()
        if not sig_id:
            continue
        m[sig_id] = {
            "class_final": (r.get("class_final") or "").strip(),
            "class_name": (r.get("class_name") or "").strip(),
            "class_geom": (r.get("class_geom") or "").strip(),
            "bbox_x": _to_float(r.get("bbox_x")),
            "bbox_y": _to_float(r.get("bbox_y")),
            "bbox_z": _to_float(r.get("bbox_z")),
        }
    return m


def main(out_dir: str = "/out") -> int:
    outp = Path(out_dir)
    inst_path = outp / "xcaf_instances.json"
    if not inst_path.exists():
        print(f"Missing: {inst_path}", flush=True)
        return 2

    instances = try_read_json(inst_path)
    if not isinstance(instances, list) or not instances:
        print("xcaf_instances.json is empty or invalid.", flush=True)
        return 2

    assets_by_ref = build_assets_map(outp)
    parent_qty = build_parent_qty_map(outp)
    def_name_map = build_def_name_map(outp)
    group_map = build_ancillary_group_map(outp)
    anc_links = build_ancillary_links(outp, parent_qty, group_map)
    main_class = build_main_class_map(outp)
    anc_class = build_anc_group_class_map(outp)

    # merge ancillary group classification into group_map
    for sig_id, g in group_map.items():
        g["classification"] = anc_class.get(sig_id, {})

    # Build adjacency: parent_def -> [child_occ]
    children_by_parent_def: Dict[str, List[str]] = defaultdict(list)
    occurrences: Dict[str, Dict[str, Any]] = {}
    parent_defs = set()

    for r in instances:
        if not isinstance(r, dict):
            continue
        parent_def = str(r.get("parent_def") or "").strip()
        child_occ = str(r.get("child_occ") or "").strip()
        ref_def = str(r.get("ref_def") or "").strip()

        if parent_def and child_occ:
            children_by_parent_def[parent_def].append(child_occ)
            parent_defs.add(parent_def)

        # store occurrence record by child_occ
        if child_occ:
            ref_name = str(r.get("ref_name") or "").strip()
            occ_name = str(r.get("occ_name") or "").strip()
            # fallback name from bom leaf definitions (if ref_name is missing)
            ref_name = best_name(ref_name, def_name_map.get(ref_def, ""))

            occurrences[child_occ] = {
                "occ_id": child_occ,
                "parent_def": parent_def,
                "ref_def": ref_def,
                "ref_name": ref_name,
                "occ_name": occ_name,
                "depth": int(r.get("depth") or 0),
                "has_ref": bool(r.get("has_ref")),
                "m_local": r.get("m_local"),
                "m_global": r.get("m_global"),
            }

    # Identify root_def as the single parent_def used at depth 0, if possible
    root_def = None
    depth0 = [r for r in instances if isinstance(r, dict) and int(r.get("depth") or 0) == 0]
    roots = sorted({str(r.get("parent_def") or "").strip() for r in depth0 if str(r.get("parent_def") or "").strip()})
    if len(roots) == 1:
        root_def = roots[0]
    else:
        # fallback: pick the parent_def with max children
        if children_by_parent_def:
            root_def = max(children_by_parent_def.items(), key=lambda kv: len(kv[1]))[0]

    # Build definitions lookup (based on what appears in occurrences)
    definitions: Dict[str, Dict[str, Any]] = {}
    ref_defs_seen = {occ["ref_def"] for occ in occurrences.values() if occ.get("ref_def")}

    for ref_def in ref_defs_seen:
        a = assets_by_ref.get(ref_def, {})
        name = best_name(str(a.get("ref_name") or ""), def_name_map.get(ref_def, ""))
        if not name:
            name = ref_def

        definitions[ref_def] = {
            "ref_def": ref_def,
            "name": name,
            "qty": int(parent_qty.get(ref_def, 0)),
            "is_assembly": ref_def in children_by_parent_def,
            "assets": {
                "stl_path": str(a.get("stl_path") or "").strip(),
                "png_path": str(a.get("png_path") or "").strip(),
            },
            "classification": main_class.get(ref_def, {}),   # <-- ADD THIS
            "ancillaries": anc_links.get(ref_def, []),
        }

    bundle = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "root_def": root_def,
        "stats": {
            "occurrences": len(occurrences),
            "parent_defs": len(children_by_parent_def),
            "definitions_seen": len(definitions),
            "ancillary_groups": len(group_map),
        },
        "children_by_parent_def": children_by_parent_def,
        "occurrences": occurrences,
        "definitions": definitions,
        "ancillary_groups": group_map,  # optional but handy for UI tooltips/details
    }

    out_json = outp / "ui_bundle.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/out"
    raise SystemExit(main(out))
