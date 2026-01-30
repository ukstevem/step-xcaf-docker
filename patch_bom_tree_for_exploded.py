#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/app/ui_runs"))

def _read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)

def _subpart_counts(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        s = str(r.get("subpart_sig") or "").strip()
        if not s:
            continue
        out[s] = out.get(s, 0) + 1
    return out

def _first_seen_order(recs: List[Dict[str, Any]]) -> List[str]:
    """Return subpart_sig in first-seen order (deterministic = manifest order)."""
    seen: Dict[str, bool] = {}
    out: List[str] = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        s = str(r.get("subpart_sig") or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen[s] = True
        out.append(s)
    return out


def _index_manifest_records(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pick a representative record for each subpart_sig (first occurrence)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        s = str(r.get("subpart_sig") or "").strip()
        if not s:
            continue
        if s not in out:
            out[s] = r
    return out


def _subpart_rollup(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      subpart_sig -> {
        "count": int,
        "stl_url": str|None,   # representative STL
        "step_relpath": str|None,
        "bbox_size": [x,y,z]|None
      }
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        s = str(r.get("subpart_sig") or "").strip()
        if not s:
            continue
        ent = out.get(s)
        if ent is None:
            ent = {"count": 0, "stl_url": None, "step_relpath": None, "bbox_size": None}
            out[s] = ent
        ent["count"] += 1

        # pick first available representative assets
        if ent["stl_url"] is None:
            u = r.get("stl_url")
            if isinstance(u, str) and u.strip():
                ent["stl_url"] = u.strip()
        if ent["step_relpath"] is None:
            u = r.get("step_relpath")
            if isinstance(u, str) and u.strip():
                ent["step_relpath"] = u.strip()
        if ent["bbox_size"] is None:
            bb = r.get("bbox")
            if isinstance(bb, dict):
                sz = bb.get("size")
                if isinstance(sz, list) and len(sz) == 3:
                    ent["bbox_size"] = sz
    return out


def _subpart_rollup(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        s = str(r.get("subpart_sig") or "").strip()
        if not s:
            continue

        ent = out.get(s)
        if ent is None:
            ent = {"count": 0, "stl_url": None, "step_relpath": None, "bbox_size": None}
            out[s] = ent

        ent["count"] += 1

        if ent["stl_url"] is None:
            u = r.get("stl_url")
            if isinstance(u, str) and u.strip():
                ent["stl_url"] = u.strip()

        if ent["step_relpath"] is None:
            u = r.get("step_relpath")
            if isinstance(u, str) and u.strip():
                ent["step_relpath"] = u.strip()

        if ent["bbox_size"] is None:
            bb = r.get("bbox")
            if isinstance(bb, dict):
                sz = bb.get("size")
                if isinstance(sz, list) and len(sz) == 3:
                    ent["bbox_size"] = sz

    return out


def patch_bom(bom: Dict[str, Any], exploded: Dict[str, Any]) -> Dict[str, Any]:
    items = bom.get("items")
    if not isinstance(items, list):
        bom["_explode_patch_note"] = "bom.items not a list; left unchanged"
        return bom

    new_items: List[Dict[str, Any]] = []

    for row in items:
        if not isinstance(row, dict):
            continue

        parent_sig = str(
            row.get("ref_def_sig") or row.get("def_sig_used") or row.get("def_sig") or ""
        ).strip()

        if not parent_sig or parent_sig not in exploded:
            new_items.append(row)
            continue

        parent_name = str(row.get("def_name") or row.get("name") or "").strip() or "Parent"
        qty_parent = int(row.get("qty_total") or row.get("qty") or 1)

        recs = exploded.get(parent_sig)
        if not isinstance(recs, list) or not recs:
            new_items.append(row)
            continue

        roll = _subpart_rollup(recs)

        # Replace parent with rolled-up subparts (BOM view)
        for sub_sig in sorted(roll.keys()):
            ent = roll[sub_sig]
            per_parent_count = int(ent.get("count") or 0)
            if per_parent_count <= 0:
                continue

            sub_short = sub_sig[:10]
            def_name = f"{parent_name} :: subpart {sub_short}"

            new_items.append({
                "kind": "exploded_subpart",
                "def_name": def_name,
                "qty_total": qty_parent * per_parent_count,
                "solid_count": 1,                 # subparts are single solids
                "ref_def_sig": sub_sig,           # UI identity for this row
                "stl_url": ent.get("stl_url"),    # <- makes selection load mesh
                "step_relpath": ent.get("step_relpath"),
                "bbox_mm": {"size": ent.get("bbox_size")} if ent.get("bbox_size") else None,
                "from_parent_def_sig": parent_sig,
                "from_parent_def_name": parent_name,
                "per_parent_count": per_parent_count,
                "parent_qty": qty_parent,
            })

    out = dict(bom)
    out["items"] = new_items
    out["_explode_patch"] = "bom_exploded_patch_v2"
    return out


def patch_tree(tree: Dict[str, Any], exploded: Dict[str, Any]) -> Dict[str, Any]:
    nodes = tree.get("nodes")
    if not isinstance(nodes, dict):
        tree["_explode_patch_note"] = "tree.nodes not a dict; left unchanged"
        return tree

    out_nodes: Dict[str, Any] = {}

    for nid, node in nodes.items():
        if not isinstance(node, dict):
            out_nodes[nid] = node
            continue

        ref = str(
            node.get("ref_def_sig") or node.get("def_sig_used") or node.get("def_sig") or ""
        ).strip()

        if ref and ref in exploded:
            recs_any = exploded.get(ref)
            recs = recs_any if isinstance(recs_any, list) else []
            sigs_in_order = _first_seen_order(recs)

            nn = dict(node)
            # keep deterministic manifest order, don’t uniq/sort beyond first-seen
            nn["exploded_subparts"] = sigs_in_order
            out_nodes[nid] = nn
        else:
            out_nodes[nid] = node

    out = dict(tree)
    out["nodes"] = out_nodes
    out["_explode_patch"] = "occ_tree_exploded_patch_v2"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ns = ap.parse_args()

    run_dir = (RUNS_DIR / ns.run_id).resolve()
    man = _read_json(run_dir / "exploded_manifest.json")
    exploded = man.get("exploded") if isinstance(man, dict) else {}
    if not isinstance(exploded, dict):
        exploded = {}

    bom_path = run_dir / "bom_global.json"
    tree_path = run_dir / "occ_tree_grouped.json"

    if bom_path.is_file():
        bom2 = patch_bom(_read_json(bom_path), exploded)
        _write_json(run_dir / "bom_global_exploded.json", bom2)

    if tree_path.is_file():
        tree2 = patch_tree(_read_json(tree_path), exploded)
        _write_json(run_dir / "occ_tree_grouped_exploded.json", tree2)

    print("[patch] done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
