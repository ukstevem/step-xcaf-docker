#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/app/ui_runs")).resolve()


def _read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def _first_seen_order(recs: List[Dict[str, Any]]) -> List[str]:
    """Deterministic order: preserve exploded_manifest record order."""
    seen: Dict[str, bool] = {}
    out: List[str] = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        s = str(r.get("subpart_sig") or "").strip()
        if not s or s in seen:
            continue
        seen[s] = True
        out.append(s)
    return out


def _rollup_subparts(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    sub_sig -> {"count": int, "sample": dict(first seen)}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        sub = str(r.get("subpart_sig") or "").strip()
        if not sub:
            continue
        ent = out.get(sub)
        if ent is None:
            out[sub] = {"count": 1, "sample": r}
        else:
            ent["count"] = int(ent["count"]) + 1
    return out


def _to_run_url(run_id: str, maybe_rel: Any) -> Optional[str]:
    """
    Match bom_global.json stl_url style:
      - if already absolute (/runs/... or http...), keep
      - else prefix /runs/<run_id>/
    """
    if not isinstance(maybe_rel, str):
        return None
    s = maybe_rel.strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://") or s.startswith("/"):
        return s
    return f"/runs/{run_id}/{s.lstrip('./')}"


def _bbox_mm_from_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prefer full-fat bbox_mm (min/max/size).
    Fallback to bbox with min/max/size.
    Final fallback: bbox with size only.
    """
    # full-fat
    for k in ("bbox_mm", "bbox"):
        bb = sample.get(k)
        if isinstance(bb, dict):
            mn = bb.get("min")
            mx = bb.get("max")
            sz = bb.get("size")
            if isinstance(mn, list) and isinstance(mx, list) and isinstance(sz, list):
                if len(mn) == 3 and len(mx) == 3 and len(sz) == 3:
                    return {
                        "min": [float(mn[0]), float(mn[1]), float(mn[2])],
                        "max": [float(mx[0]), float(mx[1]), float(mx[2])],
                        "size": [float(sz[0]), float(sz[1]), float(sz[2])],
                    }

    # size-only (if that's all you have)
    bb = sample.get("bbox")
    if isinstance(bb, dict):
        sz = bb.get("size")
        if isinstance(sz, list) and len(sz) == 3:
            return {"size": [float(sz[0]), float(sz[1]), float(sz[2])]}
    return None


def patch_bom(bom: Dict[str, Any], exploded: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    items = bom.get("items")
    if not isinstance(items, list):
        out = dict(bom)
        out["_explode_patch_note"] = "bom.items not a list; left unchanged"
        return out

    new_items: List[Dict[str, Any]] = []

    for row in items:
        if not isinstance(row, dict):
            continue

        # 1) Keep original row unchanged (full-fat)
        new_items.append(row)

        # 2) If this row has exploded subparts, append full-fat subpart rows
        parent_sig = str(row.get("ref_def_sig") or "").strip()
        if not parent_sig:
            continue

        recs_any = exploded.get(parent_sig)
        recs = recs_any if isinstance(recs_any, list) else []
        if not recs:
            continue

        roll = _rollup_subparts(recs)
        qty_parent = int(row.get("qty_total") or 1)
        parent_name = str(row.get("def_name") or "item")

        # Deterministic subpart order
        ordered_subs = _first_seen_order(recs)

        for sub_sig in ordered_subs:
            ent = roll.get(sub_sig)
            if ent is None:
                continue

            per_parent = int(ent["count"])
            sample = ent.get("sample")
            sample = sample if isinstance(sample, dict) else {}

            # Clone parent row so the schema matches "full fat"
            nr = dict(row)

            # Overwrite only what's required for the subpart identity
            nr["def_name"] = f"{parent_name} :: subpart {sub_sig[:8]}"
            nr["key"] = f"sig:{sub_sig}"
            nr["ref_def_sig"] = sub_sig
            nr["ref_def_id"] = None
            nr["shape_kind"] = "SOLID"
            nr["solid_count"] = 1
            nr["qty_total"] = qty_parent * per_parent

            # Carry bbox + stl_url from exploded manifest sample
            nr["bbox_mm"] = _bbox_mm_from_sample(sample)
            nr["stl_url"] = _to_run_url(run_id, sample.get("stl_url"))

            # Extra traceability fields (safe additions)
            nr["from_parent_def_sig"] = parent_sig
            nr["from_parent_def_name"] = parent_name
            nr["per_parent_count"] = per_parent
            nr["is_exploded_subpart"] = True

            new_items.append(nr)

    out = dict(bom)
    out["items"] = new_items
    out["_explode_patch"] = "bom_exploded_patch_fullfat_v1"
    return out


def patch_tree(tree: Dict[str, Any], exploded: Dict[str, Any]) -> Dict[str, Any]:
    nodes = tree.get("nodes")
    if not isinstance(nodes, dict):
        out = dict(tree)
        out["_explode_patch_note"] = "tree.nodes not a dict; left unchanged"
        return out

    out_nodes: Dict[str, Any] = {}

    for nid, node in nodes.items():
        if not isinstance(node, dict):
            out_nodes[nid] = node
            continue

        ref = str(node.get("ref_def_sig") or node.get("def_sig_used") or node.get("def_sig") or "").strip()
        if ref and ref in exploded:
            recs_any = exploded.get(ref)
            recs = recs_any if isinstance(recs_any, list) else []
            nn = dict(node)
            nn["exploded_subparts"] = _first_seen_order(recs)
            out_nodes[nid] = nn
        else:
            out_nodes[nid] = node

    out = dict(tree)
    out["nodes"] = out_nodes
    out["_explode_patch"] = "occ_tree_exploded_patch_fullfat_v1"
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
        bom2 = patch_bom(_read_json(bom_path), exploded, run_id=ns.run_id)
        _write_json(run_dir / "bom_global_exploded.json", bom2)

    if tree_path.is_file():
        tree2 = patch_tree(_read_json(tree_path), exploded)
        _write_json(run_dir / "occ_tree_grouped_exploded.json", tree2)

    print("[patch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
