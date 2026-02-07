#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("top-level JSON is not an object")
        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {path} ({e})") from e


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _safe_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _pick_first(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _norm_occurrences(xcaf: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    occs = xcaf.get("occurrences")
    if occs is None:
        raise RuntimeError("xcaf_instances.json missing 'occurrences'")

    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(occs, dict):
        for occ_id, rec in occs.items():
            if not isinstance(rec, dict):
                continue
            oid = _safe_str(occ_id).strip()
            if not oid:
                continue
            out[oid] = rec
        return out

    if isinstance(occs, list):
        for rec in occs:
            if not isinstance(rec, dict):
                continue
            oid = _pick_first(rec.get("occ_id"), rec.get("id"))
            if not oid:
                continue
            out[oid] = rec
        return out

    raise RuntimeError("xcaf_instances.json 'occurrences' is neither dict nor list")


def _split_instance_suffix(s: Optional[str]) -> tuple[str, str]:
    if not s:
        return "", ""
    t = str(s).strip()
    # Keep deterministic ":<int>" suffix if present
    if ":" in t:
        head, tail = t.rsplit(":", 1)
        if tail.isdigit():
            return head, ":" + tail
    return t, ""


def _occ_ref_def_id(occ: Dict[str, Any]) -> Optional[str]:
    return _pick_first(occ.get("ref_def"), occ.get("def_id"), occ.get("definition"), occ.get("ref_def_id"))


def _def_sig(def_rec: Dict[str, Any]) -> Optional[str]:
    return _pick_first(def_rec.get("def_sig_free"), def_rec.get("def_sig"))


def _occ_display_fields(
    occ: Dict[str, Any],
    defs: Dict[str, Any],
    ref_def_id: Optional[str],
    occ_id: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """
    Returns:
      display_name  (prefer definition name + occurrence :N suffix if present)
      occ_label     (original occurrence label, e.g. NAU01085:1)
      def_name      (definition name, e.g. M8 Bolt)
    """
    # occurrence label (often part number / instance label)
    occ_label = _pick_first(occ.get("display_name"), occ.get("name"), occ.get("label"))
    _, suffix = _split_instance_suffix(occ_label)

    def_name = None
    if ref_def_id and isinstance(defs.get(ref_def_id), dict):
        def_name = _pick_first(defs[ref_def_id].get("name"))

    # Prefer definition name if available
    if def_name:
        display = def_name + suffix
    elif occ_label:
        display = str(occ_label).strip()
    else:
        display = occ_id

    return (
        display,
        (str(occ_label).strip() if occ_label else None),
        (str(def_name).strip() if def_name else None),
    )


def _occ_display_name(
    occ: Dict[str, Any],
    defs: Dict[str, Any],
    ref_def_id: Optional[str],
    occ_id: str,
) -> str:
    return _occ_display_fields(occ, defs, ref_def_id, occ_id)[0]


@dataclass(frozen=True)
class ManifestHit:
    match_status: str
    part_id: str
    stl_path: Optional[str]
    ref_def: Optional[str]
    def_sig_used: Optional[str]


def _index_manifest(man: Dict[str, Any]) -> Tuple[Dict[str, List[ManifestHit]], Dict[str, List[ManifestHit]]]:
    items = man.get("items")
    by_sig: Dict[str, List[ManifestHit]] = {}
    by_def: Dict[str, List[ManifestHit]] = {}

    if not isinstance(items, list):
        return by_sig, by_def

    for it in items:
        if not isinstance(it, dict):
            continue
        hit = ManifestHit(
            match_status=_safe_str(it.get("match_status")).strip() or "unknown",
            part_id=_safe_str(it.get("part_id")).strip(),
            stl_path=it.get("stl_path") if isinstance(it.get("stl_path"), str) else None,
            ref_def=_pick_first(it.get("ref_def")),
            def_sig_used=_pick_first(it.get("def_sig_used")),
        )
        if hit.def_sig_used:
            by_sig.setdefault(hit.def_sig_used, []).append(hit)
        if hit.ref_def:
            by_def.setdefault(hit.ref_def, []).append(hit)

    return by_sig, by_def


def _pick_manifest_stl(
    ref_def_sig: Optional[str],
    ref_def_id: Optional[str],
    by_sig: Dict[str, List[ManifestHit]],
    by_def: Dict[str, List[ManifestHit]],
) -> Optional[str]:
    cands: List[ManifestHit] = []
    if ref_def_sig and ref_def_sig in by_sig:
        cands.extend(by_sig[ref_def_sig])
    if (not cands) and ref_def_id and ref_def_id in by_def:
        cands.extend(by_def[ref_def_id])

    if not cands:
        return None

    def status_rank(s: str) -> int:
        # matched first, then others
        if s == "matched":
            return 0
        return 10

    cands_sorted = sorted(
        cands,
        key=lambda h: (status_rank(h.match_status), h.part_id, _safe_str(h.stl_path)),
    )
    for h in cands_sorted:
        if h.stl_path:
            return h.stl_path
    return None

def _group_key(n: Dict[str, Any]) -> str:
    # stable key for grouping leaf parts
    sig = n.get("ref_def_sig")
    if isinstance(sig, str) and sig:
        return "sig:" + sig
    did = n.get("ref_def_id")
    if isinstance(did, str) and did:
        return "def:" + did
    return "name:" + str(n.get("display_name") or "")


def _make_group_id(parent_id: str, key: str) -> str:
    # deterministic, safe group id
    # (no hashing needed; bounded length for sanity)
    safe = key.replace("/", "_").replace(":", "_")
    if len(safe) > 60:
        safe = safe[:60]
    return f"G:{parent_id}:{safe}"


def build_bom_global(run_id: str, tree: Dict[str, Any]) -> Dict[str, Any]:
    nodes = tree.get("nodes", {})
    if not isinstance(nodes, dict):
        nodes = {}

    # Aggregate leaf occurrences by stable key
    agg: Dict[str, Dict[str, Any]] = {}

    for occ_id, n in nodes.items():
        if not isinstance(n, dict):
            continue
        kids = n.get("children")
        if isinstance(kids, list) and len(kids) > 0:
            continue  # only leaf occurrences contribute to BOM counts

        sig = n.get("ref_def_sig") if isinstance(n.get("ref_def_sig"), str) else None
        did = n.get("ref_def_id") if isinstance(n.get("ref_def_id"), str) else None

        if sig:
            key = f"sig:{sig}"
        elif did:
            key = f"def:{did}"
        else:
            # last resort (should be rare)
            key = f"name:{str(n.get('def_name') or n.get('display_name') or occ_id)}"

        rec = agg.get(key)
        if rec is None:
            rec = {
                "key": key,
                "def_name": n.get("def_name") or n.get("display_name"),
                "occ_label_sample": n.get("occ_label"),
                "ref_def_sig": sig,
                "ref_def_id": did,
                "qty_total": 0,
                "shape_kind": n.get("shape_kind"),
                "solid_count": n.get("solid_count"),
                "stl_url": n.get("stl_url"),
                "bbox_mm": n.get("bbox_mm"),
            }
            agg[key] = rec

        if not rec.get("bbox_mm") and n.get("bbox_mm"):
            rec["bbox_mm"] = n.get("bbox_mm")

        # qty: if node has qty_total int use it, else count 1
        rec["qty_total"] += 1


        # prefer having an stl_url if we didn't already
        if not rec.get("stl_url") and n.get("stl_url"):
            rec["stl_url"] = n.get("stl_url")

    items = sorted(
        agg.values(),
        key=lambda r: (
            str(r.get("def_name") or "").lower(),
            str(r.get("ref_def_sig") or ""),
            str(r.get("ref_def_id") or ""),
        ),
    )

    return {
        "schema": "bom_global.v1",
        "run_id": run_id,
        "created_utc": _utc_iso_z(),
        "items": items,
    }



def build_grouped_tree(full_tree: Dict[str, Any], *, member_cap: int = 50) -> Dict[str, Any]:
    nodes = full_tree["nodes"]
    roots = full_tree["roots"]

    # Copy non-leaf nodes as-is; we'll rewrite children lists.
    out_nodes: Dict[str, Dict[str, Any]] = {}
    for oid, n in nodes.items():
        out_nodes[oid] = dict(n)
        out_nodes[oid]["children"] = list(n.get("children") or [])

    def is_leaf(oid: str) -> bool:
        c = out_nodes[oid].get("children")
        return not isinstance(c, list) or len(c) == 0

    # For every parent, group its leaf children that share a key.
    for parent_id, parent in list(out_nodes.items()):
        kids = parent.get("children")
        if not isinstance(kids, list) or not kids:
            continue

        leaf_kids = [k for k in kids if k in out_nodes and is_leaf(k)]
        if len(leaf_kids) < 2:
            continue  # nothing to group

        # Group only leaf kids; preserve non-leaf kids untouched.
        non_leaf_kids = [k for k in kids if k not in leaf_kids]

        buckets: Dict[str, List[str]] = {}
        for k in leaf_kids:
            key = _group_key(out_nodes[k])
            buckets.setdefault(key, []).append(k)

        grouped_children: List[str] = []
        for key, members in sorted(buckets.items(), key=lambda kv: kv[0]):
            if len(members) == 1:
                grouped_children.append(members[0])
                continue

            rep = members[0]
            rep_node = out_nodes[rep]

            gid = _make_group_id(parent_id, key)
            # deterministic representative: use rep_node + qty = sum of member qty_total if present
            qty_sum = 0
            for m in members:
                q = out_nodes[m].get("qty_total")
                if isinstance(q, int):
                    qty_sum += q
                else:
                    qty_sum += 1  # fallback: count occurrences

            gnode: Dict[str, Any] = {
                "kind": "group",
                "display_name": rep_node.get("display_name"),
                "qty_total": qty_sum,
                "children": [],
                "rep_occ_id": rep,
                "ref_def_id": rep_node.get("ref_def_id"),
                "ref_def_sig": rep_node.get("ref_def_sig"),
                "stl_url": rep_node.get("stl_url"),
                "shape_kind": rep_node.get("shape_kind"),
                "solid_count": rep_node.get("solid_count"),
            }

            # Keep a small sample for "where used" / drilldown, bounded
            gnode["members"] = members[:member_cap]
            gnode["member_count"] = len(members)

            out_nodes[gid] = gnode
            grouped_children.append(gid)

        # final children list: non-leaf first (stable), then grouped leaf children
        parent["children"] = sorted(non_leaf_kids) + grouped_children

    return {
        "schema": "occ_tree_grouped.v1",
        "run_id": full_tree.get("run_id"),
        "created_utc": _utc_iso_z(),
        "roots": roots,
        "nodes": out_nodes,
    }

def build_occ_tree(run_id: str, runs_root: Path) -> Dict[str, Any]:
    run_dir = runs_root / run_id
    xcaf_path = run_dir / "xcaf_instances.json"
    man_path = run_dir / "assets_manifest.json"
    out_path = run_dir / "occ_tree.json"

    xcaf = _read_json(xcaf_path)
    man = _read_json(man_path)

    defs = xcaf.get("definitions")
    if not isinstance(defs, dict):
        raise RuntimeError("xcaf_instances.json missing 'definitions' object")

    occs = _norm_occurrences(xcaf)
    by_sig, by_def = _index_manifest(man)

    # Build parent->children links.
    children_map: Dict[str, List[str]] = {oid: [] for oid in occs.keys()}
    parent_map: Dict[str, Optional[str]] = {oid: None for oid in occs.keys()}

    # Prefer explicit children if present.
    has_explicit_children = any(isinstance(occs[oid].get("children"), list) for oid in occs.keys())

    if has_explicit_children:
        for oid, rec in occs.items():
            kids = rec.get("children")
            if not isinstance(kids, list):
                continue
            for k in kids:
                kid = _safe_str(k).strip()
                if kid and kid in occs:
                    children_map[oid].append(kid)
                    parent_map[kid] = oid
    else:
        for oid, rec in occs.items():
            parent = _pick_first(rec.get("parent_occ_id"), rec.get("parent_occ"), rec.get("parent"))
            if parent and parent in occs:
                parent_map[oid] = parent
                children_map[parent].append(oid)

    # Roots
    roots_raw = xcaf.get("roots")
    roots: List[str] = []
    if isinstance(roots_raw, list):
        for r in roots_raw:
            rid = _safe_str(r).strip()
            if rid and rid in occs and parent_map.get(rid) is None:
                roots.append(rid)

    if not roots:
        roots = [oid for oid, p in parent_map.items() if not p]


    # Build node records
    nodes: Dict[str, Dict[str, Any]] = {}
    for oid, occ in occs.items():
        ref_def_id = _occ_ref_def_id(occ)
        def_rec = defs.get(ref_def_id) if ref_def_id and isinstance(defs.get(ref_def_id), dict) else None
        ref_def_sig = _def_sig(def_rec) if isinstance(def_rec, dict) else None

        stl_path = _pick_manifest_stl(ref_def_sig, ref_def_id, by_sig, by_def)
        stl_url = None
        if stl_path and stl_path.startswith("stl/"):
            stl_url = f"/runs/{run_id}/{stl_path}"

        display, occ_label, def_name = _occ_display_fields(occ, defs, ref_def_id, oid)

        # Deterministic child ordering
        kids = children_map.get(oid, [])
        kids_dedup = list(dict.fromkeys(kids))  # stable de-dupe
        kids_sorted = sorted(
            kids_dedup,
            key=lambda k: (_occ_display_name(occs[k], defs, _occ_ref_def_id(occs[k]), k).lower(), k),
        )

        node: Dict[str, Any] = {
            "display_name": display,
            "children": kids_sorted,
            "ref_def_sig": ref_def_sig,
            "stl_url": stl_url,
        }

        # Preserve original names for UI
        if occ_label:
            node["occ_label"] = occ_label
        if def_name:
            node["def_name"] = def_name

        # Optional metadata (safe + useful)
        if ref_def_id:
            node["ref_def_id"] = ref_def_id
        if isinstance(def_rec, dict):
            if "shape_kind" in def_rec:
                node["shape_kind"] = def_rec.get("shape_kind")
            if "solid_count" in def_rec:
                node["solid_count"] = def_rec.get("solid_count")
            if "def_sig_algo" in def_rec:
                node["def_sig_algo"] = def_rec.get("def_sig_algo")
            bb = def_rec.get("bbox")
            if isinstance(bb, dict):
                node["bbox_mm"] = bb

        nodes[oid] = node

    roots_sorted = sorted(roots, key=lambda r: (nodes[r]["display_name"].lower(), r))

    tree: Dict[str, Any] = {
        "schema": "occ_tree.v1",
        "run_id": run_id,
        "created_utc": _utc_iso_z(),
        "roots": roots_sorted,
        "nodes": nodes,
    }

    _write_json(out_path, tree)
    print(f"[ok] wrote: {out_path} roots={len(roots_sorted)} nodes={len(nodes)}")
        # ALSO write grouped tree (leaf grouping)
    grouped = build_grouped_tree(tree, member_cap=50)
    grouped_path = run_dir / "occ_tree_grouped.json"
    _write_json(grouped_path, grouped)
    print(f"[ok] wrote: {grouped_path} roots={len(grouped['roots'])} nodes={len(grouped['nodes'])}")

    return tree


def main() -> int:
    ap = argparse.ArgumentParser(description="Build occ_tree.json from xcaf_instances + assets_manifest")
    ap.add_argument("--run-id", required=True, help="Run id under runs root (e.g. ca7284804e89)")
    ap.add_argument("--runs-root", default="/app/ui_runs", help="Runs root dir (default: /app/ui_runs)")
    ns = ap.parse_args()

    runs_root = Path(ns.runs_root)
    if not runs_root.exists():
        print(f"[err] runs root not found: {runs_root}", file=sys.stderr)
        return 2

    try:
       tree = build_occ_tree(ns.run_id, runs_root)

       grouped = build_grouped_tree(tree, member_cap=50)
       _write_json(runs_root / ns.run_id / "occ_tree_grouped.json", grouped)
       print("[ok] wrote grouped tree")

       bom = build_bom_global(ns.run_id, tree)
       _write_json(runs_root / ns.run_id / "bom_global.json", bom)
       print("[ok] wrote bom_global.json")
       
       return 0
    except Exception as e:
        print(f"[err] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
