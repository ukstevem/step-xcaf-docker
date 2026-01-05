#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _manifest_items(manifest: Any) -> List[Dict[str, Any]]:
    """
    Support both:
      - list[dict]                       (your current stl_manifest.json)
      - {"items": [...]} or {"entries": [...]}  (future-proofing)
    """
    if isinstance(manifest, list):
        return [x for x in manifest if isinstance(x, dict)]

    if isinstance(manifest, dict):
        for k in ("items", "entries", "manifest"):
            v = manifest.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    return []


def _index_manifest_by_ref_def(manifest: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in _manifest_items(manifest):
        ref_def = str(it.get("ref_def", "")).strip()
        if ref_def:
            out[ref_def] = it
    return out


def _group_leaf_bom(
    bom_leaf_rows: List[Dict[str, str]],
    manifest_by_def: Dict[str, Dict[str, Any]],
    sig_field: str,  # "sig_chiral" or "sig_free"
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    def_to_group: Dict[str, str] = {}

    for r in bom_leaf_rows:
        ref_def = str(r.get("ref_def", "")).strip()
        item_name = str(r.get("item_name", "")).strip()
        qty = _safe_int(r.get("qty", 0), 0)

        meta = manifest_by_def.get(ref_def, {})
        sig = str(meta.get(sig_field, "")).strip()
        gid = sig if sig else ref_def  # fallback if signature missing

        def_to_group[ref_def] = gid

        if gid not in groups:
            groups[gid] = {
                "group_id": gid,
                "sig_field": sig_field,
                "qty_total": 0,
                "member_count": 0,
                "rep_ref_def": "",
                "rep_name": "",
                "rep_stl_path": "",
                "rep_png_path": "",
                "members": [],
                "sig_chiral": str(meta.get("sig_chiral", "")).strip(),
                "sig_free": str(meta.get("sig_free", "")).strip(),
            }

        g = groups[gid]
        g["qty_total"] += qty

        display_name = str(meta.get("ref_name", "")).strip() or item_name

        g["members"].append(
            {
                "ref_def": ref_def,
                "name": display_name,
                "qty": qty,
                "stl_path": str(meta.get("stl_path", "")),
                "png_path": str(meta.get("png_path", "")),
                "sig_chiral": str(meta.get("sig_chiral", "")).strip(),
                "sig_free": str(meta.get("sig_free", "")).strip(),
            }
        )

    # Finalize: member_count, representative, stable ordering
    groups_list: List[Dict[str, Any]] = []
    for gid, g in groups.items():
        g["members"] = sorted(g["members"], key=lambda x: (x.get("name", ""), x.get("ref_def", "")))
        g["member_count"] = len(g["members"])

        # representative: highest qty, stable tie-break by ref_def
        rep = None
        for m in sorted(g["members"], key=lambda x: (-_safe_int(x.get("qty", 0), 0), x.get("ref_def", ""))):
            rep = m
            break

        if rep is not None:
            g["rep_ref_def"] = rep.get("ref_def", "")
            g["rep_name"] = rep.get("name", "")
            g["rep_stl_path"] = rep.get("stl_path", "")
            g["rep_png_path"] = rep.get("png_path", "")

        groups_list.append(g)

    groups_list = sorted(groups_list, key=lambda x: (-_safe_int(x.get("qty_total", 0), 0), x.get("group_id", "")))
    return {"groups": groups_list, "def_to_group": def_to_group}


def _enrich_leaf_bom(
    bom_leaf_rows: List[Dict[str, str]],
    manifest_by_def: Dict[str, Dict[str, Any]],
    def_to_group_chiral: Dict[str, str],
    def_to_group_free: Dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for r in bom_leaf_rows:
        ref_def = str(r.get("ref_def", "")).strip()
        qty = _safe_int(r.get("qty", 0), 0)

        meta = manifest_by_def.get(ref_def, {})
        name = str(meta.get("ref_name", "")).strip() or str(r.get("item_name", "")).strip()

        out.append(
            {
                "ref_def": ref_def,
                "name": name,
                "qty": qty,
                "stl_path": str(meta.get("stl_path", "")),
                "png_path": str(meta.get("png_path", "")),
                "sig_chiral": str(meta.get("sig_chiral", "")).strip(),
                "sig_free": str(meta.get("sig_free", "")).strip(),
                "group_chiral": def_to_group_chiral.get(ref_def, ""),
                "group_free": def_to_group_free.get(ref_def, ""),
            }
        )

    out = sorted(out, key=lambda x: (-_safe_int(x.get("qty", 0), 0), x.get("name", ""), x.get("ref_def", "")))
    return out


def main(out_dir: str) -> int:
    outp = Path(out_dir)

    xcaf_path = outp / "xcaf_instances.json"
    manifest_path = outp / "stl_manifest.json"
    leaf_csv = outp / "bom_from_xcaf_leaf.csv"
    all_csv = outp / "bom_from_xcaf_all.csv"

    if not xcaf_path.exists():
        print(f"Missing: {xcaf_path}", flush=True)
        return 2
    if not manifest_path.exists():
        print(f"Missing: {manifest_path}", flush=True)
        return 2
    if not leaf_csv.exists():
        print(f"Missing: {leaf_csv}", flush=True)
        return 2

    xcaf = _read_json(xcaf_path)
    manifest = _read_json(manifest_path)
    bom_leaf = _read_csv(leaf_csv)
    bom_all = _read_csv(all_csv) if all_csv.exists() else []

    manifest_by_def = _index_manifest_by_ref_def(manifest)

    grouping_chiral = _group_leaf_bom(bom_leaf, manifest_by_def, "sig_chiral")
    grouping_free = _group_leaf_bom(bom_leaf, manifest_by_def, "sig_free")

    enriched_leaf = _enrich_leaf_bom(
        bom_leaf,
        manifest_by_def,
        grouping_chiral["def_to_group"],
        grouping_free["def_to_group"],
    )

    bundle: Dict[str, Any] = {
        "meta": {
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "counts": {
                "manifest_defs": len(manifest_by_def),
                "bom_leaf_rows": len(bom_leaf),
                "bom_all_rows": len(bom_all),
                "groups_chiral": len(grouping_chiral["groups"]),
                "groups_free": len(grouping_free["groups"]),
            },
        },
        "xcaf": xcaf,
        "stl_manifest": manifest,
        "bom": {
            "leaf": bom_leaf,
            "all": bom_all,
            "leaf_enriched": enriched_leaf,
        },
        "grouping": {
            "chiral": grouping_chiral,
            "free": grouping_free,
        },
    }

    out_json = outp / "ui_bundle.json"
    out_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    c = bundle["meta"]["counts"]
    print(f"Wrote: {out_json}", flush=True)
    print(
        f"Counts: defs={c['manifest_defs']} leaf={c['bom_leaf_rows']} "
        f"groups(chiral)={c['groups_chiral']} groups(free)={c['groups_free']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python build_ui_bundle.py /out")
    raise SystemExit(main(sys.argv[1]))
