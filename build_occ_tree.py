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
        # Some older outputs used "occurrence_index" etc. Extend here if needed.
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
                # Can't invent a stable id. Skip with warning.
                continue
            out[oid] = rec
        return out

    raise RuntimeError("xcaf_instances.json 'occurrences' is neither dict nor list")


def _occ_display_name(occ: Dict[str, Any], defs: Dict[str, Any], ref_def_id: Optional[str], occ_id: str) -> str:
    name = _pick_first(occ.get("display_name"), occ.get("name"), occ.get("label"))
    if name:
        return name
    if ref_def_id and isinstance(defs.get(ref_def_id), dict):
        dn = _pick_first(defs[ref_def_id].get("name"))
        if dn:
            return dn
    return occ_id


def _occ_ref_def_id(occ: Dict[str, Any]) -> Optional[str]:
    return _pick_first(occ.get("ref_def"), occ.get("def_id"), occ.get("definition"), occ.get("ref_def_id"))


def _def_sig(def_rec: Dict[str, Any]) -> Optional[str]:
    return _pick_first(def_rec.get("def_sig_free"), def_rec.get("def_sig"))


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


def _pick_manifest_stl(ref_def_sig: Optional[str], ref_def_id: Optional[str],
                       by_sig: Dict[str, List[ManifestHit]],
                       by_def: Dict[str, List[ManifestHit]]) -> Optional[str]:
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
            parent = _pick_first(rec.get("parent_occ_id"), rec.get("parent"))
            if parent and parent in occs:
                parent_map[oid] = parent
                children_map[parent].append(oid)

    # Roots
    roots_raw = xcaf.get("roots")
    roots: List[str] = []
    if isinstance(roots_raw, list):
        for r in roots_raw:
            rid = _safe_str(r).strip()
            if rid and rid in occs:
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

        display = _occ_display_name(occ, defs, ref_def_id, oid)

        # Deterministic child ordering
        kids = children_map.get(oid, [])
        kids_sorted = sorted(
            list(dict.fromkeys(kids)),  # de-dupe, stable
            key=lambda k: ( _occ_display_name(occs[k], defs, _occ_ref_def_id(occs[k]), k).lower(), k),
        )

        node: Dict[str, Any] = {
            "display_name": display,
            "children": kids_sorted,
            "ref_def_sig": ref_def_sig,
            "stl_url": stl_url,
        }

        # Optional metadata (safe + useful)
        if ref_def_id:
            node["ref_def_id"] = ref_def_id
        if isinstance(def_rec, dict):
            if "qty_total" in def_rec:
                node["qty_total"] = def_rec.get("qty_total")
            if "shape_kind" in def_rec:
                node["shape_kind"] = def_rec.get("shape_kind")
            if "solid_count" in def_rec:
                node["solid_count"] = def_rec.get("solid_count")
            if "def_sig_algo" in def_rec:
                node["def_sig_algo"] = def_rec.get("def_sig_algo")

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
        build_occ_tree(ns.run_id, runs_root)
        return 0
    except Exception as e:
        print(f"[err] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
