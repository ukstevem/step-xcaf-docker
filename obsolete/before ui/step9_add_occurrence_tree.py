#!/usr/bin/env python3
"""
Step 9.3 (in-place): Add a full occurrence tree (with transforms) into out/assets_manifest_active.json
so the UI can render the full assembly from ONE file.

Inputs (read-only):
  - out/assets_manifest_active.json
  - out/xcaf_instances_active.json

Output (in-place):
  - out/assets_manifest_active.json updated with:
      derived.step9_3_occurrence_tree = {
        schema, created_utc,
        roots: [occ_id...],
        nodes: { occ_id: { parent_occ_id, children[], ref_def_sig, display_name, category,
                           local_xform_4x4?, global_xform_4x4?,
                           exploded_subparts? } },
        counts, notes
      }
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AM_ACTIVE = Path("out/assets_manifest_active.json")
XCAF_ACTIVE = Path("out/xcaf_instances_active.json")
BACKUP_SUFFIX = ".bak_step9_3"

# Occurrence parent key candidates (we still validate values)
PARENT_KEYS = ("parent_occ", "parent_occ_id", "parent", "parent_id", "parent_occurrence")

IDENTITY_4X4: List[float] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    # Deterministic JSON
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def safe_lower(v: Any) -> str:
    return safe_str(v).strip().lower()


def _parent_from_label_entry(occ_id: str) -> str:
    # LabelEntry style: "0:1:1:1:10:100" -> "0:1:1:1:10"
    s = occ_id.strip()
    if ":" not in s:
        return ""
    return s.rsplit(":", 1)[0]


def try_get_parent_raw(occ: Dict[str, Any]) -> str:
    for k in PARENT_KEYS:
        v = occ.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _pick_parent_occ_id(occ_id: str, occ: Dict[str, Any], occ_id_set: set[str]) -> str:
    """
    Prefer explicit parent field IF it points to a real occurrence id.
    Otherwise derive from label hierarchy.
    """
    raw = try_get_parent_raw(occ)
    if raw and raw in occ_id_set:
        return raw

    p = _parent_from_label_entry(occ_id)
    if p in occ_id_set:
        return p

    return ""


def try_get_ref_def_id(occ: Dict[str, Any]) -> str:
    v = occ.get("ref_def")
    if isinstance(v, str) and v.strip():
        return v.strip()
    for k in ("ref_label", "def_id", "definition", "ref_definition"):
        v2 = occ.get(k)
        if isinstance(v2, str) and v2.strip():
            return v2.strip()
    return ""


def _flatten_4x4(m: Any) -> Optional[List[float]]:
    # Accept [16] row-major OR [[4],[4],[4],[4]]
    if isinstance(m, list) and len(m) == 16:
        try:
            return [float(x) for x in m]
        except Exception:
            return None
    if isinstance(m, list) and len(m) == 4 and all(isinstance(r, list) and len(r) == 4 for r in m):
        try:
            out: List[float] = []
            for r in m:
                out.extend([float(x) for x in r])
            return out
        except Exception:
            return None
    return None


def _build_from_rot_trans(rot3x3: Any, trans3: Any) -> Optional[List[float]]:
    # rot3x3: [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]]
    # trans3: [tx,ty,tz]
    if not (isinstance(rot3x3, list) and len(rot3x3) == 3):
        return None
    if not all(isinstance(r, list) and len(r) == 3 for r in rot3x3):
        return None
    if not (isinstance(trans3, list) and len(trans3) == 3):
        return None
    try:
        r00, r01, r02 = [float(x) for x in rot3x3[0]]
        r10, r11, r12 = [float(x) for x in rot3x3[1]]
        r20, r21, r22 = [float(x) for x in rot3x3[2]]
        tx, ty, tz = [float(x) for x in trans3]
    except Exception:
        return None

    return [
        r00, r01, r02, tx,
        r10, r11, r12, ty,
        r20, r21, r22, tz,
        0.0, 0.0, 0.0, 1.0,
    ]


def extract_local_xform_4x4(occ: Dict[str, Any]) -> Optional[List[float]]:
    """
    Try to extract a 4x4 row-major matrix if present.

    Supported shapes:
      - list[16] directly
      - list[list] 4x4
      - occ["transform"] dict with {"matrix":[16]} or {"matrix":[[4x4]]}
      - dict with {"rotation_3x3":[[3x3]], "translation":[3]}
      - dict with {"rot":[[3x3]], "trans":[3]}
    """

    # Direct matrix candidates
    for k in ("local_xform_4x4", "xform_4x4", "transform_4x4", "matrix_4x4", "trsf_4x4"):
        v = occ.get(k)
        m = _flatten_4x4(v)
        if m is not None:
            return m

    # Nested dict under "transform" or "trsf" or "location"
    for container_key in ("transform", "trsf", "location", "placement"):
        v = occ.get(container_key)
        if not isinstance(v, dict):
            continue

        # 4x4 matrix in dict
        for mk in ("matrix", "m", "mat", "xform"):
            m = _flatten_4x4(v.get(mk))
            if m is not None:
                return m

        # rot+trans in dict
        rot = v.get("rotation_3x3") or v.get("rot") or v.get("R")
        trans = v.get("translation") or v.get("trans") or v.get("t") or v.get("T")
        m2 = _build_from_rot_trans(rot, trans)
        if m2 is not None:
            return m2

    return None


def mat4_mul(a: List[float], b: List[float]) -> List[float]:
    # Row-major 4x4 multiply: c = a*b
    c = [0.0] * 16
    for r in range(4):
        ar0 = a[r * 4 + 0]
        ar1 = a[r * 4 + 1]
        ar2 = a[r * 4 + 2]
        ar3 = a[r * 4 + 3]
        c[r * 4 + 0] = ar0 * b[0] + ar1 * b[4] + ar2 * b[8]  + ar3 * b[12]
        c[r * 4 + 1] = ar0 * b[1] + ar1 * b[5] + ar2 * b[9]  + ar3 * b[13]
        c[r * 4 + 2] = ar0 * b[2] + ar1 * b[6] + ar2 * b[10] + ar3 * b[14]
        c[r * 4 + 3] = ar0 * b[3] + ar1 * b[7] + ar2 * b[11] + ar3 * b[15]
    return c


def build_def_maps(xcaf: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    defs = xcaf.get("definitions", {})
    if not isinstance(defs, dict):
        return {}, {}

    id_to_sig: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}

    for def_id, d in defs.items():
        if not isinstance(def_id, str) or not isinstance(d, dict):
            continue
        sig = d.get("def_sig")
        if isinstance(sig, str) and sig.strip():
            id_to_sig[def_id] = sig.strip()
        nm = d.get("name")
        if isinstance(nm, str) and nm.strip():
            id_to_name[def_id] = nm.strip()

    return id_to_sig, id_to_name


def build_base_lookup(active: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    items = active.get("assets_manifest", {}).get("items", [])
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        sig = it.get("def_sig_used")
        if not (isinstance(sig, str) and sig.strip()):
            continue
        sig = sig.strip()
        if sig in out:
            continue
        out[sig] = {
            "display_name": safe_str(it.get("display_name")),
            "category": safe_lower(it.get("category")) or "unknown",
        }
    return out


def build_exploded_parent_map(active: Dict[str, Any]) -> Dict[str, List[str]]:
    d = active.get("derived", {})
    if not isinstance(d, dict):
        return {}

    out: Dict[str, List[str]] = {}

    eby = d.get("exploded_by_parent")
    if isinstance(eby, dict):
        for parent_sig, rec in eby.items():
            if not isinstance(parent_sig, str) or not isinstance(rec, dict):
                continue
            subs = rec.get("subparts", [])
            if not isinstance(subs, list):
                continue
            sset = set()
            for s in subs:
                if isinstance(s, dict):
                    sp = s.get("subpart_sig")
                    if isinstance(sp, str) and sp.strip():
                        sset.add(sp.strip())
            if sset:
                out[parent_sig] = sorted(sset)
        return out

    ep = d.get("exploded_parents")
    if isinstance(ep, dict):
        for parent_sig, rec in ep.items():
            if not isinstance(parent_sig, str) or not isinstance(rec, dict):
                continue
            subs = rec.get("subparts", [])
            if not isinstance(subs, list):
                continue
            sset = set()
            for s in subs:
                if isinstance(s, dict):
                    sp = s.get("subpart_sig")
                    if isinstance(sp, str) and sp.strip():
                        sset.add(sp.strip())
            if sset:
                out[parent_sig] = sorted(sset)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", default=str(AM_ACTIVE), help="Path to assets_manifest_active.json")
    ap.add_argument("--xcaf", default=str(XCAF_ACTIVE), help="Path to xcaf_instances_active.json")
    ap.add_argument("--backup", action="store_true", help="Write backup of active JSON first")
    ap.add_argument("--include_xform", action="store_true", help="Include local_xform_4x4 + global_xform_4x4")
    ns = ap.parse_args()

    active_path = Path(ns.active)
    xcaf_path = Path(ns.xcaf)

    if not active_path.is_file():
        raise SystemExit(f"Missing active JSON: {active_path}")
    if not xcaf_path.is_file():
        raise SystemExit(f"Missing XCAF active JSON: {xcaf_path}")

    active = read_json(active_path)
    xcaf_wrap = read_json(xcaf_path)

    xcaf = xcaf_wrap.get("xcaf_instances", {})
    if not isinstance(xcaf, dict):
        raise SystemExit("xcaf_instances_active.json: expected xcaf_instances object")

    id_to_sig, id_to_name = build_def_maps(xcaf)
    base_lookup = build_base_lookup(active)
    exploded_parent_map = build_exploded_parent_map(active)

    occs = xcaf.get("occurrences", {})
    if not isinstance(occs, dict):
        raise SystemExit("xcaf_instances_active.json: expected xcaf_instances.occurrences object")

    occ_id_set = set(k for k in occs.keys() if isinstance(k, str) and k)

    nodes: Dict[str, Dict[str, Any]] = {}
    children_by_parent: Dict[str, List[str]] = {}
    roots: List[str] = []

    # Pass 1: create node records + adjacency
    for occ_id, occ in occs.items():
        if not isinstance(occ_id, str) or not isinstance(occ, dict):
            continue

        parent_occ_id = _pick_parent_occ_id(occ_id, occ, occ_id_set)

        ref_def_id = try_get_ref_def_id(occ)
        ref_def_sig = id_to_sig.get(ref_def_id, "")

        disp = ""
        cat = "unknown"
        if ref_def_sig and ref_def_sig in base_lookup:
            disp = safe_str(base_lookup[ref_def_sig].get("display_name"))
            cat = safe_lower(base_lookup[ref_def_sig].get("category")) or "unknown"
        if not disp and ref_def_id:
            disp = id_to_name.get(ref_def_id, "")
        if not disp:
            disp = f"Item {ref_def_sig[:8] if ref_def_sig else occ_id[:8]}"

        node: Dict[str, Any] = {
            "occ_id": occ_id,
            "parent_occ_id": parent_occ_id,
            "ref_def_sig": ref_def_sig,
            "display_name": disp,
            "category": cat,
        }

        if ns.include_xform:
            local_m = extract_local_xform_4x4(occ)
            if local_m is not None:
                node["local_xform_4x4"] = local_m
            else:
                node["local_xform_4x4"] = IDENTITY_4X4[:]  # explicit for UI

        if ref_def_sig and ref_def_sig in exploded_parent_map:
            node["exploded_subparts"] = exploded_parent_map[ref_def_sig]

        nodes[occ_id] = node

        if parent_occ_id:
            children_by_parent.setdefault(parent_occ_id, []).append(occ_id)
        else:
            roots.append(occ_id)

    # Pass 2: attach children deterministically
    for pid, kids in children_by_parent.items():
        kids_sorted = sorted(kids)
        if pid in nodes:
            nodes[pid]["children"] = kids_sorted

    for occ_id in nodes.keys():
        if "children" not in nodes[occ_id]:
            nodes[occ_id]["children"] = []

    roots = sorted(roots)

    # Pass 3: compute global transforms deterministically (iterative traversal)
    if ns.include_xform:
        # Start each root at identity (or its own local if you prefer — here: global = local at root)
        for rid in roots:
            root_local = nodes[rid].get("local_xform_4x4")
            if isinstance(root_local, list) and len(root_local) == 16:
                nodes[rid]["global_xform_4x4"] = [float(x) for x in root_local]
            else:
                nodes[rid]["global_xform_4x4"] = IDENTITY_4X4[:]

        # Traverse: parent_global * child_local
        stack: List[str] = roots[:]  # already sorted
        visited: set[str] = set()

        while stack:
            oid = stack.pop()
            if oid in visited:
                continue
            visited.add(oid)

            parent_global = nodes[oid].get("global_xform_4x4")
            if not (isinstance(parent_global, list) and len(parent_global) == 16):
                parent_global = IDENTITY_4X4[:]
                nodes[oid]["global_xform_4x4"] = parent_global

            kids = nodes[oid].get("children", [])
            if isinstance(kids, list) and kids:
                # deterministic: push reverse so pop() visits in ascending order
                for kid in reversed(kids):
                    if kid not in nodes:
                        continue
                    child_local = nodes[kid].get("local_xform_4x4")
                    if not (isinstance(child_local, list) and len(child_local) == 16):
                        child_local = IDENTITY_4X4[:]
                        nodes[kid]["local_xform_4x4"] = child_local

                    nodes[kid]["global_xform_4x4"] = mat4_mul(parent_global, child_local)
                    stack.append(kid)

    derived = active.get("derived")
    if not isinstance(derived, dict):
        derived = {}
        active["derived"] = derived

    derived["step9_3_occurrence_tree"] = {
        "schema": "occurrence_tree_v1",
        "created_utc": utc_now_iso(),
        "roots": roots,
        "nodes": nodes,
        "counts": {
            "occurrences": len(nodes),
            "roots": len(roots),
        },
        "notes": (
            "UI should traverse from roots. "
            "node.ref_def_sig is the stable link to base items. "
            "If include_xform: local_xform_4x4 and global_xform_4x4 are row-major 4x4."
        ),
    }

    if ns.backup:
        backup = Path(str(active_path) + BACKUP_SUFFIX)
        backup.write_text(active_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[step9.3] backup: {backup}")

    write_json(active_path, active)
    print(f"[step9.3] wrote occurrence tree into: {active_path}")
    print(f"[step9.3] roots={len(roots)} nodes={len(nodes)} include_xform={bool(ns.include_xform)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
