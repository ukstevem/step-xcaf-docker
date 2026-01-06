#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def try_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def as_list_of_dicts(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return [a for a in x if isinstance(a, dict)]
    if isinstance(x, dict):
        # tolerate alternate structures
        for k in ("items", "entries", "manifest"):
            v = x.get(k)
            if isinstance(v, list):
                return [a for a in v if isinstance(a, dict)]
    return []


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pick_sig(item: Dict[str, Any], mode: str) -> str:
    if mode == "chiral":
        s = (item.get("sig_chiral") or "").strip()
        if s:
            return s
        return (item.get("sig_free") or "").strip()
    return (item.get("sig_free") or "").strip()


def main(out_dir: str, _tol: float = 0.5) -> int:
    out = Path(out_dir)

    mani_path = out / "stl_manifest_ancillary.json"
    xcaf_path = out / "xcaf_instances.json"

    mani = try_read_json(mani_path)
    if mani is None:
        print(f"Missing: {mani_path}")
        print("No ancillary manifest found; skipping grouping.")
        return 0

    xcaf = try_read_json(xcaf_path)
    if xcaf is None:
        print(f"Missing: {xcaf_path}")
        print("No xcaf_instances.json found; skipping grouping.")
        return 0

    items = as_list_of_dicts(mani)

    # Parent qty lookup from xcaf_instances.json definitions
    # Expected: xcaf["definitions"][ref_def]["qty"]
    definitions = {}
    if isinstance(xcaf, dict):
        definitions = xcaf.get("definitions", {}) or {}

    def parent_qty(parent_ref_def: str) -> int:
        if not parent_ref_def:
            return 0
        d = definitions.get(parent_ref_def)
        if isinstance(d, dict):
            return safe_int(d.get("qty"), 0)
        return 0

    # Filter ancillary items only
    anc_items = [it for it in items if (it.get("kind") == "ancillary")]

    # Build groups per mode (chiral/free)
    modes = ["chiral", "free"]
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in modes}

    for it in anc_items:
        for mode in modes:
            sig = pick_sig(it, mode)
            if not sig:
                # if no signature, ignore; exporter should provide
                continue

            g = grouped[mode].get(sig)
            if g is None:
                g = {
                    "mode": mode,
                    "sig_id": sig,
                    "members": [],  # we will store minimal per-member info
                }
                grouped[mode][sig] = g

            # minimal member info (avoid bloat)
            m = {
                "ref_def": it.get("ref_def", ""),
                "ref_name": it.get("ref_name", ""),
                "stl_path": it.get("stl_path", ""),
                "parent_ref_def": it.get("parent_ref_def", ""),
                "parent_name": it.get("parent_ref_name", it.get("parent_name", "")),
                "qty_per_parent": safe_int(it.get("qty_per_parent"), 1),
                # dims if present (from exporter)
                "dx": it.get("dx"),
                "dy": it.get("dy"),
                "dz": it.get("dz"),
            }
            g["members"].append(m)

    # Reduce groups: compute rep + parent rollups (no CSVs)
    out_modes: Dict[str, Any] = {}
    for mode in modes:
        out_groups: List[Dict[str, Any]] = []

        for sig, g in grouped[mode].items():
            members: List[Dict[str, Any]] = g["members"]

            # parent rollup
            parent_roll: Dict[str, Dict[str, Any]] = {}
            member_ref_defs = set()

            for m in members:
                member_ref_defs.add(m.get("ref_def", ""))

                pdef = (m.get("parent_ref_def") or "").strip()
                pname = (m.get("parent_name") or "").strip()
                qpp = safe_int(m.get("qty_per_parent"), 1)
                pq = parent_qty(pdef)

                pr = parent_roll.get(pdef)
                if pr is None:
                    pr = {
                        "parent_ref_def": pdef,
                        "parent_name": pname,
                        "parent_qty": pq,
                        "qty_per_parent": 0,
                        "total_qty": 0,
                    }
                    parent_roll[pdef] = pr

                # if multiple ancillaries with same signature appear on same parent,
                # qty_per_parent should accumulate.
                pr["qty_per_parent"] += qpp
                pr["total_qty"] = pr["parent_qty"] * pr["qty_per_parent"]

            # group totals
            parents_list = list(parent_roll.values())
            parents_list.sort(key=lambda x: (x["total_qty"], x["parent_name"]), reverse=True)

            total_qty = sum(p["total_qty"] for p in parents_list)
            parents_in_group = len([p for p in parents_list if p["parent_ref_def"]])
            items_in_group = len([r for r in member_ref_defs if r])

            # representative: pick member with highest implied total contribution
            # (parent_qty * qty_per_parent); tie-break by name
            def member_score(m: Dict[str, Any]) -> Tuple[int, str]:
                pdef = (m.get("parent_ref_def") or "").strip()
                pq = parent_qty(pdef)
                qpp = safe_int(m.get("qty_per_parent"), 1)
                return (pq * qpp, (m.get("ref_name") or ""))

            rep = max(members, key=member_score) if members else {}
            rep_name = rep.get("ref_name", "")
            rep_ref_def = rep.get("ref_def", "")
            rep_stl_path = rep.get("stl_path", "")

            # dims from rep if present
            dx = rep.get("dx")
            dy = rep.get("dy")
            dz = rep.get("dz")

            out_groups.append(
                {
                    "mode": mode,
                    "sig_id": sig,
                    "total_qty": total_qty,
                    "items_in_group": items_in_group,
                    "parents_in_group": parents_in_group,
                    "rep_name": rep_name,
                    "rep_ref_def": rep_ref_def,
                    "rep_stl_path": rep_stl_path,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    # keep just parent rollups; no per-member rows to avoid bloat
                    "parents": parents_list,
                    # keep member ref_defs only (optional but useful)
                    "member_ref_defs": sorted([x for x in member_ref_defs if x]),
                }
            )

        # sort by qty desc
        out_groups.sort(key=lambda x: (x["total_qty"], x["items_in_group"]), reverse=True)

        out_modes[mode] = {
            "group_count": len(out_groups),
            "groups": out_groups,
        }

    result = {
        "meta": {
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source_manifest": str(mani_path.name),
            "source_xcaf": str(xcaf_path.name),
            "ancillary_items_seen": len(anc_items),
        },
        "modes": out_modes,
    }

    out_path = out / "ancillary_groups.json"
    write_json(out_path, result)
    print(f"Wrote: {out_path}")
    print(
        "Groups:",
        f"chiral={result['modes']['chiral']['group_count']}",
        f"free={result['modes']['free']['group_count']}",
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: group_ancillaries.py OUT_DIR [tol]")
        sys.exit(2)
    out_dir = sys.argv[1]
    tol = 0.5
    if len(sys.argv) >= 3:
        try:
            tol = float(sys.argv[2])
        except Exception:
            tol = 0.5
    sys.exit(main(out_dir, tol))
