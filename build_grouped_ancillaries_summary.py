#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Small, boring helpers
# -----------------------------
def _die(msg: str) -> int:
    sys.stderr.write(msg.rstrip() + "\n")
    return 2


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def _manifest_items(manifest: Any) -> List[Dict[str, Any]]:
    # Your manifest is usually list[dict], but keep tolerant.
    if isinstance(manifest, list):
        return [x for x in manifest if isinstance(x, dict)]
    if isinstance(manifest, dict):
        for k in ("items", "entries", "manifest"):
            v = manifest.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


@dataclass(frozen=True)
class GroupInfo:
    sig_id: str
    rep_name: str
    rep_stl_path: str
    dx: str
    dy: str
    dz: str
    total_qty: int
    items_in_group: int
    parents_in_group: int


def _load_parent_names_from_stl_manifest(stl_manifest_path: Path) -> Dict[str, str]:
    """
    Map ref_def -> ref_name from stl_manifest.json.
    """
    manifest = _read_json(stl_manifest_path)
    items = _manifest_items(manifest)
    out: Dict[str, str] = {}
    for it in items:
        ref_def = str(it.get("ref_def", "")).strip()
        ref_name = str(it.get("ref_name", "")).strip()
        if ref_def and ref_name:
            out[ref_def] = ref_name
    return out


def _load_groups(groups_csv: Path) -> Dict[str, GroupInfo]:
    """
    Map sig_id -> GroupInfo
    """
    rows = _read_csv(groups_csv)
    out: Dict[str, GroupInfo] = {}
    for r in rows:
        sig_id = str(r.get("sig_id", "")).strip()
        if not sig_id:
            continue
        out[sig_id] = GroupInfo(
            sig_id=sig_id,
            rep_name=str(r.get("rep_name", "")).strip(),
            rep_stl_path=str(r.get("rep_stl_path", "")).strip(),
            dx=str(r.get("dx", "")).strip(),
            dy=str(r.get("dy", "")).strip(),
            dz=str(r.get("dz", "")).strip(),
            total_qty=_safe_int(r.get("total_qty", 0)),
            items_in_group=_safe_int(r.get("items_in_group", 0)),
            parents_in_group=_safe_int(r.get("parents_in_group", 0)),
        )
    return out


def build(out_dir: Path, mode: str) -> int:
    mode = mode.strip().lower()
    if mode not in ("chiral", "free"):
        return _die("mode must be 'chiral' or 'free'")

    stl_manifest = out_dir / "stl_manifest.json"
    groups_csv = out_dir / f"ancillary_groups_{mode}.csv"
    parent_map_csv = out_dir / f"ancillary_parent_map_{mode}.csv"

    if not stl_manifest.exists():
        return _die(f"Missing: {stl_manifest}")
    if not groups_csv.exists():
        return _die(f"Missing: {groups_csv}")
    if not parent_map_csv.exists():
        return _die(f"Missing: {parent_map_csv}")

    parent_name_by_ref_def = _load_parent_names_from_stl_manifest(stl_manifest)
    groups_by_sig = _load_groups(groups_csv)
    parent_rows = _read_csv(parent_map_csv)

    # 1) Clean grouped BOM (keep it tight and human readable)
    grouped_out_rows: List[Dict[str, Any]] = []
    for sig_id, g in groups_by_sig.items():
        grouped_out_rows.append(
            {
                "mode": mode,
                "sig_id": sig_id,
                "rep_name": g.rep_name,
                "total_qty": g.total_qty,
                "items_in_group": g.items_in_group,
                "parents_in_group": g.parents_in_group,
                "rep_stl_path": g.rep_stl_path,
                "dx": g.dx,
                "dy": g.dy,
                "dz": g.dz,
            }
        )

    grouped_out_rows.sort(key=lambda r: (-_safe_int(r.get("total_qty", 0)), str(r.get("rep_name", ""))))

    grouped_out_csv = out_dir / f"bom_ancillary_grouped_{mode}.csv"
    grouped_fields = [
        "mode",
        "sig_id",
        "rep_name",
        "total_qty",
        "items_in_group",
        "parents_in_group",
        "rep_stl_path",
        "dx",
        "dy",
        "dz",
    ]
    _write_csv(grouped_out_csv, grouped_out_rows, grouped_fields)
    print(f"Wrote: {grouped_out_csv} (rows={len(grouped_out_rows)})")

    # 2) Parent -> Group links enriched with parent_name + group rep info
    links_out_rows: List[Dict[str, Any]] = []
    for r in parent_rows:
        sig_id = str(r.get("sig_id", "")).strip()
        parent_ref_def = str(r.get("parent_ref_def", "")).strip()
        if not sig_id or not parent_ref_def:
            continue

        # Rename the problematic column
        qty_per_parent = _safe_int(r.get("qty_per_parent_sum", r.get("qty_per_parent", 0)))

        parent_qty = _safe_int(r.get("parent_qty", 0))
        total_qty = _safe_int(r.get("total_qty", 0))

        parent_name = parent_name_by_ref_def.get(parent_ref_def, "")

        g = groups_by_sig.get(sig_id)
        rep_name = g.rep_name if g else str(r.get("rep_child_name", "")).strip()
        rep_stl = g.rep_stl_path if g else str(r.get("rep_child_stl_path", "")).strip()

        links_out_rows.append(
            {
                "mode": mode,
                "parent_ref_def": parent_ref_def,
                "parent_name": parent_name,
                "parent_qty": parent_qty,
                "sig_id": sig_id,
                "group_rep_name": rep_name,
                "qty_per_parent": qty_per_parent,
                "total_qty": total_qty,
                "group_rep_stl_path": rep_stl,
            }
        )

    # Sort so UI can group nicely: parent then biggest qty first
    links_out_rows.sort(
        key=lambda r: (
            str(r.get("parent_name", "")),
            str(r.get("parent_ref_def", "")),
            -_safe_int(r.get("total_qty", 0)),
            str(r.get("group_rep_name", "")),
        )
    )

    links_out_csv = out_dir / f"parent_to_ancillary_groups_{mode}.csv"
    links_fields = [
        "mode",
        "parent_ref_def",
        "parent_name",
        "parent_qty",
        "sig_id",
        "group_rep_name",
        "qty_per_parent",
        "total_qty",
        "group_rep_stl_path",
    ]
    _write_csv(links_out_csv, links_out_rows, links_fields)
    print(f"Wrote: {links_out_csv} (rows={len(links_out_rows)})")

    return 0


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("Usage: build_grouped_ancillaries_summary.py /out [chiral|free]\n")
        return 2

    out_dir = Path(argv[1])
    mode = argv[2] if len(argv) >= 3 else "chiral"
    return build(out_dir, mode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
