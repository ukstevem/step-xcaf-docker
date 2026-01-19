#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(p: Path) -> List[Dict[str, Any]]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_csv(p: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    # Minimal CSV writer to avoid extra deps (pandas not required)
    import csv
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main(out_dir: str = "/out") -> None:
    out_dir_p = Path(out_dir)
    in_path = out_dir_p / "xcaf_instances.json"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run read_step_xcaf.py first.")

    rows = _read_json(in_path)

    # Expect these keys from your extractor; fallback safely
    for r in rows:
        r.setdefault("ref_def", "")
        r.setdefault("ref_name", "")
        r.setdefault("parent_def", "")

    # Prefer human name, else label entry
    for r in rows:
        ref_name = (r.get("ref_name") or "").strip()
        ref_def = (r.get("ref_def") or "").strip()
        r["item_name"] = ref_name if ref_name else ref_def

    # Any definition that appears as a parent_def has children => assembly (heuristic from extracted data)
    parent_defs = set((r.get("parent_def") or "").strip() for r in rows if (r.get("parent_def") or "").strip())

    def make_bom(source_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counts: Dict[tuple[str, str], int] = {}
        for r in source_rows:
            ref_def = (r.get("ref_def") or "").strip()
            name = (r.get("item_name") or "").strip()
            key = (ref_def, name)
            counts[key] = counts.get(key, 0) + 1

        out = [{"ref_def": k[0], "item_name": k[1], "qty": v} for k, v in counts.items()]
        out.sort(key=lambda x: (-int(x["qty"]), x["item_name"]))
        return out

    bom_all = make_bom(rows)

    # Leaf-only: exclude items whose ref_def is itself a parent_def anywhere
    leaf_rows = [r for r in rows if (r.get("ref_def") or "").strip() not in parent_defs]
    bom_leaf = make_bom(leaf_rows)

    all_csv = out_dir_p / "bom_from_xcaf_all.csv"
    leaf_csv = out_dir_p / "bom_from_xcaf_leaf.csv"
    _write_csv(all_csv, bom_all, ["ref_def", "item_name", "qty"])
    _write_csv(leaf_csv, bom_leaf, ["ref_def", "item_name", "qty"])

    print(f"Wrote: {all_csv}", flush=True)
    print(f"Wrote: {leaf_csv}", flush=True)
    print(f"All occurrences: unique_items={len(bom_all)} total_qty={sum(int(r['qty']) for r in bom_all)}", flush=True)
    print(f"Leaf-only:       unique_items={len(bom_leaf)} total_qty={sum(int(r['qty']) for r in bom_leaf)}", flush=True)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/out"
    main(out)
