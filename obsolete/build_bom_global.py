#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(p: Path) -> Dict[str, Any]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected object JSON: {p}")
    return obj


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def _pick_first_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _def_sig(def_rec: Dict[str, Any]) -> Optional[str]:
    return _pick_first_str(def_rec.get("def_sig_free"), def_rec.get("def_sig"))


def _index_manifest_stl(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    Map ref_def_sig -> stl_url (prefer match_status == matched).
    Manifest items are assumed to include: def_sig_used, match_status, stl_path
    """
    items = manifest.get("items")
    if not isinstance(items, list):
        return {}

    best: Dict[str, Dict[str, Any]] = {}

    def rank(status: str) -> int:
        return 0 if status == "matched" else 10

    for it in items:
        if not isinstance(it, dict):
            continue
        sig = _pick_first_str(it.get("def_sig_used"))
        stl_path = it.get("stl_path") if isinstance(it.get("stl_path"), str) else None
        if not sig or not stl_path:
            continue
        status = str(it.get("match_status") or "unknown").strip()
        cand = {"rank": rank(status), "stl_path": stl_path}

        prev = best.get(sig)
        if prev is None or cand["rank"] < prev["rank"]:
            best[sig] = cand

    out: Dict[str, str] = {}
    for sig, rec in best.items():
        stl_path = rec["stl_path"]
        # UI expects a stl_url (served via /runs/<run_id>/stl/...)
        if stl_path.startswith("stl/"):
            out[sig] = stl_path
        else:
            out[sig] = stl_path
    return out


def build_bom_global(run_dir: Path) -> Dict[str, Any]:
    xcaf_path = run_dir / "xcaf_instances.json"
    man_path = run_dir / "assets_manifest.json"
    out_path = run_dir / "bom_global.json"

    xcaf = _read_json(xcaf_path)
    man = _read_json(man_path)

    defs = xcaf.get("definitions")
    if not isinstance(defs, dict):
        raise RuntimeError("xcaf_instances.json missing 'definitions' object")

    sig_to_stl_path = _index_manifest_stl(man)

    items: List[Dict[str, Any]] = []
    for def_id, def_rec in defs.items():
        if not isinstance(def_rec, dict):
            continue

        def_name = _pick_first_str(def_rec.get("name")) or str(def_id)
        qty_total = def_rec.get("qty_total")
        if not isinstance(qty_total, int):
            # if absent, treat as 0 so it doesn't pollute global BOM
            qty_total = 0

        # If you want to hide “0 qty” defs, skip them:
        if qty_total <= 0:
            continue

        shape_kind = def_rec.get("shape_kind") if isinstance(def_rec.get("shape_kind"), str) else None
        solid_count = def_rec.get("solid_count") if isinstance(def_rec.get("solid_count"), int) else None

        sig = _def_sig(def_rec) or ""
        stl_path = sig_to_stl_path.get(sig) if sig else None

        items.append(
            {
                "def_name": def_name,
                "key": f"sig:{sig}" if sig else f"def:{def_id}",
                "qty_total": qty_total,
                "ref_def_id": str(def_id),
                "ref_def_sig": sig or None,
                "shape_kind": shape_kind,
                "solid_count": solid_count,
                # keep same contract as your UI:
                "stl_url": (f"/runs/{run_dir.name}/{stl_path}" if stl_path and stl_path.startswith("stl/") else None),
            }
        )

    # Stable ordering: biggest qty first, then name
    items.sort(key=lambda it: (-int(it.get("qty_total") or 0), str(it.get("def_name") or "").lower()))

    out = {
        "schema": "bom_global.v1",
        "run_id": run_dir.name,
        "items": items,
    }
    _write_json(out_path, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default="/app/ui_runs")
    ns = ap.parse_args()

    run_dir = Path(ns.runs_root) / ns.run_id
    if not run_dir.exists():
        raise SystemExit(f"[err] run dir not found: {run_dir}")

    out = build_bom_global(run_dir)
    print(f"[ok] wrote bom_global.json items={len(out['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
