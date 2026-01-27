#!/usr/bin/env python3
"""
Step 6: Merge explosion decisions + exploded_parts into ACTIVE canonical outputs.

Reads (truth inputs, never modified):
  /out/xcaf_instances.json
  /out/assets_manifest.json
  /out/review/multibody_decisions.json
  /out/exploded/exploded_parts.json

Writes (derived canonical outputs used downstream):
  /out/xcaf_instances_active.json
  /out/assets_manifest_active.json

Append-only log:
  /out/derived/step6_merge_log.jsonl

Notes:
- Pure merge/index step. Does NOT generate geometry.
- Joins only on stable keys: def_sig / def_sig_free / subpart_sig / subpart_sig_free
- Deterministic ordering.
- Guardrails + fail-fast.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Power-of-10 constants
# -------------------------
SCHEMA_XCAF_ACTIVE = "xcaf_instances_active_v1"
SCHEMA_MANIFEST_ACTIVE = "assets_manifest_active_v1"

DECISION_EXPLODE = "explode"
ALLOWED_DECISIONS = {"explode", "keep", "defer"}

MAX_DECISIONS = 2_000_000
MAX_STEP5_ROWS = 50_000_000
MAX_SUBPARTS_PER_PARENT = 5_000
MAX_UNIQUE_SUBPARTS = 5_000_000

LOG_REL = "derived/step6_merge_log.jsonl"


# -------------------------
# Utilities
# -------------------------
def _fail(msg: str) -> None:
    raise SystemExit(f"[step6] ERROR: {msg}")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _fail(f"Missing JSON: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON in {path}: {e}")
    if not isinstance(obj, dict):
        _fail(f"Expected JSON object at top-level: {path}")
    return obj


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _as_list3(v: Any) -> Optional[List[float]]:
    if not isinstance(v, list) or len(v) != 3:
        return None
    out: List[float] = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out


def _utc_now_iso() -> str:
    # ISO-like, stable enough for logs
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_exploded_stl_path(stl_path: str) -> str:
    """
    Normalize STL paths so downstream can resolve as /out/<stl_path>.
    We standardize exploded to: exploded/stl/<parent>/<subpart>.stl
    Step5 may already write that; if it wrote stl/<parent>/<subpart>.stl we prefix exploded/.
    """
    p = (stl_path or "").strip().lstrip("/")
    if p.startswith("exploded/stl/"):
        return p
    if p.startswith("stl/"):
        return "exploded/" + p
    # allow already-correct relative paths
    return p

def _build_def_name_by_sig(xcaf: Dict[str, Any]) -> Dict[str, str]:
    """
    Build stable name map: def_sig -> definition.name (preferred).
    Only include defs with has_shape=True and def_sig present.
    Deterministic: last-wins is avoided by only assigning once.
    """
    defs = xcaf.get("definitions")
    if not isinstance(defs, dict):
        return {}

    out: Dict[str, str] = {}
    for def_id in sorted(defs.keys()):
        d = defs.get(def_id)
        if not isinstance(d, dict):
            continue
        if not bool(d.get("has_shape", False)):
            continue
        sig = str(d.get("def_sig") or "").strip()
        if not sig:
            continue
        nm = str(d.get("name") or "").strip()
        if not nm:
            continue
        # deterministic: first assignment wins (defs iterated sorted)
        if sig not in out:
            out[sig] = nm
    return out


def _build_parent_name_by_sig(step5_by_parent: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    """
    Parent names come from Step5 rows: choose the most common non-empty parent_name.
    Deterministic tie-break: lexicographic.
    """
    out: Dict[str, str] = {}
    for parent_sig in sorted(step5_by_parent.keys()):
        rows = step5_by_parent.get(parent_sig, [])
        counts: Dict[str, int] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            nm = str(r.get("parent_name") or "").strip()
            if not nm:
                continue
            counts[nm] = counts.get(nm, 0) + 1
        if not counts:
            continue
        # pick highest count; tie-break lexicographically
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out[parent_sig] = best
    return out


def _pick_display_name_for_subpart(
    sub_sig: str,
    where_used: List[str],
    parent_name_by_sig: Dict[str, str],
) -> Tuple[str, str, str]:
    """
    Deterministic naming that never concatenates multiple parents.

    - If used under exactly one parent: "<parent_name> / subpart <8>"
    - If used under multiple parents:  "Subpart <8>"  (stable, parent-independent)

    Still returns representative parent fields for UI context.
    """
    where_used_clean = sorted(set(str(x).strip() for x in where_used if str(x).strip()))
    rep_parent_sig = where_used_clean[0] if where_used_clean else ""
    rep_parent_name = parent_name_by_sig.get(rep_parent_sig, "") if rep_parent_sig else ""

    if len(where_used_clean) == 1 and rep_parent_name:
        disp = f"{rep_parent_name} / subpart {sub_sig[:8]}"
    else:
        disp = f"Subpart {sub_sig[:8]}"

    return disp, rep_parent_sig, rep_parent_name


# -------------------------
# Core merge logic
# -------------------------
def _load_decisions(decisions_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dec = decisions_json.get("decisions")
    if not isinstance(dec, dict):
        _fail("multibody_decisions.json missing top-level decisions{}")

    if len(dec) > MAX_DECISIONS:
        _fail("Too many decisions (guardrail hit)")

    out: Dict[str, Dict[str, Any]] = {}
    for k in dec.keys():
        ks = str(k).strip()
        if not ks:
            continue
        v = dec.get(k)
        if not isinstance(v, dict):
            continue
        d = str(v.get("decision", "")).strip().lower()
        if d not in ALLOWED_DECISIONS:
            continue
        out[ks] = {
            "decision": d,
            "note": str(v.get("note", "") or ""),
            "updated_utc": str(v.get("updated_utc", "") or ""),
        }
    return out


def _index_step5_rows(exploded_parts: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    items = exploded_parts.get("items")
    if not isinstance(items, list):
        _fail("exploded_parts.json missing items[] array")

    if len(items) > MAX_STEP5_ROWS:
        _fail("exploded_parts.json too large (guardrail hit)")

    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        p = str(row.get("parent_def_sig", "")).strip()
        if not p:
            continue
        by_parent.setdefault(p, []).append(row)

    return by_parent


def _build_exploded_by_parent(
    decisions: Dict[str, Dict[str, Any]],
    step5_by_parent: Dict[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], List[str]]:
    """
    Builds:
      exploded_by_parent[parent_def_sig] = { parent_qty_total, subparts:[...] }
      parent_qty_total_by_sig[parent_def_sig] = qty_total (int)
    """
    warnings: List[str] = []
    exploded_by_parent: Dict[str, Dict[str, Any]] = {}
    parent_qty_total_by_sig: Dict[str, int] = {}

    for parent_sig in sorted(decisions.keys()):
        ent = decisions[parent_sig]
        if ent["decision"] != DECISION_EXPLODE:
            continue

        rows = step5_by_parent.get(parent_sig, [])
        if not rows:
            warnings.append(f"explode decision set but no Step5 rows for parent_def_sig={parent_sig}")
            exploded_by_parent[parent_sig] = {
                "decision": DECISION_EXPLODE,
                "note": ent["note"],
                "updated_utc": ent["updated_utc"],
                "parent_qty_total": 0,
                "subparts": [],
            }
            parent_qty_total_by_sig[parent_sig] = 0
            continue

        if len(rows) > MAX_SUBPARTS_PER_PARENT:
            _fail(f"Too many subparts for parent {parent_sig} (guardrail hit)")

        # Step5 includes parent_qty_total per row; we take max for safety
        pqty = 0
        for r in rows:
            try:
                pqty = max(pqty, int(r.get("parent_qty_total", 0) or 0))
            except Exception:
                pass
        parent_qty_total_by_sig[parent_sig] = int(pqty)

        subparts: List[Dict[str, Any]] = []
        for r in rows:
            sub_id = str(r.get("subpart_id", "")).strip()
            sub_sig = str(r.get("subpart_sig", "")).strip()
            if not sub_id or not sub_sig:
                continue

            sub_sig_free = r.get("subpart_sig_free")
            ssf = str(sub_sig_free).strip() if sub_sig_free is not None else None
            if ssf == "":
                ssf = None

            bbox = _as_list3(r.get("bbox_mm"))
            vol = r.get("volume_mm3")
            vol_f: Optional[float] = None
            if vol is not None:
                try:
                    vol_f = float(vol)
                except Exception:
                    vol_f = None

            stl_rel = _norm_exploded_stl_path(str(r.get("stl_path", "") or ""))

            subparts.append({
                "subpart_index": 0,  # assigned after sort
                "subpart_id": sub_id,
                "subpart_sig": sub_sig,
                "subpart_sig_free": ssf,
                "bbox_mm": bbox,
                "volume_mm3": vol_f,
                "stl_path": stl_rel,
                "note": str(r.get("note", "") or ""),
            })

        # Deterministic ordering
        subparts.sort(key=lambda sp: (str(sp.get("subpart_sig", "")), str(sp.get("subpart_id", ""))))
        for i, sp in enumerate(subparts):
            sp["subpart_index"] = i

        exploded_by_parent[parent_sig] = {
            "decision": DECISION_EXPLODE,
            "note": ent["note"],
            "updated_utc": ent["updated_utc"],
            "parent_qty_total": int(pqty),
            "subparts": subparts,
        }

    return exploded_by_parent, parent_qty_total_by_sig, warnings


def _build_subpart_definitions(
    exploded_by_parent: Dict[str, Dict[str, Any]],
    parent_qty_total_by_sig: Dict[str, int]
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Creates:
      subpart_definitions[subpart_sig] = {
        representative_stl_path, bbox_mm, volume_mm3,
        qty_total, where_used_parents:[...],
        subpart_sig_free
      }
    qty_total rule: for each parent instance, each subpart appears once
      => qty_total += parent_qty_total
    """
    subdefs: Dict[str, Dict[str, Any]] = {}
    subpart_instances_total = 0

    for parent_sig in sorted(exploded_by_parent.keys()):
        parent = exploded_by_parent[parent_sig]
        pqty = int(parent_qty_total_by_sig.get(parent_sig, 0))
        subs = parent.get("subparts", [])
        if not isinstance(subs, list):
            continue

        for sp in subs:
            if not isinstance(sp, dict):
                continue
            sub_sig = str(sp.get("subpart_sig", "")).strip()
            if not sub_sig:
                continue

            subpart_instances_total += pqty
            if subpart_instances_total < 0:
                _fail("integer overflow guard (unexpected)")

            ent = subdefs.get(sub_sig)
            if ent is None:
                if len(subdefs) >= MAX_UNIQUE_SUBPARTS:
                    _fail("Too many unique subparts (guardrail hit)")
                ent = {
                    "subpart_sig": sub_sig,
                    "subpart_sig_free": sp.get("subpart_sig_free"),
                    "representative_stl_path": sp.get("stl_path"),
                    "bbox_mm": sp.get("bbox_mm"),
                    "volume_mm3": sp.get("volume_mm3"),
                    "qty_total": 0,
                    "where_used_parents": [],
                }
                subdefs[sub_sig] = ent

            ent["qty_total"] = int(ent.get("qty_total", 0) or 0) + pqty

            w = ent.get("where_used_parents")
            if not isinstance(w, list):
                w = []
                ent["where_used_parents"] = w
            # deterministic unique set later; for now append
            w.append(parent_sig)

            # prefer first representative deterministically
            # (we’re iterating parents sorted, and subparts sorted within parent)
            if not ent.get("representative_stl_path"):
                ent["representative_stl_path"] = sp.get("stl_path")

    # normalize where-used to sorted unique lists
    for k in sorted(subdefs.keys()):
        w = subdefs[k].get("where_used_parents", [])
        if isinstance(w, list):
            subdefs[k]["where_used_parents"] = sorted(set(str(x) for x in w if str(x).strip()))

    return subdefs, subpart_instances_total


def _build_active_xcaf(
    xcaf_instances: Dict[str, Any],
    decisions: Dict[str, Dict[str, Any]],
    exploded_by_parent: Dict[str, Dict[str, Any]],
    subpart_definitions: Dict[str, Dict[str, Any]],
    counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_XCAF_ACTIVE,
        "created_utc": _utc_now_iso(),
        "counts": counts,
        # preserve original truth snapshot for traceability
        "xcaf_instances": xcaf_instances,
        "derived": {
            "explosion_decisions": decisions,  # only stable keyed
            "exploded_by_parent": exploded_by_parent,
            "subpart_definitions": subpart_definitions,
        },
    }


def _build_active_manifest(
    assets_manifest: Dict[str, Any],
    exploded_by_parent: Dict[str, Dict[str, Any]],
    subpart_definitions: Dict[str, Dict[str, Any]],
    counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_MANIFEST_ACTIVE,
        "created_utc": _utc_now_iso(),
        "counts": counts,
        "assets_manifest": assets_manifest,
        "derived": {
            "exploded_by_parent": exploded_by_parent,
            "subpart_definitions": subpart_definitions,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out", help="Output root (/out in container)")

    ap.add_argument("--xcaf-in", default="/out/xcaf_instances.json")
    ap.add_argument("--manifest-in", default="/out/assets_manifest.json")
    ap.add_argument("--decisions-in", default="/out/review/multibody_decisions.json")
    ap.add_argument("--exploded-in", default="/out/exploded/exploded_parts.json")

    ap.add_argument("--xcaf-out", default="/out/xcaf_instances_active.json")
    ap.add_argument("--manifest-out", default="/out/assets_manifest_active.json")

    ns = ap.parse_args()
    out_root = Path(ns.out)

    # Fail fast on required inputs
    xcaf = _read_json(Path(ns.xcaf_in))
    manifest = _read_json(Path(ns.manifest_in))
    decisions_json = _read_json(Path(ns.decisions_in))
    exploded = _read_json(Path(ns.exploded_in))

    decisions = _load_decisions(decisions_json)
    step5_by_parent = _index_step5_rows(exploded)
    def_name_by_sig = _build_def_name_by_sig(xcaf)
    parent_name_by_sig = _build_parent_name_by_sig(step5_by_parent)


    exploded_by_parent, parent_qty_total_by_sig, warnings = _build_exploded_by_parent(decisions, step5_by_parent)
    subdefs, subpart_instances_total = _build_subpart_definitions(exploded_by_parent, parent_qty_total_by_sig)

    counts = {
        "parents_explode_decisions": int(sum(1 for k in decisions.keys() if decisions[k]["decision"] == DECISION_EXPLODE)),
        "parents_with_step5_rows": int(sum(1 for k in exploded_by_parent.keys() if len(exploded_by_parent[k].get("subparts", [])) > 0)),
        "subpart_instances_total": int(subpart_instances_total),
        "subpart_unique_total": int(len(subdefs)),
    }

    xcaf_active = _build_active_xcaf(xcaf, decisions, exploded_by_parent, subdefs, counts)
    manifest_active = _build_active_manifest(manifest, exploded_by_parent, subdefs, counts)

    # ------------------------------------------------------------
    # Enrich ACTIVE manifest with human-readable names for UI
    # ------------------------------------------------------------
    # Base items (assets_manifest.items[])
    base = manifest_active.get("assets_manifest")
    if isinstance(base, dict):
        items = base.get("items")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                sig_used = str(it.get("def_sig_used") or "").strip()
                nm = def_name_by_sig.get(sig_used, "")
                if nm:
                    it["display_name"] = nm
                    it["display_name_source"] = "step1_definition_name"
                else:
                    # fallback: occurrence label or ref_def (avoid empty)
                    ref_def = str(it.get("ref_def") or "").strip()
                    it["display_name"] = ref_def if ref_def else f"def:{sig_used[:12]}" if sig_used else "part"
                    it["display_name_source"] = "fallback_ref_def_or_sig"

    # Exploded parents + subparts (derived.*)
    der = manifest_active.get("derived")
    if isinstance(der, dict):
        exploded_by_parent = der.get("exploded_by_parent")
        if isinstance(exploded_by_parent, dict):
            for parent_sig in sorted(exploded_by_parent.keys()):
                p = exploded_by_parent.get(parent_sig)
                if not isinstance(p, dict):
                    continue
                # attach parent name (prefer Step5 parent_name, else Step1 name via def_sig map)
                pn = parent_name_by_sig.get(parent_sig, "") or def_name_by_sig.get(parent_sig, "")
                if pn:
                    p["parent_name"] = pn

        subdefs = der.get("subpart_definitions")
        if isinstance(subdefs, dict):
            for sub_sig in sorted(subdefs.keys()):
                ent = subdefs.get(sub_sig)
                if not isinstance(ent, dict):
                    continue
                where_used = ent.get("where_used_parents", [])
                if not isinstance(where_used, list):
                    where_used = []
                disp, rep_parent_sig, rep_parent_name = _pick_display_name_for_subpart(
                    sub_sig=sub_sig,
                    where_used=[str(x) for x in where_used if str(x).strip()],
                    parent_name_by_sig=parent_name_by_sig
                )
                ent["display_name"] = disp
                ent["display_name_source"] = "parent_name_plus_subpart_sig"
                if rep_parent_sig:
                    ent["representative_parent_def_sig"] = rep_parent_sig
                if rep_parent_name:
                    ent["representative_parent_name"] = rep_parent_name


    xcaf_out = Path(ns.xcaf_out)
    man_out = Path(ns.manifest_out)

    _write_json_atomic(xcaf_out, xcaf_active)
    _write_json_atomic(man_out, manifest_active)

    # Append-only log
    log_path = out_root / LOG_REL
    log_rec = {
        "ts_utc": _utc_now_iso(),
        "schema": "step6_merge_log_v1",
        "xcaf_in": str(ns.xcaf_in),
        "manifest_in": str(ns.manifest_in),
        "decisions_in": str(ns.decisions_in),
        "exploded_in": str(ns.exploded_in),
        "xcaf_out": str(ns.xcaf_out),
        "manifest_out": str(ns.manifest_out),
        "counts": counts,
        "warnings": warnings[:2000],  # bounded
    }
    _append_jsonl(log_path, log_rec)

    print("[step6] OK")
    print(f"[step6] wrote: {xcaf_out}")
    print(f"[step6] wrote: {man_out}")
    print(f"[step6] log  : {log_path}")
    if warnings:
        print(f"[step6] warnings: {len(warnings)} (see log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
