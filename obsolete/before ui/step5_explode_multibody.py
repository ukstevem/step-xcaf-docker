#!/usr/bin/env python3
"""
Step 5: Explode Approved Multi-Body

- Reads:  out/review/multibody_decisions.json
- Reads:  out/xcaf_instances.json (best-effort metadata, qty/name lookup)
- Re-reads STEP via XCAF and resolves definition shapes by stable IDs (def_sig / def_sig_free)
- Explodes approved multi-body definitions into subparts (solids)
- Writes: out/exploded/stl/<parent_def_sig>/<subpart_id>.stl
- Writes: out/exploded/exploded_parts.json
- Appends: out/exploded/explode_log.jsonl

Power-of-10 style:
- deterministic ordering
- explicit constants
- bounded loops with guardrails
- clear errors
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Explicit constants / limits
# -----------------------------

MAX_APPROVED_PARENTS = 20000
MAX_SHAPES_SCANNED = 500000
MAX_SUBPARTS_PER_PARENT = 200000

# Guardrail for in-memory JSON accumulation
MAX_TOTAL_SUBPART_ROWS = 2000000

BBOX_ROUND_MM = 3          # for deterministic sort key
VOL_ROUND_MM3 = 1          # for deterministic sort key
MESH_DEFLECTION_MM = 0.2   # STL triangulation quality (tweak later if needed)


# -----------------------------
# OCC / OCP imports (inside docker)
# -----------------------------

def _import_occ():
    try:
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.TDF import TDF_LabelSequence
        from OCP.TDataStd import TDataStd_Name
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopoDS import TopoDS_Shape
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer
        return {
            "STEPCAFControl_Reader": STEPCAFControl_Reader,
            "IFSelect_RetDone": IFSelect_RetDone,
            "XCAFApp_Application": XCAFApp_Application,
            "TDocStd_Document": TDocStd_Document,
            "XCAFDoc_DocumentTool": XCAFDoc_DocumentTool,
            "TDF_LabelSequence": TDF_LabelSequence,
            "TDataStd_Name": TDataStd_Name,
            "TCollection_ExtendedString": TCollection_ExtendedString,
            "TopExp_Explorer": TopExp_Explorer,
            "TopAbs_SOLID": TopAbs_SOLID,
            "TopoDS_Shape": TopoDS_Shape,
            "Bnd_Box": Bnd_Box,
            "BRepBndLib": BRepBndLib,
            "GProp_GProps": GProp_GProps,
            "BRepGProp": BRepGProp,  # <-- canonical key (fixes blank volume)
            "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
            "StlAPI_Writer": StlAPI_Writer,
        }
    except Exception as e:
        raise RuntimeError(
            "OCP imports failed. Step 5 must run inside the same docker image as Steps 1-4.\n"
            f"Import error: {type(e).__name__}: {e}"
        )


OCC = _import_occ()


# -----------------------------
# Small utilities
# -----------------------------

def call_maybe_s(obj, method: str, *args):
    fn = getattr(obj, method, None)
    if callable(fn):
        try:
            return fn(*args)
        except TypeError:
            pass
    fn_s = getattr(obj, method + "_s", None)
    if callable(fn_s):
        return fn_s(*args)
    raise AttributeError(f"{type(obj).__name__} has no usable {method} / {method}_s")


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _require_file(p: Path, label: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing required {label}: {p}")


def _safe_int(v: Any, default: int = 1) -> int:
    try:
        i = int(v)
        return i if i >= 0 else default
    except Exception:
        return default


# -----------------------------
# JSON Helpers
# -----------------------------

def _read_json(path: Path) -> dict:
    _require_file(path, "JSON")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump_atomic(path: Path, obj: Any) -> None:
    """
    Atomic-ish write: write to temp then replace.
    Deterministic JSON (sort_keys=True).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sort_exploded_rows(rows: List[Dict[str, Any]]) -> None:
    """
    Deterministic ordering: parent_def_sig, then subpart_index, then subpart_id.
    """
    rows.sort(
        key=lambda r: (
            str(r.get("parent_def_sig", "")),
            int(r.get("subpart_index", 0)),
            str(r.get("subpart_id", "")),
        )
    )


# -----------------------------
# Signature adapter
# -----------------------------

class SignatureAdapter:
    """
    Uses the project's canonical signature functions from brep_signature.py:

        from brep_signature import DEF_SIG_ALGO, compute_def_sig, compute_def_sig_free

    This MUST match Step 1/2 behavior.
    """

    def __init__(self):
        try:
            from brep_signature import DEF_SIG_ALGO, compute_def_sig, compute_def_sig_free
        except Exception as e:
            raise RuntimeError(
                "Failed to import required signature API from brep_signature.py.\n"
                "Step 5 must run inside the repo-mounted container with /app on PYTHONPATH.\n"
                "Required exports:\n"
                "  DEF_SIG_ALGO\n"
                "  compute_def_sig(shape)\n"
                "  compute_def_sig_free(shape)\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        self._algo = str(DEF_SIG_ALGO)
        self._def_sig = compute_def_sig
        self._def_sig_free = compute_def_sig_free

    def mode(self) -> str:
        return self._algo

    def def_sig(self, shape) -> str:
        return str(self._def_sig(shape))

    def def_sig_free(self, shape) -> str:
        return str(self._def_sig_free(shape))

    def subpart_sig(self, shape) -> str:
        return str(self._def_sig(shape))

    def subpart_sig_free(self, shape) -> str:
        return str(self._def_sig_free(shape))


# -----------------------------
# XCAF reader + shape enumeration
# -----------------------------

def _load_step_to_xcaf(step_path: Path):
    STEPCAFControl_Reader = OCC["STEPCAFControl_Reader"]
    XCAFApp_Application = OCC["XCAFApp_Application"]
    TDocStd_Document = OCC["TDocStd_Document"]
    XCAFDoc_DocumentTool = OCC["XCAFDoc_DocumentTool"]

    fmt = OCC["TCollection_ExtendedString"]("MDTV-XCAF")
    app = call_maybe_s(XCAFApp_Application, "GetApplication")
    doc = TDocStd_Document(fmt)
    call_maybe_s(app, "NewDocument", fmt, doc)

    r = STEPCAFControl_Reader()
    ret = call_maybe_s(r, "ReadFile", str(step_path))

    IFSelect_RetDone = OCC.get("IFSelect_RetDone", None)
    if isinstance(ret, bool):
        if not ret:
            raise RuntimeError(f"STEP read failed: {step_path}")
    else:
        if IFSelect_RetDone is not None and ret != IFSelect_RetDone:
            raise RuntimeError(f"STEP read failed (status={ret}): {step_path}")

    ok = call_maybe_s(r, "Transfer", doc)
    if isinstance(ok, bool) and (not ok):
        raise RuntimeError(f"STEPCAF transfer failed: {step_path}")

    shape_tool = call_maybe_s(XCAFDoc_DocumentTool, "ShapeTool", doc.Main())
    return doc, shape_tool


def _label_name(label) -> str:
    try:
        TDataStd_Name = OCC["TDataStd_Name"]
        name_attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID(), name_attr):
            return str(name_attr.Get()).strip()
    except Exception:
        pass
    return ""


def _iter_simple_shapes(shape_tool) -> List[Tuple[str, Any]]:
    TDF_LabelSequence = OCC["TDF_LabelSequence"]

    seq = TDF_LabelSequence()
    call_maybe_s(shape_tool, "GetShapes", seq)

    n = seq.Length()
    out: List[Tuple[str, Any]] = []

    limit = min(n, MAX_SHAPES_SCANNED)
    for i in range(1, limit + 1):
        lbl = seq.Value(i)
        try:
            if bool(call_maybe_s(shape_tool, "IsSimpleShape", lbl)):
                out.append((_label_name(lbl), lbl))
        except Exception:
            continue

    if n > limit:
        raise RuntimeError(
            f"Shape scan exceeded MAX_SHAPES_SCANNED={MAX_SHAPES_SCANNED}. "
            f"Document had {n} shapes; increase the limit deliberately if needed."
        )

    return out


# -----------------------------
# Explode primitives
# -----------------------------

def _extract_solids(shape) -> List[Any]:
    TopExp_Explorer = OCC["TopExp_Explorer"]
    TopAbs_SOLID = OCC["TopAbs_SOLID"]

    solids: List[Any] = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()
    return solids


def _bbox_dims_mm(shape) -> Tuple[float, float, float]:
    Bnd_Box = OCC["Bnd_Box"]
    BRepBndLib = OCC["BRepBndLib"]

    box = Bnd_Box()
    call_maybe_s(BRepBndLib, "Add", shape, box)

    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    dz = float(zmax - zmin)
    return dx, dy, dz


def _volume_mm3(shape) -> Optional[float]:
    try:
        GProp_GProps = OCC["GProp_GProps"]
        BRepGProp = OCC["BRepGProp"]  # class
        props = GProp_GProps()
        call_maybe_s(BRepGProp, "VolumeProperties", shape, props)
        v = float(props.Mass())
        if v >= 0.0:
            return v
    except Exception:
        pass
    return None


def _write_stl(shape, out_path: Path) -> None:
    BRepMesh_IncrementalMesh = OCC["BRepMesh_IncrementalMesh"]
    StlAPI_Writer = OCC["StlAPI_Writer"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    BRepMesh_IncrementalMesh(shape, float(MESH_DEFLECTION_MM))

    wr = StlAPI_Writer()
    res = wr.Write(shape, str(out_path))

    # Some OCP builds return None. Treat only explicit False as failure.
    if res is False:
        raise RuntimeError(f"Failed to write STL: {out_path}")


# -----------------------------
# Decisions + metadata readers
# -----------------------------

def _load_decisions_json(decisions_json: Path) -> List[Dict[str, str]]:
    """
    Expected format:
    {
      "decisions": {
        "<def_sig>": {"decision": "explode|defer|...", "note": "...", "updated_utc": "..."},
        ...
      }
    }

    Returns a list of dicts: [{"def_sig":..., "decision":..., "note":...}, ...]
    Deterministic ordering is handled later (we sort approved keys).
    """
    _require_file(decisions_json, "multibody_decisions.json")
    with open(decisions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    decisions = data.get("decisions")
    if not isinstance(decisions, dict):
        raise RuntimeError(
            f"{decisions_json} must contain an object at key 'decisions' mapping def_sig -> record."
        )

    rows: List[Dict[str, str]] = []
    for def_sig, rec in decisions.items():
        if not isinstance(def_sig, str) or def_sig.strip() == "":
            continue
        if not isinstance(rec, dict):
            continue

        decision = str(rec.get("decision") or "").strip()
        note = str(rec.get("note") or "").strip()

        rows.append({
            "def_sig": def_sig.strip(),
            "decision": decision,
            "note": note,
            "updated_utc": str(rec.get("updated_utc") or ""),
            })

    return rows

def _load_xcaf_instances_meta(xcaf_instances_json: Path) -> Dict[str, Any]:
    if not xcaf_instances_json.is_file():
        return {}
    try:
        return _read_json(xcaf_instances_json)
    except Exception:
        return {}


def _lookup_parent_info(meta: Dict[str, Any], parent_sig: str) -> Tuple[str, int, Optional[str]]:
    defs = meta.get("definitions")
    if isinstance(defs, dict):
        for _, d in defs.items():
            if not isinstance(d, dict):
                continue
            if str(d.get("def_sig", "")) == parent_sig:
                name = str(d.get("name") or d.get("def_name") or d.get("part_name") or "")
                qty = _safe_int(d.get("qty_total") or d.get("qty") or d.get("count") or 1, 1)
                sig_free = d.get("def_sig_free")
                return (name, qty, str(sig_free) if sig_free else None)

            if str(d.get("def_sig_free", "")) == parent_sig:
                name = str(d.get("name") or d.get("def_name") or d.get("part_name") or "")
                qty = _safe_int(d.get("qty_total") or d.get("qty") or d.get("count") or 1, 1)
                sig_free = d.get("def_sig_free")
                return (name, qty, str(sig_free) if sig_free else None)

    return ("", 1, None)


# -----------------------------
# Main explode logic
# -----------------------------

def run_step5(
    *,
    step_path: Path,
    out_dir: Path,
) -> int:
    out_dir = out_dir.resolve()
    decisions_json = out_dir / "review" / "multibody_decisions.json"
    xcaf_instances_json = out_dir / "xcaf_instances.json"

    _require_file(decisions_json, "out/review/multibody_decisions.json")

    exploded_root = out_dir / "exploded"
    stl_root = exploded_root / "stl"
    exploded_root.mkdir(parents=True, exist_ok=True)
    stl_root.mkdir(parents=True, exist_ok=True)

    parts_json_path = exploded_root / "exploded_parts.json"
    log_path = exploded_root / "explode_log.jsonl"

    decisions = _load_decisions_json(decisions_json)
    meta = _load_xcaf_instances_meta(xcaf_instances_json)

    approved = [r for r in decisions if r["decision"].strip().lower() == "explode"]

    if len(approved) > MAX_APPROVED_PARENTS:
        raise RuntimeError(
            f"Approved explode rows exceed MAX_APPROVED_PARENTS={MAX_APPROVED_PARENTS}: {len(approved)}"
        )

    if not approved:
        print("[step5] No rows with decision=explode. Nothing to do.")
        return 0

    approved_keys: List[Tuple[str, Optional[str], str]] = []
    for r in approved:
        approved_keys.append((r["def_sig"], None, ""))

    _require_file(step_path, "STEP file")
    print(f"[step5] Reading STEP: {step_path}")
    doc, shape_tool = _load_step_to_xcaf(step_path)

    sig = SignatureAdapter()
    print(f"[step5] Signature algo: {sig.mode()}")

    simple_labels = _iter_simple_shapes(shape_tool)
    print(f"[step5] Simple shapes found: {len(simple_labels)}")

    sig_to_shape: Dict[str, Any] = {}
    sigfree_to_shape: Dict[str, Any] = {}
    sig_to_name: Dict[str, str] = {}
    sigfree_to_name: Dict[str, str] = {}

    for (nm, lbl) in simple_labels:
        shp = call_maybe_s(shape_tool, "GetShape", lbl)
        if shp is None:
            continue

        ds = sig.def_sig(shp)
        df = sig.def_sig_free(shp)

        if ds not in sig_to_shape:
            sig_to_shape[ds] = shp
            sig_to_name[ds] = nm
        if df not in sigfree_to_shape:
            sigfree_to_shape[df] = shp
            sigfree_to_name[df] = nm

    approved_keys_sorted = sorted(approved_keys, key=lambda t: (t[0], t[1] or "", t[2] or ""))

    parts_rows: List[Dict[str, Any]] = []
    total_written = 0

    for (parent_key, parent_sig_free_hint, name_hint) in approved_keys_sorted:
        errors: List[str] = []
        parent_sig_full = None
        parent_sig_free = None

        parent_shape = sig_to_shape.get(parent_key)
        if parent_shape is not None:
            parent_sig_full = parent_key
            parent_sig_free = sig.def_sig_free(parent_shape)
        else:
            parent_shape = sigfree_to_shape.get(parent_key)
            if parent_shape is not None:
                parent_sig_free = parent_key
                parent_sig_full = sig.def_sig(parent_shape)

        if parent_shape is None:
            raise RuntimeError(
                f"[step5] Approved parent signature not found in STEP read.\n"
                f"Signature: {parent_key}\n"
                "This usually means Step 5 is not using the same signature algorithm as Step 1.\n"
                "Fix: ensure brep_signature.py is present/importable in /app inside docker."
            )

        meta_name, meta_qty, meta_sig_free = _lookup_parent_info(meta, parent_key)
        parent_name = (
            meta_name
            or name_hint
            or sig_to_name.get(parent_sig_full or "", "")
            or sigfree_to_name.get(parent_sig_free or "", "")
        )
        parent_qty_total = int(meta_qty) if meta_qty else 1

        parent_def_sig = str(parent_sig_full) if parent_sig_full else ""
        parent_def_sig_free = str(parent_sig_free or meta_sig_free or parent_sig_free_hint or "")

        solids = _extract_solids(parent_shape)
        n_solids = len(solids)

        out_parent_dir = stl_root / parent_def_sig

        if n_solids <= 1:
            reason = "no_solids_found" if n_solids == 0 else "single_solid_only"
            _append_log(
                log_path,
                {
                    "ts_utc": _utc_now_iso(),
                    "decision": "explode",
                    "parent_def_sig": parent_def_sig,
                    "parent_def_sig_free": parent_def_sig_free or None,
                    "parent_name": parent_name or None,
                    "parent_qty_total": parent_qty_total,
                    "n_solids_found": n_solids,
                    "n_subparts_written": 0,
                    "out_parent_dir": str(out_parent_dir),
                    "parts_json_path": str(parts_json_path),
                    "errors": [],
                    "note": reason,
                },
            )
            continue

        if n_solids > MAX_SUBPARTS_PER_PARENT:
            raise RuntimeError(
                f"[step5] Parent {parent_def_sig} solids exceed MAX_SUBPARTS_PER_PARENT={MAX_SUBPARTS_PER_PARENT}: {n_solids}"
            )

        tmp: List[Dict[str, Any]] = []
        for sidx, solid in enumerate(solids):
            try:
                sp_sig = sig.subpart_sig(solid)
                sp_sig_free = sig.subpart_sig_free(solid)
                dx, dy, dz = _bbox_dims_mm(solid)
                vol = _volume_mm3(solid)

                key = (
                    str(sp_sig),
                    round(dx, BBOX_ROUND_MM),
                    round(dy, BBOX_ROUND_MM),
                    round(dz, BBOX_ROUND_MM),
                    round(vol, VOL_ROUND_MM3) if vol is not None else -1.0,
                )

                tmp.append(
                    {
                        "solid": solid,
                        "subpart_sig": str(sp_sig),
                        "subpart_sig_free": str(sp_sig_free),
                        "bbox": (float(dx), float(dy), float(dz)),
                        "vol": float(vol) if vol is not None else None,
                        "sort_key": key,
                    }
                )
            except Exception as e:
                errors.append(f"subpart build failed idx={sidx}: {type(e).__name__}: {e}")

        tmp.sort(key=lambda d: d["sort_key"])

        per_sig_count: Dict[str, int] = {}
        written_here = 0

        # Only create output folder once we have something to write
        out_parent_dir.mkdir(parents=True, exist_ok=True)

        for subpart_index, rec in enumerate(tmp):
            sp_sig = rec["subpart_sig"]
            ordinal = per_sig_count.get(sp_sig, 0)
            per_sig_count[sp_sig] = ordinal + 1

            subpart_id = _sha256_hex(f"{parent_def_sig}:{sp_sig}:{ordinal}")
            stl_rel = Path("stl") / parent_def_sig / f"{subpart_id}.stl"
            stl_abs = exploded_root / stl_rel

            note = ""
            try:
                _write_stl(rec["solid"], stl_abs)
                written_here += 1
                total_written += 1
            except Exception as e:
                errors.append(f"STL write failed subpart_index={subpart_index}: {type(e).__name__}: {e}")
                note = f"stl_write_failed:{type(e).__name__}"

            dx, dy, dz = rec["bbox"]
            parts_rows.append(
                {
                    "parent_def_sig": parent_def_sig,
                    "parent_def_sig_free": parent_def_sig_free,
                    "parent_name": parent_name,
                    "parent_qty_total": int(parent_qty_total),
                    "subpart_index": int(subpart_index),
                    "subpart_id": subpart_id,
                    "subpart_sig": rec["subpart_sig"],
                    "subpart_sig_free": rec["subpart_sig_free"],
                    "bbox_mm": [float(dx), float(dy), float(dz)],
                    "volume_mm3": (None if rec["vol"] is None else float(rec["vol"])),
                    "stl_path": str(stl_rel).replace("\\", "/"),
                    "note": note,
                }
            )

            if len(parts_rows) > MAX_TOTAL_SUBPART_ROWS:
                raise RuntimeError(
                    f"[step5] exploded parts exceed MAX_TOTAL_SUBPART_ROWS={MAX_TOTAL_SUBPART_ROWS}. "
                    "Increase deliberately if needed."
                )

        _append_log(
            log_path,
            {
                "ts_utc": _utc_now_iso(),
                "decision": "explode",
                "parent_def_sig": parent_def_sig,
                "parent_def_sig_free": parent_def_sig_free or None,
                "parent_name": parent_name or None,
                "parent_qty_total": parent_qty_total,
                "n_solids_found": n_solids,
                "n_subparts_written": written_here,
                "out_parent_dir": str(out_parent_dir),
                "parts_json_path": str(parts_json_path),
                "errors": errors,
            },
        )

    _sort_exploded_rows(parts_rows)

    payload = {
        "schema": "step5_exploded_parts_v1",
        "created_utc": _utc_now_iso(),
        "count": len(parts_rows),
        "items": parts_rows,
    }
    _json_dump_atomic(parts_json_path, payload)

    print(f"[step5] Done. Subparts written: {total_written}")
    print(f"[step5] JSON: {parts_json_path} items={len(parts_rows)}")
    print(f"[step5] Log:  {log_path}")
    return 0


def _append_log(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_path", required=True, help="Path to the input STEP file inside container (e.g. /in/model.step)")
    ap.add_argument("--out_dir", default="/out", help="Pipeline out directory inside container (default: /out)")
    ns = ap.parse_args()

    step_path = Path(ns.step_path)
    out_dir = Path(ns.out_dir)

    return run_step5(
        step_path=step_path,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
