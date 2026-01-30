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

from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs



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
            "STEPControl_Writer": STEPControl_Writer,
            "STEPControl_AsIs": STEPControl_AsIs,
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

def _write_step(shape, out_path: Path) -> None:
    STEPControl_Writer = OCC["STEPControl_Writer"]
    STEPControl_AsIs = OCC["STEPControl_AsIs"]
    IFSelect_RetDone = OCC.get("IFSelect_RetDone", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = STEPControl_Writer()
    call_maybe_s(w, "Transfer", shape, STEPControl_AsIs)
    status = call_maybe_s(w, "Write", str(out_path))

    # Some builds return bool, some IFSelect code, some None. Only explicit failure is fatal.
    if status is False:
        raise RuntimeError(f"Failed to write STEP: {out_path}")
    if IFSelect_RetDone is not None and isinstance(status, int) and status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP (status={status}): {out_path}")


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

def _load_explosion_plan_json(plan_json: Path) -> List[Dict[str, str]]:
    """
    Supports BOTH formats:

    A) list format:
      {"items":[{"ref_def_sig":"...", "ref_def_id":"...", "explode":true, "note":""}, ...]}

    B) dict format (your UI):
      {"items": {"<def_sig>": {"def_sig":"...", "note":"", ...}, ...}}

    Returns rows: [{"ref_def_sig":..., "ref_def_id":..., "note":...}, ...]
    """
    _require_file(plan_json, "explode_plan.json")
    data = _read_json(plan_json)

    items = data.get("items")
    rows: List[Dict[str, str]] = []

    # Format B: dict keyed by def_sig
    if isinstance(items, dict):
        for k, v in items.items():
            if not isinstance(v, dict):
                continue
            ds = str(v.get("def_sig") or k or "").strip()
            if not ds:
                continue
            rows.append({
                "ref_def_sig": ds,
                "ref_def_id": "",
                "note": str(v.get("note") or "").strip(),
            })
        rows.sort(key=lambda r: (r["ref_def_sig"], r["ref_def_id"], r["note"]))
        return rows

    # Format A: list
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            explode = it.get("explode", True)
            if isinstance(explode, str):
                explode = explode.strip().lower() in ("1", "true", "yes", "y", "on")
            if explode is False:
                continue

            rows.append({
                "ref_def_sig": str(it.get("ref_def_sig") or "").strip(),
                "ref_def_id": str(it.get("ref_def_id") or "").strip(),
                "note": str(it.get("note") or "").strip(),
            })

        rows = [r for r in rows if (r["ref_def_sig"] or r["ref_def_id"])]
        rows.sort(key=lambda r: (r["ref_def_sig"], r["ref_def_id"], r["note"]))
        return rows

    raise RuntimeError(f"{plan_json} must contain 'items' as a dict or a list.")


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

def run_explosion_worker_for_run_dir(*, run_dir: Path, runs_dir: Path) -> int:
    """
    Run-based exploder:
      Reads: run_dir/<EXPLODE_PLAN_REL>
      Reads: run_dir/xcaf_instances.json (for def_id->def_sig fallback)
      Resolves STEP path from run_dir/run_status.json["step_path"] OR run_dir/input.step
      Writes:
        run_dir/exploded_step/<parent_def_sig>/<n>.step
        run_dir/exploded_stl/<parent_def_sig>/<n>.stl
        run_dir/exploded_manifest.json
        run_dir/explode_worker_log.jsonl

    Incremental behavior:
      - If a parent_def_sig already exists in exploded_manifest.json AND all referenced STL files still exist,
        we skip reprocessing that parent even if the plan file changes.
    """
    run_dir = run_dir.resolve()

    plan_name = os.getenv("EXPLODE_PLAN_REL", "explode_plan.json").strip().strip('"').strip("'")
    plan_path = run_dir / plan_name
    xcaf_instances_json = run_dir / "xcaf_instances.json"
    status_path = run_dir / os.getenv("RUN_STATUS_FILENAME", "run_status.json")

    _require_file(plan_path, plan_name)
    _require_file(xcaf_instances_json, "xcaf_instances.json")

    meta = _read_json(xcaf_instances_json)

    # def_id -> def_sig fallback
    def_id_to_sig: Dict[str, str] = {}
    defs = meta.get("definitions")
    if isinstance(defs, dict):
        for def_id, d in defs.items():
            if isinstance(d, dict):
                ds = str(d.get("def_sig") or "").strip()
                if ds:
                    def_id_to_sig[str(def_id)] = ds

    # STEP path
    step_path: Optional[Path] = None
    if status_path.is_file():
        try:
            st = _read_json(status_path)
            p = str(st.get("step_path") or "").strip()
            if p:
                step_path = Path(p)
        except Exception:
            step_path = None
    if step_path is None:
        guess = run_dir / "input.step"
        if guess.is_file():
            step_path = guess
    if step_path is None:
        raise FileNotFoundError(
            f"Missing STEP path. Provide {status_path} with step_path, or run_dir/input.step"
        )

    exploded_step_root = run_dir / os.getenv("EXPLODED_STEP_DIRNAME", "exploded_step")
    exploded_stl_root = run_dir / os.getenv("EXPLODED_STL_DIRNAME", "exploded_stl")
    exploded_step_root.mkdir(parents=True, exist_ok=True)
    exploded_stl_root.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / os.getenv("WORKER_LOG_FILENAME", "explode_worker_log.jsonl")
    manifest_path = run_dir / os.getenv("EXPLODED_MANIFEST_FILENAME", "exploded_manifest.json")

    # Load existing manifest (for incremental skip + merge)
    existing_exploded: Dict[str, Any] = {}
    existing_errors: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            m = _read_json(manifest_path)
            if isinstance(m, dict):
                ex = m.get("exploded")
                er = m.get("errors")
                if isinstance(ex, dict):
                    existing_exploded = ex
                if isinstance(er, dict):
                    existing_errors = er
        except Exception:
            existing_exploded = {}
            existing_errors = {}

    # Plan items
    plan_rows = _load_explosion_plan_json(plan_path)
    if not plan_rows:
        print("[explode] No items marked for explosion.")
        return 0

    # Resolve planned keys to parent def_sig
    parent_sigs: List[Tuple[str, str]] = []
    for r in plan_rows:
        ref_sig = str(r.get("ref_def_sig") or "").strip()
        ref_id = str(r.get("ref_def_id") or "").strip()
        note = str(r.get("note") or "").strip()

        if ref_sig:
            parent_sigs.append((ref_sig, note))
        elif ref_id and ref_id in def_id_to_sig:
            parent_sigs.append((def_id_to_sig[ref_id], note))

    # Deterministic ordering
    parent_sigs.sort(key=lambda t: (t[0], t[1]))

    print(f"[explode] Reading STEP: {step_path}")
    doc, shape_tool = _load_step_to_xcaf(step_path)

    sig = SignatureAdapter()
    print(f"[explode] Signature algo: {sig.mode()}")

    simple_labels = _iter_simple_shapes(shape_tool)
    print(f"[explode] Simple shapes found: {len(simple_labels)}")

    # Build lookup: def_sig -> shape
    sig_to_shape: Dict[str, Any] = {}
    for (_nm, lbl) in simple_labels:
        shp = call_maybe_s(shape_tool, "GetShape", lbl)
        if shp is None:
            continue
        ds = sig.def_sig(shp)
        if ds not in sig_to_shape:
            sig_to_shape[ds] = shp

    exploded_new: Dict[str, List[Dict[str, Any]]] = {}
    errors_new: Dict[str, List[str]] = {}

    def _already_done(parent_def_sig: str) -> bool:
        """
        True if parent_def_sig exists in existing_exploded AND all referenced STL files still exist.
        """
        recs = existing_exploded.get(parent_def_sig)
        if not isinstance(recs, list) or not recs:
            return False
        for r in recs:
            if not isinstance(r, dict):
                return False
            rel = str(r.get("stl_url") or "").strip()
            if not rel:
                return False
            if not (run_dir / rel).is_file():
                return False
        return True

    for (parent_def_sig, note) in parent_sigs:
        # Incremental skip: if already exploded and outputs exist, do nothing
        if _already_done(parent_def_sig):
            _append_log(
                log_path,
                {
                    "ts_utc": _utc_now_iso(),
                    "event": "parent_skip_already_done",
                    "parent_def_sig": parent_def_sig,
                    "note": (note or None),
                },
            )
            continue

        per_err: List[str] = []
        parent_shape = sig_to_shape.get(parent_def_sig)

        if parent_shape is None:
            per_err.append("resolve_failed: parent def_sig not found in STEP read")
            errors_new[parent_def_sig] = per_err
            _append_log(
                log_path,
                {
                    "ts_utc": _utc_now_iso(),
                    "event": "parent_failed",
                    "parent_def_sig": parent_def_sig,
                    "note": (note or None),
                    "errors": per_err,
                },
            )
            continue

        solids = _extract_solids(parent_shape)
        if len(solids) <= 1:
            _append_log(
                log_path,
                {
                    "ts_utc": _utc_now_iso(),
                    "event": "parent_skipped",
                    "parent_def_sig": parent_def_sig,
                    "note": "single_or_empty",
                },
            )
            continue

        # Build deterministic sorted list
        tmp: List[Dict[str, Any]] = []
        for sidx, solid in enumerate(solids):
            try:
                sp_sig = sig.subpart_sig(solid)
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
                        "sidx": int(sidx),
                        "subpart_sig": str(sp_sig),
                        "bbox_mm": [float(dx), float(dy), float(dz)],
                        "vol": (None if vol is None else float(vol)),
                        "sort_key": key,
                    }
                )
            except Exception as e:
                per_err.append(f"subpart_build_failed idx={sidx}: {type(e).__name__}: {e}")

        tmp.sort(key=lambda d: d["sort_key"])

        out_step_dir = exploded_step_root / parent_def_sig
        out_stl_dir = exploded_stl_root / parent_def_sig
        out_step_dir.mkdir(parents=True, exist_ok=True)
        out_stl_dir.mkdir(parents=True, exist_ok=True)

        recs: List[Dict[str, Any]] = []
        for n, rec in enumerate(tmp):
            step_rel = f"{exploded_step_root.name}/{parent_def_sig}/{n}.step"
            stl_rel = f"{exploded_stl_root.name}/{parent_def_sig}/{n}.stl"
            step_abs = run_dir / step_rel
            stl_abs = run_dir / stl_rel

            note_out = ""
            try:
                if not step_abs.is_file():
                    _write_step(rec["solid"], step_abs)
            except Exception as e:
                note_out = f"step_write_failed:{type(e).__name__}"
                per_err.append(f"STEP write failed n={n}: {type(e).__name__}: {e}")

            try:
                if not stl_abs.is_file():
                    _write_stl(rec["solid"], stl_abs)
            except Exception as e:
                note_out = (note_out + "; " if note_out else "") + f"stl_write_failed:{type(e).__name__}"
                per_err.append(f"STL write failed n={n}: {type(e).__name__}: {e}")

            recs.append(
                {
                    "subpart_sig": rec["subpart_sig"],
                    "stl_url": stl_rel.replace("\\", "/"),
                    "step_relpath": step_rel.replace("\\", "/"),
                    "solid_index": int(rec["sidx"]),
                    "bbox": {"size": rec["bbox_mm"]},
                    "note": (note_out or None),
                }
            )

        exploded_new[parent_def_sig] = recs
        if per_err:
            errors_new[parent_def_sig] = per_err

        _append_log(
            log_path,
            {
                "ts_utc": _utc_now_iso(),
                "event": "parent_done",
                "parent_def_sig": parent_def_sig,
                "n_solids_found": len(solids),
                "n_subparts_written": len(recs),
                "note": (note or None),
                "errors": per_err,
            },
        )

    # Merge manifests (keep existing parents; overwrite only parents processed this run)
    merged_exploded: Dict[str, Any] = {}
    merged_exploded.update(existing_exploded)
    merged_exploded.update(exploded_new)

    merged_errors: Dict[str, Any] = {}
    merged_errors.update(existing_errors)
    merged_errors.update(errors_new)

    payload = {
        "schema": "exploded_manifest_v1",
        "created_utc": _utc_now_iso(),
        "signature_algo": sig.mode(),
        "exploded": merged_exploded,
        "errors": merged_errors,
    }
    _json_dump_atomic(manifest_path, payload)

    print(f"[explode] wrote: {manifest_path}")
    return 0


def _append_log(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="", help="Run folder id under RUNS_DIR (run-based explode)")
    ap.add_argument("--runs_dir", default=os.getenv("RUNS_DIR", "/app/ui_runs"))

    # legacy step5 args (keep)
    ap.add_argument("--step_path", default="", help="Legacy: STEP path (inside container)")
    ap.add_argument("--out_dir", default="/out", help="Legacy: pipeline out directory")

    ns = ap.parse_args()

    if str(ns.run_id).strip():
        runs_dir = Path(ns.runs_dir)
        run_dir = runs_dir / str(ns.run_id).strip()
        return run_explosion_worker_for_run_dir(run_dir=run_dir, runs_dir=runs_dir)


if __name__ == "__main__":
    raise SystemExit(main())
