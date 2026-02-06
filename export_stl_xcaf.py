#!/usr/bin/env python3
"""
Step 2: export_stl_xcaf.py

Deterministic across STEP rereads by matching definition shapes using Step 1 signatures.

Inputs:
  - STEP file (re-read): /in/<model>.step
  - /out/xcaf_instances.json (from Step 1)

Outputs:
  - /out/stl/<part_id>.stl
  - /out/assets_manifest.json   (delta-only: mapping + statuses + settings)

Identity rules:
  - def_id (XCAF label entry) is debug-only and NOT used for matching.
  - Match shapes ONLY by def_sig (or def_sig_free if requested) using def_sig_algo.
  - No heuristics (no name/bbox matching).

Collision rules:
  - If Step 1 reports meta.signature_collisions or duplicates within JSON:
      strict_signature_collisions => fail
      otherwise => mark ambiguous and skip export for those defs
  - If reread STEP yields multiple labels with the same signature:
      strict => fail
      otherwise => pick deterministically, and record reread_label_duplicates_for_sig

Part-id rules (stable across Step 1 reruns):
  - part_index is always 0 for now.
  - part_id = sha1(def_sig_used + "|" + part_index)
    (No bbox dependency; the signature is the identity.)

Manifest rule:
  - Only set stl_path when an STL was actually produced (or already exists + matched).
    For unmatched/ambiguous items stl_path is null, so downstream steps can safely skip.
"""

from __future__ import annotations

import argparse
<<<<<<< HEAD
import datetime as _dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------
# OCP imports (expected in docker)
# -----------------------------
_OCC_FLAVOR = "unknown"
try:
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCP.TDF import TDF_LabelSequence, TDF_Tool
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Message import Message_ProgressRange

    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    from OCP.TopoDS import TopoDS_Shape

    _OCC_FLAVOR = "OCP"
except Exception as e:
    raise RuntimeError("This Step 2 script requires OCP in the container.") from e

# -----------------------------
# Shared signature (MUST match Step 1 exactly)
# -----------------------------
from brep_signature import DEF_SIG_ALGO, compute_def_sig, compute_def_sig_free


# -----------------------------
# Constants / guards
# -----------------------------
DEFAULT_LINEAR_DEFLECTION = 0.25
DEFAULT_ANGULAR_DEFLECTION = 0.35

MAX_SHAPED_DEFS_GUARD = 300000
MAX_LABELS_GUARD = 600000

# progress cadence
PROGRESS_LABEL_EVERY = 2000
PROGRESS_DEF_EVERY = 200
=======
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------
# OCP helpers
# -----------------------------
def _mod(name: str):
    return __import__(f"OCP.{name}", fromlist=[name])
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b


# -----------------------------
# OCP binding helper
# -----------------------------
def call_maybe_s(obj: Any, method: str, *args):
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


# -----------------------------
# Time / JSON helpers
# -----------------------------
def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
<<<<<<< HEAD
# XCAF document helpers
# -----------------------------
def _get_xcaf_app():
    if hasattr(XCAFApp_Application, "GetApplication"):
        return XCAFApp_Application.GetApplication()
    if hasattr(XCAFApp_Application, "GetApplication_s"):
        return XCAFApp_Application.GetApplication_s()
    return XCAFApp_Application()

=======
# Label/name helpers
# -----------------------------
def get_label_name(lab) -> str:
    """
    Best-effort label name using TDataStd_Name.
    Returns "" if not present.
    """
    TDataStd = _mod("TDataStd")
    name_attr = TDataStd.TDataStd_Name()

    getid = getattr(TDataStd.TDataStd_Name, "GetID", None) or getattr(
        TDataStd.TDataStd_Name, "GetID_s", None
    )
    if getid is None:
        return ""
    guid = getid()
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b

def _construct_doc(fmt: str):
    try:
<<<<<<< HEAD
        return TDocStd_Document(fmt)
    except Exception:
        pass
=======
        found = lab.FindAttribute(guid, name_attr)
    except AttributeError:
        found = lab.FindAttribute_s(guid, name_attr)
    if not found:
        return ""

    try:
        s = name_attr.Get()
        for m in ("ToExtString", "ToCString", "PrintToString"):
            fn = getattr(s, m, None)
            if callable(fn):
                return fn()
        return str(s)
    except Exception:
        return ""


def _label_entry(lab) -> str:
    """
    Stable-ish label entry string like "0:1:1:18"
    """
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b
    try:
        return TDocStd_Document(TCollection_ExtendedString(fmt))
    except Exception:
        pass
    return None


def _new_xcaf_document():
    app = _get_xcaf_app()
    formats = ("MDTV-CAF", "MDTV-XCAF", "XmlXCAF", "BinXCAF", "XmlOcaf", "BinOcaf")
    errs: List[str] = []
    for fmt in formats:
        doc = _construct_doc(fmt)
        if doc is None:
            errs.append(f"ctor failed: {fmt}")
            continue
        for arg in (fmt, TCollection_ExtendedString(fmt)):
            try:
                app.NewDocument(arg, doc)
            except Exception:
                pass
        try:
            _ = doc.Main()
            return doc
        except Exception:
            errs.append(f"doc.Main failed: {fmt}")
    raise RuntimeError("Failed to create XCAF document. " + "; ".join(errs))


def _get_shape_tool(doc):
    try:
        return XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    except Exception:
        return XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def _update_assemblies(shape_tool) -> None:
    try:
        call_maybe_s(shape_tool, "UpdateAssemblies")
    except Exception:
        return


def _load_step_into_doc(step_path: Path, doc) -> None:
    if not step_path.exists():
        raise FileNotFoundError(f"STEP not found: {step_path}")
    rdr = STEPCAFControl_Reader()
    pr = Message_ProgressRange()
    stat = rdr.ReadFile(str(step_path))
    if stat != IFSelect_RetDone:
        raise RuntimeError(f"STEP ReadFile failed: {step_path} (status={stat})")
    ok = False
    try:
        ok = bool(rdr.Transfer(doc, pr))
    except Exception:
        try:
            ok = bool(rdr.Transfer(doc))
        except Exception:
            ok = False
    if not ok:
        raise RuntimeError("STEP transfer failed")


def _get_all_shape_labels(shape_tool) -> List[Any]:
    seq = TDF_LabelSequence()
    try:
        call_maybe_s(shape_tool, "GetShapes", seq)
    except Exception:
        call_maybe_s(shape_tool, "GetFreeShapes", seq)
    n = int(seq.Length())
    labs: List[Any] = []
    for i in range(1, n + 1):
        labs.append(seq.Value(i))
    return labs


def _shape_from_label(shape_tool, lab) -> Optional[Any]:
    try:
        shp = call_maybe_s(shape_tool, "GetShape", lab)
        if shp is not None and hasattr(shp, "IsNull") and not shp.IsNull():
            return shp
    except Exception:
        pass

    out_shp = TopoDS_Shape()
    try:
        _ = call_maybe_s(shape_tool, "GetShape", lab, out_shp)
        if hasattr(out_shp, "IsNull") and not out_shp.IsNull():
            return out_shp
    except Exception:
        pass

    try:
        shp = call_maybe_s(shape_tool, "Shape", lab)
        if shp is not None and hasattr(shp, "IsNull") and not shp.IsNull():
            return shp
    except Exception:
        pass

    return None


<<<<<<< HEAD
def _label_entry_str(lab) -> str:
    """
    Deterministic ordering key for labels: their "entry" string.
    """
    a = TCollection_AsciiString()
    try:
        call_maybe_s(TDF_Tool, "Entry", lab, a)
    except Exception:
        # last resort: repr
        return str(lab)
    try:
        return a.ToCString()
    except Exception:
        try:
            return str(a)
        except Exception:
            return str(lab)


# -----------------------------
# Meshing / STL writing
# -----------------------------
def _mesh_shape(shape: Any, linear_deflection: float, angular_deflection: float) -> bool:
    """Mesh a shape before STL export.

    OCCT/OCP signatures vary a bit by version, so we:
      - try the full (lin, isRelative, ang, parallel) constructor first
      - fall back to the simple (lin) constructor
      - call Perform() if available
      - check IsDone() if available

    Returns True if meshing appears to have completed.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh

    lin = float(linear_deflection)
    ang = float(angular_deflection)

    try:
        m = BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    except TypeError:
        m = BRepMesh_IncrementalMesh(shape, lin)

    try:
        m.Perform()
    except Exception:
        # some bindings perform in ctor
        pass

    try:
        if hasattr(m, "IsDone") and (not m.IsDone()):
            return False
    except Exception:
        pass

    return True

def _write_stl(
    shape: Any,
    out_path: Path,
    ascii_stl: bool,
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> tuple[bool, str]:
    """Write an STL for `shape` to `out_path`.

    Returns (ok, message). On some OCP/OCCT builds StlAPI_Writer.Write()
    returns False even when it partially wrote a file; we therefore
    validate by checking file existence + non-trivial size.
    """
    if shape is None:
        return (False, "shape is None")

    try:
        if hasattr(shape, "IsNull") and shape.IsNull():
            return (False, "shape.IsNull()")
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)

    from OCP.StlAPI import StlAPI_Writer

    writer = StlAPI_Writer()
    try:
        writer.SetASCIIMode(bool(ascii_stl))
    except Exception:
        pass

    # Retry with finer meshing if needed. Some very small parts can fail
    # to triangulate at coarse deflection and cause Write() to return False.
    lin0 = max(1e-6, float(linear_deflection))
    ang0 = max(1e-9, float(angular_deflection))
    lin_try = [lin0, lin0 * 0.5, lin0 * 0.25]

    last_msg = ""
    for lin in lin_try:
        try:
            _mesh_shape(shape, lin, ang0)
        except Exception as e:
            last_msg = f"meshing failed: {e}"
            continue

        ok = False
        try:
            ok = bool(writer.Write(shape, str(out_path)))
        except Exception as e:
            last_msg = f"Write() exception: {e}"
            ok = False

        # Confirm by checking output file; avoids false negatives.
        try:
            if out_path.exists():
                sz = out_path.stat().st_size
                if sz > 84:  # STL header-only is ~84 bytes
                    return (True, f"wrote {sz} bytes (lin={lin:g}, ok={ok})")
        except Exception as e:
            last_msg = f"post-check failed: {e}"

        last_msg = f"Write() returned {ok} (lin={lin:g})"

        # Clean up any header-only / empty output before retrying
        try:
            if out_path.exists() and out_path.stat().st_size <= 84:
                out_path.unlink(missing_ok=True)
        except Exception:
            pass

    return (False, f"Failed to write STL: {out_path} ({last_msg})")


def build_manifest(
    *,
    step_path: Path,
    out_dir: Path,
    xcaf_instances_path: Path,
    overwrite_stl: bool,
    ascii_stl: bool,
    linear_deflection: float,
    angular_deflection: float,
    use_sig_free: bool,
    strict: bool,
    strict_signature_collisions: bool,
) -> Dict[str, Any]:
    t0 = time.time()
    started_utc = _utc_now_iso()

    data = _read_json(xcaf_instances_path)
    defs = data.get("definitions")
    if not isinstance(defs, dict):
        raise RuntimeError("xcaf_instances.json missing 'definitions' dict")

    # Step 1 collisions (optional)
    step1_meta = data.get("meta", {})
    collisions = []
    if isinstance(step1_meta, dict):
        collisions = step1_meta.get("signature_collisions", []) or []
    if not isinstance(collisions, list):
        collisions = []

    # Collect shaped defs and validate signature presence
    shaped_def_ids: List[str] = []
    algo_counts: Dict[str, int] = {}
    missing_sig_fields = 0

    for def_id, defn in defs.items():
        if not isinstance(defn, dict):
            continue
        if not bool(defn.get("has_shape", False)):
            continue
        shaped_def_ids.append(str(def_id))

        algo = defn.get("def_sig_algo")
        sig = defn.get("def_sig_free") if use_sig_free else None
        if not (isinstance(sig, str) and sig):
            sig = defn.get("def_sig")

        if not (isinstance(algo, str) and algo and isinstance(sig, str) and sig):
            missing_sig_fields += 1
            continue
        algo_counts[algo] = algo_counts.get(algo, 0) + 1

    shaped_def_ids.sort()
    if len(shaped_def_ids) > MAX_SHAPED_DEFS_GUARD:
        raise RuntimeError(f"Guard: too many shaped definitions ({len(shaped_def_ids)} > {MAX_SHAPED_DEFS_GUARD})")

    if not algo_counts:
        raise RuntimeError(
            "No usable (def_sig_algo, def_sig/def_sig_free) found in Step 1 output for shaped defs. "
            "Re-run Step 1 and confirm def_sig fields are populated."
        )

    if len(algo_counts) != 1:
        raise RuntimeError(f"Multiple def_sig_algo values found in Step 1 output: {sorted(algo_counts.keys())}")

    algo = next(iter(algo_counts.keys()))
    if algo != DEF_SIG_ALGO:
        raise RuntimeError(f"Step 1 algo is '{algo}' but Step 2 expects '{DEF_SIG_ALGO}'")

    # Build Step1 collision sig set (from meta + duplicates within JSON)
    step1_collision_sigs: set[str] = set()
    for c in collisions:
        if isinstance(c, dict):
            s = c.get("def_sig") or c.get("sig")
            if isinstance(s, str) and s:
                step1_collision_sigs.add(s)
        elif isinstance(c, str) and c:
            step1_collision_sigs.add(c)

    sig_seen: set[str] = set()
    for def_id in shaped_def_ids:
        defn = defs[def_id]
        s = defn.get("def_sig_free") if use_sig_free else None
        if not (isinstance(s, str) and s):
            s = defn.get("def_sig")
        if isinstance(s, str) and s:
            if s in sig_seen:
                step1_collision_sigs.add(s)
            else:
                sig_seen.add(s)

    if step1_collision_sigs and strict_signature_collisions:
        raise RuntimeError(
            f"Step 1 signature collisions present ({len(step1_collision_sigs)} sigs). "
            "Refusing due to --strict-signature-collisions."
        )

    # Re-read STEP, enumerate labels, compute signature for each label shape
    print(f"[export_stl_xcaf] Re-reading STEP: {step_path}")
    doc = _new_xcaf_document()
    _load_step_into_doc(step_path, doc)
    shape_tool = _get_shape_tool(doc)
    _update_assemblies(shape_tool)

    labels = _get_all_shape_labels(shape_tool)
    if len(labels) > MAX_LABELS_GUARD:
        raise RuntimeError(f"Guard: too many labels returned ({len(labels)} > {MAX_LABELS_GUARD})")

    sig_to_labels: Dict[str, List[Any]] = {}
    null_shapes = 0
    sig_fail = 0

    for idx, lab in enumerate(labels, start=1):
        if (idx % PROGRESS_LABEL_EVERY) == 0:
            print(f"[export_stl_xcaf]  signature scan: {idx}/{len(labels)} labels...")

        shp = _shape_from_label(shape_tool, lab)
        if shp is None:
            null_shapes += 1
            continue
        try:
            sig = compute_def_sig_free(shp) if use_sig_free else compute_def_sig(shp)
        except Exception:
            sig_fail += 1
            continue
        sig_to_labels.setdefault(sig, []).append(lab)

    # make label list ordering deterministic per signature
    for s, labs in sig_to_labels.items():
        labs.sort(key=_label_entry_str)

    # Export per Step1 def
    out_dir.mkdir(parents=True, exist_ok=True)
    stl_dir = out_dir / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    matched = 0
    unmatched = 0
    amb_step1 = 0
    amb_reread = 0
    exported = 0
    skipped_existing = 0
    failed_write = 0

    matched_with_reread_dupes = 0

    for i, def_id in enumerate(shaped_def_ids, start=1):
        if (i % PROGRESS_DEF_EVERY) == 0:
            print(f"[export_stl_xcaf]  defs: {i}/{len(shaped_def_ids)}...")

        defn = defs[def_id]

        sig = defn.get("def_sig_free") if use_sig_free else None
        sig_source = "def_sig_free" if (use_sig_free and isinstance(sig, str) and sig) else None
        if not (isinstance(sig, str) and sig):
            sig = defn.get("def_sig")
            sig_source = "def_sig"
=======
def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s[:120] if s else "unnamed"


def _unique_base(desired: str, used: Set[str]) -> str:
    """
    Ensure filename base uniqueness within a run.
    """
    base = _safe_filename(desired)
    if base not in used:
        used.add(base)
        return base
    n = 2
    while True:
        cand = f"{base}__{n}"
        if cand not in used:
            used.add(cand)
            return cand
        n += 1


# -----------------------------
# Geometry helpers
# -----------------------------
def _iter_solids(shape) -> List[Any]:
    TopAbs = _mod("TopAbs")
    TopExp = _mod("TopExp")
    ex = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_SOLID)
    solids = []
    while ex.More():
        solids.append(ex.Current())
        ex.Next()
    return solids


def _solid_volume(solid) -> float:
    BRepGProp = _mod("BRepGProp")
    GProp = _mod("GProp")
    props = GProp.GProp_GProps()
    # VolumeProperties exists as a module-level function in OCP
    BRepGProp.brepgprop_VolumeProperties(solid, props)
    return float(props.Mass())


def _mesh_shape(shape, deflection: float) -> None:
    BRepMesh = _mod("BRepMesh")
    # (shape, lin_defl, isRelative, ang_defl, parallel)
    BRepMesh.BRepMesh_IncrementalMesh(shape, float(deflection), False, float(deflection), True)


# -----------------------------
# Main export logic
# -----------------------------
def _collect_unique_defs(st, doc) -> Dict[str, Any]:
    """
    Collect unique referred definition labels from the XCAF document.
    Returns map: ref_entry -> ref_label
    """
    TDF = _mod("TDF")

    roots = TDF.TDF_LabelSequence()
    call_maybe_s(st, "GetFreeShapes", roots)

    uniq_defs: Dict[str, Any] = {}

    def walk_def(def_label, seen: Set[str]) -> None:
        def_entry = _label_entry(def_label)
        if def_entry in seen:
            return
        seen.add(def_entry)
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b

        part_index = 0
        part_id = _sha1_hex(f"{sig}|{part_index}")

<<<<<<< HEAD
        item: Dict[str, Any] = {
            "ref_def": def_id,
            "part_index": part_index,
            "part_id": part_id,
            "stl_path": None,
            "def_sig_algo": defn.get("def_sig_algo"),
            "def_sig_used": sig,
            "def_sig_source": sig_source,
            "match_status": None,
        }

        if not (isinstance(sig, str) and sig and isinstance(defn.get("def_sig_algo"), str) and defn.get("def_sig_algo")):
            item["match_status"] = "missing_signature_fields"
            unmatched += 1
            items.append(item)
            continue

        if sig in step1_collision_sigs:
            item["match_status"] = "ambiguous_step1_signature_collision"
            item["ambiguous"] = True
            amb_step1 += 1
            items.append(item)
            continue

        labs = sig_to_labels.get(sig, [])
        if not labs:
            item["match_status"] = "unmatched_signature"
            unmatched += 1
            items.append(item)
            continue

        # reread duplicates
        if len(labs) > 1:
            item["reread_label_duplicates_for_sig"] = len(labs)
            if strict:
                item["match_status"] = "ambiguous_reread_multiple_labels"
                item["ambiguous"] = True
                item["ambiguous_label_count"] = len(labs)
                amb_reread += 1
                items.append(item)
                continue
            # non-strict: pick deterministically (labs already sorted)
            item["match_status"] = "matched_with_reread_duplicates"
            matched_with_reread_dupes += 1
        else:
            item["match_status"] = "matched"

        matched += 1

        stl_rel = f"stl/{part_id}.stl"
        stl_path = out_dir / stl_rel

        if overwrite_stl or (not stl_path.exists()):
            shp = _shape_from_label(shape_tool, labs[0])
            if shp is None:
                item["match_status"] = "matched_but_shape_missing"
                unmatched += 1
                matched -= 1
                items.append(item)
                continue
            _mesh_shape(shp, linear_deflection, angular_deflection)

            ok_write, write_msg = _write_stl(
                shp,
                stl_path,
                ascii_stl,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
            )
            if not ok_write:
                failed_write += 1
                item["stl_path"] = None
                item["stl_error"] = write_msg
                item["match_status"] = "stl_write_failed"
                if strict:
                    raise RuntimeError(write_msg)
                items.append(item)
=======
        for i in range(1, kids.Length() + 1):
            occ = kids.Value(i)

            ref = TDF.TDF_Label()
            has_ref = bool(call_maybe_s(st, "GetReferredShape", occ, ref))
            if not has_ref:
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b
                continue

            exported += 1
        else:
            skipped_existing += 1

<<<<<<< HEAD
        item["stl_path"] = stl_rel
        items.append(item)

    if strict and (unmatched > 0 or amb_step1 > 0 or amb_reread > 0):
        raise RuntimeError(
            f"Strict mode failure: matched={matched}, unmatched={unmatched}, "
            f"ambiguous_step1={amb_step1}, ambiguous_reread={amb_reread}"
        )

    finished_utc = _utc_now_iso()
    runtime_s = float(time.time() - t0)

    manifest: Dict[str, Any] = {
        "meta": {
            "created_utc": finished_utc,
            "analysis_started_utc": started_utc,
            "analysis_finished_utc": finished_utc,
            "runtime_seconds": runtime_s,
            "tool_flavor": _OCC_FLAVOR,
            "step_filename": step_path.name,
            "xcaf_instances": xcaf_instances_path.name,
            "sig_algo": algo,
            "use_sig_free": bool(use_sig_free),
            "mesh": {
                "linear_deflection_mm": float(linear_deflection),
                "angular_deflection_rad": float(angular_deflection),
                "stl_ascii": bool(ascii_stl),
            },
            "counts": {
                "defs_with_shape_step1": len(shaped_def_ids),
                "labels_reread_total": len(labels),
                "labels_with_null_shape": null_shapes,
                "sig_fail_count": sig_fail,
                "matched": matched,
                "unmatched": unmatched,
                "ambiguous_step1": amb_step1,
                "ambiguous_reread": amb_reread,
                "matched_with_reread_dupes": matched_with_reread_dupes,
                "exported": exported,
                "skipped_existing": skipped_existing,
                "failed_write": failed_write,
                "missing_sig_fields_step1": missing_sig_fields,
                "step1_signature_collision_count": len(step1_collision_sigs),
            },
            "warnings": {
                "step1_signature_collisions_present": bool(step1_collision_sigs),
            },
        },
        "items": items,
    }
    return manifest


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_truthy(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    t = str(v).strip().lower()
    return not (t in ("0", "false", "no", "off"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-path", required=True, help="Path to STEP (e.g. /in/model.step)")
    ap.add_argument("--out-dir", default="/out", help="Output directory (default: /out)")
    ap.add_argument("--xcaf-json", default="/out/xcaf_instances.json", help="Path to Step 1 xcaf_instances.json")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite STLs if they already exist")
    ap.add_argument("--ascii-stl", action="store_true", help="Write ASCII STL")

    ap.add_argument("--linear-deflection", type=float, default=None)
    ap.add_argument("--angular-deflection", type=float, default=None)

    ap.add_argument("--use-sig-free", action="store_true", help="Prefer def_sig_free when available")
    ap.add_argument("--strict", action="store_true", help="Fail if any unmatched or ambiguous defs occur")
    ap.add_argument("--strict-signature-collisions", action="store_true", help="Fail immediately if Step 1 collisions exist")

    ns = ap.parse_args()

    # allow .env defaults via docker --env-file, while still letting CLI override
    linear_deflection = float(ns.linear_deflection) if ns.linear_deflection is not None else _env_float("STL_LINEAR_DEFLECTION", DEFAULT_LINEAR_DEFLECTION)
    angular_deflection = float(ns.angular_deflection) if ns.angular_deflection is not None else _env_float("STL_ANGULAR_DEFLECTION", DEFAULT_ANGULAR_DEFLECTION)
    ascii_stl = bool(ns.ascii_stl) or _env_truthy("STL_ASCII", False)

    step_path = Path(ns.step_path)
    out_dir = Path(ns.out_dir)
    xcaf_path = Path(ns.xcaf_json)

    manifest = build_manifest(
        step_path=step_path,
        out_dir=out_dir,
        xcaf_instances_path=xcaf_path,
        overwrite_stl=bool(ns.overwrite),
        ascii_stl=ascii_stl,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        use_sig_free=bool(ns.use_sig_free) or _env_truthy("USE_SIG_FREE", False),
        strict=bool(ns.strict),
        strict_signature_collisions=bool(ns.strict_signature_collisions),
    )

    out_manifest = out_dir / "assets_manifest.json"
    _write_json(out_manifest, manifest)

    c = manifest["meta"]["counts"]
    print("[export_stl_xcaf] DONE")
    print(f"  manifest: {out_manifest}")
    print(f"  stl_dir  : {out_dir / 'stl'}")
    print(
        f"  matched={c['matched']} unmatched={c['unmatched']} "
        f"amb_step1={c['ambiguous_step1']} amb_reread={c['ambiguous_reread']} "
        f"matched_with_dupes={c['matched_with_reread_dupes']} "
        f"exported={c['exported']} skipped_existing={c['skipped_existing']}"
    )
    if manifest["meta"]["warnings"]["step1_signature_collisions_present"]:
        print(f"  WARNING: Step 1 signature collisions present: {c['step1_signature_collision_count']} sigs")
=======
            # Recurse only through assemblies to discover deeper referred defs
            if bool(call_maybe_s(st, "IsAssembly", ref)):
                walk_def(ref, seen)

    seen: Set[str] = set()
    for i in range(1, roots.Length() + 1):
        walk_def(roots.Value(i), seen)

    return uniq_defs


def _is_leaf_def(st, ref_lab) -> bool:
    """
    Leaf definition in XCAF terms: not an assembly.
    """
    return not bool(call_maybe_s(st, "IsAssembly", ref_lab))


def export_stls(
    step_path: Path,
    out_dir: Path,
    def_limit: int,
    mesh_deflection: float,
    export_leaf_ancillaries: bool,
    min_anc_abs_vol: float,
    min_anc_rel_vol: float,
    max_ancillaries_per_parent: int,
) -> None:
    if not step_path.exists():
        raise FileNotFoundError(step_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    stl_dir = out_dir / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)
    anc_dir = stl_dir / "ancillary"
    anc_dir.mkdir(parents=True, exist_ok=True)

    app = _get_app()
    doc = _new_doc(app)
    st = _shape_tool(doc)

    print("Reading STEP:", str(step_path), flush=True)
    _read_step_into_doc(str(step_path), doc)
    call_maybe_s(st, "UpdateAssemblies")

    uniq_defs = _collect_unique_defs(st, doc)
    print("Unique definitions found:", len(uniq_defs), flush=True)

    StlAPI = _mod("StlAPI")
    writer = StlAPI.StlAPI_Writer()

    used_names: Set[str] = set()
    defs_manifest: List[Dict[str, Any]] = []
    anc_manifest: List[Dict[str, Any]] = []

    # Deterministic iteration: sort by ref_entry
    items = sorted(uniq_defs.items(), key=lambda kv: kv[0])

    exported = 0
    for ref_entry, ref_lab in items:
        if def_limit and exported >= def_limit:
            break

        ref_name = get_label_name(ref_lab)
        desired = ref_name if ref_name else ref_entry.replace(":", "_")
        base = _unique_base(desired, used_names)

        shape = call_maybe_s(st, "GetShape", ref_lab)
        _mesh_shape(shape, mesh_deflection)

        parent_stl = stl_dir / f"{base}.stl"
        writer.Write(shape, str(parent_stl))

        defs_manifest.append(
            {
                "ref_def": ref_entry,
                "ref_name": ref_name,
                "stl_path": str(parent_stl).replace(str(out_dir), "").lstrip("/"),
            }
        )

        exported += 1
        if exported % 50 == 0:
            print("Exported:", exported, flush=True)

        # Optional: split multi-solid leaf definitions into ancillary solids
        if not export_leaf_ancillaries:
            continue

        if not _is_leaf_def(st, ref_lab):
            continue

        solids = _iter_solids(shape)
        if len(solids) <= 1:
            continue

        if max_ancillaries_per_parent > 0 and len(solids) > max_ancillaries_per_parent:
            # Guardrail: skip insane splits
            anc_manifest.append(
                {
                    "kind": "ancillary_skip",
                    "parent_ref_def": ref_entry,
                    "parent_ref_name": ref_name,
                    "reason": f"too_many_solids:{len(solids)}",
                }
            )
            continue

        vols: List[Tuple[int, float]] = []
        for i, s in enumerate(solids):
            try:
                v = _solid_volume(s)
            except Exception:
                v = 0.0
            vols.append((i, float(v)))

        # Sort by volume desc for stable A### ordering
        vols_sorted = sorted(vols, key=lambda t: t[1], reverse=True)
        main_vol = vols_sorted[0][1] if vols_sorted else 0.0

        # Export each solid as its own STL
        for rank, (orig_idx, v) in enumerate(vols_sorted, start=1):
            # Skip tiny junk solids (absolute or relative to main)
            if v < float(min_anc_abs_vol):
                continue
            if main_vol > 0.0 and v < (main_vol * float(min_anc_rel_vol)):
                continue

            child_tag = f"A{rank:03d}"
            child_ref_def = f"{ref_entry}#{child_tag}"

            child_stl = anc_dir / f"{base}__{child_tag}.stl"
            try:
                _mesh_shape(solids[orig_idx], mesh_deflection)
            except Exception:
                # If meshing fails on the sub-solid, still attempt write (may fail too)
                pass
            writer.Write(solids[orig_idx], str(child_stl))

            anc_manifest.append(
                {
                    "kind": "ancillary",
                    "ref_def": child_ref_def,
                    "ref_name": (ref_name + "__" + child_tag) if ref_name else (base + "__" + child_tag),
                    "stl_path": str(child_stl).replace(str(out_dir), "").lstrip("/"),
                    "parent_ref_def": ref_entry,
                    "parent_ref_name": ref_name,
                    "child_index": rank,
                    "child_volume": v,
                    "main_volume": main_vol,
                    "qty_per_parent": 1,
                }
            )

    # Write manifests
    defs_path = out_dir / "stl_manifest.json"
    defs_path.write_text(json.dumps(defs_manifest, indent=2), encoding="utf-8")
    print("Wrote:", str(defs_path), flush=True)

    if export_leaf_ancillaries:
        anc_path = out_dir / "stl_manifest_ancillary.json"
        anc_path.write_text(json.dumps(anc_manifest, indent=2), encoding="utf-8")
        print("Wrote:", str(anc_path), flush=True)

        all_path = out_dir / "stl_manifest_all.json"
        all_items = []
        for d in defs_manifest:
            dd = dict(d)
            dd["kind"] = "definition"
            all_items.append(dd)
        all_items.extend(anc_manifest)
        all_path.write_text(json.dumps(all_items, indent=2), encoding="utf-8")
        print("Wrote:", str(all_path), flush=True)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export unique definition STLs from STEP via XCAF; optionally split multi-solid leaf defs into ancillary STLs."
    )
    p.add_argument("step", help="Input STEP file path (inside container: /in/model.step)")
    p.add_argument("out", nargs="?", default="/out", help="Output dir (default: /out)")
    p.add_argument("def_limit", nargs="?", type=int, default=0, help="Optional limit on defs exported (0=all)")

    p.add_argument("--mesh", type=float, default=0.5, help="Meshing deflection (default 0.5)")
    p.add_argument(
        "--export-leaf-ancillaries",
        action="store_true",
        help="Split multi-solid leaf definitions into ancillary STLs and write stl_manifest_ancillary.json",
    )
    p.add_argument(
        "--min-anc-abs-vol",
        type=float,
        default=0.0,
        help="Skip ancillary solids with volume below this absolute threshold (default 0.0)",
    )
    p.add_argument(
        "--min-anc-rel-vol",
        type=float,
        default=0.0,
        help="Skip ancillary solids with volume below this fraction of the main solid volume (default 0.0)",
    )
    p.add_argument(
        "--max-ancillaries-per-parent",
        type=int,
        default=200,
        help="Guardrail: skip splitting if a leaf contains more than this many solids (default 200; 0 disables)",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    step_path = Path(args.step)
    out_dir = Path(args.out)
    def_limit = int(args.def_limit)

    export_stls(
        step_path=step_path,
        out_dir=out_dir,
        def_limit=def_limit,
        mesh_deflection=float(args.mesh),
        export_leaf_ancillaries=bool(args.export_leaf_ancillaries),
        min_anc_abs_vol=float(args.min_anc_abs_vol),
        min_anc_rel_vol=float(args.min_anc_rel_vol),
        max_ancillaries_per_parent=int(args.max_ancillaries_per_parent),
    )
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b
    return 0


if __name__ == "__main__":
<<<<<<< HEAD
    raise SystemExit(main())
=======
    raise SystemExit(main(sys.argv[1:]))
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b
