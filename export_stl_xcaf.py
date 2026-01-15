#!/usr/bin/env python3
"""
Step 2: export_stl_xcaf.py

Deterministic across STEP rereads by matching definition shapes using Step 1 signatures.

Inputs:
  - STEP file (re-read): /in/<model>.step
  - /out/xcaf_instances.json (from Step 1)

Outputs:
  - /out/stl/<part_id>.stl
  - /out/stl_manifest.json

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
      otherwise => mark ambiguous and skip export for those defs

Part-id rules (stable across Step 1 reruns):
  - part_index is always 0 for now.
  - part_id = sha1(def_sig_used + "|" + part_index + "|" + bbox_q)
  - bbox_q derived from Step 1 bbox (quantized) so IDs are stable.

Manifest rule:
  - Only set stl_path when an STL was actually produced (or already exists + matched).
    For unmatched/ambiguous items stl_path is null, so Step 3 can safely skip.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------
# Env helpers (read from .env via docker --env-file)
# -----------------------------
def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    v = v.strip()
    return v if v else default

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    raw = raw.strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}='{raw}' (expected float)") from e

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    raw = raw.strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}='{raw}' (expected int)") from e

# -----------------------------
# OCP imports (expected in docker)
# -----------------------------
_OCC_FLAVOR = "unknown"
try:
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
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
# Constants / guards (env-backed defaults)
# -----------------------------
# Quantization tolerance for bbox_q (affects part_id stability)
BBOX_TOL_MM = _env_float("STEP_BBOX_TOL_MM", 0.1)
if BBOX_TOL_MM <= 0.0:
    raise RuntimeError(f"STEP_BBOX_TOL_MM must be > 0, got {BBOX_TOL_MM}")

BBOX_SCALE = int(round(1.0 / BBOX_TOL_MM))
if BBOX_SCALE <= 0:
    raise RuntimeError(f"Invalid BBOX_SCALE from STEP_BBOX_TOL_MM={BBOX_TOL_MM}")

# Meshing defaults (used unless CLI overrides)
DEFAULT_LINEAR_DEFLECTION = _env_float("STEP2_LINEAR_DEFLECTION", 0.25)
DEFAULT_ANGULAR_DEFLECTION = _env_float("STEP2_ANGULAR_DEFLECTION", 0.35)

# Guards
MAX_SHAPED_DEFS_GUARD = _env_int("STEP2_MAX_SHAPED_DEFS_GUARD", 300000)
MAX_LABELS_GUARD = _env_int("STEP2_MAX_LABELS_GUARD", 600000)

# -----------------------------
# OCP binding helper
# -----------------------------
def call_maybe_s(obj: Any, method: str, *args):
    """
    Try obj.method(*args) then obj.method_s(*args) for OCP binding variants.

    If obj.method exists but raises TypeError due to signature mismatch,
    fall back to method_s.
    """
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

def _json_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)

def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------------
# bbox quantization (from Step 1 bbox)
# -----------------------------
def _q_mm_to_int(v_mm: float) -> int:
    return int(round(float(v_mm) * BBOX_SCALE))

def _bbox_q_from_step1_def(defn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bb = defn.get("bbox")
    if not isinstance(bb, dict):
        return None
    mn = bb.get("min")
    mx = bb.get("max")
    if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3):
        return None
    try:
        mnf = [float(mn[0]), float(mn[1]), float(mn[2])]
        mxf = [float(mx[0]), float(mx[1]), float(mx[2])]
    except Exception:
        return None
    return {
        "tol_mm": float(BBOX_TOL_MM),
        "scale": int(BBOX_SCALE),
        "min_i": [_q_mm_to_int(mnf[0]), _q_mm_to_int(mnf[1]), _q_mm_to_int(mnf[2])],
        "max_i": [_q_mm_to_int(mxf[0]), _q_mm_to_int(mxf[1]), _q_mm_to_int(mxf[2])],
    }

# -----------------------------
# XCAF document helpers
# -----------------------------
def _get_xcaf_app():
    if hasattr(XCAFApp_Application, "GetApplication"):
        return XCAFApp_Application.GetApplication()
    if hasattr(XCAFApp_Application, "GetApplication_s"):
        return XCAFApp_Application.GetApplication_s()
    return XCAFApp_Application()

def _construct_doc(fmt: str):
    try:
        return TDocStd_Document(fmt)
    except Exception:
        pass
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
    """
    Prefer GetShapes(seq). If not present, fall back to GetFreeShapes(seq).
    """
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
    """
    Robust label -> TopoDS_Shape for OCP binding variants.
    """
    # 1) direct returning form
    try:
        shp = call_maybe_s(shape_tool, "GetShape", lab)
        if shp is not None and hasattr(shp, "IsNull") and not shp.IsNull():
            return shp
    except Exception:
        pass

    # 2) out-arg form
    out_shp = TopoDS_Shape()
    try:
        _ = call_maybe_s(shape_tool, "GetShape", lab, out_shp)
        if hasattr(out_shp, "IsNull") and not out_shp.IsNull():
            return out_shp
    except Exception:
        pass

    # 3) alternate name
    try:
        shp = call_maybe_s(shape_tool, "Shape", lab)
        if shp is not None and hasattr(shp, "IsNull") and not shp.IsNull():
            return shp
    except Exception:
        pass

    return None

def _label_key(lab: Any) -> str:
    """
    Stable-ish key for a TDF_Label so we can pick a canonical label deterministically.
    Prefer EntryDumpToString() (same style as Step 1 def_id).
    """
    try:
        if hasattr(lab, "EntryDumpToString"):
            s = lab.EntryDumpToString()
            if isinstance(s, str) and s:
                return s
    except Exception:
        pass

    # Fallback: best-effort string form
    try:
        return str(lab)
    except Exception:
        return repr(lab)


# -----------------------------
# Meshing / STL writing
# -----------------------------
def _mesh_shape(shape, linear_deflection: float, angular_deflection: float) -> None:
    try:
        mesh = BRepMesh_IncrementalMesh(shape, float(linear_deflection), False, float(angular_deflection), False)
        if hasattr(mesh, "Perform"):
            mesh.Perform()
    except TypeError:
        mesh = BRepMesh_IncrementalMesh(shape, float(linear_deflection))
        if hasattr(mesh, "Perform"):
            mesh.Perform()

def _write_stl(shape, out_path: Path, ascii_mode: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wr = StlAPI_Writer()
    if hasattr(wr, "SetASCIIMode"):
        try:
            wr.SetASCIIMode(bool(ascii_mode))
        except Exception:
            pass
    ok = False
    try:
        ok = bool(wr.Write(shape, str(out_path)))
    except Exception:
        ok = False
    if not ok and not out_path.exists():
        raise RuntimeError(f"Failed to write STL: {out_path}")

# -----------------------------
# Main build
# -----------------------------
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
    doc = _new_xcaf_document()
    _load_step_into_doc(step_path, doc)
    shape_tool = _get_shape_tool(doc)
    _update_assemblies(shape_tool)

    labels = _get_all_shape_labels(shape_tool)
    if len(labels) > MAX_LABELS_GUARD:
        raise RuntimeError(f"Guard: too many labels returned ({len(labels)} > {MAX_LABELS_GUARD})")

    # Map signature -> canonical label (deterministic) and count of labels seen for that sig
    sig_to_label: Dict[str, Any] = {}
    sig_label_counts: Dict[str, int] = {}

    null_shapes = 0
    sig_fail = 0

    for lab in labels:
        shp = _shape_from_label(shape_tool, lab)
        if shp is None:
            null_shapes += 1
            continue
        try:
            sig = compute_def_sig_free(shp) if use_sig_free else compute_def_sig(shp)
        except Exception:
            sig_fail += 1
            continue

        sig_label_counts[sig] = sig_label_counts.get(sig, 0) + 1

        # Pick a canonical label for this signature deterministically
        if sig not in sig_to_label:
            sig_to_label[sig] = lab
        else:
            # choose lowest label key
            if _label_key(lab) < _label_key(sig_to_label[sig]):
                sig_to_label[sig] = lab

    reread_sig_dupe_count = sum(1 for k, v in sig_label_counts.items() if v > 1)
    reread_sig_dupe_max = max([v for v in sig_label_counts.values() if v > 1], default=0)


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

    for def_id in shaped_def_ids:
        defn = defs[def_id]

        sig = defn.get("def_sig_free") if use_sig_free else None
        sig_source = "def_sig_free" if (use_sig_free and isinstance(sig, str) and sig) else None
        if not (isinstance(sig, str) and sig):
            sig = defn.get("def_sig")
            sig_source = "def_sig"

        bbox_q = _bbox_q_from_step1_def(defn)
        if bbox_q is None:
            raise RuntimeError(
                f"Step 1 definition '{def_id}' has_shape=true but has no bbox; required for part_id stability."
            )

        part_index = 0
        pid_src = f"{sig}|{part_index}|{_json_compact(bbox_q)}"
        part_id = _sha1_hex(pid_src)

        item: Dict[str, Any] = {
            "ref_def": def_id,
            "part_index": part_index,
            "part_id": part_id,
            "stl_path": None,
            "bbox_q": bbox_q,
            "name": defn.get("name"),
            "shape_kind": defn.get("shape_kind"),
            "solid_count": defn.get("solid_count"),
            "qty_total": defn.get("qty_total"),
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

        lab = sig_to_label.get(sig)
        if lab is None:
            item["match_status"] = "unmatched_signature"
            unmatched += 1
            items.append(item)
            continue

        # matched (even if there were multiple reread labels with same sig)
        matched += 1
        item["match_status"] = "matched"
        dup_ct = sig_label_counts.get(sig, 1)
        if dup_ct > 1:
            item["reread_label_duplicates_for_sig"] = int(dup_ct)

        stl_rel = f"stl/{part_id}.stl"
        stl_path = out_dir / stl_rel

        if overwrite_stl or (not stl_path.exists()):
            shp = _shape_from_label(shape_tool, lab)
            if shp is None:
                item["match_status"] = "matched_but_shape_missing"
                unmatched += 1
                matched -= 1
                items.append(item)
                continue
            _mesh_shape(shp, linear_deflection, angular_deflection)
            _write_stl(shp, stl_path, ascii_stl)
            exported += 1
        else:
            skipped_existing += 1

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
            "env": {
                "STEP_BBOX_TOL_MM": float(BBOX_TOL_MM),
                "STEP2_LINEAR_DEFLECTION": float(DEFAULT_LINEAR_DEFLECTION),
                "STEP2_ANGULAR_DEFLECTION": float(DEFAULT_ANGULAR_DEFLECTION),
                "STEP2_MAX_SHAPED_DEFS_GUARD": int(MAX_SHAPED_DEFS_GUARD),
                "STEP2_MAX_LABELS_GUARD": int(MAX_LABELS_GUARD),
            },
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
                "reread_sig_duplicate_count": int(reread_sig_dupe_count),
                "reread_sig_duplicate_max": int(reread_sig_dupe_max),
                "matched": matched,
                "unmatched": unmatched,
                "ambiguous_step1": amb_step1,
                "ambiguous_reread": amb_reread,
                "exported": exported,
                "skipped_existing": skipped_existing,
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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-path", required=True, help="Path to STEP (e.g. /in/model.step)")
    ap.add_argument("--out-dir", default="/out", help="Output directory (default: /out)")
    ap.add_argument("--xcaf-json", default="/out/xcaf_instances.json", help="Path to Step 1 xcaf_instances.json")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite STLs if they already exist")
    ap.add_argument("--ascii-stl", action="store_true", help="Write ASCII STL")

    # Defaults come from env-backed constants above
    ap.add_argument("--linear-deflection", type=float, default=DEFAULT_LINEAR_DEFLECTION)
    ap.add_argument("--angular-deflection", type=float, default=DEFAULT_ANGULAR_DEFLECTION)

    ap.add_argument("--use-sig-free", action="store_true", help="Prefer def_sig_free when available")
    ap.add_argument("--strict", action="store_true", help="Fail if any unmatched or ambiguous defs occur")
    ap.add_argument("--strict-signature-collisions", action="store_true", help="Fail immediately if Step 1 collisions exist")

    ns = ap.parse_args()

    step_path = Path(ns.step_path)
    out_dir = Path(ns.out_dir)
    xcaf_path = Path(ns.xcaf_json)

    manifest = build_manifest(
        step_path=step_path,
        out_dir=out_dir,
        xcaf_instances_path=xcaf_path,
        overwrite_stl=bool(ns.overwrite),
        ascii_stl=bool(ns.ascii_stl),
        linear_deflection=float(ns.linear_deflection),
        angular_deflection=float(ns.angular_deflection),
        use_sig_free=bool(ns.use_sig_free),
        strict=bool(ns.strict),
        strict_signature_collisions=bool(ns.strict_signature_collisions),
    )

    out_manifest = out_dir / "stl_manifest.json"
    _write_json(out_manifest, manifest)

    c = manifest["meta"]["counts"]
    print("[export_stl_xcaf] DONE")
    print(f"  manifest: {out_manifest}")
    print(f"  stl_dir  : {out_dir / 'stl'}")
    print(
        f"  matched={c['matched']} unmatched={c['unmatched']} "
        f"amb_step1={c['ambiguous_step1']} amb_reread={c['ambiguous_reread']} "
        f"exported={c['exported']} skipped_existing={c['skipped_existing']}"
    )
    if manifest["meta"]["warnings"]["step1_signature_collisions_present"]:
        print(f"  WARNING: Step 1 signature collisions present: {c['step1_signature_collision_count']} sigs")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
