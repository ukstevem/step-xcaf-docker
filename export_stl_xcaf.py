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

        part_index = 0
        part_id = _sha1_hex(f"{sig}|{part_index}")

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
                continue

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())