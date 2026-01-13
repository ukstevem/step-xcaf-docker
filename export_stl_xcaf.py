#!/usr/bin/env python3
"""
export_stl_xcaf.py (Step 2)

Reads:
  - STEP: /in/<model>.step
  - /out/xcaf_instances.json

Writes:
  - /out/stl/<part_id>.stl
  - /out/stl_manifest.json

Determinism goals:
  - stable ordering (sorted ref_def)
  - stable bbox quantization (tol=0.1mm)
  - stable part_id = sha1(ref_def|part_index|bbox_q_json)
  - do not delete old STLs; overwrite only if requested
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------
# OCC / OCP compatibility
# ---------------------------

_OCC_FLAVOR = "unknown"

try:
    # OCP
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCP.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool, TDF_ChildIterator
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Message import Message_ProgressRange

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    _OCC_FLAVOR = "OCP"
except Exception:
    # pythonocc-core
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
    from OCC.Core.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool, TDF_ChildIterator
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Message import Message_ProgressRange

    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib as BRepBndLib  # pythonocc naming
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    _OCC_FLAVOR = "pythonocc"


# ---------------------------
# Constants (Power-of-10)
# ---------------------------

BBOX_TOL_MM = 0.1
BBOX_SCALE = int(round(1.0 / BBOX_TOL_MM))  # 10 for 0.1mm

DEFAULT_LINEAR_DEFLECTION = 0.25  # mm
DEFAULT_ANGULAR_DEFLECTION = 0.35  # radians (approx)
MAX_DEFS = 250000  # hard guard (you won't hit this, but it's explicit)


# ---------------------------
# Small helpers
# ---------------------------

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


def _json_compact(obj: Any) -> str:
    # deterministic serialization
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _quantize_mm_to_int(v_mm: float) -> int:
    # quantize to 0.1mm using integer scaling
    return int(round(v_mm * BBOX_SCALE))


def _bbox_of_shape(shape) -> Tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    # Some OCC builds support SetGap; keep deterministic by not setting random gaps.
    BRepBndLib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax))


def _bbox_q_from_bbox(b: Tuple[float, float, float, float, float, float]) -> Dict[str, Any]:
    xmin, ymin, zmin, xmax, ymax, zmax = b
    min_i = [_quantize_mm_to_int(xmin), _quantize_mm_to_int(ymin), _quantize_mm_to_int(zmin)]
    max_i = [_quantize_mm_to_int(xmax), _quantize_mm_to_int(ymax), _quantize_mm_to_int(zmax)]
    return {"tol_mm": BBOX_TOL_MM, "scale": BBOX_SCALE, "min_i": min_i, "max_i": max_i}


def _label_entry_str(label: TDF_Label) -> str:
    # Prefer EntryDumpToString when present
    if hasattr(label, "EntryDumpToString"):
        try:
            return str(label.EntryDumpToString())
        except Exception:
            pass
    # Fallback: TDF_Tool.Entry(label, ascii)
    asc = TCollection_AsciiString()
    try:
        ok = TDF_Tool.Entry(label, asc)
        if ok:
            return asc.ToCString()
    except Exception:
        pass
    # Last resort
    return ""


def _get_xcaf_app():
    if hasattr(XCAFApp_Application, "GetApplication"):
        return XCAFApp_Application.GetApplication()
    if hasattr(XCAFApp_Application, "GetApplication_s"):
        return XCAFApp_Application.GetApplication_s()
    return XCAFApp_Application()


def _construct_doc(fmt: str):
    # Try both ctor styles
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

    last_errs: List[str] = []
    for fmt in formats:
        doc = _construct_doc(fmt)
        if doc is None:
            last_errs.append(f"ctor failed: {fmt}")
            continue

        # Try NewDocument in both calling styles; accept if doc.Main() usable
        ok_any = False
        for arg in (fmt, TCollection_ExtendedString(fmt)):
            try:
                ok = app.NewDocument(arg, doc)
                if ok is None:
                    ok_any = True
                else:
                    ok_any = bool(ok) or ok_any
            except Exception:
                pass

        try:
            _ = doc.Main()
            # If Main works, accept
            return doc
        except Exception:
            last_errs.append(f"doc.Main failed: {fmt}")

    raise RuntimeError("Failed to create XCAF document. Tried: " + ", ".join(formats) + " / " + "; ".join(last_errs))


def _get_shape_tool(doc):
    # OCP and pythonocc differ on ShapeTool vs ShapeTool_s
    try:
        return XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    except Exception:
        return XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def _update_assemblies(shape_tool) -> None:
    # Some builds require UpdateAssemblies_s
    if hasattr(shape_tool, "UpdateAssemblies"):
        try:
            shape_tool.UpdateAssemblies()
            return
        except Exception:
            pass
    if hasattr(shape_tool, "UpdateAssemblies_s"):
        shape_tool.UpdateAssemblies_s()


def _load_step_into_doc(step_path: Path, doc) -> None:
    if not step_path.exists():
        raise FileNotFoundError(f"STEP not found: {step_path}")

    rdr = STEPCAFControl_Reader()
    # Some builds support SetColorMode/SetNameMode etc; keep minimal & deterministic.
    pr = Message_ProgressRange()
    stat = rdr.ReadFile(str(step_path))
    if stat != IFSelect_RetDone:
        raise RuntimeError(f"STEPCAFControl_Reader.ReadFile failed: {step_path} (status={stat})")

    ok = False
    try:
        ok = bool(rdr.Transfer(doc, pr))
    except Exception:
        try:
            ok = bool(rdr.Transfer(doc))
        except Exception:
            ok = False
    if not ok:
        raise RuntimeError("STEPCAF transfer failed")


def _build_entry_to_label_map(doc, shape_tool) -> Dict[str, TDF_Label]:
    # Preferred: shape_tool.GetShapes(seq) if available.
    out: Dict[str, TDF_Label] = {}

    if hasattr(shape_tool, "GetShapes"):
        try:
            seq = TDF_LabelSequence()
            shape_tool.GetShapes(seq)
            # LabelSequence is 1-based
            n = int(seq.Length())
            for i in range(1, n + 1):
                lab = seq.Value(i)
                ent = _label_entry_str(lab)
                if ent:
                    out[ent] = lab
            if out:
                return out
        except Exception:
            out = {}

    # Fallback: traverse entire label tree
    root = doc.Main()
    it = TDF_ChildIterator(root, True)
    while it.More():
        lab = it.Value()
        ent = _label_entry_str(lab)
        if ent:
            out[ent] = lab
        it.Next()

    return out


def _mesh_shape(shape, linear_deflection: float, angular_deflection: float) -> None:
    # Deterministic mesh params; avoid parallel if binding exposes it.
    try:
        mesh = BRepMesh_IncrementalMesh(shape, float(linear_deflection), False, float(angular_deflection), False)
        # Some builds need Perform()
        if hasattr(mesh, "Perform"):
            mesh.Perform()
    except TypeError:
        # Older signature: (shape, deflection)
        mesh = BRepMesh_IncrementalMesh(shape, float(linear_deflection))
        if hasattr(mesh, "Perform"):
            mesh.Perform()


def _write_stl(shape, out_path: Path, ascii_mode: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wr = StlAPI_Writer()
    # Prefer explicit ASCII control if supported
    if hasattr(wr, "SetASCIIMode"):
        try:
            wr.SetASCIIMode(bool(ascii_mode))
        except Exception:
            pass
    elif hasattr(wr, "ASCIIMode"):
        try:
            wr.ASCIIMode = bool(ascii_mode)
        except Exception:
            pass

    ok = False
    try:
        ok = bool(wr.Write(shape, str(out_path)))
    except Exception:
        ok = False
    if not ok and not out_path.exists():
        raise RuntimeError(f"Failed to write STL: {out_path}")


# ---------------------------
# Main logic
# ---------------------------

def build_manifest(
    step_path: Path,
    out_dir: Path,
    xcaf_instances_path: Path,
    *,
    overwrite_stl: bool,
    ascii_stl: bool,
    linear_deflection: float,
    angular_deflection: float,
) -> Dict[str, Any]:
    data = _read_json(xcaf_instances_path)
    defs: Dict[str, Any] = data.get("definitions", {})
    if not isinstance(defs, dict):
        raise RuntimeError("xcaf_instances.json missing 'definitions' dict")

    def_ids = [k for k, v in defs.items() if isinstance(v, dict) and bool(v.get("has_shape", False))]
    def_ids.sort()

    if len(def_ids) > MAX_DEFS:
        raise RuntimeError(f"Guard: too many definitions with shapes ({len(def_ids)} > {MAX_DEFS})")

    # Load STEP -> XCAF doc
    doc = _new_xcaf_document()
    _load_step_into_doc(step_path, doc)
    shape_tool = _get_shape_tool(doc)
    _update_assemblies(shape_tool)

    entry_map = _build_entry_to_label_map(doc, shape_tool)

    stl_dir = out_dir / "stl"
    items: List[Dict[str, Any]] = []

    for ref_def in def_ids:
        d = defs.get(ref_def, {})
        name = d.get("name")
        shape_kind = d.get("shape_kind")
        solid_count = d.get("solid_count")

        lab = entry_map.get(ref_def)
        if lab is None:
            # Try direct label lookup if map didn't contain it
            found = False
            try:
                lab2 = TDF_Label()
                ok = False
                # different bindings expose TDF_Tool.Label differently
                if hasattr(TDF_Tool, "Label"):
                    try:
                        ok = bool(TDF_Tool.Label(doc.Main().Data(), TCollection_AsciiString(ref_def), lab2, False))
                    except Exception:
                        ok = False
                if ok:
                    lab = lab2
                    found = True
            except Exception:
                found = False

            if not found or lab is None:
                raise RuntimeError(f"Could not locate XCAF label for ref_def={ref_def}")

        # GetShape
        try:
            shape = shape_tool.GetShape(lab)
        except Exception as e:
            raise RuntimeError(f"shape_tool.GetShape failed for ref_def={ref_def}: {e}")

        # bbox + quantization for stable ID
        bbox = _bbox_of_shape(shape)
        bbox_q = _bbox_q_from_bbox(bbox)

        part_index = 0
        part_id_src = f"{ref_def}|{part_index}|{_json_compact(bbox_q)}"
        part_id = _sha1_hex(part_id_src)
        stl_rel = f"stl/{part_id}.stl"
        stl_path = out_dir / stl_rel

        # export STL if needed
        if overwrite_stl or (not stl_path.exists()):
            _mesh_shape(shape, linear_deflection, angular_deflection)
            _write_stl(shape, stl_path, ascii_stl)

        items.append(
            {
                "ref_def": ref_def,
                "part_index": part_index,
                "part_id": part_id,
                "stl_path": stl_rel,
                "bbox_q": bbox_q,
                # traceability (optional but useful)
                "name": name,
                "shape_kind": shape_kind,
                "solid_count": solid_count,
            }
        )

    manifest: Dict[str, Any] = {
        "meta": {
            "created_utc": _utc_now_iso(),
            "occ_flavor": _OCC_FLAVOR,
            "step_filename": str(step_path.name),
            "xcaf_instances": str(xcaf_instances_path.name),
            "bbox_tol_mm": BBOX_TOL_MM,
            "mesh": {
                "linear_deflection_mm": float(linear_deflection),
                "angular_deflection_rad": float(angular_deflection),
                "stl_ascii": bool(ascii_stl),
            },
            "counts": {
                "defs_with_shape": len(def_ids),
                "items": len(items),
            },
        },
        "items": items,
    }

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-path", required=True, help="Path to STEP (e.g. /in/model.step)")
    ap.add_argument("--out-dir", default="/out", help="Output directory (default: /out)")
    ap.add_argument("--xcaf-json", default="/out/xcaf_instances.json", help="Path to xcaf_instances.json")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite STLs if they already exist")
    ap.add_argument("--ascii-stl", action="store_true", help="Write ASCII STL (default is binary if supported)")
    ap.add_argument("--linear-deflection", type=float, default=DEFAULT_LINEAR_DEFLECTION, help="Mesh linear deflection (mm)")
    ap.add_argument("--angular-deflection", type=float, default=DEFAULT_ANGULAR_DEFLECTION, help="Mesh angular deflection (rad)")
    ns = ap.parse_args()

    step_path = Path(ns.step_path)
    out_dir = Path(ns.out_dir)
    xcaf_path = Path(ns.xcaf_json)

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        step_path=step_path,
        out_dir=out_dir,
        xcaf_instances_path=xcaf_path,
        overwrite_stl=bool(ns.overwrite),
        ascii_stl=bool(ns.ascii_stl),
        linear_deflection=float(ns.linear_deflection),
        angular_deflection=float(ns.angular_deflection),
    )

    out_manifest = out_dir / "stl_manifest.json"
    _write_json(out_manifest, manifest)

    print(f"[export_stl_xcaf] wrote: {out_manifest}")
    print(f"[export_stl_xcaf] stl dir: {out_dir / 'stl'}")
    print(f"[export_stl_xcaf] items: {manifest['meta']['counts']['items']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
