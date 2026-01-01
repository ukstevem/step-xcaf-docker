#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Set


def _mod(name: str):
    return __import__(f"OCP.{name}", fromlist=[name])


def call_maybe_s(obj: Any, method: str, *args):
    fn = getattr(obj, method, None)
    if callable(fn):
        return fn(*args)
    fn_s = getattr(obj, method + "_s", None)
    if callable(fn_s):
        return fn_s(*args)
    raise AttributeError(f"{type(obj).__name__} has no {method} / {method}_s")


def _get_app():
    XCAFApp = _mod("XCAFApp")
    return call_maybe_s(XCAFApp.XCAFApp_Application, "GetApplication")


def _new_doc(app):
    TDocStd = _mod("TDocStd")
    TCollection = _mod("TCollection")
    fmt = TCollection.TCollection_ExtendedString("MDTV-XCAF")
    doc = TDocStd.TDocStd_Document(fmt)
    call_maybe_s(app, "NewDocument", fmt, doc)
    _ = doc.Main()
    return doc


def _shape_tool(doc):
    XCAFDoc = _mod("XCAFDoc")
    Tool = XCAFDoc.XCAFDoc_DocumentTool
    fn = getattr(Tool, "ShapeTool", None)
    if callable(fn):
        return fn(doc.Main())
    return Tool.ShapeTool_s(doc.Main())


def _read_step_into_doc(step_path: str, doc):
    STEPCAFControl = _mod("STEPCAFControl")
    IFSelect = _mod("IFSelect")

    r = STEPCAFControl.STEPCAFControl_Reader()
    call_maybe_s(r, "SetNameMode", True)

    status = r.ReadFile(step_path)
    if status != IFSelect.IFSelect_RetDone:
        raise RuntimeError(f"ReadFile failed: {status}")

    ok = r.Transfer(doc)
    if not ok:
        raise RuntimeError("Transfer(doc) returned False")


def get_label_name(lab) -> str:
    TDataStd = _mod("TDataStd")
    name_attr = TDataStd.TDataStd_Name()

    getid = getattr(TDataStd.TDataStd_Name, "GetID", None) or getattr(TDataStd.TDataStd_Name, "GetID_s", None)
    if getid is None:
        return ""
    guid = getid()

    try:
        found = lab.FindAttribute(guid, name_attr)
    except AttributeError:
        found = lab.FindAttribute_s(guid, name_attr)
    if not found:
        return ""

    try:
        s = name_attr.Get()
        # Best-effort conversion
        for m in ("ToExtString", "ToCString", "PrintToString"):
            fn = getattr(s, m, None)
            if callable(fn):
                return fn()
        return str(s)
    except Exception:
        return ""


def _label_entry(lab) -> str:
    # Stable label entry string like "0:1:2"
    try:
        ent = lab.Entry()
        return ent.ToCString() if hasattr(ent, "ToCString") else str(ent)
    except Exception:
        pass

    TCollection = _mod("TCollection")
    TDF = _mod("TDF")
    asc = TCollection.TCollection_AsciiString()
    try:
        TDF.TDF_Tool.Entry(lab, asc)
    except AttributeError:
        TDF.TDF_Tool.Entry_s(lab, asc)
    return asc.ToCString() if hasattr(asc, "ToCString") else str(asc)


def _safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120] if s else "unnamed"


def main(step_path: str, out_dir: str = "/out", def_limit: int = 0):
    step_path_p = Path(step_path)
    if not step_path_p.exists():
        raise FileNotFoundError(step_path_p)

    out_dir_p = Path(out_dir)
    stl_dir = out_dir_p / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)

    app = _get_app()
    doc = _new_doc(app)
    st = _shape_tool(doc)

    print("Reading STEP:", step_path_p, flush=True)
    _read_step_into_doc(str(step_path_p), doc)
    call_maybe_s(st, "UpdateAssemblies")

    # Collect unique definition labels from the expanded tree
    TDF = _mod("TDF")
    roots = TDF.TDF_LabelSequence()
    call_maybe_s(st, "GetFreeShapes", roots)

    uniq_defs: Dict[str, Any] = {}  # entry -> label

    def collect_defs(def_label, seen: Set[str]):
        def_entry = _label_entry(def_label)
        if def_entry in seen:
            return
        seen.add(def_entry)

        kids = TDF.TDF_LabelSequence()
        call_maybe_s(st, "GetComponents", def_label, kids)

        for i in range(1, kids.Length() + 1):
            occ = kids.Value(i)
            ref = TDF.TDF_Label()
            has_ref = bool(call_maybe_s(st, "GetReferredShape", occ, ref))
            if not has_ref:
                continue

            ref_entry = _label_entry(ref)
            if ref_entry not in uniq_defs:
                uniq_defs[ref_entry] = ref

            if bool(call_maybe_s(st, "IsAssembly", ref)):
                collect_defs(ref, seen)

    seen: Set[str] = set()
    for i in range(1, roots.Length() + 1):
        collect_defs(roots.Value(i), seen)

    print("Unique definitions found:", len(uniq_defs), flush=True)

    # STL writer + mesher
    BRepMesh = _mod("BRepMesh")
    StlAPI = _mod("StlAPI")

    writer = StlAPI.StlAPI_Writer()
    # Optional: writer.ASCIIMode = True  # if you want ASCII STL (bigger)

    manifest = []
    count = 0

    for ref_entry, ref_lab in uniq_defs.items():
        if def_limit and count >= def_limit:
            break

        # Get the definition shape from the label
        shape = call_maybe_s(st, "GetShape", ref_lab)

        # Mesh it (linear deflection in model units; tune as needed)
        # Smaller deflection => finer mesh, larger files.
        BRepMesh.BRepMesh_IncrementalMesh(shape, 0.5, False, 0.5, True)

        ref_name = get_label_name(ref_lab)
        base = _safe_filename(ref_name) if ref_name else ref_entry
        out_path = stl_dir / f"{base}.stl"

        writer.Write(shape, str(out_path))

        manifest.append({
            "ref_def": ref_entry,
            "ref_name": ref_name,
            "stl_path": str(out_path).replace(str(out_dir_p), "").lstrip("/"),
        })

        count += 1
        if count % 50 == 0:
            print("Exported:", count, flush=True)

    man_path = out_dir_p / "stl_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote:", man_path, flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python export_stl_xcaf.py /in/model.step [/out] [def_limit]")
    step = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/out"
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(step, out, lim)
