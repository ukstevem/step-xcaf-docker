#!/usr/bin/env python3
"""
read_step_xcaf.py

Reads a STEP file into an XCAF (OCAF) document using OCP/OCCT, then extracts an
*expanded* occurrence list.

Key points:
- XCAF represents assemblies with references:
    occurrence(label) --> referred definition(label)
  Children typically live under the *definition* label, not the occurrence.
  So we expand definitions while accumulating transforms.
- To preserve names, STEPCAFControl_Reader must have NameMode enabled.
- Names are usually on referred definition labels; occurrences may be blank/NAUO.

Output: /out/xcaf_instances.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set


def _mod(name: str):
    return __import__(f"OCP.{name}", fromlist=[name])


def call_maybe_s(obj: Any, method: str, *args):
    """Try obj.method(*args) then obj.method_s(*args) for OCP binding variants."""
    fn = getattr(obj, method, None)
    if callable(fn):
        return fn(*args)
    fn_s = getattr(obj, method + "_s", None)
    if callable(fn_s):
        return fn_s(*args)
    raise AttributeError(f"{type(obj).__name__} has no {method} / {method}_s")


def _to_py_str(x) -> str:
    for m in ("ToExtString", "ToCString", "PrintToString"):
        fn = getattr(x, m, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return str(x)


def get_label_name(lab) -> str:
    """Return TDataStd_Name on a label, if present."""
    TDataStd = _mod("TDataStd")
    name_attr = TDataStd.TDataStd_Name()

    getid = getattr(TDataStd.TDataStd_Name, "GetID", None) or getattr(
        TDataStd.TDataStd_Name, "GetID_s", None
    )
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
        return _to_py_str(name_attr.Get())
    except Exception:
        return ""


def _get_app():
    XCAFApp = _mod("XCAFApp")
    return call_maybe_s(XCAFApp.XCAFApp_Application, "GetApplication")


def _new_doc(app):
    # TDocStd_Document expects a TCollection_ExtendedString in OCP builds
    TDocStd = _mod("TDocStd")
    TCollection = _mod("TCollection")

    fmt = TCollection.TCollection_ExtendedString("MDTV-XCAF")
    doc = TDocStd.TDocStd_Document(fmt)
    call_maybe_s(app, "NewDocument", fmt, doc)
    _ = doc.Main()  # ensure Main() is available
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

    # CRUCIAL: preserve names into XCAF label attributes
    call_maybe_s(r, "SetNameMode", True)
    # Optional (sometimes helpful):
    # call_maybe_s(r, "SetPropsMode", True)
    # call_maybe_s(r, "SetColorMode", True)
    # call_maybe_s(r, "SetLayerMode", True)

    status = r.ReadFile(step_path)
    if status != IFSelect.IFSelect_RetDone:
        raise RuntimeError(f"ReadFile failed: {status}")

    ok = r.Transfer(doc)
    if not ok:
        raise RuntimeError("Transfer(doc) returned False")


def _label_entry(lab) -> str:
    """Stable label entry string like '0:1:2'."""
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


def _trsf_to_rows(trsf) -> List[List[float]]:
    """Return 3x4 matrix rows (R|t)."""
    m = trsf.VectorialPart()
    t = trsf.TranslationPart()
    tx, ty, tz = float(t.X()), float(t.Y()), float(t.Z())
    rows: List[List[float]] = []
    for r in (1, 2, 3):
        rows.append(
            [
                float(m.Value(r, 1)),
                float(m.Value(r, 2)),
                float(m.Value(r, 3)),
                [tx, ty, tz][r - 1],
            ]
        )
    return rows


def _trsf_identity():
    gp = _mod("gp")
    return gp.gp_Trsf()  # identity


def _trsf_mul(a, b):
    """Return a*b (apply b after a) in OCCT convention."""
    try:
        return a.Multiplied(b)
    except Exception:
        gp = _mod("gp")
        c = gp.gp_Trsf(a)
        c.Multiply(b)
        return c


def main(step_path: str, out_dir: str = "/out"):
    step_path_p = Path(step_path)
    if not step_path_p.exists():
        raise FileNotFoundError(step_path_p)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    app = _get_app()
    doc = _new_doc(app)
    st = _shape_tool(doc)

    print("Reading STEP:", step_path_p, flush=True)
    _read_step_into_doc(str(step_path_p), doc)

    # Helps OCCT build assembly relations
    call_maybe_s(st, "UpdateAssemblies")

    TDF = _mod("TDF")
    roots = TDF.TDF_LabelSequence()
    call_maybe_s(st, "GetFreeShapes", roots)
    print("FreeShapes:", roots.Length(), flush=True)

    occ_rows: List[Dict[str, Any]] = []
    recursion_stack: Set[str] = set()
    MAX_DEPTH = 200

    def expand_definition(def_label, parent_def_entry: str, depth: int, acc_trsf):
        if depth > MAX_DEPTH:
            return

        def_entry = _label_entry(def_label)
        if def_entry in recursion_stack:
            return  # cycle guard

        recursion_stack.add(def_entry)
        try:
            kids = TDF.TDF_LabelSequence()
            call_maybe_s(st, "GetComponents", def_label, kids)

            for i in range(1, kids.Length() + 1):
                child_occ = kids.Value(i)

                ref = TDF.TDF_Label()
                has_ref = bool(call_maybe_s(st, "GetReferredShape", child_occ, ref))

                loc = call_maybe_s(st, "GetLocation", child_occ)
                local_trsf = loc.Transformation()
                glob_trsf = _trsf_mul(acc_trsf, local_trsf)

                occ_rows.append(
                    {
                        "depth": int(depth),
                        "parent_def": parent_def_entry,
                        "child_occ": _label_entry(child_occ),
                        "has_ref": has_ref,
                        "ref_def": _label_entry(ref) if has_ref else "",
                        "m_local": _trsf_to_rows(local_trsf),
                        "m_global": _trsf_to_rows(glob_trsf),
                        "occ_name": get_label_name(child_occ),
                        "ref_name": get_label_name(ref) if has_ref else "",
                    }
                )

                # Recurse into referred definition if it is an assembly
                if has_ref and bool(call_maybe_s(st, "IsAssembly", ref)):
                    expand_definition(ref, _label_entry(ref), depth + 1, glob_trsf)

        finally:
            recursion_stack.remove(def_entry)

    for i in range(1, roots.Length() + 1):
        root_def = roots.Value(i)
        expand_definition(root_def, _label_entry(root_def), 0, _trsf_identity())

    print("Expanded occurrences found:", len(occ_rows), flush=True)

    # Summary: unique definitions
    def_counts: Dict[str, int] = {}
    for row in occ_rows:
        key = row["ref_def"] if row["ref_def"] else "(no-ref)"
        def_counts[key] = def_counts.get(key, 0) + 1

    no_ref = def_counts.get("(no-ref)", 0)
    unique_defs = len([k for k in def_counts.keys() if k != "(no-ref)"])
    print("Unique referred definitions:", unique_defs, flush=True)
    print("Occurrences with no referred shape:", no_ref, flush=True)

    top = sorted(
        [(v, k) for k, v in def_counts.items() if k != "(no-ref)"], reverse=True
    )[:10]
    print("Top repeated defs (count, ref_label):", flush=True)
    for v, k in top:
        print(f"  {v:6d}  {k}", flush=True)

    out_json = out_dir_p / "xcaf_instances.json"
    out_json.write_text(json.dumps(occ_rows, indent=2), encoding="utf-8")
    print("Wrote:", out_json, flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python read_step_xcaf.py /in/model.step [/out]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/out")
