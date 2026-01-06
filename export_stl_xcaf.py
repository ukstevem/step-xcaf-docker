#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


# -----------------------------
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

    try:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
