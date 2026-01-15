#!/usr/bin/env python3
"""read_step_xcaf.py

STEP → XCAF → single source-of-truth JSON (Step 1)

Creates /out/xcaf_instances.json capturing:
  - full assembly tree (parent_def → child occurrences)
  - what each occurrence refers to (definition)
  - local + global transforms
  - basic per-definition shape facts (shape_kind, solid_count, bbox) FOR PARTS
  - stable per-definition signatures (def_sig/def_sig_free) via shared brep_signature.py
  - fast indexes (children_by_parent_def, children_by_parent_occ, occs_by_ref_def, leaf_occ_ids)

No STL/PNG generation happens here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Single source of truth for definition signatures (shared with Step 2).
from brep_signature import DEF_SIG_ALGO, compute_def_sig, compute_def_sig_free


# -----------------------------------------------------------------------------
# .env loader (no external deps)
# -----------------------------------------------------------------------------

def _parse_env_line(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # allow: export KEY=VAL
    if s.lower().startswith("export "):
        s = s[7:].strip()
    if "=" not in s:
        return None
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        return None
    # strip quotes
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    return k, v


def _load_dotenv(path: Path, *, override: bool = False) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    - If override=False, existing os.environ values are preserved.
    - Returns dict of keys actually set (after applying override rule).
    """
    if not path.exists() or not path.is_file():
        return {}

    out: Dict[str, str] = {}
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    for raw in txt.splitlines():
        parsed = _parse_env_line(raw)
        if parsed is None:
            continue
        k, v = parsed
        if (not override) and (k in os.environ):
            continue
        os.environ[k] = v
        out[k] = v
    return out


def _env_float(name: str, default: float) -> float:
    s = os.environ.get(name, "").strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    s = os.environ.get(name, "").strip()
    if not s:
        return int(default)
    try:
        return int(float(s))
    except Exception:
        return int(default)


# -----------------------------------------------------------------------------
# Deterministic bbox quantization (must match Step 2 when used)
# -----------------------------------------------------------------------------

def _q_tol_mm(x: float, tol_mm: float) -> float:
    """Quantize x to nearest tol_mm using deterministic HALF_UP rounding."""
    if tol_mm <= 0.0:
        return float(x)
    dx = Decimal(str(float(x)))
    dt = Decimal(str(float(tol_mm)))
    q = (dx / dt).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * dt
    return float(q)


# -----------------------------------------------------------------------------
# OCP compatibility helpers (bindings differ between builds)
# -----------------------------------------------------------------------------

def _mod(name: str):
    return __import__(f"OCP.{name}", fromlist=[name])


def call_maybe_s(obj: Any, method: str, *args):
    """
    Try obj.method(*args) then obj.method_s(*args) for OCP binding variants.

    Important: if obj.method exists but raises TypeError due to SWIG signature
    mismatch, fall back to method_s.
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
        return _to_py_str(name_attr.Get()).strip()
    except Exception:
        return ""


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


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _get_label_shape(shape_tool, lab):
    """Best-effort: return TopoDS_Shape for a label, or None."""
    try:
        shp = call_maybe_s(shape_tool, "GetShape", lab)
        if shp is not None and not shp.IsNull():
            return shp
    except Exception:
        pass

    # Some builds expose an overload that fills a provided TopoDS_Shape.
    try:
        TopoDS = _mod("TopoDS")
        shp = TopoDS.TopoDS_Shape()
        ok = bool(call_maybe_s(shape_tool, "GetShape", lab, shp))
        if ok and (shp is not None) and (not shp.IsNull()):
            return shp
    except Exception:
        pass

    return None


def _shape_kind(shape) -> str:
    if shape is None or shape.IsNull():
        return "EMPTY"
    TopAbs = _mod("TopAbs")
    st = shape.ShapeType()
    if st == TopAbs.TopAbs_SOLID:
        return "SOLID"
    if st == TopAbs.TopAbs_COMPOUND:
        return "COMPOUND"
    if st == TopAbs.TopAbs_SHELL:
        return "SHELL"
    return "UNKNOWN"


def _solid_count(shape) -> int:
    if shape is None or shape.IsNull():
        return 0
    TopExp = _mod("TopExp")
    TopAbs = _mod("TopAbs")
    exp = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_SOLID)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return int(n)


def _shape_bbox(shape, *, tol_mm: float) -> Optional[Dict[str, List[float]]]:
    if shape is None or shape.IsNull():
        return None

    Bnd = _mod("Bnd")
    BRepBndLib = _mod("BRepBndLib")
    box = Bnd.Bnd_Box()

    # Try free-function + class-wrapper + _s variants deterministically.
    cands = []

    # module-level
    for nm in ("Add", "Add_s"):
        fn = getattr(BRepBndLib, nm, None)
        if callable(fn):
            cands.append(fn)

    # class-wrapper
    cls = getattr(BRepBndLib, "BRepBndLib", None)
    if cls is not None:
        for nm in ("Add", "Add_s"):
            fn = getattr(cls, nm, None)
            if callable(fn):
                cands.append(fn)

    if not cands:
        return None

    added = False
    for fn in cands:
        # preferred overload first: (shape, box, useTriangulation)
        try:
            fn(shape, box, True)
            added = True
            break
        except TypeError:
            pass
        except Exception:
            pass

        # fallback overload: (shape, box)
        try:
            fn(shape, box)
            added = True
            break
        except TypeError:
            pass
        except Exception:
            pass

    if not added:
        return None

    try:
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    except Exception:
        return None

    mn = [_q_tol_mm(float(xmin), tol_mm), _q_tol_mm(float(ymin), tol_mm), _q_tol_mm(float(zmin), tol_mm)]
    mx = [_q_tol_mm(float(xmax), tol_mm), _q_tol_mm(float(ymax), tol_mm), _q_tol_mm(float(zmax), tol_mm)]
    sz = [
        _q_tol_mm(float(xmax - xmin), tol_mm),
        _q_tol_mm(float(ymax - ymin), tol_mm),
        _q_tol_mm(float(zmax - zmin), tol_mm),
    ]
    return {"min": mn, "max": mx, "size": sz}



def _shape_massprops(shape) -> Optional[Dict[str, float]]:
    """Best-effort mass props (NOT used for def_sig). Kept for optional reporting only."""
    if shape is None or shape.IsNull():
        return None

    try:
        GProp = _mod("GProp")
        BRepGProp = _mod("BRepGProp")

        props_v = GProp.GProp_GProps()
        props_s = GProp.GProp_GProps()

        # OCP binding variants: VolumeProperties vs VolumeProperties_s
        call_maybe_s(BRepGProp, "VolumeProperties", shape, props_v)
        call_maybe_s(BRepGProp, "SurfaceProperties", shape, props_s)

        vol = float(props_v.Mass())
        area = float(props_s.Mass())

        # Guard: sometimes you get zeros for weird shapes
        return {"volume": vol, "area": area}
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Transform helpers
# -----------------------------------------------------------------------------

def _trsf_identity():
    gp = _mod("gp")
    return gp.gp_Trsf()


def _as_trsf(x):
    """Normalize gp_Trsf / TopLoc_Location -> gp_Trsf."""
    gp = _mod("gp")
    if x is None:
        return gp.gp_Trsf()

    # gp_Trsf already
    if hasattr(x, "VectorialPart") and hasattr(x, "TranslationPart"):
        return x

    # TopLoc_Location -> gp_Trsf
    if hasattr(x, "Transformation"):
        try:
            return x.Transformation()
        except Exception:
            pass
        try:
            return x.Transformation_s()
        except Exception:
            pass

    return gp.gp_Trsf()

def _trsf_mul(a, b):
    """Multiply two transforms (gp_Trsf or TopLoc_Location)."""
    a_t = _as_trsf(a)
    b_t = _as_trsf(b)
    try:
        return a_t.Multiplied(b_t)
    except Exception:
        gp = _mod("gp")
        c = gp.gp_Trsf(a_t)
        c.Multiply(b_t)
        return c


def _trsf_to_4x4(trsf) -> List[float]:
    """Convert gp_Trsf or TopLoc_Location to row-major 4x4 list."""
    t = _as_trsf(trsf)  # <-- CRITICAL: normalize TopLoc_Location -> gp_Trsf
    m = t.VectorialPart()
    tr = t.TranslationPart()
    tx, ty, tz = float(tr.X()), float(tr.Y()), float(tr.Z())
    return [
        float(m.Value(1, 1)), float(m.Value(1, 2)), float(m.Value(1, 3)), tx,
        float(m.Value(2, 1)), float(m.Value(2, 2)), float(m.Value(2, 3)), ty,
        float(m.Value(3, 1)), float(m.Value(3, 2)), float(m.Value(3, 3)), tz,
        0.0, 0.0, 0.0, 1.0,
    ]

def _sorted_labels_from_seq(seq) -> List[Any]:
    """Return labels from a TDF_LabelSequence in deterministic order."""
    labs = [seq.Value(i) for i in range(1, seq.Length() + 1)]
    labs.sort(key=_label_entry)
    return labs


def _tool_versions() -> Dict[str, str]:
    vers: Dict[str, str] = {"python": sys.version.split()[0]}
    try:
        import OCP  # type: ignore
        v = getattr(OCP, "__version__", None)
        if v:
            vers["ocp"] = str(v)
    except Exception:
        pass
    try:
        Std = _mod("Standard")
        fn = getattr(Std, "Standard_Version", None) or getattr(Std, "Standard_Version_s", None)
        if callable(fn):
            vers["occt"] = _to_py_str(fn())
    except Exception:
        pass
    return vers


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_xcaf_instances_json(
    step_path: str,
    out_dir: str,
    *,
    with_massprops: bool,
    with_signature: bool,
    strict_signature_collisions: bool = False,
    env_path: Optional[str] = None,
) -> Path:
    step_path_p = Path(step_path)
    if not step_path_p.exists():
        raise FileNotFoundError(step_path_p)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Load .env (but do not override docker --env-file or shell env)
    dotenv_used: Dict[str, str] = {}
    dotenv_sources: List[str] = []
    if env_path:
        p = Path(env_path)
        dotenv_used = _load_dotenv(p, override=False)
        if dotenv_used:
            dotenv_sources.append(str(p))
    else:
        # Common locations in your docker run:
        # - working dir: /app
        # - repo root: /app
        for p in (Path("/app/.env"), Path(".env")):
            used = _load_dotenv(p, override=False)
            if used:
                dotenv_sources.append(str(p))
                dotenv_used.update(used)

    # Read shared settings (Step 1 + Step 2 must match)
    bbox_tol_mm = _env_float("BBOX_TOL_MM", 0.0)
    use_sig_free = _env_int("USE_SIG_FREE", 1)
    skip_chirality = _env_int("SKIP_CHIRALITY", 1)

    analysis_started_utc = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    app = _get_app()
    doc = _new_doc(app)
    st = _shape_tool(doc)

    print("Reading STEP:", step_path_p, flush=True)
    _read_step_into_doc(str(step_path_p), doc)

    call_maybe_s(st, "UpdateAssemblies")

    TDF = _mod("TDF")
    roots = TDF.TDF_LabelSequence()
    call_maybe_s(st, "GetFreeShapes", roots)
    free_n = roots.Length()
    print("FreeShapes:", free_n, flush=True)

    warnings: List[Dict[str, Any]] = []
    definitions: Dict[str, Dict[str, Any]] = {}
    defs_in_traversal_order: List[str] = []
    occurrences: Dict[str, Dict[str, Any]] = {}

    children_by_parent_def: Dict[str, List[str]] = {}
    children_by_parent_occ: Dict[str, List[str]] = {}
    occs_by_ref_def: Dict[str, List[str]] = {}
    leaf_occ_ids: List[str] = []

    DEF_NOREF = "DEF-NOREF"

    def ensure_def(def_label, *, force_id: Optional[str] = None) -> str:
        def_id = force_id or _label_entry(def_label)
        if def_id in definitions:
            return def_id

        name = get_label_name(def_label)

        # Assemblies: don't compute shape facts (prevents root solid_count insanity)
        is_asm = False
        try:
            is_asm = bool(call_maybe_s(st, "IsAssembly", def_label))
        except Exception:
            is_asm = False

        shp = None
        has_shape = False
        shape_kind = "EMPTY"
        solid_count = 0
        bbox = None
        mp = None

        if not is_asm:
            shp = _get_label_shape(st, def_label)
            has_shape = bool(shp is not None and (not shp.IsNull()))
            shape_kind = _shape_kind(shp)
            solid_count = _solid_count(shp) if has_shape else 0
            bbox = _shape_bbox(shp, tol_mm=bbox_tol_mm) if has_shape else None
            if with_massprops and has_shape:
                mp = _shape_massprops(shp)

        # Stable B-Rep signature (shared module; Step 2 must compute identically).
        def_sig = None
        def_sig_free = None
        def_sig_algo = None
        if with_signature and has_shape:
            def_sig_algo = DEF_SIG_ALGO
            def_sig = compute_def_sig(shp)
            def_sig_free = compute_def_sig_free(shp)

        rec: Dict[str, Any] = {
            "def_id": def_id,
            "name": name,
            "has_shape": bool(has_shape),
            "shape_kind": shape_kind,
            "solid_count": int(solid_count),
            "qty_total": 0,
            "assets": {"stl": None, "png": None},
            "def_sig": def_sig,
            "def_sig_free": def_sig_free,
            "def_sig_algo": def_sig_algo,
        }
        if bbox is not None:
            rec["bbox"] = bbox
        if mp is not None:
            rec["massprops"] = mp
        if not rec["name"]:
            warnings.append({"type": "missing_name", "scope": "definition", "def_id": def_id})

        definitions[def_id] = rec
        defs_in_traversal_order.append(def_id)
        return def_id

    def _ensure_noref_def():
        if DEF_NOREF in definitions:
            return
        definitions[DEF_NOREF] = {
            "def_id": DEF_NOREF,
            "name": "",
            "has_shape": False,
            "shape_kind": "EMPTY",
            "solid_count": 0,
            "qty_total": 0,
            "assets": {"stl": None, "png": None},
            "def_sig": None,
            "def_sig_free": None,
            "def_sig_algo": None,
        }
        defs_in_traversal_order.append(DEF_NOREF)

    recursion_stack: Set[str] = set()
    MAX_DEPTH = 200

    def add_occurrence(
        *,
        occ_id: str,
        parent_occ: Optional[str],
        occ_label,
        parent_def_id: str,
        ref_def_id: str,
        depth: int,
        local_trsf,
        global_trsf,
        is_leaf: bool,
    ) -> None:
        if occ_id in occurrences:
            warnings.append(
                {
                    "type": "duplicate_occ_id",
                    "occ_id": occ_id,
                    "parent_def": parent_def_id,
                    "ref_def": ref_def_id,
                }
            )
            return

        occ_name = get_label_name(occ_label) or definitions.get(ref_def_id, {}).get("name", "")

        occurrences[occ_id] = {
            "occ_id": occ_id,
            "parent_def": parent_def_id,
            "parent_occ": parent_occ,
            "ref_def": ref_def_id,
            "name": occ_name,
            "depth": int(depth),
            "local_xform": _trsf_to_4x4(local_trsf),
            "global_xform": _trsf_to_4x4(global_trsf),
            "is_leaf": bool(is_leaf),
        }

        children_by_parent_def.setdefault(parent_def_id, []).append(occ_id)
        if parent_occ is not None:
            children_by_parent_occ.setdefault(parent_occ, []).append(occ_id)
        occs_by_ref_def.setdefault(ref_def_id, []).append(occ_id)
        if is_leaf:
            leaf_occ_ids.append(occ_id)

    def expand_definition(def_label, *, parent_def_id: str, parent_occ_id: str, depth: int, acc_trsf):
        """Expand a definition into child occurrences.

        occ_id must be unique per *instance path*, not per child label entry.
        We use: occ_id = f"{parent_occ_id}:{child_pos}" where child_pos is stable due to sorting.
        """
        if depth > MAX_DEPTH:
            warnings.append({"type": "max_depth_exceeded", "def_id": _label_entry(def_label), "max_depth": MAX_DEPTH})
            return

        def_id = ensure_def(def_label)
        if def_id in recursion_stack:
            warnings.append({"type": "cycle_detected", "def_id": def_id})
            return

        recursion_stack.add(def_id)
        try:
            kids = TDF.TDF_LabelSequence()
            call_maybe_s(st, "GetComponents", def_label, kids)
            sorted_kids = _sorted_labels_from_seq(kids)

            for child_pos, child_occ in enumerate(sorted_kids, start=1):
                occ_id = f"{parent_occ_id}:{child_pos}"

                ref = TDF.TDF_Label()
                has_ref = bool(call_maybe_s(st, "GetReferredShape", child_occ, ref))

                if has_ref:
                    ref_def_id = ensure_def(ref)
                else:
                    _ensure_noref_def()
                    ref_def_id = DEF_NOREF
                    warnings.append(
                        {
                            "type": "no_referred_shape",
                            "occ_id": occ_id,
                            "occ_label_entry": _label_entry(child_occ),
                            "parent_def": parent_def_id,
                        }
                    )

                # GetLocation returns TopLoc_Location in OCP.
                # Keep it as-is; _as_trsf handles conversion.
                try:
                    local_loc = call_maybe_s(st, "GetLocation", child_occ)
                    local_trsf = local_loc
                except Exception as e:
                    local_trsf = _trsf_identity()
                    warnings.append(
                        {
                            "type": "transform_failed",
                            "occ_id": occ_id,
                            "occ_label_entry": _label_entry(child_occ),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

                glob_trsf = _trsf_mul(acc_trsf, local_trsf)

                is_asm = False
                if has_ref:
                    try:
                        is_asm = bool(call_maybe_s(st, "IsAssembly", ref))
                    except Exception:
                        is_asm = False

                add_occurrence(
                    occ_id=occ_id,
                    parent_occ=parent_occ_id,
                    occ_label=child_occ,
                    parent_def_id=parent_def_id,
                    ref_def_id=ref_def_id,
                    depth=depth,
                    local_trsf=local_trsf,
                    global_trsf=glob_trsf,
                    is_leaf=(not is_asm),
                )

                if has_ref and is_asm:
                    expand_definition(
                        ref,
                        parent_def_id=ref_def_id,
                        parent_occ_id=occ_id,
                        depth=depth + 1,
                        acc_trsf=glob_trsf,
                    )

        finally:
            recursion_stack.remove(def_id)

    # Root handling
    if free_n <= 0:
        raise RuntimeError("No FreeShapes found (empty or unreadable STEP?)")

    if free_n == 1:
        root_label = roots.Value(1)
        root_def_id = ensure_def(root_label)
        expand_definition(
            root_label,
            parent_def_id=root_def_id,
            parent_occ_id=root_def_id,
            depth=0,
            acc_trsf=_trsf_identity(),
        )
    else:
        root_def_id = "A-ROOT"
        if root_def_id not in definitions:
            definitions[root_def_id] = {
                "def_id": root_def_id,
                "name": step_path_p.stem,
                "has_shape": False,
                "shape_kind": "EMPTY",
                "solid_count": 0,
                "qty_total": 0,
                "assets": {"stl": None, "png": None},
                "def_sig": None,
                "def_sig_free": None,
                "def_sig_algo": None,
            }
            defs_in_traversal_order.append(root_def_id)

        warnings.append({"type": "multiple_free_shapes", "count": free_n})

        for i, root_label in enumerate(_sorted_labels_from_seq(roots), start=1):
            real_root_def_id = ensure_def(root_label)
            occ_id = f"{root_def_id}:{i}"

            occurrences[occ_id] = {
                "occ_id": occ_id,
                "parent_def": root_def_id,
                "parent_occ": None,
                "ref_def": real_root_def_id,
                "name": definitions.get(real_root_def_id, {}).get("name", "") or real_root_def_id,
                "depth": 0,
                "local_xform": _trsf_to_4x4(_trsf_identity()),
                "global_xform": _trsf_to_4x4(_trsf_identity()),
                "is_leaf": False,
            }
            children_by_parent_def.setdefault(root_def_id, []).append(occ_id)
            # NOTE: legacy behaviour retained for compatibility:
            children_by_parent_occ.setdefault(root_def_id, []).append(occ_id)
            occs_by_ref_def.setdefault(real_root_def_id, []).append(occ_id)

            expand_definition(
                root_label,
                parent_def_id=real_root_def_id,
                parent_occ_id=occ_id,
                depth=1,
                acc_trsf=_trsf_identity(),
            )

    # qty_total
    for occ in occurrences.values():
        ref_def = occ.get("ref_def")
        if ref_def in definitions:
            definitions[ref_def]["qty_total"] = int(definitions[ref_def].get("qty_total", 0) + 1)

    # Guard checks (warnings, not crashes)
    for occ_id, occ in occurrences.items():
        ref_def = occ.get("ref_def")
        if ref_def not in definitions:
            warnings.append({"type": "dangling_ref_def", "occ_id": occ_id, "ref_def": ref_def})

    for parent_def, occ_ids in children_by_parent_def.items():
        for oid in occ_ids:
            if occurrences.get(oid, {}).get("parent_def") != parent_def:
                warnings.append({"type": "children_index_mismatch", "parent_def": parent_def, "occ_id": oid})

    # --- Signature collision check ---------------------------------------------
    signature_collisions: List[Dict[str, Any]] = []
    if with_signature:
        sig_map: Dict[str, List[str]] = {}
        for def_id, d in definitions.items():
            if not bool(d.get("has_shape", False)):
                continue
            sig = str(d.get("def_sig") or "")
            if not sig:
                continue
            sig_map.setdefault(sig, []).append(def_id)

        for sig, def_ids in sig_map.items():
            if len(def_ids) > 1:
                def_ids_sorted = sorted(def_ids)
                signature_collisions.append(
                    {
                        "def_sig": sig,
                        "def_ids": def_ids_sorted,
                        "names": [str(definitions.get(i, {}).get("name", "") or "") for i in def_ids_sorted],
                    }
                )

        if signature_collisions:
            print(f"WARNING: signature collisions detected: {len(signature_collisions)}", flush=True)
            if strict_signature_collisions:
                raise RuntimeError("Signature collisions detected (strict mode).")

    analysis_finished_utc = datetime.now(timezone.utc)
    runtime_s = time.perf_counter() - t0

    # --- Deterministic ordering for JSON output -------------------------------
    for _k, _lst in children_by_parent_def.items():
        _lst.sort()
    for _k, _lst in children_by_parent_occ.items():
        _lst.sort()
    for _k, _lst in occs_by_ref_def.items():
        _lst.sort()
    leaf_occ_ids.sort()

    definitions_out = {k: definitions[k] for k in sorted(definitions.keys())}
    occurrences_out = {k: occurrences[k] for k in sorted(occurrences.keys())}
    children_by_parent_def_out = {k: children_by_parent_def[k] for k in sorted(children_by_parent_def.keys())}
    children_by_parent_occ_out = {k: children_by_parent_occ[k] for k in sorted(children_by_parent_occ.keys())}
    occs_by_ref_def_out = {k: occs_by_ref_def[k] for k in sorted(occs_by_ref_def.keys())}

    out = {
        "meta": {
            "step_filename": step_path_p.name,
            "created_utc": analysis_finished_utc.isoformat(timespec="seconds"),
            "tool_versions": _tool_versions(),
            "counts": {"defs": len(definitions), "occs": len(occurrences), "leaf_occs": len(leaf_occ_ids)},
            "analysis_started_utc": analysis_started_utc.isoformat(timespec="seconds"),
            "analysis_finished_utc": analysis_finished_utc.isoformat(timespec="seconds"),
            "runtime_seconds": round(float(runtime_s), 3),
            "with_signature": bool(with_signature),
            "signature_algo": (DEF_SIG_ALGO if with_signature else ""),
            "signature_collisions_count": len(signature_collisions),
            "signature_collisions": signature_collisions,
            "env": {
                "dotenv_sources": dotenv_sources,
                "dotenv_loaded_keys": sorted(dotenv_used.keys()),
                "BBOX_TOL_MM": float(bbox_tol_mm),
                "USE_SIG_FREE": int(use_sig_free),
                "SKIP_CHIRALITY": int(skip_chirality),
            },
        },
        "root_def": root_def_id,
        "definitions": definitions_out,
        "occurrences": occurrences_out,
        "indexes": {
            "children_by_parent_def": children_by_parent_def_out,
            "children_by_parent_occ": children_by_parent_occ_out,
            "occs_by_ref_def": occs_by_ref_def_out,
            "leaf_occ_ids": leaf_occ_ids,
            "defs_in_traversal_order": defs_in_traversal_order,
        },
        "warnings": warnings,
    }

    mp_count = sum(1 for d in definitions.values() if "massprops" in d)
    print(f"massprops defs  : {mp_count}", flush=True)

    out_json = out_dir_p / "xcaf_instances.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    # --- Synopsis -------------------------------------------------------------
    print("\n=== Step 1 Synopsis ===", flush=True)
    print(f"Analysis started : {analysis_started_utc.isoformat(timespec='seconds')}", flush=True)
    print(f"Analysis finished: {analysis_finished_utc.isoformat(timespec='seconds')}", flush=True)
    print(f"Runtime (s)      : {round(float(runtime_s), 3)}", flush=True)
    print(f"Definitions      : {len(definitions)}", flush=True)
    print(f"Occurrences      : {len(occurrences)}", flush=True)
    print(f"Leaf occurrences : {len(leaf_occ_ids)}", flush=True)
    print(f"Warnings         : {len(warnings)}", flush=True)
    print(f"BBOX_TOL_MM      : {bbox_tol_mm}", flush=True)
    if dotenv_sources:
        print(f".env sources     : {', '.join(dotenv_sources)}", flush=True)
    if with_signature:
        print(f"Sig collisions   : {len(signature_collisions)}", flush=True)
        print(f"Sig algo         : {DEF_SIG_ALGO}", flush=True)

    print("\nWrote:", out_json, flush=True)
    print(f"Definitions: {len(definitions)} | Occurrences: {len(occurrences)} | Leaf occs: {len(leaf_occ_ids)}", flush=True)
    return out_json


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="STEP→XCAF→xcaf_instances.json (Step 1)")
    p.add_argument("step_path", help="Input STEP file (e.g. /in/model.step)")
    p.add_argument("out_dir", nargs="?", default="/out", help="Output dir (default: /out)")
    p.add_argument("--with_massprops", action="store_true", help="Include volume/area (best-effort; not used for def_sig)")

    sig_group = p.add_mutually_exclusive_group()
    sig_group.add_argument("--with_signature", dest="with_signature", action="store_true", help="Compute def_sig for shaped definitions (default)")
    sig_group.add_argument("--no_signature", dest="with_signature", action="store_false", help="Disable def_sig computation")
    p.set_defaults(with_signature=True)

    p.add_argument("--strict_signature_collisions", action="store_true", help="Crash if def_sig collisions are detected")

    # Optional explicit .env path (if omitted, tries /app/.env then ./.env)
    p.add_argument("--env", dest="env_path", default="", help="Path to .env (default: /app/.env then ./.env)")

    ns = p.parse_args(argv)

    # Env default for massprops (CLI overrides env)
    env_massprops = _env_int("WITH_MASSPROPS", 0)
    with_massprops_effective = bool(ns.with_massprops) or (env_massprops == 1)

    build_xcaf_instances_json(
        ns.step_path,
        ns.out_dir,
        with_massprops=with_massprops_effective,
        with_signature=bool(ns.with_signature),
        strict_signature_collisions=bool(ns.strict_signature_collisions),
        env_path=(ns.env_path.strip() or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
