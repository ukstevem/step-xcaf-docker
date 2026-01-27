from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


# ----------------------------
# Config (env-driven)
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent
UI_DIR = ROOT_DIR / "ui"
RUNS_DIR = ROOT_DIR / "ui_runs"

DEFAULT_ANALYZE_CMD = 'python /repo/read_step_xcaf.py --with_massprops "{step}" "{out}"'
ANALYZE_CMD = os.environ.get("ANALYZE_CMD", DEFAULT_ANALYZE_CMD).strip()

# STL outputs (relative to run dir)
ASSEMBLY_STL_REL = os.environ.get("ASSEMBLY_STL_REL", "assembly.stl")
ASSEMBLY_PREVIEW_STL_REL = os.environ.get("ASSEMBLY_PREVIEW_STL_REL", "assembly_preview.stl")
MAX_ASSEMBLY_STL_MB = float(os.environ.get("MAX_ASSEMBLY_STL_MB", "200"))
PREVIEW_MESH_DEFLECTION = float(os.environ.get("PREVIEW_MESH_DEFLECTION", "8.0"))
FULL_MESH_DEFLECTION = float(os.environ.get("FULL_MESH_DEFLECTION", "3.0"))

# Metadata files (relative to run dir)
IMPORT_PACK_REL = os.environ.get("IMPORT_PACK_REL", "import_pack.json")
VIEWER_META_REL = os.environ.get("VIEWER_META_REL", "viewer_meta.json")
XCAF_INSTANCES_REL = os.environ.get("XCAF_INSTANCES_REL", "xcaf_instances.json")

# Preflight output (fast preview)
PREFLIGHT_PACK_REL = os.environ.get("PREFLIGHT_PACK_REL", "preflight_pack.json")
ORIENTATION_REL = os.environ.get("ORIENTATION_REL", "orientation.json")
ANALYSIS_PACK_REL = os.environ.get("ANALYSIS_PACK_REL", "analysis_pack.json")

PREFLIGHT_IMG_SIZE = int(os.environ.get("PREFLIGHT_IMG_SIZE", "560"))
PREFLIGHT_MESH_DEFLECTION = float(os.environ.get("PREFLIGHT_MESH_DEFLECTION", "18.0"))
PREFLIGHT_TARGET_STL_MB = float(os.environ.get("PREFLIGHT_TARGET_STL_MB", "40"))
PREFLIGHT_MAX_STL_MB = float(os.environ.get("PREFLIGHT_MAX_STL_MB", "120"))
PREFLIGHT_MAX_DEFLECTION = float(os.environ.get("PREFLIGHT_MAX_DEFLECTION", "250.0"))

MAX_TOPLEVEL = 2000  # guardrail


app = FastAPI(title="STEP UI Starter")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")

# ----------------------------
# Step 2 (fast XCAF + tree)
# ----------------------------
DEFAULT_STEP2_CMD = 'python /repo/read_step_xcaf.py "{step}" "{out}"'
STEP2_CMD = os.environ.get("STEP2_CMD", DEFAULT_STEP2_CMD).strip()

OCC_TREE_REL = os.environ.get("OCC_TREE_REL", "occurrence_tree.json")
STEP2_PACK_REL = os.environ.get("STEP2_PACK_REL", "step2_pack.json")


# ----------------------------
# Progress helpers
# ----------------------------
def _append_progress(run_dir: Path, msg: str) -> None:
    p = run_dir / "progress.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
        f.flush()


@app.get("/api/progress/{run_id}")
async def progress(run_id: str):
    run_dir = RUNS_DIR / run_id
    log_path = run_dir / "progress.log"

    async def event_stream():
        last_pos = 0
        while True:
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as f:
                    f.seek(last_pos)
                    chunk = f.read()
                    last_pos = f.tell()
                if chunk:
                    for line in chunk.splitlines():
                        yield f"data: {line}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ----------------------------
# Models
# ----------------------------
@dataclass(frozen=True)
class TopLevelItem:
    id: str
    name: str
    ref: Optional[str] = None


# ----------------------------
# Run lifecycle
# ----------------------------
def _safe_run_id() -> str:
    return uuid.uuid4().hex[:12]


@app.post("/api/create_run")
def create_run() -> Dict[str, Any]:
    run_id = _safe_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _append_progress(run_dir, "Run created.")
    return {"run_id": run_id}


# ----------------------------
# Small utilities
# ----------------------------
def _mb(p: Path) -> float:
    return float(p.stat().st_size) / (1024.0 * 1024.0)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_orientation(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / ORIENTATION_REL
    if p.exists():
        try:
            d = _read_json(p)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    # default
    return {"plan_source": "plan", "rotation_deg": 0}


def _write_orientation(run_dir: Path, plan_source: str, rotation_deg: int) -> Dict[str, Any]:
    if plan_source not in ("plan", "front", "side"):
        plan_source = "plan"
    if rotation_deg not in (0, 90, 180, 270):
        rotation_deg = 0
    d = {"plan_source": plan_source, "rotation_deg": int(rotation_deg)}
    _write_json(run_dir / ORIENTATION_REL, d)
    return d


# ----------------------------
# Pipeline invocation (existing extractor)
# ----------------------------
def _run_pipeline(step_path: Path, out_dir: Path) -> None:
    if not ANALYZE_CMD:
        raise RuntimeError("ANALYZE_CMD is not set.")

    cmd = ANALYZE_CMD.format(step=str(step_path), out=str(out_dir))
    _append_progress(out_dir, "Starting analysis…")
    _append_progress(out_dir, f"CMD: {cmd}")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        _append_progress(out_dir, line.rstrip())

    rc = proc.wait()
    if rc != 0:
        _append_progress(out_dir, f"FAILED (exit {rc})")
        tail = ""
        try:
            lines = (out_dir / "progress.log").read_text(encoding="utf-8").splitlines()
            tail = "\n".join(lines[-20:])
        except Exception:
            pass
        raise RuntimeError(
            "STEP import failed (pipeline error). "
            f"(exit {rc}).\n\nLast output:\n{tail}"
        )

    _append_progress(out_dir, "Analysis: Done.")


# ----------------------------
# STL helpers (UI-only)
# ----------------------------
def _write_stl_from_step(step_path: Path, stl_path: Path, deflection: float) -> None:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    BRepMesh_IncrementalMesh(shape, float(deflection))

    w = StlAPI_Writer()
    try:
        w.SetASCIIMode(False)
    except Exception:
        pass

    ok = w.Write(shape, str(stl_path))
    if not ok:
        raise RuntimeError(f"Failed to write STL: {stl_path}")


def _ensure_assembly_stls(step_path: Path, run_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    preview_path = run_dir / ASSEMBLY_PREVIEW_STL_REL
    full_path = run_dir / ASSEMBLY_STL_REL

    if not preview_path.exists():
        _write_stl_from_step(step_path, preview_path, PREVIEW_MESH_DEFLECTION)

    if not full_path.exists():
        _write_stl_from_step(step_path, full_path, FULL_MESH_DEFLECTION)

    full_rel = ASSEMBLY_STL_REL if full_path.exists() else None
    prev_rel = ASSEMBLY_PREVIEW_STL_REL if preview_path.exists() else None
    return full_rel, prev_rel


def _write_preflight_stl_adaptive(step_path: Path, stl_path: Path) -> Tuple[float, float]:
    """
    Create a coarse STL aiming for PREFLIGHT_TARGET_STL_MB.
    Bounded loop (8 iterations max).
    """
    if stl_path.exists():
        return (PREFLIGHT_MESH_DEFLECTION, _mb(stl_path))

    defl = float(PREFLIGHT_MESH_DEFLECTION)
    last_mb = 0.0

    for _ in range(8):
        if stl_path.exists():
            try:
                stl_path.unlink()
            except Exception:
                pass

        _write_stl_from_step(step_path, stl_path, defl)
        last_mb = _mb(stl_path)

        if last_mb <= PREFLIGHT_TARGET_STL_MB:
            return (defl, last_mb)

        # Increase deflection more aggressively if it's huge
        if last_mb > PREFLIGHT_MAX_STL_MB:
            defl = min(defl * 2.0, PREFLIGHT_MAX_DEFLECTION)
        else:
            defl = min(defl * 1.4, PREFLIGHT_MAX_DEFLECTION)

    return (defl, float(last_mb))


def _ensure_4view_pngs_from_stl(stl_path: Path, run_dir: Path) -> Dict[str, str]:
    """
    Render plan/front/side/iso from STL using PyVista (off-screen).
    """
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    if not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception:
            pass

    mesh = pv.read(str(stl_path))

    def _render(setup_fn, out_png: Path) -> None:
        pl = pv.Plotter(off_screen=True, window_size=(PREFLIGHT_IMG_SIZE, PREFLIGHT_IMG_SIZE))
        pl.set_background("white")
        pl.add_mesh(mesh, smooth_shading=True)
        pl.enable_parallel_projection()
        setup_fn(pl)
        pl.camera.zoom(1.15)
        pl.show(screenshot=str(out_png), auto_close=True)

    plan_png = run_dir / "view_plan.png"
    front_png = run_dir / "view_front.png"
    side_png = run_dir / "view_side.png"
    iso_png = run_dir / "view_iso.png"

    _render(lambda pl: pl.view_xy(), plan_png)       # look along +Z
    _render(lambda pl: pl.view_yz(), front_png)      # look along +X
    _render(lambda pl: pl.view_xz(), side_png)       # look along +Y
    _render(lambda pl: pl.view_isometric(), iso_png)

    out: Dict[str, str] = {}
    for k, p in (("plan", plan_png), ("front", front_png), ("side", side_png), ("iso", iso_png)):
        if p.exists():
            out[k] = f"/runs/{run_dir.name}/{p.name}"
    return out


def _preflight_pack(step_path: Path, run_dir: Path, upload_name: str, upload_bytes: int) -> Dict[str, Any]:
    """
    Fast preview: create coarse STL -> render 4 PNG views -> bbox from STL bounds.
    Avoids OCP TopoDS helper imports and BRepBndLib binding differences.
    """
    _append_progress(run_dir, "Preflight: hashing…")
    sha = _sha256_file(step_path)

    _append_progress(run_dir, "Preflight: generating coarse STL…")
    pre_stl = run_dir / "preflight_preview.stl"
    defl_used, stl_mb = _write_preflight_stl_adaptive(step_path, pre_stl)

    preview_views: Dict[str, str] = {}
    bbox_mm: Optional[Dict[str, Any]] = None

    try:
        _append_progress(run_dir, "Preflight: rendering 4-view PNGs…")
        preview_views = _ensure_4view_pngs_from_stl(pre_stl, run_dir)

        import pyvista as pv
        m = pv.read(str(pre_stl))
        xmin, xmax, ymin, ymax, zmin, zmax = m.bounds
        bbox_mm = {
            "min": [float(xmin), float(ymin), float(zmin)],
            "max": [float(xmax), float(ymax), float(zmax)],
            "size": [float(xmax - xmin), float(ymax - ymin), float(zmax - zmin)],
        }
    except Exception as e:
        _append_progress(run_dir, f"Preflight: PNG render failed: {e}")

    # simple counts without traversing deep topology (fast-ish)
    counts = {"solids": None, "faces": None}
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE

        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status == IFSelect_RetDone:
            reader.TransferRoots()
            shape = reader.OneShape()

            solid_count = 0
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            while exp.More():
                solid_count += 1
                exp.Next()

            face_count = 0
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            while exp.More():
                face_count += 1
                exp.Next()

            counts = {"solids": int(solid_count), "faces": int(face_count)}
    except Exception:
        pass

    pack = {
        "schema": "preflight_pack_v1",
        "run_id": run_dir.name,
        "source_file": {"name": upload_name, "bytes": int(upload_bytes), "sha256": sha},
        "bbox_mm": bbox_mm,
        "counts": counts,
        "preview_views": preview_views,
        "preflight_mesh": {
            "stl_url": f"/runs/{run_dir.name}/{pre_stl.name}" if pre_stl.exists() else None,
            "stl_mb": float(stl_mb),
            "deflection": float(defl_used),
        },
        "created_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    _write_json(run_dir / PREFLIGHT_PACK_REL, pack)
    _append_progress(run_dir, "Preflight: done.")
    return pack


# ----------------------------
# Top-level extractors
# ----------------------------
def _top_level_from_import_pack(pack: Dict[str, Any]) -> List[TopLevelItem]:
    occs: Dict[str, Any] = pack.get("occurrences", {})
    roots: List[str] = pack.get("roots", []) or []

    top_ids: List[str] = []
    for r in roots:
        o = occs.get(r)
        if not o:
            continue
        kids = o.get("children") or []
        if kids:
            top_ids.extend([k for k in kids if k in occs])
        else:
            top_ids.append(r)

    seen = set()
    dedup: List[str] = []
    for x in top_ids:
        if x not in seen:
            seen.add(x)
            dedup.append(x)

    if len(dedup) > MAX_TOPLEVEL:
        dedup = dedup[:MAX_TOPLEVEL]

    defs: Dict[str, Any] = pack.get("definitions", {})
    items: List[TopLevelItem] = []
    for occ_id in dedup:
        o = occs.get(occ_id, {})
        def_sig = o.get("ref_def_sig")
        name = o.get("display_name") or ""
        if (not name) and def_sig in defs:
            name = defs[def_sig].get("display_name") or def_sig
        if not name:
            name = occ_id
        items.append(TopLevelItem(id=occ_id, name=name, ref=def_sig))
    return items


def _top_level_from_viewer_meta(meta: Dict[str, Any]) -> List[TopLevelItem]:
    by_inst: Dict[str, Any] = meta.get("by_instance_id", {})
    top: List[TopLevelItem] = []
    for inst_id, rec in by_inst.items():
        if rec.get("parent_uid") == "A-ROOT":
            name = rec.get("name") or inst_id
            ref = rec.get("part_uid")
            top.append(TopLevelItem(id=inst_id, name=name, ref=ref))

    top.sort(key=lambda x: x.name)
    if len(top) > MAX_TOPLEVEL:
        top = top[:MAX_TOPLEVEL]
    return top


def _top_level_from_xcaf_instances(x: Dict[str, Any]) -> List[TopLevelItem]:
    defs: Dict[str, Any] = x.get("definitions", {}) or {}
    occs: Dict[str, Any] = x.get("occurrences", {}) or {}

    root_def = x.get("root_def")
    if not isinstance(root_def, str) or not root_def:
        return []

    top_occ_ids: List[str] = []
    for occ_id, o in occs.items():
        if isinstance(o, dict) and o.get("parent_def") == root_def:
            top_occ_ids.append(occ_id)

    def _occ_sort_key(occ_id: str) -> Tuple[str, str]:
        o = occs.get(occ_id, {}) or {}
        nm = (o.get("name") or o.get("occ_id") or occ_id)
        return (nm, occ_id)

    top_occ_ids.sort(key=_occ_sort_key)
    if len(top_occ_ids) > MAX_TOPLEVEL:
        top_occ_ids = top_occ_ids[:MAX_TOPLEVEL]

    items: List[TopLevelItem] = []
    for occ_id in top_occ_ids:
        o = occs.get(occ_id, {}) or {}
        def_id = o.get("ref_def")

        name = None
        if def_id and def_id in defs:
            name = defs[def_id].get("name")
        if not name:
            name = o.get("name") or occ_id

        ref = def_id
        if def_id and def_id in defs:
            ref = defs[def_id].get("def_sig") or def_id

        items.append(TopLevelItem(id=occ_id, name=name, ref=ref))

    return items


def _load_top_level(run_dir: Path) -> Tuple[List[TopLevelItem], str]:
    pack_path = run_dir / IMPORT_PACK_REL
    meta_path = run_dir / VIEWER_META_REL
    xcaf_path = run_dir / XCAF_INSTANCES_REL

    if pack_path.exists():
        return _top_level_from_import_pack(_read_json(pack_path)), "import_pack"
    if meta_path.exists():
        return _top_level_from_viewer_meta(_read_json(meta_path)), "viewer_meta"
    if xcaf_path.exists():
        return _top_level_from_xcaf_instances(_read_json(xcaf_path)), "xcaf_instances"
    return [], "none"

def _compute_leaf_occurrences_from_occs(occs: Any) -> Optional[int]:
    """
    Leaf = occurrence that is not a parent of any other occurrence.
    Fast: single pass, no recursion.
    """
    if not isinstance(occs, dict):
        return None

    PARENT_KEYS = ("parent_occ", "parent_occ_id", "parent_id", "parent_uid", "parent")

    has_children = set()
    for _, o in occs.items():
        if not isinstance(o, dict):
            continue
        parent = None
        for pk in PARENT_KEYS:
            pv = o.get(pk)
            if isinstance(pv, str) and pv:
                parent = pv
                break
        if parent:
            has_children.add(parent)

    leaf = 0
    for occ_id in occs.keys():
        if occ_id not in has_children:
            leaf += 1
    return int(leaf)


# ----------------------------
# Orientation: store a deterministic transform matrix
# (so downstream exports can apply consistently)
# ----------------------------
def _deg_to_rad(d: float) -> float:
    return float(d) * 3.141592653589793 / 180.0


def _mat4_identity() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    # bounded, explicit loops (Power-of-10 style)
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for r in range(4):
        ar0, ar1, ar2, ar3 = a[r]
        for c in range(4):
            out[r][c] = (
                ar0 * b[0][c] +
                ar1 * b[1][c] +
                ar2 * b[2][c] +
                ar3 * b[3][c]
            )
    return out


def _rot_x(deg: int) -> List[List[float]]:
    import math
    th = _deg_to_rad(float(deg))
    c = float(math.cos(th))
    s = float(math.sin(th))
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c,   -s,  0.0],
        [0.0, s,    c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rot_y(deg: int) -> List[List[float]]:
    import math
    th = _deg_to_rad(float(deg))
    c = float(math.cos(th))
    s = float(math.sin(th))
    return [
        [c,   0.0, s,   0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0, c,   0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rot_z(deg: int) -> List[List[float]]:
    import math
    th = _deg_to_rad(float(deg))
    c = float(math.cos(th))
    s = float(math.sin(th))
    return [
        [c,   -s,  0.0, 0.0],
        [s,    c,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _orientation_matrix(orient: Dict[str, Any]) -> List[List[float]]:
    """
    Your preflight renders:
      plan  = view_xy (look along +Z)
      front = view_yz (look along +X)
      side  = view_xz (look along +Y)

    If user says "front is true PLAN", they mean: make +X become +Z (top-down).
    If user says "side is true PLAN", they mean: make +Y become +Z.
    Then apply rotation_deg about +Z in the corrected frame.
    """
    plan_source = str(orient.get("plan_source") or "plan")
    rot_deg = int(orient.get("rotation_deg") or 0)
    if rot_deg not in (0, 90, 180, 270):
        rot_deg = 0

    base = _mat4_identity()
    if plan_source == "front":
        # X -> Z
        base = _rot_y(-90)
    elif plan_source == "side":
        # Y -> Z
        base = _rot_x(+90)

    rz = _rot_z(rot_deg)
    return _mat4_mul(rz, base)


# ----------------------------
# Step 2 runner + lightweight tree builder
# ----------------------------
def _run_step2(step_path: Path, out_dir: Path) -> None:
    if not STEP2_CMD:
        raise RuntimeError("STEP2_CMD is not set.")
    cmd = STEP2_CMD.format(step=str(step_path), out=str(out_dir))
    _append_progress(out_dir, "Step2: starting fast XCAF read…")
    _append_progress(out_dir, f"CMD: {cmd}")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        _append_progress(out_dir, line.rstrip())

    rc = proc.wait()
    if rc != 0:
        _append_progress(out_dir, f"Step2: FAILED (exit {rc})")
        raise RuntimeError(f"Step2 failed (exit {rc}). See progress.log for details.")

    _append_progress(out_dir, "Step2: XCAF instances written.")


def _as_mat4(val: Any) -> Optional[List[List[float]]]:
    # Accept [[...],[...],[...],[...]] or flat 16
    if isinstance(val, list) and len(val) == 4 and all(isinstance(r, list) and len(r) == 4 for r in val):
        try:
            return [[float(x) for x in row] for row in val]
        except Exception:
            return None
    if isinstance(val, list) and len(val) == 16:
        try:
            f = [float(x) for x in val]
            return [f[0:4], f[4:8], f[8:12], f[12:16]]
        except Exception:
            return None
    return None


def _mat4_to_flat16(m: List[List[float]]) -> List[float]:
    return [m[r][c] for r in range(4) for c in range(4)]


def _apply_orientation_metadata(x: Dict[str, Any], orient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Don’t assume exact transform field names in xcaf_instances.json.
    For Step 2 deliverable: store the matrix + orientation clearly.
    If we find known matrix fields, we *also* pre-multiply them.
    """
    m = _orientation_matrix(orient)

    meta = x.get("ui", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["orientation"] = {
        "plan_source": orient.get("plan_source", "plan"),
        "rotation_deg": int(orient.get("rotation_deg", 0)),
    }
    meta["orientation_matrix_4x4"] = m
    meta["orientation_matrix_flat16"] = _mat4_to_flat16(m)
    x["ui"] = meta

    # Best-effort transform patching (safe no-ops if keys don’t exist)
    occs = x.get("occurrences", {})
    if isinstance(occs, dict):
        CAND_KEYS = (
            "trsf_4x4", "trsf_local_4x4", "trsf_global_4x4",
            "matrix_4x4", "matrix_local_4x4", "matrix_global_4x4",
            "transform_4x4", "transform_local_4x4", "transform_global_4x4",
        )
        for _, o in occs.items():
            if not isinstance(o, dict):
                continue
            for k in CAND_KEYS:
                mm = _as_mat4(o.get(k))
                if mm is None:
                    continue
                o[k] = _mat4_mul(m, mm)

    return x


def _build_occurrence_tree(x: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    defs = x.get("definitions", {}) or {}
    occs = x.get("occurrences", {}) or {}

    # Parent key discovery (don’t guess too hard; keep deterministic)
    PARENT_KEYS = ("parent_occ", "parent_occ_id", "parent_id", "parent_uid", "parent")

    # Build parent->children adjacency
    children_by_parent: Dict[str, List[str]] = {}
    nodes: Dict[str, Any] = {}

    for occ_id, o in occs.items():
        if not isinstance(o, dict):
            continue

        parent = None
        for pk in PARENT_KEYS:
            pv = o.get(pk)
            if isinstance(pv, str) and pv:
                parent = pv
                break

        # Record child link
        if parent:
            children_by_parent.setdefault(parent, []).append(occ_id)

        # Node basics
        ref_def = o.get("ref_def")
        display_name = o.get("name") or o.get("display_name") or occ_id

        ref_def_sig = None
        if isinstance(ref_def, str) and ref_def and isinstance(defs, dict) and ref_def in defs:
            drec = defs.get(ref_def) or {}
            if isinstance(drec, dict):
                ref_def_sig = drec.get("def_sig") or None

        nodes[occ_id] = {
            "occ_id": occ_id,
            "display_name": display_name,
            "ref_def": ref_def,
            "ref_def_sig": ref_def_sig,
            "children": [],  # filled below
        }

    # Fill children deterministically
    for parent, kids in children_by_parent.items():
        kids_sorted = sorted(kids)
        if parent in nodes:
            nodes[parent]["children"] = kids_sorted

    # Roots: prefer explicit root markers if present; else “not a child of anyone”
    roots: List[str] = []
    root_def = x.get("root_def")
    if isinstance(root_def, str) and root_def:
        # Try: occurrences whose parent_def == root_def
        for occ_id, o in occs.items():
            if isinstance(o, dict) and o.get("parent_def") == root_def:
                roots.append(occ_id)

    if not roots:
        all_ids = set([k for k in occs.keys() if isinstance(k, str)])
        non_roots = set()
        for _, kids in children_by_parent.items():
            for kid in kids:
                non_roots.add(kid)
        roots = sorted(list(all_ids - non_roots))

    tree = {
        "schema": "occurrence_tree_v1",
        "run_id": run_id,
        "roots": roots,
        "nodes": nodes,
        "counts": {
            "occurrences": int(len(nodes)),
            "roots": int(len(roots)),
            "definitions": int(len(defs)) if isinstance(defs, dict) else None,
        },
    }
    return tree


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _find_first(d: Any, keys: List[str]) -> Any:
    """
    Shallow+1 search: checks d[key] and d['synopsis'][key] and d['analysis'][key]
    without getting clever. Deterministic, bounded.
    """
    if not isinstance(d, dict):
        return None

    for k in keys:
        if k in d:
            return d.get(k)

    for parent in ("synopsis", "analysis", "meta"):
        sub = d.get(parent)
        if isinstance(sub, dict):
            for k in keys:
                if k in sub:
                    return sub.get(k)

    return None


def _extract_counts(x: Dict[str, Any]) -> Dict[str, Optional[int]]:
    defs = x.get("definitions")
    occs = x.get("occurrences")

    leaf = _safe_int(_find_first(x, ["leaf_occurrences", "leaf_occ_ids_count", "leaf_count"]))
    if leaf is None:
        leaf = _compute_leaf_occurrences_from_occs(occs)

    out = {
        "definitions": _safe_int(len(defs)) if isinstance(defs, dict) else None,
        "occurrences": _safe_int(len(occs)) if isinstance(occs, dict) else None,
        "leaf_occurrences": leaf,
        "free_shapes": _safe_int(_find_first(x, ["free_shapes", "freeShapes"])),
    }
    return out


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (UI_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/state/{run_id}")
def get_state(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    preflight: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {}
    orient = _read_orientation(run_dir)

    pf = run_dir / PREFLIGHT_PACK_REL
    if pf.exists():
        try:
            preflight = _read_json(pf)
        except Exception:
            preflight = {}

    ap = run_dir / ANALYSIS_PACK_REL
    if ap.exists():
        try:
            analysis = _read_json(ap)
        except Exception:
            analysis = {}

    return {"run_id": run_id, "preflight": preflight, "analysis": analysis, "orientation": orient}


@app.get("/api/orientation/{run_id}")
def get_orientation(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "orientation": _read_orientation(run_dir)}


@app.post("/api/orientation/{run_id}")
def set_orientation(
    run_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    plan_source = str(payload.get("plan_source") or "plan")
    rotation_deg = int(payload.get("rotation_deg") or 0)

    orient = _write_orientation(run_dir, plan_source, rotation_deg)
    _append_progress(run_dir, f"Orientation set: plan_source={orient['plan_source']} rotation={orient['rotation_deg']}")

    return {"run_id": run_id, "orientation": orient}


@app.post("/api/preview/{run_id}")
async def preview_step(run_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    step_path = run_dir / "input.step"
    nbytes = 0
    upload_name = file.filename or "input.step"

    try:
        with step_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                nbytes += len(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    _append_progress(run_dir, f"Upload saved: {nbytes} bytes")

    try:
        preflight = _preflight_pack(step_path, run_dir, upload_name, nbytes)
    except Exception as e:
        _append_progress(run_dir, f"Preflight failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preflight failed: {e}")

    orient = _read_orientation(run_dir)
    return {"run_id": run_id, "preflight": preflight, "orientation": orient}


@app.post("/api/analyze/{run_id}")
def analyze_run(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    step_path = run_dir / "input.step"
    if not step_path.exists():
        raise HTTPException(status_code=400, detail="No uploaded STEP for this run_id. Use Upload + Preview first.")

    # Run pipeline
    try:
        _run_pipeline(step_path, run_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Generate assembly STLs (optional 3D)
    full_rel = None
    prev_rel = None
    try:
        full_rel, prev_rel = _ensure_assembly_stls(step_path, run_dir)
    except Exception as e:
        _append_progress(run_dir, f"WARNING: assembly STL not generated: {e}")

    # Load top-level
    try:
        items, source = _load_top_level(run_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top-level parse failed: {e}")

    # Decide which STL to serve
    stl_url = None
    stl_kind = "none"
    stl_size_mb = None

    full_path = run_dir / ASSEMBLY_STL_REL
    prev_path = run_dir / ASSEMBLY_PREVIEW_STL_REL

    if full_rel and full_path.exists():
        size = _mb(full_path)
        if size <= MAX_ASSEMBLY_STL_MB:
            stl_url = f"/runs/{run_id}/{ASSEMBLY_STL_REL}"
            stl_kind = "full"
            stl_size_mb = size

    if stl_url is None and prev_rel and prev_path.exists():
        size = _mb(prev_path)
        stl_url = f"/runs/{run_id}/{ASSEMBLY_PREVIEW_STL_REL}"
        stl_kind = "preview"
        stl_size_mb = size

    analysis_pack = {
        "run_id": run_id,
        "meta_source": source,
        "assembly_stl_url": stl_url,
        "assembly_stl_kind": stl_kind,
        "assembly_stl_mb": stl_size_mb,
        "top_level": [{"id": i.id, "name": i.name, "ref": i.ref} for i in items],
    }
    _write_json(run_dir / ANALYSIS_PACK_REL, analysis_pack)
    return analysis_pack


# Back-compat: one-call import (preview + analyze)
@app.post("/api/import/{run_id}")
async def import_step(run_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    # preview upload
    prev = await preview_step(run_id, file)
    # analyze
    analysis = analyze_run(run_id)
    # merge
    out = dict(analysis)
    out["preflight"] = prev.get("preflight", {})
    out["orientation"] = prev.get("orientation", {})
    return out


@app.post("/api/step2/{run_id}")
def step2(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    step_path = run_dir / "input.step"
    if not step_path.exists():
        raise HTTPException(status_code=400, detail="No uploaded STEP for this run_id. Use Upload + Preview first.")

    orient = _read_orientation(run_dir)

    # Run fast XCAF read (no full assembly STL)
    try:
        _run_step2(step_path, run_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    xcaf_path = run_dir / XCAF_INSTANCES_REL
    if not xcaf_path.exists():
        raise HTTPException(status_code=500, detail="Step2 did not produce xcaf_instances.json")

    # Load xcaf_instances.json, attach orientation matrix (and best-effort patch matrices)
    x: Dict[str, Any] = {}
    try:
        x = _read_json(xcaf_path)
        x = _apply_orientation_metadata(x, orient)
        _write_json(xcaf_path, x)  # overwrite with ui.orientation metadata included
    except Exception as e:
        _append_progress(run_dir, f"Step2: WARNING could not attach orientation metadata: {e}")
        # keep going with whatever is on disk

    # Build lightweight occurrence tree
    try:
        x2 = _read_json(xcaf_path)
        tree = _build_occurrence_tree(x2, run_id)
        _write_json(run_dir / OCC_TREE_REL, tree)
    except Exception as e:
        _append_progress(run_dir, f"Step2: WARNING could not build occurrence tree: {e}")

    # Provide top-level (pills) using existing helper, but sourced from xcaf_instances
    top_level: List[TopLevelItem] = []
    try:
        top_level = _top_level_from_xcaf_instances(_read_json(xcaf_path))
    except Exception as e:
        _append_progress(run_dir, f"Step2: WARNING top-level parse failed: {e}")

    # UI summary for header:
    # - bbox from preview preflight (already computed cheaply + reliably)
    # - counts from xcaf_instances.json
    preflight_info: Dict[str, Any] = {"bbox_mm": None, "counts": {}}

    # bbox: use preflight_pack.json if present (stored as {"min":[...], "max":[...], "size":[...]})
    try:
        pf_path = run_dir / PREFLIGHT_PACK_REL
        if pf_path.exists():
            pf = _read_json(pf_path)
            bb = pf.get("bbox_mm")
            if isinstance(bb, dict) and isinstance(bb.get("size"), list) and len(bb["size"]) == 3:
                preflight_info["bbox_mm"] = {
                    "x": float(bb["size"][0]),
                    "y": float(bb["size"][1]),
                    "z": float(bb["size"][2]),
                }
    except Exception as e:
        _append_progress(run_dir, f"Step2: WARNING could not load bbox from preflight_pack: {e}")

    # counts: from xcaf_instances.json
    try:
        x3 = _read_json(xcaf_path)
        preflight_info["counts"] = _extract_counts(x3)
    except Exception as e:
        _append_progress(run_dir, f"Step2: WARNING could not extract counts from xcaf_instances: {e}")
        preflight_info["counts"] = {}

    ui_meta = x.get("ui") if isinstance(x, dict) else None
    orientation_matrix_flat16 = ui_meta.get("orientation_matrix_flat16") if isinstance(ui_meta, dict) else None

    pack = {
        "run_id": run_id,
        "meta_source": "step2_xcaf",
        "assembly_stl_url": None,
        "assembly_stl_kind": "none",
        "assembly_stl_mb": None,
        "top_level": [{"id": i.id, "name": i.name, "ref": i.ref} for i in top_level],
        "occurrence_tree_url": f"/runs/{run_id}/{OCC_TREE_REL}" if (run_dir / OCC_TREE_REL).exists() else None,
        "xcaf_instances_url": f"/runs/{run_id}/{XCAF_INSTANCES_REL}" if xcaf_path.exists() else None,
        "preflight": preflight_info,
        "orientation": orient,
        "orientation_matrix_flat16": orientation_matrix_flat16,
        "created_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    try:
        c = preflight_info.get("counts", {}) if isinstance(preflight_info, dict) else {}
        _append_progress(run_dir, f"Step2: done defs={c.get('definitions')} occs={c.get('occurrences')}")
    except Exception:
        pass

    _write_json(run_dir / STEP2_PACK_REL, pack)
    _write_json(run_dir / ANALYSIS_PACK_REL, pack)  # keep /api/state restore path unchanged

    return pack
