from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.responses import HTMLResponse, StreamingResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone

# ----------------------------
# Config
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent
UI_DIR = ROOT_DIR / "ui"
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(ROOT_DIR / "ui_runs"))).resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

EXPLODE_PLAN_REL = os.environ.get("EXPLODE_PLAN_REL", "explode_plan.json")

PREFLIGHT_PACK_REL = os.environ.get("PREFLIGHT_PACK_REL", "preflight_pack.json")
ORIENTATION_REL = os.environ.get("ORIENTATION_REL", "orientation.json")
STATUS_REL = os.environ.get("STATUS_REL", "status.json")

# (Worker outputs later; Step 1 doesn't require them)
XCAF_INSTANCES_REL = os.environ.get("XCAF_INSTANCES_REL", "xcaf_instances.json")
OCC_TREE_REL = os.environ.get("OCC_TREE_REL", "occ_tree.json")
ANALYSIS_PACK_REL = os.environ.get("ANALYSIS_PACK_REL", "analysis_pack.json")

app = FastAPI(title="STEP UI Starter (Step 1)")

# Serve UI static
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
# Serve run outputs (png/json/etc)
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


# ----------------------------
# Helpers (deterministic / bounded)
# ----------------------------

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/sw.js")
def sw():
    return Response(status_code=204)

def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _run_dir(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_progress(run_dir: Path, msg: str) -> None:
    p = run_dir / "progress.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
        f.flush()


def _read_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _set_status(run_dir: Path, stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    s: Dict[str, Any] = {
        "schema": "run_status_v1",
        "run_id": run_dir.name,
        "stage": stage,  # created|uploaded|preflight|error
        "updated_utc": _now_utc_iso(),
    }
    if extra:
        s.update(extra)
    _write_json(run_dir / STATUS_REL, s)


def _default_orientation(run_id: str) -> Dict[str, Any]:
    plan_source = "top"
    rotation_deg = 0
    return {
        "schema": "orientation_v1",
        "run_id": run_id,
        "plan_source": plan_source,
        "rotation_deg": rotation_deg,
        "transform": _orientation_matrix4(plan_source, rotation_deg),
        "updated_utc": _now_utc_iso(),
    }


def _load_or_init_orientation(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / ORIENTATION_REL
    o = _read_json_if_exists(p)

    if isinstance(o, dict) and o.get("schema") == "orientation_v1":
        # Backfill / normalize fields
        ps = str(o.get("plan_source") or "top").strip().lower()
        rd = int(o.get("rotation_deg") or 0)

        if ps not in ("top", "bottom", "front", "back", "left", "right"):
            ps = "top"
        if rd not in (0, 90, 180, 270):
            rd = 0

        # Ensure transform matches current ps/rd
        o["plan_source"] = ps
        o["rotation_deg"] = rd
        o["transform"] = _orientation_matrix4(ps, rd)
        if not o.get("updated_utc"):
            o["updated_utc"] = _now_utc_iso()

        _write_json(p, o)
        return o

    o = _default_orientation(run_dir.name)
    _write_json(p, o)
    return o


def _load_explode_plan(run_dir: Path, run_id: str) -> Dict[str, Any]:
    p = run_dir / EXPLODE_PLAN_REL
    j = _read_json_if_exists(p)
    if isinstance(j, dict):
        # ensure required fields exist
        j.setdefault("schema", "explode_plan_v1")
        j.setdefault("run_id", run_id)
        j.setdefault("items", {})
        return j

    return {
        "schema": "explode_plan_v1",
        "run_id": run_id,
        "created_utc": _now_utc_iso(),
        "modified_utc": _now_utc_iso(),
        "items": {},  # def_sig -> record
    }

def _prefer_json(run_dir: Path, preferred_name: str, fallback_name: str) -> Path:
    """
    Prefer `preferred_name` if it exists, else `fallback_name`.
    Always returns a path under run_dir.
    """
    p = run_dir / preferred_name
    if p.is_file():
        return p
    return run_dir / fallback_name


def _json_fileresponse(p: Path) -> FileResponse:
    # Avoid stale UI when files are rewritten
    return FileResponse(
        p,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )



async def _save_upload_to(path: Path, up: UploadFile, *, chunk_size: int = 1024 * 1024) -> Tuple[int, str]:
    """
    Save UploadFile to disk in bounded chunks.
    Returns: (nbytes, original_filename)
    """
    nbytes = 0
    name = up.filename or path.name
    try:
        with path.open("wb") as f:
            while True:
                chunk = await up.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                nbytes += len(chunk)
    finally:
        try:
            await up.close()
        except Exception:
            pass
    return int(nbytes), str(name)

import math
from typing import List, Dict, Any

# ----------------------------
# Orientation math (discrete, matches step1_worker.py view_* calls)
# ----------------------------

_ALLOWED_PLAN_SOURCES = ("top", "bottom", "front", "back", "left", "right")
_ALLOWED_ROT_DEG = (0, 90, 180, 270)


def _mat3_mul(a, b):
    # 3x3 row-major multiply (bounded loops)
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            out[r][c] = (
                a[r][0] * b[0][c] +
                a[r][1] * b[1][c] +
                a[r][2] * b[2][c]
            )
    return out


def _mat3_T(m):
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


def _rotz_cw(deg: int):
    """
    Rotate in-plane CLOCKWISE about +Z (screen-like), discrete 0/90/180/270.
    """
    if deg == 0:
        return [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]
    if deg == 90:
        # x' = y, y' = -x
        return [[0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]]
    if deg == 180:
        return [[-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0]]
    # 270
    return [[0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]]


def _basis_for_plan_source(src: str):
    """
    Returns (x_can_in_orig, y_can_in_orig, z_can_in_orig) as unit vectors in ORIGINAL coords.

    This is defined to match the thumbnails produced by step1_worker.py:

      top    -> pl.view_xy():  camera +Z, screen axes: X right, Y up
      bottom -> pl.view_yx():  camera -Z, screen axes: Y right, X up
      front  -> pl.view_yz():  camera +X, screen axes: Y right, Z up
      back   -> pl.view_zy():  camera -X, screen axes: Z right, Y up
      right  -> pl.view_xz():  camera +Y, screen axes: X right, Z up
      left   -> pl.view_zx():  camera -Y, screen axes: Z right, X up

    We store canonical axes so that when transformed, the chosen view becomes a "TOP" view:
      canonical +Z is "toward the camera" in the chosen view,
      canonical +X is screen-right,
      canonical +Y is screen-up.
    """
    s = (src or "top").strip().lower()
    if s == "top":
        return ((1.0, 0.0, 0.0),  # +X right
                (0.0, 1.0, 0.0),  # +Y up
                (0.0, 0.0, 1.0))  # +Z toward camera
    if s == "bottom":
        return ((0.0, 1.0, 0.0),  # +Y right
                (1.0, 0.0, 0.0),  # +X up
                (0.0, 0.0,-1.0))  # -Z toward camera
    if s == "front":
        return ((0.0, 1.0, 0.0),  # +Y right
                (0.0, 0.0, 1.0),  # +Z up
                (1.0, 0.0, 0.0))  # +X toward camera
    if s == "back":
        return ((0.0, 0.0, 1.0),  # +Z right
                (0.0, 1.0, 0.0),  # +Y up
                (-1.0,0.0, 0.0))  # -X toward camera
    if s == "right":
        return ((1.0, 0.0, 0.0),  # +X right
                (0.0, 0.0, 1.0),  # +Z up
                (0.0, 1.0, 0.0))  # +Y toward camera
    if s == "left":
        return ((0.0, 0.0, 1.0),  # +Z right
                (1.0, 0.0, 0.0),  # +X up
                (0.0,-1.0, 0.0))  # -Y toward camera
    raise ValueError("plan_source must be one of: top, bottom, front, back, left, right")


def _orientation_matrix4(plan_source: str, rotation_deg: int) -> Dict[str, Any]:
    """
    Build transform dict:
      - matrix_row_major: original -> canonical (4x4)
      - matrix_inv_row_major: canonical -> original (4x4)
    """
    ps = (plan_source or "top").strip().lower()
    if ps not in _ALLOWED_PLAN_SOURCES:
        raise ValueError("plan_source must be one of: top, bottom, front, back, left, right")

    deg = int(rotation_deg or 0)
    if deg not in _ALLOWED_ROT_DEG:
        raise ValueError("rotation_deg must be one of: 0, 90, 180, 270")

    xax, yax, zax = _basis_for_plan_source(ps)

    # B columns are canonical axes expressed in original coords
    B = [
        [xax[0], yax[0], zax[0]],
        [xax[1], yax[1], zax[1]],
        [xax[2], yax[2], zax[2]],
    ]

    # original -> canonical is B^T (orthonormal)
    R = _mat3_T(B)

    # Apply in-plane clockwise rotation about canonical +Z
    Rz = _rotz_cw(deg)
    Rf = _mat3_mul(Rz, R)

    M = [
        [Rf[0][0], Rf[0][1], Rf[0][2], 0.0],
        [Rf[1][0], Rf[1][1], Rf[1][2], 0.0],
        [Rf[2][0], Rf[2][1], Rf[2][2], 0.0],
        [0.0,      0.0,      0.0,      1.0],
    ]

    # Inverse of rotation-only is transpose
    Rt = _mat3_T(Rf)
    Minv = [
        [Rt[0][0], Rt[0][1], Rt[0][2], 0.0],
        [Rt[1][0], Rt[1][1], Rt[1][2], 0.0],
        [Rt[2][0], Rt[2][1], Rt[2][2], 0.0],
        [0.0,      0.0,      0.0,      1.0],
    ]

    return {
        "kind": "rigid_4x4",
        "space": "global",
        "matrix_row_major": M,
        "matrix_inv_row_major": Minv,
        "definition": "Apply to global points: p' = M * [x,y,z,1]. rotation_deg is clockwise about +Z in canonical frame.",
    }


# ----------------------------
# Minimal STEP -> mesh -> thumbnails (OCP + pyvista)
# ----------------------------
def _mod(name: str):
    return __import__(f"OCP.{name}", fromlist=[name])


def _call_maybe(obj, base: str, *args):
    fn = getattr(obj, base, None)
    if callable(fn):
        return fn(*args)
    fn_s = getattr(obj, base + "_s", None)
    if callable(fn_s):
        return fn_s(*args)
    return None


def _read_step_one_shape(step_path: Path):
    STEPControl = _mod("STEPControl")
    IFSelect = _mod("IFSelect")
    reader = STEPControl.STEPControl_Reader()
    stat = reader.ReadFile(str(step_path))
    ok = (stat == IFSelect.IFSelect_RetDone)
    if not ok:
        raise RuntimeError(f"STEP read failed (status={stat})")
    reader.TransferRoots()
    shp = reader.OneShape()
    if shp is None or shp.IsNull():
        raise RuntimeError("STEP read returned null shape")
    return shp


def _bbox_mm(shape) -> Dict[str, Any]:
    Bnd = _mod("Bnd")
    BRepBndLib = _mod("BRepBndLib")

    box = Bnd.Bnd_Box()
    # Try module-level and class-wrapper variants
    added = False
    for host in (BRepBndLib, getattr(BRepBndLib, "BRepBndLib", None)):
        if host is None:
            continue
        for nm in ("Add", "Add_s"):
            fn = getattr(host, nm, None)
            if not callable(fn):
                continue
            try:
                fn(shape, box, True)
                added = True
                break
            except TypeError:
                pass
            except Exception:
                pass
            try:
                fn(shape, box)
                added = True
                break
            except Exception:
                pass
        if added:
            break
    if not added:
        raise RuntimeError("Could not compute bbox (BRepBndLib.Add unavailable)")

    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    sx = float(xmax - xmin)
    sy = float(ymax - ymin)
    sz = float(zmax - zmin)
    return {
        "min": [float(xmin), float(ymin), float(zmin)],
        "max": [float(xmax), float(ymax), float(zmax)],
        "size": [sx, sy, sz],
    }


def _counts(shape) -> Dict[str, int]:
    TopExp = _mod("TopExp")
    TopAbs = _mod("TopAbs")
    exp_face = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_FACE)
    faces = 0
    while exp_face.More():
        faces += 1
        exp_face.Next()
    exp_sol = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_SOLID)
    solids = 0
    while exp_sol.More():
        solids += 1
        exp_sol.Next()
    return {"faces": int(faces), "solids": int(solids)}


def _triangulate_to_pyvista(shape, *, deflection: float):
    import numpy as np  # local import
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError(f"pyvista not available: {e}")

    # headless safety
    try:
        pv.start_xvfb()
    except Exception:
        pass

    BRepMesh = _mod("BRepMesh")
    _call_maybe(BRepMesh, "BRepMesh_IncrementalMesh", shape, float(deflection))

    TopExp = _mod("TopExp")
    TopAbs = _mod("TopAbs")
    BRep = _mod("BRep")

    # BRep_Tool host differs by build
    BRep_Tool = getattr(BRep, "BRep_Tool", None)
    if BRep_Tool is None:
        BRep_Tool = getattr(BRep, "BRep", None)

    verts = []
    faces = []

    exp = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_FACE)
    while exp.More():
        face = exp.Current()

        loc = None
        try:
            loc = face.Location()
        except Exception:
            loc = None

        tri = None
        if BRep_Tool is not None:
            try:
                tri = _call_maybe(BRep_Tool, "Triangulation", face, loc)
            except Exception:
                tri = None

        if tri is None:
            exp.Next()
            continue

        try:
            nb_nodes = int(tri.NbNodes())
            nb_tris = int(tri.NbTriangles())
        except Exception:
            exp.Next()
            continue

        if nb_nodes <= 0 or nb_tris <= 0:
            exp.Next()
            continue

        base = len(verts)
        for i in range(1, nb_nodes + 1):
            p = tri.Node(i)
            verts.append([float(p.X()), float(p.Y()), float(p.Z())])

        for i in range(1, nb_tris + 1):
            t = tri.Triangle(i)
            n1, n2, n3 = t.Get()
            faces.append([3, base + int(n1) - 1, base + int(n2) - 1, base + int(n3) - 1])

        exp.Next()

    if not verts or not faces:
        raise RuntimeError("No triangulation produced (empty verts/faces)")

    faces_np = np.hstack(faces).astype(np.int32)
    mesh = pv.PolyData(np.asarray(verts, dtype=float), faces_np)
    return mesh


def _render_views(mesh, out_dir: Path, *, px: int = 320) -> Dict[str, str]:
    import numpy as np
    import pyvista as pv

    b = mesh.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
    cx = 0.5 * (b[0] + b[1])
    cy = 0.5 * (b[2] + b[3])
    cz = 0.5 * (b[4] + b[5])
    dx = max(1e-6, float(b[1] - b[0]))
    dy = max(1e-6, float(b[3] - b[2]))
    dz = max(1e-6, float(b[5] - b[4]))

    # A simple, deterministic margin
    margin = 1.08

    def _shot(name: str, pos, up, parallel_scale: float):
        pl = pv.Plotter(off_screen=True, window_size=(px, px))
        pl.set_background("white")
        pl.add_mesh(mesh, show_edges=False)
        pl.enable_parallel_projection()
        pl.camera.focal_point = (cx, cy, cz)
        pl.camera.position = pos
        pl.camera.up = up
        pl.camera.parallel_scale = float(parallel_scale)

        # generous clipping
        dist = np.linalg.norm(np.asarray(pos, float) - np.asarray([cx, cy, cz], float))
        pl.camera.clipping_range = (max(0.1, dist * 0.01), dist * 50.0)

        out = out_dir / f"{name}.png"
        pl.show(screenshot=str(out))
        return out.name

    d = max(dx, dy, dz) * 3.0

    # "plan" = look down +Z
    plan_scale = 0.5 * max(dx, dy) * margin
    front_scale = 0.5 * max(dx, dz) * margin
    side_scale = 0.5 * max(dy, dz) * margin
    iso_scale = 0.5 * max(dx, dy, dz) * margin

    views = {
        "plan": _shot("plan", (cx, cy, cz + d), (0, 1, 0), plan_scale),
        "front": _shot("front", (cx, cy + d, cz), (0, 0, 1), front_scale),
        "side": _shot("side", (cx + d, cy, cz), (0, 0, 1), side_scale),
        "iso": _shot("iso", (cx + d, cy + d, cz + d), (0, 0, 1), iso_scale),
    }
    return views


def _preflight_pack(step_path: Path, run_dir: Path, upload_name: str, nbytes: int) -> Dict[str, Any]:
    """
    Generates:
      - run_dir/plan.png, front.png, side.png, iso.png
      - run_dir/preflight_pack.json
    """
    _append_progress(run_dir, "Preflight: reading STEP…")
    shp = _read_step_one_shape(step_path)

    bb = _bbox_mm(shp)
    cnt = _counts(shp)

    # choose deflection deterministically from size (bounded)
    sx, sy, sz = bb["size"]
    diag = max(1e-6, (sx * sx + sy * sy + sz * sz) ** 0.5)
    defl = diag / 600.0
    if defl < 0.15:
        defl = 0.15
    if defl > 3.0:
        defl = 3.0

    views: Dict[str, Optional[str]] = {"plan": None, "front": None, "side": None, "iso": None}

    try:
        _append_progress(run_dir, f"Preflight: meshing (deflection={defl:.3f})…")
        mesh = _triangulate_to_pyvista(shp, deflection=float(defl))
        _append_progress(run_dir, "Preflight: rendering views…")
        v = _render_views(mesh, run_dir)
        for k in views.keys():
            views[k] = v.get(k)
    except Exception as e:
        # Still write pack (UI can show bbox and warn)
        _append_progress(run_dir, f"Preflight: thumbnail render skipped/failed: {e}")

    pack: Dict[str, Any] = {
        "schema": "preflight_pack_v1",
        "run_id": run_dir.name,
        "created_utc": _now_utc_iso(),
        "upload": {"filename": upload_name, "bytes": int(nbytes)},
        "bbox_mm": bb,
        "counts": cnt,
        "preview_views": views,
    }

    _write_json(run_dir / PREFLIGHT_PACK_REL, pack)
    _set_status(run_dir, "preflight")
    _append_progress(run_dir, "Preflight: done.")
    return pack


# ----------------------------
# Progress stream
# ----------------------------
@app.get("/api/progress/{run_id}")
async def progress(run_id: str):
    run_dir = _run_dir(run_id)
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
# Routes (Step 1)
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (UI_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/create_run")
def create_run() -> Dict[str, Any]:
    run_id = _safe_run_id()
    run_dir = _run_dir(run_id)
    _append_progress(run_dir, "Run created.")
    _set_status(run_dir, "created")
    return {"run_id": run_id}


# Legacy upload endpoint (unused by Step 1 UI; kept)
@app.post("/api/upload/{run_id}")
async def upload_step(
    run_id: str,
    file: UploadFile = File(...),
    project_number: str = Form(""),
    client: str = Form(""),
) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)
    step_path = run_dir / "input.step"
    nbytes, upload_name = await _save_upload_to(step_path, file)

    meta = {
        "schema": "run_meta_v1",
        "run_id": run_id,
        "created_utc": _now_utc_iso(),
        "upload": {"filename": upload_name, "bytes": int(nbytes)},
        "project": {"project_number": (project_number or "").strip(), "client": (client or "").strip()},
    }
    _write_json(run_dir / "run_meta.json", meta)

    _append_progress(run_dir, f"Upload saved: {nbytes} bytes")
    _set_status(run_dir, "uploaded")
    return {"run_id": run_id, "bytes": nbytes}


@app.get("/api/status/{run_id}")
def run_status(run_id: str) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)
    s = _read_json_if_exists(run_dir / STATUS_REL) or {
        "schema": "run_status_v1",
        "run_id": run_id,
        "stage": "unknown",
        "updated_utc": _now_utc_iso(),
    }

    # Presence flags only
    s["has"] = {
        "preflight_pack": (run_dir / PREFLIGHT_PACK_REL).exists(),
        "orientation": (run_dir / ORIENTATION_REL).exists(),
        "xcaf_instances": (run_dir / XCAF_INSTANCES_REL).exists(),
        "occurrence_tree": (run_dir / OCC_TREE_REL).exists(),
        "analysis_pack": (run_dir / ANALYSIS_PACK_REL).exists(),
    }
    return s


@app.get("/api/state/{run_id}")
def state(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")

    preflight = _read_json_if_exists(run_dir / PREFLIGHT_PACK_REL)
    orientation = _load_or_init_orientation(run_dir)

    return {
        "schema": "ui_state_v1",
        "run_id": run_id,
        "preflight": preflight,
        "orientation": orientation,
    }


@app.post("/api/orientation/{run_id}")
def save_orientation(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)

    plan_source = str(payload.get("plan_source") or "top").strip().lower()
    rotation_deg = int(payload.get("rotation_deg") or 0)

    if plan_source not in ("top", "bottom", "front", "back", "left", "right"):
        raise HTTPException(
            status_code=400,
            detail="plan_source must be one of: top, bottom, front, back, left, right",
        )
    if rotation_deg not in (0, 90, 180, 270):
        raise HTTPException(status_code=400, detail="rotation_deg must be 0, 90, 180, 270")

    orient = {
        "schema": "orientation_v1",
        "run_id": run_id,
        "plan_source": plan_source,
        "rotation_deg": rotation_deg,
        "transform": _orientation_matrix4(plan_source, rotation_deg),
        "updated_utc": _now_utc_iso(),
    }

    _write_json(run_dir / ORIENTATION_REL, orient)
    _append_progress(run_dir, f"Orientation saved: plan_source={plan_source} rotation={rotation_deg}")

    return orient


@app.post("/api/preview/{run_id}")
async def preview(run_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)

    step_path = run_dir / "input.step"
    nbytes, upload_name = await _save_upload_to(step_path, file)
    _append_progress(run_dir, f"Upload saved: {nbytes} bytes")

    # 🔑 this is what the worker is waiting for
    _set_status(run_dir, "uploaded", {"upload_name": upload_name, "upload_bytes": int(nbytes)})

    # ensure orientation exists for UI restore
    _load_or_init_orientation(run_dir)

    return {"run_id": run_id, "queued": True}


@app.get("/api/tree/{run_id}")
def get_tree(run_id: str):
    p = RUNS_DIR / run_id / OCC_TREE_REL
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{OCC_TREE_REL} not found")
    return FileResponse(p, media_type="application/json")

@app.get("/api/tree_grouped/{run_id}")
def get_tree_grouped(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")
    # Prefer patched tree if present
    p = _prefer_json(run_dir, "occ_tree_grouped_exploded.json", "occ_tree_grouped.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{p.name} not found")
    return _json_fileresponse(p)


@app.get("/api/bom/{run_id}")
def get_bom(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")
    # Prefer patched BOM if present
    p = _prefer_json(run_dir, "bom_global_exploded.json", "bom_global.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{p.name} not found")
    return _json_fileresponse(p)


@app.get("/api/explode_plan/{run_id}")
def get_explode_plan(run_id: str) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)
    # Return empty plan if it doesn't exist (don’t 404 the UI)
    plan = _load_explode_plan(run_dir, run_id)
    return plan

@app.post("/api/explode_plan/{run_id}")
def post_explode_plan(run_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)

    def_sig = (payload.get("def_sig") or "").strip()
    if not def_sig:
        raise HTTPException(status_code=400, detail="def_sig is required")

    action = (payload.get("action") or "").strip().lower()
    if action not in ("mark", "unmark"):
        raise HTTPException(status_code=400, detail="action must be 'mark' or 'unmark'")

    def_name = payload.get("def_name")
    solid_count = payload.get("solid_count")
    note = payload.get("note") or ""

    plan = _load_explode_plan(run_dir, run_id)
    items = plan.get("items")
    if not isinstance(items, dict):
        items = {}
        plan["items"] = items

    if action == "mark":
        items[def_sig] = {
            "def_sig": def_sig,
            "def_name": def_name,
            "solid_count": solid_count,
            "note": note,
            "marked_utc": _now_utc_iso(),
        }
    else:
        items.pop(def_sig, None)

    plan["modified_utc"] = _now_utc_iso()
    _write_json(run_dir / EXPLODE_PLAN_REL, plan)
    return plan