from __future__ import annotations

import json
import os
import subprocess
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ----------------------------
# Config
# ----------------------------
RUNS_DIR = Path(os.environ.get("RUNS_DIR", "./ui_runs")).resolve()

DEFAULT_ANALYZE_CMD = 'python /repo/read_step_xcaf.py --with_massprops "{step}" "{out}"'
ANALYZE_CMD = os.environ.get("ANALYZE_CMD", DEFAULT_ANALYZE_CMD).strip()

STATUS_REL = os.environ.get("STATUS_REL", "status.json")
PREFLIGHT_PACK_REL = os.environ.get("PREFLIGHT_PACK_REL", "preflight_pack.json")

# Preflight rendering sizes
PREFLIGHT_IMG_SIZE = int(os.environ.get("PREFLIGHT_IMG_SIZE", "560"))

# Coarse STL meshing parameters (single pass)
PREFLIGHT_STL_DEFLECTION = float(os.environ.get("PREFLIGHT_STL_DEFLECTION", "600.0"))

# If the coarse STL is huge, we keep it only if <= this cap (else delete after making render STL)
PREFLIGHT_KEEP_STL_MAX_MB = float(os.environ.get("PREFLIGHT_KEEP_STL_MAX_MB", "20.0"))

# If the STL (coarse or render) exceeds this, we attempt to decimate for render
PREFLIGHT_DECIMATE_IF_MB_OVER = float(os.environ.get("PREFLIGHT_DECIMATE_IF_MB_OVER", "80.0"))

# Target poly count for render STL (decimate_pro)
PREFLIGHT_RENDER_MAX_CELLS = int(os.environ.get("PREFLIGHT_RENDER_MAX_CELLS", "250000"))

# Filenames
PREFLIGHT_STL_NAME = os.environ.get("PREFLIGHT_STL_NAME", "preflight_preview.stl")
PREFLIGHT_RENDER_STL_NAME = os.environ.get("PREFLIGHT_RENDER_STL_NAME", "preflight_render.stl")

SLEEP_SEC = float(os.environ.get("WORKER_SLEEP_SEC", "2.0"))


# ----------------------------
# Helpers
# ----------------------------

def _start_heartbeat(run_dir: Path, label: str, every_sec: float = 5.0) -> Tuple[threading.Event, threading.Thread]:
    """
    Background heartbeat that appends a line every `every_sec` seconds.
    Keeps UI alive during long OCC operations.
    """
    stop_evt = threading.Event()

    def _loop() -> None:
        i = 0
        while not stop_evt.wait(every_sec):
            i += 1
            _append_progress(run_dir, f"{label} … ({i * every_sec:.0f}s)")

    t = threading.Thread(target=_loop, name="preflight-heartbeat", daemon=True)
    t.start()
    return stop_evt, t


@contextmanager
def _heartbeat(run_dir: Path, label: str, every_sec: float = 5.0):
    stop_evt, t = _start_heartbeat(run_dir, label, every_sec=every_sec)
    try:
        yield
    finally:
        stop_evt.set()
        try:
            t.join(timeout=1.0)
        except Exception:
            pass


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _set_status(run_dir: Path, stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    s = _read_json_if_exists(run_dir / STATUS_REL) or {}
    s.update(
        {
            "schema": "run_status_v1",
            "run_id": run_dir.name,
            "stage": stage,
            "updated_utc": _now_utc_iso(),
        }
    )
    if extra:
        s.update(extra)
    _write_json(run_dir / STATUS_REL, s)


def _mb(p: Path) -> float:
    return float(p.stat().st_size) / (1024.0 * 1024.0)


def _try_lock(run_dir: Path) -> bool:
    lock = run_dir / ".worker_lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _unlock(run_dir: Path) -> None:
    lock = run_dir / ".worker_lock"
    try:
        lock.unlink(missing_ok=True)
    except Exception:
        pass


def _poly_cells_count(pd) -> int:
    # PyVista: for PolyData, triangles are cells
    try:
        return int(getattr(pd, "n_cells"))
    except Exception:
        return 0


# ----------------------------
# STEP -> coarse STL
# ----------------------------
def _write_coarse_stl_from_step(step_path: Path, stl_path: Path, deflection: float, run_dir: Path) -> None:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    _append_progress(run_dir, "Preflight: [stl] reading STEP…")
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed (status={status})")

    _append_progress(run_dir, "Preflight: [stl] transfer roots…")
    reader.TransferRoots()
    shape = reader.OneShape()

    _append_progress(run_dir, f"Preflight: [stl] meshing deflection={deflection:.1f}…")
    BRepMesh_IncrementalMesh(shape, float(deflection))

    _append_progress(run_dir, "Preflight: [stl] writing STL…")
    w = StlAPI_Writer()
    try:
        w.SetASCIIMode(False)
    except Exception:
        pass

    ok = w.Write(shape, str(stl_path))
    if not ok:
        raise RuntimeError("STL write failed")


# ----------------------------
# Decimate for render (optional)
# ----------------------------
def _make_render_stl(src_stl: Path, dst_stl: Path, run_dir: Path) -> Tuple[Path, float, str]:
    """
    Return (stl_to_render, mb, note).
    If src is small enough -> use src.
    If src is huge -> try decimate_pro into dst and use dst.
    """
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    src_mb = _mb(src_stl)

    if src_mb <= PREFLIGHT_DECIMATE_IF_MB_OVER:
        return src_stl, src_mb, "render: using coarse STL (no decimate)"

    _append_progress(run_dir, f"Preflight: coarse STL {src_mb:.1f} MB -> decimating for render…")
    m = pv.read(str(src_stl))
    n0 = _poly_cells_count(m)
    if n0 <= 0:
        return src_stl, src_mb, "render: decimate skipped (no cells)"

    if n0 <= PREFLIGHT_RENDER_MAX_CELLS:
        # already small in poly terms, just copy
        try:
            if dst_stl.exists():
                dst_stl.unlink()
        except Exception:
            pass
        dst_stl.write_bytes(src_stl.read_bytes())
        mb = _mb(dst_stl)
        return dst_stl, mb, f"render: copied (cells={n0})"

    reduction = 1.0 - (float(PREFLIGHT_RENDER_MAX_CELLS) / float(max(1, n0)))
    if reduction < 0.05:
        reduction = 0.05
    if reduction > 0.98:
        reduction = 0.98

    m2 = m.decimate_pro(reduction)
    n1 = _poly_cells_count(m2)
    if n1 <= 0:
        return src_stl, src_mb, "render: decimate failed (0 cells)"

    try:
        if dst_stl.exists():
            dst_stl.unlink()
    except Exception:
        pass
    m2.save(str(dst_stl))
    mb = _mb(dst_stl)
    return dst_stl, mb, f"render: decimate {n0} -> {n1} cells (reduction={reduction:.3f})"


# ----------------------------
# Render 6 faces via PyVista from STL
# ----------------------------
def _render_6faces_from_stl(stl_path: Path, run_dir: Path) -> Dict[str, str]:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from PIL import Image

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
        pl.camera.zoom(1.10)
        pl.show(screenshot=str(out_png), auto_close=True)

    p_top = run_dir / "face_top.png"
    p_bottom = run_dir / "face_bottom.png"
    p_front = run_dir / "face_front.png"
    p_back = run_dir / "face_back.png"
    p_left = run_dir / "face_left.png"
    p_right = run_dir / "face_right.png"

    _render(lambda pl: pl.view_xy(), p_top)
    _render(lambda pl: pl.view_yx(), p_bottom)
    _render(lambda pl: pl.view_yz(), p_front)
    _render(lambda pl: pl.view_zy(), p_back)
    _render(lambda pl: pl.view_zx(), p_left)
    _render(lambda pl: pl.view_xz(), p_right)

    # composite
    size = PREFLIGHT_IMG_SIZE
    comp = Image.new("RGB", (size * 3, size * 2), (255, 255, 255))

    def _paste(src: Path, col: int, row: int) -> None:
        im = Image.open(src).convert("RGB")
        comp.paste(im, (col * size, row * size))

    _paste(p_top, 0, 0)
    _paste(p_front, 1, 0)
    _paste(p_right, 2, 0)
    _paste(p_left, 0, 1)
    _paste(p_back, 1, 1)
    _paste(p_bottom, 2, 1)

    cube = run_dir / "cube_faces.png"
    comp.save(cube)

    return {
        "cube_faces": f"/runs/{run_dir.name}/{cube.name}",
        "top": f"/runs/{run_dir.name}/{p_top.name}",
        "bottom": f"/runs/{run_dir.name}/{p_bottom.name}",
        "front": f"/runs/{run_dir.name}/{p_front.name}",
        "back": f"/runs/{run_dir.name}/{p_back.name}",
        "left": f"/runs/{run_dir.name}/{p_left.name}",
        "right": f"/runs/{run_dir.name}/{p_right.name}",
    }


# ----------------------------
# Preflight
# ----------------------------
def _preflight(run_dir: Path) -> None:
    step_path = run_dir / "input.step"
    if not step_path.exists():
        raise RuntimeError("Missing input.step")

    step_mb = 0.0
    try:
        step_mb = _mb(step_path)
    except Exception:
        pass

    _append_progress(run_dir, f"Preflight: step size={step_mb:.1f} MB")
    _append_progress(run_dir, "Preflight: generating ONE coarse STL…")

    stl_coarse = run_dir / PREFLIGHT_STL_NAME
    stl_render = run_dir / PREFLIGHT_RENDER_STL_NAME

    try:
        stl_coarse.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        stl_render.unlink(missing_ok=True)
    except Exception:
        pass

    # Heartbeat during the heavyweight OCC mesh+write
    with _heartbeat(run_dir, "Preflight: meshing+writing STL", every_sec=5.0):
        _write_coarse_stl_from_step(step_path, stl_coarse, PREFLIGHT_STL_DEFLECTION, run_dir)

    coarse_mb = _mb(stl_coarse)
    _append_progress(run_dir, f"Preflight: coarse STL size={coarse_mb:.1f} MB")

    # Heartbeat during decimation (can be slow on huge meshes)
    with _heartbeat(run_dir, "Preflight: preparing render mesh", every_sec=5.0):
        stl_to_render, render_mb, render_note = _make_render_stl(stl_coarse, stl_render, run_dir)

    _append_progress(run_dir, f"Preflight: {render_note} (render STL {render_mb:.1f} MB)")

    # Heartbeat during 6-face rendering (can be slow if mesh is heavy)
    with _heartbeat(run_dir, "Preflight: rendering 6 views", every_sec=5.0):
        views = _render_6faces_from_stl(stl_to_render, run_dir)

    if coarse_mb > PREFLIGHT_KEEP_STL_MAX_MB:
        _append_progress(run_dir, f"Preflight: deleting coarse STL ({coarse_mb:.1f} MB > keep cap {PREFLIGHT_KEEP_STL_MAX_MB:.1f} MB)")
        try:
            stl_coarse.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        stl_render.unlink(missing_ok=True)
    except Exception:
        pass

    pack = {
        "schema": "preflight_pack_v1",
        "run_id": run_dir.name,
        "created_utc": _now_utc_iso(),
        "bbox_mm": None,
        "preview_views": views,
        "preflight_mesh": {
            "stl_url": None,
            "stl_mb": 0.0,
            "deflection": float(PREFLIGHT_STL_DEFLECTION),
            "note": "views from coarse STL via PyVista (HLR disabled)",
            "render_note": render_note,
        },
    }

    _write_json(run_dir / PREFLIGHT_PACK_REL, pack)
    _append_progress(run_dir, "Preflight: done.")


# ----------------------------
# Heavy step: run read_step_xcaf.py
# ----------------------------
def _run_pipeline(run_dir: Path) -> None:
    step_path = run_dir / "input.step"
    if not step_path.exists():
        raise RuntimeError("Missing input.step")
    if not ANALYZE_CMD:
        raise RuntimeError("ANALYZE_CMD is empty")

    cmd = ANALYZE_CMD.format(step=str(step_path), out=str(run_dir))
    _append_progress(run_dir, "Analysis: starting…")
    _append_progress(run_dir, f"CMD: {cmd}")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd="/",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_progress(run_dir, line.rstrip())

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Pipeline failed (exit {rc})")

    _append_progress(run_dir, "Analysis: done.")


# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[worker] watching: {RUNS_DIR}")

    while True:
        try:
            run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
        except Exception:
            time.sleep(SLEEP_SEC)
            continue

        for run_dir in run_dirs:
            status = _read_json_if_exists(run_dir / STATUS_REL) or {}
            stage = str(status.get("stage") or "")

            if stage != "uploaded":
                continue

            if not _try_lock(run_dir):
                continue

            try:
                _set_status(run_dir, "preflight")
                _preflight(run_dir)

                _set_status(run_dir, "running")
                _run_pipeline(run_dir)

                _set_status(run_dir, "ready")
            except Exception as e:
                _append_progress(run_dir, f"ERROR: {e}")
                _set_status(run_dir, "error", {"error": str(e)})
            finally:
                _unlock(run_dir)

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
