from __future__ import annotations

import json
import os
import subprocess
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# New: occurrence tree output (Step 1)
OCC_TREE_REL = os.environ.get("OCC_TREE_REL", "occ_tree.json")
XCAF_INSTANCES_REL = os.environ.get("XCAF_INSTANCES_REL", "xcaf_instances.json")
ASSETS_MANIFEST_REL = os.environ.get("ASSETS_MANIFEST_REL", "assets_manifest.json")


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
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_json_required(p: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {p} ({e})") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid JSON (expected object): {p}")
    return obj


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    # atomic-ish write
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


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
    try:
        return int(getattr(pd, "n_cells"))
    except Exception:
        return 0


def _safe_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _pick_first(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


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

    with _heartbeat(run_dir, "Preflight: meshing+writing STL", every_sec=5.0):
        _write_coarse_stl_from_step(step_path, stl_coarse, PREFLIGHT_STL_DEFLECTION, run_dir)

    coarse_mb = _mb(stl_coarse)
    _append_progress(run_dir, f"Preflight: coarse STL size={coarse_mb:.1f} MB")

    with _heartbeat(run_dir, "Preflight: preparing render mesh", every_sec=5.0):
        stl_to_render, render_mb, render_note = _make_render_stl(stl_coarse, stl_render, run_dir)

    _append_progress(run_dir, f"Preflight: {render_note} (render STL {render_mb:.1f} MB)")

    with _heartbeat(run_dir, "Preflight: rendering 6 views", every_sec=5.0):
        views = _render_6faces_from_stl(stl_to_render, run_dir)

    if coarse_mb > PREFLIGHT_KEEP_STL_MAX_MB:
        _append_progress(
            run_dir,
            f"Preflight: deleting coarse STL ({coarse_mb:.1f} MB > keep cap {PREFLIGHT_KEEP_STL_MAX_MB:.1f} MB)",
        )
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
# Occurrence tree (Step 1)
# ----------------------------
@dataclass(frozen=True)
class _ManifestHit:
    match_status: str
    part_id: str
    stl_path: Optional[str]
    ref_def: Optional[str]
    def_sig_used: Optional[str]


def _norm_occurrences(xcaf: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    occs = xcaf.get("occurrences")
    if occs is None:
        raise RuntimeError("xcaf_instances missing 'occurrences'")

    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(occs, dict):
        for occ_id, rec in occs.items():
            if isinstance(rec, dict):
                oid = _safe_str(occ_id).strip()
                if oid:
                    out[oid] = rec
        return out

    if isinstance(occs, list):
        for rec in occs:
            if not isinstance(rec, dict):
                continue
            oid = _pick_first(rec.get("occ_id"), rec.get("id"))
            if not oid:
                continue
            out[oid] = rec
        return out

    raise RuntimeError("xcaf_instances.occurrences is neither dict nor list")


def _occ_ref_def_id(occ: Dict[str, Any]) -> Optional[str]:
    return _pick_first(occ.get("ref_def"), occ.get("def_id"), occ.get("definition"), occ.get("ref_def_id"))


def _def_sig(def_rec: Dict[str, Any]) -> Optional[str]:
    return _pick_first(def_rec.get("def_sig_free"), def_rec.get("def_sig"))


def _occ_display_name(occ: Dict[str, Any], defs: Dict[str, Any], ref_def_id: Optional[str], occ_id: str) -> str:
    name = _pick_first(occ.get("display_name"), occ.get("name"), occ.get("label"))
    if name:
        return name
    if ref_def_id and isinstance(defs.get(ref_def_id), dict):
        dn = _pick_first(defs[ref_def_id].get("name"))
        if dn:
            return dn
    return occ_id


def _index_manifest(man: Dict[str, Any]) -> Tuple[Dict[str, List[_ManifestHit]], Dict[str, List[_ManifestHit]]]:
    items = man.get("items")
    by_sig: Dict[str, List[_ManifestHit]] = {}
    by_def: Dict[str, List[_ManifestHit]] = {}

    if not isinstance(items, list):
        return by_sig, by_def

    for it in items:
        if not isinstance(it, dict):
            continue
        hit = _ManifestHit(
            match_status=_safe_str(it.get("match_status")).strip() or "unknown",
            part_id=_safe_str(it.get("part_id")).strip(),
            stl_path=it.get("stl_path") if isinstance(it.get("stl_path"), str) else None,
            ref_def=_pick_first(it.get("ref_def")),
            def_sig_used=_pick_first(it.get("def_sig_used")),
        )
        if hit.def_sig_used:
            by_sig.setdefault(hit.def_sig_used, []).append(hit)
        if hit.ref_def:
            by_def.setdefault(hit.ref_def, []).append(hit)

    return by_sig, by_def


def _pick_manifest_stl(
    ref_def_sig: Optional[str],
    ref_def_id: Optional[str],
    by_sig: Dict[str, List[_ManifestHit]],
    by_def: Dict[str, List[_ManifestHit]],
) -> Optional[str]:
    cands: List[_ManifestHit] = []
    if ref_def_sig and ref_def_sig in by_sig:
        cands.extend(by_sig[ref_def_sig])
    if (not cands) and ref_def_id and ref_def_id in by_def:
        cands.extend(by_def[ref_def_id])

    if not cands:
        return None

    def _rank(status: str) -> int:
        return 0 if status == "matched" else 10

    cands_sorted = sorted(cands, key=lambda h: (_rank(h.match_status), h.part_id, _safe_str(h.stl_path)))
    for h in cands_sorted:
        if h.stl_path:
            return h.stl_path
    return None


def _build_occ_tree(run_dir: Path) -> None:
    """
    Build occ_tree.json as soon as xcaf_instances.json exists.
    Uses xcaf_instances for hierarchy; optionally uses assets_manifest to resolve stl_url.
    """
    xcaf_path = run_dir / XCAF_INSTANCES_REL
    if not xcaf_path.exists():
        raise RuntimeError(f"Missing {XCAF_INSTANCES_REL}")

    xcaf = _read_json_required(xcaf_path)

    defs = xcaf.get("definitions")
    if not isinstance(defs, dict):
        raise RuntimeError("xcaf_instances missing 'definitions' object")

    occs = _norm_occurrences(xcaf)

    man = _read_json_if_exists(run_dir / ASSETS_MANIFEST_REL) or {}
    by_sig, by_def = _index_manifest(man)

    # Build children map
    children_map: Dict[str, List[str]] = {oid: [] for oid in occs.keys()}
    parent_map: Dict[str, Optional[str]] = {oid: None for oid in occs.keys()}

    has_explicit_children = any(isinstance(occs[oid].get("children"), list) for oid in occs.keys())

    if has_explicit_children:
        for oid, rec in occs.items():
            kids = rec.get("children")
            if not isinstance(kids, list):
                continue
            for k in kids:
                kid = _safe_str(k).strip()
                if kid and kid in occs:
                    children_map[oid].append(kid)
                    parent_map[kid] = oid
    else:
        for oid, rec in occs.items():
            parent = _pick_first(rec.get("parent_occ_id"), rec.get("parent"))
            if parent and parent in occs:
                parent_map[oid] = parent
                children_map[parent].append(oid)

    # Roots (prefer xcaf.roots if present)
    roots_raw = xcaf.get("roots")
    roots: List[str] = []
    if isinstance(roots_raw, list):
        for r in roots_raw:
            rid = _safe_str(r).strip()
            if rid and rid in occs:
                roots.append(rid)
    if not roots:
        roots = [oid for oid, p in parent_map.items() if not p]

    nodes: Dict[str, Dict[str, Any]] = {}

    for oid, occ in occs.items():
        ref_def_id = _occ_ref_def_id(occ)
        def_rec = defs.get(ref_def_id) if ref_def_id and isinstance(defs.get(ref_def_id), dict) else None
        ref_def_sig = _def_sig(def_rec) if isinstance(def_rec, dict) else None

        stl_path = _pick_manifest_stl(ref_def_sig, ref_def_id, by_sig, by_def)
        stl_url = None
        if stl_path and stl_path.startswith("stl/"):
            stl_url = f"/runs/{run_dir.name}/{stl_path}"

        display = _occ_display_name(occ, defs, ref_def_id, oid)

        kids = children_map.get(oid, [])
        # deterministic order
        kids_sorted = sorted(
            list(dict.fromkeys(kids)),
            key=lambda k: (_occ_display_name(occs[k], defs, _occ_ref_def_id(occs[k]), k).lower(), k),
        )

        node: Dict[str, Any] = {
            "display_name": display,
            "children": kids_sorted,
            "ref_def_sig": ref_def_sig,
            "stl_url": stl_url,
        }

        if ref_def_id:
            node["ref_def_id"] = ref_def_id

        if isinstance(def_rec, dict):
            if "qty_total" in def_rec:
                node["qty_total"] = def_rec.get("qty_total")
            if "shape_kind" in def_rec:
                node["shape_kind"] = def_rec.get("shape_kind")
            if "solid_count" in def_rec:
                node["solid_count"] = def_rec.get("solid_count")
            if "def_sig_algo" in def_rec:
                node["def_sig_algo"] = def_rec.get("def_sig_algo")

        nodes[oid] = node

    roots_sorted = sorted(roots, key=lambda r: (nodes[r]["display_name"].lower(), r))

    tree = {
        "schema": "occ_tree.v1",
        "run_id": run_dir.name,
        "created_utc": _now_utc_iso(),
        "roots": roots_sorted,
        "nodes": nodes,
    }

    out_path = run_dir / OCC_TREE_REL
    _write_json(out_path, tree)


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

                # New: build occurrence tree immediately after xcaf_instances exists
                _append_progress(run_dir, "Tree: building occ_tree.json…")
                try:
                    _build_occ_tree(run_dir)
                    _append_progress(run_dir, f"Tree: done. ({OCC_TREE_REL})")
                except Exception as e:
                    # Don't fail the whole run if tree failed; record and continue.
                    _append_progress(run_dir, f"Tree: WARNING: {e}")

                _set_status(run_dir, "ready")
            except Exception as e:
                _append_progress(run_dir, f"ERROR: {e}")
                _set_status(run_dir, "error", {"error": str(e)})
            finally:
                _unlock(run_dir)

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
