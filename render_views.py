#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

# Headless rendering config (same idea as render_thumbnails.py)
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv


def _start_headless():
    # If no display is available, start a virtual framebuffer (best-effort).
    if not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception:
            pass


def _safe_view(pl: pv.Plotter, name: str) -> None:
    # Use built-in view helpers (robust).
    if name == "plan":
        # Looking down Z
        pl.view_xy()
    elif name == "front":
        # Looking down Y
        pl.view_xz()
    elif name == "side":
        # Looking down X
        pl.view_yz()
    elif name == "iso":
        pl.view_isometric()
    else:
        pl.view_isometric()


def render_4_views(stl_path: Path, out_dir: Path, size: int = 960, decimate: float = 0.0) -> Dict[str, str]:
    """
    Renders plan/front/side/iso PNGs from a single STL.
    Optional decimate in [0..1): proportion of reduction (e.g. 0.9 keeps ~10%).
    """
    _start_headless()

    mesh = pv.read(str(stl_path))

    # Optional: decimate huge meshes before rendering.
    # decimate=0.9 -> reduce triangles ~90%
    if decimate and 0.0 < decimate < 1.0:
        try:
            mesh = mesh.decimate_pro(decimate)
        except Exception:
            # If decimation fails, render as-is.
            pass

    out_dir.mkdir(parents=True, exist_ok=True)

    views = ["plan", "front", "side", "iso"]
    out: Dict[str, str] = {}

    for v in views:
        png_path = out_dir / f"view_{v}.png"

        pl = pv.Plotter(off_screen=True, window_size=(size, size))
        pl.set_background("white")

        # Shaded mesh (this gives you form cues)
        pl.add_mesh(mesh, smooth_shading=True)

        _safe_view(pl, v)

        # Slight zoom for readability
        try:
            pl.camera.zoom(1.15)
        except Exception:
            pass

        pl.show(auto_close=False)
        pl.screenshot(str(png_path), return_img=False)
        pl.close()

        out[v] = str(png_path)

    return out


def main():
    import sys

    if len(sys.argv) < 3:
        print("usage: render_views.py <stl_path> <out_dir> [size] [decimate]")
        raise SystemExit(2)

    stl_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 960
    dec = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    if not stl_path.exists():
        raise FileNotFoundError(stl_path)

    render_4_views(stl_path, out_dir, size=size, decimate=dec)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
