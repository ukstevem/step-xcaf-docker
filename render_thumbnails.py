#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def main(out_dir: str = "/out", size: int = 512) -> None:
    out_dir_p = Path(out_dir)
    man_path = out_dir_p / "stl_manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}. Run STL export first.")

    # Headless rendering
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    import pyvista as pv

    # If no display is available, start a virtual framebuffer.
    # This is a common approach for headless/docker use. :contentReference[oaicite:3]{index=3}
    if not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception:
            # If this fails, you can still run via `xvfb-run -a ...` (see run command below)
            pass

    manifest: List[Dict[str, Any]] = json.loads(man_path.read_text(encoding="utf-8"))

    png_dir = out_dir_p / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    updated: List[Dict[str, Any]] = []
    ok = 0
    fail = 0

    for item in manifest:
        stl_rel = item.get("stl_path", "")
        stl_path = out_dir_p / stl_rel
        if not stl_path.exists():
            item["png_path"] = ""
            item["thumb_error"] = f"missing STL: {stl_rel}"
            updated.append(item)
            fail += 1
            continue

        # Name PNG after ref_def (stable) unless you want filenames from ref_name
        ref_def = item.get("ref_name","ref_def").replace(":", "_")
        png_path = png_dir / f"{ref_def}.png"

        try:
            mesh = pv.read(str(stl_path))

            pl = pv.Plotter(off_screen=True, window_size=(size, size))
            pl.set_background("white")

            # Render
            pl.add_mesh(mesh, smooth_shading=True)
            pl.view_isometric()
            pl.camera.zoom(1.2)

            # Ensure the scene is rendered before screenshot
            pl.show(auto_close=False)

            # Save screenshot (supports transparent_background if you want)
            # PyVista screenshot API: Plotter.screenshot(...) :contentReference[oaicite:4]{index=4}
            pl.screenshot(str(png_path), return_img=False)
            pl.close()

            item["png_path"] = str(png_path).replace(str(out_dir_p), "").lstrip("/")

            ok += 1
        except Exception as e:
            item["png_path"] = ""
            item["thumb_error"] = repr(e)
            fail += 1

        updated.append(item)

        if (ok + fail) % 50 == 0:
            print(f"thumbs: ok={ok} fail={fail}", flush=True)

    out_manifest = out_dir_p / "stl_manifest.json"
    out_manifest.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"Done thumbnails: ok={ok} fail={fail}", flush=True)
    print(f"Updated manifest: {out_manifest}", flush=True)
    print(f"PNG dir: {png_dir}", flush=True)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/out"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    main(out, size)
