#!/usr/bin/env python3
"""
Step 4 (Gallery) — build an all-items HTML gallery from existing Step 1–3 JSON outputs.

Inputs (defaults):
  /out/xcaf_instances.json
  /out/stl_manifest.json

Outputs:
  /out/step4_gallery/index.html
  /out/step4_gallery/items.json
  /out/step4_gallery/items.js
  /out/step4_gallery/serve.ps1

No STL copying. Uses stl_manifest stl_path as links.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from step4_gallery import GalleryConfig, write_step4_gallery


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xcaf", default=os.getenv("STEP4_XCAF", "/out/xcaf_instances.json"))
    ap.add_argument("--manifest", default=os.getenv("STEP4_MANIFEST", "/out/stl_manifest.json"))
    ap.add_argument("--out-dir", default=os.getenv("STEP4_OUT_DIR", "/out"))
    ap.add_argument("--max-items", type=int, default=_env_int("STEP4_GALLERY_MAX_ITEMS", 50000))
    ap.add_argument("--prefer-with-stl", action="store_true", default=True)

    ns, _ = ap.parse_known_args()

    out_dir = Path(ns.out_dir)
    xcaf_path = Path(ns.xcaf)
    manifest_path = Path(ns.manifest)

    cfg = GalleryConfig(max_items=int(ns.max_items), prefer_with_stl=bool(ns.prefer_with_stl))
    gdir = write_step4_gallery(out_dir, xcaf_path, manifest_path, cfg=cfg)

    print(f"[step4-gallery] Wrote: {gdir / 'index.html'}")
    print(f"[step4-gallery] Items : {(gdir / 'items.json')}")
    print(f"[step4-gallery] Serve : {(gdir / 'serve.ps1')}")
    print("Step 4 gallery complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
