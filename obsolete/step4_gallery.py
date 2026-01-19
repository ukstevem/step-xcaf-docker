#!/usr/bin/env python3
"""
Step4 Gallery helper (all items).

Creates /out/step4_gallery:
  - index.html : list-first gallery, optional 3D previews (lazy, capped)
  - items.json : merged metadata from xcaf_instances.json + stl_manifest.json
  - items.js   : items as a JS global (no fetch/CORS hassles)
  - serve.ps1  : starts a local web server rooted at /out (so /stl works)
  - vendor/    : three.js ES modules copied from vendor sources if present

No STL copying. Links point to existing stl_manifest stl_path under /out.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class GalleryConfig:
    max_items: int = 50000
    prefer_with_stl: bool = True
    max_active_previews: int = 12  # prevent WebGL context explosion


_VENDOR_FILES = ("three.module.js", "OrbitControls.js", "STLLoader.js")


# ----------------------------
# Public API
# ----------------------------

def write_step4_gallery(
    out_dir: Path,
    xcaf_path: Path,
    manifest_path: Path,
    cfg: Optional[GalleryConfig] = None,
) -> Path:
    """
    Build merged items from Step 1 + Step 2/3 JSON and write /out/step4_gallery.
    Returns the gallery folder path.
    """
    if cfg is None:
        cfg = GalleryConfig()

    out_dir = Path(out_dir)
    gallery_dir = out_dir / "step4_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    xcaf = _read_json(xcaf_path)
    man = _read_json(manifest_path)

    items = build_items_from_json(xcaf, man)
    items_sorted = sorted(items, key=_sort_key(cfg.prefer_with_stl), reverse=True)[: int(cfg.max_items)]

    payload = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "count": len(items_sorted),
        "items": items_sorted,
    }

    (gallery_dir / "items.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (gallery_dir / "items.js").write_text(
        "window.__STEP4_GALLERY__ = " + json.dumps(payload) + ";\n",
        encoding="utf-8",
    )

    # IMPORTANT: serve from OUT_DIR (parent), not from inside step4_gallery,
    # so that ../stl/ resolves correctly and /stl is available.
    (gallery_dir / "serve.ps1").write_text(_build_serve_ps1(), encoding="utf-8")

    vendor_ok = _ensure_vendor_three(gallery_dir)

    (gallery_dir / "index.html").write_text(
        _build_html(vendor_ok=vendor_ok, max_active=cfg.max_active_previews),
        encoding="utf-8",
    )

    return gallery_dir


def build_items_from_json(xcaf: Dict[str, Any], man: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns one row per definition (has_shape=true) merged with best manifest entry (if any).
    """
    defs = xcaf.get("definitions") if isinstance(xcaf.get("definitions"), dict) else {}
    best_by_def = _pick_best_manifest_by_ref_def(man)

    items: List[Dict[str, Any]] = []

    for def_id, d in defs.items():
        if not bool(d.get("has_shape", False)):
            continue

        name = _as_str(d.get("name"))
        shape_kind = _as_str(d.get("shape_kind"))
        solid_count = _as_str(d.get("solid_count"))
        qty_total = _as_str(d.get("qty_total"))

        bbox_min = d.get("bbox", {}).get("min")
        bbox_max = d.get("bbox", {}).get("max")
        dx, dy, dz = _bbox_deltas(bbox_min, bbox_max)

        mp = d.get("massprops", {}) if isinstance(d.get("massprops"), dict) else {}
        mass_kg = _fmt_num(mp.get("mass_kg"))
        density = _fmt_num(mp.get("density_kg_m3"))
        vol = _fmt_num(mp.get("volume"))
        area = _fmt_num(mp.get("area"))

        m = best_by_def.get(def_id, {})
        match_status = _as_str(m.get("match_status"))
        stl_path = _normalize_rel_path(_as_str(m.get("stl_path")))
        part_id = _as_str(m.get("part_id"))
        def_sig_used = _as_str(m.get("def_sig_used"))
        def_sig_source = _as_str(m.get("def_sig_source"))

        # index.html is /out/step4_gallery/index.html
        # stl folder is /out/stl
        # so relative link is ../stl/<file>.stl
        stl_href = ("../" + stl_path) if stl_path else ""

        items.append(
            {
                "def_id": def_id,
                "name": name,
                "qty_total": qty_total,
                "shape_kind": shape_kind,
                "solid_count": solid_count,
                "bbox_dx": _fmt_num(dx),
                "bbox_dy": _fmt_num(dy),
                "bbox_dz": _fmt_num(dz),
                "mass_kg": mass_kg,
                "density_kg_m3": density,
                "volume": vol,
                "area": area,
                "match_status": match_status,
                "stl_path": stl_path,
                "stl_href": stl_href,
                "part_id": part_id,
                "def_sig_used": def_sig_used,
                "def_sig_source": def_sig_source,
            }
        )

    return items


# ----------------------------
# Manifest merge
# ----------------------------

def _pick_best_manifest_by_ref_def(man: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = man.get("items", [])
    if not isinstance(items, list):
        return {}

    best: Dict[str, Dict[str, Any]] = {}

    def score(it: Dict[str, Any]) -> Tuple[int, int]:
        stl = 1 if _as_str(it.get("stl_path")).strip() else 0
        matched = 1 if _as_str(it.get("match_status")).strip().lower() == "matched" else 0
        return (matched, stl)

    for it in items:
        if not isinstance(it, dict):
            continue
        ref_def = _as_str(it.get("ref_def")).strip()
        if not ref_def:
            continue

        cur = best.get(ref_def)
        if cur is None or score(it) > score(cur):
            best[ref_def] = it

    return best


# ----------------------------
# Vendor copy (self-contained)
# ----------------------------

def _patch_jsm_imports_to_local_three(src_text: str) -> str:
    """
    Patch three.js "examples/jsm" modules that import 'three' to import local three.module.js.
    """
    return re.sub(r"from\s+['\"]three['\"]", "from './three.module.js'", src_text)


def _candidate_vendor_dirs() -> List[Path]:
    """
    Search a few likely places for vendor/three files both inside Docker (/app)
    and on Windows (repo layout mirrored into container via bind mount).
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    cands: List[Path] = []

    # Typical repo layout: /app/vendor/three
    cands.append(here / "vendor" / "three")
    cands.append(cwd / "vendor" / "three")

    # Common alternatives we’ve seen in repos
    cands.append(here / "vendor")
    cands.append(cwd / "vendor")

    # If script lives in /app, allow /app/vendor/three explicitly
    cands.append(Path("/app/vendor/three"))
    cands.append(Path("/app/vendor"))

    # De-dup while keeping order
    seen = set()
    out: List[Path] = []
    for p in cands:
        rp = p.resolve() if p.exists() else p
        if str(rp) in seen:
            continue
        seen.add(str(rp))
        out.append(p)
    return out


def _find_vendor_file(src_dir: Path, filename: str) -> Optional[Path]:
    """
    Try exact match first. If not found, try a recursive search under src_dir.
    """
    p = src_dir / filename
    if p.is_file():
        return p

    # If src_dir is already 'vendor', try vendor/three/*
    p2 = src_dir / "three" / filename
    if p2.is_file():
        return p2

    # Last resort: recursive search (bounded to this directory)
    try:
        for found in src_dir.rglob(filename):
            if found.is_file():
                return found
    except Exception:
        return None

    return None


def _ensure_vendor_three(gallery_dir: Path) -> bool:
    """
    Copies required three.js files into /out/step4_gallery/vendor/.
    Returns True if all required files are present.
    """
    dst_dir = gallery_dir / "vendor"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Locate sources
    src_candidates = _candidate_vendor_dirs()

    copied: Dict[str, bool] = {fn: False for fn in _VENDOR_FILES}

    for fn in _VENDOR_FILES:
        src_path: Optional[Path] = None
        for cand in src_candidates:
            sp = _find_vendor_file(cand, fn)
            if sp is not None:
                src_path = sp
                break

        if src_path is None:
            continue

        dst_path = dst_dir / fn

        if fn in ("OrbitControls.js", "STLLoader.js"):
            txt = src_path.read_text(encoding="utf-8", errors="replace")
            txt2 = _patch_jsm_imports_to_local_three(txt)
            dst_path.write_text(txt2, encoding="utf-8")
        else:
            shutil.copy2(src_path, dst_path)

        copied[fn] = True

    # Verify final presence on disk (not just “copied=True”)
    ok = True
    for fn in _VENDOR_FILES:
        if not (dst_dir / fn).is_file():
            ok = False

    return ok


# ----------------------------
# Sorting
# ----------------------------

def _sort_key(prefer_with_stl: bool):
    def key(r: Dict[str, Any]) -> Tuple[int, float, str]:
        has_stl = 1 if _as_str(r.get("stl_path")).strip() else 0
        qty = _safe_float(r.get("qty_total"))
        name = _as_str(r.get("name")).lower()
        if prefer_with_stl:
            return (has_stl, qty, name)
        return (0, qty, name)
    return key


# ----------------------------
# Utils
# ----------------------------

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def _fmt_num(v: Any) -> str:
    try:
        if v is None:
            return ""
        return f"{float(v):.6f}".rstrip("0").rstrip(".")
    except Exception:
        return _as_str(v)


def _bbox_deltas(vmin: Any, vmax: Any) -> Tuple[float, float, float]:
    if not (isinstance(vmin, (list, tuple)) and isinstance(vmax, (list, tuple)) and len(vmin) == 3 and len(vmax) == 3):
        return (0.0, 0.0, 0.0)
    try:
        dx = float(vmax[0]) - float(vmin[0])
        dy = float(vmax[1]) - float(vmin[1])
        dz = float(vmax[2]) - float(vmin[2])
        return (dx, dy, dz)
    except Exception:
        return (0.0, 0.0, 0.0)


def _normalize_rel_path(p: str) -> str:
    """
    Normalize a path that is supposed to be under /out (e.g. "stl/abc.stl").
    - Converts backslashes to slashes
    - Strips leading slashes
    - Strips leading "out/" if someone wrote "out/stl/..."
    """
    s = (p or "").replace("\\", "/").strip()
    s = re.sub(r"^/+", "", s)
    if s.lower().startswith("out/"):
        s = s[4:]
    return s


def _build_serve_ps1() -> str:
    # Run server from OUT_DIR (parent of this step4_gallery folder)
    return r"""Param(
  [int]$Port = 8000
)
$gallery = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Split-Path -Parent $gallery
Push-Location $out
Write-Host "Serving /out at http://localhost:$Port/ (Ctrl+C to stop)"
Write-Host "Open: http://localhost:$Port/step4_gallery/index.html"
python -m http.server $Port
Pop-Location
"""


# ----------------------------
# HTML
# ----------------------------

def _build_html(vendor_ok: bool, max_active: int) -> str:
    vendor_flag = "true" if vendor_ok else "false"
    max_active_js = int(max_active)

    # IMPORTANT:
    # - No CDN.
    # - Only attempt module imports if vendor_ok AND served over HTTP.
    # - Lazy preview creation + cap to avoid WebGL context loss.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Step4 Gallery (All Items)</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 12px; }}
  .top {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  input {{ padding:6px; min-width:320px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; margin-top: 12px; }}
  .card {{ border:1px solid #ddd; border-radius: 8px; padding: 8px; }}
  .meta {{ font-size: 12px; color:#333; line-height:1.35; }}
  canvas {{ width: 100%; height: 220px; display:block; background:#f6f6f6; border-radius:6px; }}
  .name {{ font-weight: bold; margin:6px 0 2px 0; }}
  .muted {{ color:#777; }}
  .row {{ display:flex; justify-content:space-between; gap:8px; }}
  a {{ color:#0b57d0; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#eee; font-size:12px; }}
  .warn {{ color:#b45309; }}
</style>
</head>
<body>
<div class="top">
  <div><b>Step4 Gallery</b> — all parts (definitions with has_shape=true)</div>
  <div class="muted">Count: <span id="count">0</span></div>
  <input id="q" placeholder="Filter by name / def_id / match (e.g. 2515, matched)"/>
  <span id="status" class="muted"></span>
</div>

<div id="hint" class="meta"></div>
<div class="grid" id="grid"></div>

<script src="./items.js"></script>

<script>
(function() {{
  const payload = window.__STEP4_GALLERY__ || {{ items: [], count: 0 }};
  const items = payload.items || [];
  const grid = document.getElementById('grid');
  const count = document.getElementById('count');
  const q = document.getElementById('q');
  const status = document.getElementById('status');
  const hint = document.getElementById('hint');

  const vendorOk = {vendor_flag};
  count.textContent = String(items.length);

  if (!vendorOk) {{
    hint.innerHTML = '<span class="warn">Vendor three.js files not found. 3D previews disabled. List + links still work.</span>';
  }} else if (location.protocol === 'file:') {{
    hint.innerHTML = '<span class="warn"><b>3D previews need HTTP.</b> Run <code>serve.ps1</code> and open http://localhost:8000/step4_gallery/index.html</span>';
  }} else {{
    hint.textContent = 'Tip: scroll to load a few previews. Click a preview area to force-load that one.';
  }}

  function pill(txt) {{
    if (!txt) return '';
    return '<span class="pill">' + txt + '</span>';
  }}

  function cardHtml(item) {{
    const stlUrl = item.stl_href || '';
    const linkHtml = stlUrl
      ? '<div><a href="' + stlUrl + '" target="_blank">Open STL</a></div>'
      : '<div class="muted">No STL available</div>';

    return (
      '<div class="card">' +
        '<canvas data-stl="' + (stlUrl || '') + '"></canvas>' +
        '<div class="name">' + (item.name || '(no name)') + ' ' + pill(item.match_status) + '</div>' +
        '<div class="meta">' +
          '<div class="row"><span>def_id:</span><span><b>' + (item.def_id || '') + '</b></span></div>' +
          '<div class="row"><span>qty:</span><span>' + (item.qty_total || '') + '</span></div>' +
          '<div class="row"><span>bbox:</span><span>' +
            (item.bbox_dx || '') + ' × ' + (item.bbox_dy || '') + ' × ' + (item.bbox_dz || '') + ' mm</span></div>' +
          '<div class="row"><span>mass:</span><span>' + (item.mass_kg || '') + ' kg</span></div>' +
        '</div>' +
        '<div class="meta">' + linkHtml + '</div>' +
      '</div>'
    );
  }}

  function renderList(list) {{
    grid.innerHTML = list.map(cardHtml).join('');
    count.textContent = String(list.length);
  }}

  function applyFilter() {{
    const s = (q.value || '').trim().toLowerCase();
    if (!s) {{ renderList(items); return; }}
    const f = items.filter(it =>
      (it.name || '').toLowerCase().includes(s) ||
      (it.def_id || '').toLowerCase().includes(s) ||
      (it.match_status || '').toLowerCase().includes(s)
    );
    renderList(f);
  }}

  q.addEventListener('input', applyFilter);
  renderList(items);

  status.textContent = vendorOk
    ? 'List loaded. Previews are lazy-loaded.'
    : 'List loaded. Previews disabled (vendor missing).';
}})();
</script>

<script type="module">
  const vendorOk = {vendor_flag};
  const status = document.getElementById('status');
  const MAX_ACTIVE = {max_active_js};

  if (!vendorOk || location.protocol === 'file:') {{
    // no previews
  }} else {{
    try {{
      const THREE = await import('./vendor/three.module.js');
      const {{ OrbitControls }} = await import('./vendor/OrbitControls.js');
      const {{ STLLoader }} = await import('./vendor/STLLoader.js');

      let activeCount = 0;
      const previews = new Map();

      function makePreview(canvas, url) {{
        if (!url) return null;

        const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: false }});
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf6f6f6);

        const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
        camera.position.set(200, 200, 200);

        const controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const dl = new THREE.DirectionalLight(0xffffff, 0.7);
        dl.position.set(1, 1, 1);
        scene.add(dl);

        const loader = new STLLoader();
        let raf = 0;
        let ro = null;
        let disposed = false;

        function resize() {{
          const w = canvas.clientWidth || 1;
          const h = canvas.clientHeight || 1;
          renderer.setSize(w, h, false);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        }}

        function animate() {{
          if (disposed) return;
          controls.update();
          renderer.render(scene, camera);
          raf = requestAnimationFrame(animate);
        }}

        function dispose() {{
          disposed = true;
          if (raf) cancelAnimationFrame(raf);
          if (ro) ro.disconnect();

          scene.traverse(obj => {{
            if (obj.isMesh) {{
              if (obj.geometry) obj.geometry.dispose();
              if (obj.material) {{
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
              }}
            }}
          }});

          renderer.dispose();
          activeCount = Math.max(0, activeCount - 1);
        }}

        loader.load(url, (geom) => {{
          if (disposed) return;
          geom.computeBoundingBox();
          geom.center();

          const mat = new THREE.MeshStandardMaterial({{ color: 0x8c8c8c, metalness: 0.1, roughness: 0.8 }});
          scene.add(new THREE.Mesh(geom, mat));

          const size = new THREE.Vector3();
          geom.boundingBox.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z) || 1;

          camera.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.5);
          controls.target.set(0, 0, 0);
          controls.update();

          resize();
          ro = new ResizeObserver(resize);
          ro.observe(canvas);

          animate();
        }});

        return {{ dispose }};
      }}

      function startPreview(canvas) {{
        if (previews.has(canvas)) return;
        if (activeCount >= MAX_ACTIVE) return;

        const url = canvas.getAttribute('data-stl') || '';
        if (!url) return;

        activeCount += 1;
        const h = makePreview(canvas, url);
        if (!h) {{
          activeCount = Math.max(0, activeCount - 1);
          return;
        }}
        previews.set(canvas, h);
      }}

      function stopPreview(canvas) {{
        const h = previews.get(canvas);
        if (!h) return;
        previews.delete(canvas);
        h.dispose();
      }}

      // Click-to-force preview (if cap blocks auto start)
      document.querySelectorAll('canvas[data-stl]').forEach(c => {{
        c.addEventListener('click', () => startPreview(c));
        c.title = 'Click to preview (auto-previews are capped)';
      }});

      const io = new IntersectionObserver((entries) => {{
        for (const e of entries) {{
          const canvas = e.target;
          if (e.isIntersecting) startPreview(canvas);
          else stopPreview(canvas);
        }}
      }}, {{ threshold: 0.25 }});

      document.querySelectorAll('canvas[data-stl]').forEach(c => io.observe(c));

      status.textContent = '3D previews enabled (lazy, capped at ' + MAX_ACTIVE + ').';
    }} catch (e) {{
      console.warn(e);
      status.textContent = '3D previews failed to load vendor modules. List + links still available.';
    }}
  }}
</script>

</body>
</html>
"""


