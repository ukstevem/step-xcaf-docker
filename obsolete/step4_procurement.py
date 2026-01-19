#!/usr/bin/env python3
"""step4_procurement.py

Step 4 – Procurement outputs (CSV-first) + optional review gallery.

Reads:
  - xcaf_instances.json   (Step 1 truth file)
  - assets_manifest.json  (Step 2–3 asset index only)

Writes (under out_dir, default: /out/step4):
  - master_items.csv
  - plates.csv, sections.csv, hardware.csv, handrail.csv, treads.csv, grating.csv, review.csv
  - review_overrides.json   (small persisted user decisions)
  - gallery.html            (optional, plus --serve for a simple save flow)

Notes:
- Minimal deps: stdlib only.
- Deterministic ordering: category order then name then def_id.

Usage examples (inside your docker container):
  python step4_procurement.py \
    --xcaf /out/xcaf_instances.json \
    --assets /out/assets_manifest.json \
    --out /out/step4 \
    --write-html

  # Start a simple local review server (writes review_overrides.json on Save)
  python step4_procurement.py \
    --xcaf /out/xcaf_instances.json \
    --assets /out/assets_manifest.json \
    --out /out/step4 \
    --write-html --serve 8000
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Constants / category ordering
# ----------------------------

CATEGORY_ORDER: Tuple[str, ...] = (
    "PLATE",
    "SECTION",
    "HARDWARE",
    "HANDRAIL",
    "TREAD",
    "GRATING",
    "REVIEW",
)

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "def_id",
    "name",
    "qty_total",
    "shape_kind",
    "solid_count",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "def_sig_used",
    "chirality_sig_free",
    "stl_path",
    "category",
    "confidence",
    "reason",
    "needs_review",
    "explode_recommended",
)

# ----------------------------
# JSON loading (robust)
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return False


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return first found key in dict."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _ensure_defs_map(definitions: Any) -> Dict[str, Dict[str, Any]]:
    """Accept dict-of-defs or list-of-defs; return {def_id: def_dict}."""
    if isinstance(definitions, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in definitions.items():
            if isinstance(v, dict):
                def_id = str(v.get("def_id") or k)
                out[def_id] = v
        return out

    if isinstance(definitions, list):
        out = {}
        for v in definitions:
            if isinstance(v, dict):
                def_id = str(v.get("def_id") or "")
                if def_id:
                    out[def_id] = v
        return out

    return {}


def _extract_bbox_dims(defn: Dict[str, Any]) -> Tuple[float, float, float]:
    """Best-effort bbox dims (mm)."""
    # common direct keys
    x = _get(defn, "bbox_x", "bbox_dx", "dx")
    y = _get(defn, "bbox_y", "bbox_dy", "dy")
    z = _get(defn, "bbox_z", "bbox_dz", "dz")
    if x is not None and y is not None and z is not None:
        return (_as_float(x), _as_float(y), _as_float(z))

    # nested bbox object
    bbox = _get(defn, "bbox", default=None)
    if isinstance(bbox, dict):
        # dims may be stored as {x,y,z} or {dx,dy,dz}
        x2 = _get(bbox, "x", "dx", "bbox_x")
        y2 = _get(bbox, "y", "dy", "bbox_y")
        z2 = _get(bbox, "z", "dz", "bbox_z")
        if x2 is not None and y2 is not None and z2 is not None:
            return (_as_float(x2), _as_float(y2), _as_float(z2))

        # or {min:[...], max:[...]}
        mn = _get(bbox, "min", "mins", default=None)
        mx = _get(bbox, "max", "maxs", default=None)
        if isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple)) and len(mn) == 3 and len(mx) == 3:
            return (
                abs(_as_float(mx[0]) - _as_float(mn[0])),
                abs(_as_float(mx[1]) - _as_float(mn[1])),
                abs(_as_float(mx[2]) - _as_float(mn[2])),
            )

    # unknown
    return (0.0, 0.0, 0.0)


# ----------------------------
# Assets manifest join
# ----------------------------

@dataclass(frozen=True)
class AssetRec:
    ref_def: str
    part_id: str
    stl_path: str
    chirality_sig_free: str
    def_sig_used: str


def _assets_index(assets_json: Dict[str, Any]) -> Dict[str, AssetRec]:
    """Return {ref_def: AssetRec} choosing a deterministic representative per ref_def."""
    items = assets_json.get("items")
    if not isinstance(items, list):
        # tolerate older schema
        items = assets_json.get("assets")
    if not isinstance(items, list):
        return {}

    buckets: Dict[str, List[AssetRec]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        ref_def = str(_get(it, "ref_def", "def_id", "definition", default="")).strip()
        if not ref_def:
            continue
        rec = AssetRec(
            ref_def=ref_def,
            part_id=str(_get(it, "part_id", default="")) or "",
            stl_path=str(_get(it, "stl_path", default="")) or "",
            chirality_sig_free=str(_get(it, "chirality_sig_free", default="")) or "",
            def_sig_used=str(_get(it, "def_sig_used", default="")) or "",
        )
        buckets.setdefault(ref_def, []).append(rec)

    out: Dict[str, AssetRec] = {}
    for ref_def, recs in buckets.items():
        # deterministic pick
        recs_sorted = sorted(recs, key=lambda r: (r.part_id, r.stl_path, r.chirality_sig_free, r.def_sig_used))
        out[ref_def] = recs_sorted[0]
    return out


# ----------------------------
# Overrides (small persisted file)
# ----------------------------

def _stable_key_for(def_id: str, def_sig_used: str, chirality_sig_free: str, bbox_q: str) -> str:
    """A stable-ish key for overrides.

    Prefer signature-based; fall back to def_id (stable for the current run).
    """
    a = (def_sig_used or "").strip()
    c = (chirality_sig_free or "").strip()
    if a or c:
        return f"sig:{a}:{c}:{bbox_q}"
    return f"def:{def_id}"


def _read_overrides(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema": "step4_review_overrides_v1", "items": {}}
    try:
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("items"), dict):
            return data
    except Exception:
        pass
    return {"schema": "step4_review_overrides_v1", "items": {}}


def _write_overrides(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _apply_override(
    ovr_items: Dict[str, Any],
    key: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (confirmed_category, explode_decision, notes)."""
    v = ovr_items.get(key)
    if not isinstance(v, dict):
        return (None, None, None)
    cat = v.get("confirmed_category")
    exp = v.get("explode_decision")
    notes = v.get("notes")
    return (
        str(cat).strip().upper() if isinstance(cat, str) and cat.strip() else None,
        str(exp).strip() if isinstance(exp, str) and exp.strip() else None,
        str(notes).strip() if isinstance(notes, str) and notes.strip() else None,
    )


# ----------------------------
# Classification
# ----------------------------

@dataclass(frozen=True)
class ClassResult:
    category: str
    confidence: float
    reason: str


def _norm_name(name: str) -> str:
    return (name or "").strip().lower()


def _has_any(s: str, terms: Iterable[str]) -> bool:
    for t in terms:
        if t and t in s:
            return True
    return False


_HARDWARE_TERMS = (
    "bolt", "nut", "washer", "stud", "screw", "anchor", "rawl", "rivet", "pin", "dowel",
    "thread", "m6", "m8", "m10", "m12", "m16", "m20", "m24", "m30",
)

_HANDRAIL_TERMS = (
    "handrail", "hand rail", "railing", "rail ", "baluster", "stanchion", "newel",
)

_GRATING_TERMS = (
    "grating", "grate", "walkway", "mesh", "bar grate",
)

_TREAD_TERMS = (
    "tread", "stair", "step", "nosing", "riser",
)

_PLATE_NAME_TERMS = (
    "plate", "pl ", "pl-", "chequer", "checker", "treadplate", "tread plate", "durbar", "floorplate",
)

_SECTION_TERMS = (
    "ub", "uc", "pfc", "rsj", "shs", "rhs", "chs", "ea", "ua", "angle", "channel", "c-section",
    "ipe", "hea", "heb", "hem", "hss", "tee", "t-section", "flat bar", "fb ",
)


def classify_item(
    *,
    name: str,
    bbox_xyz: Tuple[float, float, float],
    override_category: Optional[str] = None,
    override_notes: Optional[str] = None,
) -> ClassResult:
    """Layered name-first classification with a simple plate geometry fallback."""

    if override_category and override_category in CATEGORY_ORDER:
        reason = "User override"
        if override_notes:
            reason = f"User override: {override_notes}"
        return ClassResult(override_category, 1.0, reason)

    nm = _norm_name(name)

    # 1) HARDWARE (name)
    if _has_any(nm, _HARDWARE_TERMS):
        return ClassResult("HARDWARE", 0.95, "Name matched hardware keyword")

    # 2) HANDRAIL / GRATING / TREAD (name)
    if _has_any(nm, _HANDRAIL_TERMS):
        return ClassResult("HANDRAIL", 0.95, "Name matched handrail keyword")
    if _has_any(nm, _GRATING_TERMS):
        return ClassResult("GRATING", 0.95, "Name matched grating keyword")
    if _has_any(nm, _TREAD_TERMS):
        return ClassResult("TREAD", 0.95, "Name matched tread/stair keyword")

    # 3) SECTIONS (name)
    if _has_any(nm, _SECTION_TERMS):
        return ClassResult("SECTION", 0.90, "Name matched section keyword")

    # 4) PLATE by name keywords
    if _has_any(nm, _PLATE_NAME_TERMS):
        return ClassResult("PLATE", 0.90, "Name matched plate keyword")

    # 5) PLATE by bbox thickness heuristic
    x, y, z = bbox_xyz
    dims = sorted([abs(x), abs(y), abs(z)])
    t = dims[0]
    a = dims[1]
    b = dims[2]

    # Guard unknown/degenerate dims
    if t > 0 and a > 0 and b > 0:
        # plate-ish if one dimension is much smaller than the other two
        ratio1 = t / a
        ratio2 = t / b
        # Practical defaults (tweakable later)
        if (t <= 50.0) and (ratio1 <= 0.20) and (ratio2 <= 0.20):
            return ClassResult("PLATE", 0.80, f"BBox thin-dimension heuristic (t={t:.1f}mm)")

    return ClassResult("REVIEW", 0.20, "No strong name match; geometry inconclusive")


# ----------------------------
# CSV writing
# ----------------------------

def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


# ----------------------------
# Gallery + tiny review server
# ----------------------------

def _relpath_or_abs(target: str, base_dir: Path) -> str:
    """Try to make a forward-slash relpath for browser loading; fall back to string."""
    if not target:
        return ""
    try:
        rel = os.path.relpath(str(Path(target).resolve()), start=str(base_dir.resolve()))
        return rel.replace("\\", "/")
    except Exception:
        return str(target)


def _pick_thumb(defn: Dict[str, Any], asset: Optional[AssetRec]) -> str:
    # Try common fields from Step 1
    for k in ("thumb_path", "png_path", "preview_png", "thumbnail", "thumb"):
        v = defn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Assets manifest might carry png in future
    if asset is not None:
        v2 = _get(asset.__dict__, "png_path", default="")
        if isinstance(v2, str) and v2.strip():
            return v2.strip()
    return ""


def write_gallery_html(
    out_dir: Path,
    review_rows: List[Dict[str, Any]],
    overrides_path: Path,
    *,
    title: str = "Step 4 Review Queue",
) -> Path:
    """Write a simple gallery that can save overrides when served via --serve."""

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "gallery.html"

    # Build cards HTML
    cards = []
    for r in review_rows:
        key = str(r.get("override_key", ""))
        name = escape(str(r.get("name", "")))
        def_id = escape(str(r.get("def_id", "")))
        qty = escape(str(r.get("qty_total", "")))
        solids = escape(str(r.get("solid_count", "")))
        bbox = f"{r.get('bbox_x','')} × {r.get('bbox_y','')} × {r.get('bbox_z','')}"
        bbox = escape(bbox)
        cat = escape(str(r.get("category", "REVIEW")))
        conf = escape(str(r.get("confidence", "")))
        reason = escape(str(r.get("reason", "")))
        thumb = str(r.get("thumb_rel", ""))
        img_html = f'<img src="{escape(thumb)}" alt="thumb" />' if thumb else ""

        cards.append(
            f"""
            <div class="card" data-key="{escape(key)}">
              <div class="thumb">{img_html}</div>
              <div class="meta">
                <div class="name">{name}</div>
                <div class="sub">def_id: <code>{def_id}</code> • qty: <b>{qty}</b> • solids: <b>{solids}</b></div>
                <div class="sub">bbox: <code>{bbox}</code></div>
                <div class="pred">Predicted: <b>{cat}</b> (conf {conf})</div>
                <div class="reason">{reason}</div>

                <div class="actions">
                  <label>Confirm category:
                    <select class="cat">
                      {"".join([f'<option value="{c}"' + (' selected' if c==r.get('category') else '') + f'>{c}</option>' for c in CATEGORY_ORDER])}
                    </select>
                  </label>

                  <label>Explode?
                    <select class="explode">
                      <option value="">(unset)</option>
                      <option value="Yes">Yes</option>
                      <option value="No">No</option>
                      <option value="Not sure">Not sure</option>
                    </select>
                  </label>

                  <label class="notes">Notes:
                    <input class="note" type="text" placeholder="optional" />
                  </label>

                  <button class="save">Save</button>
                  <span class="status"></span>
                </div>
              </div>
            </div>
            """
        )

    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 18px; background:#f6f7fb; }
      h1 { margin: 0 0 8px 0; font-size: 20px; }
      p  { margin: 0 0 14px 0; color:#444; }
      .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }
      .card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,.05); display:flex; overflow:hidden; }
      .thumb { width: 160px; min-height: 140px; background:#111827; display:flex; align-items:center; justify-content:center; }
      .thumb img { max-width: 160px; max-height: 160px; object-fit: contain; display:block; }
      .meta { padding: 12px; flex:1; }
      .name { font-weight: 700; margin-bottom: 4px; }
      .sub { color:#555; font-size: 13px; margin-bottom: 3px; }
      .pred { margin: 8px 0 4px 0; }
      .reason { color:#333; font-size: 13px; }
      .actions { margin-top: 10px; display:flex; flex-wrap: wrap; gap: 8px; align-items: center; }
      label { font-size: 13px; color:#222; display:flex; gap:6px; align-items:center; }
      select, input { padding: 4px 6px; border: 1px solid #d1d5db; border-radius: 6px; }
      button { padding: 6px 10px; border: 1px solid #111827; background:#111827; color:white; border-radius: 8px; cursor:pointer; }
      button:hover { filter: brightness(1.08); }
      .status { font-size: 13px; color:#065f46; }
      .warn { color:#b45309; }
      .error { color:#b91c1c; }
      code { background:#f3f4f6; padding: 1px 4px; border-radius: 4px; }
      .banner { display:flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      .pill { font-size: 12px; padding: 3px 8px; border-radius: 999px; border: 1px solid #d1d5db; background:white; }
    </style>
    """

    js = f"""
    <script>
      async function postJSON(url, obj){{
        const r = await fetch(url, {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(obj) }});
        if(!r.ok) throw new Error('HTTP '+r.status);
        return await r.json();
      }}

      function setStatus(card, text, cls){{
        const el = card.querySelector('.status');
        el.textContent = text || '';
        el.className = 'status ' + (cls||'');
      }}

      document.addEventListener('click', async (ev)=>{{
        const btn = ev.target.closest('button.save');
        if(!btn) return;
        const card = btn.closest('.card');
        const key  = card.getAttribute('data-key');
        const cat  = card.querySelector('select.cat').value;
        const exp  = card.querySelector('select.explode').value;
        const note = card.querySelector('input.note').value;

        setStatus(card, 'Saving…');
        try {{
          const res = await postJSON('/api/save', {{ key, confirmed_category: cat, explode_decision: exp, notes: note }});
          if(res && res.ok) setStatus(card, 'Saved ✓');
          else setStatus(card, 'Save failed', 'error');
        }} catch(e) {{
          console.error(e);
          setStatus(card, 'Save failed (are you using --serve?)', 'error');
        }}
      }});
    </script>
    """

    meta = {
        "generated_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "overrides_path": str(overrides_path),
        "review_count": len(review_rows),
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{escape(title)}</title>
  {css}
</head>
<body>
  <div class="banner">
    <h1>{escape(title)}</h1>
    <span class="pill">items needing attention: <b>{len(review_rows)}</b></span>
    <span class="pill">Save writes to: <code>{escape(str(overrides_path))}</code> (only when served via <code>--serve</code>)</span>
  </div>
  <p>Review only what matters: items in <b>REVIEW</b> and/or <b>multi-body</b> items (solid_count &gt; 1). Choose a category and decide whether to explode.</p>

  <div class="grid">
    {"".join(cards)}
  </div>

  <script type="application/json" id="meta">{escape(json.dumps(meta))}</script>
  {js}
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


class _ReviewHandler(SimpleHTTPRequestHandler):
    """Serve static files and accept POST /api/save to update review_overrides.json."""

    # Set by factory
    overrides_path: Path = Path("review_overrides.json")

    def _send_json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/save":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n <= 0 or n > (1 << 20):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "bad content length"})
            return

        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})
            return

        key = str(payload.get("key") or "").strip()
        if not key:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing key"})
            return

        confirmed_category = payload.get("confirmed_category")
        explode_decision = payload.get("explode_decision")
        notes = payload.get("notes")

        data = _read_overrides(self.overrides_path)
        items = data.get("items")
        if not isinstance(items, dict):
            items = {}
            data["items"] = items

        rec: Dict[str, Any] = items.get(key) if isinstance(items.get(key), dict) else {}
        if isinstance(confirmed_category, str) and confirmed_category.strip():
            rec["confirmed_category"] = confirmed_category.strip().upper()
        if isinstance(explode_decision, str):
            # allow empty to clear
            rec["explode_decision"] = explode_decision.strip()
        if isinstance(notes, str):
            rec["notes"] = notes.strip()
        rec["updated_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")

        items[key] = rec
        _write_overrides(self.overrides_path, data)

        self._send_json(HTTPStatus.OK, {"ok": True})


def serve_review(out_dir: Path, overrides_path: Path, port: int) -> None:
    """Serve out_dir and accept override saves."""

    # Bound port, no daemon threads, deterministic and simple.
    class Handler(_ReviewHandler):
        overrides_path = overrides_path

    os.chdir(str(out_dir))
    with TCPServer(("", port), Handler) as httpd:
        sa = httpd.socket.getsockname()
        print(f"[step4] Serving review UI at http://{sa[0] or 'localhost'}:{sa[1]}/gallery.html")
        print("[step4] Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[step4] Server stopped.")


# ----------------------------
# Main build
# ----------------------------

def build_step4(
    xcaf_path: Path,
    assets_path: Path,
    out_dir: Path,
    *,
    write_html: bool,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Return (master_csv_path, review_rows_for_gallery)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    overrides_path = out_dir / "review_overrides.json"

    xcaf = _read_json(xcaf_path)
    defs_map = _ensure_defs_map(xcaf.get("definitions"))

    assets = _read_json(assets_path) if assets_path.exists() else {}
    asset_by_def = _assets_index(assets)

    overrides = _read_overrides(overrides_path)
    ovr_items = overrides.get("items") if isinstance(overrides, dict) else {}
    if not isinstance(ovr_items, dict):
        ovr_items = {}

    rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []

    for def_id, defn in defs_map.items():
        if not isinstance(defn, dict):
            continue

        name = str(_get(defn, "name", "def_name", default=""))
        qty_total = _as_int(_get(defn, "qty_total", "qty", "quantity", default=0), default=0)
        shape_kind = str(_get(defn, "shape_kind", "kind", "shapeType", default=""))
        solid_count = _as_int(_get(defn, "solid_count", "solids", "n_solids", default=0), default=0)

        # Include if it has a shape, or Step 2/3 managed to make an asset for it
        has_shape = _as_bool(_get(defn, "has_shape", default=False)) or (shape_kind.strip() != "")
        asset = asset_by_def.get(def_id)
        has_asset = asset is not None and bool(asset.stl_path)
        if not (has_shape or has_asset):
            continue

        bbox_x, bbox_y, bbox_z = _extract_bbox_dims(defn)

        def_sig_used = asset.def_sig_used if asset else ""
        chirality_sig_free = asset.chirality_sig_free if asset else ""
        stl_path = asset.stl_path if asset else ""

        # Quantize bbox into a coarse key to help overrides survive tiny numeric noise
        dims_sorted = sorted([abs(bbox_x), abs(bbox_y), abs(bbox_z)])
        bbox_q = f"{dims_sorted[0]:.1f},{dims_sorted[1]:.1f},{dims_sorted[2]:.1f}"
        override_key = _stable_key_for(def_id, def_sig_used, chirality_sig_free, bbox_q)

        o_cat, o_exp, o_notes = _apply_override(ovr_items, override_key)

        cls = classify_item(
            name=name,
            bbox_xyz=(bbox_x, bbox_y, bbox_z),
            override_category=o_cat,
            override_notes=o_notes,
        )

        explode_recommended = bool(solid_count > 1)

        # needs_review:
        # - category is REVIEW
        # - or multi-body with no user decision
        needs_review = (cls.category == "REVIEW")
        if explode_recommended and not (isinstance(o_exp, str) and o_exp.strip()):
            needs_review = True

        row = {
            "def_id": def_id,
            "name": name,
            "qty_total": qty_total,
            "shape_kind": shape_kind,
            "solid_count": solid_count,
            "bbox_x": round(bbox_x, 3) if bbox_x else 0.0,
            "bbox_y": round(bbox_y, 3) if bbox_y else 0.0,
            "bbox_z": round(bbox_z, 3) if bbox_z else 0.0,
            "def_sig_used": def_sig_used,
            "chirality_sig_free": chirality_sig_free,
            "stl_path": stl_path,
            "category": cls.category,
            "confidence": round(float(cls.confidence), 3),
            "reason": cls.reason,
            "needs_review": "1" if needs_review else "0",
            "explode_recommended": "1" if explode_recommended else "0",
            # internal (not written to CSV):
            "override_key": override_key,
        }

        rows.append(row)

        if needs_review:
            thumb_path = _pick_thumb(defn, asset)
            thumb_rel = _relpath_or_abs(thumb_path, out_dir) if thumb_path else ""
            rr = dict(row)
            rr["thumb_rel"] = thumb_rel
            rr["explode_decision"] = o_exp or ""
            rr["notes"] = o_notes or ""
            review_rows.append(rr)

    # Deterministic sort
    cat_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            cat_index.get(str(r.get("category", "REVIEW")), len(CATEGORY_ORDER)),
            str(r.get("name", "")).lower(),
            str(r.get("def_id", "")),
        ),
    )

    # Output paths
    master_csv = out_dir / "master_items.csv"
    _write_csv(master_csv, rows_sorted, OUTPUT_COLUMNS)

    # Split outputs
    by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORY_ORDER}
    for r in rows_sorted:
        c = str(r.get("category", "REVIEW"))
        if c not in by_cat:
            c = "REVIEW"
        by_cat[c].append(r)

    # File naming map
    file_map = {
        "PLATE": "plates.csv",
        "SECTION": "sections.csv",
        "HARDWARE": "hardware.csv",
        "HANDRAIL": "handrail.csv",
        "TREAD": "treads.csv",
        "GRATING": "grating.csv",
        "REVIEW": "review.csv",
    }

    for cat, fname in file_map.items():
        _write_csv(out_dir / fname, by_cat.get(cat, []), OUTPUT_COLUMNS)

    # Ensure overrides file exists (even if empty) so review server can write predictably
    if not overrides_path.exists():
        _write_overrides(overrides_path, {"schema": "step4_review_overrides_v1", "items": {}})

    if write_html:
        # Only show the queue (review rows), and keep it deterministic
        review_rows_sorted = sorted(
            review_rows,
            key=lambda r: (
                str(r.get("category", "REVIEW")) != "REVIEW",  # REVIEW first
                str(r.get("name", "")).lower(),
                str(r.get("def_id", "")),
            ),
        )
        write_gallery_html(out_dir, review_rows_sorted, overrides_path)

    return master_csv, review_rows


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4 procurement CSVs + optional review gallery.")
    p.add_argument("--xcaf", default="/out/xcaf_instances.json", help="Path to xcaf_instances.json")
    p.add_argument("--assets", default="/out/assets_manifest.json", help="Path to assets_manifest.json")
    p.add_argument("--out", default="/out/step4", help="Output directory (will contain CSVs + overrides)")
    p.add_argument("--write-html", action="store_true", help="Write gallery.html for the review queue")
    p.add_argument("--serve", type=int, default=0, help="Serve out_dir and enable Save → review_overrides.json (port)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = _parse_args(argv)

    xcaf_path = Path(ns.xcaf)
    assets_path = Path(ns.assets)
    out_dir = Path(ns.out)

    if not xcaf_path.exists():
        print(f"[step4] ERROR: xcaf_instances.json not found: {xcaf_path}")
        return 2

    if not assets_path.exists():
        print(f"[step4] WARN: assets_manifest.json not found: {assets_path} (continuing without assets join)")

    master_csv, review_rows = build_step4(
        xcaf_path=xcaf_path,
        assets_path=assets_path,
        out_dir=out_dir,
        write_html=bool(ns.write_html or ns.serve),
    )

    print(f"[step4] Wrote: {master_csv}")
    print(f"[step4] Review queue size: {len(review_rows)}")
    print(f"[step4] Outputs: {out_dir}")

    if int(ns.serve) > 0:
        overrides_path = out_dir / "review_overrides.json"
        serve_review(out_dir, overrides_path, int(ns.serve))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
