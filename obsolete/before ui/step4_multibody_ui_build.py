from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DECISION_ENUM = ("keep_as_one", "explode", "defer")


DECISION_ENUM = ("keep_as_one", "explode", "defer")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_review_items(path: Path) -> List[Dict[str, Any]]:
    """
    Reads multibody_review.json written by step4_multibody_review.py

    Expected:
      {"schema":"multibody_review_v1", "items":[...]}
    """
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"review JSON must be an object: {path}")
    items = obj.get("items")
    if not isinstance(items, list):
        raise ValueError(f"review JSON missing items[]: {path}")
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(it)
    return out


def _load_decisions(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Reads multibody_decisions.json

    Expected:
      {"schema":"multibody_decisions_v1", "decisions": {"<def_sig>": {"decision":"...", "note":"..."}}}
    Returns:
      def_sig -> {"decision": "...", "note": "..."}
    """
    out: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return out

    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"decisions JSON must be an object: {path}")

    decs = obj.get("decisions")
    if not isinstance(decs, dict):
        raise ValueError(f"decisions JSON missing decisions{{}}: {path}")

    for sig, rec in decs.items():
        if not isinstance(sig, str) or not sig.strip():
            continue
        if not isinstance(rec, dict):
            continue

        decision = rec.get("decision", "defer")
        note = rec.get("note", "")

        if not isinstance(decision, str):
            decision = "defer"
        if not isinstance(note, str):
            note = ""

        decision = decision.strip() or "defer"
        note = note.strip()

        if decision not in DECISION_ENUM:
            decision = "defer"

        out[sig.strip()] = {"decision": decision, "note": note}

    return out


def _load_assets_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("assets_manifest.json not in expected format (list or {'items':[...]}).")


def _build_sig_to_stl_map(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map signature -> stl_path.

    We key by:
      - def_sig_used (preferred, since your manifest has it)
      - chirality_sig (sometimes useful later)
      - part_id (not used for join, but ok)
    """
    out: Dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        stl_path = it.get("stl_path")
        if not isinstance(stl_path, str) or not stl_path:
            continue
        for k in ("def_sig_used", "chirality_sig", "chirality_sig_free"):
            v = it.get(k)
            if isinstance(v, str) and v:
                # first win; keep deterministic
                if v not in out:
                    out[v] = stl_path
    return out

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _safe_copy(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_ui(out_dir: Path) -> None:
    review_json = out_dir / "review" / "multibody_review.json"
    decisions_json = out_dir / "review" / "multibody_decisions.json"
    assets_manifest = out_dir / "assets_manifest.json"

    if not review_json.exists():
        raise SystemExit(f"Missing: {review_json}")
    if not decisions_json.exists():
        raise SystemExit(f"Missing: {decisions_json}")
    if not assets_manifest.exists():
        raise SystemExit(f"Missing: {assets_manifest}")

    review_items = _load_review_items(review_json)
    decisions = _load_decisions(decisions_json)
    manifest_items = _load_assets_manifest(assets_manifest)
    sig_to_stl = _build_sig_to_stl_map(manifest_items)
    manifest_items = _load_assets_manifest(assets_manifest)
    sig_to_stl = _build_sig_to_stl_map(manifest_items)

    ui_dir = out_dir / "review_ui"
    stl_cache_dir = ui_dir / "stl"
    ui_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    copied = 0
    missing = 0

    for r in review_items:
        def_sig = (r.get("def_sig") or "").strip()
        def_sig_free = (r.get("def_sig_free") or "").strip()
        name = (r.get("name") or "").strip()

        if not def_sig:
            continue

        # Join to assets manifest:
        # manifest uses def_sig_used. That might match def_sig OR def_sig_free.
        stl_rel = ""
        stl_rel = sig_to_stl.get(def_sig, "")
        if (not stl_rel) and def_sig_free:
            stl_rel = sig_to_stl.get(def_sig_free, "")

        stl_url = ""  # served from /stl/<file>
        if stl_rel:
            src = out_dir / stl_rel
            if src.exists():
                dst = stl_cache_dir / src.name
                if _safe_copy(src, dst):
                    copied += 1
                    stl_url = f"stl/{dst.name}"
                else:
                    missing += 1
            else:
                missing += 1
        else:
            missing += 1

        d = decisions.get(def_sig, {"decision": "defer", "note": ""})

        items.append(
            {
                "def_sig": def_sig,
                "def_sig_free": def_sig_free,
                "name": name,
                "qty_total": _safe_int(r.get("qty_total"), 0),
                "solid_count": _safe_int(r.get("solid_count"), 0),
                "bbox": (r.get("bbox") or "").strip(),
                "bucket": (r.get("bucket") or "").strip(),
                "reason": (r.get("reason") or "").strip(),
                "stl_url": stl_url,
                "decision": d["decision"],
                "note": d["note"],
            }
        )

    payload = {
        "generated": True,
        "items_count": len(items),
        "items": items,
    }

    _write_json(ui_dir / "items.json", payload)
    _write_text(ui_dir / "index.html", _INDEX_HTML)
    _write_text(ui_dir / "styles.css", _STYLES_CSS)
    _write_text(ui_dir / "app.js", _APP_JS)

    # ---- Offline-friendly vendor copy (serve local three.js modules) ----
    vendor_src = Path("vendor")  # repo root
    vendor_dst = ui_dir / "vendor"
    if vendor_src.exists() and vendor_src.is_dir():
        vendor_dst.mkdir(parents=True, exist_ok=True)
        for fn in ("three.module.js", "OrbitControls.js", "STLLoader.js"):
            src = vendor_src / fn
            if src.exists():
                shutil.copy2(src, vendor_dst / fn)
            else:
                print(f"[warn] missing vendor file: {src}")
    else:
        print("[warn] vendor/ directory not found at repo root; UI may fail without CDN.")

    print(f"[ok] UI written: {ui_dir}")
    print(f"[ok] items: {len(items)}")
    print(f"[ok] stl copied: {copied}   missing: {missing}")
    print(f"[hint] run server: python step4_multibody_ui_server.py --outdir {out_dir}")


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Step 4 - Multi-body Review</title>

  <!-- Import map is required because OrbitControls/STLLoader use `import ... from "three"` -->
  <script type="importmap">
  {
    "imports": {
      "three": "./vendor/three.module.js"
    }
  }
  </script>

  <link rel="stylesheet" href="styles.css"/>
</head>
<body>
  <header>
    <div class="title">
      <div class="h1">Step 4 - Multi-body Review</div>
      <div class="h2">Select an item to preview STL, set decision, save.</div>
    </div>
    <div class="actions">
      <button id="btnSave" class="primary">Save decisions</button>
      <span id="saveStatus" class="status"></span>
    </div>
  </header>

  <main>
    <section class="left">
      <div class="filters">
        <button class="pill" data-bucket="all">All</button>
        <button class="pill" data-bucket="likely_explode">Likely explode</button>
        <button class="pill" data-bucket="review">Review</button>
        <button class="pill" data-bucket="auto_keep">Auto keep</button>
        <input id="q" class="search" placeholder="Search name or sig..."/>
      </div>

      <div id="list" class="list"></div>
    </section>

    <section class="right">
      <div class="viewerHeader">
        <div id="selTitle" class="selTitle">No item selected</div>
        <div id="selMeta" class="selMeta"></div>
      </div>

      <div class="viewerWrap">
        <canvas id="cv"></canvas>
        <div id="viewerHint" class="viewerHint">Select an item with an STL to preview</div>
      </div>

      <div class="editor">
        <div class="row">
          <label>Decision</label>
          <div class="radioGroup">
            <label><input type="radio" name="decision" value="keep_as_one"> keep_as_one</label>
            <label><input type="radio" name="decision" value="explode"> explode</label>
            <label><input type="radio" name="decision" value="defer"> defer</label>
          </div>
        </div>
        <div class="row">
          <label>Note</label>
          <input id="note" class="note" placeholder="Optional note..."/>
        </div>
      </div>
    </section>
  </main>

  <script type="module" src="app.js"></script>
</body>
</html>
"""

_STYLES_CSS = r"""
:root { font-family: system-ui, Segoe UI, Arial, sans-serif; }

/* Lock the page to the viewport; prevent body scrolling */
html, body { height: 100%; overflow: hidden; }

body { margin: 0; background: #0f1115; color: #e8e8e8; }

header {
  display:flex; align-items:center; justify-content:space-between;
  padding: 12px 16px; border-bottom: 1px solid #222;
}

.h1 { font-size: 18px; font-weight: 700; }
.h2 { font-size: 12px; opacity: 0.7; margin-top: 2px; }
.actions { display:flex; align-items:center; gap: 10px; }
.primary { background: #2f6fed; border: 0; color: #fff; padding: 10px 12px; border-radius: 10px; cursor:pointer; }
.status { font-size: 12px; opacity: 0.8; }

/* Main layout is fixed height; no scrolling here */
main {
  display:grid;
  grid-template-columns: 420px 1fr;
  height: calc(100vh - 58px);
  overflow: hidden;
}

/* Left pane: fixed column, only the list scrolls */
.left {
  border-right: 1px solid #222;
  display:flex;
  flex-direction:column;
  min-height: 0;
  overflow: hidden;
}

/* Right pane: fixed, no scrolling; viewer fills space */
.right {
  display:flex;
  flex-direction:column;
  min-height: 0;
  overflow: hidden;
}

.filters { display:flex; gap:8px; padding: 10px; flex-wrap:wrap; border-bottom: 1px solid #222; }
.pill { background: #1a1d25; border: 1px solid #2a2f3b; color:#e8e8e8; padding: 6px 10px; border-radius: 999px; cursor:pointer; font-size: 12px; }
.pill.active { border-color: #2f6fed; }
.search { flex:1; min-width: 160px; background:#0c0e12; border:1px solid #2a2f3b; color:#e8e8e8; padding: 8px 10px; border-radius: 10px; }

/* Only this scrolls */
.list {
  flex: 1;
  overflow: auto;
  padding: 8px;
  min-height: 0;
}

/* Cards */
.card {
  border: 1px solid #242a35;
  background:#12151c;
  border-radius: 14px;
  padding: 10px;
  margin-bottom: 8px;
  cursor:pointer;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.card.selected { outline: 2px solid #2f6fed; }

/* Decision color borders */
.card.decision-keep   { border-color: #2f6fed; box-shadow: 0 0 0 1px rgba(47,111,237,0.20); }
.card.decision-explode{ border-color: #2ad17a; box-shadow: 0 0 0 1px rgba(42,209,122,0.20); }
.card.decision-defer  { border-color: #ff4d4d; box-shadow: 0 0 0 1px rgba(255,77,77,0.18); }

.cardTop { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
.nameRow { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.name { font-weight: 700; font-size: 13px; }
.small { font-size: 12px; opacity: 0.8; }

.tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; background:#1a1d25; border:1px solid #2a2f3b; align-self:flex-start; }

.row2 { display:flex; gap:10px; margin-top:6px; flex-wrap:wrap; }
.kv { font-size: 12px; opacity:0.85; }
.sig { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; opacity:0.75; margin-top:6px; word-break:break-all; }

/* Decision buttons inside card */
.decBtns { display:flex; gap:6px; }
.decBtn {
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid #2a2f3b;
  background: #0c0e12;
  color: #e8e8e8;
  cursor: pointer;
  user-select: none;
}
.decBtn:hover { border-color: #3a4150; }

.decBtn.keep.active    { border-color: #2f6fed; }
.decBtn.explode.active { border-color: #2ad17a; }
.decBtn.defer.active   { border-color: #ff4d4d; }

/* Small dot indicator next to name */
.dot {
  width: 8px; height: 8px; border-radius: 999px; display:inline-block;
  border: 1px solid #2a2f3b;
}
.dot.keep    { background: #2f6fed; border-color: #2f6fed; }
.dot.explode { background: #2ad17a; border-color: #2ad17a; }
.dot.defer   { background: #ff4d4d; border-color: #ff4d4d; }

.viewerHeader { padding: 10px 12px; border-bottom: 1px solid #222; }
.selTitle { font-weight: 700; }
.selMeta { font-size: 12px; opacity: 0.75; margin-top: 2px; }

/* Viewer stays pinned and fills available space; no scrolling */
.viewerWrap {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  background:#0b0d11;
}

#cv { width: 100%; height: 100%; display:block; }

.viewerHint {
  position:absolute; left: 14px; bottom: 14px;
  font-size: 12px; opacity: 0.75;
  background: rgba(0,0,0,0.35);
  padding: 6px 10px; border-radius: 10px;
}

.editor { padding: 12px; border-top: 1px solid #222; display:flex; flex-direction:column; gap: 12px; }
.row label { display:block; font-size: 12px; opacity: 0.8; margin-bottom: 6px; }
.radioGroup { display:flex; gap:14px; flex-wrap:wrap; }
.note { width: 100%; background:#0c0e12; border:1px solid #2a2f3b; color:#e8e8e8; padding: 10px 10px; border-radius: 10px; }
"""

_APP_JS = r"""
import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { STLLoader } from "./vendor/STLLoader.js";

const $ = (id) => document.getElementById(id);

let state = {
  items: [],
  filtered: [],
  selectedIndex: -1,
  activeBucket: "all",
  q: "",
};

function bucketLabel(b) {
  if (b === "likely_explode") return "likely_explode";
  if (b === "auto_keep") return "auto_keep";
  return "review";
}

function matchesQuery(item, q) {
  if (!q) return true;
  const s = q.toLowerCase();
  return (item.name || "").toLowerCase().includes(s) ||
         (item.def_sig || "").toLowerCase().includes(s) ||
         (item.def_sig_free || "").toLowerCase().includes(s);
}

function applyFilters() {
  state.filtered = state.items.filter(it => {
    const bucketOk = (state.activeBucket === "all") ? true : (it.bucket === state.activeBucket);
    const qOk = matchesQuery(it, state.q);
    return bucketOk && qOk;
  });
  renderList();
}

function setDecision(it, decision) {
  it.decision = decision;
  renderList();
  // keep editor radios in sync if selected item changed
  const sel = state.filtered[state.selectedIndex];
  if (sel && sel.def_sig === it.def_sig) {
    const radios = document.querySelectorAll('input[name="decision"]');
    radios.forEach(r => { r.checked = (r.value === it.decision); });
  }
}

function decisionBadge(decision) {
  if (decision === "explode") return "explode";
  if (decision === "keep_as_one") return "keep";
  return "defer";
}

function renderList() {
  const list = $("list");
  list.innerHTML = "";

  for (let i = 0; i < state.filtered.length; i++) {
    const it = state.filtered[i];

    const div = document.createElement("div");
    div.className = "card";
    if (i === state.selectedIndex) div.classList.add("selected");

    const tag = bucketLabel(it.bucket);
    const dtag = decisionBadge(it.decision);

    div.classList.add("decision-" + dtag);

    const dotClass = (dtag === "keep") ? "keep" : (dtag === "explode") ? "explode" : "defer";

    div.innerHTML = `
      <div class="cardTop">
        <div>
          <div class="nameRow">
            <span class="dot ${dotClass}"></span>
            <div class="name">${escapeHtml(it.name || "(unnamed)")}</div>

            <div class="decBtns">
              <button class="decBtn keep ${it.decision === "keep_as_one" ? "active" : ""}" data-dec="keep_as_one">Keep</button>
              <button class="decBtn explode ${it.decision === "explode" ? "active" : ""}" data-dec="explode">Explode</button>
              <button class="decBtn defer ${it.decision === "defer" ? "active" : ""}" data-dec="defer">Defer</button>
            </div>
          </div>

          <div class="small">${escapeHtml(it.bbox || "")}</div>
        </div>

        <div class="tag">${tag}</div>
      </div>

      <div class="row2">
        <div class="kv">qty: <b>${it.qty_total}</b></div>
        <div class="kv">solids: <b>${it.solid_count}</b></div>
        <div class="kv">${escapeHtml(it.reason || "")}</div>
      </div>

      <div class="sig">${escapeHtml(it.def_sig)}</div>
    `;

    // Card click selects item
    div.addEventListener("click", () => selectIndex(i));

    // Decision button clicks: update decision without forcing selection changes/mouse travel
    div.querySelectorAll(".decBtn").forEach(btn => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const dec = btn.dataset.dec;
        if (!dec) return;
        setDecision(it, dec);
      });
    });

    list.appendChild(div);
  }

  if (state.filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "small";
    empty.style.padding = "10px";
    empty.textContent = "No items match filters.";
    list.appendChild(empty);
  }
}

function selectIndex(idx) {
  state.selectedIndex = idx;
  renderList();

  const it = state.filtered[idx];
  if (!it) return;

  $("selTitle").textContent = it.name || "(unnamed)";
  $("selMeta").textContent = `bucket=${it.bucket}  qty=${it.qty_total}  solids=${it.solid_count}  bbox=${it.bbox || ""}`;

  // Set editor fields
  const radios = document.querySelectorAll('input[name="decision"]');
  radios.forEach(r => { r.checked = (r.value === it.decision); });
  $("note").value = it.note || "";

  // Load STL into single viewer
  if (it.stl_url) {
    $("viewerHint").style.display = "none";
    loadSTL(it.stl_url);
    onResize();
  } else {
    clearScene();
    $("viewerHint").style.display = "block";
    $("viewerHint").textContent = "No STL available for this item (stl_url missing).";
  }
}

function escapeHtml(s) {
  return (s || "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
                  .replaceAll('"',"&quot;").replaceAll("'","&#039;");
}

// Persist changes from editor into state (only selected item)
function wireEditor() {
  document.querySelectorAll('input[name="decision"]').forEach(r => {
    r.addEventListener("change", () => {
      const it = state.filtered[state.selectedIndex];
      if (!it) return;
      it.decision = r.value;
      renderList();
    });
  });

  $("note").addEventListener("input", () => {
    const it = state.filtered[state.selectedIndex];
    if (!it) return;
    it.note = $("note").value;
  });

  $("btnSave").addEventListener("click", saveDecisions);
}

async function saveDecisions() {
  // We must save decisions for ALL items (not just filtered view).
  // state.items is the master list.
  const payload = {
    decisions: state.items.map(it => ({
      def_sig: it.def_sig,
      decision: it.decision || "defer",
      note: it.note || ""
    }))
  };

  $("saveStatus").textContent = "Saving...";
  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("save failed");
    const j = await res.json();
    $("saveStatus").textContent = `Saved (${j.updated || 0})`;
    setTimeout(() => { $("saveStatus").textContent = ""; }, 1500);
  } catch (e) {
    $("saveStatus").textContent = "Save error";
  }
}

// ---- Three.js single viewer ----

let renderer, scene, camera, controls, mesh, lightA, lightB;
const loader = new STLLoader();

function initViewer() {
  const canvas = $("cv");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d11);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
  camera.position.set(200, 200, 200);

  // Better shading: hemi + directional
  const hemi = new THREE.HemisphereLight(0xffffff, 0x202030, 0.75);
  scene.add(hemi);

  lightA = new THREE.DirectionalLight(0xffffff, 1.1);
  lightA.position.set(2, 3, 4);
  scene.add(lightA);

  lightB = new THREE.AmbientLight(0xffffff, 0.18);
  scene.add(lightB);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  camera.position.set(200, 200, 200);

  lightA = new THREE.DirectionalLight(0xffffff, 1.0);
  lightA.position.set(1, 1, 1);
  scene.add(lightA);

  lightB = new THREE.AmbientLight(0xffffff, 0.35);
  scene.add(lightB);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  window.addEventListener("resize", onResize);
  onResize();
  animate();
}

function onResize() {
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w > 0 && h > 0) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

function clearScene() {
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
    mesh = null;
  }
}

function loadSTL(url) {
  clearScene();
  loader.load(url, (geom) => {
    geom.computeVertexNormals();
        const mat = new THREE.MeshStandardMaterial({
      metalness: 0.05,
      roughness: 0.65
    });
    mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    // Compute bounds
    const box = new THREE.Box3().setFromObject(mesh);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    // Robust scale measures
    const maxDim = Math.max(size.x, size.y, size.z);
    const diag = size.length();

    // If mesh is degenerate, bail gracefully
    if (!isFinite(maxDim) || maxDim <= 0) {
      controls.target.set(0, 0, 0);
      camera.near = 0.1;
      camera.far = 100000;
      camera.updateProjectionMatrix();
      controls.update();
      return;
    }

    // Frame distance based on camera fov
    const fovRad = (camera.fov * Math.PI) / 180.0;
    const dist = (maxDim / (2 * Math.tan(fovRad / 2))) * 1.4; // 1.4 padding

    // Place camera on a nice diagonal
    const dir = new THREE.Vector3(1, 1, 1).normalize();
    camera.position.copy(center).addScaledVector(dir, dist);

    // Set target
    controls.target.copy(center);

    // Dynamic clipping planes to avoid "invisible until zoom" issues
    // near must be small enough; far must be comfortably beyond the model
    camera.near = Math.max(dist / 1000, 0.01);
    camera.far  = Math.max(dist * 100, diag * 10, 1000);
    camera.updateProjectionMatrix();

    // Controls limits (optional but helps UX)
    controls.minDistance = camera.near * 2;
    controls.maxDistance = camera.far * 0.5;

    controls.update();
  });
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

// ---- boot ----

async function boot() {
  initViewer();
  wireEditor();

  const res = await fetch("/api/items");
  const j = await res.json();
  state.items = j.items || [];

  // Default filter button state
  document.querySelectorAll(".pill").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeBucket = btn.dataset.bucket || "all";
      applyFilters();
    });
  });
  document.querySelector('.pill[data-bucket="all"]').classList.add("active");

  $("q").addEventListener("input", () => {
    state.q = $("q").value || "";
    applyFilters();
  });

  applyFilters();

  // Auto-select first item
  if (state.filtered.length > 0) selectIndex(0);
}

boot();
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory (host or container path), e.g. out or /out")
    ns = ap.parse_args()
    build_ui(Path(ns.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
