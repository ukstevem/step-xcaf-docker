// ui/app.js (full replacement) — clean, deterministic, no injected DOM
// Requires index.html to contain ids: treeSection, treeTitle, treeHint, viewAssembly, viewBom, tree, viewer, nodeMeta
// Uses API endpoints in ui_server.py: /api/create_run, /api/preview/{run}, /api/state/{run}, /api/progress/{run},
// /api/orientation/{run}, /api/tree_grouped/{run}, /api/bom/{run}, /api/explode_plan/{run}

const statusEl = document.getElementById("status");
const warningEl = document.getElementById("warning");
const progressEl = document.getElementById("progress");
const preflightInfoEl = document.getElementById("preflightInfo");
const selTopEl = document.getElementById("selTop");

const btnGo = document.getElementById("go");
const fileInp = document.getElementById("file");
const previewWrap = document.getElementById("previewViews");

// Tree/BOM pane elements (must exist in index.html)
const treeSectionEl = document.getElementById("treeSection");
const treeTitleEl = document.getElementById("treeTitle");
const treeHintEl = document.getElementById("treeHint");
const viewAssemblyBtn = document.getElementById("viewAssembly");
const viewBomBtn = document.getElementById("viewBom");
const treeEl = document.getElementById("tree");
const viewerEl = document.getElementById("viewer");
const nodeMetaEl = document.getElementById("nodeMeta");
const btnBomViewer = document.getElementById("btnBomViewer");
const btnBomPurchasing = document.getElementById("btnBomPurchasing");


// 6-face images (required in index.html)
const imgTop = document.getElementById("pv_top");
const imgBottom = document.getElementById("pv_bottom");
const imgFront = document.getElementById("pv_front");
const imgBack = document.getElementById("pv_back");
const imgLeft = document.getElementById("pv_left");
const imgRight = document.getElementById("pv_right");

// Optional “Use as TOP” buttons (OK if null)
const btnTop = document.getElementById("btnTop");
const btnBottom = document.getElementById("btnBottom");
const btnFront = document.getElementById("btnFront");
const btnBack = document.getElementById("btnBack");
const btnLeft = document.getElementById("btnLeft");
const btnRight = document.getElementById("btnRight");

// rotation controls
const rotSel = document.getElementById("rotSel");
const btnApplyOrient = document.getElementById("applyOrient");

const FACE_KEYS = ["top", "bottom", "front", "back", "left", "right"];

const faceImg = {
  top: imgTop,
  bottom: imgBottom,
  front: imgFront,
  back: imgBack,
  left: imgLeft,
  right: imgRight,
};

// ---- app state ----
let currentRunId = null;
let currentPreflight = null;
let currentOrientation = { plan_source: "top", rotation_deg: 0 };

let progressES = null;
let statePollTimer = null;

let currentView = "bom"; // "assembly" | "bom" | "bom_flat"
let treeData = null;
let bomData = null;
let treePollTimer = null;
let bomPollTimer = null;
let bomFlatData = null;

// explode plan (server-backed)
let explodePlan = { schema: "explode_plan.v1", run_id: null, items: {} };

// ---- viewer (three.js lazy) ----
let _THREE = null;
let _OrbitControls = null;
let _STLLoader = null;

let _viewerInit = false;
let _renderer = null;
let _scene = null;
let _camera = null;
let _controls = null;
let _currentMesh = null;

// ------------------------------
// small helpers
// ------------------------------
function setStatus(s) {
  if (statusEl) statusEl.textContent = s || "";
}

function firstNonEmpty(obj, keys) {
  for (const k of keys) {
    const v = obj?.[k];
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return "";
}

function makeLinkCell(label, url, title) {
  const td = document.createElement("td");
  td.className = "cell-link";

  const a = document.createElement("a");
  a.className = "lnk";
  a.textContent = label;
  a.title = title || label;

  if (!url) {
    a.classList.add("missing");
    a.href = "#";
  } else {
    // Most of our URLs are run-relative. If it’s already absolute, keep it.
    const abs = url.startsWith("http://") || url.startsWith("https://") || url.startsWith("/")
      ? url
      : resolveRunUrl(currentRunId, url);

    if (!abs) {
      a.classList.add("missing");
      a.href = "#";
    } else {
      a.href = abs;
      a.target = "_blank";
      a.rel = "noopener";
    }
  }

  td.appendChild(a);
  return td;
}

function fmtBbox(it) {
  // Supports several likely shapes: bbox_mm.size, bbox.size, bbox_mm, bbox
  const b = it?.bbox_mm || it?.bbox || null;
  const sz =
    (Array.isArray(b?.size) && b.size.length >= 3 && b.size) ||
    (Array.isArray(b) && b.length >= 3 && b) ||
    null;

  if (!sz) return "";
  const x = Number(sz[0]), y = Number(sz[1]), z = Number(sz[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return "";
  return `${x.toFixed(1)}×${y.toFixed(1)}×${z.toFixed(1)}`;
}

function setWarning(msg) {
  if (!warningEl) return;
  if (!msg) {
    warningEl.style.display = "none";
    warningEl.textContent = "";
    return;
  }
  warningEl.style.display = "block";
  warningEl.textContent = String(msg);
}

function showProgress(show) {
  if (!progressEl) return;
  progressEl.style.display = show ? "block" : "none";
  if (show) progressEl.textContent = "";
}

function tsLocal() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour12: false });
}

function appendProgress(raw) {
  if (!progressEl) return;

  const incoming = String(raw)
    .split(/\r?\n/g)
    .map((s) => s.trim())
    .filter(Boolean);

  if (incoming.length === 0) return;

  const existing = progressEl.textContent
    ? progressEl.textContent.split("\n").filter(Boolean)
    : [];

  for (const line of incoming) existing.push(`[${tsLocal()}] ${line}`);

  progressEl.textContent = existing.slice(-220).join("\n");
  progressEl.scrollTop = progressEl.scrollHeight;
}

function stopProgress() {
  try {
    if (progressES) progressES.close();
  } catch (_) {}
  progressES = null;
}

function stopStatePolling() {
  if (statePollTimer) clearInterval(statePollTimer);
  statePollTimer = null;
}

function stopTreePolling() {
  if (treePollTimer) clearInterval(treePollTimer);
  treePollTimer = null;
}

function stopBomPolling() {
  if (bomPollTimer) clearInterval(bomPollTimer);
  bomPollTimer = null;
}

function resolveRunUrl(runId, u) {
  if (!u) return null;
  const s = String(u);
  if (s.startsWith("http://") || s.startsWith("https://") || s.startsWith("/")) return s;
  return `/runs/${runId}/${s.replace(/^\.\/+/, "")}`;
}

function clearPreviewUI() {
  if (previewWrap) previewWrap.style.display = "none";
  if (preflightInfoEl) preflightInfoEl.textContent = "";

  FACE_KEYS.forEach((k) => {
    const img = faceImg[k];
    if (!img) return;
    img.classList.remove("selected");
    img.removeAttribute("src");
    img.style.transform = "";
  });

  currentPreflight = null;
  currentOrientation = { plan_source: "top", rotation_deg: 0 };
  if (rotSel) rotSel.value = "0";
  if (selTopEl) selTopEl.textContent = "TOP";

  setWarning("");
}

function highlightPlanSource(planSource) {
  FACE_KEYS.forEach((k) => faceImg[k]?.classList.remove("selected"));
  const src = (planSource || "top").toLowerCase();
  if (faceImg[src]) faceImg[src].classList.add("selected");
  if (selTopEl) selTopEl.textContent = src.toUpperCase();
}

function applyRotationCss(planSource, rotDeg) {
  const rot = Number(rotDeg || 0);
  const src = (planSource || "top").toLowerCase();

  FACE_KEYS.forEach((k) => {
    const img = faceImg[k];
    if (img) img.style.transform = "";
  });

  const target = faceImg[src];
  if (target) target.style.transform = `rotate(${rot}deg)`;
}

function setPreviewImages(runId, views) {
  const bust = Date.now();
  FACE_KEYS.forEach((k) => {
    const img = faceImg[k];
    if (!img) return;

    const baseUrl = resolveRunUrl(runId, views?.[k]);
    if (!baseUrl) return;

    const sep = baseUrl.includes("?") ? "&" : "?";
    img.src = `${baseUrl}${sep}v=${bust}`;
  });
}

function renderPreflight(preflight, orientation) {
  currentPreflight = preflight;
  currentOrientation = orientation || { plan_source: "top", rotation_deg: 0 };

  const views = preflight?.preview_views || {};
  setPreviewImages(currentRunId, views);

  const any = FACE_KEYS.some((k) => Boolean(views?.[k]));
  if (previewWrap) previewWrap.style.display = any ? "flex" : "none";

  const src = (currentOrientation?.plan_source || "top").toLowerCase();
  const rot = Number(currentOrientation?.rotation_deg || 0);
  if (rotSel) rotSel.value = String(rot);

  highlightPlanSource(src);
  applyRotationCss(src, rot);

  if (preflightInfoEl) preflightInfoEl.textContent = "";
}

// ------------------------------
// API calls
// ------------------------------
async function createRun() {
  const res = await fetch("/api/create_run", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const j = await res.json();
  if (!j?.run_id) throw new Error("create_run returned no run_id");
  return j.run_id;
}

async function fetchState(runId) {
  const res = await fetch(`/api/state/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function postOrientation(runId, plan_source, rotation_deg) {
  const res = await fetch(`/api/orientation/${encodeURIComponent(runId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_source, rotation_deg }),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function fetchTreeGrouped(runId) {
  const res = await fetch(`/api/tree_grouped/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function fetchBom(runId) {
  const res = await fetch(`/api/bom/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function fetchBomFlat(runId) {
  const res = await fetch(`/api/bom_flat/${encodeURIComponent(runId)}`);
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`bom_flat fetch failed: ${res.status} ${txt}`);
  }
  return await res.json();
}
async function fetchExplodePlan(runId) {
  const res = await fetch(`/api/explode_plan/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function postExplodePlan(runId, payload) {
  const res = await fetch(`/api/explode_plan/${encodeURIComponent(runId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

// ------------------------------
// progress + polling
// ------------------------------
async function refreshFromState(runId) {
  const st = await fetchState(runId);

  if (st?.status?.stage === "error") {
    setWarning(st?.status?.error || "Run failed.");
  }

  if (st?.preflight?.preview_views) {
    if (currentRunId === runId) {
      renderPreflight(st.preflight, st.orientation || currentOrientation);
      setStatus("Preview ready. Loading…");
      stopStatePolling();

      // show tree/bom section
      if (treeSectionEl) treeSectionEl.style.display = "block";

      // load current view
      await loadViewForRun(currentRunId, currentView);
    }
    return true;
  }
  return false;
}

function startStatePolling(runId) {
  stopStatePolling();
  statePollTimer = setInterval(async () => {
    try {
      await refreshFromState(runId);
    } catch (_) {}
  }, 1500);
}

function startProgress(runId) {
  stopProgress();

  const es = new EventSource(`/api/progress/${encodeURIComponent(runId)}`);
  progressES = es;

  es.onmessage = async (ev) => {
    if (!ev?.data) return;
    appendProgress(ev.data);

    const s = String(ev.data);
    if (
      s.includes("Preflight: done") ||
      s.includes("Preflight: done.") ||
      s.includes("preflight_pack") ||
      s.includes("Preview ready")
    ) {
      try {
        await refreshFromState(runId);
      } catch (_) {}
    }
  };

  es.onerror = () => {
    // ignore
  };
}

// ------------------------------
// orientation controls
// ------------------------------
async function saveOrientation(planSource) {
  if (!currentRunId) return;
  const src = String(planSource || "top").toLowerCase();
  const rot = Number(rotSel?.value || 0);

  await postOrientation(currentRunId, src, rot);

  const st = await fetchState(currentRunId);
  renderPreflight(st.preflight || currentPreflight, st.orientation || currentOrientation);
  setStatus("Orientation saved.");
}

function wireFaceClicks() {
  FACE_KEYS.forEach((k) => {
    const img = faceImg[k];
    if (img) img.onclick = () => saveOrientation(k);
  });

  if (btnTop) btnTop.onclick = () => saveOrientation("top");
  if (btnBottom) btnBottom.onclick = () => saveOrientation("bottom");
  if (btnFront) btnFront.onclick = () => saveOrientation("front");
  if (btnBack) btnBack.onclick = () => saveOrientation("back");
  if (btnLeft) btnLeft.onclick = () => saveOrientation("left");
  if (btnRight) btnRight.onclick = () => saveOrientation("right");
}

// ------------------------------
// view switching
// ------------------------------
function setViewUi(view) {
  currentView = view;

  const isBom = (view === "bom" || view === "bom_flat");

  if (treeTitleEl) {
    treeTitleEl.textContent =
      view === "bom_flat" ? "Purchasing BOM" :
      view === "bom" ? "BOM (Global)" :
      "Assembly (Grouped)";
  }

  if (treeHintEl) {
    treeHintEl.textContent =
      view === "bom"
        ? "Tick Explode where needed. Click a row to load STL."
        : isBom
          ? "Purchasing view. Click a row to load STL."
          : "Click a row to load STL";
  }

  const bomSearch = document.getElementById("bomSearch");
  if (bomSearch) bomSearch.style.display = isBom ? "block" : "none";

  if (viewAssemblyBtn) viewAssemblyBtn.disabled = view === "assembly";
  if (viewBomBtn) viewBomBtn.disabled = isBom; // disable BOM (Global) button if any BOM view is active

  if (btnBomViewer) btnBomViewer.disabled = view === "bom";
  if (btnBomPurchasing) btnBomPurchasing.disabled = view === "bom_flat";
}



function wireViewButtons() {
  if (viewAssemblyBtn) {
    viewAssemblyBtn.onclick = async () => {
      if (!currentRunId) return;
      setViewUi("assembly");
      await loadViewForRun(currentRunId, "assembly");
    };
  }
  if (viewBomBtn) {
    viewBomBtn.onclick = async () => {
      if (!currentRunId) return;
      setViewUi("bom");
      await loadViewForRun(currentRunId, "bom");
    };
  }
}

function wireBomModeButtons() {
  if (btnBomViewer) {
    btnBomViewer.onclick = async () => {
      if (!currentRunId) return;
      setViewUi("bom");
      await loadViewForRun(currentRunId, "bom");
    };
  }

  if (btnBomPurchasing) {
    btnBomPurchasing.onclick = async () => {
      if (!currentRunId) return;
      setViewUi("bom_flat");
      await loadViewForRun(currentRunId, "bom_flat");
    };
  }
}


async function loadViewForRun(runId, view) {
  // always refresh explode plan (cheap + keeps checkboxes accurate)
  try {
    explodePlan = await fetchExplodePlan(runId);
  } catch (_) {
    explodePlan = { schema: "explode_plan.v1", run_id: runId, items: {} };
  }

  if (view === "bom") {
    stopTreePolling();
    await tryLoadBom(runId);
    return;
  }

  if (view === "bom_flat") {
    stopTreePolling();
    await tryLoadBomFlat(runId);
    return;
  }

  // assembly
  stopBomPolling();
  await tryLoadTree(runId);
}


// ------------------------------
// explode plan helpers
// ------------------------------
function isMarked(defSig) {
  const items = explodePlan?.items;
  if (!defSig || typeof defSig !== "string") return false;
  return Boolean(items && typeof items === "object" && items[defSig]);
}

async function toggleMark(runId, defSig, defName, solidCount, checked) {
  const action = checked ? "mark" : "unmark";
  const payload = {
    def_sig: defSig,
    action,
    def_name: defName || null,
    solid_count: typeof solidCount === "number" ? solidCount : null,
    note: "",
  };
  explodePlan = await postExplodePlan(runId, payload);
}

// ------------------------------
// Three.js viewer
// ------------------------------
async function loadViewerModules() {
  if (_THREE && _OrbitControls && _STLLoader) return;
  const THREE = await import("/ui/vendor/three.module.js");
  const oc = await import("/ui/vendor/OrbitControls.js");
  const stl = await import("/ui/vendor/STLLoader.js");
  _THREE = THREE;
  _OrbitControls = oc;
  _STLLoader = stl;
}

function initViewerOnce() {
  if (_viewerInit) return;
  if (!viewerEl) return;

  const THREE = _THREE;

  _scene = new THREE.Scene();
  _camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
  _camera.position.set(200, 200, 200);

  _renderer = new THREE.WebGLRenderer({ antialias: true });
  _renderer.setPixelRatio(window.devicePixelRatio || 1);
  viewerEl.innerHTML = "";
  viewerEl.appendChild(_renderer.domElement);

  _controls = new _OrbitControls.OrbitControls(_camera, _renderer.domElement);
  _controls.enableDamping = true;

  _scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dl = new THREE.DirectionalLight(0xffffff, 0.6);
  dl.position.set(1, 2, 3);
  _scene.add(dl);

  function applySize() {
    const w = Math.max(1, viewerEl.clientWidth || 800);
    const h = Math.max(1, viewerEl.clientHeight || 600);
    _camera.aspect = w / h;
    _camera.updateProjectionMatrix();
    _renderer.setSize(w, h, false);
  }

  // Window resize still helps, but panel resize is the real fix
  window.addEventListener("resize", applySize);

  // Observe viewer element size changes (flexbox, split panes, etc.)
  const ro = new ResizeObserver(() => {
    applySize();
    // Optional: if something is loaded, refit after resize so it stays framed
    if (_currentMesh) fitToObject(_currentMesh);                                  //remove if jittery
  });
  ro.observe(viewerEl);

  applySize();


  (function animate() {
    requestAnimationFrame(animate);
    _controls.update();
    _renderer.render(_scene, _camera);
  })();

  _viewerInit = true;
}

function clearMesh() {
  if (!_currentMesh || !_scene) return;
  _scene.remove(_currentMesh);
  try { _currentMesh.geometry.dispose(); } catch (_) {}
  try { _currentMesh.material.dispose(); } catch (_) {}
  _currentMesh = null;
}

function fitToObject(obj, offset = 1.25) {
  const THREE = _THREE;
  if (!_camera || !_controls || !obj) return;

  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z) || 1;

  // Keep current view direction
  const viewDir = new THREE.Vector3()
    .subVectors(_camera.position, _controls.target)
    .normalize();

  // Fit using FOV
  const fov = (_camera.fov * Math.PI) / 180.0;
  let distance = (maxDim * 0.5) / Math.tan(fov * 0.5);
  distance *= offset;

  _controls.target.copy(center);
  _camera.position.copy(center).add(viewDir.multiplyScalar(distance));

  _camera.near = Math.max(distance / 100.0, 0.1);
  _camera.far = distance * 100.0;
  _camera.updateProjectionMatrix();

  _controls.maxDistance = distance * 20.0;
  _controls.minDistance = distance / 20.0;
  _controls.update();
}


async function loadSTL(url) {
  await loadViewerModules();
  initViewerOnce();
  clearMesh();

  if (!url) return;

  const THREE = _THREE;
  const loader = new _STLLoader.STLLoader();

  const bust = Date.now();
  const sep = url.includes("?") ? "&" : "?";
  const u = `${url}${sep}v=${bust}`;

  return new Promise((resolve, reject) => {
    loader.load(
      u,
      (geom) => {
        const mat = new THREE.MeshStandardMaterial();
        const mesh = new THREE.Mesh(geom, mat);

        // If your STL is already in the right orientation, remove this line.
        mesh.rotation.set(-Math.PI / 2, 0, 0);

        _scene.add(mesh);
        _currentMesh = mesh;
        fitToObject(mesh);
        resolve();
      },
      undefined,
      (err) => reject(err)
    );
  });
}


// ------------------------------
// Assembly (Grouped) renderer
// ------------------------------
function nodeRecord(occId) {
  return treeData?.nodes?.[occId] || null;
}

function renderNodeMeta(obj) {
  if (!nodeMetaEl) return;
  nodeMetaEl.textContent = JSON.stringify(obj || {}, null, 2);
}

function makeNodeRow(occId) {
  const n = nodeRecord(occId);
  const li = document.createElement("li");
  li.style.margin = "2px 0";

  const hasKids = Array.isArray(n?.children) && n.children.length > 0;

  const btnExp = document.createElement("button");
  btnExp.textContent = hasKids ? "▸" : " ";
  btnExp.style.width = "26px";
  btnExp.style.border = "none";
  btnExp.style.background = "none";
  btnExp.style.cursor = hasKids ? "pointer" : "default";

  const btnNode = document.createElement("button");
  btnNode.className = "nodebtn";
  btnNode.textContent = n?.display_name || occId;

  const sub = document.createElement("div");
  sub.style.marginLeft = "26px";
  sub.style.display = "none";

  let expanded = false;

  btnExp.addEventListener("click", () => {
    if (!hasKids) return;
    expanded = !expanded;
    btnExp.textContent = expanded ? "▾" : "▸";
    sub.style.display = expanded ? "block" : "none";

    if (expanded && sub.childNodes.length === 0) {
      const ul = document.createElement("ul");
      ul.style.listStyle = "none";
      ul.style.margin = "0";
      ul.style.paddingLeft = "16px";
      for (const kid of n.children) ul.appendChild(makeNodeRow(kid));
      sub.appendChild(ul);
    }
  });

  btnNode.addEventListener("click", async () => {
    document.querySelectorAll(".nodebtn.selected").forEach((x) => x.classList.remove("selected"));
    btnNode.classList.add("selected");

    renderNodeMeta({ occ_id: occId, ...n });

    const url = n?.stl_url || null;
    if (!url) {
      setStatus("No STL for this node.");
      clearMesh();
      return;
    }

    const stlAbs = resolveRunUrl(currentRunId, url);
    if (!stlAbs) {
      setStatus("Bad STL URL for this node.");
      clearMesh();
      return;
    }

    try {
      setStatus("Loading STL…");
      await loadSTL(stlAbs);
      setStatus("");
    } catch (e) {
      console.error(e);
      setStatus("Failed to load STL.");
      clearMesh();
    }
  });

  li.appendChild(btnExp);
  li.appendChild(btnNode);
  li.appendChild(sub);
  return li;
}



function renderTreeUI() {
  if (!treeEl) return;
  treeEl.innerHTML = "";

  const ul = document.createElement("ul");
  ul.style.listStyle = "none";
  ul.style.margin = "0";
  ul.style.paddingLeft = "0";

  const roots = Array.isArray(treeData?.roots) ? treeData.roots : [];
  for (const r of roots) ul.appendChild(makeNodeRow(r));

  treeEl.appendChild(ul);
  renderNodeMeta({});
  if (treeSectionEl) treeSectionEl.style.display = "block";
}

async function tryLoadTree(runId) {
  stopTreePolling();

  const attempt = async () => {
    try {
      const t = await fetchTreeGrouped(runId);
      if (t?.nodes && Object.keys(t.nodes).length > 0) {
        treeData = t;
        renderTreeUI();
        setStatus("Assembly loaded.");
        stopTreePolling();
        return true;
      }
    } catch (_) {}
    return false;
  };

  const ok = await attempt();
  if (ok) return;

  setStatus("Waiting for assembly tree…");
  treePollTimer = setInterval(attempt, 1500);
}

// ------------------------------
// BOM renderer (Global)
// ------------------------------
function renderBomUI() {
  if (!treeEl) return;
  treeEl.innerHTML = "";

  const items = Array.isArray(bomData?.items) ? bomData.items : [];

  // filter
  const q = String(document.getElementById("bomSearch")?.value || "").trim().toLowerCase();
  const filtered = q
    ? items.filter((it) => String(it.part_number || it.def_name || "").toLowerCase().includes(q))
    : items;

  const table = document.createElement("table");
  table.className = "bom";

  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th title="Explode decision">Explode</th>
      <th>Part Number</th>
      <th class="cell-right">Qty</th>
      <th>Parent</th>
      <th class="cell-mono">BBox (mm)</th>
      <th>Category</th>
      <th class="cell-right">Solids</th>
      <th class="cell-link">STL</th>
      <th class="cell-link">DWG</th>
      <th class="cell-link">DXF</th>
      <th class="cell-link">NC</th>
      <th class="cell-link">DRILL</th>
      <th class="cell-link">PNG</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  for (const it of filtered) {
    const partNo = String(it.part_number || it.def_name || "—");
    const qty = typeof it.qty_total === "number" ? it.qty_total : null;
    const parent = String(it.from_parent_def_name || "");
    const category = String(it.category || "");
    const solidCount = typeof it.solid_count === "number" ? it.solid_count : null;

    const defSig =
      (typeof it.ref_def_sig === "string" && it.ref_def_sig) ||
      (typeof it.def_sig_used === "string" && it.def_sig_used) ||
      (typeof it.def_sig === "string" && it.def_sig) ||
      "";

    const tr = document.createElement("tr");

    // explode checkbox
    const tdExplode = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.title = "Mark for explosion (split solids)";
    cb.disabled = !(solidCount != null && solidCount > 1) || !defSig;
    cb.checked = defSig ? isMarked(defSig) : false;

    // don't trigger row click when toggling checkbox
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.onchange = async () => {
      try {
        await toggleMark(currentRunId, defSig, partNo, solidCount, cb.checked);
        setStatus(cb.checked ? "Marked for explosion." : "Unmarked.");
        // keep node meta accurate if selected
        renderNodeMeta({ ...it, explode_marked: defSig ? isMarked(defSig) : false });
      } catch (e) {
        cb.checked = !cb.checked;
        setWarning(e?.message || String(e));
      }
    };
    tdExplode.appendChild(cb);

    const tdPart = document.createElement("td");
    tdPart.textContent = partNo;

    const tdQty = document.createElement("td");
    tdQty.className = "cell-right";
    tdQty.textContent = qty != null ? String(qty) : "";

    const tdParent = document.createElement("td");
    tdParent.className = "cell-muted";
    tdParent.textContent = parent;

    const tdBbox = document.createElement("td");
    tdBbox.className = "cell-mono";
    tdBbox.textContent = fmtBbox(it);

    const tdCat = document.createElement("td");
    tdCat.className = "cell-muted";
    tdCat.textContent = category;

    const tdSol = document.createElement("td");
    tdSol.className = "cell-right cell-muted";
    tdSol.textContent = solidCount != null ? String(solidCount) : "";

    const stl = firstNonEmpty(it, ["stl_url", "stl_path", "stl"]);
    const dwg = firstNonEmpty(it, ["dwg_url", "dwg_path", "dwg"]);
    const dxf = firstNonEmpty(it, ["dxf_url", "dxf_path", "dxf"]);
    const nc  = firstNonEmpty(it, ["nc_url", "nc1_url", "nc1_path", "nc_path", "nc"]);
    const drl = firstNonEmpty(it, ["drilling_url", "drill_url", "drilling_path", "drill_path", "html_url"]);
    const png = firstNonEmpty(it, ["png_url", "thumb_url", "thumbnail_url", "png_path", "thumb_path"]);


    tr.appendChild(tdExplode);
    tr.appendChild(tdPart);
    tr.appendChild(tdQty);
    tr.appendChild(tdParent);
    tr.appendChild(tdBbox);
    tr.appendChild(tdCat);
    tr.appendChild(tdSol);
    tr.appendChild(makeLinkCell("STL", stl, "Open STL"));
    tr.appendChild(makeLinkCell("DWG", dwg, "Open DWG"));
    tr.appendChild(makeLinkCell("DXF", dxf, "Open DXF"));
    tr.appendChild(makeLinkCell("NC",  nc,  "Open NC1"));
    tr.appendChild(makeLinkCell("DRILL", drl, "Open drilling drawing"));
    tr.appendChild(makeLinkCell("PNG", png, "Open thumbnail image"));

    tr.onclick = async () => {
      // selection highlight
      treeEl.querySelectorAll("tr.selected").forEach((x) => x.classList.remove("selected"));
      tr.classList.add("selected");

      renderNodeMeta({ ...it, explode_marked: defSig ? isMarked(defSig) : false });

      const url = it.stl_url || null;
      if (!url) {
        setStatus("No STL for this item.");
        clearMesh();
        return;
      }

      const stlAbs = resolveRunUrl(currentRunId, url);
      if (!stlAbs) {
        setWarning("Bad STL URL for this item.");
        return;
      }

      try {
        setStatus("Loading STL…");
        await loadSTL(stlAbs);
        setStatus("");
      } catch (e) {
        console.error(e);
        setWarning("Failed to load STL.");
      }
    };

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  treeEl.appendChild(table);

  renderNodeMeta({});
  if (treeSectionEl) treeSectionEl.style.display = "block";
}

function fmtBboxSorted(it) {
  const s = it?.bbox_sorted_mm;
  if (!Array.isArray(s) || s.length < 3) return "";
  const a = Number(s[0]), b = Number(s[1]), c = Number(s[2]);
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return "";
  return `${a.toFixed(1)}×${b.toFixed(1)}×${c.toFixed(1)}`;
}

function renderBomFlatUI() {
  if (!treeEl) return;
  treeEl.innerHTML = "";

  const items = Array.isArray(bomData?.items) ? bomData.items : [];

  // filter
  const q = String(document.getElementById("bomSearch")?.value || "").trim().toLowerCase();
  const filtered = q
    ? items.filter((it) => String(it.example_def_name || it.common_part_id || "").toLowerCase().includes(q))
    : items;

  const table = document.createElement("table");
  table.className = "bom";

  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>Part</th>
      <th class="cell-right">Qty</th>
      <th>Category</th>
      <th class="cell-mono">BBox (mm)</th>
      <th class="cell-mono">Group</th>
      <th class="cell-right">Members</th>
      <th class="cell-link">STL</th>
      <th class="cell-link">DWG</th>
      <th class="cell-link">DXF</th>
      <th class="cell-link">NC</th>
      <th class="cell-link">DRILL</th>
      <th class="cell-link">PNG</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  for (const it of filtered) {
    const tr = document.createElement("tr");

    const tdPart = document.createElement("td");
    tdPart.textContent = String(it.example_def_name || "—");

    const tdQty = document.createElement("td");
    tdQty.className = "cell-right";
    tdQty.textContent = (typeof it.qty === "number") ? String(it.qty) : "";

    const tdCat = document.createElement("td");
    tdCat.className = "cell-muted";
    tdCat.textContent = String(it.category || "");

    const tdBbox = document.createElement("td");
    tdBbox.className = "cell-mono";
    tdBbox.textContent = fmtBboxSorted(it);

    const tdGroup = document.createElement("td");
    tdGroup.className = "cell-mono cell-muted";
    tdGroup.textContent = String(it.common_part_id || "");

    const tdMembers = document.createElement("td");
    tdMembers.className = "cell-right cell-muted";
    tdMembers.textContent = (typeof it.members_count === "number") ? String(it.members_count) : "";

    // representative STL + future link columns (same pattern as global BOM)
    const stl = firstNonEmpty(it, ["representative_stl_url", "stl_url", "stl_path", "stl"]);
    const dwg = firstNonEmpty(it, ["dwg_url", "dwg_path", "dwg"]);
    const dxf = firstNonEmpty(it, ["dxf_url", "dxf_path", "dxf"]);
    const nc  = firstNonEmpty(it, ["nc_url", "nc1_url", "nc1_path", "nc_path", "nc"]);
    const drl = firstNonEmpty(it, ["drilling_url", "drill_url", "drilling_path", "drill_path", "html_url"]);
    const png = firstNonEmpty(it, ["png_url", "thumb_url", "thumbnail_url", "png_path", "thumb_path"]);

    tr.appendChild(tdPart);
    tr.appendChild(tdQty);
    tr.appendChild(tdCat);
    tr.appendChild(tdBbox);
    tr.appendChild(tdGroup);
    tr.appendChild(tdMembers);

    tr.appendChild(makeLinkCell("STL", stl, "Open representative STL"));
    tr.appendChild(makeLinkCell("DWG", dwg, "Open DWG"));
    tr.appendChild(makeLinkCell("DXF", dxf, "Open DXF"));
    tr.appendChild(makeLinkCell("NC",  nc,  "Open NC1"));
    tr.appendChild(makeLinkCell("DRILL", drl, "Open drilling drawing"));
    tr.appendChild(makeLinkCell("PNG", png, "Open thumbnail image"));

    tr.onclick = async () => {
      treeEl.querySelectorAll("tr.selected").forEach((x) => x.classList.remove("selected"));
      tr.classList.add("selected");

      renderNodeMeta(it);

      const url = it.representative_stl_url || null;
      if (!url) {
        setStatus("No STL for this item.");
        clearMesh();
        return;
      }

      const stlAbs = resolveRunUrl(currentRunId, url);
      if (!stlAbs) {
        setWarning("Bad STL URL for this item.");
        return;
      }

      try {
        setStatus("Loading STL…");
        await loadSTL(stlAbs);
        setStatus("");
      } catch (e) {
        console.error(e);
        setWarning("Failed to load STL.");
      }
    };

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  treeEl.appendChild(table);

  renderNodeMeta({});
  if (treeSectionEl) treeSectionEl.style.display = "block";
}


async function tryLoadBom(runId) {
  stopBomPolling();

  const attempt = async () => {
    try {
      const b = await fetchBom(runId);
      if (Array.isArray(b?.items)) {
        bomData = b;
        renderBomUI();
        setStatus("BOM loaded.");
        stopBomPolling();
        return true;
      }
    } catch (_) {}
    return false;
  };

  const ok = await attempt();
  if (ok) return;

  setStatus("Waiting for BOM…");
  bomPollTimer = setInterval(attempt, 1500);
}

// ------------------------------
// wiring + boot
// ------------------------------
wireFaceClicks();
wireViewButtons();
wireBomModeButtons();

if (fileInp) {
  fileInp.addEventListener("change", () => {
    clearPreviewUI();
    showProgress(false);
    setStatus("File selected. Click Upload + Preview.");
  });
}

if (btnGo) {
  btnGo.onclick = async () => {
    if (!fileInp?.files || fileInp.files.length === 0) {
      setStatus("Pick a STEP file first.");
      return;
    }

    stopProgress();
    stopStatePolling();
    stopTreePolling();
    stopBomPolling();

    clearPreviewUI();
    setWarning("");
    showProgress(true);
    setStatus("Creating run…");

    try {
      currentRunId = await createRun();
      localStorage.setItem("last_run_id", currentRunId); 

      startProgress(currentRunId);
      startStatePolling(currentRunId);

      const fd = new FormData();
      fd.append("file", fileInp.files[0]);

      setStatus("Uploading…");
      const res = await fetch(`/api/preview/${encodeURIComponent(currentRunId)}`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const txt = await res.text();
        setWarning(`Preview request returned ${res.status}. Worker may still finish.\n${txt}`);
        setStatus("Working… (watch progress)");
        return;
      }

      setStatus("Queued. Working… (watch progress)");
    } catch (e) {
      setWarning(e?.message || String(e));
      setStatus("Working… (watch progress)");
      if (currentRunId) startStatePolling(currentRunId);
    }
  };
}

if (btnApplyOrient) {
  btnApplyOrient.onclick = async () => {
    const src = (currentOrientation?.plan_source || "top").toLowerCase();
    await saveOrientation(src);
  };
}

if (rotSel) {
  rotSel.onchange = () => {
    const src = (currentOrientation?.plan_source || "top").toLowerCase();
    const rot = Number(rotSel?.value || 0);
    applyRotationCss(src, rot);
  };
}

const bomSearchEl = document.getElementById("bomSearch");
if (bomSearchEl) {
  bomSearchEl.addEventListener("input", () => {
    if (currentView === "bom") renderBomUI();
    if (currentView === "bom_flat") renderBomFlatUI();

  });
}

// Boot: restore last run
(async function boot() {
  const last = localStorage.getItem("last_run_id");
  if (!last) return;

  try {
    const st = await fetchState(last);
    currentRunId = st.run_id || last;

    showProgress(true);
    startProgress(currentRunId);

    if (st.preflight?.preview_views) {
      renderPreflight(st.preflight, st.orientation);
      setStatus("Restored last run. Loading…");
      if (treeSectionEl) treeSectionEl.style.display = "block";

      setViewUi(currentView);
      await loadViewForRun(currentRunId, currentView);
    } else {
      setStatus("Restoring… waiting for thumbnails…");
      startStatePolling(currentRunId);
    }
  } catch (_) {
    // ignore
  }
})();
