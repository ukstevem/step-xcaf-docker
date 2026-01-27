const canvas = document.getElementById("viewer");
const statusEl = document.getElementById("status");
const topEl = document.getElementById("toplevel");
const progressEl = document.getElementById("progress");
const warningEl = document.getElementById("warning");

const previewWrap = document.getElementById("previewViews");
const imgPlan = document.getElementById("pv_plan");
const imgFront = document.getElementById("pv_front");
const imgSide = document.getElementById("pv_side");
const imgIso = document.getElementById("pv_iso");

const btnGo = document.getElementById("go");
const btnAnalyze = document.getElementById("analyze");

const btnPlanAsPlan = document.getElementById("btnPlanAsPlan");
const btnFrontAsPlan = document.getElementById("btnFrontAsPlan");
const btnSideAsPlan = document.getElementById("btnSideAsPlan");

const rotSel = document.getElementById("rotSel");
const btnApplyOrient = document.getElementById("applyOrient");

let currentRunId = null;
let currentPreflight = null;

// 3D viewer (optional)
let THREE = null;
let OrbitControls = null;
let STLLoader = null;
let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let loader = null;
let modelMesh = null;
let edgeLines = null;

// Guardrail: avoid locking the tab
const MAX_TRIANGLES = 2_500_000;

// ----------------------------
// UI helpers
// ----------------------------
function setStatus(s) {
  if (statusEl) statusEl.textContent = s;
}

function setWarning(msg) {
  if (!warningEl) return;
  if (!msg) {
    warningEl.style.display = "none";
    warningEl.textContent = "";
    return;
  }
  warningEl.style.display = "block";
  warningEl.textContent = msg;
}

function showProgressBox(show) {
  if (!progressEl) return;
  progressEl.style.display = show ? "block" : "none";
  if (show) progressEl.textContent = "";
}

function appendProgressLine(line) {
  if (!progressEl) return;
  const lines = progressEl.textContent.split("\n").filter(Boolean);
  lines.push(line);
  const trimmed = lines.slice(-120);
  progressEl.textContent = trimmed.join("\n");
  progressEl.scrollTop = progressEl.scrollHeight;
}

function startProgress(runId) {
  const es = new EventSource(`/api/progress/${runId}`);
  es.onmessage = (ev) => {
    if (ev && ev.data) {
      appendProgressLine(ev.data);
      setStatus(ev.data);
    }
  };
  es.onerror = () => {
    // EventSource auto-retries
  };
  return es;
}

function fmtMm(n) {
  if (typeof n !== "number" || !isFinite(n)) return "-";
  return `${n.toFixed(1)} mm`;
}

function fmtMb(n) {
  if (typeof n !== "number" || !isFinite(n)) return "-";
  return `${n.toFixed(1)} MB`;
}

// ----------------------------
// Preflight rendering (4 PNG views + info)
// ----------------------------
function computeDisplayMapping(planSource) {
  // Which underlying image key is shown in each slot
  // Keep it simple and deterministic.
  if (planSource === "front") {
    return { plan: "front", front: "plan", side: "side", iso: "iso" };
  }
  if (planSource === "side") {
    return { plan: "side", front: "front", side: "plan", iso: "iso" };
  }
  return { plan: "plan", front: "front", side: "side", iso: "iso" };
}

function applyOrientationToUI(orientation) {
  const planSource = (orientation?.plan_source || "plan");
  const rot = Number(orientation?.rotation_deg || 0);

  if (rotSel) rotSel.value = String(rot);

  // Button highlight
  const on = (btn, yes) => {
    if (!btn) return;
    btn.style.borderColor = yes ? "#2f6fed" : "#bbb";
    btn.style.boxShadow = yes ? "0 0 0 2px rgba(47,111,237,0.15)" : "none";
  };
  on(btnPlanAsPlan, planSource === "plan");
  on(btnFrontAsPlan, planSource === "front");
  on(btnSideAsPlan, planSource === "side");

  // Image border highlight (PLAN slot)
  if (imgPlan) {
    imgPlan.style.borderColor = "#2f6fed";
    imgPlan.style.boxShadow = "0 0 0 2px rgba(47,111,237,0.15)";
  }

  // Apply rotation to PLAN image only (cheap + reliable)
  if (imgPlan) {
    imgPlan.style.transformOrigin = "center center";
    imgPlan.style.transform = `rotate(${rot}deg)`;
  }
}

function renderPreflight(preflight, orientation) {
  currentPreflight = preflight || null;

  if (!previewWrap) return;
  if (!preflight || !preflight.preview_views) {
    previewWrap.style.display = "none";
    return;
  }

  previewWrap.style.display = "flex";

  const pv = preflight.preview_views || {};
  const planSource = (orientation?.plan_source || "plan");
  const map = computeDisplayMapping(planSource);

  if (imgPlan) imgPlan.src = pv[map.plan] || "";
  if (imgFront) imgFront.src = pv[map.front] || "";
  if (imgSide) imgSide.src = pv[map.side] || "";
  if (imgIso) imgIso.src = pv[map.iso] || "";

  applyOrientationToUI(orientation);

  // Preflight info in the warning bar (handy + always visible)
  const bbox = preflight.bbox_mm?.size;
  const solids = preflight.counts?.solids;
  const faces = preflight.counts?.faces;
  const stlMb = preflight.preflight_mesh?.stl_mb;

  let info = [];
  if (bbox && bbox.length === 3) {
    info.push(`BBox: ${fmtMm(bbox[0])} × ${fmtMm(bbox[1])} × ${fmtMm(bbox[2])}`);
  }
  if (typeof solids === "number") info.push(`Solids: ${solids}`);
  if (typeof faces === "number") info.push(`Faces: ${faces}`);
  if (typeof stlMb === "number") info.push(`Preview STL: ${fmtMb(stlMb)}`);

  setWarning(info.join("   |   "));
}

function renderPreflightInfo(preflight) {
  const el = document.getElementById("preflightInfo");
  if (!el) return;

  if (!preflight) {
    el.textContent = "";
    return;
  }

  const bbox = preflight.bbox_mm;
  const counts = preflight.counts || {};

  const bboxStr = bbox
    ? `BBOX (mm): X=${bbox.x.toFixed(1)} Y=${bbox.y.toFixed(1)} Z=${bbox.z.toFixed(1)}`
    : "BBOX (mm): (not reported)";

  const countsStr = `Defs=${counts.definitions ?? "?"}, Occs=${counts.occurrences ?? "?"}, Leaf=${counts.leaf_occurrences ?? "?"}, FreeShapes=${counts.free_shapes ?? "?"}`;

  el.textContent = `${bboxStr} | ${countsStr}`;
}


// ----------------------------
// Top-level pills
// ----------------------------
function renderTopLevel(list) {
  if (!topEl) return;
  topEl.innerHTML = "";

  if (!list || list.length === 0) {
    const d = document.createElement("div");
    d.textContent = "No top-level items found.";
    d.style.color = "#666";
    topEl.appendChild(d);
    return;
  }

  for (const item of list) {
    const pill = document.createElement("div");
    pill.className = "pill";
    pill.textContent = item.name;
    pill.title = item.ref ? `ref: ${item.ref}` : item.id;
    pill.onclick = () => setStatus(`Selected: ${item.name} (${item.id})`);
    topEl.appendChild(pill);
  }
}

// ----------------------------
// 3D viewer (optional) – dynamic import so UI still works if vendor files are missing
// ----------------------------
async function init3D() {
  try {
    THREE = await import("/ui/vendor/three.module.js");
    ({ OrbitControls } = await import("/ui/vendor/OrbitControls.js"));
    ({ STLLoader } = await import("/ui/vendor/STLLoader.js"));
  } catch (e) {
    THREE = null;
    setStatus("3D viewer unavailable. Showing 2D previews instead.");
    if (canvas) canvas.style.display = "none";
    return false;
  }

  // Setup scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf6f7f9);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x888888, 0.9);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(1, 1.2, 0.8);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(-1, 0.4, -0.6);
  scene.add(fill);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.physicallyCorrectLights = true;

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1e7);
  camera.position.set(300, 200, 300);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  loader = new STLLoader();

  function onResize() {
    const w = window.innerWidth;
    const h = canvas.clientHeight || (window.innerHeight - 220);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", onResize);
  onResize();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return true;
}

function fitToObject(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  const dist = maxDim * 1.6 + 1;

  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(dist, dist * 0.6, dist));
  camera.near = Math.max(0.1, maxDim / 1000);
  camera.far = Math.max(1000, maxDim * 20);
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadStl(url) {
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (geom) => resolve(geom),
      (xhr) => {
        try {
          if (xhr && xhr.total) {
            const pct = Math.round((xhr.loaded / xhr.total) * 100);
            setStatus(`Downloading STL… ${pct}%`);
          } else if (xhr && typeof xhr.loaded === "number") {
            const mb = Math.round(xhr.loaded / (1024 * 1024));
            setStatus(`Downloading STL… ${mb} MB`);
          }
        } catch (_) {}
      },
      (err) => reject(err)
    );
  });
}

function clearModel() {
  if (modelMesh) {
    scene.remove(modelMesh);
    if (modelMesh.geometry) modelMesh.geometry.dispose();
    if (modelMesh.material) modelMesh.material.dispose();
    modelMesh = null;
  }
  if (edgeLines) {
    scene.remove(edgeLines);
    if (edgeLines.geometry) edgeLines.geometry.dispose();
    if (edgeLines.material) edgeLines.material.dispose();
    edgeLines = null;
  }
}

async function showAssembly(url) {
  if (!THREE) return; // no 3D
  if (!url) {
    setStatus("No 3D model preview available (STL URL not returned).");
    return;
  }

  setStatus("Loading STL…");
  clearModel();

  let geom;
  try {
    geom = await loadStl(url);
  } catch (e) {
    setStatus("Failed to load STL.");
    return;
  }

  const pos = geom?.attributes?.position;
  const triCount = pos ? Math.floor(pos.count / 3) : 0;
  if (triCount && triCount > MAX_TRIANGLES) {
    setStatus(`STL too heavy to render safely (${triCount.toLocaleString()} triangles).`);
    return;
  }

  try { geom.computeVertexNormals(); } catch (_) {}

  const mat = new THREE.MeshStandardMaterial({
    color: 0x9aa3ad,
    metalness: 0.05,
    roughness: 0.65,
    flatShading: false,
  });

  modelMesh = new THREE.Mesh(geom, mat);
  scene.add(modelMesh);

  if (triCount && triCount < 1_000_000) {
    const edges = new THREE.EdgesGeometry(geom, 20);
    edgeLines = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x2b2f33, transparent: true, opacity: 0.25 })
    );
    scene.add(edgeLines);
  }

  if (canvas) canvas.style.display = "block";
  fitToObject(modelMesh);
  setStatus("3D loaded.");
}

// ----------------------------
// API helpers
// ----------------------------
async function createRun() {
  const res = await fetch("/api/create_run", { method: "POST" });
  if (!res.ok) throw new Error(`create_run failed: ${await res.text()}`);
  const data = await res.json();
  if (!data?.run_id) throw new Error("create_run returned no run_id");
  return data.run_id;
}

async function fetchState(runId) {
  const res = await fetch(`/api/state/${runId}`);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function postOrientation(runId, plan_source, rotation_deg) {
  const res = await fetch(`/api/orientation/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_source, rotation_deg }),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

// ----------------------------
// Orientation actions
// ----------------------------
async function setPlanSource(planSource) {
  if (!currentRunId) return;
  const rot = Number(rotSel?.value || 0);

  try {
    await postOrientation(currentRunId, planSource, rot);
    const state = await fetchState(currentRunId);
    renderPreflight(state.preflight, state.orientation);
    setStatus("Plan selection saved.");
  } catch (e) {
    setStatus("Failed to save plan selection.");
  }
}

async function applyRotation() {
  if (!currentRunId) return;
  const rot = Number(rotSel?.value || 0);

  // Keep current plan source
  let planSource = "plan";
  try {
    const state = await fetchState(currentRunId);
    planSource = state?.orientation?.plan_source || "plan";
  } catch (_) {}

  try {
    await postOrientation(currentRunId, planSource, rot);
    const state = await fetchState(currentRunId);
    renderPreflight(state.preflight, state.orientation);
    setStatus("Rotation saved.");
  } catch (e) {
    setStatus("Failed to save rotation.");
  }
}

// ----------------------------
// Main buttons
// ----------------------------
btnGo.onclick = async () => {
  const inp = document.getElementById("file");
  if (!inp?.files || inp.files.length === 0) {
    setStatus("Pick a STEP file first.");
    return;
  }

  setWarning("");
  showProgressBox(true);
  if (topEl) topEl.innerHTML = "";
  setStatus("Creating run…");

  let es = null;

  try {
    currentRunId = await createRun();
    localStorage.setItem("last_run_id", currentRunId);

    es = startProgress(currentRunId);

    const fd = new FormData();
    fd.append("file", inp.files[0]);

    setStatus("Uploading + previewing…");
    const res = await fetch(`/api/preview/${currentRunId}`, { method: "POST", body: fd });
    if (!res.ok) {
      const txt = await res.text();
      if (es) es.close();
      setStatus("Failed: " + txt);
      return;
    }

    const data = await res.json();
    if (es) es.close();

    renderPreflight(data.preflight, data.orientation);
    setStatus("Preview ready. Choose plan + rotation, then run analysis.");
    if (btnAnalyze) btnAnalyze.disabled = false;

  } catch (e) {
    if (es) es.close();
    setStatus("Failed: " + (e?.message || String(e)));
  }
};

btnAnalyze.onclick = async () => {
  if (!currentRunId) {
    setStatus("No run_id. Use Upload + Preview first.");
    return;
  }

  showProgressBox(true);
  setStatus("Running Step 2 (XCAF + tree)…");

  let es = null;
  try {
    es = startProgress(currentRunId);

    // Step 2 endpoint (fast XCAF + occurrence tree; no full-model STL)
    const res = await fetch(`/api/step2/${currentRunId}`, { method: "POST" });
    if (!res.ok) {
      const txt = await res.text();
      if (es) es.close();
      setStatus("Failed: " + txt);
      return;
    }

    // IMPORTANT: parse JSON BEFORE touching `data`
    const data = await res.json();
    if (es) es.close();

    renderTopLevel(data.top_level);
    renderPreflightInfo(data.preflight);

    // No STL expected for large assemblies in Step 2
    await showAssembly(data.assembly_stl_url);

    if (data.occurrence_tree_url) {
      setWarning(`Step 2 ready. Tree: ${data.occurrence_tree_url}`);
    }

    setStatus(`Step 2 done. Top-level from: ${data.meta_source || "step2"}`);

  } catch (e) {
    if (es) es.close();
    setStatus("Failed: " + (e?.message || String(e)));
  }
};


// Plan selection buttons
if (btnPlanAsPlan) btnPlanAsPlan.onclick = () => setPlanSource("plan");
if (btnFrontAsPlan) btnFrontAsPlan.onclick = () => setPlanSource("front");
if (btnSideAsPlan) btnSideAsPlan.onclick = () => setPlanSource("side");
if (btnApplyOrient) btnApplyOrient.onclick = () => applyRotation();

// ----------------------------
// On load: restore last run state + init 3D if possible
// ----------------------------
(async function boot() {
  await init3D();

  const last = localStorage.getItem("last_run_id");
  if (!last) return;

  try {
    const state = await fetchState(last);
    currentRunId = state.run_id;
    if (btnAnalyze) btnAnalyze.disabled = false;

    if (state.preflight && state.preflight.preview_views) {
      renderPreflight(state.preflight, state.orientation);
      setStatus("Restored last run preview.");
    }
    if (state.analysis && state.analysis.top_level) {
      renderTopLevel(state.analysis.top_level);
      await showAssembly(state.analysis.assembly_stl_url);
      setStatus("Restored last run analysis.");
    }
  } catch (_) {
    // ignore
  }
})();

