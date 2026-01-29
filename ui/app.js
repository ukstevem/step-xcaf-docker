const statusEl = document.getElementById("status");
const warningEl = document.getElementById("warning");
const progressEl = document.getElementById("progress");
const preflightInfoEl = document.getElementById("preflightInfo");
const selTopEl = document.getElementById("selTop");

const btnGo = document.getElementById("go");
const fileInp = document.getElementById("file");

const previewWrap = document.getElementById("previewViews");

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

let currentRunId = null;
let currentPreflight = null;
let currentOrientation = { plan_source: "top", rotation_deg: 0 };

// runtime helpers
let progressES = null;
let statePollTimer = null;
let lastRenderedRunId = null;

const FACE_KEYS = ["top", "bottom", "front", "back", "left", "right"];

const faceImg = {
  top: imgTop,
  bottom: imgBottom,
  front: imgFront,
  back: imgBack,
  left: imgLeft,
  right: imgRight,
};

function setStatus(s) {
  if (statusEl) statusEl.textContent = s || "";
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

function showProgress(show) {
  if (!progressEl) return;
  progressEl.style.display = show ? "block" : "none";
  if (show) progressEl.textContent = "";
}

function tsLocal() {
  const d = new Date();
  // HH:MM:SS local time
  return d.toLocaleTimeString([], { hour12: false });
}

function appendProgress(raw) {
  if (!progressEl) return;

  // SSE may deliver multiple lines in one message; split and append each line
  const incoming = String(raw)
    .split(/\r?\n/g)
    .map((s) => s.trim())
    .filter(Boolean);

  if (incoming.length === 0) return;

  const existing = progressEl.textContent
    ? progressEl.textContent.split("\n").filter(Boolean)
    : [];

  for (const line of incoming) {
    existing.push(`[${tsLocal()}] ${line}`);
  }

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
  if (statePollTimer) {
    clearInterval(statePollTimer);
    statePollTimer = null;
  }
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

function renderInfo(preflight) {
  // You said bbox is not important here; keep this minimal.
  if (!preflightInfoEl) return;
  preflightInfoEl.textContent = "";
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
  // Cache-bust so big-run images don't show the previous run without refresh.
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
  renderInfo(preflight);
}

async function createRun() {
  const res = await fetch("/api/create_run", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const j = await res.json();
  if (!j?.run_id) throw new Error("create_run returned no run_id");
  return j.run_id;
}

async function fetchState(runId) {
  const res = await fetch(`/api/state/${runId}`, { cache: "no-store" });
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

async function refreshFromState(runId) {
  const st = await fetchState(runId);

  // If server returns error stage, show it
  if (st?.status?.stage === "error") {
    setWarning(st?.status?.error || "Run failed.");
  }

  if (st?.preflight?.preview_views) {
    // Only re-render if we are on the same run
    if (currentRunId === runId) {
      renderPreflight(st.preflight, st.orientation || currentOrientation);
      setStatus("Preview ready. Choose TOP source + rotation.");
      lastRenderedRunId = runId;
      stopStatePolling(); // we’re done
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
    } catch (_) {
      // ignore transient fetch errors
    }
  }, 1500);
}

function startProgress(runId) {
  stopProgress();

  const es = new EventSource(`/api/progress/${runId}`);
  progressES = es;

  es.onmessage = async (ev) => {
    if (!ev?.data) return;

    // Important: split into lines + timestamp inside appendProgress
    appendProgress(ev.data);

    // If worker tells us preflight is done, refresh state immediately.
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
    // SSE may drop on long jobs; polling will still pick it up.
    // Don't spam warnings here.
  };

  return es;
}

async function saveOrientation(planSource) {
  if (!currentRunId) return;

  const src = String(planSource || "top").toLowerCase();
  const rot = Number(rotSel?.value || 0);

  await postOrientation(currentRunId, src, rot);

  // Refresh state and re-render (ensures stored orientation matches)
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

wireFaceClicks();

// Clear old run visuals when selecting a new file
if (fileInp) {
  fileInp.addEventListener("change", () => {
    clearPreviewUI();
    showProgress(false);
    setStatus("File selected. Click Upload + Preflight.");
  });
}

btnGo.onclick = async () => {
  if (!fileInp?.files || fileInp.files.length === 0) {
    setStatus("Pick a STEP file first.");
    return;
  }

  // Stop any prior run watchers
  stopProgress();
  stopStatePolling();

  clearPreviewUI();
  setWarning("");
  showProgress(true);
  setStatus("Creating run…");

  try {
    currentRunId = await createRun();
    localStorage.setItem("last_run_id", currentRunId);

    // Start SSE + Polling immediately
    startProgress(currentRunId);
    startStatePolling(currentRunId);

    const fd = new FormData();
    fd.append("file", fileInp.files[0]);

    setStatus("Uploading…");
    const res = await fetch(`/api/preview/${currentRunId}`, { method: "POST", body: fd });

    // IMPORTANT: do not “fail” the UI just because this request times out / errors on large models.
    // The worker may still be running and polling will pick up the preflight_pack when it appears.
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

// Save rotation for whichever source is currently selected
btnApplyOrient.onclick = async () => {
  const src = (currentOrientation?.plan_source || "top").toLowerCase();
  await saveOrientation(src);
};

// Rotate visually immediately (no persistence until Save)
rotSel.onchange = () => {
  const src = (currentOrientation?.plan_source || "top").toLowerCase();
  const rot = Number(rotSel?.value || 0);
  applyRotationCss(src, rot);
};

// Boot: restore last run (and poll if needed)
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
      setStatus("Restored last run.");
    } else {
      setStatus("Restoring… waiting for thumbnails…");
      startStatePolling(currentRunId);
    }
  } catch (_) {
    // ignore
  }
})();
