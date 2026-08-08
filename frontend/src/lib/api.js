const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

function endpoint(path) {
  return `${API_BASE}${path}`;
}

// --- Response cache for static endpoints ---
const _cache = new Map();
const CACHE_TTL_MS = 60_000;

function getCached(key) {
  const entry = _cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL_MS) {
    _cache.delete(key);
    return null;
  }
  return entry.data;
}

function setCache(key, data) {
  _cache.set(key, { data, ts: Date.now() });
}

// --- Response parsing ---
async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  const isJson = contentType.includes('application/json');
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = payload?.detail || payload;
    const message =
      detail?.message ||
      detail?.detail ||
      payload?.message ||
      'Recognition service is currently unavailable.';
    const error = new Error(message);
    error.status = response.status;
    error.code = detail?.error || payload?.error || 'request_failed';
    error.detail = detail;
    throw error;
  }

  return payload;
}

// --- Default timeout signal ---
function timeoutSignal(ms = 8000) {
  if (typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

function mergeSignals(signal, timeoutMs = 8000) {
  if (!signal) return { signal: timeoutSignal(timeoutMs) };
  // If caller provides a signal, don't add extra timeout (they manage it)
  return { signal };
}

// --- API Functions ---

export async function getHealth() {
  const cached = getCached('health');
  if (cached) return cached;

  const response = await fetch(endpoint('/api/health'), mergeSignals());
  const data = await parseResponse(response);
  setCache('health', data);
  return data;
}

export async function getModelInfo() {
  const cached = getCached('modelInfo');
  if (cached) return cached;

  const response = await fetch(endpoint('/api/model/info'), mergeSignals());
  const data = await parseResponse(response);
  setCache('modelInfo', data);
  return data;
}

export async function inferVideo(video, { onPhase, saveHistory = true } = {}) {
  const formData = new FormData();
  const fallbackFilename = video.type?.includes('mp4') ? 'bdsl-video.mp4' : 'bdsl-video.webm';
  formData.append('file', video, video.name || fallbackFilename);
  formData.append('save_history', String(saveHistory));

  onPhase?.('uploading');
  const request = fetch(endpoint('/api/inference/video'), {
    method: 'POST',
    body: formData,
  });
  onPhase?.('processing');
  return parseResponse(await request);
}

export async function inferFrames(frames, { saveHistory = false } = {}) {
  const response = await fetch(endpoint('/api/inference/frames'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      frames,
      save_history: saveHistory,
    }),
    ...mergeSignals(),
  });
  return parseResponse(response);
}

export async function inferLive({ sessionId, frames, reset = false, predict = true, saveHistory = false, signal }) {
  const response = await fetch(endpoint('/api/inference/live'), {
    method: 'POST',
    ...mergeSignals(signal),
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId || null,
      frames,
      reset,
      predict,
      save_history: saveHistory,
    }),
  });
  return parseResponse(response);
}

export async function getHistory() {
  const response = await fetch(endpoint('/api/history'), mergeSignals());
  return parseResponse(response);
}

export async function clearHistory() {
  const response = await fetch(endpoint('/api/history'), { method: 'DELETE', ...mergeSignals() });
  if (response.status === 204) return null;
  return parseResponse(response);
}

export async function deleteHistoryItem(id) {
  const response = await fetch(endpoint(`/api/history/${encodeURIComponent(id)}`), { method: 'DELETE', ...mergeSignals() });
  if (response.status === 204) return null;
  return parseResponse(response);
}
