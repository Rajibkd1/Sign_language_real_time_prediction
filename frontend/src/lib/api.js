const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

function endpoint(path) {
  return `${API_BASE}${path}`;
}

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

export async function getHealth() {
  const response = await fetch(endpoint('/api/health'));
  return parseResponse(response);
}

export async function getModelInfo() {
  const response = await fetch(endpoint('/api/model/info'));
  return parseResponse(response);
}

export async function inferVideo(blob, { onPhase } = {}) {
  const formData = new FormData();
  const filename = blob.type.includes('mp4') ? 'bdsl-recording.mp4' : 'bdsl-recording.webm';
  formData.append('file', blob, filename);

  onPhase?.('uploading');
  const request = fetch(endpoint('/api/inference/video'), {
    method: 'POST',
    body: formData
  });
  onPhase?.('processing');
  return parseResponse(await request);
}

export async function getHistory() {
  const response = await fetch(endpoint('/api/history'));
  return parseResponse(response);
}

export async function clearHistory() {
  const response = await fetch(endpoint('/api/history'), { method: 'DELETE' });
  if (response.status === 204) return null;
  return parseResponse(response);
}

export async function deleteHistoryItem(id) {
  const response = await fetch(endpoint(`/api/history/${encodeURIComponent(id)}`), { method: 'DELETE' });
  if (response.status === 204) return null;
  return parseResponse(response);
}

