import { useCallback, useRef } from 'react';

const DEFAULT_FRAME_WIDTH = 560;
const DEFAULT_JPEG_QUALITY = 0.82;
const DEFAULT_MAX_FRAMES = 40;
const DEFAULT_SAMPLE_MS = 80;

/**
 * Ring buffer that avoids array allocations on every frame capture.
 */
export class FrameRingBuffer {
  constructor(capacity) {
    this._buffer = new Array(capacity);
    this._capacity = capacity;
    this._head = 0;
    this._size = 0;
  }

  push(item) {
    this._buffer[this._head] = item;
    this._head = (this._head + 1) % this._capacity;
    if (this._size < this._capacity) this._size += 1;
  }

  drain(maxCount) {
    const count = Math.min(maxCount, this._size);
    if (count === 0) return [];

    const start = (this._head - this._size + this._capacity) % this._capacity;
    const ordered = new Array(this._size);

    for (let i = 0; i < this._size; i++) {
      ordered[i] = this._buffer[(start + i) % this._capacity];
    }

    const result = ordered.slice(0, count);
    const remaining = ordered.slice(count);
    this.clear();
    for (const item of remaining) {
      this.push(item);
    }

    return result;
  }

  drainLatest(maxCount) {
    const count = Math.min(maxCount, this._size);
    if (count === 0) return [];

    const start = (this._head - this._size + this._capacity) % this._capacity;
    const ordered = new Array(this._size);

    for (let i = 0; i < this._size; i++) {
      ordered[i] = this._buffer[(start + i) % this._capacity];
    }

    const result = ordered.slice(this._size - count);
    this.clear();
    return result;
  }

  get size() {
    return this._size;
  }

  clear() {
    this._size = 0;
    this._head = 0;
  }
}

/**
 * Converts a canvas to a base64 JPEG data URL.
 * Uses toBlob + FileReader for non-blocking encoding when available,
 * falls back to synchronous toDataURL.
 */
function canvasToDataURL(canvas, quality) {
  return new Promise((resolve) => {
    if (typeof canvas.toBlob === 'function') {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            resolve(canvas.toDataURL('image/jpeg', quality));
            return;
          }
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        },
        'image/jpeg',
        quality
      );
    } else {
      resolve(canvas.toDataURL('image/jpeg', quality));
    }
  });
}

/**
 * Hook that manages webcam frame capture with a ring buffer,
 * non-blocking JPEG encoding, and requestAnimationFrame gating.
 */
export function useFrameCapture({
  videoRef,
  canvasRef,
  frameWidth = DEFAULT_FRAME_WIDTH,
  jpegQuality = DEFAULT_JPEG_QUALITY,
  maxFrames = DEFAULT_MAX_FRAMES,
  sampleMs = DEFAULT_SAMPLE_MS,
} = {}) {
  const bufferRef = useRef(new FrameRingBuffer(maxFrames));
  const startTimeRef = useRef(0);
  const timerRef = useRef(null);
  const activeRef = useRef(false);
  const pendingCaptureRef = useRef(false);

  const captureFrame = useCallback(async () => {
    if (pendingCaptureRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

    const width = Math.min(frameWidth, video.videoWidth);
    const height = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * width));

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.drawImage(video, 0, 0, width, height);

    const timestampMs = Date.now() - startTimeRef.current;

    pendingCaptureRef.current = true;
    try {
      const image = await canvasToDataURL(canvas, jpegQuality);
      if (activeRef.current) {
        bufferRef.current.push({ image, timestampMs });
      }
    } finally {
      pendingCaptureRef.current = false;
    }
  }, [videoRef, canvasRef, frameWidth, jpegQuality]);

  const startCapture = useCallback(() => {
    stopCapture();
    bufferRef.current = new FrameRingBuffer(maxFrames);
    startTimeRef.current = Date.now();
    activeRef.current = true;

    timerRef.current = window.setInterval(() => {
      // Skip capture when the tab is backgrounded
      if (document.hidden) return;
      captureFrame();
    }, sampleMs);
  }, [captureFrame, maxFrames, sampleMs]);

  const stopCapture = useCallback(() => {
    activeRef.current = false;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const drainFrames = useCallback((maxCount) => {
    return bufferRef.current.drainLatest(maxCount);
  }, []);

  const clearFrames = useCallback(() => {
    bufferRef.current.clear();
  }, []);

  const getFrameCount = useCallback(() => {
    return bufferRef.current.size;
  }, []);

  const getStartTime = useCallback(() => {
    return startTimeRef.current;
  }, []);

  return {
    startCapture,
    stopCapture,
    drainFrames,
    clearFrames,
    getFrameCount,
    getStartTime,
  };
}
