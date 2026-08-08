import { useCallback, useEffect, useRef, useReducer } from 'react';
import { inferLive } from '../lib/api.js';
import { chooseLiveDraft, hasEnoughLiveSignal, isSoftLiveError, wordCount } from '../features/recognition/liveDraft.js';

// --- Constants ---
const SEND_BATCH_MS = 280;
const PREDICT_MS = 850;
const ENABLE_LIVE_DRAFTS = true;
const CLOCK_TICK_MS = 250;
const MIN_BATCH_FRAMES = 1;
const MAX_BATCH_SEND_FRAMES = 8;
const REQUEST_TIMEOUT_MS = 15000;
const MAX_CONSECUTIVE_FAILURES = 3;
const BACKOFF_BASE_MS = 500;
const BACKOFF_MAX_MS = 4000;

// --- State ---
const initialState = {
  backendState: 'idle',
  isAnalyzing: false,
  elapsedSeconds: 0,
  draftSentence: '',
  finalSentence: '',
  latestResult: null,
  updates: 0,
  bufferedFrames: 0,
  motionScore: 0,
  captureFps: 0,
  detections: {
    pose_rate: 0,
    face_rate: 0,
    left_hand_rate: 0,
    right_hand_rate: 0,
  },
  latestLatencyMs: null,
  hint: 'Allow camera access once. Sentences will be detected automatically.',
  error: '',
};

function reducer(state, action) {
  switch (action.type) {
    case 'TICK':
      return { ...state, elapsedSeconds: action.elapsed };

    case 'SET_ANALYZING':
      return { ...state, isAnalyzing: action.value };

    case 'LIVE_RESPONSE': {
      const { backendState, bufferedFrames, motionScore, captureFps, detections, latencyMs } = action;
      return {
        ...state,
        backendState,
        bufferedFrames,
        motionScore,
        captureFps,
        detections,
        latestLatencyMs: latencyMs,
        isAnalyzing: false,
      };
    }

    case 'DRAFT_UPDATE': {
      const { result, draftSentence, hint } = action;
      return {
        ...state,
        latestResult: result,
        updates: state.updates + 1,
        draftSentence,
        hint,
      };
    }

    case 'FINAL_UPDATE': {
      const { result, hint } = action;
      return {
        ...state,
        latestResult: result,
        updates: state.updates + 1,
        finalSentence: result.prediction?.trim() || '',
        draftSentence: '',
        hint,
      };
    }

    case 'SET_HINT':
      return { ...state, hint: action.hint };

    case 'SET_ERROR':
      return { ...state, error: action.error, hint: action.hint || state.hint };

    case 'CLEAR_ERROR':
      return { ...state, error: '' };

    case 'NEW_SEGMENT':
      return { ...state, draftSentence: '', finalSentence: '' };

    case 'CLEAR_DRAFT':
      return { ...state, draftSentence: '' };

    case 'RESET':
      return { ...initialState, hint: action.hint || initialState.hint };

    default:
      return state;
  }
}

// --- Helper ---
function makeFramePayload(frame) {
  return {
    image: frame.image,
    timestamp_ms: Math.max(0, Math.round(frame.timestampMs)),
  };
}

function isPublishableDraft(currentSentence, result, candidateHits) {
  const prediction = String(result?.prediction || '').trim();
  if (!prediction || !hasEnoughLiveSignal(result)) return false;

  const confidence = result?.confidence?.value || 0;
  const currentWords = wordCount(currentSentence);
  const nextWords = wordCount(prediction);

  if (candidateHits >= 2) return true;
  if (confidence >= 0.36) return true;
  if (nextWords > currentWords && confidence >= 0.22) return true;
  return !currentSentence && confidence >= 0.26;
}

/**
 * Hook managing the live inference loop: batching, abort, backoff, state.
 */
export function useLiveInference({ drainFrames, clearFrames, cameraReady }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const sessionIdRef = useRef('');
  const startedAtRef = useRef(0);
  const lastPredictAtRef = useRef(0);
  const inFlightRef = useRef(false);
  const candidateRef = useRef({ prediction: '', hits: 0 });
  const sendTimerRef = useRef(null);
  const clockTimerRef = useRef(null);
  const streamGenRef = useRef(0);
  const abortRef = useRef(null);
  const activeSegmentRef = useRef(0);
  const lastFinalSegmentRef = useRef(0);
  const consecutiveFailuresRef = useRef(0);
  const backoffMsRef = useRef(0);

  // Keep latest state accessible in callbacks without re-creating them
  const stateRef = useRef(state);
  stateRef.current = state;

  const invalidateTransport = useCallback(() => {
    streamGenRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    inFlightRef.current = false;
    clearFrames();
    dispatch({ type: 'SET_ANALYZING', value: false });
  }, [clearFrames]);

  const resetOutput = useCallback(() => {
    invalidateTransport();
    sessionIdRef.current = '';
    activeSegmentRef.current = 0;
    lastFinalSegmentRef.current = 0;
    lastPredictAtRef.current = 0;
    candidateRef.current = { prediction: '', hits: 0 };
    consecutiveFailuresRef.current = 0;
    backoffMsRef.current = 0;
    dispatch({ type: 'RESET', hint: 'Watching for signing...' });
  }, [invalidateTransport]);

  const publishDraft = useCallback((result) => {
    const prediction = String(result?.prediction || '').trim();
    if (!prediction) return;

    if (candidateRef.current.prediction === prediction) {
      candidateRef.current = { prediction, hits: candidateRef.current.hits + 1 };
    } else {
      candidateRef.current = { prediction, hits: 1 };
    }

    const current = stateRef.current.draftSentence;
    if (isPublishableDraft(current, result, candidateRef.current.hits)) {
      const newDraft = chooseLiveDraft(current, result);
      dispatch({ type: 'DRAFT_UPDATE', result, draftSentence: newDraft, hint: 'Recognizing live signing...' });
    } else {
      dispatch({ type: 'DRAFT_UPDATE', result, draftSentence: current, hint: 'Recognizing live signing...' });
    }
  }, []);

  const publishFinal = useCallback((result) => {
    const prediction = String(result?.prediction || '').trim();
    if (!prediction) return;
    candidateRef.current = { prediction: '', hits: 0 };
    dispatch({ type: 'FINAL_UPDATE', result, hint: 'Final sentence ready. Watching for the next sign...' });
  }, []);

  const runBatch = useCallback(async () => {
    if (inFlightRef.current) return;

    // Backoff: skip this tick if we're in backoff window
    if (backoffMsRef.current > 0) {
      backoffMsRef.current = Math.max(0, backoffMsRef.current - SEND_BATCH_MS);
      return;
    }

    const frames = drainFrames(MAX_BATCH_SEND_FRAMES);
    if (frames.length < MIN_BATCH_FRAMES) {
      return;
    }

    const shouldPredict = ENABLE_LIVE_DRAFTS && Date.now() - lastPredictAtRef.current >= PREDICT_MS;
    const gen = streamGenRef.current;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    inFlightRef.current = true;
    abortRef.current = controller;
    dispatch({ type: 'SET_ANALYZING', value: shouldPredict });
    dispatch({ type: 'CLEAR_ERROR' });

    try {
      const t0 = performance.now();
      const payload = await inferLive({
        sessionId: sessionIdRef.current,
        frames: frames.map(makeFramePayload),
        reset: !sessionIdRef.current,
        predict: shouldPredict,
        saveHistory: true,
        signal: controller.signal,
      });
      const latencyMs = Math.round(performance.now() - t0);

      if (gen !== streamGenRef.current || controller.signal.aborted) return;

      // Success — reset failure tracking
      consecutiveFailuresRef.current = 0;
      backoffMsRef.current = 0;

      const segmentId = Number(payload.segment_id || 0);
      const payloadSessionId = payload.session_id || '';
      if (sessionIdRef.current && payloadSessionId && payloadSessionId !== sessionIdRef.current) return;
      sessionIdRef.current = payload.session_id;

      if (payload.state === 'signing' && segmentId > 0 && segmentId !== activeSegmentRef.current) {
        activeSegmentRef.current = segmentId;
        candidateRef.current = { prediction: '', hits: 0 };
        dispatch({ type: 'NEW_SEGMENT' });
      }

      dispatch({
        type: 'LIVE_RESPONSE',
        backendState: payload.state || 'idle',
        bufferedFrames: payload.buffered_frames || 0,
        motionScore: payload.motion_score || 0,
        captureFps: payload.capture_fps || 0,
        detections: payload.detections || initialState.detections,
        latencyMs,
      });

      if (shouldPredict || payload.is_final) {
        lastPredictAtRef.current = Date.now();
      }

      if (payload.result) {
        if (payload.is_final) {
          if (segmentId > 0 && segmentId <= lastFinalSegmentRef.current) {
            clearFrames();
            dispatch({ type: 'SET_HINT', hint: 'Watching for signing...' });
            return;
          }
          if (segmentId > 0) {
            lastFinalSegmentRef.current = segmentId;
            activeSegmentRef.current = segmentId;
          }
          clearFrames();
          publishFinal(payload.result);
        } else if (payload.state === 'signing') {
          if (segmentId > 0 && segmentId <= lastFinalSegmentRef.current) return;
          publishDraft(payload.result);
        } else {
          dispatch({ type: 'SET_HINT', hint: 'Watching for signing...' });
        }
      } else if (payload.state === 'signing') {
        dispatch({ type: 'SET_HINT', hint: 'Sign motion detected. Building sentence buffer...' });
      } else if (payload.state === 'finalizing') {
        dispatch({ type: 'SET_HINT', hint: 'Sentence ended. Running Model 6 final decode...' });
      } else if (payload.state === 'result' && stateRef.current.finalSentence) {
        dispatch({ type: 'SET_HINT', hint: 'Final sentence ready. Watching for the next sign...' });
      } else {
        dispatch({ type: 'SET_HINT', hint: 'Watching for signing...' });
        if (payload.state === 'idle' && !stateRef.current.draftSentence) {
          dispatch({ type: 'CLEAR_DRAFT' });
        }
      }
    } catch (err) {
      if (err?.name === 'AbortError') return;

      consecutiveFailuresRef.current += 1;

      if (isSoftLiveError(err)) {
        dispatch({ type: 'SET_HINT', hint: 'Waiting for clearer hand and body landmarks...' });
      } else if (consecutiveFailuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        // Exponential backoff
        backoffMsRef.current = Math.min(
          BACKOFF_BASE_MS * Math.pow(2, consecutiveFailuresRef.current - MAX_CONSECUTIVE_FAILURES),
          BACKOFF_MAX_MS
        );
        dispatch({
          type: 'SET_ERROR',
          error: 'Connection unstable. Retrying...',
          hint: 'Reconnecting to recognition service...',
        });
      } else {
        dispatch({
          type: 'SET_ERROR',
          error: err.message || 'Recognition service is currently unavailable.',
          hint: 'Live update failed.',
        });
      }
    } finally {
      clearTimeout(timeoutId);
      if (abortRef.current === controller) {
        abortRef.current = null;
        inFlightRef.current = false;
        dispatch({ type: 'SET_ANALYZING', value: false });
      }
    }
  }, [drainFrames, clearFrames, publishDraft, publishFinal]);

  // Keep runBatch reference fresh
  const runBatchRef = useRef(runBatch);
  runBatchRef.current = runBatch;

  const startSession = useCallback((startTime) => {
    resetOutput();
    startedAtRef.current = startTime || Date.now();
    lastPredictAtRef.current = 0;

    sendTimerRef.current = window.setInterval(() => {
      runBatchRef.current();
    }, SEND_BATCH_MS);

    clockTimerRef.current = window.setInterval(() => {
      dispatch({ type: 'TICK', elapsed: (Date.now() - startedAtRef.current) / 1000 });
    }, CLOCK_TICK_MS);
  }, [resetOutput]);

  const stopSession = useCallback(() => {
    window.clearInterval(sendTimerRef.current);
    window.clearInterval(clockTimerRef.current);
    sendTimerRef.current = null;
    clockTimerRef.current = null;
    invalidateTransport();
    sessionIdRef.current = '';
    activeSegmentRef.current = 0;
    lastFinalSegmentRef.current = 0;
    consecutiveFailuresRef.current = 0;
    backoffMsRef.current = 0;
  }, [invalidateTransport]);

  const clearOutput = useCallback(() => {
    resetOutput();
    dispatch({ type: 'SET_HINT', hint: cameraReady ? 'Watching for signing...' : 'Camera is idle.' });
  }, [resetOutput, cameraReady]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      window.clearInterval(sendTimerRef.current);
      window.clearInterval(clockTimerRef.current);
      abortRef.current?.abort();
    };
  }, []);

  return {
    state,
    dispatch,
    startSession,
    stopSession,
    clearOutput,
    resetOutput,
    invalidateTransport,
  };
}
