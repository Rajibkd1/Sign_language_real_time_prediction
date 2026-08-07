import { AlertTriangle, Play, RotateCcw, ShieldCheck, Square } from 'lucide-react';
import { useCallback, useEffect, useReducer, useRef } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { useCamera } from '../hooks/useCamera.js';
import { inferVideo } from '../lib/api.js';
import CameraStage from '../features/recognition/CameraStage.jsx';
import DiagnosticsPanel from '../features/recognition/DiagnosticsPanel.jsx';
import ProcessingPanel from '../features/recognition/ProcessingPanel.jsx';
import ResultPanel from '../features/recognition/ResultPanel.jsx';
import {
  canStart,
  canStop,
  initialRecognitionState,
  isBusy,
  recognitionReducer,
  recognitionStates
} from '../features/recognition/recognitionMachine.js';

const MAX_SECONDS = 384 / 25;
const WARN_SECONDS = 12;
const MIN_SECONDS = 0.65;

function bestMimeType() {
  if (!window.MediaRecorder) return '';
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

export default function RecognitionPage() {
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const startedAtRef = useRef(0);
  const autoStopRef = useRef(false);
  const [state, dispatch] = useReducer(recognitionReducer, initialRecognitionState);
  const camera = useCamera();

  useEffect(() => {
    if (videoRef.current && camera.stream) {
      videoRef.current.srcObject = camera.stream;
    }
  }, [camera.stream]);

  useEffect(() => {
    if (state.status !== recognitionStates.RECORDING) return undefined;
    const timer = window.setInterval(() => {
      const elapsedSeconds = (Date.now() - startedAtRef.current) / 1000;
      dispatch({ type: 'TICK', elapsedSeconds });
      if (elapsedSeconds >= MAX_SECONDS && !autoStopRef.current) {
        autoStopRef.current = true;
        stopRecording();
      }
    }, 120);
    return () => window.clearInterval(timer);
  }, [state.status]);

  const enableCamera = useCallback(async () => {
    dispatch({ type: 'REQUEST_CAMERA' });
    try {
      await camera.requestCamera();
      dispatch({ type: 'CAMERA_READY' });
    } catch (error) {
      dispatch({ type: 'CAMERA_ERROR', error: error.message });
    }
  }, [camera]);

  const handleRecordingComplete = useCallback(
    async (blob, durationSeconds) => {
      if (durationSeconds < MIN_SECONDS) {
        dispatch({
          type: 'ERROR',
          error: 'The recording was too short. Perform one complete sentence and try again.'
        });
        return;
      }

      try {
        dispatch({ type: 'UPLOAD' });
        const result = await inferVideo(blob, {
          onPhase: (phase) => {
            dispatch({ type: phase === 'uploading' ? 'UPLOAD' : 'PROCESS', phase });
          }
        });
        dispatch({ type: 'SUCCESS', result });
      } catch (error) {
        dispatch({
          type: 'ERROR',
          error: error.message || 'Recognition service is currently unavailable.'
        });
      }
    },
    []
  );

  const startRecording = useCallback(async () => {
    if (!camera.stream || !window.MediaRecorder) return;
    chunksRef.current = [];
    autoStopRef.current = false;
    const mimeType = bestMimeType();
    const recorder = new MediaRecorder(camera.stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data?.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = () => {
      const durationSeconds = (Date.now() - startedAtRef.current) / 1000;
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' });
      handleRecordingComplete(blob, durationSeconds);
    };

    startedAtRef.current = Date.now();
    dispatch({ type: 'START_RECORDING', startedAt: startedAtRef.current });
    recorder.start(250);
  }, [camera.stream, handleRecordingComplete]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;
    dispatch({ type: 'STOPPING' });
    recorder.stop();
  }, []);

  const clear = () => {
    dispatch({ type: 'CLEAR', keepCamera: Boolean(camera.stream) });
  };

  const switchCamera = async (deviceId) => {
    try {
      await camera.switchCamera(deviceId);
      dispatch({ type: 'CAMERA_READY' });
    } catch (error) {
      dispatch({ type: 'CAMERA_ERROR', error: error.message });
    }
  };

  const busy = isBusy(state);
  const startEnabled = canStart(state, Boolean(camera.stream)) && !busy;
  const stopEnabled = canStop(state);
  const showProcessing = [recognitionStates.UPLOADING, recognitionStates.PROCESSING, recognitionStates.STOPPING].includes(
    state.status
  );

  return (
    <main className="page">
      <PageHeader eyebrow="Continuous Sentence Recognition" title="Bangla Sign Language Sentence Recognition">
        Start, sign one complete sentence, stop, then Model 6 decodes the Bangla sentence with word-level CTC.
      </PageHeader>

      <div className="recognition-layout">
        <CameraStage
          videoRef={videoRef}
          stream={camera.stream}
          devices={camera.devices}
          selectedDeviceId={camera.selectedDeviceId}
          onEnableCamera={enableCamera}
          onSwitchCamera={switchCamera}
          state={state}
          maxSeconds={MAX_SECONDS}
          warnSeconds={WARN_SECONDS}
        />

        <aside className="stack" aria-label="Recognition controls and result">
          <section className="controls-panel">
            <div>
              <p className="eyebrow">Recognition</p>
              <h2>Record Sentence</h2>
              <p className="subtle">Maximum supported clip length is {MAX_SECONDS.toFixed(2)} seconds.</p>
            </div>
            <div className="controls-row">
              <button
                type="button"
                className="button primary"
                onClick={startRecording}
                disabled={!startEnabled}
              >
                <Play size={18} aria-hidden="true" />
                Start Recognition
              </button>
              <button
                type="button"
                className="button danger"
                onClick={stopRecording}
                disabled={!stopEnabled}
              >
                <Square size={18} aria-hidden="true" />
                Stop & Predict
              </button>
              <button type="button" className="button ghost" onClick={clear} disabled={state.status === recognitionStates.RECORDING}>
                <RotateCcw size={18} aria-hidden="true" />
                Clear
              </button>
            </div>

            {state.status === recognitionStates.REQUESTING_CAMERA ? (
              <StatusBadge tone="loading">Requesting camera</StatusBadge>
            ) : null}
            {state.status === recognitionStates.STOPPING ? (
              <StatusBadge tone="loading">Preparing recording</StatusBadge>
            ) : null}
          </section>

          {showProcessing ? <ProcessingPanel phase={state.phase} /> : null}

          {state.status === recognitionStates.ERROR ? (
            <div className="error-box" role="alert">
              <h3>
                <AlertTriangle size={18} aria-hidden="true" /> Recognition unavailable
              </h3>
              <p>{state.error || camera.error}</p>
            </div>
          ) : null}

          <ResultPanel result={state.result} onTryAgain={clear} />
          <DiagnosticsPanel result={state.result} />

          <div className="privacy-note">
            <ShieldCheck size={20} aria-hidden="true" />
            <span>
              Camera preview stays in the browser. The completed recording is uploaded only after Stop,
              processed for inference, then deleted from temporary server storage. Raw video is not saved by default.
            </span>
          </div>
        </aside>
      </div>
    </main>
  );
}

