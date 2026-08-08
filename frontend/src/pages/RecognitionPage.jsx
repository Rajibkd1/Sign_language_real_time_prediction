import { AlertTriangle, Copy, RotateCcw, ShieldCheck, Sparkles } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import AnimatedPage from '../components/AnimatedPage.jsx';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { showToast } from '../components/ToastHost.jsx';
import TypewriterText from '../components/TypewriterText.jsx';
import LiveCameraStage from '../features/recognition/LiveCameraStage.jsx';
import { useCamera } from '../hooks/useCamera.js';
import { useFrameCapture } from '../hooks/useFrameCapture.js';
import { useLiveInference } from '../hooks/useLiveInference.js';

function phaseDetails({ cameraReady, backendState, isAnalyzing, hasFinal }) {
  if (!cameraReady) {
    return { label: 'Camera idle', hint: 'Allow camera access once. Sentences will be detected automatically.', tone: 'ready' };
  }
  if (isAnalyzing && backendState === 'signing') {
    return { label: 'Recognizing', hint: 'Recognizing live signing...', tone: 'loading' };
  }
  if (backendState === 'signing') {
    return { label: 'Recognizing', hint: 'Sign motion detected. Building sentence buffer...', tone: 'recording' };
  }
  if (backendState === 'finalizing') {
    return { label: 'Processing', hint: 'Sentence ended. Running Model 6 final decode...', tone: 'loading' };
  }
  if (backendState === 'result' && hasFinal) {
    return { label: 'Final', hint: 'Final sentence ready. Watching for the next sign...', tone: 'success' };
  }
  return { label: 'Watching', hint: 'Watching for signing...', tone: 'ready' };
}

export default function RecognitionPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lastToastSentenceRef = useRef('');
  const [cameraStatus, setCameraStatus] = useState('idle');
  const camera = useCamera();

  const cameraReady = Boolean(camera.stream);
  const captureProfile = useMemo(
    () => (
      camera.facingMode === 'environment'
        ? { frameWidth: 640, jpegQuality: 0.82, maxFrames: 56, sampleMs: 90 }
        : { frameWidth: 560, jpegQuality: 0.8, maxFrames: 48, sampleMs: 100 }
    ),
    [camera.facingMode]
  );

  const { startCapture, stopCapture, drainFrames, clearFrames } = useFrameCapture({
    videoRef,
    canvasRef,
    ...captureProfile,
  });

  const { state, startSession, stopSession, clearOutput } = useLiveInference({
    drainFrames,
    clearFrames,
    cameraReady,
  });

  const phase = useMemo(
    () => phaseDetails({ cameraReady, backendState: state.backendState, isAnalyzing: state.isAnalyzing, hasFinal: Boolean(state.finalSentence) }),
    [state.backendState, cameraReady, state.finalSentence, state.isAnalyzing]
  );

  const displayedSentence = state.backendState === 'signing' && state.draftSentence ? state.draftSentence : state.finalSentence || state.draftSentence;
  const outputMode = state.backendState === 'signing' && state.draftSentence ? 'Live Draft' : state.finalSentence ? 'Final Sentence' : 'Recognized Sentence';
  const confidence = Math.max(0, Math.min(1, state.latestResult?.confidence?.value || 0));

  // Attach the active camera stream to the preview element.
  const prevStreamRef = useRef(null);
  if (videoRef.current && camera.stream && camera.stream !== prevStreamRef.current) {
    videoRef.current.srcObject = camera.stream;
    prevStreamRef.current = camera.stream;
  }

  const startLiveCapture = useCallback(() => {
    const startTime = Date.now();
    startCapture();
    startSession(startTime);
    setCameraStatus('watching');
  }, [startCapture, startSession]);

  const enableCamera = useCallback(async () => {
    setCameraStatus('requesting');
    try {
      await camera.requestCamera();
      startLiveCapture();
    } catch (err) {
      setCameraStatus('error');
      showToast({ type: 'error', message: err.message || 'Camera permission denied' });
    }
  }, [camera, startLiveCapture]);

  const pauseCamera = useCallback(() => {
    stopCapture();
    stopSession();
    camera.stopCamera();
    setCameraStatus('idle');
  }, [camera, stopCapture, stopSession]);

  const switchCamera = async (deviceId) => {
    try {
      await camera.switchCamera(deviceId);
    } catch (_err) {
      setCameraStatus('error');
    }
  };

  const clearLiveSentence = () => {
    clearOutput();
  };

  const copySentence = async () => {
    await navigator.clipboard?.writeText(displayedSentence);
  };

  useEffect(() => {
    const sentence = state.finalSentence?.trim();
    if (sentence && sentence !== lastToastSentenceRef.current) {
      lastToastSentenceRef.current = sentence;
      showToast({ type: 'success', message: 'Translation complete' });
    }
  }, [state.finalSentence]);

  const isStarting = cameraStatus === 'requesting';

  return (
    <AnimatedPage>
      <main className="page live-page">
        <PageHeader eyebrow="Automatic Real-Time Recognition" title="Live BdSL Sentence Translation">
          Keep the camera active. The system detects signing motion, streams sentence drafts, and finalizes when the sentence ends.
        </PageHeader>

        <div className="recognition-layout live-layout">
          <LiveCameraStage
            videoRef={videoRef}
            stream={camera.stream}
            devices={camera.devices}
            selectedDeviceId={camera.selectedDeviceId}
            shouldMirrorPreview={camera.shouldMirrorPreview}
            onEnableCamera={enableCamera}
            onPauseCamera={pauseCamera}
            onSwitchCamera={switchCamera}
            isStarting={isStarting}
            isLive={cameraReady}
            isAnalyzing={state.isAnalyzing || state.backendState === 'finalizing'}
            elapsedSeconds={state.elapsedSeconds}
            phaseLabel={phase.label}
            phaseTone={phase.tone}
          />
          <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />

          <aside className="stack" aria-label="Live translation result">
            <section className="result-panel live-output-panel" aria-live="polite">
              <div className="result-topline">
                <div>
                  <p className="eyebrow">{outputMode}</p>
                  <h2>Recognized Sentence</h2>
                </div>
                <StatusBadge tone={phase.tone}>{phase.label}</StatusBadge>
              </div>

              <div className="result-console live-result-console">
                <div className="console-meta" aria-hidden="true">
                  <span>BN</span>
                  <span className={state.backendState === 'signing' ? 'console-mode-badge is-draft' : 'console-mode-badge is-auto'}>
                    <Sparkles size={13} />
                    {state.backendState === 'signing' ? 'Live Draft' : 'Auto Segmentation'}
                  </span>
                </div>
                {displayedSentence ? (
                  <p className="bangla-result" lang="bn">
                    <TypewriterText text={displayedSentence} speed={state.backendState === 'signing' ? 12 : 16} startDelay={30} />
                  </p>
                ) : (
                  <p className="result-placeholder">Allow camera access to begin real-time sentence recognition.</p>
                )}
                {state.backendState === 'signing' && state.finalSentence ? (
                  <p className="candidate-line" lang="bn">Last final: {state.finalSentence}</p>
                ) : null}
                <div className="live-confidence" aria-label={`Confidence ${Math.round(confidence * 100)} percent`}>
                  <span style={{ width: `${Math.round(confidence * 100)}%` }} />
                </div>
              </div>

              <div className="controls-row">
                <button type="button" className="button primary" onClick={copySentence} disabled={!displayedSentence}>
                  <Copy size={18} aria-hidden="true" />
                  Copy Sentence
                </button>
                <button type="button" className="button ghost" onClick={clearLiveSentence}>
                  <RotateCcw size={18} aria-hidden="true" />
                  Clear Output
                </button>
              </div>
            </section>

            {state.error ? (
              <div className="error-box" role="alert">
                <h3>
                  <AlertTriangle size={18} aria-hidden="true" /> Recognition unavailable
                </h3>
                <p>{state.error || camera.error}</p>
              </div>
            ) : null}

            <div className="privacy-note">
              <ShieldCheck size={18} aria-hidden="true" />
              <span>
                Model 6 remains unchanged. Live motion gating decides sentence start and end before streaming draft and final CTC output.
              </span>
            </div>
          </aside>
        </div>
      </main>
    </AnimatedPage>
  );
}
