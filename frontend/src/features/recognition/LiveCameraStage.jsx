import { Camera, PauseCircle, Power, Timer } from 'lucide-react';
import StatusBadge from '../../components/StatusBadge.jsx';
import { formatSeconds } from '../../lib/formatters.js';

export default function LiveCameraStage({
  videoRef,
  stream,
  devices,
  selectedDeviceId,
  shouldMirrorPreview,
  onEnableCamera,
  onPauseCamera,
  onSwitchCamera,
  isStarting,
  isLive,
  isAnalyzing,
  elapsedSeconds,
  phaseLabel,
  phaseTone,
}) {
  const isCameraReady = Boolean(stream);
  const processing = isCameraReady && (isAnalyzing || phaseTone === 'recording');

  return (
    <section className="camera-stage live-camera-stage" aria-label="Live recognition camera">
      <div className={`camera-frame${processing ? ' is-processing' : ''}`}>
        <video
          ref={videoRef}
          className={`camera-video${shouldMirrorPreview ? ' mirrored' : ''}`}
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview"
        />
        {!isCameraReady ? (
          <div className="camera-empty">
            <div className="camera-empty-inner">
              <div className="camera-empty-icon" aria-hidden="true">
                <Camera size={28} />
              </div>
              <h2>Allow Camera</h2>
              <p className="subtle">Enable camera access to start continuous sign detection and sentence translation.</p>
              <button type="button" className="button primary" onClick={onEnableCamera} disabled={isStarting}>
                <Power size={18} aria-hidden="true" />
                Enable Camera
              </button>
            </div>
          </div>
        ) : null}

      </div>

      {isCameraReady ? (
        <div className="live-rail" aria-live="polite">
          <StatusBadge tone={phaseTone}>{phaseLabel}</StatusBadge>
          <StatusBadge tone={isAnalyzing ? 'loading' : 'success'}>
            {isAnalyzing ? 'Model 6 active' : 'Watching'}
          </StatusBadge>
          <span className="timer">
            <Timer size={15} aria-hidden="true" />
            {formatSeconds(elapsedSeconds)}
          </span>
        </div>
      ) : null}

      <div className="stage-footer">
        <div className="camera-tools">
          <StatusBadge tone={isLive ? 'recording' : isCameraReady ? 'success' : 'ready'}>
            {isLive ? 'Auto detection on' : isCameraReady ? 'Camera ready' : 'Camera needed'}
          </StatusBadge>
          {devices.length > 1 ? (
            <label>
              <span className="sr-only">Camera device</span>
              <select
                className="select"
                value={selectedDeviceId}
                onChange={(event) => onSwitchCamera(event.target.value)}
                disabled={isLive}
              >
                {devices.map((device, index) => (
                  <option key={device.deviceId || index} value={device.deviceId}>
                    {device.label || `Camera ${index + 1}`}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>

        {isCameraReady ? (
          <div className="camera-actions">
            <button type="button" className="button ghost icon-button" onClick={onPauseCamera} aria-label="Pause camera" data-tooltip="Pause camera">
              <PauseCircle size={18} aria-hidden="true" />
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
