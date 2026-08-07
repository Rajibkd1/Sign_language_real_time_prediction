import { Camera, ChevronDown, Timer } from 'lucide-react';
import StatusBadge from '../../components/StatusBadge.jsx';
import { formatSeconds } from '../../lib/formatters.js';
import { recognitionStates } from './recognitionMachine.js';

export default function CameraStage({
  videoRef,
  stream,
  devices,
  selectedDeviceId,
  onEnableCamera,
  onSwitchCamera,
  state,
  maxSeconds,
  warnSeconds
}) {
  const progress = Math.min(100, (state.elapsedSeconds / maxSeconds) * 100);
  const isRecording = state.status === recognitionStates.RECORDING;
  const isCameraReady = Boolean(stream);

  return (
    <section className="camera-stage" aria-label="Recognition camera">
      <div className="camera-frame">
        <video
          ref={videoRef}
          className="camera-video"
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview"
        />

        {!isCameraReady ? (
          <div className="camera-empty">
            <div className="camera-empty-inner">
              <div className="camera-empty-icon" aria-hidden="true">
                <Camera size={34} />
              </div>
              <h2>Enable Camera</h2>
              <p className="subtle">Record one complete Bangla sign sentence, then send the finished clip to Model 6.</p>
              <button
                type="button"
                className="button primary"
                onClick={onEnableCamera}
                disabled={state.status === recognitionStates.REQUESTING_CAMERA}
              >
                <Camera size={18} aria-hidden="true" />
                Enable Camera
              </button>
            </div>
          </div>
        ) : null}

        {isRecording ? (
          <div className="recording-rail" aria-live="polite">
            <StatusBadge tone={state.elapsedSeconds >= warnSeconds ? 'warning' : 'recording'}>
              Recording
            </StatusBadge>
            <div className="duration-bar" aria-hidden="true">
              <span
                className={`duration-fill${state.elapsedSeconds >= warnSeconds ? ' warn' : ''}`}
                style={{ '--progress': `${progress}%` }}
              />
            </div>
            <span className="timer">
              <Timer size={16} aria-hidden="true" />
              {formatSeconds(state.elapsedSeconds)}
            </span>
          </div>
        ) : null}
      </div>

      <div className="stage-footer">
        <div className="camera-tools">
          <StatusBadge tone={isRecording ? 'recording' : isCameraReady ? 'success' : 'ready'}>
            {isRecording ? 'Perform one complete sign sentence' : isCameraReady ? 'Camera ready' : 'Camera inactive'}
          </StatusBadge>
          {devices.length > 1 ? (
            <label>
              <span className="sr-only">Camera device</span>
              <select
                className="select"
                value={selectedDeviceId}
                onChange={(event) => onSwitchCamera(event.target.value)}
                disabled={isRecording}
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
        <span className="subtle" aria-hidden="true">
          <ChevronDown size={16} /> Preview mirrored only
        </span>
      </div>
    </section>
  );
}

