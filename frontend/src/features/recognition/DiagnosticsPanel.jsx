import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import { formatMs, formatRate, formatSeconds } from '../../lib/formatters.js';

export default function DiagnosticsPanel({ result }) {
  const [expanded, setExpanded] = useState(false);

  if (!result) return null;

  return (
    <section className="diagnostics-panel" aria-label="Recognition diagnostics">
      <button
        type="button"
        className="diagnostics-toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <h3>Diagnostics</h3>
        {expanded ? <ChevronUp size={18} aria-hidden="true" /> : <ChevronDown size={18} aria-hidden="true" />}
      </button>

      <div className={`diagnostics-body${expanded ? ' open' : ''}`}>
        <div className="metric-grid">
          <div className="metric">
            <span>Duration</span>
            <strong>{formatSeconds(result.duration_seconds)}</strong>
          </div>
          <div className="metric">
            <span>Input frames</span>
            <strong>{result.input_frames}</strong>
          </div>
          <div className="metric">
            <span>MediaPipe</span>
            <strong>{formatMs(result.timings?.mediapipe_ms)}</strong>
          </div>
          <div className="metric">
            <span>Model 6</span>
            <strong>{formatMs(result.timings?.model_ms)}</strong>
          </div>
          <div className="metric">
            <span>Pose</span>
            <strong>{formatRate(result.detections?.pose_rate)}</strong>
          </div>
          <div className="metric">
            <span>Face</span>
            <strong>{formatRate(result.detections?.face_rate)}</strong>
          </div>
          <div className="metric">
            <span>Left hand</span>
            <strong>{formatRate(result.detections?.left_hand_rate)}</strong>
          </div>
          <div className="metric">
            <span>Right hand</span>
            <strong>{formatRate(result.detections?.right_hand_rate)}</strong>
          </div>
        </div>
      </div>
    </section>
  );
}
