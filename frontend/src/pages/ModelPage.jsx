import {
  BrainCircuit,
  Cpu,
  Database,
  FileVideo,
  Fingerprint,
  Layers3,
  Route,
  ScanFace,
  Sigma,
  Workflow,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import AnimatedPage from '../components/AnimatedPage.jsx';
import PageHeader from '../components/PageHeader.jsx';
import SkeletonLoader from '../components/SkeletonLoader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { getModelInfo } from '../lib/api.js';

const pipeline = [
  ['Webcam Video', FileVideo],
  ['MediaPipe', ScanFace],
  ['1662-D Sequence', Database],
  ['Raw + Velocity', Route],
  ['Region Projection', Layers3],
  ['Fusion Gate', Fingerprint],
  ['Temporal Conv', Workflow],
  ['Transformer', BrainCircuit],
  ['Word-Level CTC', Sigma],
];

export default function ModelPage() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getModelInfo()
      .then((data) => {
        setInfo(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Recognition service is currently unavailable.');
        setLoading(false);
      });
  }, []);

  return (
    <AnimatedPage>
      <main className="page page-narrow">
        <PageHeader eyebrow="Runtime Model" title="Model 6">
          Actual backend status for Lightweight Region-Motion Depthwise Conv-Transformer + Word-Level CTC.
        </PageHeader>

        <div className="page-grid">
          {error ? (
            <div className="error-box" role="alert">
              {error}
            </div>
          ) : null}

          <section className="plain-panel">
            <div className="controls-row" style={{ justifyContent: 'space-between' }}>
              {loading ? (
                <>
                  <SkeletonLoader width="140px" height="2rem" />
                  <SkeletonLoader width="120px" height="2rem" />
                </>
              ) : (
                <>
                  <StatusBadge tone={info?.ready ? 'success' : 'warning'}>{info?.ready ? 'Model ready' : 'Model not ready'}</StatusBadge>
                  <StatusBadge tone="ready">
                    <Cpu size={14} aria-hidden="true" />
                    {info?.device || 'device unavailable'}
                  </StatusBadge>
                </>
              )}
            </div>
          </section>

          {loading ? (
            <section className="model-grid">
              {Array.from({ length: 9 }, (_, i) => (
                <SkeletonLoader key={i} variant="metric" />
              ))}
            </section>
          ) : (
            <section className="model-grid" aria-label="Model runtime details">
              <Metric label="Parameters" value={info?.parameters?.toLocaleString() || 'Unavailable'} />
              <Metric label="Input" value={info?.input_shape || '[T, 1662]'} />
              <Metric label="Vocabulary" value={info?.vocab_size ?? 'Unavailable'} />
              <Metric label="Blank index" value={info?.blank_index ?? 'Unavailable'} />
              <Metric label="Target FPS" value={info?.target_fps ?? 25} />
              <Metric label="Max sequence" value={info?.max_sequence_length ?? 384} />
              <Metric label="Checkpoint source" value={info?.weight_source || 'Unavailable'} />
              <Metric label="Output length" value={info?.output_temporal_formula || 'floor((T - 1) / 2) + 1'} />
              <Metric label="Architecture" value="Word-Level CTC" />
            </section>
          )}

          <section className="plain-panel">
            <h2>Pipeline</h2>
            <div className="pipeline" aria-label="Model 6 pipeline visualization">
              {pipeline.map(([label, Icon], index) => (
                <div className="pipeline-node" key={label} style={{ animationDelay: `${index * 60}ms` }}>
                  <Icon size={20} aria-hidden="true" />
                  <strong>{label}</strong>
                </div>
              ))}
            </div>
          </section>

          {info?.error ? (
            <div className="error-box" role="alert">
              {info.error}
            </div>
          ) : null}
        </div>
      </main>
    </AnimatedPage>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
