import { Copy, Download, RotateCcw } from 'lucide-react';
import StatusBadge from '../../components/StatusBadge.jsx';
import { downloadJson } from '../../lib/formatters.js';

export default function ResultPanel({ result, onTryAgain }) {
  if (!result) return null;

  const copySentence = async () => {
    await navigator.clipboard?.writeText(result.prediction);
  };

  return (
    <section className="result-panel" aria-live="polite">
      <StatusBadge tone="success">Sentence decoded</StatusBadge>
      <p className="bangla-result" lang="bn">
        {result.prediction}
      </p>
      <p className="subtle">
        Confidence is shown as an uncalibrated confidence estimate, not a calibrated probability.
      </p>
      <div className="controls-row">
        <button type="button" className="button primary" onClick={copySentence}>
          <Copy size={18} aria-hidden="true" />
          Copy Sentence
        </button>
        <button type="button" className="button" onClick={onTryAgain}>
          <RotateCcw size={18} aria-hidden="true" />
          Try Again
        </button>
        <button
          type="button"
          className="button ghost"
          onClick={() => downloadJson(`bdsl-inference-${result.id}.json`, result)}
        >
          <Download size={18} aria-hidden="true" />
          JSON
        </button>
      </div>
    </section>
  );
}

