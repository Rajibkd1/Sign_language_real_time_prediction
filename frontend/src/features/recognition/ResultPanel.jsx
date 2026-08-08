import { Copy, Download, Info, RotateCcw, Sparkles } from 'lucide-react';
import StatusBadge from '../../components/StatusBadge.jsx';
import TypewriterText from '../../components/TypewriterText.jsx';
import { downloadJson } from '../../lib/formatters.js';

export default function ResultPanel({ result, onTryAgain }) {
  if (!result) return null;
  const warnings = Array.isArray(result.warnings) ? result.warnings.filter(Boolean) : [];

  const copySentence = async () => {
    await navigator.clipboard?.writeText(result.prediction);
  };

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-topline">
        <div>
          <p className="eyebrow">Translation Result</p>
          <h2>Sentence Output</h2>
        </div>
        <StatusBadge tone="success">Sentence decoded</StatusBadge>
      </div>

      <div className="result-console">
        <div className="console-meta" aria-hidden="true">
          <span>BN</span>
          <span>
            <Sparkles size={13} />
            Model 6 Output
          </span>
        </div>
        <p className="bangla-result" lang="bn">
          <TypewriterText text={result.prediction} speed={22} startDelay={140} />
        </p>
      </div>

      <p className="subtle result-disclaimer">
        Confidence is shown as an uncalibrated estimate, not a calibrated probability.
      </p>
      {warnings.length > 0 ? (
        <div className="result-warnings" role="note">
          {warnings.map((warning) => (
            <p className="result-warning-item" key={warning}>
              <Info size={16} aria-hidden="true" />
              <span>{warning}</span>
            </p>
          ))}
        </div>
      ) : null}
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
