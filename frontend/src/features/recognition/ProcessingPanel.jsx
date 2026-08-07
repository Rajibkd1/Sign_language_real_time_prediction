import { CheckCircle2, Loader2 } from 'lucide-react';

const steps = ['Uploading', 'Extracting landmarks', 'Running Model 6', 'Decoding sentence'];

export default function ProcessingPanel({ phase }) {
  const activeIndex = phase === 'uploading' ? 0 : 1;
  return (
    <section className="plain-panel" aria-live="polite" aria-busy="true">
      <p className="eyebrow">Recognition Pipeline</p>
      <h2>Processing recording</h2>
      <ul className="processing-list">
        {steps.map((step, index) => (
          <li key={step} className={`processing-step${index >= activeIndex ? ' active' : ''}`}>
            {index < activeIndex ? <CheckCircle2 size={16} /> : <Loader2 size={16} />}
            <span>{step}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

