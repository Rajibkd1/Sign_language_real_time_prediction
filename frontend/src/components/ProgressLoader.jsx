import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

const defaultSteps = ['Uploading', 'Extracting landmarks', 'Running Model 6', 'Decoding sentence'];

export default function ProgressLoader({
  steps = defaultSteps,
  activeIndex = 0,
  title = 'Processing video',
  subtitle,
}) {
  return (
    <div className="progress-loader" aria-live="polite" aria-busy="true">
      <div className="progress-loader-header">
        <div className="progress-spinner" aria-hidden="true">
          <svg viewBox="0 0 44 44" width="44" height="44">
            <circle className="progress-spinner-track" cx="22" cy="22" r="18" fill="none" strokeWidth="3" />
            <circle className="progress-spinner-fill" cx="22" cy="22" r="18" fill="none" strokeWidth="3"
              strokeDasharray="80 33" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <p className="eyebrow">Recognition Pipeline</p>
          <h2>{title}</h2>
          {subtitle ? <p className="subtle">{subtitle}</p> : null}
        </div>
      </div>

      <div className="progress-track" aria-hidden="true">
        <div className="progress-track-fill" style={{ width: `${Math.min(100, ((activeIndex + 0.5) / steps.length) * 100)}%` }} />
      </div>

      <ul className="progress-steps">
        {steps.map((step, index) => {
          const isComplete = index < activeIndex;
          const isActive = index === activeIndex;
          return (
            <li key={step} className={`progress-step${isActive ? ' active' : ''}${isComplete ? ' complete' : ''}`}>
              <span className="progress-step-icon">
                {isComplete ? <CheckCircle2 size={18} /> : isActive ? <Loader2 size={18} /> : <Circle size={18} />}
              </span>
              <span className="progress-step-label">{step}</span>
              {isActive ? <span className="progress-step-dots"><span /><span /><span /></span> : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
