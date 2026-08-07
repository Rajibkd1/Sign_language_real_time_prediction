import { AlertTriangle, CheckCircle2, CircleDot, Loader2 } from 'lucide-react';

const icons = {
  ready: CircleDot,
  recording: CircleDot,
  success: CheckCircle2,
  warning: AlertTriangle,
  loading: Loader2
};

export default function StatusBadge({ tone = 'ready', children }) {
  const Icon = icons[tone] || CircleDot;
  return (
    <span className={`status-badge ${tone}`} role="status">
      <Icon size={15} aria-hidden="true" />
      {children}
    </span>
  );
}

