import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

const icons = {
  success: CheckCircle2,
  warning: AlertTriangle,
  error: AlertTriangle,
  info: Info,
};

export function showToast({ type = 'info', message }) {
  if (!message) return;
  window.dispatchEvent(new CustomEvent('bdsl-toast', { detail: { type, message } }));
}

export default function ToastHost() {
  const [items, setItems] = useState([]);

  const dismiss = useCallback((id) => {
    setItems((value) => value.filter((item) => item.id !== id));
  }, []);

  useEffect(() => {
    const onToast = (event) => {
      const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
      const toast = {
        id,
        type: event.detail?.type || 'info',
        message: event.detail?.message || '',
      };
      setItems((value) => [toast, ...value].slice(0, 4));
      window.setTimeout(() => dismiss(id), 4000);
    };

    window.addEventListener('bdsl-toast', onToast);
    return () => window.removeEventListener('bdsl-toast', onToast);
  }, [dismiss]);

  return (
    <div className="toast-viewport" role="region" aria-label="Notifications">
      {items.map((item) => {
        const Icon = icons[item.type] || Info;
        return (
          <div key={item.id} className={`toast toast-${item.type}`} role="status">
            <Icon size={18} aria-hidden="true" />
            <p>{item.message}</p>
            <button type="button" className="toast-dismiss" onClick={() => dismiss(item.id)} aria-label="Dismiss notification">
              <X size={14} aria-hidden="true" />
            </button>
            <span className="toast-timer" aria-hidden="true" />
          </div>
        );
      })}
    </div>
  );
}
