import { Search, Trash2, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { clearHistory, deleteHistoryItem, getHistory } from '../lib/api.js';
import { formatDate, formatMs, formatRate, formatSeconds } from '../lib/formatters.js';

export default function HistoryPage() {
  const [items, setItems] = useState([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const payload = await getHistory();
      setItems(payload.items || []);
    } catch (err) {
      setError(err.message || 'Recognition service is currently unavailable.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return items;
    return items.filter((item) => item.prediction?.toLowerCase().includes(normalized));
  }, [items, query]);

  const remove = async (id) => {
    await deleteHistoryItem(id);
    setItems((value) => value.filter((item) => item.id !== id));
  };

  const clearAll = async () => {
    await clearHistory();
    setItems([]);
  };

  return (
    <main className="page page-narrow">
      <PageHeader eyebrow="Local Predictions" title="History">
        Raw videos are not stored by default. This list keeps prediction metadata for recent sessions.
      </PageHeader>

      <div className="page-grid">
        <section className="plain-panel">
          <div className="history-toolbar">
            <label>
              <span className="sr-only">Search history</span>
              <div style={{ position: 'relative' }}>
                <Search size={17} aria-hidden="true" style={{ position: 'absolute', left: 12, top: 12 }} />
                <input
                  className="input"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search Bangla sentence"
                  style={{ paddingLeft: 38 }}
                />
              </div>
            </label>
            <button type="button" className="button ghost" onClick={clearAll} disabled={!items.length}>
              <Trash2 size={18} aria-hidden="true" />
              Clear All
            </button>
          </div>
        </section>

        {loading ? <StatusBadge tone="loading">Loading history</StatusBadge> : null}
        {error ? (
          <div className="error-box" role="alert">
            {error}
          </div>
        ) : null}

        <section className="history-list" aria-label="Prediction history">
          {!loading && !filtered.length ? (
            <div className="plain-panel">
              <h2>No saved predictions</h2>
              <p className="subtle">Recognition results will appear here after successful decoding.</p>
            </div>
          ) : null}

          {filtered.map((item) => (
            <article className="history-item" key={item.id}>
              <div>
                <div className="history-sentence" lang="bn">
                  {item.prediction}
                </div>
                <div className="history-meta">
                  <span>{formatDate(item.created_at)}</span>
                  <span>{formatSeconds(item.duration_seconds)}</span>
                  <span>{formatMs(item.total_latency_ms)}</span>
                  <span>
                    detection {formatRate(Math.max(item.detections?.left_hand_rate || 0, item.detections?.right_hand_rate || 0))}
                  </span>
                </div>
              </div>
              <button
                type="button"
                className="button ghost icon-button"
                onClick={() => remove(item.id)}
                aria-label="Delete history item"
                title="Delete"
              >
                <X size={18} aria-hidden="true" />
              </button>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}

