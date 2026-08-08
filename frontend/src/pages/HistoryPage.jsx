import { History as HistoryIcon, Search, Trash2, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import AnimatedPage from '../components/AnimatedPage.jsx';
import PageHeader from '../components/PageHeader.jsx';
import SkeletonLoader from '../components/SkeletonLoader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { showToast } from '../components/ToastHost.jsx';
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
      showToast({ type: 'error', message: err.message || 'Could not load history' });
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
    showToast({ type: 'success', message: 'History item deleted' });
  };

  const clearAll = async () => {
    await clearHistory();
    setItems([]);
    showToast({ type: 'success', message: 'History cleared' });
  };

  return (
    <AnimatedPage>
      <main className="page page-narrow">
        <PageHeader eyebrow="Local Predictions" title="History">
          Raw videos are not stored by default. This list keeps prediction metadata for recent sessions.
        </PageHeader>

        <div className="page-grid">
          <section className="plain-panel">
            <div className="history-toolbar">
              <label className="search-field">
                <span className="sr-only">Search history</span>
                <Search size={16} aria-hidden="true" className="search-field-icon" />
                <input
                  className="input"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search Bangla sentence"
                />
              </label>
              <button type="button" className="button ghost" onClick={clearAll} disabled={!items.length}>
                <Trash2 size={18} aria-hidden="true" />
                Clear All
              </button>
            </div>
          </section>

          {loading ? (
            <div className="page-grid">
              <SkeletonLoader variant="card" height="80px" />
              <SkeletonLoader variant="card" height="80px" />
              <SkeletonLoader variant="card" height="80px" />
            </div>
          ) : null}

          {error ? (
            <div className="error-box" role="alert">
              {error}
            </div>
          ) : null}

          <section className="history-list" aria-label="Prediction history">
            {!loading && !filtered.length ? (
              <div className="plain-panel empty-state">
                <div className="empty-state-illustration">
                  <HistoryIcon size={24} aria-hidden="true" />
                  <span />
                </div>
                <h2>No saved predictions</h2>
                <p className="subtle">Recognition results will appear here after successful decoding.</p>
              </div>
            ) : null}

            {filtered.map((item, index) => (
              <article
                className="history-item"
                key={item.id}
                style={{ animationDelay: `${Math.min(index * 40, 400)}ms` }}
              >
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
                  data-tooltip="Delete"
                >
                  <X size={18} aria-hidden="true" />
                </button>
              </article>
            ))}
          </section>
        </div>
      </main>
    </AnimatedPage>
  );
}
