import { ArrowRight, Camera, FileVideo, Layers, ShieldCheck, Zap } from 'lucide-react';
import { useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import TypewriterText from '../components/TypewriterText.jsx';

const sampleSentence = '\u09ae\u09be \u09ad\u09be\u09a4 \u0996\u09be\u09ac\u09c7';
const previewCells = Array.from({ length: 48 }, (_, index) => index);

const features = [
  {
    icon: Camera,
    title: 'Live Camera',
    desc: 'Camera on, sign naturally, sentence draft updates continuously.',
  },
  {
    icon: FileVideo,
    title: 'Video Upload',
    desc: 'Decode completed MP4, WebM, MOV, M4V, or AVI clips.',
  },
  {
    icon: Zap,
    title: 'Real-Time Translation',
    desc: 'Instant sentence drafts with sub-second latency feedback.',
  },
  {
    icon: ShieldCheck,
    title: 'Private By Default',
    desc: 'Temporary inference files are removed after processing.',
  },
];

export default function SplashPage() {
  const productRef = useRef(null);

  const handleProductMove = useCallback((event) => {
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;
    const node = productRef.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    node.style.setProperty('--tilt-x', `${(-y * 8).toFixed(2)}deg`);
    node.style.setProperty('--tilt-y', `${(x * 8).toFixed(2)}deg`);
  }, []);

  const resetProductTilt = useCallback(() => {
    const node = productRef.current;
    if (!node) return;
    node.style.setProperty('--tilt-x', '0deg');
    node.style.setProperty('--tilt-y', '0deg');
  }, []);

  return (
    <main className="splash-page">
      <section className="splash-hero">
        <div className="splash-content">
          <div className="live-model-pill">
            <span aria-hidden="true" />
            Model 6 - Live Inference
          </div>
          <div className="splash-brand-badge">
            <Layers size={16} aria-hidden="true" />
            <span>BdSL Vision</span>
          </div>
          <h1>Professional <span className="gradient-text">Bangla Sign</span> Language Translation</h1>
          <p className="splash-copy">
            Real-time sentence recognition for Bangla Sign Language, powered by advanced deep learning with dedicated workflows for live camera and uploaded video clips.
          </p>
          <div className="splash-actions">
            <Link className="button primary" to="/live">
              <Camera size={18} aria-hidden="true" />
              Open Live Camera
              <ArrowRight size={18} aria-hidden="true" className="button-arrow" />
            </Link>
            <Link className="button ghost" to="/upload">
              <FileVideo size={18} aria-hidden="true" />
              Upload Video
            </Link>
          </div>
        </div>

        <div
          ref={productRef}
          className="splash-product"
          aria-hidden="true"
          onMouseMove={handleProductMove}
          onMouseLeave={resetProductTilt}
        >
          <div className="splash-camera-surface">
            <div className="splash-camera-bar">
              <span className="splash-dot" />
              <span>Live Camera</span>
              <span>Model 6</span>
            </div>
            <div className="splash-hand-frame">
              <div className="splash-grid-overlay">
                {previewCells.map((cell) => (
                  <span key={cell} />
                ))}
              </div>
              <span className="splash-frame-line" />
              <span className="splash-frame-line" />
              <span className="splash-frame-line" />
              <span className="splash-scan-window" />
            </div>
          </div>
          <div className="splash-console">
            <span>BN Live Draft</span>
            <strong lang="bn">
              <TypewriterText text={sampleSentence} speed={32} startDelay={400} />
            </strong>
          </div>
        </div>
      </section>

      <section className="splash-band" aria-label="Platform capabilities">
        {features.map((feat, index) => {
          const Icon = feat.icon;
          return (
            <div key={feat.title} className={`feature-card feature-card-${index + 1}`}>
              <div className="feature-card-icon">
                <Icon size={20} aria-hidden="true" />
              </div>
              <strong>{feat.title}</strong>
              <span>{feat.desc}</span>
            </div>
          );
        })}
      </section>
    </main>
  );
}
