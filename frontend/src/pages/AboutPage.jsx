import { BrainCircuit, Camera, Database, ShieldCheck } from 'lucide-react';
import AnimatedPage from '../components/AnimatedPage.jsx';
import PageHeader from '../components/PageHeader.jsx';

const infoBlocks = [
  {
    icon: Camera,
    title: 'Browser Capture',
    text: 'The webcam preview runs in the browser and streams sampled frames to the local recognition API.',
  },
  {
    icon: Database,
    title: 'Landmark Sequence',
    text: 'MediaPipe extracts pose, face, left-hand, and right-hand landmarks into [T, 1662] features.',
  },
  {
    icon: BrainCircuit,
    title: 'Model 6',
    text: 'The architecture combines raw and velocity streams, region projection, temporal convolution, Transformer encoding, and word-level CTC.',
  },
  {
    icon: ShieldCheck,
    title: 'Privacy',
    text: 'Temporary recordings are deleted after inference. Raw video is not stored by default.',
  },
];

export default function AboutPage() {
  return (
    <AnimatedPage>
      <main className="page page-narrow">
        <PageHeader eyebrow="Research System" title="BdSL Vision">
          Bangla Sign Language Sentence Recognition
        </PageHeader>

        <div className="page-grid">
          <section className="plain-panel">
            <h2>Continuous Sentence Recognition</h2>
            <p className="subtle">
              BdSL Vision translates sentence-level signing from live camera input or uploaded video. Model 6 remains the
              final sequence decoder, while the interface handles live motion gating and sentence boundaries.
            </p>
          </section>

          <section className="model-grid">
            {infoBlocks.map((block, index) => (
              <InfoBlock key={block.title} {...block} delay={index * 60} />
            ))}
          </section>
        </div>
      </main>
    </AnimatedPage>
  );
}

function InfoBlock({ icon: Icon, title, text, delay = 0 }) {
  return (
    <article className="plain-panel info-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="info-card-icon">
        <Icon size={22} aria-hidden="true" />
      </div>
      <h2>{title}</h2>
      <p className="subtle">{text}</p>
    </article>
  );
}
