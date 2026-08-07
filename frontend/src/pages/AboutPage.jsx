import { BrainCircuit, Camera, Database, ShieldCheck } from 'lucide-react';
import PageHeader from '../components/PageHeader.jsx';

export default function AboutPage() {
  return (
    <main className="page page-narrow">
      <PageHeader eyebrow="Research System" title="BdSL Vision">
        Bangla Sign Language Sentence Recognition
      </PageHeader>

      <div className="page-grid">
        <section className="plain-panel">
          <h2>Continuous Sentence Recognition</h2>
          <p className="subtle">
            BdSL Vision recognizes one completed sign sentence at a time. Model 6 is a full-sequence,
            non-causal encoder, so the final Bangla sentence is generated after signing stops.
          </p>
        </section>

        <section className="model-grid">
          <InfoBlock
            icon={Camera}
            title="Browser Capture"
            text="The webcam preview runs locally in the browser. The completed short clip is sent after Stop."
          />
          <InfoBlock
            icon={Database}
            title="Landmark Sequence"
            text="MediaPipe extracts pose, face, left-hand, and right-hand landmarks into [T, 1662] features."
          />
          <InfoBlock
            icon={BrainCircuit}
            title="Model 6"
            text="The architecture combines raw and velocity streams, region projection, temporal convolution, Transformer encoding, and word-level CTC."
          />
          <InfoBlock
            icon={ShieldCheck}
            title="Privacy"
            text="Temporary recordings are deleted after inference. Raw video is not stored by default."
          />
        </section>
      </div>
    </main>
  );
}

function InfoBlock({ icon: Icon, title, text }) {
  return (
    <article className="plain-panel">
      <Icon size={24} aria-hidden="true" />
      <h2>{title}</h2>
      <p className="subtle">{text}</p>
    </article>
  );
}

