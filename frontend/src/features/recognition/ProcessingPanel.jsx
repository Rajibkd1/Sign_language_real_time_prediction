import ProgressLoader from '../../components/ProgressLoader.jsx';

export default function ProcessingPanel({ phase }) {
  const activeIndex = phase === 'uploading' ? 0 : 1;
  return (
    <section className="plain-panel processing-panel-wrapper">
      <ProgressLoader
        activeIndex={activeIndex}
        title="Processing video"
        subtitle="Analyzing your sign language clip..."
      />
    </section>
  );
}
