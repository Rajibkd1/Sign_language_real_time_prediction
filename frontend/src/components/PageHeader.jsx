export default function PageHeader({ eyebrow, title, children }) {
  return (
    <div className="section-header">
      {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
      <h1>{title}</h1>
      {children ? <p className="subtle">{children}</p> : null}
      <div className="section-header-line" aria-hidden="true" />
    </div>
  );
}
