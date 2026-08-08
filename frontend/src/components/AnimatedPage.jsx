export default function AnimatedPage({ children, className = '' }) {
  return (
    <div className={`animated-page ${className}`}>
      {children}
    </div>
  );
}
