export default function SkeletonLoader({
  width = '100%',
  height = '1rem',
  borderRadius = 'var(--radius-sm)',
  variant = 'text',
  count = 1,
  className = '',
}) {
  const style = {
    width,
    height: variant === 'circle' ? width : height,
    borderRadius: variant === 'circle' ? '50%' : borderRadius,
  };

  if (variant === 'card') {
    style.height = height || '120px';
    style.borderRadius = 'var(--radius)';
  }

  if (variant === 'metric') {
    style.height = '5rem';
    style.borderRadius = 'var(--radius)';
  }

  if (count <= 1) {
    return <span className={`skeleton ${className}`} style={style} aria-hidden="true" />;
  }

  return (
    <span className={`skeleton-group ${className}`} aria-hidden="true">
      {Array.from({ length: count }, (_, i) => (
        <span
          key={i}
          className="skeleton"
          style={{
            ...style,
            width: i === count - 1 ? '60%' : style.width,
            animationDelay: `${i * 80}ms`,
          }}
        />
      ))}
    </span>
  );
}
