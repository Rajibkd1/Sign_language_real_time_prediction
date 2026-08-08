import { memo, useEffect, useMemo, useRef, useState } from 'react';

function TypewriterTextInner({ text = '', className = '', speed = 18, startDelay = 80, onDone }) {
  const content = useMemo(() => String(text || ''), [text]);
  const [visible, setVisible] = useState('');
  const prevTextRef = useRef('');

  useEffect(() => {
    // If text hasn't changed, don't restart animation
    if (prevTextRef.current === content && visible === content) return undefined;
    prevTextRef.current = content;

    setVisible('');
    if (!content) return undefined;

    let index = 0;
    let rafId;
    let lastTime = 0;
    const prefersReduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    if (prefersReduced) {
      setVisible(content);
      onDone?.();
      return undefined;
    }

    const tick = (timestamp) => {
      if (!lastTime) {
        lastTime = timestamp;
      }

      const elapsed = timestamp - lastTime;
      const char = content[index - 1] || '';
      const pause = /[.!?\u0964\uff01\uff1f]/.test(char) ? speed * 8 : speed;

      if (elapsed >= (index === 0 ? startDelay : pause)) {
        index += 1;
        setVisible(content.slice(0, index));
        lastTime = timestamp;

        if (index >= content.length) {
          onDone?.();
          return;
        }
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [content, onDone, speed, startDelay]);

  return (
    <span className={`typewriter ${className}`}>
      {visible}
      {visible.length < content.length ? <span className="typewriter-cursor" aria-hidden="true" /> : null}
    </span>
  );
}

const TypewriterText = memo(TypewriterTextInner);
export default TypewriterText;
