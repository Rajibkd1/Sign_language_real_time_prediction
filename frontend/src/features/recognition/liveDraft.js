export function wordCount(sentence = '') {
  return sentence.trim().split(/\s+/).filter(Boolean).length;
}

function words(sentence = '') {
  return sentence.trim().split(/\s+/).filter(Boolean);
}

export function mergeOverlappingDraft(current, next) {
  const currentWords = words(current);
  const nextWords = words(next);
  if (!currentWords.length) return nextWords.join(' ');
  if (!nextWords.length) return currentWords.join(' ');

  const maxOverlap = Math.min(currentWords.length, nextWords.length);
  for (let size = maxOverlap; size > 0; size -= 1) {
    const currentTail = currentWords.slice(currentWords.length - size).join('\u0001');
    const nextHead = nextWords.slice(0, size).join('\u0001');
    if (currentTail === nextHead) {
      return [...currentWords, ...nextWords.slice(size)].join(' ');
    }
  }

  return currentWords.join(' ');
}

export function chooseLiveDraft(current, result) {
  const next = String(result?.prediction || '').trim();
  if (!next) return current;
  if (!current) return next;
  const merged = mergeOverlappingDraft(current, next);
  if (wordCount(merged) > wordCount(current)) return merged;
  if (wordCount(next) >= wordCount(current)) return next;
  if ((result?.confidence?.value || 0) >= 0.62) return next;
  return current;
}

export function isSoftLiveError(error) {
  return ['recording_too_short', 'no_ctc_words', 'mediapipe_failure'].includes(error?.code);
}

export function hasEnoughLiveSignal(result) {
  const detections = result?.detections || {};
  return Math.max(detections.left_hand_rate || 0, detections.right_hand_rate || 0) >= 0.08 || (detections.pose_rate || 0) >= 0.35;
}
