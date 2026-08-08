import { describe, expect, it } from 'vitest';
import { chooseLiveDraft, hasEnoughLiveSignal, isSoftLiveError, mergeOverlappingDraft, wordCount } from './liveDraft.js';

describe('live draft helpers', () => {
  it('counts sentence tokens', () => {
    expect(wordCount('ma vaat khabe')).toBe(3);
  });

  it('allows a longer sentence draft to replace the previous draft', () => {
    expect(
      chooseLiveDraft('ma khabe', {
        prediction: 'ma vaat khabe',
        confidence: { value: 0.41 }
      })
    ).toBe('ma vaat khabe');
  });

  it('merges overlapping rolling-window drafts', () => {
    expect(mergeOverlappingDraft('ma khabe', 'khabe vaat')).toBe('ma khabe vaat');
  });

  it('keeps a shorter low-confidence regression', () => {
    expect(
      chooseLiveDraft('ma vaat khabe', {
        prediction: 'ma',
        confidence: { value: 0.4 }
      })
    ).toBe('ma vaat khabe');
  });

  it('suppresses expected incomplete live-window errors', () => {
    expect(isSoftLiveError({ code: 'no_ctc_words' })).toBe(true);
    expect(isSoftLiveError({ code: 'model_not_ready' })).toBe(false);
  });

  it('requires hand or strong pose signal for live publication', () => {
    expect(hasEnoughLiveSignal({ detections: { left_hand_rate: 0.1, right_hand_rate: 0, pose_rate: 0 } })).toBe(true);
    expect(hasEnoughLiveSignal({ detections: { left_hand_rate: 0, right_hand_rate: 0, pose_rate: 0.4 } })).toBe(true);
    expect(hasEnoughLiveSignal({ detections: { left_hand_rate: 0, right_hand_rate: 0, pose_rate: 0.1 } })).toBe(false);
  });
});
