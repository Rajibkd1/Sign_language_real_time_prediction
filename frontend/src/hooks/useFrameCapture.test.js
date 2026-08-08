import { describe, expect, it } from 'vitest';
import { FrameRingBuffer } from './useFrameCapture.js';

describe('FrameRingBuffer', () => {
  it('drains the oldest frames and preserves queued continuity', () => {
    const buffer = new FrameRingBuffer(5);
    [1, 2, 3, 4, 5].forEach((timestampMs) => buffer.push({ timestampMs }));

    expect(buffer.drain(2).map((frame) => frame.timestampMs)).toEqual([1, 2]);
    expect(buffer.size).toBe(3);
    expect(buffer.drain(2).map((frame) => frame.timestampMs)).toEqual([3, 4]);
    expect(buffer.drain(2).map((frame) => frame.timestampMs)).toEqual([5]);
  });

  it('keeps chronological order after capacity overflow', () => {
    const buffer = new FrameRingBuffer(3);
    [1, 2, 3, 4].forEach((timestampMs) => buffer.push({ timestampMs }));

    expect(buffer.drain(2).map((frame) => frame.timestampMs)).toEqual([2, 3]);
    buffer.push({ timestampMs: 5 });

    expect(buffer.drain(3).map((frame) => frame.timestampMs)).toEqual([4, 5]);
  });

  it('drains the newest frames and discards stale backlog for realtime streaming', () => {
    const buffer = new FrameRingBuffer(8);
    [1, 2, 3, 4, 5, 6].forEach((timestampMs) => buffer.push({ timestampMs }));

    expect(buffer.drainLatest(3).map((frame) => frame.timestampMs)).toEqual([4, 5, 6]);
    expect(buffer.size).toBe(0);
  });
});
