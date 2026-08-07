import { describe, expect, it } from 'vitest';
import {
  canStart,
  canStop,
  initialRecognitionState,
  recognitionReducer,
  recognitionStates
} from './recognitionMachine.js';

describe('recognition state machine', () => {
  it('moves from camera ready to recording and blocks duplicate start', () => {
    const ready = recognitionReducer(initialRecognitionState, { type: 'CAMERA_READY' });
    expect(canStart(ready, true)).toBe(true);
    const recording = recognitionReducer(ready, { type: 'START_RECORDING', startedAt: 100 });
    expect(recording.status).toBe(recognitionStates.RECORDING);
    expect(canStart(recording, true)).toBe(false);
    expect(canStop(recording)).toBe(true);
  });

  it('keeps success result and can clear while preserving camera', () => {
    const success = recognitionReducer(initialRecognitionState, {
      type: 'SUCCESS',
      result: { prediction: 'বাবা সকালে কফি খায়নি' }
    });
    expect(success.status).toBe(recognitionStates.SUCCESS);
    const cleared = recognitionReducer(success, { type: 'CLEAR', keepCamera: true });
    expect(cleared.status).toBe(recognitionStates.CAMERA_READY);
    expect(cleared.result).toBe(null);
  });

  it('normalizes errors into the error state', () => {
    const next = recognitionReducer(initialRecognitionState, { type: 'ERROR', error: 'Backend unavailable' });
    expect(next.status).toBe(recognitionStates.ERROR);
    expect(next.error).toBe('Backend unavailable');
  });
});

