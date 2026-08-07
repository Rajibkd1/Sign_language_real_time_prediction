export const recognitionStates = {
  IDLE: 'idle',
  REQUESTING_CAMERA: 'requesting_camera',
  CAMERA_READY: 'camera_ready',
  RECORDING: 'recording',
  STOPPING: 'stopping',
  UPLOADING: 'uploading',
  PROCESSING: 'processing',
  SUCCESS: 'success',
  ERROR: 'error'
};

export const initialRecognitionState = {
  status: recognitionStates.IDLE,
  elapsedSeconds: 0,
  result: null,
  error: '',
  phase: '',
  startedAt: 0
};

export function recognitionReducer(state, action) {
  switch (action.type) {
    case 'REQUEST_CAMERA':
      return { ...initialRecognitionState, status: recognitionStates.REQUESTING_CAMERA };
    case 'CAMERA_READY':
      return { ...state, status: recognitionStates.CAMERA_READY, error: '' };
    case 'CAMERA_ERROR':
      return { ...state, status: recognitionStates.ERROR, error: action.error || "We couldn't access your camera." };
    case 'START_RECORDING':
      return {
        ...initialRecognitionState,
        status: recognitionStates.RECORDING,
        startedAt: action.startedAt || Date.now()
      };
    case 'TICK':
      if (state.status !== recognitionStates.RECORDING) return state;
      return { ...state, elapsedSeconds: action.elapsedSeconds };
    case 'STOPPING':
      return { ...state, status: recognitionStates.STOPPING };
    case 'UPLOAD':
      return { ...state, status: recognitionStates.UPLOADING, phase: 'uploading' };
    case 'PROCESS':
      return { ...state, status: recognitionStates.PROCESSING, phase: action.phase || 'processing' };
    case 'SUCCESS':
      return {
        ...state,
        status: recognitionStates.SUCCESS,
        result: action.result,
        error: '',
        phase: ''
      };
    case 'ERROR':
      return {
        ...state,
        status: recognitionStates.ERROR,
        error: action.error || 'Recognition failed. Please try again.',
        phase: ''
      };
    case 'CLEAR':
      return {
        ...initialRecognitionState,
        status: action.keepCamera ? recognitionStates.CAMERA_READY : recognitionStates.IDLE
      };
    default:
      return state;
  }
}

export function canStart(state, hasCamera) {
  return hasCamera && [recognitionStates.CAMERA_READY, recognitionStates.SUCCESS, recognitionStates.ERROR].includes(state.status);
}

export function canStop(state) {
  return state.status === recognitionStates.RECORDING;
}

export function isBusy(state) {
  return [
    recognitionStates.REQUESTING_CAMERA,
    recognitionStates.STOPPING,
    recognitionStates.UPLOADING,
    recognitionStates.PROCESSING
  ].includes(state.status);
}

