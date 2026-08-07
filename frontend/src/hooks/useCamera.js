import { useCallback, useEffect, useRef, useState } from 'react';

function normalizeCameraError(error) {
  if (!error) return "We couldn't access your camera.";
  if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
    return "We couldn't access your camera. Check browser camera permission and try again.";
  }
  if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    return 'No camera was found on this device.';
  }
  if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
    return 'The camera is already in use by another application.';
  }
  if (error.name === 'OverconstrainedError') {
    return 'The selected camera could not be opened.';
  }
  return error.message || "We couldn't access your camera.";
}

export function useCamera() {
  const [stream, setStream] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [error, setError] = useState('');
  const streamRef = useRef(null);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
  }, []);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const all = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = all.filter((device) => device.kind === 'videoinput');
    setDevices(videoInputs);
    return videoInputs;
  }, []);

  const requestCamera = useCallback(
    async (deviceId = selectedDeviceId) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        const message = 'This browser does not support camera capture.';
        setError(message);
        throw new Error(message);
      }

      stopCamera();
      setError('');
      try {
        const constraints = {
          audio: false,
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
            ...(deviceId ? { deviceId: { exact: deviceId } } : {})
          }
        };
        const nextStream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = nextStream;
        setStream(nextStream);
        const videoDevices = await refreshDevices();
        const activeTrack = nextStream.getVideoTracks()[0];
        const activeSettings = activeTrack?.getSettings?.() || {};
        if (activeSettings.deviceId) {
          setSelectedDeviceId(activeSettings.deviceId);
        } else if (!selectedDeviceId && videoDevices[0]?.deviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
        return nextStream;
      } catch (err) {
        const message = normalizeCameraError(err);
        setError(message);
        throw new Error(message);
      }
    },
    [refreshDevices, selectedDeviceId, stopCamera]
  );

  const switchCamera = useCallback(
    async (deviceId) => {
      setSelectedDeviceId(deviceId);
      return requestCamera(deviceId);
    },
    [requestCamera]
  );

  useEffect(() => {
    refreshDevices().catch(() => {});
    return () => stopCamera();
  }, [refreshDevices, stopCamera]);

  return {
    stream,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    requestCamera,
    switchCamera,
    stopCamera,
    error
  };
}

