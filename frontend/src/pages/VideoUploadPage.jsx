import { AlertTriangle, FileVideo, UploadCloud, X } from 'lucide-react';
import { useRef, useState } from 'react';
import AnimatedPage from '../components/AnimatedPage.jsx';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { showToast } from '../components/ToastHost.jsx';
import DiagnosticsPanel from '../features/recognition/DiagnosticsPanel.jsx';
import ProcessingPanel from '../features/recognition/ProcessingPanel.jsx';
import ResultPanel from '../features/recognition/ResultPanel.jsx';
import { inferVideo } from '../lib/api.js';
import { formatFileSize, isSupportedVideoFile, MAX_UPLOAD_BYTES, VIDEO_ACCEPT } from '../lib/videoFiles.js';

export default function VideoUploadPage() {
  const fileInputRef = useRef(null);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [result, setResult] = useState(null);
  const [phase, setPhase] = useState('');
  const [busy, setBusy] = useState(false);

  const selectUploadedVideo = (file) => {
    setResult(null);
    if (!file) return;
    if (!isSupportedVideoFile(file)) {
      setUploadedVideo(null);
      setUploadError('Upload a video file in MP4, WebM, MOV, M4V, or AVI format.');
      showToast({ type: 'warning', message: 'Unsupported video format' });
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadedVideo(null);
      setUploadError(`Video must be ${formatFileSize(MAX_UPLOAD_BYTES)} or smaller.`);
      showToast({ type: 'warning', message: 'Video file is too large' });
      return;
    }

    setUploadedVideo(file);
    setUploadError('');
    showToast({ type: 'success', message: 'Video ready for translation' });
  };

  const removeUploadedVideo = () => {
    setUploadedVideo(null);
    setUploadError('');
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const translateUploadedVideo = async () => {
    if (!uploadedVideo) {
      setUploadError('Choose a sign-language video first.');
      return;
    }

    setBusy(true);
    setPhase('uploading');
    setUploadError('');
    setResult(null);
    try {
      const payload = await inferVideo(uploadedVideo, {
        saveHistory: true,
        onPhase: (nextPhase) => setPhase(nextPhase),
      });
      setResult(payload);
      showToast({ type: 'success', message: 'Translation complete' });
    } catch (error) {
      setUploadError(error.message || 'Recognition service is currently unavailable.');
      showToast({ type: 'error', message: error.message || 'Video translation failed' });
    } finally {
      setBusy(false);
      setPhase('');
    }
  };

  return (
    <AnimatedPage>
      <main className="page page-narrow">
        <PageHeader eyebrow="Video File Translation" title="Upload BdSL Video">
          Upload a completed Bangla sign-language sentence clip and decode it with Model 6.
        </PageHeader>

        <div className="upload-page-layout">
          <section className="upload-panel">
            <div className="panel-topline">
              <div>
                <p className="eyebrow">Video Upload</p>
                <h2>Translate Existing Clip</h2>
                <p className="subtle">MP4, WebM, MOV, M4V, or AVI up to {formatFileSize(MAX_UPLOAD_BYTES)}.</p>
              </div>
              <StatusBadge tone={uploadedVideo ? 'success' : 'ready'}>
                {uploadedVideo ? 'Video selected' : 'Waiting'}
              </StatusBadge>
            </div>

            <label className={`upload-dropzone${uploadedVideo ? ' has-file' : ''}${busy ? ' disabled' : ''}`}>
              <input
                ref={fileInputRef}
                type="file"
                accept={VIDEO_ACCEPT}
                onChange={(event) => selectUploadedVideo(event.target.files?.[0])}
                disabled={busy}
              />
              <span className="upload-icon" aria-hidden="true">
                <FileVideo size={28} />
              </span>
              <strong title={uploadedVideo?.name || undefined}>
                {uploadedVideo ? uploadedVideo.name : 'Choose a BdSL video'}
              </strong>
              <span>
                {uploadedVideo
                  ? `${formatFileSize(uploadedVideo.size)} ready for sentence recognition`
                  : 'Drop in a completed sign sentence clip'}
              </span>
            </label>

            <div className="upload-actions">
              <button type="button" className="button primary" onClick={translateUploadedVideo} disabled={busy || !uploadedVideo}>
                <UploadCloud size={18} aria-hidden="true" />
                Translate Video
              </button>
              <button type="button" className="button ghost icon-button" onClick={removeUploadedVideo} disabled={busy} aria-label="Remove selected video" data-tooltip="Remove video">
                <X size={18} aria-hidden="true" />
              </button>
            </div>

            {uploadError ? (
              <p className="field-error" role="alert">
                <AlertTriangle size={15} aria-hidden="true" />
                {uploadError}
              </p>
            ) : null}
          </section>

          {busy ? <ProcessingPanel phase={phase} /> : null}
          <ResultPanel result={result} onTryAgain={removeUploadedVideo} />
          <DiagnosticsPanel result={result} />
        </div>
      </main>
    </AnimatedPage>
  );
}
