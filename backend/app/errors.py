from __future__ import annotations


class BdSLVisionError(Exception):
    status_code = 500
    error_code = "internal_error"
    user_message = "Recognition failed. Please try again."

    def __init__(self, message: str | None = None, *, detail: str | None = None):
        super().__init__(message or self.user_message)
        self.detail = detail


class InvalidUploadError(BdSLVisionError):
    status_code = 400
    error_code = "invalid_upload"
    user_message = "The uploaded file is not a valid recording."


class UploadTooLargeError(BdSLVisionError):
    status_code = 413
    error_code = "upload_too_large"
    user_message = "The recording is too large. Record a shorter sentence and try again."


class UnsupportedVideoError(BdSLVisionError):
    status_code = 415
    error_code = "unsupported_video"
    user_message = "We couldn't read this video format. Try recording again in the browser."


class EmptyVideoError(BdSLVisionError):
    status_code = 422
    error_code = "empty_video"
    user_message = "The recording did not contain readable video frames."


class RecordingTooShortError(BdSLVisionError):
    status_code = 422
    error_code = "recording_too_short"
    user_message = "The recording was too short. Perform one complete sentence and try again."


class SequenceTooLongError(BdSLVisionError):
    status_code = 422
    error_code = "sequence_too_long"
    user_message = "The recording is longer than Model 6 supports. Keep one sentence under 15 seconds."


class MediaPipeFailureError(BdSLVisionError):
    status_code = 422
    error_code = "mediapipe_failure"
    user_message = "We couldn't detect enough signing landmarks. Improve framing and try again."


class CheckpointFailureError(BdSLVisionError):
    status_code = 503
    error_code = "checkpoint_failure"
    user_message = "Model 6 checkpoint could not be loaded."


class VocabMismatchError(BdSLVisionError):
    status_code = 503
    error_code = "vocab_mismatch"
    user_message = "Model 6 vocabulary does not match the checkpoint."


class ModelNotReadyError(BdSLVisionError):
    status_code = 503
    error_code = "model_not_ready"
    user_message = "Recognition service is currently unavailable."


class NoCTCWordsError(BdSLVisionError):
    status_code = 422
    error_code = "no_ctc_words"
    user_message = "No confident sentence was decoded. Try again with clearer signing."

