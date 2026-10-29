# BdSL Vision Model 6

## Intended use

Research and development of Bangla Sign Language sentence recognition from
completed video clips or browser camera segments. The application returns a
closed-vocabulary Bangla word sequence with greedy word-level CTC. Output needs
human checking; the project has not established accuracy for use as a sole
interpreter in healthcare, legal, emergency, or other consequential decisions.

## Model identity and interface

- Implementation: `backend/app/model/model6.py`.
- Default weights: `backend/models/model6_best_wer_checkpoint.pt`.
- Vocabulary: `backend/models/bangla_vocab_tsv_fixed.json`, 39 classes, blank 0.
- Input: float32 `[T,1662]`, pose 132, face 1404, left hand 63, right hand 63.
- Temporal preparation: 25 FPS target, at most 384 frames, stride two in model.
- Architecture: raw/velocity regional projection, learned gate, temporal depthwise
  convolutions, sinusoidal position encoding, four Transformer encoder layers,
  final CTC classifier; 4,924,163 trainable parameters with this vocabulary.
- Runtime prefers EMA state when available; CUDA and autocast are optional.
- The loader handles full PyTorch checkpoint objects. Load only trusted
  checkpoints; a `.pt` filename is not a trust boundary.

The repository includes duplicate root-level checkpoint and vocabulary files for
the standalone webcam script. The backend paths above are authoritative for the
web application. Evaluation outputs fingerprint the actual files used.

## Training and evaluation provenance

The training corpus version, split manifests, signer composition, consent record,
training configuration and per-sample evaluation for this bundled checkpoint are
not supplied in this checkout. Do not assume that the later final-year report
describes this model. Its proposed model uses 646 features and 4,129,350
parameters; [evaluation documentation](EVALUATION.md) explains the discrepancy
and attributes its historical results separately.

Current-checkpoint WER, CER, exact sentence accuracy, real labeled examples,
cross-signer robustness and complete application latency remain unverified.
The new evaluation runner can record actual predictions once appropriately
licensed, labeled clips are available. Synthetic latency measurements and
passing tests do not establish recognition quality.

## Failure modes

Out-of-vocabulary signs cannot be represented reliably. This is a full-sequence,
non-causal encoder, with heuristic live sentence boundaries and uncalibrated
confidence. Missed hands, occlusion, lighting, blur, camera angle and signing
style can alter the feature sequence. Greedy CTC has no separate grammatical
reordering or language-model rescoring. Multi-candidate upload decoding may
favor longer output; the effect on insertions needs measurement.

## Privacy and rights

Camera frames go to the configured backend, which may be on another computer.
History can store predicted text and diagnostics; optional artifact saving can
retain landmark arrays. Raw uploaded temporary video is deleted after inference.
Do not commit user recordings, local history, features or identifying transcripts.

Code is MIT licensed. Weight, vocabulary, dataset and MediaPipe bundle rights
are separate and have unresolved provenance checks. See [NOTICE.md](../NOTICE.md).
