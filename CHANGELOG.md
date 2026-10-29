# Changelog

## 0.1.0-rc.1 (2026-10-29)

Code-only prerelease. Checkpoints, vocabularies, model binaries and datasets are
excluded from archives. Research asset permissions and full labeled video/live
evaluation remain open in issues #2 and #3.

### Added

- Reproducible source packaging, per-file release manifest, SHA-256 checksums,
  and archive tests that reject leaked model/data assets.
- Pinned MediaPipe model versions/generations with verified SHA-256 and official
  Apache-2.0 model-card attribution.
- Split/recording preparation and explicit training-compatible text scoring.
- Historical validation metadata extracted from the exact bundled checkpoint,
  explicitly distinguished from a new inference run.

- MIT license for original code and documentation, with separate research asset
  rights and provenance notes.
- Model card and evaluation guide distinguishing the final report's 646D model
  from this repository's 1662D Model 6.
- Corpus WER scoring, labeled-video evaluation and synthetic latency benchmark
  tools, plus scoring regression tests.
- An archived synthetic CPU benchmark with checkpoint/source fingerprints and
  measured timing scope; recognition accuracy remains unevaluated for Model 6.
- Contribution and maintainer guides, CODEOWNERS, issue and pull request
  templates, and backend/frontend CI checks.

### Changed

- Updated frontend tooling and transitive packages to address npm audit findings;
  Vite 8.3.5 and Vitest 4.1.12 are the minimum requested versions.

### Evidence still needed

- Matching 646D research artifacts, independently reproduced accuracy, real
  labeled Model 6 examples, complete application latency and asset permissions.

### Fixed

- Invalid upload metadata now returns HTTP 400 even when inference is
  unavailable; valid metadata still returns HTTP 503 without storing the upload.
- Linux CI installs MediaPipe's EGL/GL runtime libraries and explicitly tests
  readiness with the bundled model assets.
