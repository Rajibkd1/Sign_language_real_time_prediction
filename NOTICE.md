# License scope and research assets

The [MIT license](LICENSE) applies to original BdSL Vision source code and
repository documentation. It does not relicense third-party dependencies,
model weights, vocabulary/data assets, research recordings, or the author's
separate university report. Existing third-party notices remain applicable.

## Assets requiring separate review

| Asset | Source and current evidence | License status in this repository |
| --- | --- | --- |
| `model6_best_wer_checkpoint.pt` and `backend/models/model6_best_wer_checkpoint.pt` | Bundled Model 6 checkpoint; training provenance must be supplied by the maintainer | Excluded from MIT; no separate weight license or redistribution permission has been verified |
| `bangla_vocab_tsv_fixed.json` and `backend/models/bangla_vocab_tsv_fixed.json` | Bundled 39-class vocabulary paired with Model 6 | Excluded from MIT pending provenance and rights review |
| `backend/assets/mediapipe/*.task` | Google MediaPipe task bundles; download URLs are in `backend/scripts/download_mediapipe_models.py` | Third-party assets, excluded from MIT; verify the exact bundle's upstream terms before redistribution |
| Ban-Sign-Sent-9K-V1 videos, annotations and derived features | [Publisher's dataset](https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1) | No explicit dataset license was visible in the card or root file listing checked on 2026-10-29; public download availability does not establish reuse permission |
| Final-year report | Rajib Kumar Dhar, *Developing a Real-Time Vision-Based System for Continuous Bangla Sign Language Translation*, September 1, 2026 | Used as a supplied research source; full document is not redistributed here |

The dataset card lists 1,922 sentences and 9,610 videos. It does not establish
which dataset or permissions were used for the bundled older Model 6 checkpoint.
Do not infer that provenance from the later report.

MediaPipe software and downloaded task weights are separate artifacts. See the
upstream [software license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
and the [pose](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker),
[hand](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker),
and [face](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
model documentation. The software license alone is not recorded here as proof
of the exact binary bundles' permissions.

## Maintainer follow-up

Record each asset's immutable source revision, SHA-256, copyright holder,
license text, and evidence of distribution/use permission. For training data,
also record consent and derivative-model restrictions where applicable.
Until that audit is resolved, this repository offers MIT-licensed application
code with separately unresolved research assets, not a fully cleared model or
dataset release. No broader rights are granted by adding the code license.
