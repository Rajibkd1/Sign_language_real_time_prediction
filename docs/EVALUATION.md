# Evaluation and reproducibility

BdSL Vision currently contains the **1662D Model 6 application**. The supplied
final-year report evaluates a **different 646D proposed model**. The report's
recognition results are historical, author-reported evidence and have not been
reproduced from this checkout. They must not be advertised as the accuracy of
the bundled Model 6 checkpoint.

## Results reported in the final-year study

Source: Rajib Kumar Dhar, *Developing a Real-Time Vision-Based System for
Continuous Bangla Sign Language Translation*, final-year report, Institute of
Information Technology, Noakhali Science and Technology University, September
1, 2026. Tables 7.2, 7.3, 7.4 and 7.6 and Sections 3.2, 3.3 and 8.2 are the
basis of this summary. The user supplied the DOCX for inspection; the full report
and its personal/administrative pages are not included in this repository.

Supplied source DOCX SHA-256:
`782186c6a0f3dbeff7eecf3ba4904ec1f31644941c375d6dfb4331d7789e1325`.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Videos | 960 | 480 |
| Sentence groups | 192 | 96 |
| Reference words | 4,360 | 2,175 |
| Word substitutions / deletions / insertions | 68 / 0 / 1 | 58 / 3 / 1 |
| WER | 1.582569% | 2.850575% |
| CER | 0.931099% | 1.901675% |
| Exact sentence accuracy | 92.812500% (891/960) | 87.500000% (420/480) |

The study reports Ban-Sign-Sent-9K-V1, with 8,170 training videos in 1,634
sentence groups, a seed-42 sentence-group-preserving 85/10/5 split, train-only
vocabulary, and EMA checkpoint selection at epoch 23 using validation WER.
All five recordings of each base sentence remain in one split. These split
manifests, training code, per-sample predictions, and the matching 646D checkpoint
are not present in this checkout, so the split and scores cannot yet be rerun here.

The report records 4,129,350 parameters and 1.085 ms/sequence for model plus
decoder (1.050 ms forward, 0.035 ms decode) in a Kaggle session with two T4 GPUs
available. The benchmark's full procedure is not included here. **That timing
excludes video decoding, MediaPipe, transfers, browser work and sentence-end
waiting**, and is not current application latency or proof of real-time operation.
The report's T=512 GFLOP estimate is a standardized complexity measurement;
its prepared inference sequence cap is 384 frames.

## Why these results do not identify this checkpoint

| Contract | Final report's proposed model | Current repository |
| --- | --- | --- |
| Input per frame | 646: pose, 128 semantic face points, hands, four presence flags | 1662: pose, 468 face points, hands |
| Parameters | 4,129,350 | 4,924,163 with 39 classes |
| Encoder | RoPE, RMSNorm, SwiGLU; auxiliary CTC | Sinusoidal positions; PyTorch TransformerEncoder, GELU/LayerNorm |
| Class vocabulary | Train-only CTC words; PAD/SOS/EOS excluded | 39 classes including special tokens in the supplied mapping |
| Identity | M5 in the report's 13-model comparison | Named Model 6 in code/checkpoint |

The report's M6 comparison row is an Enhanced Pure Transformer, so the number
"6" alone does not establish a match either. Feature conversion is not enough
to make incompatible weights or experimental results interchangeable.

## Current application evidence

- Recognition WER and real labeled prediction examples: **not yet measured
  from an available labeled video set in this checkout**.
- Full video-pipeline and browser/live latency: **not yet measured**.
- A reproducible synthetic-input Model 6 latency runner is provided below.
  It measures execution cost, not recognition quality. Any published run belongs
  to its recorded hardware, checkpoint hash and configuration only.
- Automated unit/API tests validate behavior and contracts; mocked responses in
  tests are not real prediction examples.

## Measured synthetic CPU latency

The [2026-10-29 CI run](https://github.com/Rajibkd1/Sign_language_real_time_prediction/actions/runs/40878967048)
passed 42 backend tests, 9 frontend tests, the frontend build and the checkpoint
benchmark at code revision `e514b7b7835316934d029bda39a45fc7909ba024`.
The [archived raw JSON](evaluation/model6-synthetic-cpu-2026-10-29.json) records
the exact checkpoint, vocabulary and Python source hashes.

Environment: Linux GitHub runner, AMD EPYC 9V45 CPU, Python 3.12.15, PyTorch
2.15.0+cpu, NumPy 2.5.4, one PyTorch thread, batch size one, FP32 CPU execution.
Each length used five warm-up calls and 30 measured calls with all-zero
float32 `[T,1662]` input.

| Frames | Forward p50 | Forward p95 | Predict wall p50 | Predict wall p95 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 10.650 ms | 10.962 ms | 10.873 ms | 11.196 ms |
| 384 | 33.945 ms | 40.399 ms | 34.221 ms | 40.678 ms |

`predict` wall time includes tensor preparation, model forward, greedy CTC,
confidence and cleanup. These measurements exclude video decoding, MediaPipe,
network transfer, browser overhead, startup and sentence-end waiting. This is
one synthetic execution benchmark, with no recognition accuracy or real signing
examples measured. Shared runner performance can vary; these numbers are not a
claim about the maintainer's laptop or a user's full application latency.

## Evaluate labeled videos

Use the backend environment from the README. Create a private UTF-8 JSONL file,
with one record per complete labeled clip. Video paths are relative to the
manifest. `group_id` identifies a base sentence, shared by all its recordings.

```json
{"id":"clip-001","video":"clips/clip-001.mp4","reference":"আমি ভাত খাব","group_id":"sentence-001"}
```

This line demonstrates the input format; it is **not a real model prediction**.
Use transcripts checked by a qualified BdSL annotator, record dataset revision
and annotation method, and obtain permission before sharing videos or transcripts.

From the repository root:

```bash
python backend/scripts/evaluate_videos.py evaluation-private/test.jsonl --split test --exclude-manifest evaluation-private/train.jsonl --exclude-manifest evaluation-private/validation.jsonl --output evaluation-results/model6-test.json
```

The script checks sample IDs, files, and sentence-group overlap against supplied
exclusions before loading the model. All manifests use the same schema. The
`--split` name is a user-supplied label; it does not prove a held-out split. The
script does not establish signer independence, source-wide deduplication, or
alignment with the report's original split. Keep test groups out of training,
tuning and checkpoint selection, and publish the exclusion manifest hashes.

Corpus WER is `100 * (S + D + I) / total reference words`, rather than the mean
of per-clip WER. Text uses Unicode NFC and collapsed whitespace, preserving
punctuation and case. This explicit scoring convention may differ from the
report's preprocessing. Exact sentence accuracy uses the same normalization.
Equal-cost edit alignments prefer substitution, then deletion, then insertion.

Failed inference is saved with an empty prediction, counts all reference words
as deletions, and makes the command exit with code 1 **after writing the report**.
Failures are never silently removed from WER. Missing files or invalid manifests
abort before evaluation. The output contains every reference, prediction/error,
video hash, selected response diagnostics, model/vocabulary/task hashes, runtime
settings, environment and Git revision. Review local paths and identifying text
before publishing an approved copy in `docs/evaluation/`.

`pipeline_wall_latency` includes video decode, MediaPipe, candidate predictions
and CTC within `predict_video`. Startup and HTTP/browser time are excluded. The
response's `model_ms` times only the selected candidate's forward; upload logic
may try multiple candidates. Record this distinction when comparing timings.
Run several independent sessions and report failure counts, p50/p95, hardware
and sample lengths. The runner leaves settings intact; disable inference artifact
saving if you do not intend to retain derived landmarks.

## Benchmark model execution

```bash
python backend/scripts/benchmark_model.py --frames 128 384 --warmup 5 --runs 30 --threads 1 --output evaluation-results/model6-latency.json
```

The runner uses batch size one and all-zero float32 `[T,1662]` inputs, records
checkpoint/vocabulary hashes and environment, warms up each length, synchronizes
CUDA when applicable, and reports nearest-rank p50/p95 for forward and whole
`ModelRuntime.predict` wall time. The latter includes tensor preparation, greedy
CTC, confidence calculation and cleanup. Both exclude video processing and user
waiting. Synthetic outputs are intentionally not presented as signed examples.

The `Checks` GitHub Actions workflow also runs this benchmark on its Linux
runner and retains the JSON as a `model6-synthetic-latency` artifact for 14 days.
Download a run's artifact to inspect its exact hardware/software and source
fingerprints. Runner hardware can vary across runs. Archive reviewed results
in `docs/evaluation/` when a lasting record is needed.

## Limitations and remaining evidence

The encoder uses a complete clip and is non-causal. Live segmentation waits
about 0.9 seconds of low motion before finalization; model milliseconds alone
cannot describe the user's waiting time. The vocabulary is closed, confidence
is an uncalibrated heuristic, and lighting, occlusion, framing, signer variation
and landmark failures can affect output. Upload candidate selection can prefer
longer decoded word sequences and needs labeled evaluation for insertion bias.

The report uses single saved runs without multi-seed uncertainty, a single
dataset and incomplete signer metadata. Neither the report nor this update
establishes signer-independent, cross-domain or production performance. See
[the model card](MODEL_CARD.md), [asset rights](../NOTICE.md), and the
[roadmap](ROADMAP.md) for the work required to close these gaps.
