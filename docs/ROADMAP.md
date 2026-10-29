# Maintainer roadmap

This roadmap tracks evidence and engineering gaps found while comparing the
current application with the final-year report. These are open tasks, without
promised completion dates.

## Align research and deployed model

Tracked in [issue #1](https://github.com/Rajibkd1/Sign_language_real_time_prediction/issues/1).

The report's proposed architecture uses 646 features, RoPE/RMSNorm/SwiGLU and
4,129,350 parameters. The app uses 1662-feature Model 6 with 4,924,163 parameters.
Obtain the original research preprocessing schema, matching checkpoint,
vocabulary, training code and split manifests. Either preserve both as clearly
versioned models or implement a validated migration. Do not silently reshape
features or relabel Model 6 results.

Acceptance: an explicit model version, exact hashes, compatible loader and tests,
matching preprocessing, and a repeatable evaluation with documented provenance.

## Resolve research asset permissions

Tracked in [issue #2](https://github.com/Rajibkd1/Sign_language_real_time_prediction/issues/2).

Obtain explicit licensing and distribution evidence for the bundled Model 6
weights/vocabulary and the training dataset. Verify exact MediaPipe task bundle
terms and record immutable sources. Replace or stop distributing any artifact
whose required permission cannot be established. The MIT code license does
not clear these assets.

Acceptance: a reviewed asset inventory with source revisions, hashes, license
texts, attribution and use/redistribution permissions.

## Publish current-checkpoint video evidence

Tracked in [issue #3](https://github.com/Rajibkd1/Sign_language_real_time_prediction/issues/3).

Use `backend/scripts/evaluate_videos.py` on consented, labeled clips with
disjoint sentence groups and documented annotation. Publish all predictions
including failures, WER, exact sentence accuracy, and pipeline latency p50/p95.
Evaluate the insertion effect of the current multi-candidate decode heuristic.
Add a separate live-camera benchmark including capture, upload and sentence-end
waiting. Include changed lighting/framing and new signers where permission and
identity metadata allow. Keep exploratory demos separate from held-out testing.

Acceptance: public, permission-reviewed results with checkpoint, vocabulary,
manifest and source hashes; real successful and failed examples; hardware,
scoring convention and limitations. A synthetic benchmark cannot close this task.
