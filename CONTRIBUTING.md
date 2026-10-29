# Contributing to BdSL Vision

Bug reports, documentation corrections, reproducible evaluations, accessibility
improvements and focused code changes are welcome. Start with the
[README](README.md), [model card](docs/MODEL_CARD.md), [evaluation guide](docs/EVALUATION.md)
and [roadmap](docs/ROADMAP.md). Use the repository's issue templates to describe
work; discuss changes to the model or feature contract before implementation.

## Local workflow

1. Fork or clone the repository and create a descriptive branch.
2. Set up Python 3.12 and Node.js 22.12+ using the README instructions.
3. Make a focused change with a clear explanation of the observed problem.
4. Run the checks below and include commands, outcomes and any skipped checks
   in the pull request. Add regression coverage for changed behavior where useful.
5. Open a pull request and link the issue it addresses. Keep generated artifacts,
   dependency folders, private recordings, credentials and history out of commits.

From the repository root, with the backend virtual environment active:

```bash
python -m pytest backend/tests
```

From `frontend/`:

```bash
npm ci
npm test
npm run build
```

The backend suite needs `backend/requirements.txt`; some tests exercise real
checkpoint loading and MediaPipe startup. The lightweight scoring tests can
also run with only pytest: `python -m pytest backend/tests/test_evaluation.py`.
No camera or private dataset is required for unit tests. Mocked API responses
validate plumbing and must never be described as model-quality evidence.

## Evaluation and asset changes

Preserve the `[T,1662]`, vocabulary and CTC contract unless a change explicitly
introduces a different version with compatible checkpoints and tests. Document
hashes, preprocessing, split ownership, scoring conventions, hardware and failure
counts for model comparisons. Never assign the report's 646D results to Model 6.
Do not tune on the held-out test split or publish only successful clips.

Only submit code/assets you have permission to contribute. Original project code
contributions are under the [MIT license](LICENSE); third-party and research
assets retain separate terms described in [NOTICE.md](NOTICE.md). Include source,
license and permission evidence with any proposed weight/data addition. Do not
include identifiable signing recordings without permission to publish them.

## Review

The maintainer checks scope, correctness, tests, documentation, asset permissions
and evidence before merging. Contributors can ask for clarification in the
linked issue or pull request. Treat other contributors and BdSL users
respectfully, avoid assumptions about signing ability, and welcome corrections
from people with lived language expertise. See [MAINTAINERS.md](MAINTAINERS.md)
for ownership and review responsibilities.
