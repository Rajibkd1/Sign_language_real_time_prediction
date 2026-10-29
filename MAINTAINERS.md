# Maintainers and project stewardship

## Current contact

[Rajib Kumar Dhar (@Rajibkd1)](https://github.com/Rajibkd1) is the repository
owner and project contact. The supplied final-year report identifies him as
the research author. The project originated as a final-year software engineering
project at the Institute of Information Technology, Noakhali Science and
Technology University. Institutional affiliation does not imply institutional
ownership approval, endorsement or production support.

Use [GitHub issues](https://github.com/Rajibkd1/Sign_language_real_time_prediction/issues)
for reproducible bugs, evaluation requests and contribution proposals. No response
time or release schedule is promised. Activity is visible in public commits,
issues and pull requests; this document does not establish a historical cadence
or claim external adoption.

## Responsibilities

The maintainer role covers issue triage, code review, dependency updates,
regression testing, documentation, model/preprocessing compatibility, evaluation
provenance and asset-rights review. Rights to code, training data and model
weights are reviewed separately. Significant model changes require a compatible
checkpoint, explicit version/feature contract and new evaluation evidence.

Routine fixes should link a concrete problem to a tested behavior change.
Model results must identify the evaluated files and environment, and explain
remaining limitations. Unsupported metrics or user counts must not be added to
the README or funding applications.

## Review and release procedure

1. Reproduce the issue, define the change and check overlapping work.
2. Review source and documentation; run the checks in [CONTRIBUTING.md](CONTRIBUTING.md).
3. For model/data changes, verify compatibility, permission evidence and
   evaluation output before distributing assets or making new accuracy claims.
4. Record notable changes in [CHANGELOG.md](CHANGELOG.md). Release notes should
   identify the commit, supported environment and known limitations.
5. Create a tagged release only after its checks pass. A roadmap entry or merged
   documentation update is not by itself a release or a commitment to ship.

CI checks and an issue roadmap support future maintenance. They cannot create
a genuine record of regular maintenance retroactively. New maintainers should
be listed here and in CODEOWNERS only after they accept the role and receive
the corresponding repository access.
