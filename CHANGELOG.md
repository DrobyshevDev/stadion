# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Property-based tests for the paired bootstrap. `hypothesis` had been a dev
  dependency since the start without ever being imported; the tests state the
  invariants the scoring argument rests on — a verdict says what the interval
  says and nothing more, naming the arms in the other order negates the
  comparison, and shifting every return by a constant shifts the difference by
  that constant. Replacing the bootstrap with one that resamples the two arms
  independently leaves the example-based suite green and fails three of these.
- Build provenance on releases. `actions/attest-build-provenance` signs a SLSA
  statement naming the repository, workflow and commit that produced each
  distribution, and the distributions are attached to the GitHub release beside
  it, so `gh attestation verify <file> --repo DrobyshevDev/stadion` has both
  halves. PyPI-side PEP 740 attestations are unchanged.
- An OpenSSF Scorecard workflow and its badge — the one measurement here that
  this project does not produce about itself.
- A badge row in both READMEs. The repository previously carried none at all,
  which made it the only one in the organisation where a reader could not tell
  at a glance that the suite runs on three operating systems or that the package
  is on PyPI as `stadion-rl`.
- Coverage is measured in CI and reported to Codecov rather than discarded at
  the end of the job.
- `Documentation` link in the PyPI project metadata.

### Changed
- Every GitHub Action is pinned to a commit, with the tag kept as a trailing
  comment. `pypa/gh-action-pypi-publish@release/v1` was a *branch* — it moves by
  design — on the step that publishes to PyPI.

## [0.1.0] — 2026-08-09

### Added
- First release. A proving ground for operational decisions: an agent is scored
  against the classical operations-research method for the problem and against
  the exact optimum, on the same instances with the same episode seeds, and the
  result is a normalised score with a bootstrap confidence interval.
- Six tasks — inventory, joint pricing, pricing, queueing, supply chain and
  energy — each small enough that the optimum can be computed by backward
  induction rather than approximated.
- "Indistinguishable from the classical method" as a first-class outcome. Where
  the textbook rule is already optimal the normalised score has no denominator,
  and that is reported rather than papered over with an epsilon.

[Unreleased]: https://github.com/DrobyshevDev/stadion/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DrobyshevDev/stadion/releases/tag/v0.1.0
