# Changelog

## Documentation and Pages publishing

- Adopted standard MkDocs Material with Markdown pages, quick start and implementation referenceping.
- Updated API, configuration and integration examples to match the current implementation.
- Added strict static-site and documentation-example checks.
- Published through GitHub Pages artifacts from `main`.
- Implemented named-node adjacency filtering for the LangGraph adapter.

## 0.2.0 — Joint node/edge probability model

- Added bounded energy dynamics, Gaussian exceedance probabilities and inducing-variable GP moments.
- Added the fixed-target lazy coordinate MH sampler and KL-regularized covariance-gradient calibration.
- Deployment samples nodes and edges, repairs cycles deterministically and applies strict frequency consensus.
- Selected DAGs execute directly with request-local buffers; Bayesian routing works through `TokenOptimizer`.
- Version-2 checkpoints preserve the GP, energy, optimizer and sampling state.
- Added finite-state tests and a runnable calibration/sampling/deployment example.
- Added Python 3.9 and 3.12 CI coverage.
