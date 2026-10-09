# Change Log

---


## 0.2.0 — Joint node/edge probability model

- Added bounded energy dynamics, Gaussian exceedance probabilities and inducing-variable GP moments.
- Added the fixed-target lazy coordinate MH sampler and KL-regularized covariance-gradient calibration.
- Deployment samples nodes and edges, repairs cycles deterministically and applies strict frequency consensus.
- Selected DAGs execute directly with request-local buffers; Bayesian routing now works through TokenOptimizer.
- Version-2 checkpoints preserve the GP, energy, optimizer and sampling state.
- Added offline tests and a runnable calibration/sampling/deployment example.
- Documentation publication is manually triggered; pushes to main run read-only tests without modifying gh-pages.
