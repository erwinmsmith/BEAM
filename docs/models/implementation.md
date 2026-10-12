# Implementation reference

BEAM combines joint graph selection, utility-based calibration and feasible DAG execution. The table below locates the core components and their regression checks.

| Component | Code | Check |
| --- | --- | --- |
| Feasible graph | `GraphState` | All 25 DAGs on three active nodes |
| Energy and coupling | `update_energy`, `coupling` | Exact seeded transition and positive-domain handling |
| Gaussian node/edge probabilities | `_score_moments`, `_probabilities` | Mean, variance and exceedance formulas |
| Conditional graph mass | `conditional_log_mass` | Active-pair factors and zero inactive-pair gradients |
| Variational GP | `SparsePairGP` | Dense covariance and Gaussian KL formulas |
| Task and regularizers | `score_loss`, `_regularization` | Exact joint gradient over all node masks and DAGs |
| Single-edge MH | `CoordinateSampler` | Detailed balance, stationarity, multi-step mixing and retained rejections |
| Deployment | `deployment_state`, `repair_state` | Fixed mask, mean-based repair and strict frequency thresholds |
| Execution and persistence | `arun_state`, `save`, `load` | Request isolation and identical checkpoint continuation |

## Calibration and deployment

Calibration first draws a node mask, then samples the conditional DAG distribution by single-edge Metropolis–Hastings. Task utility drives the node score and centered edge-score update. Nuclear-norm, variance and inducing-variable KL penalties complete the objective.

Deployment keeps one active set across Gaussian edge draws, removes the lowest-mean edge in each cycle and forms a strict frequency consensus. Final cycle repair uses edge frequencies.

## Sources and tests

- [Probability primitives](https://github.com/erwinmsmith/BEAM/blob/main/beam/core/probability.py)
- [Bayesian strategy](https://github.com/erwinmsmith/BEAM/blob/main/beam/strategies/bayesian.py)
- [Graph execution](https://github.com/erwinmsmith/BEAM/blob/main/beam/core/graph.py)
- [Finite-state and numerical tests](https://github.com/erwinmsmith/BEAM/blob/main/tests/test_probability.py)
- [Calibration and execution tests](https://github.com/erwinmsmith/BEAM/blob/main/tests/test_bayesian.py)

[Detailed equations and implementation choices](beam-theory.md)
