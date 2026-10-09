# Implementation reference

BEAM combines joint graph selection, utility-based calibration and feasible DAG execution. The table below locates the core components and their regression checks.

| Component | Code | Check |
| --- | --- | --- |
| Feasible state `x=(z,A)` | `GraphState.feasible`, `topological_order` | Enumerate all 38 feasible three-agent states |
| Clipped positive energy | `AgentBayesian.update_energy` | Energy bounds and complete-observation limit |
| Coupling and Gaussian scores | `_refresh_scores`, `interior_cdf` | Gaussian exceedance probabilities and one completeness variance factor |
| Structural Bernoulli mass | `bernoulli_log_mass` | Absent-node and absent-edge factors |
| Inducing-variable joint | `SparsePairGP.fit`, `moments` | Full conditional covariance and positive inducing covariance |
| KL-regularized objective and gradient | `covariance_loss`, `_train_batch` | Finite differences of the exact constrained objective |
| Fixed-target MH | `CoordinateSampler`, `sample_reference` | Detailed balance, stationarity, lazy transitions and multi-step minorization |
| Deployment | `deployment_state`, `repair_state` | Candidate support, endpoint validity, deterministic cycle repair, strict thresholds |
| Selected-state execution | `AgentGraph.arun_state` | Exact edges, active nodes, round override and request isolation |

## Calibration and deployment

The reference chain targets `p0(x) exp(beta*u(x))` on feasible joint states. Variational calibration samples the current constrained graph family and uses the centered covariance gradient. Deployment draws independent nodes and candidate edges, repairs cycles, then forms an edge-frequency consensus. These stages have distinct entry points and explicit probability laws.

The source implements the same strict consensus threshold at both boundaries: a frequency must exceed zero to create an edge when the threshold is zero, and no edge survives a threshold of one. Cycle repair deletes edges while preserving active endpoints.

## Sources and tests

- [Probability primitives](https://github.com/erwinmsmith/BEAM/blob/main/beam/core/probability.py)
- [Bayesian strategy](https://github.com/erwinmsmith/BEAM/blob/main/beam/strategies/bayesian.py)
- [Graph execution](https://github.com/erwinmsmith/BEAM/blob/main/beam/core/graph.py)
- [Finite-state and numerical tests](https://github.com/erwinmsmith/BEAM/blob/main/tests/test_probability.py)
- [Calibration and execution tests](https://github.com/erwinmsmith/BEAM/blob/main/tests/test_bayesian.py)

[Detailed equations and implementation choices](beam-theory.md)
