# Architecture

BEAM represents a communication state as a joint node mask and directed adjacency matrix. Participation decides which agents run; edges decide which same-round messages they receive. Feasible states have no self-loops or directed cycles, and every retained edge has active endpoints.

## The model

1. **Observe.** Supply each agent's retained-information fraction. Update its bounded energy from contribution and cost feedback.
2. **Represent uncertainty.** Pairwise GP moments feed Gaussian edge scores. Incident-edge probabilities and energy determine node probabilities.
3. **Calibrate.** Fit a constrained Bernoulli graph family using expected bounded utility minus KL divergence to the structural reference.
4. **Deploy.** Sample independent node/edge states, repair cycles and aggregate edge frequencies into a feasible DAG.
5. **Execute.** Run active agents in topological order, with per-request output and memory buffers.

## Package layout

| Module | Responsibility |
| --- | --- |
| `beam/core/probability.py` | Graph states, MH transitions, covariance loss, GP moments, cycle repair and consensus |
| `beam/strategies/bayesian.py` | Calibration, energy/observation updates, deployment and persistence |
| `beam/core/graph.py` | Agent registration and execution |
| `beam/core/config.py` | Dataclass configuration |
| `beam/core/optimizer.py` | High-level training and inference entry point |
| `beam/integrations/` | LangChain and LangGraph adapters |
| `tests/` | Finite-state, numerical and execution checks |

[Read the probability model](../models/beam-theory.md) or [run the example](quickstart.md).
