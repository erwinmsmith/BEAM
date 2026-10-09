<p align="center">
  <img src="assets/beamlogo.png" alt="BEAM Logo" width="400">
</p>

# BEAM

**A Bayesian Energy-Aware Framework for Multi-Agent Communication Optimization under Incomplete Information**

BEAM jointly selects agent participation and communication links. It combines task utility, communication cost, and observation uncertainty in a probabilistic graph, then executes the selected agents in a directed acyclic graph (DAG).

## What BEAM provides

- **Joint node–edge selection:** choose participating agents and their communication routes together.
- **Uncertainty-aware scoring:** bounded energy and sparse Gaussian-process moments inform selection probabilities.
- **Utility-based calibration:** optimize a KL-regularized objective with a covariance-gradient estimator.
- **Feasible execution:** deterministic cycle repair and edge-frequency consensus produce an executable DAG.
- **Framework integration:** use native asynchronous execution or connect graph masks to LangChain and LangGraph workflows.

## Quick start

This CPU example runs calibration, reference sampling, and deployment with deterministic callbacks. No API key is required.

```bash
git clone https://github.com/erwinmsmith/BEAM.git
cd BEAM
python -m pip install -e .
python -m beam.examples.bayesian_workflow
```

## Documentation

| Start here | Learn more |
| --- | --- |
| [Installation](getting-started/installation.md) | Set up Python and optional integrations |
| [Quick start](getting-started/quickstart.md) | Run a complete local example |
| [Probability model](models/beam-theory.md) | Follow the model and sampling algorithms |
| [Implementation reference](models/implementation.md) | Locate the implementation of each algorithm |
| [Configuration](getting-started/config.md) | Configure calibration, energy, and deployment |
| [API reference](api-reference/apireference.md) | Build and execute agent graphs |
| [Engineering](advanced.md) | Run tests and maintain these Markdown docs |

## Algorithm checks

The test suite enumerates all 38 feasible three-agent states and checks detailed balance, stationary mass, the covariance gradient, Gaussian-process conditional variance, and exact DAG execution.

These docs use [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Edit Markdown files in `docs/`; changes pushed to `main` are built and published by GitHub Actions.
