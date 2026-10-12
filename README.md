<p align="center">
  <img src="assets/beamlogo.png" alt="BEAM Logo" width="400">
</p>

<h1 align="center">BEAM</h1>

<p align="center">
  <b>A Bayesian Energy-Aware Framework for Multi-Agent Communication Optimization under Incomplete Information</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

[Documentation](docs/index.md) · [Quick start](docs/getting-started/quickstart.md) · [Tests](tests/)

BEAM jointly models agent participation and communication links. Energy combines contribution and communication cost, pairwise Gaussian-process moments represent uncertainty, and task feedback with nuclear-norm, variance and GP prior regularization trains the graph distribution. Deployment constructs a feasible DAG and executes its selected agents and links.

## Run locally

Download and extract the repository, then open a terminal in its root directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m beam.examples.bayesian_workflow
```

The example runs calibration, fixed-target reference sampling and deployment on CPU with deterministic callbacks. It requires no API key. Python 3.9+ and PyTorch 2.0+ are supported.

## Algorithm interfaces

| Stage | Implementation | Purpose |
| --- | --- | --- |
| Observation and energy | `set_completeness`, `update_energy` | Retained-information coefficients and square-root energy updates |
| Score uncertainty | `SparsePairGP`, `probabilities` | Inducing-variable moments and Gaussian exceedance probabilities |
| Calibration | `train` / `atrain` | Node and centered edge scores with three regularizers |
| Reference chain | `sample_reference` | Fixed-mask single-edge Metropolis–Hastings |
| Deployment | `run` / `run_sync` | Gaussian edge draws, cycle repair and strict frequency consensus |
| Execution | `AgentGraph.arun_state` | Exact active-node and edge execution with isolated request buffers |

```python
import asyncio
from beam.examples.bayesian_workflow import build_example

async def main():
    strategy = build_example()
    stats = await strategy.atrain(
        [{"task": "What is 2 + 2?", "answer": "4"}],
        lambda prediction, answer: float(prediction == answer),
        num_iterations=1,
    )
    answers, graph = await strategy.run({"task": "What is 2 + 2?"})
    print(answers, graph["node_mask"], graph["adjacency"])

asyncio.run(main())
```

Use `AgentNode` callbacks or an LLM client to define agents. Register nodes with `graph.add_node(node)` in the intended state-coordinate order, then call `strategy.set_graph(graph)`. See the [API reference](docs/api-reference/apireference.md) for complete construction and persistence interfaces.

## Engineering checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python scripts/check_site.py
```

Tests cover all 25 DAGs on three active nodes and all node masks. They verify detailed balance, stationarity, the joint score gradient, GP covariance and KL, energy transitions, DAG execution, integration masks and checkpoint continuation. CI runs Python 3.9 and 3.12. The static documentation site deploys through GitHub Pages artifacts from `main`.

## Integrations and strategy interfaces

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-Integration-blue?style=for-the-badge" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Integration-purple?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

- **AgentBayesian** implements the joint model and algorithms above; `TokenOptimizer(config, graph)` delegates to it when `config.optimization.strategy="bayesian"`.
- **AgentPrune** and **AgentDropout** expose separate edge-pruning and agent-skipping interfaces.
- **LangChain** runnables can supply an agent's execution callback.
- **LangGraph** adapters map named nodes to explicit adjacency coordinates.

Optional integrations: `python -m pip install -e '.[langchain,langgraph]'`.

## License

[MIT](LICENSE).
