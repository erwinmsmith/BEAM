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

[Documentation](https://erwinmsmith.github.io/BEAM/) · [Quick start](https://erwinmsmith.github.io/BEAM/getting-started/quickstart/) · [Tests](https://github.com/erwinmsmith/BEAM/actions/workflows/tests.yml)

BEAM jointly models agent participation and communication links. Bounded energy combines contribution and cost, pairwise Gaussian-process moments represent uncertainty, and a KL-regularized objective calibrates the graph distribution. Deployment constructs a feasible DAG and executes its selected agents and links.

## Run locally

```bash
git clone https://github.com/erwinmsmith/BEAM.git
cd BEAM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m beam.examples.bayesian_workflow
```

The example runs calibration, fixed-target reference sampling and deployment on CPU with deterministic callbacks. It requires no API key. Python 3.9+ and PyTorch 2.0+ are supported.

## Algorithm interfaces

| Stage | Implementation | Purpose |
| --- | --- | --- |
| Observation and energy | `set_completeness`, `update_energy` | Retained-information coefficients and clipped positive energy |
| Score uncertainty | `SparsePairGP`, `probabilities` | Inducing-variable moments and Gaussian exceedance probabilities |
| Calibration | `train` / `atrain` | Centered gradient of expected utility minus KL regularization |
| Reference chain | `sample_reference` | Fixed-target, lazy coordinate Metropolis–Hastings |
| Deployment | `run` / `run_sync` | Independent graph draws, cycle repair and strict frequency consensus |
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

Use `AgentNode` callbacks or an LLM client to define agents. Register nodes with `graph.add_node(node)` in the intended state-coordinate order, then call `strategy.set_graph(graph)`. See the [API reference](https://erwinmsmith.github.io/BEAM/api-reference/apireference/) for complete construction and persistence interfaces.

## Engineering checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python scripts/check_site.py
```

Tests enumerate all 38 feasible three-agent states and verify detailed balance, stationarity, multi-step minorization, the covariance gradient, GP conditional variance, energy bounds, DAG execution, integration masks and checkpoint continuation. CI runs Python 3.9 and 3.12. The static documentation site deploys through GitHub Pages artifacts from `main`.

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
