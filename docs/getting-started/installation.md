# Installation

Use Python 3.9 or later. The core implementation uses PyTorch, NumPy and the package dependencies declared in `pyproject.toml`.

```bash
git clone https://github.com/erwinmsmith/BEAM.git
cd BEAM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Optional tools

```bash
# Development and tests
python -m pip install -e '.[dev]'

# Framework adapters
python -m pip install -e '.[langchain,langgraph]'

# Static documentation
python -m pip install -r requirements-docs.txt
```

The `bayesian` installation extra remains an alias for the core implementation. Both the discrete sampler and GP use PyTorch.

## Verify the installation

```bash
python -m beam.examples.bayesian_workflow
python -m pytest -q
```

The example uses deterministic agent callbacks. To connect a live model, provide your own LLM client to `AgentNode` or use a [LangChain runnable](../models/LangChain.md). Keep API keys in environment variables.
