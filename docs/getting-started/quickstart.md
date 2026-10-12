# Quick start

Run the complete offline workflow:

```bash
python -m beam.examples.bayesian_workflow
```

It builds three agents, sets their completeness coefficients, calibrates the joint graph distribution, draws 10 retained conditional graph states and executes a deployment DAG.

## Complete example

This code is included directly from the example exercised by CI.

```python
--8<-- "beam/examples/bayesian_workflow.py"
```

## Use your own task

Replace each agent's `execute_fn` with your implementation and supply calibration records containing `task` and `answer`. The evaluation function returns a finite score in `[0,1]`.

`atrain` is the asynchronous calibration entry point. In a synchronous script, use `train`. Call `run` for deployment and inspect `node_mask` and `adjacency` in its metadata.

[Configuration](config.md) · [Execution example](../examples/examples.md) · [Implementation reference](../models/implementation.md)
