# Strategy interfaces

| Interface | Optimization unit | Entry point |
| --- | --- | --- |
| `AgentBayesian` | Joint participation and communication edges | `train` / `atrain`, `run`, `sample_reference` |
| `AgentPrune` | Edge masks | `train`, `run`, `get_pruned_edges` |
| `AgentDropout` | Per-round agent skip decisions | `train`, `run`, `get_skip_statistics` |

## The BEAM paper implementation

`AgentBayesian` combines bounded energy, pairwise GP uncertainty, a KL-regularized graph objective and feasible DAG deployment. Use this interface for the manuscript's joint model.

```python
from beam.examples.theory_alignment import build_example

strategy = build_example()
strategy.set_completeness([1.0, 0.8, 0.6])
strategy.train(
    [{"task": "2 + 2", "answer": "4"}],
    lambda prediction, answer: float(prediction == answer),
    num_iterations=1,
)
answers, metadata = strategy.run_sync({"task": "2 + 2"})
```

`TokenOptimizer` dispatches `strategy="bayesian"` to this implementation, including training, execution and checkpoints. Direct use of `AgentBayesian` also exposes completeness, energy and reference-sampling controls.

## Other strategy classes

Instantiate `AgentPrune` or `AgentDropout` directly, supply a configured graph with registered nodes, and call `graph.init_potential_edges()` for their graph-construction path. These classes retain their separate configuration and persistence interfaces.

The manuscript's deterministic and node-only limits are mathematical restrictions of the joint probability model. The [paper-to-code map](paper-to-code.md) documents the concrete joint algorithms.
