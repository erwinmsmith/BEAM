# API reference

## `AgentBayesian`

Construct with `AgentBayesian(config)`, register all agents on an `AgentGraph`, and attach it using `set_graph(graph)`.

| Method | Behavior |
| --- | --- |
| `set_completeness(completeness)` | One retained-information coefficient in `[0,1]` per agent |
| `update_energy(contributions, costs)` | Clipped energy update from two bounded per-agent vectors |
| `probabilities(fitted=True)` | Returns node vector and off-diagonal edge matrix |
| `train(data, eval_fn, ...)` | Synchronous calibration |
| `await atrain(data, eval_fn, ...)` | Asynchronous calibration |
| `await sample_reference(utility, initial=None, transitions=None, burn_in=None)` | Retained fixed-target graph states |
| `await run(inputs, num_rounds=None, use_mean=True)` | Answers and deployment metadata |
| `run_sync(inputs, num_rounds=None)` | Synchronous deployment |
| `get_edge_statistics()` | Pair moments, probabilities, energies and completeness |
| `save(path)` / `load(path)` | Version-2 model and sampler state |

`use_mean=True` aggregates `deployment_samples` independent draws. `False` uses one draw. Metadata contains `node_mask`, `adjacency`, `active_nodes`, `active_edges`, `calibration_mcmc` and `deployment_samples`.

## `GraphState`

```python
from beam import GraphState

state = GraphState(
    nodes=(True, True, False),
    edges=((False, True, False), (False, False, False), (False, False, False)),
)
assert state.feasible()
```

Coordinates follow graph insertion order. `bits()` places the node bits first and off-diagonal edge bits second. `flip(coordinate)` constructs a one-bit proposal; `feasible()` checks its constraints.

## `AgentGraph` and `AgentNode`

```python
from beam import AgentConfig, AgentGraph, AgentNode, BEAMConfig

config = BEAMConfig(agents=[AgentConfig(name="Solver", count=2)], num_rounds=1)
graph = AgentGraph(config)
for i in range(2):
    graph.add_node(AgentNode(id=str(i), role=f"Solver {i}", execute_fn=lambda s: "4"))
```

An execution callback receives `task`, `context`, `spatial_info` and `temporal_info`. It can be synchronous or asynchronous. To use an LLM, pass a client with `gen` / `agen` and the node's system/user prompt templates.

`await graph.arun_state(inputs, state, num_rounds=None)` executes a selected feasible state exactly. `graph.run` is synchronous; `await graph.arun` is the separate logits-and-mask sampling path used by the pruning interface. Initialize potential edges before using that latter path.

## `TokenOptimizer`

With `strategy="bayesian"`, `train`, `run`, `run_sync`, `get_edge_statistics`, `save_state` and `load_state` delegate to the joint Bayesian strategy. Pass a complete registered graph to `TokenOptimizer(config, graph)`.

[Configuration reference](../getting-started/config.md) · [Prompt templates](../examples/prompts-examples.md)
