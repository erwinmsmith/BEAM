# LangGraph

BEAM's selected adjacency can be mapped to named workflow edges using an explicit node order. Use the same order as `graph.nodes`.

```python
from beam.integrations.langgraph import apply_beam_masks

node_order = ["researcher", "solver", "reviewer"]
edges = [("researcher", "solver"), ("researcher", "reviewer"), ("solver", "reviewer")]
adjacency = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
selected = apply_beam_masks(edges, adjacency, node_order=node_order)
assert selected == [("researcher", "solver"), ("solver", "reviewer")]
```

Use `metadata["node_mask"]` to select participating workflow nodes and `metadata["adjacency"]` for their communication links. When using BEAM's own executor, `AgentGraph.arun_state` handles both directly.

## Helper interfaces

- `get_langgraph_config(num_agents, strategy="bayesian")` produces a dictionary accepted by `BEAMConfig.from_dict`.
- `create_beam_node(node_fn, role)` adapts a synchronous node function to the `BEAMState` convention.
- `create_skip_condition(skip_agents)` returns the routing labels `skip` and `continue`.
- `apply_beam_masks(edges, spatial_masks, node_order=None)` filters a list of existing edges. Named endpoints use an explicit node order; integer endpoints can index the mask directly.

Install framework support with `python -m pip install -e '.[langgraph]'`.
