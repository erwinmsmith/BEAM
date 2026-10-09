# Execute a selected graph

Use `arun_state` to execute an explicit DAG with request-local buffers.

```python
import asyncio
from beam import AgentConfig, AgentGraph, AgentNode, BEAMConfig, GraphState

config = BEAMConfig(agents=[AgentConfig(name="Agent", count=2)], num_rounds=1)
graph = AgentGraph(config)
graph.add_node(AgentNode(id="solver", role="Solver", execute_fn=lambda s: "4"))
graph.add_node(AgentNode(
    id="reader", role="Reader",
    execute_fn=lambda s: s["spatial_info"]["solver"]["output"],
))
state = GraphState((True, True), ((False, True), (False, False)))
answers, _ = asyncio.run(graph.arun_state({"task": "2 + 2"}, state))
assert answers == ["4"]
```

Both nodes are active, and the reader receives the solver's answer through the selected edge. Set a node bit to false and remove its incident edges to exclude it from execution. An empty state returns an empty answer list.

For more than one round, the configured temporal mask connects previous-round memory between active agents. The spatial state remains fixed for the execution call.
