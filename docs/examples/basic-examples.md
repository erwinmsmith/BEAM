# Offline calibration and deployment

Run:

```bash
python -m beam.examples.bayesian_workflow
```

The example uses three deterministic solver callbacks and an aggregation callback. It prints the calibration score, number of retained reference states, final answer and deployment metadata.

## Read the selected graph

```python
import asyncio
from beam.examples.bayesian_workflow import build_example

strategy = build_example()
answers, metadata = asyncio.run(strategy.run({"task": "What is 2 + 2?"}))
print(metadata["node_mask"])
print(metadata["adjacency"])
```

The boolean mask records participation, including isolated active nodes. Each true adjacency entry records a selected directed communication link. [The complete source](../getting-started/quickstart.md) is included from the same file exercised by CI.

The example sets `node_threshold=0.0` to favor participation in its three-agent demonstration. General configurations default to `n-1`.
