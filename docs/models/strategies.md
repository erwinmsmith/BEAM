# Strategies

BEAM provides three specialized optimization strategies designed to balance performance and token efficiency. These strategies dynamically adjust the agent network by removing redundant paths, skipping unnecessary agents, or using probabilistic models to find the optimal configuration.

---

## AgentPrune

Learns edge importance through policy gradient optimization and prunes low-importance connections.

**Training:**

```python

from beam import AgentPrune, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="prune",
        optimize_spatial=True,
        optimize_temporal=True,
        pruning_rate=0.3,        # Prune 30% of edges
        initial_spatial_probability=0.5,
        initial_temporal_probability=0.5
)
)
strategy = AgentPrune(config)

```

**Best for:**

- Dense agent networks where many connections are redundant.

**How it works:**

- Initializes learnable logits for each potential edge
- Uses Gumbel-Softmax for differentiable edge sampling
- Optimizes via policy gradient based on task performance
- Prunes edges below learned threshold

**Persistence**

`AgentPrune` provides a robust persistence interface, allowing you to save the trained "optimal topology" and reuse it in production environments. This ensures that your training results are preserved and that your deployment utilizes a deterministic, lightweight reasoning path.

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| **`save(path)`** | `str` | `None` | Serializes and saves the learned `logits` (connection weights) and `masks` (pruning status). This captures the complete evolutionary state of the graph. |
| **`load(path)`** | `str` | `None` | Restores connection weights and pruning masks from a specified file, instantly returning the graph to its optimized state. |

---

---

## AgentDropout

Learns which agents can be skipped entirely during inference.

```python

from beam import AgentDropout, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="dropout",
        optimize_spatial=True,
        dropout_rate=0.2,        # Target 20% agent skip rate
    )
)

strategy = AgentDropout(config)
```

**Best for:**

- Systems with redundant agents where some can be skipped without quality loss.

**How it works:**

- Learns skip probability for each agent
- During training, samples skip decisions
- Agents with high skip probability are dropped during inference
- Maintains ensemble diversity while reducing computation

---

## AgentBayesian

Uses Bayesian optimization with optional MCMC sampling for uncertainty-aware pruning.

```python
from beam import AgentBayesian, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="bayesian",
        optimize_spatial=True,
        optimize_temporal=True,
        use_mcmc=True,           # Enable MCMC sampling
        mcmc_samples=100,        # Number of MCMC samples
    )
)

strategy = AgentBayesian(config)
```

**Best for:**

- When you need confidence estimates or have limited training data.

**How it works:**

- Maintains posterior distribution over edge weights
- Uses MCMC (optional) to sample from posterior
- Provides uncertainty estimates for pruning decisions
- More robust with small datasets

---

## Spatial & Temporal

**Spatial connections** define how information is passed horizontally between agents within the **same execution round (Round)**.

**How it Works：**
 It determines an agent’s "immediate collaborative circle." When `Agent_A` generates an output, spatial connections decide which other agents (e.g., `Agent_B`) can immediately see that information in their current context. Logically, this forms a Directed Acyclic Graph (DAG).
**Optimization Logic：**
 By enabling `optimize_spatial`, the system learns which communications are redundant. For instance, in a "Dev-Test" workflow, if the testing agent finds it only needs the source code and not the developer's logs, the system will automatically prune the spatial connection from the developer to the tester, reducing context noise.

---

**Temporal connections** define **cross-round persistence**, governing how agents utilize "past" experiences.

- **How it Works**: It determines an agent’s "long-term memory." As the task progresses to Round $n$, temporal connections dictate which intermediate conclusions from previous rounds need to be retained and injected into the current prompt.
- **Optimization Logic**: Enabling `optimize_temporal` helps prevent "hallucination accumulation" and "information overload." If historical data contains faulty reasoning or if past dialogues become too verbose, the system learns to close these temporal edges, achieving "precise forgetting" to keep the agent focused on the most valuable decision history.

---

## Topology Comparison Matrix

| Topology Mode | Structure Description | Token Cost | Core Advantage | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`FULL_CONNECTED`** | Every agent is connected to every other agent. | **Extreme** | Maximum search space; no potential collaboration is missed. | **Architecture Discovery**. Ideal for complex tasks where the optimal path is unknown. |
| **`CHAIN`** | Sequential flow: $A \to B \to C$. Agents only see their direct predecessor. | **Very Low** | Maximum efficiency; eliminates noise from unrelated agents. | **Standard SOPs**. Linear tasks such as `Research -> Summary -> Translation`. |
| **`STAR`** | A central "Manager" (Agent 0) connects to all "Workers." | **Moderate** | Centralized control; prevents peer-to-peer "chatter." | **Manager-Worker Pattern**. A lead agent dispatches tasks and synthesizes results. |
| **`LAYERED`** | Agents are grouped into stages; each group feeds into the next. | **Moderate** | Balances group collaboration with global task progress. | **Multi-stage Refinement**. E.g., `Drafting Team -> Review Team -> Final Approver`. |
| **`DEBATE`** | Full-mesh connection with deep cross-round (temporal) memory. | **High** | Reduces hallucinations through iterative cross-examination. | **Fact-Checking & Logic**. Scenarios where agents must challenge and verify each other. |
| **`RANDOM`** | Connections are initialized based on random probability. | **Variable** | Breaks human design bias and tests robustness. | **Benchmarking**. Testing BEAM's ability to self-organize and prune from a chaotic start. |

---
