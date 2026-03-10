# Config

This guide outlines how to define Team Structure, Collaboration Logic, and Evolutionary Strategies for your multi-agent system.

---

## Minimal YAML Template

```YAML
domain: "dev_ops"
num_rounds: 3

agents:
  - name: "Lead"
    role: "System Architect"
    count: 1
  - name: "Dev"
    role: "Python Engineer"
    count: 2

connection_mode: "star"
decision_method: "DecisionMethod.REFER"

llm:
  model_name: "gpt-4o"
  temperature: 0.2

optimization:
  strategy: "prune"
  pruning_rate: 0.2
```

---

## Core Logic

The configuration follows a top-down hierarchy:

Brain (LLM) → Members (Agents) → Organization (Topology) → Evolution (Optimization).

---

## Parameter Definitions

**LLM Layer:**

Defines the cognitive engine for all agents.

| Parameter | Range | Recommendation |
| :--- | :--- | :--- |
| model_name | str | Use gpt-4o for logic; gpt-4o-mini for cost-efficiency. |
| temperature | 0.0 - 1.0 | 0.1-0.3: Coding/Logic; 0.7-0.9: Creative writing. |
| max_tokens | int | Prevents truncation; 2048+ recommended for code generation. |

---

**Agent Layer:**

Defines the specialized roles in your team.

**role**: The System Prompt. Be specific about expertise and output format.

**count**: Number of instances. Increase to gain diverse perspectives via ensemble logic.

---
**Topology & Decision**
Defines how information flows and how the final result is reached.

**connection_mode:**

- **full_connected:** Everyone sees everything. High transparency, high noise.
- **star:** First agent is the Leader. Centralized command and control.
- **chain:** $A \to B \to C$. Ideal for sequential pipelines (Design → Code → Test).
- **layered:** Mimics organizational hierarchies.

**decision_method:**

- **major_vote:** Best for objective tasks (Math/Logic).

- **refer:** Final summary by a designated expert (usually the last agent).

- **weighted:** Aggregates results based on agent importance.

---

**Optimization Layer**
Automates the refinement of communication paths over time.

**strategy:**

- **prune:** Removes redundant or low-value communication edges.

- **dropout:** Randomly breaks connections to force agent independence.
- **Bayesian:** Uses Bayesian optimization with optional MCMC sampling for uncertainty-aware pruning.

**pruning_rate:** (0.0-1.0). Default 0.2. High values lead to a more sparse, efficient network.

**optimize_spatial:** Set to true to allow the system to re-route "who talks to whom."

---
