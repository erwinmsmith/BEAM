# Project Overview

Understanding BEAM's architecture and workflow will help you use it effectively.

---

## Architecture Overview

BEAM is a framework designed to optimize communication efficiency in multi-agent LLM systems while maintaining output quality under incomplete information. It addresses the challenge of high inference costs in multi-agent architectures by learning which agent connections and communications are essential using Bayesian optimization techniques.
THETA consists of three main components:

**How It Works:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Multi-Agent System                             │
│                                                                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐         │
│   │ Agent 1 │───▶│ Agent 2 │───▶│ Agent 3 │───▶│ Decision │         │
│   └─────────┘    └─────────┘    └─────────┘    └──────────┘         │
│        │              │              │                              │
│        └──────────────┴──────────────┘                              │
│                       │                                             │
│               BEAM Optimization                                     │
│          (Bayesian Edge Learning)                                   │
│                       ▼                                             │
│   ┌─────────┐              ┌─────────┐    ┌──────────┐              │
│   │ Agent 1 │─────────────▶│ Agent 3 │───▶│ Decision │              │
│   └─────────┘              └─────────┘    └──────────┘              │
│                                                                     │
│   * Fewer tokens consumed    * Maintained accuracy                  │
│   * Reduced latency          * Lower inference cost                 │
└─────────────────────────────────────────────────────────────────────┘
```

---
**Key Features:**

| Feature | Description |
| ------ |------------- |
| Three Optimization Strategies | Unsupervised learning |
| Extensible Prompt Management | Domain-specific prompt templates with registry |
| Framework Integration | Lightweight utilities for popular frameworks |
| Flexible Agent Design | Support custom functions or LLM-based agents |
| Graph-based Architecture | Spatial and temporal agent connections |

---

## Directory Structure

BEAM organizes files in the following structure:

### Project Directory

```
BEAM/
├── beam/
│   ├── __init__.py          # Package exports
│   ├── core/
│   │   ├── config.py        # BEAMConfig, OptimizationConfig, AgentConfig
│   │   ├── graph.py         # AgentGraph - manages agent connections
│   │   ├── node.py          # AgentNode - base agent class
│   │   ├── llm.py           # BaseLLM, LLMRegistry
│   │   └── optimizer.py     # TokenOptimizer - training orchestration
│   ├── strategies/
│   │   ├── prune.py         # AgentPrune strategy
│   │   ├── dropout.py       # AgentDropout strategy
│   │   └── bayesian.py      # AgentBayesian strategy
│   ├── prompts/
│   │   ├── template.py      # PromptTemplate, PromptSet
│   │   └── registry.py      # PromptRegistry
│   ├── integrations/
│   │   ├── langchain.py     # LangChain utilities
│   │   └── langgraph.py     # LangGraph utilities
│   └── examples/
│       ├── basic_usage.py
│       └── prompts_example.py
├── pyproject.toml           # Package configuration
├── README.md
└── requirements.txt
```


## Workflow Summary

The BEAM framework follows a structured process to transform high-level objectives into optimized agent execution. The typical workflow consists of four stages:

**Stage 1: Initialization & Configuration**
1. Configure Settings: Define parameters in BEAMConfig, including API keys, model hyperparameters, and execution constraints.

2. Register LLMs: Use the LLMRegistry to register and load the underlying large language models (e.g., GPT-4, Claude, or local models).

3. Prepare Prompts: Set up task-specific instruction templates within the PromptRegistry for dynamic retrieval.

**Stage 2: Architecture Construction**

1. Define Nodes: Create multiple AgentNode instances, assigning specific functional roles and attributes to each.
2. Build the Graph: Utilize AgentGraph to establish logical connections (Directed Acyclic Graphs) between agents.
3. Initialize Optimizer: Set up the TokenOptimizer engine to oversee the orchestration and efficiency of the graph.

**Stage 3: Optimization & Execution**

1. Run Orchestration: Execute optimizer.py to manage the task flow across the agent network.
2. Apply Strategies: Implement Prune or Dropout strategies to dynamically trim redundant paths or nodes.
3. Token Allocation: Real-time adjustment of token distribution based on feedback to achieve the highest performance-to-cost ratio.

**Stage 4: Output & Evaluation**

1. Generate Results: Retrieve the final output generated through the optimized execution path.

2. Analyze Metrics: Review token consumption reports and perform a cost-benefit analysis.

3. Validate Robustness: Use provided scripts in the examples/ directory to verify the reliability and stability of the agent collaboration.

---

## Next Steps

Now that you understand the architecture, you can:

- Three implementation strategies for BEAM **[strategies](../models/strategies.md)** 
