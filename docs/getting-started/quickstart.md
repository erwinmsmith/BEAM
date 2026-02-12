# Quick Start

This tutorial demonstrates how to train a BEAM model on your dataset in under 5 minutes.

---

## Step 1: Prepare Your Data

Create a json file with your text data. The Json must contain a column with text content.

**Example JSOn format:**

```json
{"task": "What is 15% of 80?", "answer": "12"},
{"task": "Solve: 2x + 5 = 13", "answer": "4"},
{"task": "Calculate 3^4", "answer": "81"},
```

**Required columns:**

| Column Name | Type | Required | Description |
|------------|------|----------|-------------|
| task / clean_text | string | Yes | Math problem description|
| answer | float/int | Yes | True answer |

---

## Step 2: Configure Your System


```python
from beam import BEAMConfig, AgentConfig, OptimizationConfig

config = BEAMConfig(
    # Define your agents
    agents=[
        AgentConfig(name="Analyzer", count=2),
        AgentConfig(name="Solver", count=3),
        AgentConfig(name="Verifier", count=1),
    ],
    
    # Multi-round reasoning
    num_rounds=2,
    
    # Optimization settings
    optimization=OptimizationConfig(
        strategy="prune",           # "prune", "dropout", or "bayesian"
        optimize_spatial=True,      # Optimize same-round connections
        optimize_temporal=True,     # Optimize cross-round connections
        pruning_rate=0.25,          # Target 25% edge reduction
        learning_rate=0.01,
    ),
    
    # Decision aggregation
    decision_method="reference",    # "reference", "majority", or "direct"
    domain="math"
)
```

---

## Step 3: Create Agents

**Option A: Using Custom Functions**

```python
from beam import create_agent_node

def analyze_problem(state):
    """Custom analysis logic."""
    task = state["task"]
    context = state.get("context", "")
    # Your analysis logic here
    return f"Analysis: The problem requires solving {task}"

def solve_problem(state):
    """Custom solving logic."""
    task = state["task"]
    analysis = state.get("context", "")
    # Your solving logic here
    return f"Solution based on analysis: {analysis}"

# Create nodes
analyzer = create_agent_node(role="Analyzer", execute_fn=analyze_problem)
solver = create_agent_node(role="Solver", execute_fn=solve_problem)
```

**Option B: Using LLM with Prompts**

```python
from beam import create_agent_node, LLMRegistry

# Get LLM instance
llm = LLMRegistry.get("gpt-4o")

# Create agent with prompts
analyzer = create_agent_node(
    role="Analyzer",
    llm=llm,
    system_prompt="""You are a problem analyzer. Your task is to:
1. Identify key information in the problem
2. Determine what type of problem this is
3. Outline the approach to solve it""",
    user_prompt_template="""Problem: {task}

Previous analysis (if any):
{context}

Provide your analysis:"""
)

solver = create_agent_node(
    role="Solver", 
    llm=llm,
    system_prompt="""You are a problem solver. Based on the analysis provided,
solve the problem step by step and provide a clear final answer.""",
    user_prompt_template="""Problem: {task}

Analysis from other agents:
{context}

Your solution:"""
)
```

---

## Step 4: Use Prompt Templates (Recommended)

```python
from beam import PromptSet, PromptRegistry

# Create a reusable prompt set
math_prompts = PromptSet(name="math", description="Math problem solving")

# Add role-specific prompts
math_prompts.add_role(
    role="analyzer",
    system="""You are a mathematical analyst. Break down problems into components:
- Identify given information
- Identify what needs to be found
- Suggest solution strategies""",
    user="Problem: {task}\n\nContext: {context}\n\nYour analysis:"
)

math_prompts.add_role(
    role="solver",
    system="""You are a math solver. Show your work step by step.
Be precise with calculations and clearly state your final answer.""",
    user="Problem: {task}\n\nAnalysis: {context}\n\nSolution:"
)

math_prompts.add_role(
    role="verifier",
    system="""You verify mathematical solutions. Check for:
- Calculation errors
- Logic errors
- Missing steps
Confirm or correct the answer.""",
    user="Problem: {task}\n\nSolution to verify: {context}\n\nVerification:"
)

# Set decision template
math_prompts.set_decision_template(
    system="Synthesize multiple solutions and provide the final answer.",
    user="Problem: {task}\n\nSolutions:\n{context}\n\nFinal answer:"
)

# Register for global access
PromptRegistry.register("math", math_prompts)

# Use anywhere in your code
prompts = PromptRegistry.get("math")
system, user = prompts.get_prompt("solver", task="2+2=?", context="Simple addition")
```

## Step 5: Train and Run Optimization

```python
from beam import AgentPrune, AgentGraph

# Create strategy
strategy = AgentPrune(config)

# Build graph (connects agents based on config)
graph = AgentGraph(config)
graph.add_nodes([analyzer, solver, verifier])
strategy.set_graph(graph)

# Prepare training data
train_data = [
    {"task": "What is 15% of 80?", "answer": "12"},
    {"task": "Solve: 2x + 5 = 13", "answer": "4"},
    {"task": "Calculate 3^4", "answer": "81"},
    # ... more examples
]

# Define evaluation function
def eval_fn(prediction: str, answer: str) -> float:
    """Return 1.0 for correct, 0.0 for incorrect."""
    try:
        # Extract numbers and compare
        pred_nums = [float(s) for s in prediction.split() if s.replace('.','').isdigit()]
        ans_num = float(answer)
        return 1.0 if ans_num in pred_nums else 0.0
    except:
        return 0.0

# Train (learns which edges to prune)
training_stats = strategy.train(
    train_data=train_data,
    eval_fn=eval_fn,
    epochs=10,
    batch_size=4
)

print(f"Training accuracy: {training_stats['final_accuracy']:.2%}")
print(f"Edges pruned: {training_stats['edges_pruned']}")

# Run optimized inference
result, metadata = await strategy.run({"task": "What is 25% of 200?"})
print(f"Answer: {result}")
print(f"Tokens used: {metadata['tokens_used']}")
print(f"Agents activated: {metadata['active_agents']}")
```

---

## What's Next?

- Three implementation strategies for BEAM **[strategies](../models/strategies.md)** 
- [Examples](../examples/examples.md) - Real-world use cases
