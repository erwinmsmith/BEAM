# Examples


This example showcases a "Multi-Solver + Verifier" architecture. It demonstrates how to use AgentPrune to identify which solver paths are redundant for specific math tasks, thereby reducing token consumption without sacrificing accuracy.

---


## Complete Math Solving Example


```python
import asyncio
from beam import (
    BEAMConfig, AgentConfig, OptimizationConfig,
    AgentPrune, AgentGraph, create_agent_node,
    PromptSet, PromptRegistry, LLMRegistry
)

async def main():
    # 1. Setup prompts
    prompts = PromptSet(name="math")
    prompts.add_role("solver", 
        system="Solve math problems step by step.",
        user="Problem: {task}\nContext: {context}\nSolution:")
    prompts.add_role("verifier",
        system="Verify mathematical solutions.",
        user="Problem: {task}\nSolution: {context}\nVerification:")
    PromptRegistry.register("math", prompts)
    
    # 2. Configure system
    config = BEAMConfig(
        agents=[
            AgentConfig(name="solver", count=3),
            AgentConfig(name="verifier", count=1),
        ],
        num_rounds=1,
        optimization=OptimizationConfig(
            strategy="prune",
            optimize_spatial=True,
            pruning_rate=0.25,
        )
    )
    
    # 3. Create agents
    llm = LLMRegistry.get("gpt-4o")
    math_prompts = PromptRegistry.get("math")
    
    agents = []
    for i in range(3):
        sys, usr = math_prompts.get_prompt("solver", task="{task}", context="{context}")
        agents.append(create_agent_node(
            role=f"Solver_{i}",
            llm=llm,
            system_prompt=sys,
            user_prompt_template=usr
        ))
    
    sys, usr = math_prompts.get_prompt("verifier", task="{task}", context="{context}")
    agents.append(create_agent_node(
        role="Verifier",
        llm=llm,
        system_prompt=sys,
        user_prompt_template=usr
    ))
    
    # 4. Build and train
    strategy = AgentPrune(config)
    graph = AgentGraph(config)
    graph.add_nodes(agents)
    strategy.set_graph(graph)
    
    train_data = [
        {"task": "2 + 2", "answer": "4"},
        {"task": "10 * 5", "answer": "50"},
        {"task": "100 / 4", "answer": "25"},
    ]
    
    def eval_fn(pred, ans):
        return 1.0 if ans in pred else 0.0
    
    stats = strategy.train(train_data, eval_fn, epochs=5)
    
    # 5. Run inference
    result, meta = await strategy.run({"task": "What is 15% of 200?"})
    print(f"Result: {result}")
    print(f"Tokens saved: {meta.get('tokens_saved', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
    
```

**Definitions:**

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| **`set_graph(graph)`** | `AgentGraph` | `None` | Binds an `AgentGraph` to the optimizer for pruning. |
| **`train(data, eval_fn, epochs)`** | `List[dict], Callable, int` | `dict` | Trains the pruning masks using training data and an evaluation function. |
| **`run(inputs)`** | `dict` | `(str, dict)` | Executes optimized inference, returning the result and metadata (e.g., tokens saved). |

---