# Prompt templates

`PromptTemplate` accepts string templates or callables. `PromptSet` groups templates by role and adds an aggregation prompt.

```python
from beam import PromptSet, PromptRegistry

prompts = PromptSet(name="arithmetic")
prompts.add_role(
    "solver",
    system="Solve the arithmetic task carefully.",
    user="Task: {task}\nContext: {context}",
)
prompts.set_decision_template(
    system="Combine the agents' answers.",
    user="Task: {task}\nContext: {context}",
)
PromptRegistry.register("arithmetic", prompts)
system, user = prompts.get_prompt("solver", task="2 + 2", context="")
assert "2 + 2" in user
```

Pass the rendered system prompt and a user template to an `AgentNode` using an LLM client. `PromptSet.save(path)` and `PromptSet.load(path)` serialize string-based prompt sets as JSON.
