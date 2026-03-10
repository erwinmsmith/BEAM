# Examples

Here are six examples of applications that include prompt words.

---

## Example 1: Create a prompt set for math problem solving

```python
def create_math_prompts():
    """Create prompts for mathematical problem solving."""
    
    prompts = PromptSet(
        name="math",
        description="Prompts for mathematical problem solving with multiple agents"
    )
    
    # Add solver role
    prompts.add_role(
        role="solver",
        system="""You are an expert mathematician. Your task is to solve mathematical problems step by step.

Guidelines:
1. Read the problem carefully
2. Identify the key information and what is being asked
3. Show your work step by step
4. Verify your answer
5. State the final answer clearly""",
        user="""Problem: {task}

{context}

Please solve this problem step by step and provide the final answer."""
    )
    
    # Add verifier role
    prompts.add_role(
        role="verifier",
        system="""You are a mathematical verifier. Your task is to check solutions for correctness.

Guidelines:
1. Review the solution carefully
2. Check each step for errors
3. Verify the final answer
4. If incorrect, explain the error and provide the correct solution""",
        user="""Problem: {task}

Solutions to verify:
{context}

Please verify these solutions and confirm or correct the answer."""
    )
    
    # Add decision template
    prompts.set_decision_template(
        system="""You are a decision agent that synthesizes multiple mathematical solutions.
Review all provided solutions and determine the correct final answer.""",
        user="""Problem: {task}

Agent solutions:
{context}

Based on the above solutions, provide the final answer (just the number/result):"""
    )
    
    return prompts
```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Team Composition** | agents | Solver(3), Verifier(1) | Defines agent roles and the number of instances for each. | **Increase count**: Gathers diverse perspectives; ideal for complex reasoning tasks. |
| **Workflow** | num\_rounds | 2 | The number of back-and-forth communication cycles. | **Increase**: Deepens reasoning but increases Token usage linearly. |
| **Strategy** | strategy | "prune" | The core optimization algorithm. | **Prune**: Removes useless connections to save money and reduce noise. |
| **Spatial Opt.** | optimize\_spatial | True | Whether to optimize "who talks to whom" (Topology). | Setting to True lets the system cut redundant chat paths between agents. |
| **Temporal Opt.** | optimize\_temporal | True | Whether to optimize "memory retention" across rounds. | Prevents the context window from being cluttered by irrelevant history. |
| **Pruning Rate** | pruning\_rate | 0.25 | The percentage of connections to attempt to remove during optimization. | **Higher (0.5+)**: Aggressive cost-saving. **Lower (0.1)**: Conservative optimization. |
| **Metadata** | domain | "math" | A label for the task domain. | Helps the system apply domain-specific prompt templates or evaluation metrics. |

---

## Example 2: Create agents with custom execution functions

```python

def create_agents_with_functions():
    """Create agents using custom execution functions."""
    # Define custom execution functions
    def solver_fn(state):
        """Custom solver logic."""
        task = state["task"]
        context = state.get("context", "")
        # Your solving logic here
        return f"Solution for: {task}"
    
    def verifier_fn(state):
        """Custom verifier logic."""
        task = state["task"]
        context = state.get("context", "")
        # Your verification logic here
        return f"Verified: {context}"
    
    # Create nodes
    solver = create_agent_node(
        role="Solver",
        execute_fn=solver_fn
    )
    
    verifier = create_agent_node(
        role="Verifier",
        execute_fn=verifier_fn
    )
    
    return [solver, verifier]
```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Logic Injection** | execute\_fn | solver\_fn | The Python function containing the agent's core logic. | **Adjustment**: Use for deterministic tasks (Regex, API calls). Overrides LLM behavior for this node. |
| **Data Input** | state | {"task": "...", ...} | The dictionary object passed into the function. | **Tuning**: Access state\["task"\] for the goal and state.get("context") for messages from other agents. |
| **Identity** | role | "Verifier" | The unique identifier for the agent node. | **Impact**: Must match the AgentConfig name to ensure the BEAM Graph routes data correctly. |
| **Output** | return | f"Verified: {context}" | The string result sent back to the graph. | **Requirement**: Must return a string (or serializable data) to be used as "context" for the next agent. |

---

## Example 3: Create agents with LLM and prompts

```python
def create_agents_with_llm(llm):
    """Create agents using LLM with prompt templates."""
    
    solver = create_agent_node(
        role="Solver",
        llm=llm,
        system_prompt="""You are a problem solver. Solve the given problem step by step.
Show your reasoning and provide a clear final answer.""",
        user_prompt_template="""Problem: {task}

Other solutions:
{context}

Your solution:"""
    )
    
    verifier = create_agent_node(
        role="Verifier",
        llm=llm,
        system_prompt="""You verify solutions for correctness.
Check the work and confirm or correct the answer.""",
        user_prompt_template="""Problem: {task}

Solutions to verify:
{context}

Your verification:"""
    )
    
    return [solver, verifier]

```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Persona** | system\_prompt | "You are a problem solver..." | Defines the agent's identity, expertise, and behavioral constraints. | **Tuning**: Add specific formatting rules (e.g., "Always use JSON") to make the output easier for the next agent to parse. |
| **Input Schema** | user\_prompt\_template | "Problem: {task}..." | A string template containing placeholders that BEAM fills at runtime. | **Impact**: Determines how the agent perceives the current task and the history of the conversation. |
| **Task Variable** | {task} | *Runtime Variable* | Placeholder for the initial problem or query provided to the system. | **Logic**: Stays constant across all agents and rounds to ensure the team stays focused on the original goal. |
| **Context Variable** | {context} | *Runtime Variable* | Placeholder for the accumulated outputs from other connected agents. | **Impact**: This is the "memory" or "discussion" part. Tuning the surrounding text helps the LLM distinguish its own work from others. |
| **Engine** | llm | LLMConfig(...) | The specific model instance (e.g., GPT-4o) used to power the node. | **Tuning**: You can assign different models to different roles (e.g., a "cheap" model for simple solving and a "powerful" model for final verification). |

---

## Example 4: Using the AgentPrune strategy

```python
async def prune_example():
    """Example of using AgentPrune for edge optimization."""
    
    config = create_basic_config()
    
    # Create the pruning strategy
    prune = AgentPrune(config)
    
    # You would set up your graph here
    # prune.set_graph(your_graph)
    
    # Training data format
    train_data = [
        {"task": "What is 2 + 2?", "answer": "4"},
        {"task": "What is 3 * 5?", "answer": "15"},
    ]
    
    # Evaluation function
    def eval_fn(prediction, answer):
        # Extract number from prediction and compare
        try:
            pred_num = float(''.join(c for c in prediction if c.isdigit() or c == '.'))
            ans_num = float(answer)
            return 1.0 if pred_num == ans_num else 0.0
        except:
            return 0.0
    
    # Train (requires graph to be set)
    # stats = prune.train(train_data, eval_fn)
    
    # Run inference
    # result, metadata = await prune.run({"task": "What is 10 / 2?"})
    
    print("AgentPrune example setup complete")

```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Optimization Engine** | AgentPrune | prune \= AgentPrune(config) | The class responsible for managing edge weights and executing the pruning algorithm. | **Logic**: It requires a BEAMConfig with strategy="prune" to function correctly. |
| **Training Set** | train\_data | \[{"task": "...", "answer": "..."}\] | A list of historical tasks and their ground-truth answers used for calibration. | **Tuning**: A larger, more diverse dataset (20–50 samples) leads to a more robust and efficient pruned graph. |
| **Metric** | eval\_fn | return 1.0 if pred \== ans else 0.0 | A custom function that scores the agent's output against the ground truth. | **Impact**: This is the "Loss Function." It tells the pruner which communication paths are valuable vs. which are "noise." |
| **Execution** | prune.train() | *Internal Method* | The process of running the graph multiple times to identify low-impact edges. | **Impact**: After training, the internal spatial\_masks are updated to block specific agent-to-agent channels. |

---

## Example 5: Using the AgentDropout strategy

```python
async def dropout_example():
    """Example of using AgentDropout for agent skipping."""
    
    config = BEAMConfig(
        agents=[AgentConfig(name="Agent", count=5)],
        num_rounds=3,
        optimization=OptimizationConfig(
            strategy="dropout",
            optimize_spatial=True,
        )
    )
    
    dropout = AgentDropout(config)
    
    # After training, dropout learns which agents to skip
    # This reduces token usage while maintaining quality
    
    print("AgentDropout example setup complete")

```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Optimization Engine** | AgentDropout | dropout \= AgentDropout(config) | The class that manages node-level deactivation (skipping agents). | **Logic**: Unlike Pruning, this treats agents as binary (Active/Inactive) for specific steps. |
| **Agent Scaling** | count | 5 | The number of identical or similar agents in the pool. | **Impact**: Dropout is most effective when you have high redundancy (e.g., 5 agents doing similar tasks). |
| **Strategy Choice** | strategy | "dropout" | Forces the optimizer to evaluate agent necessity rather than connection strength. | **Tuning**: Use this when your goal is to reduce the total number of LLM calls per round. |
| **Spatial Opt.** | optimize\_spatial | True | Allows the system to identify which physical agents to drop. | **Impact**: If an agent's output consistently fails to change the final outcome, it is "dropped." |

---

## Example 6: Full workflow

```python
async def full_workflow_example():
    """Complete example workflow."""
    
    # 1. Create configuration
    config = BEAMConfig(
        agents=[
            AgentConfig(name="Analyst", count=2),
            AgentConfig(name="Solver", count=2),
        ],
        num_rounds=2,
        optimization=OptimizationConfig(
            strategy="prune",
            optimize_spatial=True,
            pruning_rate=0.2,
        )
    )
    # 2. Create prompt templates
    prompts = PromptSet(name="problem_solving")
    prompts.add_role(
        "Analyst",
        system="You analyze problems and identify key information.",
        user="Analyze: {task}\n\nContext: {context}"
    )
    prompts.add_role(
        "Solver",
        system="You solve problems based on analysis.",
        user="Problem: {task}\n\nAnalysis: {context}"
    )
    
    # 3. Create agents (with mock execution for demo)
    def mock_execute(state):
        return f"Processed: {state['task'][:50]}..."
    
    agents = []
    for agent_config in config.agents:
        for i in range(agent_config.count):
            node = create_agent_node(
                role=agent_config.name,
                execute_fn=mock_execute
            )
            agents.append(node)
    
    print(f"Created {len(agents)} agents")
    print(f"Configuration: {config.optimization.strategy} strategy")
    print(f"Rounds: {config.num_rounds}")
    
    # 4. In real usage, you would:
    # - Set up the graph with these agents
    # - Train on your dataset
    # - Run optimized inference

```

**Definitions:**

| Category | Parameter / Step | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Blueprint** | BEAMConfig | Analyst(2), Solver(2) | Initializes the team structure and sets the optimization goal. | **Logic**: Starting with 2 agents per role allows the prune strategy to later identify which specific analyst/solver pair works best. |
| **Instructions** | PromptSet | prompts.add\_role(...) | Centralizes all System and User prompts for the different roles. | **Tuning**: Ensure the user prompt includes {task} and {context} so data flows correctly between rounds. |
| **Logic Bind** | mock\_execute | f"Processed: ..." | The actual code or LLM call that executes when an agent "speaks." | **Impact**: Replacing this with a real LLM call (using llm=config.llm) turns the nodes from static mocks into active AI. |
| **Assembly** | create\_agent\_node | *Factory Method* | Combines the configuration, the prompts, and the execution logic into a Node. | **Requirement**: Nodes must be stored in a list (like agents) to be later injected into an AgentGraph. |

---
