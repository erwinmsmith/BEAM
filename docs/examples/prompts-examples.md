# Examples

Here are six basic application examples.

---

## Example 1: Basic configuration

```python
def create_basic_config():
    """Create a basic BEAM configuration."""
    
    config = BEAMConfig(
        # Define agents
        agents=[
            AgentConfig(name="Solver", count=3),
            AgentConfig(name="Verifier", count=1),
        ],
        
        # Graph settings
        num_rounds=2,
        
        # LLM settings (configure with your API)
        # llm=LLMConfig(
        #     model_name="gpt-4o",
        #     api_key="your-api-key"
        # ),
        
        # Optimization settings
        optimization=OptimizationConfig(
            strategy="prune",
            optimize_spatial=True,
            optimize_temporal=True,
            pruning_rate=0.25,
        ),
        
        domain="math"
    )
    
    return config
```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Identity** | role | "solver" / "verifier" | Unique ID for the agent. | **Logic**: Must match the name in BEAMConfig for the system to route prompts correctly. |
| **Persona** | system | "You are an expert..." | Core instructions and constraints. | **Tuning**: Adding "Show your work step-by-step" (Chain-of-Thought) significantly boosts accuracy. |
| **The Goal** | {task} | *Runtime Variable* | Placeholder for the original math problem. | **Logic**: Stays constant so all agents focus on the same core problem throughout the rounds. |
| **The Memory** | {context} | *Runtime Variable* | Placeholder for outputs from previous agents. | **Impact**: Crucial for collaboration. This is how a verifier sees the solver's work. |
| **Final Synthesis** | decision\_template | *Global Role* | Instructions for the final aggregation step. | **Tuning**: Use this to force a specific output format (e.g., "Output only the final number"). |

---

## Example 2: Create prompts for code generation

```python
def create_coding_prompts():
    """Create prompts for code generation tasks."""
    
    prompts = PromptSet(
        name="coding",
        description="Prompts for collaborative code generation"
    )
    
    prompts.add_role(
        role="architect",
        system="""You are a software architect. Design the high-level structure and approach for solving coding problems.

Focus on:
- Algorithm selection
- Data structure choices
- Edge cases to consider
- Time/space complexity""",
        user="""Task: {task}

{context}

Provide your architectural analysis and recommended approach."""
    )
    
    prompts.add_role(
        role="coder",
        system="""You are an expert programmer. Write clean, efficient, well-documented code.

Guidelines:
- Follow best practices
- Handle edge cases
- Write readable code
- Include brief comments for complex logic""",
        user="""Task: {task}

{context}

Write the code solution."""
    )
    
    prompts.add_role(
        role="reviewer",
        system="""You are a code reviewer. Review code for correctness, efficiency, and best practices.

Check for:
- Bugs and logic errors
- Edge case handling
- Code style and readability
- Potential optimizations""",
        user="""Task: {task}

Code to review:
{context}

Provide your review and any suggested improvements."""
    )
    
    prompts.set_decision_template(
        system="Synthesize the code solutions and reviews into a final, correct implementation.",
        user="""Task: {task}

Solutions and reviews:
{context}

Provide the final code solution:"""
    )
    
    return prompts
```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Architect** | system | "Design high-level structure..." | Focuses on the "Plan." Prevents the Coder from jumping into a bad approach. | **Tuning**: Instruct the architect to output pseudo-code to provide a clear roadmap for the Coder. |
| **Coder** | system | "Write clean, efficient code..." | Focuses on "Implementation." Translates the Architect's plan into syntax. | **Impact**: By seeing the Architect's design in {context}, the Coder's output becomes much more structured. |
| **Reviewer** | system | "Check for bugs and style..." | Focuses on "Quality." Acts as a final gatekeeper for logic errors. | **Tuning**: Specify a language (e.g., "Check for Python PEP8 compliance") to get more specific feedback. |
| **Decision** | system | "Synthesize... into a final implementation." | The final "Merger." Combines the fix suggestions into the final code block. | **Impact**: Use this to ensure the final output contains **only code** without excessive conversational filler. |

---

## Example 3: Register prompts with the registry

```python
def setup_prompts():
    """Register all prompt sets with the global registry."""
    
    # Register math prompts
    math_prompts = create_math_prompts()
    PromptRegistry.register("math", math_prompts)
    
    # Register coding prompts
    coding_prompts = create_coding_prompts()
    PromptRegistry.register("coding", coding_prompts)
    
    print(f"Registered prompt sets: {PromptRegistry.keys()}")
```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Registration** | register | "math", math\_prompts | Links a unique string key to a specific PromptSet object. | **Logic**: This "saves" the prompts into a global dictionary for later use by the AgentGraph. |
| **Identification** | key | "coding" | The unique name used to look up the instructions. | **Impact**: Must be consistent. If you register as "coding", you must call it as "coding". |
| **Management** | keys() | *Method* | Returns a list of all currently registered prompt sets. | **Impact**: Useful for debugging or verifying that your setup script ran correctly. |

---

## Example 4: Using prompts with agents

```python
def example_usage():
    """Demonstrate how to use prompts with BEAM agents."""
    
    from beam import create_agent_node, LLMRegistry
    
    # Setup prompts
    setup_prompts()
    
    # Get the math prompt set
    math_prompts = PromptRegistry.get("math")
    
    # Get prompts for a specific role
    system, user_template = math_prompts.get_prompt(
        "solver",
        task="What is 15% of 80?",
        context=""
    )
    
    print("System prompt:")
    print(system)
    print("\nUser prompt:")
    print(user_template)
    
    # Create an agent node with these prompts
    # (Assuming you have an LLM configured)
    # llm = LLMRegistry.get("gpt-4o")
    # 
    # solver_node = create_agent_node(
    #     role="solver",
    #     llm=llm,
    #     system_prompt=system,
    #     user_prompt_template="Problem: {task}\n\n{context}\n\nSolve step by step:"
    # )

```

**Definitions:**

| Category | Parameter | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Retrieval** | PromptRegistry.get | "math" | Fetches the entire collection of math-related roles and templates. | **Logic**: Use this to ensure all agents in a graph share the same domain-specific instruction set. |
| **Extraction** | get\_prompt | "solver" | Pulls the system and user strings specifically for that role. | **Impact**: Allows for manual inspection or modification of a prompt before the agent actually uses it. |
| **Dynamic Filling** | task / context | "What is 15%..." | Replaces the curly-brace placeholders in the template with actual data. | **Logic**: In standard usage, BEAM handles this automatically, but manual extraction is useful for debugging. |
| **Node Binding** | create\_agent\_node | *Factory Method* | Finalizes the agent by attaching the LLM and the retrieved prompts. | **Impact**: This is the "Birth" of the agent. Once created, it is ready to be added to an AgentGraph. |

---

## Example 5: Save and load prompts

```python
def save_load_example():
    """Show how to save and load prompt sets."""
    
    prompts = create_math_prompts()
    
    # Save to file
    prompts.save("math_prompts.json")
    print("Saved prompts to math_prompts.json")
    
    # Load from file
    loaded = PromptSet.load("math_prompts.json")
    print(f"Loaded prompt set: {loaded.name}")
    print(f"Available roles: {loaded.get_roles()}")

```

**Definitions:**

| Category | Parameter / Method | Example Value | Description | Tuning Logic & Impact |
| :---- | :---- | :---- | :---- | :---- |
| **Serialization** | prompts.save | "math\_prompts.json" | Converts the PromptSet object (including all roles and templates) into a structured JSON file. | **Logic**: Saves the entire state, including the name, description, and every add\_role configuration. |
| **Deserialization** | PromptSet.load | "math\_prompts.json" | A class method that reads a JSON file and recreates a functional PromptSet object. | **Impact**: Allows you to reconstruct the exact same agent logic in a completely different environment or session. |
| **Inspection** | get\_roles() | *Method* | Returns a list of all roles defined within the loaded prompt set. | **Usage**: Use this after loading to verify that all roles (e.g., solver, verifier) were successfully recovered. |

---
