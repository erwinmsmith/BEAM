# API Reference

The BEAM API Reference provides a detailed technical overview of the framework's components. It is organized into three key areas: Core Classes for building and managing agent graphs, LLM Classes for model interfacing and registration, and Integrations for bridging BEAM with external ecosystems like LangChain and LangGraph.

---

## Core Classes

**BEAMConfig:**

Main configuration class for BEAM systems.
```python
@dataclass
class BEAMConfig:
    agents: List[AgentConfig]           # Agent definitions
    num_rounds: int = 1                 # Number of reasoning rounds
    optimization: OptimizationConfig    # Optimization settings
    decision_method: str = "reference"  # "reference", "majority", "direct"
    domain: str = ""                    # Task domain
```

**Definitions:**

| Attribute | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `agents`| `List[AgentConfig]` | *Required* | Definitions of roles and counts for agents. |
| `num_rounds` | `int` | `1` | Number of reasoning/interaction iterations. |
| `optimization` | `OptimizationConfig` | *Required* | Settings for pruning or efficiency strategies. |
| `decision_method` | `str` | `"reference"` | Strategy: `"reference"`, `"majority"`, or `"direct"`. |
| `domain` | `str` | `""` | The specific task area (e.g., "math", "medical"). |

**AgentNode:**

Base class for all agents in the graph.

```python
node = AgentNode(
    id="unique_id",                     # Optional, auto-generated if not provided
    agent_name="Solver",                # Agent type name
    role="Problem Solver",              # Role description
    domain="math",                      # Task domain
    llm=llm_instance,                   # LLM for generation (optional)
    execute_fn=custom_function,         # Custom execution (optional)
    system_prompt="...",                # System prompt for LLM
    user_prompt_template="...",         # User prompt template
)

# Execute
result = node.execute({"task": "..."})
result = await node.async_execute({"task": "..."})
```

**Definitions:**

| Attribute | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | `str` | `None` | Unique identifier. Auto-generated if not provided. |
| `agent_name` | `str` | *Required* | Type name of the agent (e.g., "Solver"). |
| `role` | `str` | *Required* | Brief description of the agent's role/responsibility. |
| `domain` | `str` | `""` | The task domain this node belongs to. |
| `llm` | `LLM` | `None` | The LLM instance used for text generation. |
| `execute_fn` | `Callable` | `None` | Optional custom Python function for non-LLM tasks. |
| `system_prompt` | `str` | `""` | The system-level instruction for the LLM. |
| `user_prompt_template` | `str` | `""` | Template for user input with `{variable}` placeholders. |

**AgentGraph:**

Manages agent connections and execution flow.

```python
graph = AgentGraph(config)
graph.add_node(node)
graph.add_nodes([node1, node2, node3])

# Run inference
results = await graph.run({"task": "..."}, num_rounds=2)
```

**Definitions:**

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| **`add_node(node)`** | `node: AgentNode` | `None` | Adds a single agent node to the graph. |
| **`add_nodes(nodes)`** | `nodes: List[Node]` | `None` | Batch adds multiple agent nodes. |
| **`run(inputs, rounds)`** | `dict, int` | `List[Res]` | Asynchronously executes the reasoning flow. |

**PromptSet:**

Collection of prompts for a domain.


```python
prompts = PromptSet(name="domain_name")
prompts.add_role(role, system, user)
prompts.set_decision_template(system, user)
prompts.get_prompt(role, **variables)
prompts.save("prompts.json")
prompts.load("prompts.json")
```

**Definitions:**

| Method | Arguments | Description |
| :--- | :--- | :--- |
| **`add_role(role, system, user)`** | `str, str, str` | Adds a new role with specific system and user templates. |
| **`set_decision_template(sys, usr)`** | `str, str` | Sets global templates for agent decision-making. |
| **`get_prompt(role, **vars)`** | `str, kwargs` | Renders a prompt by injecting variables into the template. |
| **`save(file_path)`** | `str` | Serializes the prompt set to a JSON file. |
| **`load(file_path)`** | `str` | Loads a prompt set configuration from a JSON file. |


**PromptRegistry:**

Global registry for prompt sets.
```python
PromptRegistry.register(name, prompt_set)
PromptRegistry.get(name)
PromptRegistry.keys()
PromptRegistry.load_from_file(name, path)
```

**Definitions:**

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| **`register(name, prompt_set)`** | `str, PromptSet` | `None` | Registers a `PromptSet` under a specific global name. |
| **`get(name)`** | `str` | `PromptSet` | Retrieves a registered `PromptSet` by its name. |
| **`keys()`** | `None` | `List[str]` | Returns a list of all registered `PromptSet` names. |
| **`load_from_file(name, path)`** | `str, str` | `None` | Loads a `PromptSet` from a JSON file and registers it. |

---

## LLM Classes
**BaseLLM:**

Abstract base class for LLM implementations.
```python
class BaseLLM(ABC):
    @abstractmethod
    def gen(self, messages: List[Dict]) -> str: ...
    
    @abstractmethod
    async def agen(self, messages: List[Dict]) -> str: ...
```

**Definitions:**

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| **`gen(messages)`** | `List[Dict]` | `str` | **Synchronous** generation. Takes a list of message dictionaries (role/content). |
| **`agen(messages)`** | `List[Dict]` | `str` | **Asynchronous** generation. Recommended for high-concurrency multi-agent tasks. |

**LLMRegistry:**

Registry for LLM implementations.
```python
# Get LLM instance
llm = LLMRegistry.get("gpt-4o")
llm = LLMRegistry.get("deepseek-chat")

# Register custom LLM
@LLMRegistry.register("custom")
class CustomLLM(BaseLLM):
    ...
```

**Definitions:**

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| **`get(name)`** | `str` | `BaseLLM` | Retrieves an initialized LLM instance by its registered name. |
| **`register(name)`** | `str` | `Decorator` | A decorator to register a custom class (must inherit from `BaseLLM`). |

---

## Integration
**LangChain:**

```python
from beam.integrations.langchain import (
    LangChainLLMWrapper,
    LangChainCallbackHandler,
    wrap_langchain_runnable,
)

# Wrap LangChain LLM for use with BEAM
from langchain_openai import ChatOpenAI
langchain_llm = ChatOpenAI(model="gpt-4")
beam_llm = LangChainLLMWrapper(langchain_llm)

# Use with BEAM agents
agent = create_agent_node(role="Solver", llm=beam_llm, ...)

# Track token usage with callback
callback = LangChainCallbackHandler()
# Use callback in your LangChain chains

# Wrap existing runnable
from langchain_core.runnables import RunnableSequence
wrapped = wrap_langchain_runnable(your_chain, beam_config)
```

**Definitions:**

| Component | Type | Description |
| :--- | :--- | :--- |
| **`LangChainLLMWrapper`** | `Class` | Wraps a LangChain LLM instance to make it compatible with BEAM's `BaseLLM`. |
| **`LangChainCallbackHandler`** | `Class` | A standard callback handler to track token usage and events within LangChain. |
| **`wrap_langchain_runnable`** | `Function` | Converts an existing LangChain `Runnable` or `Chain` into a BEAM-compatible node. |

**LangGraph:**

Registry for LLM implementations.
```python
from beam.integrations.langgraph import (
    BEAMState,
    create_beam_node,
    create_skip_condition,
    apply_beam_masks,
)

# Use BEAM state in your graph
from langgraph.graph import StateGraph

class MyState(BEAMState):
    custom_field: str

# Create BEAM-aware node
@create_beam_node(agent_id="solver")
def solver_node(state: MyState) -> dict:
    # Your logic here
    return {"result": "..."}

# Create skip condition based on BEAM masks
skip_condition = create_skip_condition("solver", beam_strategy)

# Build graph
graph = StateGraph(MyState)
graph.add_node("solver", solver_node)
graph.add_conditional_edges("start", skip_condition, {...})
```

**Definitions:**

| Component | Type | Description |
| :--- | :--- | :--- |
| **`BEAMState`** | `Class` | A specialized state class that tracks BEAM-specific metadata (masks, weights) within LangGraph. |
| **`create_beam_node`** | `Decorator` | Transforms a standard function into a BEAM-aware node that respects pruning and optimization. |
| **`create_skip_condition`** | `Function` | Creates routing logic for LangGraph edges based on BEAM's pruning results (skipping inactive nodes). |
| **`apply_beam_masks`** | `Function` | Utility to filter or weight LangGraph nodes based on the current BEAM optimization state. |

---