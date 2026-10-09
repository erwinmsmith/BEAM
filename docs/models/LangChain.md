# LangChain

LangChain runnables can supply execution callbacks for BEAM agents. The BEAM graph controls participation and message routing; the runnable handles the agent's task.

```bash
python -m pip install -e '.[langchain]'
```

```python
from beam import AgentNode
from beam.integrations.langchain import wrap_langchain_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "Solve the task using the available context."),
    ("human", "Task: {task}\nContext: {context}"),
])
runnable = prompt | ChatOpenAI(model="gpt-4o")
node = AgentNode(
    id="solver", role="Solver",
    execute_fn=wrap_langchain_runnable(runnable, role="Solver"),
)
```

Register each node with `graph.add_node(node)` and attach the complete graph to `AgentBayesian`. Supply credentials through your model provider's environment variables.

`LangChainCallbackHandler` records prompt, completion and total token counts from LLM callbacks. Use those measurements in an evaluation protocol or cost analysis. The Bayesian calibration utility uses its configured normalized spatial-edge cost.
