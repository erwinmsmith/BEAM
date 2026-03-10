# LangGraph

BEAM’s integration with LangChain goes beyond simple compatibility; it introduces an **algorithmic efficiency control layer** on top of standard declarative chains (LCEL), enabling cost-aware multi-agent orchestration.

---

## Integrated Workflow

BEAM transforms LangChain Runnables into "cost-sensitive" nodes, evolving the workflow from basic execution to intelligent scheduling:

### Protocol & Message Adaptation

The LangChainLLMWrapper unifies communication protocols. It dynamically converts BEAM’s generic message schemas into LangChain-specific objects (e.g., HumanMessage), making the underlying optimization logic transparent to the model layer.

```python

# Wrap an existing LLM instance to grant it BEAM's async generation capabilities
beam_llm = LangChainLLMWrapper(ChatOpenAI(model="gpt-4o"))
```

### Active Token Monitoring & Feedback

The LangChainCallbackHandler hooks directly into the execution flow. During the lifecycle of a LangChain chain, it captures real-time token metrics and routes them as feedback signals to the BEAM optimization engine.

```python

handler = LangChainCallbackHandler()
# Inject the monitoring hook automatically during invocation
llm.invoke(messages, config={"callbacks": [handler]})
stats = handler.get_stats() # Retrieve precise Prompt/Completion metrics
```

### Node Transformation & Strategy Mapping

Using wrap\_langchain\_runnable, complex LCEL chains are encapsulated into named nodes with specific roles. Finally, create\_beam\_chain defines the global optimization strategy (e.g., Pruning or Bayesian).

```Python

# Wrap a LangChain chain as a functional node with a semantic role
node_fn = wrap_langchain_runnable(research_chain, role="Researcher")

# Generate a BEAM configuration utilizing the Pruning strategy
chain_config = create_beam_chain([research_chain, writer_chain], strategy="prune")
```

## Technical Advancements

### Automated Cost-Quality Equilibrium

In traditional LangChain development, developers must manually optimize prompts to reduce token usage. BEAM introduces **Automated Optimization Strategies** that use real-time data from the CallbackHandler to select the most cost-effective path among multiple chain candidates using Bayesian optimization.

### Seamless Cross-Framework Migration

The LangChainLLMWrapper offers full compatibility with the LangChain ecosystem. This allows developers to inject BEAM’s multi-agent optimization into existing RAG or Tool-use chains via a lightweight wrapper, significantly reducing the risk and effort of system refactoring.

### Fine-Grained Marginal Contribution Assessment

By injecting semantic roles through wrap\_langchain\_runnable, BEAM quantifies the "marginal contribution" of each LangChain node to the final output. This enables **Spatial Pruning**—the ability to identify and eliminate energy-intensive steps that appear important but offer negligible improvements to accuracy.

### Core Comparison Matrix

| Feature Dimension | Standard LangChain | BEAM-Enhanced LangChain |
| :---- | :---- | :---- |
| **Message Handling** | Manual construction of Message objects | **Automatic protocol adaptation** |
| **Token Monitoring** | Manual parsing from response usage | **Automatic callback-driven feedback** |
| **Execution Logic** | Static sequential execution | **Dynamic Bayesian path selection** |
| **Performance Scaling** | Linear cost growth with complexity | **Non-linear savings via Node Pruning** |

---
