# FAQ - Frequently Asked Questions

---

## 1. How to fix `asyncio` conflicts in Jupyter Notebook?

**Issue**: Running `loop.run_until_complete` in Notebooks causes `RuntimeError: This event loop is already running`.
**Cause**: Jupyter already has a running event loop. Standard `asyncio` does not support nested loops by default.
**Solution**:
Install and use `nest_asyncio` to allow nested event loops:

```python
import nest_asyncio
nest_asyncio.apply()

# Now you can run BEAM asynchronous methods (like agen) normally
```

## 2. How to configure API Keys? (Env Vars vs. Code)

**Option A: Hardcoded in code (Best for quick experiments)**

Pass the api_key directly into LLMRegistry.get():

```Python

llm = LLMRegistry.get(model_name="gpt-4o", api_key="sk-...")

```

**Option B: Environment Variables**

Set them in your system terminal or .env file. BEAM automatically scans for these if the parameter is missing:

* **OpenAI**: export OPENAI\_API\_KEY='sk-...'  
* **DeepSeek**: export DEEPSEEK\_API\_KEY='sk-...'

---

## 3. How to handle CUDA Out of Memory (OOM) during training?

**Solutions**:

1. **Reduce Batch Size**: Lower the batch\_size in your optimizer configuration to reduce immediate VRAM usage.  
2. **Decrease Initial Density**: Lower initial\_spatial\_probability in OptimizationConfig to create a sparser initial graph.  
3. **Gradient Accumulation**: If a large batch is necessary, compute gradients over smaller steps before updating.  
4. **CPU Execution**: For purely structural/topological optimizations that aren't compute-heavy, set device='cpu'.

---

## 4. What if performance drops too much after Pruning?

**Solutions**:

* **Lower Pruning Rate**: Set a smaller pruning\_rate (e.g., 0.05 or 0.1) for a more gradual removal of edges.  
* **Extend Warm-up Period**: Increase the training rounds before triggering update\_masks to ensure edge weights (logits) have converged correctly.  
* **Refine Reward Function**: If the reward signal is too sparse (mostly zeros), the algorithm might prune critical paths randomly due to lack of feedback.  
* **Checkpoint Rollback**: Use strategy.save() to save checkpoints regularly. If accuracy collapses, use strategy.load() to revert to the last stable topology.

---
**Routing Logic**:

BEAM’s LLMRegistry uses keyword-based routing based on the model\_name:

* Contains **gpt** \-\> Routes to OpenAILLM  
* Contains **deepseek** \-\> Routes to DeepSeekLLM (automatically sets DeepSeek endpoints)  
* Contains **llama** or **qwen** \-\> Routes to LocalLLM (requires base\_url)
