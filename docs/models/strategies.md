# Strategies

BEAM provides three specialized optimization strategies designed to balance performance and token efficiency. These strategies dynamically adjust the agent network by removing redundant paths, skipping unnecessary agents, or using probabilistic models to find the optimal configuration.

---

## AgentPrune

Learns edge importance through policy gradient optimization and prunes low-importance connections.

**Training:**
```python
from beam import AgentPrune, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="prune",
        optimize_spatial=True,
        optimize_temporal=True,
        pruning_rate=0.3,        # Prune 30% of edges
        initial_probability=0.5, # Initial edge probability
    )
)

strategy = AgentPrune(config)
```
**Best for:**
- Dense agent networks where many connections are redundant.

**How it works:**
- Initializes learnable logits for each potential edge
- Uses Gumbel-Softmax for differentiable edge sampling
- Optimizes via policy gradient based on task performance
- Prunes edges below learned threshold
---

## AgentDropout

Learns which agents can be skipped entirely during inference.
```python
from beam import AgentDropout, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="dropout",
        optimize_spatial=True,
        dropout_rate=0.2,        # Target 20% agent skip rate
    )
)

strategy = AgentDropout(config)
```
**Best for:**
- Systems with redundant agents where some can be skipped without quality loss.

**How it works::**
- Learns skip probability for each agent
- During training, samples skip decisions
- Agents with high skip probability are dropped during inference
- Maintains ensemble diversity while reducing computation
---

## AgentBayesian

Uses Bayesian optimization with optional MCMC sampling for uncertainty-aware pruning.

```python
from beam import AgentBayesian, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="bayesian",
        optimize_spatial=True,
        optimize_temporal=True,
        use_mcmc=True,           # Enable MCMC sampling
        mcmc_samples=100,        # Number of MCMC samples
        prior_mean=0.5,          # Prior edge probability
    )
)

strategy = AgentBayesian(config)
```

**Best for:**
- When you need confidence estimates or have limited training data.

**How it works:**
- Maintains posterior distribution over edge weights
- Uses MCMC (optional) to sample from posterior
- Provides uncertainty estimates for pruning decisions
- More robust with small datasets

---
