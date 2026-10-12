# FAQ

## How do I use joint agent and edge selection?

Use `AgentBayesian`, or `TokenOptimizer` with `strategy="bayesian"`. The [implementation reference](models/implementation.md) describes the core components and their tests.

## How do I run the project without an API key?

Run `python -m beam.examples.bayesian_workflow`. It uses deterministic callbacks on CPU.

## What does the completeness coefficient mean?

It is the retained fraction of the observation available to an agent. Supply one coefficient in `[0,1]` per agent using `set_completeness`. Your observation protocol determines the actual input content.

## How are reference sampling and deployment related?

The reference chain samples the DAG-conditioned Bernoulli edge law under one fixed node mask. Deployment draws candidate graphs and applies deterministic repair and frequency consensus. See the [probability model](models/beam-theory.md) for their respective laws.

## Can a sampled graph have no active agents?

Yes. The empty graph is feasible, returns no agent answer and has zero calibration accuracy. Isolated active agents also remain valid and execute normally.

## How do I run inside a notebook or asynchronous application?

Use `await strategy.atrain(...)` and `await strategy.run(...)`. The `train` and `run_sync` entry points are for synchronous scripts.

## How do I restore a model?

Construct the same agent graph, call `set_graph`, then `load(path)`. Version-3 checkpoints store the variational GP and sampler state. Initialize and train a version-3 model when upgrading from graph-offset checkpoints.
