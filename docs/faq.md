# FAQ

## Which interface implements the paper's joint model?

Use `AgentBayesian`, or `TokenOptimizer` with `strategy="bayesian"`. The [paper-to-code map](models/paper-to-code.md) lists the matching equations, algorithms and tests.

## How do I run the project without an API key?

Run `python -m beam.examples.theory_alignment`. It uses deterministic callbacks on CPU.

## What does the completeness coefficient mean?

It is the retained fraction of the observation available to an agent. Supply one coefficient in `[0,1]` per agent using `set_completeness`. Your observation protocol determines the actual input content.

## How are reference sampling and deployment related?

The reference chain targets the utility-tilted feasible graph distribution. Deployment draws candidate graphs and applies deterministic repair and frequency consensus. See the [probability model](models/beam-theory.md) for their respective laws.

## Can a sampled graph have no active agents?

Yes. The empty graph is feasible, returns no agent answer and has zero calibration accuracy. Isolated active agents also remain valid and execute normally.

## How do I run inside a notebook or asynchronous application?

Use `await strategy.atrain(...)` and `await strategy.run(...)`. The `train` and `run_sync` entry points are for synchronous scripts.

## How do I restore a model?

Construct the same agent graph, call `set_graph`, then `load(path)`. Version-2 checkpoints store the joint model and sampler state. For an older edge-only checkpoint, calibrate the joint model and save a version-2 checkpoint.
