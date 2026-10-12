# Configuration

`BEAMConfig` describes the agents and execution protocol. `OptimizationConfig` controls the probability model, calibration and deployment. `BEAMConfig.from_dict` and `BEAMConfig.from_yaml` accept the same field names.

## Core settings

| Field | Default | Meaning |
| --- | --- | --- |
| `num_rounds` | `3` | Execution rounds for a selected graph |
| `connection_mode` | `full_connected` | Generates candidate spatial and temporal masks |
| `agents` | `[]` | Agent names and counts; register the same number of graph nodes |
| `spatial_masks` | `None` | Optional deployment candidate adjacency |
| `temporal_masks` | `None` | Optional cross-round memory mask |

For custom topology, supply both masks. Node order is the insertion order of `graph.nodes`. The decision node is an output aggregator, outside the sampled agent state.

## Bayesian settings

| Field | Default | Meaning |
| --- | --- | --- |
| `strategy` | `prune` | Set `bayesian` for the joint model |
| `use_mcmc` | `True` | Single-edge sampling under one fixed node mask |
| `mcmc_samples` | `10` | Transitions retained per calibration batch |
| `deployment_samples` | `10` | Gaussian edge draws for consensus |
| `consensus_threshold` | `0.5` | Strict edge-frequency threshold |
| `candidate_threshold` | `None` | Optional strict candidate mean threshold |
| `random_seed` | `None` | Private sampler seed |
| `learning_rate` | `0.001` | Adam learning rate |
| `batch_size` | `40` | Records per batch |
| `num_iterations` | `20` | Passes over the training records |
| `nuclear_weight` | `0.01` | Nuclear norm of the mean matrix |
| `variance_weight` | `0.01` | Sum of off-diagonal score variances |
| `prior_weight` | `0.005` | Inducing-variable KL penalty |

## Energy and GP settings

| Fields | Defaults |
| --- | --- |
| `energy_initial` | `1.0` |
| `energy_gain`, `energy_cost`, `energy_dt` | `0.1`, `0.1`, `1.0` |
| `uncertainty_scale` | `0.1` |
| `coupling_epsilon`, `edge_threshold` | `1e-6`, `0.5` |
| `node_energy_weight`, `node_threshold` | `0.1`, `None` (uses `n-1`) |
| `gp_inducing_points`, `gp_length_scale`, `gp_scale` | `10`, `1.0`, `1.0` |
| `gp_jitter` | `1e-6` |

`set_graph(graph, embeddings=...)` supplies agent features. `set_completeness([...])` sets the retained observation fractions. The initial auxiliary edge-score distribution has mean `0.5` and variance `1.0` with default GP settings.

Observation masking and node/edge deletion protocols belong to task data preparation. They are distinct from Gaussian participation and edge thresholds; a deletion rate does not set a completeness coefficient or Gaussian threshold automatically. `dropout_rate` controls the separate `AgentDropout` strategy.

The `optimize_spatial` and `optimize_temporal` switches configure the pruning strategy. Temporal masks remain part of the Bayesian execution protocol.

## Upgrading

The GP variational parameters replace the version-2 graph offsets. Remove `beta`, `communication_cost_weight`, `mcmc_warmup`, `energy_min`, `energy_max`, `baseline_std`, `node_std` and `gp_observation_variance` from old Bayesian configurations. `sample_reference()` accepts a fixed mask through `nodes=...`; task feedback is passed to `train` or `atrain` through `eval_fn`.

[Probability model](../models/beam-theory.md)
