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
| `strategy` | `prune` | Set to `bayesian` for the joint model |
| `use_mcmc` | `True` | Coordinate sampling during variational calibration |
| `mcmc_samples` | `50` | Retained calibration or reference states |
| `mcmc_warmup` | `5` | Discarded transitions |
| `beta` | `1.0` | Utility/KL trade-off and reference inverse temperature |
| `communication_cost_weight` | `0.1` | Normalized spatial-edge cost coefficient |
| `deployment_samples` | `10` | Independent draws for consensus |
| `consensus_threshold` | `0.5` | Retain an edge when frequency is strictly greater |
| `random_seed` | `None` | Optional private sampler seed |
| `learning_rate` | `0.1` | Adam learning rate for graph logit offsets |
| `batch_size` | `40` | Calibration records per batch |
| `num_iterations` | `2` | Passes over calibration records |

## Energy and GP settings

| Fields | Defaults |
| --- | --- |
| `energy_min`, `energy_initial`, `energy_max` | `0.1`, `1.0`, `2.0` |
| `energy_gain`, `energy_cost`, `energy_dt` | `0.1`, `0.1`, `1.0` |
| `uncertainty_scale`, `baseline_std` | `0.1`, `0.1` |
| `coupling_epsilon`, `edge_threshold` | `1e-6`, `0.5` |
| `node_energy_weight`, `node_threshold`, `node_std` | `0.1`, `None` (uses `n−1`), `1.0` |
| `gp_inducing_points`, `gp_length_scale`, `gp_scale` | `10`, `1.0`, `1.0` |
| `gp_observation_variance`, `gp_jitter` | `0.1`, `1e-6` |

Set completeness with `strategy.set_completeness([...])`. During Bayesian calibration, both node and edge offsets are fitted. The generic `optimize_spatial` and `optimize_temporal` switches configure the separate pruning path; temporal masks remain part of the Bayesian execution protocol.

See the [probability model](../models/beam-theory.md) for each parameter's mathematical role.
