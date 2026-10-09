# BEAM probability model and implementation

BEAM jointly chooses participating agents and directed communication edges. A positive energy proxy integrates contribution and communication cost. Pairwise Gaussian-process moments express interaction uncertainty; calibration learns a graph distribution, and deployment constructs an executable DAG.

## State and probabilities

`GraphState(nodes, edges)` represents `x = (z, A)` in the insertion order of `graph.nodes`. The diagonal of `A` is zero, every edge has active endpoints, and `A` is acyclic. Isolated active agents remain part of the state. The empty state returns no agent answer and receives zero calibration accuracy.

For `n` agents, the reference distribution uses `n²` bits: first `n` node bits, then `n(n−1)` ordered off-diagonal edge bits. `bernoulli_log_mass` includes both present and absent bits. Conditioning this mass on feasible states defines `p0`; neither MH nor the covariance gradient needs its normalizer.

The energy update is

```text
E_i(next) = clip(E_i + (energy_gain * contribution_i - energy_cost * cost_i) * dt
                    + uncertainty_scale * (1-alpha_i) * sqrt(E_i * dt) * Normal(0,1),
                energy_min, energy_max)
```

`set_completeness(alpha)` sets the observed retained fraction, with `0 <= alpha_i <= 1`. The caller supplies observations produced by its masking or observation protocol. `update_energy(contributions, costs)` accepts per-agent scores in `[0,1]`; the built-in calibration uses accuracy-weighted participation and normalized outgoing degree. Energy updates occur between calibration batches.

For off-diagonal pairs:

```text
m_ij = normalized_coupling(E_i, E_j) + GP_mean_ij
s_ij² = GP_variance_ij + baseline_std²
        + uncertainty_scale² * ((1-alpha_i)² + (1-alpha_j)²)
p_ij = Phi((m_ij - edge_threshold) / s_ij)
r_i  = Phi((sum_j(p_ij + p_ji) + node_energy_weight * E_i - node_threshold) / node_std)
```

`node_threshold` defaults to `n−1`. Gaussian probabilities are clipped only at machine-epsilon endpoints to retain positive finite masses. Learned logit offsets parameterize the variational family relative to this structural reference.

## Pairwise GP

`SparsePairGP` uses the ordered features `(i/(n−1), j/(n−1))`, an RBF kernel and a deterministic subset of inducing inputs. Its variational joint is `q(f,v) = p(f|v) q(v)` with Gaussian `q(v)`. Prediction retains the conditional covariance:

```text
B = K_XZ K_ZZ^(-1)
mean = B m
covariance = K_XX - B K_ZX + B S B^T
```

Cholesky solves implement the matrix operations. Before observations, `m=0` and `S=K_ZZ`, so the full prior variance is recovered even with fewer inducing inputs than pair inputs. Calibration forms one empirical contribution score per observed pair by averaging its active-state rewards. A Gaussian observation model gives closed-form inducing mean and covariance updates. GP configuration controls the kernel length scale, amplitude, observation variance, inducing count and diagonal jitter.

## Fixed-target reference chain

`await strategy.sample_reference(utility)` samples the target

```text
pi_beta(x) proportional to p0(x) * exp(beta * utility(x)).
```

The callback must specify a fixed deterministic utility in `[0,1]`, and can be synchronous or asynchronous. The sampler copies all logits at chain creation and memoizes each state's utility. Each transition stays put with probability `1/2`; otherwise it uniformly chooses one of the `n²` coordinates and flips that bit. Infeasible proposals are rejected. For a feasible neighbor `y`, the log acceptance threshold is

```text
min(0, log_base_mass(y) - log_base_mass(x) + beta * (utility(y)-utility(x))).
```

Node deactivation with incident edges is rejected as a single-bit proposal. No edges are silently deleted by the reference sampler. Self-transitions are retained. `transitions` includes burn-in; the returned length is `transitions - burn_in`. The configuration defaults use `mcmc_warmup + mcmc_samples` transitions to retain exactly `mcmc_samples` states.

## KL-regularized calibration

`train` / `atrain` fit `F_beta(q) = E_q[u] - KL(q || p0)/beta`. The built-in fixed-batch utility is

```text
u(x) = (accuracy(x) - lambda * edge_density(x) + lambda) / (1+lambda),
edge_density(x) = sum(A_ij) / (n*(n-1)).
```

`eval_fn(prediction, answer)` supplies a score in `[0,1]`. Each distinct sampled graph is evaluated once on the batch and its outputs and score are reused within that update. Both node and edge offsets receive the centered covariance gradient:

```text
Cov_q(u - (log_base_q - log_base_p0)/beta, grad(log_base_q)).
```

Centering cancels both DAG normalizers. With `use_mcmc=True`, the lazy coordinate kernel samples the current constrained variational distribution. With `False`, independent Bernoulli proposals are accepted only when already feasible; a finite rejection budget reports when coordinate sampling is needed. Both paths fit the same objective. Sampling the utility-tilted reference target is available separately through `sample_reference`.

`accuracies` and `utilities` summarize retained variational states. `total_executed` counts evaluated graph/example combinations, including the empty state's defined zero score. GP and energy feedback is applied after the graph gradient step.

## Deployment

`await strategy.run(inputs, num_rounds=...)` draws nodes and candidate edges independently, removes edges with inactive endpoints, then repairs cycles. A deterministic DFS selects the first cycle, and its lowest-score edge is removed; ties use `(sender_index, receiver_index)`.

With multiple samples, active nodes are their union and edges are retained exactly when their frequency is **strictly greater** than `consensus_threshold`. The consensus graph receives the same cycle repair, using edge frequencies as scores. `use_mean=True` selects `deployment_samples` draws; `False` selects one draw.

The configured spatial mask defines the deployment candidate set. Reference calibration has full feasible support; deployment applies that candidate set. This gives the reference target and the repaired deployment distribution explicit, separate definitions.

`AgentGraph.arun_state` executes the selected state directly. It does not resample edges or execute inactive agents. Each call uses separate connection, output and memory buffers, while sharing the LLM clients and execution functions. Across multiple rounds, temporal links follow the fixed configured temporal mask restricted to active endpoints. The Bayesian objective charges spatial edge density; temporal masks are part of the fixed execution protocol. Agent failures propagate after retries.

## Entry points and saved state

- `AgentBayesian(config).set_graph(graph)` enables direct control of completeness, energies, calibration and reference sampling.
- `TokenOptimizer` dispatches `strategy="bayesian"` training, inference and persistence to `AgentBayesian`.
- `save` / `load` preserve joint logits, GP moments, energies, completeness, optimizer state and random-generator state. Restore into a graph with the same agent order and execution protocol.
- Version-2 checkpoints encode the joint state; older edge-only checkpoints need recalibration with this model.
- Core PyTorch supplies the GP and discrete sampler. `pip install -e '.[bayesian]'` remains accepted as an installation alias.

Run the complete offline example:

```bash
python -m beam.examples.bayesian_workflow
python -m pytest -q
```

Tests enumerate all 38 feasible three-agent states and check detailed balance, stationarity, the multi-step minorization, and the centered gradient against the exact finite-state objective. Additional tests cover GP conditional covariance, energy bounds, strict consensus thresholds, request isolation and checkpoint continuation.
