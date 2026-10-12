# BEAM probability model

BEAM learns agent participation and directed communication jointly. Energy reflects task contribution and communication cost. A sparse variational Gaussian process shares information across agent pairs, while Gaussian score uncertainty controls node and edge selection.

## Energy and observations

`set_completeness(alpha)` sets each agent's retained-information fraction in `[0,1]`. The task's observation protocol supplies the corresponding masked inputs. Energy evolves between calibration batches:

```text
w_ij = E_i E_j / sqrt(E_i² + E_j² + coupling_epsilon)
E_i(next) = E_i + (energy_gain * contribution_i - energy_cost * cost_i) * dt
                  + uncertainty_scale * (1-alpha_i) * sqrt(E_i * dt) * Normal(0,1)
```

The diagonal coupling is zero. `update_energy(contributions, costs=None)` takes contributions in `[0,1]` and nonnegative outgoing-strength sums. Omitting costs uses the full coupling row sums. During training, contributions are accuracy-weighted participation and costs are coupling strengths weighted by the retained adjacency frequency. An energy proposal outside the positive domain raises a `ValueError` before changing energies or consuming the noise draw; callers can reduce `energy_dt` and retry with the same feedback.

The conditional energy noise variance is `uncertainty_scale² * (1-alpha_i)² * E_i * dt`. The GP supplies the learned edge-score variance. These quantities enter their respective energy and score models.

## Sparse variational GP

`set_graph(graph, embeddings=None)` accepts an `n × d` matrix of agent embeddings in registration order. The default is a one-hot identity embedding; callers can supply embeddings of roles or prompts. Each pair input is `(embedding_i, embedding_j, E_i, E_j)`. Energy updates change the query inputs; inducing inputs stay fixed at their initialized locations.

The RBF GP uses `q(f,u)=p(f|u)q(u)`, with `q(u)=Normal(m,S)`:

```text
B = K_XZ K_ZZ^(-1)
mean_f = B m
cov_f = K_XX - B K_ZX + B S B^T
mu_ij = sigmoid(mean_f_ij)
sigma_ij² = diag(cov_f)_ij
```

Cholesky solves retain the conditional covariance. The trainable Gaussian uses whitened coordinates `u=L v`, where `K_ZZ=L L^T` and `q(v)=Normal(a,C C^T)`. Exponentiated Cholesky diagonals keep `S` positive definite. The inducing penalty is

```text
KL(q(u)||p(u)) = (||C||_F² + ||a||² - m - 2 sum(log(diag(C)))) / 2.
```

Initialization uses `a=0, C=I`, giving score means `0.5` and variance `gp_scale²` (default `1.0`). Adam updates both variational mean and covariance from task feedback and the regularizers.

## Node and edge probabilities

```text
p_ij = Phi((mu_ij - edge_threshold) / sigma_ij)
rho_i = sum_j(mu_ij + mu_ji) + node_energy_weight * E_i
s_i² = sum_j(sigma_ij² + sigma_ji²)
r_i = Phi((rho_i - node_threshold) / s_i)
```

All sums exclude the diagonal. `node_threshold=None` selects `n-1`. Gaussian probabilities use machine-epsilon interior endpoints. Sample the node mask `z` once per batch. On its active ordered pairs, define

```text
h(A|z) = product p_ij^A_ij (1-p_ij)^(1-A_ij)
pi(A|z) = 1{A is a DAG} h(A|z) / Z(z)
P(z,A) = product r_i^z_i (1-r_i)^(1-z_i) * pi(A|z).
```

Inactive pairs are fixed to zero; every absent active-pair edge contributes its Bernoulli factor. Empty and singleton masks have a point-mass edge law.

## Conditional edge sampling

`await strategy.sample_reference(initial=None, transitions=None, nodes=None)` draws one mask or uses the supplied mask. Starting from an empty DAG (or the supplied feasible graph), each step toggles one uniformly chosen active directed edge. Cyclic proposals are rejected. The acceptance probability is `min(1, h(A_prime|z)/h(A|z))`.

Each transition contributes one retained graph, including rejected steps. The default is `mcmc_samples=10`. Node masks and score parameters remain fixed for the entire chain. `use_mcmc=False` uses independent edge draws with DAG rejection under the same fixed mask.

## Task-feedback objective

Training maximizes

```text
J = E_P[utility] - nuclear_weight * ||mu||_*
                - variance_weight * sum(sigma_ij²)
                - prior_weight * KL(q(u)||p(u)).
```

`eval_fn(prediction, answer)` supplies the utility in `[0,1]`. Each distinct retained graph is evaluated on the batch; repeated states receive the same batch score. Empty masks receive zero accuracy. With a fixed sampled mask, the utility gradient combines the mean utility times the node log-probability gradient and the centered covariance of utility with the edge log-mass gradient. Centering accounts for the conditional DAG normalizer. The finite-chain estimator averages over every retained transition.

The complete update differentiates all three penalties through the GP moments. A single Adam step updates its mean and covariance parameters together. Defaults are `learning_rate=0.001`, `num_iterations=20`, and regularizer weights `0.01`, `0.01`, `0.005`. The statistics report the score surrogate loss, task accuracy and regularized objective.

## Deployment and execution

Deployment draws an active set once and holds it across the edge draws. Candidate pairs come from the spatial mask, optionally restricted by `mu_ij > candidate_threshold`. Each candidate receives an independent `Normal(mu_ij,sigma_ij²)` draw and becomes an edge if its score exceeds `edge_threshold`.

A deterministic DFS locates a cycle and deletes its smallest-mean edge, breaking ties by pair index. With multiple draws, an edge survives if its frequency is strictly greater than `consensus_threshold`. A final repair deletes the smallest-frequency edge in each detected cycle. Active isolated agents are preserved.

`AgentGraph.arun_state` executes the selected DAG with request-local connection, output and memory buffers. Temporal communication follows the configured temporal mask on active endpoints. Agent failures propagate after retries.

## Persistence

Version-3 checkpoints save embedding rows, agent order, variational parameters, fixed inducing inputs, energies, completeness, Adam state and the private random-generator state. Restoring a checkpoint reproduces the next graph draw. Version-2 graph-offset checkpoints use a different parameterization; initialize and train the variational GP when upgrading.

Run the offline workflow and regression checks:

```bash
python -m beam.examples.bayesian_workflow
python -m pytest -q
```
