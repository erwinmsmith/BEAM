"""BEAM: joint node/edge calibration and uncertainty-aware DAG deployment."""

import asyncio
from dataclasses import asdict
import math
from typing import Any, Dict, List, Optional

import torch

from beam.core.config import BEAMConfig, OptimizationConfig
from beam.core.graph import AgentGraph
from beam.core.probability import (
    CoordinateSampler,
    GraphState,
    SparsePairGP,
    conditioned_samples,
    covariance_loss,
    deployment_state,
    interior_cdf,
    pairs,
    validate_unit,
)


class AgentBayesian:
    """Fit the KL-regularized graph objective and deploy repaired graph samples.

    Calibration fixes the structural reference and caches task utility for each
    state within a batch. GP and energy updates occur between calibration batches.
    ``sample_reference`` exposes the fixed-target chain from Algorithm 1;
    ``run`` uses the independent draws and consensus from Algorithm 2.
    """

    def __init__(self, config: BEAMConfig):
        self.config = config
        self.graph: Optional[AgentGraph] = None
        self.training_history: List[Dict[str, Any]] = []
        self.observation_history: List[Dict[str, Any]] = []
        self.optimizer = None
        self.gp = None
        self.edge_means = self.edge_vars = None
        self._validate_config()
        self.generator = torch.Generator()
        if config.optimization.random_seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(config.optimization.random_seed)

    def _validate_config(self):
        c = self.config.optimization
        positive = [
            "beta",
            "energy_min",
            "energy_max",
            "energy_dt",
            "baseline_std",
            "coupling_epsilon",
            "node_std",
            "gp_length_scale",
            "gp_scale",
            "gp_observation_variance",
            "gp_jitter",
            "learning_rate",
        ]
        nonnegative = [
            "communication_cost_weight",
            "energy_gain",
            "energy_cost",
            "uncertainty_scale",
            "node_energy_weight",
        ]
        for name in positive + nonnegative:
            value = getattr(c, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid {name}")
        if not c.energy_min < c.energy_max or not c.energy_min <= c.energy_initial <= c.energy_max:
            raise ValueError(
                "Require 0 < energy_min <= energy_initial <= energy_max, with distinct bounds"
            )
        for name in [
            "mcmc_samples",
            "deployment_samples",
            "gp_inducing_points",
            "batch_size",
            "num_iterations",
        ]:
            if not isinstance(getattr(c, name), int) or getattr(c, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if c.mcmc_samples < 2 or not isinstance(c.mcmc_warmup, int) or c.mcmc_warmup < 0:
            raise ValueError("Require mcmc_samples >= 2 and mcmc_warmup >= 0")
        if not 0 <= c.consensus_threshold <= 1 or not math.isfinite(c.edge_threshold):
            raise ValueError("Invalid graph threshold")
        if c.node_threshold is not None and not math.isfinite(c.node_threshold):
            raise ValueError("Node threshold must be finite")

    def set_graph(self, graph: AgentGraph):
        n = len(graph.nodes)
        if n < 2 or n != len(graph.agent_names):
            raise ValueError("Register all configured agents (at least two) before set_graph")
        self.graph, self.n = graph, n
        self.training_history = []
        self.observation_history = []
        self.pairs = pairs(n)
        self.features = torch.tensor(self.pairs, dtype=torch.float64) / (n - 1)
        c = self.config.optimization
        self.gp = SparsePairGP(
            self.features,
            c.gp_inducing_points,
            c.gp_length_scale,
            c.gp_scale,
            c.gp_observation_variance,
            c.gp_jitter,
        )
        self.energy = torch.full((n,), c.energy_initial, dtype=torch.float64)
        self.completeness = torch.ones(n, dtype=torch.float64)
        self.offsets = torch.nn.Parameter(torch.zeros(n * n, dtype=torch.float64))
        self.optimizer = torch.optim.Adam([self.offsets], lr=c.learning_rate)
        self._refresh_scores()

    def _require_graph(self):
        if self.graph is None:
            raise ValueError("Call set_graph with registered agents first")

    def set_completeness(self, completeness):
        """Set the retained information fraction for each agent in the next round."""
        self._require_graph()
        values = validate_unit(completeness, "Completeness")
        if values.shape != (self.n,):
            raise ValueError("One completeness coefficient is required per agent")
        self.completeness = values.clone()
        self._refresh_scores()

    def update_energy(self, contributions, costs):
        """Apply the clipped energy transition (Eq. 3), between fixed-target runs."""
        self._require_graph()
        contribution = validate_unit(contributions, "Contributions")
        cost = validate_unit(costs, "Costs")
        if contribution.shape != (self.n,) or cost.shape != (self.n,):
            raise ValueError("Contribution and cost vectors must match the agents")
        c = self.config.optimization
        drift = c.energy_gain * contribution - c.energy_cost * cost
        noise = torch.randn(self.n, dtype=torch.float64, generator=self.generator)
        updated = (
            self.energy
            + drift * c.energy_dt
            + (
                c.uncertainty_scale
                * (1 - self.completeness)
                * torch.sqrt(self.energy * c.energy_dt)
                * noise
            )
        )
        self.energy = updated.clamp(c.energy_min, c.energy_max)
        self._refresh_scores()

    def _refresh_scores(self):
        c = self.config.optimization
        mean, variance = self.gp.moments()
        e = self.energy
        normalizer = c.energy_max**2 / math.sqrt(2 * c.energy_max**2 + c.coupling_epsilon)
        coupling = (
            e[:, None]
            * e[None, :]
            / (normalizer * torch.sqrt(e[:, None] ** 2 + e[None, :] ** 2 + c.coupling_epsilon))
        )
        self.edge_means = coupling.clone()
        self.edge_vars = torch.zeros_like(coupling)
        for k, (i, j) in enumerate(self.pairs):
            self.edge_means[i, j] += mean[k]
            self.edge_vars[i, j] = (
                variance[k]
                + c.baseline_std**2
                + c.uncertainty_scale**2
                * ((1 - self.completeness[i]) ** 2 + (1 - self.completeness[j]) ** 2)
            )
        p = torch.zeros_like(coupling)
        rows, cols = zip(*self.pairs)
        p[rows, cols] = interior_cdf(
            self.edge_means[rows, cols], self.edge_vars[rows, cols], c.edge_threshold
        )
        node_mean = p.sum(0) + p.sum(1) + c.node_energy_weight * e
        threshold = self.n - 1 if c.node_threshold is None else c.node_threshold
        r = interior_cdf(node_mean, torch.full_like(e, c.node_std**2), threshold)
        self.prior_logits = torch.logit(torch.cat((r, p[rows, cols]))).detach()

    def probabilities(self, fitted=True):
        """Node and off-diagonal edge probabilities, in agent insertion order."""
        self._require_graph()
        logits = self.prior_logits + self.offsets.detach() if fitted else self.prior_logits
        eps = torch.finfo(logits.dtype).eps
        probabilities = logits.sigmoid().clamp(eps, 1 - eps)
        edges = torch.zeros((self.n, self.n), dtype=torch.float64)
        rows, cols = zip(*self.pairs)
        edges[rows, cols] = probabilities[self.n :]
        return probabilities[: self.n], edges

    async def sample_reference(self, utility, initial=None, transitions=None, burn_in=None):
        """Sample pi_beta proportional to p0 exp(beta*u), caching u per state.

        ``utility(state)`` is a deterministic [0,1] callback (sync or async).
        ``transitions`` counts all transitions; ``burn_in`` counts discarded ones.
        All structural logits are copied when the chain starts.
        """
        self._require_graph()
        c = self.config.optimization
        burn_in = c.mcmc_warmup if burn_in is None else burn_in
        transitions = burn_in + c.mcmc_samples if transitions is None else transitions
        sampler = CoordinateSampler(self.prior_logits, self.n, utility, c.beta, self.generator)
        return await sampler.sample(transitions, burn_in, initial)

    async def _variational_samples(self):
        c = self.config.optimization
        if c.use_mcmc:
            sampler = CoordinateSampler(
                self.prior_logits + self.offsets.detach(), self.n, generator=self.generator
            )
            return await sampler.sample(c.mcmc_warmup + c.mcmc_samples, c.mcmc_warmup)
        nodes, edges = self.probabilities()
        return conditioned_samples(nodes, edges, c.mcmc_samples, self.generator)

    async def _train_batch(self, batch, eval_fn):
        self._require_graph()
        if not batch:
            raise ValueError("Calibration batches must be nonempty")
        c = self.config.optimization
        prior = self.prior_logits.clone()
        states = await self._variational_samples()
        utilities, accuracies = {}, {}
        observations = []
        for state in dict.fromkeys(states):
            scores = []
            for record in batch:
                if any(state.nodes):
                    answers, _ = await self.graph.arun_state(
                        {"task": record["task"]}, state, self.config.num_rounds
                    )
                    score = float(eval_fn(answers[0] if answers else "", record["answer"]))
                    validate_unit(score, "Calibration score")
                else:
                    score = 0.0
                scores.append(score)
            accuracy = sum(scores) / len(scores)
            cost = sum(sum(row) for row in state.edges) / len(self.pairs)
            utilities[state] = (
                accuracy - c.communication_cost_weight * cost + c.communication_cost_weight
            ) / (1 + c.communication_cost_weight)
            accuracies[state] = accuracy
            observations.append(
                {
                    "nodes": list(state.nodes),
                    "active_edges": [i * self.n + j for i, j in self.pairs if state.edges[i][j]],
                    "reward": accuracy,
                }
            )
        logits = prior + self.offsets
        loss = covariance_loss(logits, states, [utilities[x] for x in states], prior, c.beta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if not torch.isfinite(self.offsets).all():
            raise RuntimeError("Non-finite fitted graph logits")
        self.observation_history.extend(observations)
        self._fit_gp()
        # Contribution and cost feedback use the same retained calibration states.
        contributions = torch.tensor(
            [[accuracies[x] * bit for bit in x.nodes] for x in states], dtype=torch.float64
        ).mean(0)
        costs = torch.tensor(
            [[sum(row) / (self.n - 1) for row in x.edges] for x in states], dtype=torch.float64
        ).mean(0)
        self.update_energy(contributions, costs)
        return {
            "loss": float(loss.detach()),
            "accuracy": sum(accuracies[x] for x in states) / len(states),
            "utility": sum(utilities[x] for x in states) / len(states),
            "solved": sum(accuracies.values()) * len(batch),
            "executed": len(utilities) * len(batch),
        }

    def _fit_gp(self):
        # Sufficient statistics per pair keep inducing updates independent of history length.
        sums = torch.zeros(len(self.pairs), dtype=torch.float64)
        counts = torch.zeros_like(sums)
        index = {i * self.n + j: k for k, (i, j) in enumerate(self.pairs)}
        for observation in self.observation_history:
            for edge in observation["active_edges"]:
                k = index[edge]
                sums[k] += observation["reward"]
                counts[k] += 1
        observed = counts > 0
        if observed.any():
            # One empirical contribution score per ordered pair is the GP observation.
            self.gp.fit(self.features[observed], sums[observed] / counts[observed])
        self._refresh_scores()

    def train(self, train_data, eval_fn, num_iterations=None, batch_size=None, verbose=True):
        """Synchronous calibration; async applications can await ``atrain``."""
        return asyncio.run(self.atrain(train_data, eval_fn, num_iterations, batch_size, verbose))

    async def atrain(self, train_data, eval_fn, num_iterations=None, batch_size=None, verbose=True):
        self._require_graph()
        c = self.config.optimization
        iterations = c.num_iterations if num_iterations is None else num_iterations
        size = c.batch_size if batch_size is None else batch_size
        if (
            not train_data
            or not isinstance(iterations, int)
            or iterations < 1
            or not isinstance(size, int)
            or size < 1
        ):
            raise ValueError(
                "Training data, iteration count and batch size must be nonempty/positive"
            )
        stats = {
            "total_solved": 0.0,
            "total_executed": 0,
            "losses": [],
            "accuracies": [],
            "utilities": [],
            "edge_uncertainty": [],
        }
        for iteration in range(iterations):
            for start in range(0, len(train_data), size):
                result = await self._train_batch(train_data[start : start + size], eval_fn)
                stats["total_solved"] += result["solved"]
                stats["total_executed"] += result["executed"]
                stats["losses"].append(result["loss"])
                stats["accuracies"].append(result["accuracy"])
                stats["utilities"].append(result["utility"])
                rows, cols = zip(*self.pairs)
                stats["edge_uncertainty"].append(float(self.edge_vars[rows, cols].mean()))
                if verbose:
                    print(
                        f"Iteration {iteration + 1}: accuracy={result['accuracy']:.4f}, utility={result['utility']:.4f}"
                    )
        self.training_history.append(stats)
        return stats

    async def run(self, inputs, num_rounds=None, use_mean=True):
        """Deploy Algorithm 2; use_mean=True aggregates deployment_samples draws."""
        self._require_graph()
        c = self.config.optimization
        node_probs, edge_probs = self.probabilities()
        state = deployment_state(
            node_probs,
            edge_probs,
            self.graph.fixed_spatial_masks,
            c.deployment_samples if use_mean else 1,
            c.consensus_threshold,
            self.generator,
        )
        answers, _ = await self.graph.arun_state(inputs, state, num_rounds)
        metadata = {
            "strategy": "bayesian",
            "sampling": "independent-repair-consensus",
            "active_nodes": sum(state.nodes),
            "active_edges": sum(map(sum, state.edges)),
            "node_mask": list(state.nodes),
            "adjacency": [list(row) for row in state.edges],
            "calibration_mcmc": c.use_mcmc,
            "deployment_samples": c.deployment_samples if use_mean else 1,
        }
        return answers, metadata

    def run_sync(self, inputs, num_rounds=None):
        return asyncio.run(self.run(inputs, num_rounds))

    def get_edge_statistics(self):
        if self.graph is None:
            return {}
        r, p = self.probabilities()
        rows, cols = zip(*self.pairs)
        return {
            "spatial": {
                "mean": float(self.edge_means[rows, cols].mean()),
                "std": float(self.edge_means[rows, cols].std()),
                "uncertainty_mean": float(self.edge_vars[rows, cols].mean()),
                "probabilities": p.tolist(),
            },
            "node_probabilities": r.tolist(),
            "energy": self.energy.tolist(),
            "completeness": self.completeness.tolist(),
            "observations": len(self.observation_history),
        }

    def save(self, path):
        self._require_graph()
        optimization = asdict(self.config.optimization)
        strategy = optimization["strategy"]
        optimization["strategy"] = getattr(strategy, "value", strategy)
        torch.save(
            {
                "version": 2,
                "n": self.n,
                "optimization": optimization,
                "offsets": self.offsets.detach(),
                "energy": self.energy,
                "completeness": self.completeness,
                "gp_mean": self.gp.mean,
                "gp_covariance": self.gp.covariance,
                "rng": self.generator.get_state(),
                "optimizer": self.optimizer.state_dict(),
                "training_history": self.training_history,
                "observation_history": self.observation_history,
            },
            path,
        )

    def load(self, path):
        self._require_graph()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version") != 2 or state["n"] != self.n:
            raise ValueError(
                "Expected a version-2 joint node/edge checkpoint with matching agent count"
            )
        config = OptimizationConfig(**state["optimization"])
        old_config = self.config.optimization
        self.config.optimization = config
        try:
            self._validate_config()
        except (ValueError, TypeError):
            self.config.optimization = old_config
            raise
        self.set_graph(self.graph)
        for name in ["energy", "completeness"]:
            value = state[name]
            if value.shape != (self.n,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint {name}")
        validate_unit(state["completeness"], "Checkpoint completeness")
        if ((state["energy"] < config.energy_min) | (state["energy"] > config.energy_max)).any():
            raise ValueError("Checkpoint energies are outside configured bounds")
        if (
            state["offsets"].shape != self.offsets.shape
            or not torch.isfinite(state["offsets"]).all()
        ):
            raise ValueError("Invalid checkpoint logits")
        if (
            state["gp_mean"].shape != self.gp.mean.shape
            or state["gp_covariance"].shape != self.gp.covariance.shape
        ):
            raise ValueError("Invalid checkpoint GP dimensions")
        if (
            not torch.isfinite(state["gp_mean"]).all()
            or not torch.isfinite(state["gp_covariance"]).all()
        ):
            raise ValueError("Invalid checkpoint GP moments")
        covariance = state["gp_covariance"]
        if (
            not torch.allclose(covariance, covariance.T)
            or torch.linalg.eigvalsh(covariance).min() < -1e-10
        ):
            raise ValueError("Checkpoint GP covariance must be positive semidefinite")
        with torch.no_grad():
            self.offsets.copy_(state["offsets"])
        self.energy, self.completeness = state["energy"], state["completeness"]
        self.gp.mean, self.gp.covariance = state["gp_mean"], state["gp_covariance"]
        self.generator.set_state(state["rng"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.observation_history = state["observation_history"]
        self.training_history = state["training_history"]
        self._refresh_scores()
