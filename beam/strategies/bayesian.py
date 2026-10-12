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
    SparsePairGP,
    conditioned_samples,
    score_loss,
    deployment_state,
    interior_cdf,
    pairs,
    validate_unit,
)


class AgentBayesian:
    """Learn Gaussian graph scores with task utility and three regularizers."""

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
            "energy_initial",
            "energy_dt",
            "coupling_epsilon",
            "gp_length_scale",
            "gp_scale",
            "gp_jitter",
            "learning_rate",
        ]
        nonnegative = [
            "energy_gain",
            "energy_cost",
            "uncertainty_scale",
            "node_energy_weight",
            "nuclear_weight",
            "variance_weight",
            "prior_weight",
        ]
        for name in positive + nonnegative:
            value = getattr(c, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid {name}")
        for name in [
            "mcmc_samples",
            "deployment_samples",
            "gp_inducing_points",
            "batch_size",
            "num_iterations",
        ]:
            if not isinstance(getattr(c, name), int) or getattr(c, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= c.consensus_threshold <= 1 or not math.isfinite(c.edge_threshold):
            raise ValueError("Invalid graph threshold")
        for name in ["node_threshold", "candidate_threshold"]:
            value = getattr(c, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def set_graph(self, graph: AgentGraph, embeddings=None):
        """Register agents and their embedding rows in graph insertion order.

        Omitted embeddings use one-hot agent identities. Supply semantic agent
        embeddings to share information by role or prompt similarity.
        """
        n = len(graph.nodes)
        if n < 2 or n != len(graph.agent_names):
            raise ValueError("Register all configured agents (at least two) before set_graph")
        embedding = (
            torch.eye(n, dtype=torch.float64)
            if embeddings is None
            else torch.as_tensor(embeddings, dtype=torch.float64)
        )
        if (
            embedding.ndim != 2
            or embedding.shape[0] != n
            or embedding.shape[1] < 1
            or not torch.isfinite(embedding).all()
        ):
            raise ValueError("Embeddings must be a finite n by d matrix")
        self.graph, self.n = graph, n
        self.embeddings = embedding.detach().clone()
        self.training_history, self.observation_history = [], []
        self.pairs = pairs(n)
        c = self.config.optimization
        self.energy = torch.full((n,), c.energy_initial, dtype=torch.float64)
        self.completeness = torch.ones(n, dtype=torch.float64)
        self.gp = SparsePairGP(
            self._pair_features(), c.gp_inducing_points, c.gp_length_scale, c.gp_scale, c.gp_jitter
        )
        self.optimizer = torch.optim.Adam(self.gp.parameters(), lr=c.learning_rate)
        self._refresh_scores()

    def _pair_features(self):
        rows, cols = zip(*self.pairs)
        return torch.cat(
            (
                self.embeddings[list(rows)],
                self.embeddings[list(cols)],
                self.energy[list(rows), None],
                self.energy[list(cols), None],
            ),
            dim=1,
        )

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

    def coupling(self):
        """Unnormalized energy communication strengths with zero diagonal."""
        e = self.energy
        strength = (
            e[:, None]
            * e[None, :]
            / torch.sqrt(
                e[:, None].square()
                + e[None, :].square()
                + self.config.optimization.coupling_epsilon
            )
        )
        return strength * (1 - torch.eye(self.n, dtype=e.dtype))

    def update_energy(self, contributions, costs=None):
        """Apply the square-root transition on the positive-energy domain.

        costs are sums of outgoing communication strengths. An omitted vector
        uses the full coupling row sums. A nonpositive proposal raises before
        changing the energy state.
        """
        self._require_graph()
        contribution = validate_unit(contributions, "Contributions")
        cost = (
            self.coupling().sum(1) if costs is None else torch.as_tensor(costs, dtype=torch.float64)
        )
        if contribution.shape != (self.n,) or cost.shape != (self.n,):
            raise ValueError("Contribution and cost vectors must match the agents")
        if not torch.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Communication costs must be finite and nonnegative")
        c = self.config.optimization
        rng = self.generator.get_state()
        noise = torch.randn(self.n, dtype=torch.float64, generator=self.generator)
        updated = self.energy + (c.energy_gain * contribution - c.energy_cost * cost) * c.energy_dt
        updated += (
            c.uncertainty_scale
            * (1 - self.completeness)
            * torch.sqrt(self.energy * c.energy_dt)
            * noise
        )
        if not torch.isfinite(updated).all() or (updated <= 0).any():
            self.generator.set_state(rng)
            raise ValueError(
                "Energy transition leaves the positive domain; reduce energy_dt or adjust feedback"
            )
        self.energy = updated
        self._refresh_scores()

    def _score_moments(self):
        mean, variance = self.gp.moments(self._pair_features())
        rows, cols = zip(*self.pairs)
        means = torch.zeros((self.n, self.n), dtype=torch.float64)
        variances = torch.zeros_like(means)
        means[rows, cols] = mean.sigmoid()
        variances[rows, cols] = variance
        return means, variances

    def _probabilities(self, means, variances):
        c = self.config.optimization
        rows, cols = zip(*self.pairs)
        edges = torch.zeros_like(means)
        edges[rows, cols] = interior_cdf(means[rows, cols], variances[rows, cols], c.edge_threshold)
        importance = means.sum(0) + means.sum(1) + c.node_energy_weight * self.energy
        uncertainty = variances.sum(0) + variances.sum(1)
        threshold = self.n - 1 if c.node_threshold is None else c.node_threshold
        return interior_cdf(importance, uncertainty, threshold), edges

    def _refresh_scores(self):
        with torch.no_grad():
            self.edge_means, self.edge_vars = self._score_moments()

    def probabilities(self):
        """Gaussian node and edge exceedance probabilities."""
        self._require_graph()
        with torch.no_grad():
            self.edge_means, self.edge_vars = self._score_moments()
            return self._probabilities(self.edge_means, self.edge_vars)

    def _draw_nodes(self, probabilities):
        return tuple(
            (
                torch.rand(self.n, dtype=torch.float64, generator=self.generator) < probabilities
            ).tolist()
        )

    async def sample_reference(self, initial=None, transitions=None, nodes=None):
        """Draw one node mask and retain every fixed-mask single-edge transition."""
        self._require_graph()
        r, p = self.probabilities()
        if nodes is None:
            nodes = initial.nodes if initial is not None else self._draw_nodes(r)
        sampler = CoordinateSampler(p, nodes, self.generator)
        return await sampler.sample(
            self.config.optimization.mcmc_samples if transitions is None else transitions, initial
        )

    async def _variational_samples(self):
        c = self.config.optimization
        r, p = self.probabilities()
        nodes = self._draw_nodes(r)
        if c.use_mcmc:
            return await CoordinateSampler(p, nodes, self.generator).sample(c.mcmc_samples)
        return conditioned_samples(nodes, p, c.mcmc_samples, self.generator)

    def _regularization(self, means, variances):
        c = self.config.optimization
        return (
            c.nuclear_weight * torch.linalg.matrix_norm(means, ord="nuc")
            + c.variance_weight * variances.sum()
            + c.prior_weight * self.gp.kl_divergence()
        )

    async def _train_batch(self, batch, eval_fn):
        self._require_graph()
        if not batch:
            raise ValueError("Calibration batches must be nonempty")
        states = await self._variational_samples()
        utilities = {}
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
            utilities[state] = sum(scores) / len(scores)
        means, variances = self._score_moments()
        r, p = self._probabilities(means, variances)
        penalty = self._regularization(means, variances)
        loss = score_loss(r, p, states, [utilities[x] for x in states]) + penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if any(not torch.isfinite(param).all() for param in self.gp.parameters()):
            raise RuntimeError("Non-finite GP variational parameters")
        self.observation_history.extend(
            {
                "nodes": list(x.nodes),
                "active_edges": [i * self.n + j for i, j in self.pairs if x.edges[i][j]],
                "reward": utilities[x],
            }
            for x in utilities
        )
        contributions = torch.tensor(
            [[utilities[x] * bit for bit in x.nodes] for x in states], dtype=torch.float64
        ).mean(0)
        adjacency = torch.tensor([x.edges for x in states], dtype=torch.float64).mean(0)
        costs = (adjacency * self.coupling()).sum(1)
        self.update_energy(contributions, costs)
        accuracy = sum(utilities[x] for x in states) / len(states)
        return {
            "loss": float(loss.detach()),
            "accuracy": accuracy,
            "utility": accuracy,
            "objective": accuracy - float(penalty.detach()),
            "solved": sum(utilities.values()) * len(batch),
            "executed": len(utilities) * len(batch),
        }

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
            "objectives": [],
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
                stats["objectives"].append(result["objective"])
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
        node_probs, _ = self.probabilities()
        nodes = self._draw_nodes(node_probs)
        candidates = self.graph.fixed_spatial_masks.bool().clone()
        if c.candidate_threshold is not None:
            candidates &= self.edge_means > c.candidate_threshold
        state = deployment_state(
            nodes,
            self.edge_means,
            self.edge_vars,
            candidates,
            c.deployment_samples if use_mean else 1,
            c.consensus_threshold,
            c.edge_threshold,
            self.generator,
        )
        answers, _ = await self.graph.arun_state(inputs, state, num_rounds)
        metadata = {
            "strategy": "bayesian",
            "sampling": "gaussian-repair-consensus",
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
                "version": 3,
                "n": self.n,
                "optimization": optimization,
                "agent_ids": list(self.graph.nodes),
                "embeddings": self.embeddings,
                "energy": self.energy,
                "completeness": self.completeness,
                "gp": self.gp.state_dict(),
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
        if state.get("version") != 3 or state["n"] != self.n:
            raise ValueError(
                "Expected a version-3 variational GP checkpoint with matching agent count"
            )
        if state["agent_ids"] != list(self.graph.nodes):
            raise ValueError("Checkpoint agent order must match the registered graph")
        for name in ["energy", "completeness"]:
            value = state[name]
            if value.shape != (self.n,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint {name}")
        validate_unit(state["completeness"], "Checkpoint completeness")
        if (state["energy"] <= 0).any():
            raise ValueError("Checkpoint energies must be positive")
        config = BEAMConfig(optimization=OptimizationConfig(**state["optimization"]))
        restored = AgentBayesian(config)
        restored.set_graph(self.graph, state["embeddings"])
        restored.gp.load_state_dict(state["gp"])
        if any(not torch.isfinite(x).all() for x in restored.gp.state_dict().values()):
            raise ValueError("Checkpoint GP tensors must be finite")
        restored.energy, restored.completeness = state["energy"], state["completeness"]
        restored.generator.set_state(state["rng"])
        restored.optimizer.load_state_dict(state["optimizer"])
        restored.training_history, restored.observation_history = (
            state["training_history"],
            state["observation_history"],
        )
        restored._refresh_scores()
        self.config.optimization = config.optimization
        for name in [
            "embeddings",
            "energy",
            "completeness",
            "gp",
            "generator",
            "optimizer",
            "training_history",
            "observation_history",
            "edge_means",
            "edge_vars",
        ]:
            setattr(self, name, getattr(restored, name))
