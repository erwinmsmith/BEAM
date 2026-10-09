"""Finite node/edge states, sparse GP moments, and BEAM graph distributions."""

from dataclasses import dataclass
import inspect
import math
from typing import Tuple

import torch


@dataclass(frozen=True)
class GraphState:
    """A node mask z and an adjacency matrix A, in graph insertion order."""

    nodes: Tuple[bool, ...]
    edges: Tuple[Tuple[bool, ...], ...]

    def __post_init__(self):
        n = len(self.nodes)
        if n < 2 or len(self.edges) != n or any(len(row) != n for row in self.edges):
            raise ValueError("A state needs n >= 2 nodes and an n by n adjacency matrix")
        values = list(self.nodes) + [v for row in self.edges for v in row]
        if any(v not in (0, 1) for v in values):
            raise ValueError("State coordinates must be binary")
        object.__setattr__(self, "nodes", tuple(bool(v) for v in self.nodes))
        object.__setattr__(self, "edges", tuple(tuple(bool(v) for v in r) for r in self.edges))

    @classmethod
    def empty(cls, n):
        return cls((False,) * n, ((False,) * n,) * n)

    def bits(self):
        return torch.tensor(
            self.nodes + tuple(self.edges[i][j] for i, j in pairs(len(self.nodes))),
            dtype=torch.float64,
        )

    def flip(self, coordinate):
        n = len(self.nodes)
        if not 0 <= coordinate < n * n:
            raise ValueError("Coordinate outside the node/edge state")
        z, a = list(self.nodes), [list(row) for row in self.edges]
        if coordinate < n:
            z[coordinate] = not z[coordinate]
        else:
            i, j = pairs(n)[coordinate - n]
            a[i][j] = not a[i][j]
        return GraphState(tuple(z), tuple(tuple(row) for row in a))

    def topological_order(self):
        n = len(self.nodes)
        degree = [sum(self.edges[j][i] for j in range(n)) for i in range(n)]
        queue = [i for i in range(n) if degree[i] == 0]
        order = []
        while queue:
            i = queue.pop(0)
            order.append(i)
            for j in range(n):
                if self.edges[i][j]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        queue.append(j)
        return order if len(order) == n else None

    def feasible(self):
        for i, row in enumerate(self.edges):
            for j, bit in enumerate(row):
                if bit and (i == j or not self.nodes[i] or not self.nodes[j]):
                    return False
        return self.topological_order() is not None


def pairs(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def validate_unit(values, name):
    values = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return values


def interior_cdf(mean, variance, threshold):
    """Gaussian exceedance probability, with interior floating-point endpoints."""
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("Gaussian moments must be finite")
    if (variance <= 0).any():
        raise ValueError("Gaussian variances must be positive")
    probability = 0.5 * torch.erfc((threshold - mean) / torch.sqrt(2 * variance))
    eps = torch.finfo(probability.dtype).eps
    return probability.clamp(eps, 1 - eps)


def bernoulli_log_mass(logits, state):
    """Unnormalized constrained mass: includes every inactive node/edge factor."""
    bits = state.bits().to(logits)
    if logits.shape != bits.shape:
        raise ValueError("Expected n^2 logits (nodes, then off-diagonal edges)")
    return (bits * logits - torch.nn.functional.softplus(logits)).sum()


def covariance_loss(logits, states, utilities, prior_logits, beta):
    """Negative empirical covariance gradient of E_q[u] - KL(q || p0)/beta."""
    if len(states) < 2:
        raise ValueError("At least two retained states are needed for graph fitting")
    log_q = torch.stack([bernoulli_log_mass(logits, x) for x in states])
    log_p = torch.stack([bernoulli_log_mass(prior_logits, x) for x in states])
    reward = torch.as_tensor(utilities, dtype=logits.dtype, device=logits.device)
    # DAG normalizers are state-independent and cancel under centering.
    weights = (reward - (log_q - log_p) / beta).detach()
    return -((weights - weights.mean()) * log_q).mean()


class CoordinateSampler:
    """Algorithm 1: lazy, symmetric, single-bit MH with a frozen utility cache."""

    def __init__(self, logits, n, utility=None, beta=1.0, generator=None):
        self.logits = torch.as_tensor(logits, dtype=torch.float64).detach().clone()
        if self.logits.shape != (n * n,) or not torch.isfinite(self.logits).all():
            raise ValueError("Expected n^2 finite logits")
        if n < 2 or not math.isfinite(beta) or beta <= 0:
            raise ValueError("n >= 2 and finite beta > 0 are required")
        self.n, self.beta, self.utility = n, beta, utility
        self.generator = generator
        self.utilities = {}

    async def value(self, state):
        if state not in self.utilities:
            value = 0.0 if self.utility is None else self.utility(state)
            if inspect.isawaitable(value):
                value = await value
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("The fixed utility must lie in [0, 1]")
            self.utilities[state] = value
        return self.utilities[state]

    async def log_acceptance(self, state, proposal):
        if not state.feasible() or len(state.nodes) != self.n:
            raise ValueError("The current state must be feasible")
        if len(proposal.nodes) != self.n or not proposal.feasible():
            return -math.inf
        if int((state.bits() != proposal.bits()).sum()) != 1:
            return -math.inf
        difference = bernoulli_log_mass(self.logits, proposal) - bernoulli_log_mass(
            self.logits, state
        )
        difference = float(difference) + self.beta * (
            await self.value(proposal) - await self.value(state)
        )
        return min(0.0, difference)

    async def sample(self, transitions, burn_in=0, initial=None):
        if not isinstance(transitions, int) or not isinstance(burn_in, int):
            raise ValueError("Transition and burn-in counts must be integers")
        if not 0 <= burn_in < transitions:
            raise ValueError("Require 0 <= burn_in < transitions")
        state = GraphState.empty(self.n) if initial is None else initial
        if len(state.nodes) != self.n or not state.feasible():
            raise ValueError("Initial state must be feasible and match n")
        retained = []
        for step in range(transitions):
            if torch.rand((), generator=self.generator).item() < 0.5:
                coordinate = int(torch.randint(self.n**2, (), generator=self.generator))
                proposal = state.flip(coordinate)
                log_accept = await self.log_acceptance(state, proposal)
                uniform = torch.rand((), dtype=torch.float64, generator=self.generator).item()
                if (math.log(uniform) if uniform > 0 else -math.inf) < log_accept:
                    state = proposal
            if step >= burn_in:
                retained.append(state)
        return retained


def independent_state(node_probs, edge_probs, generator=None):
    n = len(node_probs)
    z = torch.rand(n, dtype=torch.float64, generator=generator) < node_probs
    a = torch.rand((n, n), dtype=torch.float64, generator=generator) < edge_probs
    a.fill_diagonal_(False)
    return GraphState(tuple(z.tolist()), tuple(tuple(row) for row in a.tolist()))


def conditioned_samples(node_probs, edge_probs, count, generator=None, max_attempts=100000):
    """Independent draws from q conditioned on feasibility, by rejection."""
    retained = []
    for _ in range(max_attempts):
        state = independent_state(node_probs, edge_probs, generator)
        if state.feasible():
            retained.append(state)
            if len(retained) == count:
                return retained
    raise RuntimeError("DAG rejection budget exhausted; use_mcmc=True enables coordinate sampling")


def repair_state(state, scores):
    """Delete the lowest-score edge of the first DFS cycle; ties use (i, j)."""
    n = len(state.nodes)
    a = [
        [bool(state.edges[i][j] and i != j and state.nodes[i] and state.nodes[j]) for j in range(n)]
        for i in range(n)
    ]
    outgoing = [[j for j in range(n) if a[i][j]] for i in range(n)]

    def first_cycle():
        color, path = [0] * n, []

        def visit(i):
            color[i] = 1
            path.append(i)
            for j in outgoing[i]:
                if color[j] == 1:
                    cycle = path[path.index(j) :] + [j]
                    return list(zip(cycle[:-1], cycle[1:]))
                if color[j] == 0:
                    found = visit(j)
                    if found:
                        return found
            path.pop()
            color[i] = 2
            return None

        for i in range(n):
            if color[i] == 0:
                found = visit(i)
                if found:
                    return found
        return None

    cycle = first_cycle()
    while cycle:
        i, j = min(cycle, key=lambda edge: (float(scores[edge]), edge))
        a[i][j] = False
        outgoing[i].remove(j)
        cycle = first_cycle()
    return GraphState(state.nodes, tuple(tuple(row) for row in a))


def deployment_state(node_probs, edge_probs, candidates, count, threshold, generator=None):
    """Algorithm 2: independent draws, deterministic repair, and strict consensus."""
    node_probs = validate_unit(node_probs, "Node probabilities")
    edge_probs = validate_unit(edge_probs, "Edge probabilities")
    n = len(node_probs)
    candidates = torch.as_tensor(candidates, dtype=torch.bool).clone()
    if n < 2 or edge_probs.shape != (n, n) or candidates.shape != (n, n):
        raise ValueError("Deployment dimensions must agree, with n >= 2")
    if not isinstance(count, int) or count < 1 or not 0 <= threshold <= 1:
        raise ValueError("Require sample count >= 1 and consensus threshold in [0, 1]")
    candidates.fill_diagonal_(False)
    samples = []
    for _ in range(count):
        raw = independent_state(node_probs, edge_probs * candidates, generator)
        samples.append(repair_state(raw, edge_probs))
    if count == 1:
        return samples[0]
    nodes = tuple(any(x.nodes[i] for x in samples) for i in range(n))
    frequencies = torch.tensor([x.edges for x in samples], dtype=torch.float64).mean(0)
    edges = (frequencies > threshold) & candidates
    state = GraphState(nodes, tuple(tuple(row) for row in edges.tolist()))
    return repair_state(state, frequencies)


class SparsePairGP:
    """RBF inducing-variable GP with q(f,v)=p(f|v)q(v), Gaussian observations."""

    def __init__(
        self, inputs, inducing_count=10, length_scale=1.0, scale=1.0, noise=0.1, jitter=1e-6
    ):
        self.inputs = torch.as_tensor(inputs, dtype=torch.float64).clone()
        if self.inputs.ndim != 2 or not len(self.inputs) or not torch.isfinite(self.inputs).all():
            raise ValueError("Pair features must be a nonempty finite matrix")
        if not isinstance(inducing_count, int) or inducing_count < 1:
            raise ValueError("Inducing count must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in [length_scale, scale, noise, jitter]):
            raise ValueError("GP scales, observation variance and jitter must be positive")
        self.length_scale, self.scale, self.noise = length_scale, scale, noise
        indices = torch.linspace(0, len(inputs) - 1, min(inducing_count, len(inputs))).long()
        self.inducing = self.inputs[indices]
        self.kzz = self.kernel(self.inducing, self.inducing) + jitter * torch.eye(
            len(indices), dtype=torch.float64
        )
        self.factor = torch.linalg.cholesky(self.kzz)
        self.mean = torch.zeros(len(indices), dtype=torch.float64)
        self.covariance = self.kzz.clone()

    def kernel(self, x, y):
        distances = torch.cdist(x / self.length_scale, y / self.length_scale).square()
        return self.scale**2 * torch.exp(-0.5 * distances)

    def fit(self, inputs, targets):
        x = torch.as_tensor(inputs, dtype=torch.float64)
        y = torch.as_tensor(targets, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.inputs.shape[1] or y.shape != (len(x),):
            raise ValueError("GP training features and targets must have matching dimensions")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("GP observations must be finite")
        kzx = self.kernel(self.inducing, x)
        a = self.kzz + kzx @ kzx.T / self.noise
        factor = torch.linalg.cholesky(a)
        self.mean = self.kzz @ torch.cholesky_solve((kzx @ y / self.noise)[:, None], factor)[:, 0]
        self.covariance = self.kzz @ torch.cholesky_solve(self.kzz, factor)

    def moments(self):
        kxz = self.kernel(self.inputs, self.inducing)
        b = torch.cholesky_solve(kxz.T, self.factor).T
        conditional = self.scale**2 - (b * kxz).sum(1)
        variance = conditional + ((b @ self.covariance) * b).sum(1)
        return b @ self.mean, variance.clamp_min(0)
