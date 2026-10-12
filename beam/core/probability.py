"""Finite node/edge states, sparse GP moments, and BEAM graph distributions."""

from dataclasses import dataclass
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


def conditional_log_mass(edge_logits, state):
    """Bernoulli edge mass on active ordered pairs, before DAG normalization."""
    n = len(state.nodes)
    if edge_logits.shape != (n, n):
        raise ValueError("Expected an n by n edge-logit matrix")
    active = [(i, j) for i, j in pairs(n) if state.nodes[i] and state.nodes[j]]
    if not active:
        return edge_logits.sum() * 0
    rows, cols = zip(*active)
    eta = edge_logits[rows, cols]
    bits = torch.tensor([state.edges[i][j] for i, j in active], dtype=eta.dtype)
    return (bits * eta - torch.nn.functional.softplus(eta)).sum()


def score_loss(node_probs, edge_probs, states, utilities):
    """Negative node score plus the within-mask centered edge-score estimate."""
    if not states or any(x.nodes != states[0].nodes for x in states):
        raise ValueError("Retained states must share one fixed node mask")
    reward = validate_unit(utilities, "Task utilities").to(node_probs)
    if reward.shape != (len(states),):
        raise ValueError("One utility per retained state is required")
    z = torch.tensor(states[0].nodes, dtype=node_probs.dtype)
    node_log = (z * node_probs.log() + (1 - z) * torch.log1p(-node_probs)).sum()
    # The diagonal is excluded from the mass; give it a finite logit.
    safe = edge_probs + torch.eye(len(node_probs), dtype=edge_probs.dtype) * 0.5
    logs = torch.stack([conditional_log_mass(torch.logit(safe), x) for x in states])
    return -(
        reward.mean().detach() * node_log
        + ((reward - reward.mean()).detach() * (logs - logs.mean())).mean()
    )


class CoordinateSampler:
    """Fixed-node, uniformly proposed single-edge Metropolis--Hastings chain."""

    def __init__(self, edge_probs, nodes, generator=None):
        self.nodes = GraphState(tuple(nodes), ((False,) * len(nodes),) * len(nodes)).nodes
        self.n = len(self.nodes)
        probabilities = validate_unit(edge_probs, "Edge probabilities")
        if self.n < 2 or probabilities.shape != (self.n, self.n):
            raise ValueError("Expected n >= 2 nodes and an n by n probability matrix")
        self.active = [(i, j) for i, j in pairs(self.n) if self.nodes[i] and self.nodes[j]]
        if any(not 0 < float(probabilities[i, j]) < 1 for i, j in self.active):
            raise ValueError("Active-pair probabilities must be strictly between zero and one")
        safe = probabilities.clone()
        safe.fill_diagonal_(0.5)
        self.logits = torch.logit(safe).detach().clone()
        self.generator = generator

    async def log_acceptance(self, state, proposal):
        if state.nodes != self.nodes or not state.feasible():
            raise ValueError("The current state must be a DAG on the fixed node mask")
        if proposal.nodes != self.nodes or not proposal.feasible():
            return -math.inf
        changed = [(i, j) for i, j in pairs(self.n) if state.edges[i][j] != proposal.edges[i][j]]
        if len(changed) != 1 or changed[0] not in self.active:
            return -math.inf
        i, j = changed[0]
        return min(0.0, float(self.logits[i, j]) * (1 if proposal.edges[i][j] else -1))

    async def sample(self, transitions, initial=None):
        if not isinstance(transitions, int) or transitions < 1:
            raise ValueError("Transition count must be a positive integer")
        state = (
            GraphState(self.nodes, ((False,) * self.n,) * self.n) if initial is None else initial
        )
        if state.nodes != self.nodes or not state.feasible():
            raise ValueError("Initial graph must be a DAG on the fixed node mask")
        retained = []
        for _ in range(transitions):
            if self.active:
                i, j = self.active[
                    int(torch.randint(len(self.active), (), generator=self.generator))
                ]
                proposal = state.flip(self.n + pairs(self.n).index((i, j)))
                log_accept = await self.log_acceptance(state, proposal)
                uniform = torch.rand((), dtype=torch.float64, generator=self.generator).item()
                if (math.log(uniform) if uniform else -math.inf) < log_accept:
                    state = proposal
            retained.append(state)
        return retained


def conditioned_samples(nodes, edge_probs, count, generator=None, max_attempts=100000):
    """Independent DAG-conditioned edge draws with the node mask held fixed."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("Sample count must be positive")
    nodes = GraphState(tuple(nodes), ((False,) * len(nodes),) * len(nodes)).nodes
    n = len(nodes)
    probabilities = validate_unit(edge_probs, "Edge probabilities")
    if n < 2 or probabilities.shape != (n, n):
        raise ValueError("Node and edge dimensions must agree")
    endpoint = torch.tensor(nodes)[:, None] & torch.tensor(nodes)[None, :]
    retained = []
    for _ in range(max_attempts):
        edges = (
            torch.rand((n, n), dtype=torch.float64, generator=generator) < probabilities
        ) & endpoint
        edges.fill_diagonal_(False)
        state = GraphState(nodes, tuple(tuple(row) for row in edges.tolist()))
        if state.feasible():
            retained.append(state)
            if len(retained) == count:
                return retained
    raise RuntimeError("DAG rejection budget exhausted; use_mcmc=True enables edge sampling")


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


def deployment_state(
    nodes, means, variances, candidates, count, threshold, edge_threshold=0.5, generator=None
):
    """Gaussian edge draws, smallest-mean cycle repair, strict frequency consensus."""
    nodes = GraphState(tuple(nodes), ((False,) * len(nodes),) * len(nodes)).nodes
    n = len(nodes)
    means = torch.as_tensor(means, dtype=torch.float64)
    variances = torch.as_tensor(variances, dtype=torch.float64)
    candidates = torch.as_tensor(candidates, dtype=torch.bool).clone()
    if n < 2 or any(x.shape != (n, n) for x in [means, variances, candidates]):
        raise ValueError("Deployment dimensions must agree, with n >= 2")
    if not isinstance(count, int) or count < 1 or not 0 <= threshold <= 1:
        raise ValueError("Require sample count >= 1 and consensus threshold in [0, 1]")
    if (
        not torch.isfinite(means).all()
        or not torch.isfinite(variances).all()
        or (variances < 0).any()
    ):
        raise ValueError("Gaussian moments must be finite with nonnegative variances")
    if not math.isfinite(edge_threshold):
        raise ValueError("Edge threshold must be finite")
    candidates.fill_diagonal_(False)
    candidates &= torch.tensor(nodes)[:, None] & torch.tensor(nodes)[None, :]
    samples = []
    for _ in range(count):
        draws = means + variances.sqrt() * torch.randn(
            (n, n), dtype=torch.float64, generator=generator
        )
        edges = (draws > edge_threshold) & candidates
        raw = GraphState(nodes, tuple(tuple(row) for row in edges.tolist()))
        samples.append(repair_state(raw, means))
    if count == 1:
        return samples[0]
    frequencies = torch.tensor([x.edges for x in samples], dtype=torch.float64).mean(0)
    edges = (frequencies > threshold) & candidates
    state = GraphState(nodes, tuple(tuple(row) for row in edges.tolist()))
    return repair_state(state, frequencies)


class SparsePairGP(torch.nn.Module):
    """RBF inducing GP with trainable whitened Gaussian variational parameters."""

    def __init__(self, inputs, inducing_count=10, length_scale=1.0, scale=1.0, jitter=1e-6):
        super().__init__()
        inputs = torch.as_tensor(inputs, dtype=torch.float64).detach().clone()
        if inputs.ndim != 2 or not len(inputs) or not torch.isfinite(inputs).all():
            raise ValueError("Pair features must be a nonempty finite matrix")
        if not isinstance(inducing_count, int) or inducing_count < 1:
            raise ValueError("Inducing count must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in [length_scale, scale, jitter]):
            raise ValueError("GP scales and jitter must be positive")
        self.length_scale, self.scale = length_scale, scale
        indices = torch.linspace(0, len(inputs) - 1, min(inducing_count, len(inputs))).long()
        self.register_buffer("inputs", inputs)
        self.register_buffer("inducing", inputs[indices].clone())
        kzz = self.kernel(self.inducing, self.inducing) + jitter * torch.eye(
            len(indices), dtype=torch.float64
        )
        self.register_buffer("kzz", kzz)
        self.register_buffer("factor", torch.linalg.cholesky(kzz))
        self.location = torch.nn.Parameter(torch.zeros(len(indices), dtype=torch.float64))
        self.raw_cholesky = torch.nn.Parameter(torch.zeros_like(kzz))

    @property
    def cholesky(self):
        return self.raw_cholesky.tril(-1) + torch.diag(self.raw_cholesky.diag().exp())

    @property
    def mean(self):
        return self.factor @ self.location

    @property
    def covariance(self):
        factor = self.factor @ self.cholesky
        return factor @ factor.T

    def kernel(self, x, y):
        distances = torch.cdist(x / self.length_scale, y / self.length_scale).square()
        return self.scale**2 * torch.exp(-0.5 * distances)

    def moments(self, inputs=None):
        x = self.inputs if inputs is None else inputs
        kxz = self.kernel(x, self.inducing)
        projection = torch.linalg.solve_triangular(self.factor, kxz.T, upper=False).T
        conditional = self.scale**2 - projection.square().sum(1)
        variance = conditional + (projection @ self.cholesky).square().sum(1)
        return projection @ self.location, variance.clamp_min(torch.finfo(x.dtype).eps)

    def kl_divergence(self):
        chol = self.cholesky
        return 0.5 * (
            chol.square().sum()
            + self.location.square().sum()
            - len(self.location)
            - 2 * self.raw_cholesky.diag().sum()
        )
