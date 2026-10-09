import asyncio
import itertools
import math

import pytest
import torch

from beam.core.probability import (
    CoordinateSampler,
    GraphState,
    SparsePairGP,
    bernoulli_log_mass,
    covariance_loss,
    deployment_state,
    interior_cdf,
    pairs,
    repair_state,
)


def feasible_states(n):
    result = []
    for bits in itertools.product([False, True], repeat=n * n):
        a = [[False] * n for _ in range(n)]
        for bit, (i, j) in zip(bits[n:], pairs(n)):
            a[i][j] = bit
        state = GraphState(bits[:n], tuple(tuple(row) for row in a))
        if state.feasible():
            result.append(state)
    return result


def test_full_three_node_kernel_and_minorization():
    async def check():
        states = feasible_states(3)
        assert len(states) == 38
        logits = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
        utility = lambda x: (sum(x.nodes) + sum(map(sum, x.edges))) / 6
        sampler = CoordinateSampler(logits, 3, utility, beta=1.7)
        mass = torch.tensor(
            [float(bernoulli_log_mass(logits, x)) + 1.7 * utility(x) for x in states],
            dtype=torch.float64,
        ).softmax(0)
        kernel = torch.zeros((38, 38), dtype=torch.float64)
        for i, state in enumerate(states):
            for coordinate in range(9):
                proposal = state.flip(coordinate)
                alpha = math.exp(await sampler.log_acceptance(state, proposal))
                if proposal in states:
                    kernel[i, states.index(proposal)] += alpha / 18
                kernel[i, i] += (1 - alpha) / 18
            kernel[i, i] += 0.5
        assert torch.allclose(kernel.sum(1), torch.ones(38, dtype=torch.float64))
        assert torch.all(kernel.diag() >= 0.5)
        flow = mass[:, None] * kernel
        assert torch.allclose(flow, flow.T, atol=1e-14)
        assert torch.allclose(mass @ kernel, mass, atol=1e-14)
        kd = torch.linalg.matrix_power(kernel, 9)
        assert (kd > 0).all()
        delta = (mass.min() / mass.max() / 18) ** 9
        assert torch.all(kd[:, states.index(GraphState.empty(3))] >= delta)

    asyncio.run(check())


def test_mass_includes_inactive_edge_factors():
    probabilities = torch.tensor([0.2, 0.3, 0.4, 0.6], dtype=torch.float64)
    actual = bernoulli_log_mass(torch.logit(probabilities), GraphState.empty(2))
    assert float(actual) == pytest.approx(math.log(0.8 * 0.7 * 0.6 * 0.4))


def test_incident_node_flip_rejected_and_utility_cached():
    calls = {}

    def utility(x):
        calls[x] = calls.get(x, 0) + 1
        return 0.5

    async def check():
        state = GraphState((1, 1), ((0, 1), (0, 0)))
        sampler = CoordinateSampler(torch.zeros(4), 2, utility)
        assert await sampler.log_acceptance(state, state.flip(0)) == -math.inf
        proposal = state.flip(2)
        await sampler.log_acceptance(state, proposal)
        await sampler.log_acceptance(state, proposal)
        assert calls == {proposal: 1, state: 1}
        original = sampler.logits.clone()
        retained = await sampler.sample(17, 3)
        assert len(retained) == 14 and all(x.feasible() for x in retained)
        assert torch.equal(sampler.logits, original)

    asyncio.run(check())


def test_covariance_gradient_matches_exact_objective_finite_difference():
    states = feasible_states(3)
    prior = torch.linspace(-0.6, 0.7, 9, dtype=torch.float64)
    utilities = torch.tensor([sum(x.nodes) / 3 for x in states], dtype=torch.float64)
    beta = 2.1
    logits = torch.zeros(9, dtype=torch.float64, requires_grad=True)
    loss = covariance_loss(logits, states, utilities, prior, beta)
    loss.backward()
    p0 = torch.stack([bernoulli_log_mass(prior, x) for x in states]).softmax(0)

    def objective(eta):
        q = torch.stack([bernoulli_log_mass(eta, x) for x in states]).softmax(0)
        return float((q * (utilities - (q.log() - p0.log()) / beta)).sum())

    for j in range(9):
        h = torch.zeros(9, dtype=torch.float64)
        h[j] = 1e-5
        derivative = (objective(h) - objective(-h)) / 2e-5
        assert float(-logits.grad[j]) == pytest.approx(derivative, abs=1e-9)


def test_gp_keeps_conditional_covariance_and_fits_pair_features():
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
    gp = SparsePairGP(x, inducing_count=1, scale=1.5)
    mean, variance = gp.moments()
    assert torch.allclose(mean, torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(variance, torch.full((3,), 2.25, dtype=torch.float64))
    gp.fit(x[:2], torch.tensor([0.8, 0.2]))
    mean, variance = gp.moments()
    b = gp.kernel(x, gp.inducing) @ torch.linalg.inv(gp.kzz)
    expected = (gp.kernel(x, x) - b @ gp.kernel(gp.inducing, x) + b @ gp.covariance @ b.T).diag()
    assert torch.allclose(variance, expected, atol=1e-12)
    assert (mean > 0).all() and (variance > 0).all()
    assert (torch.linalg.eigvalsh(gp.covariance) >= 0).all()


def test_cycle_repair_fixed_tie_breaking_and_endpoint_validity():
    state = GraphState((1, 1, 1), ((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    scores = torch.ones((3, 3))
    scores[1, 2] = 0.1
    repaired = repair_state(state, scores)
    assert repaired.feasible() and not repaired.edges[1][2]
    tied = repair_state(state, torch.ones((3, 3)))
    assert not tied.edges[0][1]
    off = GraphState((0, 1, 1), state.edges)
    repaired = repair_state(off, scores)
    assert repaired.feasible() and not any(repaired.edges[0])
    assert not any(row[0] for row in repaired.edges)


@pytest.mark.parametrize("threshold,expected", [(0, True), (0.5, False), (1, False)])
def test_consensus_uses_strict_threshold_and_union_nodes(monkeypatch, threshold, expected):
    samples = iter([GraphState((1, 1), ((0, 1), (0, 0))), GraphState((1, 0), ((0, 0), (0, 0)))])
    monkeypatch.setattr("beam.core.probability.independent_state", lambda *args: next(samples))
    result = deployment_state(torch.ones(2), torch.ones(2, 2), torch.ones(2, 2), 2, threshold)
    assert result.nodes == (True, True)
    assert result.edges[0][1] == expected and result.feasible()


def test_deployment_candidate_support_and_empty_graph():
    for threshold in [0, 0.5, 1]:
        result = deployment_state(torch.ones(3), torch.ones(3, 3), torch.zeros(3, 3), 4, threshold)
        assert result.feasible() and sum(map(sum, result.edges)) == 0
    result = deployment_state(torch.zeros(3), torch.ones(3, 3), torch.ones(3, 3), 1, 0)
    assert result == GraphState.empty(3)


def test_gaussian_probability_is_exceedance_not_sigmoid():
    p = interior_cdf(torch.tensor([0.0, 1.0, -1.0]), torch.ones(3), 0.0)
    assert torch.allclose(p, torch.tensor([0.5, 0.841344746, 0.158655254]), atol=1e-7)
    extremes = interior_cdf(
        torch.tensor([-1000.0, 1000.0], dtype=torch.float64), torch.ones(2), 0.0
    )
    assert ((extremes > 0) & (extremes < 1)).all()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_utility_rejected(value):
    sampler = CoordinateSampler(torch.zeros(4), 2, lambda x: value)
    with pytest.raises(ValueError):
        asyncio.run(sampler.value(GraphState.empty(2)))
