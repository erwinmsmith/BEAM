import asyncio
import itertools
import math

import pytest
import torch

from beam.core.probability import (
    CoordinateSampler,
    GraphState,
    SparsePairGP,
    conditional_log_mass,
    score_loss,
    deployment_state,
    interior_cdf,
    pairs,
    repair_state,
)


def feasible_states(nodes):
    n = len(nodes)
    active = [(i, j) for i, j in pairs(n) if nodes[i] and nodes[j]]
    result = []
    for bits in itertools.product([False, True], repeat=len(active)):
        a = [[False] * n for _ in range(n)]
        for bit, (i, j) in zip(bits, active):
            a[i][j] = bit
        state = GraphState(nodes, tuple(tuple(row) for row in a))
        if state.feasible():
            result.append(state)
    return result


def test_full_fixed_mask_kernel_detailed_balance_stationarity_and_mixing():
    async def check():
        states = feasible_states((True, True, True))
        assert len(states) == 25
        p = torch.linspace(0.15, 0.8, 9, dtype=torch.float64).reshape(3, 3)
        sampler = CoordinateSampler(p, states[0].nodes)
        mass = torch.stack([conditional_log_mass(torch.logit(p), x) for x in states]).softmax(0)
        kernel = torch.zeros((25, 25), dtype=torch.float64)
        for i, state in enumerate(states):
            for coordinate in range(3, 9):
                proposal = state.flip(coordinate)
                alpha = math.exp(await sampler.log_acceptance(state, proposal))
                if proposal in states:
                    kernel[i, states.index(proposal)] += alpha / 6
                kernel[i, i] += (1 - alpha) / 6
        assert torch.allclose(kernel.sum(1), torch.ones(25, dtype=torch.float64))
        assert torch.allclose(mass[:, None] * kernel, (mass[:, None] * kernel).T, atol=1e-14)
        assert torch.allclose(mass @ kernel, mass, atol=1e-14)
        assert (torch.linalg.matrix_power(kernel, 12) > 0).all()
        assert await sampler.log_acceptance(states[0], states[0].flip(0)) == -math.inf
        samples = await sampler.sample(10)
        assert len(samples) == 10 and all(
            x.nodes == states[0].nodes and x.feasible() for x in samples
        )

    asyncio.run(check())


def test_inactive_pairs_are_absent_from_conditional_mass():
    logits = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    state = GraphState((1, 1, 0), ((0, 1, 0), (0, 0, 0), (0, 0, 0)))
    mass = conditional_log_mass(logits, state)
    assert float(mass.detach()) == pytest.approx(math.log(0.25))
    mass.backward()
    assert logits.grad[0, 1] == 0.5 and logits.grad[1, 0] == -0.5
    assert logits.grad[2].abs().sum() == 0 and logits.grad[:, 2].abs().sum() == 0


def test_all_rejections_and_singleton_steps_are_retained():
    async def check():
        sampler = CoordinateSampler(torch.full((2, 2), 0.5), (1, 1))

        async def reject(*args):
            return -math.inf

        sampler.log_acceptance = reject
        samples = await sampler.sample(10)
        assert len(samples) == 10 and len(set(samples)) == 1
        singleton = await CoordinateSampler(torch.full((2, 2), 0.5), (1, 0)).sample(10)
        assert len(singleton) == 10 and all(x.nodes == (True, False) for x in singleton)

    asyncio.run(check())


def test_node_and_centered_edge_gradient_matches_exact_joint_objective():
    eta = torch.tensor([-0.4, 0.2, 0.7], dtype=torch.float64, requires_grad=True)
    edge_eta = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    r, p = eta.sigmoid(), edge_eta.sigmoid()
    p = p * (1 - torch.eye(3, dtype=torch.float64))
    estimated = torch.tensor(0.0, dtype=torch.float64)
    exact = torch.tensor(0.0, dtype=torch.float64)
    for nodes in itertools.product([False, True], repeat=3):
        states = feasible_states(nodes)
        z = torch.tensor(nodes, dtype=torch.float64)
        b = (r**z * (1 - r) ** (1 - z)).prod()
        u = torch.tensor(
            [(sum(x.nodes) + 2 * sum(map(sum, x.edges))) / 9 for x in states], dtype=torch.float64
        )
        estimated = estimated - b.detach() * score_loss(r, p, states, u)
        masses = torch.stack([conditional_log_mass(edge_eta, x) for x in states]).softmax(0)
        exact = exact + b * (masses * u).sum()
    estimated_grad = torch.autograd.grad(estimated, (eta, edge_eta), retain_graph=True)
    exact_grad = torch.autograd.grad(exact, (eta, edge_eta))
    for a, b in zip(estimated_grad, exact_grad):
        assert torch.allclose(a, b, atol=1e-12)
    assert estimated_grad[0].abs().sum() > 0 and estimated_grad[1].abs().sum() > 0


def test_gp_covariance_and_kl_match_dense_gaussian_formulas():
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
    gp = SparsePairGP(x, inducing_count=2, scale=1.5)
    mean, variance = gp.moments()
    assert torch.allclose(mean, torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(variance, torch.full((3,), 2.25, dtype=torch.float64))
    assert float(gp.kl_divergence().detach()) == pytest.approx(0.0)
    with torch.no_grad():
        gp.location.copy_(torch.tensor([0.2, -0.3]))
        gp.raw_cholesky.copy_(torch.tensor([[0.1, 0.0], [0.3, -0.2]]))
    b = gp.kernel(x, gp.inducing) @ torch.linalg.inv(gp.kzz)
    expected = (gp.kernel(x, x) - b @ gp.kernel(gp.inducing, x) + b @ gp.covariance @ b.T).diag()
    assert torch.allclose(gp.moments()[1], expected, atol=1e-12)
    inverse = torch.linalg.inv(gp.kzz)
    kl = 0.5 * (
        torch.trace(inverse @ gp.covariance)
        + gp.mean @ inverse @ gp.mean
        - 2
        + torch.logdet(gp.kzz)
        - torch.logdet(gp.covariance)
    )
    assert torch.allclose(gp.kl_divergence(), kl, atol=1e-12)
    assert torch.linalg.eigvalsh(gp.covariance).min() > 0


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
def test_gaussian_deployment_fixed_nodes_and_strict_consensus(monkeypatch, threshold, expected):
    draws = iter([torch.tensor([[0.0, 1.0], [-1.0, 0.0]]), torch.full((2, 2), -1.0)])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: next(draws))
    result = deployment_state(
        (1, 1), torch.zeros(2, 2), torch.ones(2, 2), torch.ones(2, 2), 2, threshold, 0.0
    )
    assert result.nodes == (True, True) and result.edges[0][1] == expected and result.feasible()


def test_deployment_repairs_by_mean_and_respects_active_candidates(monkeypatch):
    means = torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.7, 0.0, 0.0]])
    variances = torch.ones(3, 3)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.ones(3, 3))
    candidates = means > 0
    result = deployment_state((1, 1, 1), means, variances, candidates, 1, 0.5)
    assert result.feasible() and not result.edges[1][2]
    assert result.edges[0][1] and result.edges[2][0]
    result = deployment_state((1, 1, 0), means, variances, candidates, 1, 0.5)
    assert result.feasible() and sum(map(sum, result.edges)) == 1


def test_gaussian_probability_is_exceedance_not_sigmoid():
    p = interior_cdf(torch.tensor([0.0, 1.0, -1.0]), torch.ones(3), 0.0)
    assert torch.allclose(p, torch.tensor([0.5, 0.841344746, 0.158655254]), atol=1e-7)
    extremes = interior_cdf(
        torch.tensor([-1000.0, 1000.0], dtype=torch.float64), torch.ones(2), 0.0
    )
    assert ((extremes > 0) & (extremes < 1)).all()


def test_sampler_rejects_nonbinary_masks():
    with pytest.raises(ValueError, match="binary"):
        CoordinateSampler(torch.full((2, 2), 0.5), (1, 2))
