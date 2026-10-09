import asyncio
import copy

import pytest
import torch

from beam import (
    AgentBayesian,
    AgentConfig,
    AgentGraph,
    AgentNode,
    BEAMConfig,
    GraphState,
    TokenOptimizer,
)
from beam.core.config import OptimizationConfig, OptimizationStrategy


def make_strategy(**options):
    c = BEAMConfig(
        agents=[AgentConfig(name="Agent", count=2)],
        num_rounds=1,
        optimization=OptimizationConfig(
            strategy="bayesian",
            use_mcmc=True,
            mcmc_samples=120,
            mcmc_warmup=30,
            random_seed=14,
            **options,
        ),
    )
    graph = AgentGraph(c)
    for i in range(2):
        graph.add_node(AgentNode(id=str(i), role=str(i), execute_fn=lambda s: "yes"))
    graph.decision_node = AgentNode(id="decision", execute_fn=lambda s: "yes")
    strategy = AgentBayesian(c)
    strategy.set_graph(graph)
    return strategy


def test_energy_bounds_and_completeness_variance_once():
    strategy = make_strategy(energy_gain=100, energy_cost=100)
    before = strategy.edge_vars.clone()
    strategy.set_completeness([0.5, 0.25])
    expected = strategy.config.optimization.uncertainty_scale**2 * (0.5**2 + 0.75**2)
    assert float(strategy.edge_vars[0, 1] - before[0, 1]) == pytest.approx(expected)
    strategy.set_completeness([1, 1])
    strategy.update_energy([1, 0], [0, 1])
    assert strategy.energy.tolist() == [2.0, 0.1]
    r, p = strategy.probabilities(fitted=False)
    assert ((r > 0) & (r < 1)).all() and p[0, 0] == 0 and p[1, 1] == 0
    assert ((p[[0, 1], [1, 0]] > 0) & (p[[0, 1], [1, 0]] < 1)).all()


def test_calibration_updates_joint_logits_gp_and_energy():
    strategy = make_strategy()
    original = strategy.offsets.detach().clone()
    original_mean = strategy.gp.mean.clone()
    prior_energy = strategy.energy.clone()
    stats = strategy.train(
        [{"task": "q", "answer": "yes"}],
        lambda p, a: float(p == a),
        num_iterations=1,
        verbose=False,
    )
    assert stats["total_executed"] > 0
    assert not torch.equal(strategy.offsets, original)
    assert not torch.equal(strategy.gp.mean, original_mean)
    assert not torch.equal(strategy.energy, prior_energy)
    assert torch.isfinite(strategy.offsets).all()
    assert strategy.offsets[:2].abs().sum() > 0
    assert strategy.offsets[2:].abs().sum() > 0


def test_execution_preserves_graph_and_inactive_nodes_exactly():
    strategy = make_strategy()
    graph = strategy.graph
    calls = []

    def execute(s):
        calls.append(s)
        return s["task"] + "!"

    graph.nodes["0"].execute_fn = execute
    graph.nodes["1"].execute_fn = execute
    graph.spatial_logits.data.fill_(-100.0)
    graph.nodes["0"].outputs = ["stale"]
    state = GraphState((1, 1), ((0, 1), (0, 0)))
    asyncio.run(graph.arun_state({"task": "fresh"}, state, num_rounds=2))
    assert len(calls) == 4
    assert calls[1]["spatial_info"]["0"]["output"] == "fresh!"
    assert calls[2]["temporal_info"]["0"]["output"] == "fresh!"
    assert graph.nodes["0"].outputs == ["stale"]
    assert not graph.nodes["0"].spatial_successors
    calls.clear()
    asyncio.run(graph.arun_state({"task": "fresh"}, GraphState((1, 0), ((0, 0), (0, 0)))))
    assert len(calls) == 1
    calls.clear()
    assert asyncio.run(graph.arun_state({}, GraphState.empty(2)))[0] == []
    assert not calls


def test_concurrent_state_executions_isolate_context():
    strategy = make_strategy()

    async def execute(s):
        await asyncio.sleep(0)
        return s["task"] + str(sorted(v["output"] for v in s["spatial_info"].values()))

    for node in strategy.graph.nodes.values():
        node.execute_fn = execute
    strategy.graph.decision_node = None
    state = GraphState((1, 1), ((0, 1), (0, 0)))

    async def run():
        return await asyncio.gather(
            *(strategy.graph.arun_state({"task": q}, state) for q in ["alpha", "beta"])
        )

    a, b = asyncio.run(run())
    assert "beta" not in a[0][0] and "alpha" not in b[0][0]
    assert "alpha[]" in a[0][0] and "beta[]" in b[0][0]


def test_state_execution_propagates_failure():
    strategy = make_strategy()

    def fail(s):
        raise RuntimeError("agent failed")

    strategy.graph.nodes["0"].execute_fn = fail
    with pytest.raises(RuntimeError, match="agent failed"):
        asyncio.run(
            strategy.graph.arun_state({}, GraphState((1, 0), ((0, 0), (0, 0))), max_retries=1)
        )


def test_checkpoint_restores_next_draw_and_fitted_distribution(tmp_path):
    strategy = make_strategy()
    strategy.train(
        [{"task": "q", "answer": "yes"}],
        lambda p, a: float(p == a),
        num_iterations=1,
        verbose=False,
    )
    path = tmp_path / "beam.pt"
    strategy.save(path)
    other = make_strategy()
    other.load(path)
    for first, second in zip(strategy.probabilities(), other.probabilities()):
        assert torch.equal(first, second)
    assert torch.equal(strategy.energy, other.energy)
    assert torch.equal(strategy.gp.covariance, other.gp.covariance)
    assert asyncio.run(strategy.run({"task": "q"})) == asyncio.run(other.run({"task": "q"}))


def test_token_optimizer_dispatches_bayesian_and_round_override(tmp_path):
    strategy = make_strategy()
    optimizer = TokenOptimizer(strategy.config, strategy.graph)
    optimizer.train(
        [{"task": "q", "answer": "yes"}],
        lambda p, a: float(p == a),
        num_iterations=1,
        verbose=False,
    )
    _, meta = asyncio.run(optimizer.run({"task": "q"}, num_rounds=2))
    assert meta["strategy"] == "bayesian" and "node_mask" in meta
    path = tmp_path / "optimizer.pt"
    optimizer.save_state(path)
    optimizer.load_state(path)
    assert "node_probabilities" in optimizer.get_edge_statistics()


def test_reference_is_frozen_during_external_updates():
    strategy = make_strategy()

    async def utility(x):
        strategy.set_completeness([0.1, 0.2])
        return sum(x.nodes) / 2

    states = asyncio.run(strategy.sample_reference(utility, transitions=30, burn_in=5))
    assert len(states) == 25 and all(x.feasible() for x in states)


def test_independent_calibration_path():
    strategy = make_strategy()
    strategy.config.optimization.use_mcmc = False
    states = asyncio.run(strategy._variational_samples())
    assert len(states) == 120 and all(x.feasible() for x in states)


@pytest.mark.parametrize(
    "field,value",
    [
        ("beta", 0),
        ("baseline_std", 0),
        ("energy_min", -1),
        ("energy_initial", float("nan")),
        ("consensus_threshold", 1.2),
        ("gp_scale", 0),
        ("deployment_samples", 0),
        ("mcmc_samples", 1),
    ],
)
def test_configuration_rejects_invalid_theory_domains(field, value):
    c = BEAMConfig()
    setattr(c.optimization, field, value)
    with pytest.raises(ValueError):
        AgentBayesian(c)


def test_config_from_dict_preserves_input():
    source = {"agents": [{"name": "A", "count": 2}], "optimization": {"strategy": "bayesian"}}
    original = copy.deepcopy(source)
    BEAMConfig.from_dict(source)
    assert source == original


def test_checkpoint_accepts_enum_strategy_and_rejects_old_schema(tmp_path):
    strategy = make_strategy()
    strategy.config.optimization.strategy = OptimizationStrategy.BAYESIAN
    path = tmp_path / "enum.pt"
    strategy.save(path)
    other = make_strategy()
    other.load(path)
    assert torch.equal(strategy.offsets, other.offsets)
    torch.save({"edge_means": torch.zeros(4)}, path)
    with pytest.raises(ValueError, match="version-2"):
        other.load(path)


def test_sync_optimizer_inference_after_training():
    strategy = make_strategy()
    optimizer = TokenOptimizer(strategy.config, strategy.graph)
    optimizer.train(
        [{"task": "q", "answer": "yes"}],
        lambda p, a: float(p == a),
        num_iterations=1,
        verbose=False,
    )
    assert optimizer.run_sync({"task": "q"})[1]["strategy"] == "bayesian"


def test_prune_entry_point_still_executes():
    config = BEAMConfig(
        agents=[AgentConfig(name="A", count=2)],
        num_rounds=1,
        optimization=OptimizationConfig(
            strategy="prune", optimize_spatial=False, optimize_temporal=False
        ),
    )
    graph = AgentGraph(config)
    graph.add_node(AgentNode(id="a", execute_fn=lambda s: "one"))
    graph.add_node(AgentNode(id="b", execute_fn=lambda s: "two"))
    graph.init_potential_edges()
    answers, _ = asyncio.run(TokenOptimizer(config, graph).run({"task": "q"}))
    assert answers == ["two"]
