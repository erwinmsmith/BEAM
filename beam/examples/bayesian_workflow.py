"""CPU-only calibration, fixed-target sampling and deployment; no API key needed.

Run: python -m beam.examples.bayesian_workflow
"""

import asyncio

from beam import AgentBayesian, AgentConfig, AgentGraph, AgentNode, BEAMConfig, OptimizationConfig


def build_example():
    config = BEAMConfig(
        agents=[AgentConfig(name="Solver", count=3)],
        num_rounds=1,
        optimization=OptimizationConfig(
            strategy="bayesian",
            use_mcmc=True,
            mcmc_samples=10,
            deployment_samples=10,
            node_threshold=0.0,
            random_seed=7,
        ),
    )
    graph = AgentGraph(config)
    for i in range(3):
        graph.add_node(AgentNode(id=str(i), role=f"Solver {i}", execute_fn=lambda s: "4"))
    graph.decision_node = AgentNode(id="decision", execute_fn=lambda s: "4")
    strategy = AgentBayesian(config)
    strategy.set_graph(graph)
    strategy.set_completeness([1.0, 0.8, 0.6])
    return strategy


async def main():
    strategy = build_example()
    stats = await strategy.atrain(
        [{"task": "What is 2 + 2?", "answer": "4"}],
        lambda prediction, answer: float(prediction == answer),
        num_iterations=1,
        verbose=False,
    )

    samples = await strategy.sample_reference()
    answers, metadata = await strategy.run({"task": "What is 2 + 2?"})
    print("Calibration accuracy:", stats["accuracies"][-1])
    print("Feasible reference samples:", len(samples))
    print("Answers:", answers)
    print("Deployment:", metadata)


if __name__ == "__main__":
    asyncio.run(main())
