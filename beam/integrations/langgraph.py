"""
LangGraph integration utilities for BEAM toolkit.

This module provides lightweight utilities to integrate BEAM's token
optimization strategies with LangGraph workflows.

BEAM extends LangGraph with token-efficient multi-agent optimization.
"""

from typing import Dict, Any, List, Optional, Callable, TypedDict
import asyncio


class BEAMState(TypedDict, total=False):
    """
    Recommended state type for BEAM-optimized LangGraph workflows.
    
    Use this as a base for your LangGraph state when integrating with BEAM.
    
    Example:
        ```python
        from langgraph.graph import StateGraph
        from beam.integrations.langgraph import BEAMState
        
        class MyState(BEAMState):
            custom_field: str
        
        workflow = StateGraph(MyState)
        ```
    """
    task: str
    context: str
    agent_outputs: Dict[str, str]
    final_answer: str
    round: int
    active_agents: List[str]
    skipped_agents: List[str]


def create_beam_node(
    node_fn: Callable,
    role: str = "Assistant"
) -> Callable:
    """
    Wrap a LangGraph node function with BEAM context handling.
    
    This wrapper adds context aggregation from other agents to your
    node function, making it compatible with BEAM's multi-agent optimization.
    
    Args:
        node_fn: Original LangGraph node function (state) -> state
        role: Role description for this node
        
    Returns:
        Wrapped node function with BEAM context handling
        
    Example:
        ```python
        def my_researcher(state):
            task = state["task"]
            context = state.get("context", "")
            # ... do research
            return {"output": result}
        
        beam_researcher = create_beam_node(my_researcher, role="Researcher")
        ```
    """
    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        # Aggregate context from agent_outputs if available
        agent_outputs = state.get("agent_outputs", {})
        if agent_outputs:
            context_parts = [
                f"[{k}]: {v}" for k, v in agent_outputs.items()
                if v and v != "None."
            ]
            state["context"] = "\n".join(context_parts)
        
        # Call original function
        result = node_fn(state)
        
        # Store output in agent_outputs
        if isinstance(result, dict) and "output" in result:
            if "agent_outputs" not in result:
                result["agent_outputs"] = state.get("agent_outputs", {}).copy()
            result["agent_outputs"][role] = result["output"]
        
        return result
    
    wrapped.__beam_role__ = role
    wrapped.__name__ = f"beam_{node_fn.__name__}"
    return wrapped


def create_skip_condition(
    skip_agents: List[str]
) -> Callable:
    """
    Create a conditional edge function that skips specified agents.
    
    Use this with LangGraph's conditional edges to implement BEAM's
    agent dropout optimization.
    
    Args:
        skip_agents: List of agent names to skip
        
    Returns:
        Conditional function for LangGraph
        
    Example:
        ```python
        from langgraph.graph import StateGraph
        
        # After BEAM training, get skip decisions
        skip_agents = dropout_strategy.skip_nodes
        
        workflow.add_conditional_edges(
            "router",
            create_skip_condition(["agent_2"]),
            {"skip": "next_node", "continue": "agent_2"}
        )
        ```
    """
    def should_skip(state: Dict[str, Any]) -> str:
        current_agent = state.get("current_agent", "")
        if current_agent in skip_agents:
            return "skip"
        return "continue"
    
    return should_skip


def apply_beam_masks(
    edges: List[tuple],
    spatial_masks: List[List[int]],
    node_order: Optional[List[str]] = None,
) -> List[tuple]:
    """Filter existing edges using the selected binary adjacency matrix.

    Named endpoints require the same ``node_order`` used to construct the
    matrix. Without it, endpoints must be integer matrix indices. The order
    of retained edges is preserved.
    """
    n = len(spatial_masks)
    if any(len(row) != n for row in spatial_masks):
        raise ValueError("spatial_masks must be square")
    if any(value not in (0, 1) for row in spatial_masks for value in row):
        raise ValueError("spatial_masks must contain binary values")
    if node_order is not None:
        if len(node_order) != n or len(set(node_order)) != n:
            raise ValueError("node_order must contain one unique name per matrix row")
        indices = {name: index for index, name in enumerate(node_order)}

        def index_of(node):
            if node not in indices:
                raise ValueError(f"Unknown node: {node!r}")
            return indices[node]
    else:
        def index_of(node):
            if isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < n:
                raise ValueError("Use integer indices in range or provide node_order")
            return node

    selected = []
    for source, target in edges:
        i, j = index_of(source), index_of(target)
        if spatial_masks[i][j]:
            selected.append((source, target))
    return selected


def get_langgraph_config(
    num_agents: int,
    strategy: str = "prune",
    connection_mode: str = "full_connected"
) -> Dict[str, Any]:
    """
    Generate BEAM configuration suitable for LangGraph integration.
    
    Args:
        num_agents: Number of agents in your LangGraph workflow
        strategy: Optimization strategy ('prune', 'dropout', 'bayesian')
        connection_mode: How agents are connected
        
    Returns:
        Configuration dict for BEAMConfig.from_dict()
        
    Example:
        ```python
        from beam import BEAMConfig
        from beam.integrations.langgraph import get_langgraph_config
        
        config_dict = get_langgraph_config(
            num_agents=4,
            strategy="dropout",
            connection_mode="full_connected"
        )
        config = BEAMConfig.from_dict(config_dict)
        ```
    """
    return {
        "agents": [
            {"name": f"Agent_{i}", "count": 1}
            for i in range(num_agents)
        ],
        "connection_mode": connection_mode,
        "optimization": {
            "strategy": strategy,
            "optimize_spatial": True,
            "optimize_temporal": True
        }
    }
