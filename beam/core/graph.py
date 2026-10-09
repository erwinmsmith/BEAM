"""Agent graph implementation for BEAM toolkit."""

import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import copy

from beam.core.node import AgentNode
from beam.core.config import BEAMConfig


class AgentGraph(ABC):
    """
    Base graph class for managing agent networks.
    
    This class provides the foundation for building and executing
    multi-agent workflows with optimizable connections.
    
    Attributes:
        config: BEAM configuration
        nodes: Dictionary of agent nodes
        decision_node: Final decision-making node
        spatial_logits: Learnable spatial connection weights
        temporal_logits: Learnable temporal connection weights
        spatial_masks: Fixed spatial connection masks
        temporal_masks: Fixed temporal connection masks
    """

    def __init__(self, config: BEAMConfig):
        self.config = config
        self.id: str = shortuuid.ShortUUID().random(length=4)
        
        # Core attributes from config
        self.domain = config.domain
        self.llm_name = config.llm.model_name
        self.agent_names = config.get_agent_names()
        self.num_rounds = config.num_rounds
        
        # Optimization settings
        self.optimized_spatial = config.optimization.optimize_spatial
        self.optimized_temporal = config.optimization.optimize_temporal
        
        # Initialize nodes and edges
        self.nodes: Dict[str, AgentNode] = {}
        self.decision_node: Optional[AgentNode] = None
        self.potential_spatial_edges: List[List[str]] = []
        self.potential_temporal_edges: List[List[str]] = []
        
        # Generate masks
        spatial_masks, temporal_masks = config.generate_masks()
        self.fixed_spatial_masks = torch.tensor(spatial_masks)
        self.fixed_temporal_masks = torch.tensor(temporal_masks)
        
        # Flatten masks
        n = len(self.agent_names)
        flat_spatial = torch.tensor(spatial_masks).view(-1).float()
        flat_temporal = torch.tensor(temporal_masks).view(-1).float()
        
        # Initialize logits
        init_spatial_logit = self._compute_init_logit(
            config.optimization.initial_spatial_probability,
            self.optimized_spatial
        )
        init_temporal_logit = self._compute_init_logit(
            config.optimization.initial_temporal_probability,
            self.optimized_temporal
        )
        
        # Create parameters
        self.spatial_masks = torch.nn.Parameter(flat_spatial, requires_grad=False)
        self.temporal_masks = torch.nn.Parameter(flat_temporal, requires_grad=False)
        
        self.spatial_logits = torch.nn.Parameter(
            torch.ones(n * n, requires_grad=self.optimized_spatial) * init_spatial_logit,
            requires_grad=self.optimized_spatial
        )
        self.temporal_logits = torch.nn.Parameter(
            torch.ones(n * n, requires_grad=self.optimized_temporal) * init_temporal_logit,
            requires_grad=self.optimized_temporal
        )

    def _compute_init_logit(self, probability: float, optimized: bool) -> float:
        """Compute initial logit from probability."""
        if optimized and 0 < probability < 1:
            return float(torch.log(torch.tensor(probability / (1 - probability))))
        return 10.0

    @property
    def spatial_adj_matrix(self) -> np.ndarray:
        """Get spatial adjacency matrix."""
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self) -> np.ndarray:
        """Get temporal adjacency matrix."""
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self) -> int:
        """Get total number of active edges."""
        return sum(len(node.spatial_successors) for node in self.nodes.values())

    @property
    def num_nodes(self) -> int:
        """Get number of nodes."""
        return len(self.nodes)

    def find_node(self, node_id: str) -> AgentNode:
        """Find a node by ID."""
        if node_id in self.nodes:
            return self.nodes[node_id]
        raise ValueError(f"Node not found: {node_id}")

    def add_node(self, node: AgentNode) -> AgentNode:
        """Add a node to the graph."""
        node_id = node.id if node.id else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_potential_edges(self):
        """Initialize potential edges between all nodes."""
        self.potential_spatial_edges = []
        self.potential_temporal_edges = []
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    def clear_spatial_connections(self):
        """Clear all spatial connections."""
        for node in self.nodes.values():
            node.spatial_predecessors = []
            node.spatial_successors = []
        if self.decision_node:
            self.decision_node.spatial_predecessors = []
            self.decision_node.spatial_successors = []

    def clear_temporal_connections(self):
        """Clear all temporal connections."""
        for node in self.nodes.values():
            node.temporal_predecessors = []
            node.temporal_successors = []

    def connect_decision_node(self, last_node_id: Optional[str] = None):
        """Connect nodes to the decision node."""
        if not self.decision_node:
            return
        for node_id, node in self.nodes.items():
            if last_node_id is None or last_node_id == node_id:
                node.add_successor(self.decision_node)

    def check_cycle(self, new_node: AgentNode, target_nodes: set) -> bool:
        """Check if adding an edge would create a cycle."""
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def construct_spatial_connections(
        self,
        temperature: float = 1.0,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """Construct spatial connections based on logits and masks."""
        self.clear_spatial_connections()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for i, (edge, logit, mask) in enumerate(zip(
            self.potential_spatial_edges,
            self.spatial_logits,
            self.spatial_masks
        )):
            out_node = self.find_node(edge[0])
            in_node = self.find_node(edge[1])
            
            if mask == 0.0:
                continue
            elif mask == 1.0 and not self.optimized_spatial:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                continue
            
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node, 'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
        
        return torch.sum(torch.stack(log_probs))

    def construct_temporal_connections(
        self,
        round_idx: int = 0,
        temperature: float = 1.0,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """Construct temporal connections based on logits and masks."""
        self.clear_temporal_connections()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        
        if round_idx == 0:
            return torch.sum(torch.stack(log_probs))
        
        for edge, logit, mask in zip(
            self.potential_temporal_edges,
            self.temporal_logits,
            self.temporal_masks
        ):
            out_node = self.find_node(edge[0])
            in_node = self.find_node(edge[1])
            
            if mask == 0.0:
                continue
            elif mask == 1.0 and not self.optimized_temporal:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')
                continue
            
            edge_prob = torch.sigmoid(logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node, 'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
        
        return torch.sum(torch.stack(log_probs))

    def update_memory(self):
        """Update memory for all nodes."""
        for node in self.nodes.values():
            node.update_memory()

    def update_masks(self, pruning_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update masks by pruning low-weight edges."""
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masked = (self.spatial_masks == 0).sum()
            prune_count = max(1, int(torch.round(num_edges * pruning_rate)))
            
            logits = self.spatial_logits.clone()
            min_logit = logits.min()
            logits[self.spatial_masks == 0] = min_logit - 1.0
            
            sorted_idx = torch.argsort(logits)
            prune_idx = sorted_idx[:int(prune_count + num_masked)]
            self.spatial_masks.data[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masked = (self.temporal_masks == 0).sum()
            prune_count = max(1, int(torch.round(num_edges * pruning_rate)))
            
            logits = self.temporal_logits.clone()
            min_logit = logits.min()
            logits[self.temporal_masks == 0] = min_logit - 1.0
            
            sorted_idx = torch.argsort(logits)
            prune_idx = sorted_idx[:int(prune_count + num_masked)]
            self.temporal_masks.data[prune_idx] = 0
        
        return self.spatial_masks, self.temporal_masks

    def run(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None,
        max_retries: int = 3,
        max_time: int = 600
    ) -> Tuple[List[Any], torch.Tensor]:
        """Execute the graph synchronously."""
        num_rounds = num_rounds or self.num_rounds
        log_probs = torch.tensor(0.0)
        
        for round_idx in range(num_rounds):
            log_probs = log_probs + self.construct_spatial_connections()
            log_probs = log_probs + self.construct_temporal_connections(round_idx)
            
            # Topological execution
            in_degree = {
                node_id: len(node.spatial_predecessors)
                for node_id, node in self.nodes.items()
            }
            queue = [nid for nid, deg in in_degree.items() if deg == 0]
            
            while queue:
                current_id = queue.pop(0)
                for attempt in range(max_retries):
                    try:
                        self.nodes[current_id].execute(inputs)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Node {current_id} failed: {e}")
                
                for successor in self.nodes[current_id].spatial_successors:
                    if successor.id in self.nodes:
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            queue.append(successor.id)
            
            self.update_memory()
        
        # Execute decision node
        self.connect_decision_node()
        if self.decision_node:
            self.decision_node.execute(inputs)
            final_answers = self.decision_node.outputs
        else:
            final_answers = list(self.nodes.values())[-1].outputs if self.nodes else []
        
        if not final_answers:
            final_answers = ["No answer from decision node"]
        
        return final_answers, log_probs

    async def arun(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None,
        max_retries: int = 3,
        max_time: int = 600
    ) -> Tuple[List[Any], torch.Tensor]:
        """Execute the graph asynchronously."""
        num_rounds = num_rounds or self.num_rounds
        log_probs = torch.tensor(0.0)
        
        for round_idx in range(num_rounds):
            log_probs = log_probs + self.construct_spatial_connections()
            log_probs = log_probs + self.construct_temporal_connections(round_idx)
            
            # Topological execution
            in_degree = {
                node_id: len(node.spatial_predecessors)
                for node_id, node in self.nodes.items()
            }
            queue = [nid for nid, deg in in_degree.items() if deg == 0]
            
            while queue:
                current_id = queue.pop(0)
                for attempt in range(max_retries):
                    try:
                        await asyncio.wait_for(
                            self.nodes[current_id].async_execute(inputs),
                            timeout=max_time
                        )
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Node {current_id} failed: {e}")
                
                for successor in self.nodes[current_id].spatial_successors:
                    if successor.id in self.nodes:
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            queue.append(successor.id)
            
            self.update_memory()
        
        # Execute decision node
        self.connect_decision_node()
        if self.decision_node:
            await self.decision_node.async_execute(inputs)
            final_answers = self.decision_node.outputs
        else:
            final_answers = list(self.nodes.values())[-1].outputs if self.nodes else []
        
        if not final_answers:
            final_answers = ["No answer from decision node"]
        
        return final_answers, log_probs

    async def arun_state(
        self, inputs, state, num_rounds=None, max_retries=3, max_time=600
    ) -> Tuple[List[Any], torch.Tensor]:
        """Execute a feasible state exactly, with request-local agent buffers.

        Spatial edges are fixed by A. Cross-round memory follows the configured
        temporal mask, restricted to active endpoints, and is deterministic.
        LLM clients and execution callbacks are shared; outputs/connections are not.
        """
        from beam.core.probability import GraphState

        if not isinstance(state, GraphState) or not state.feasible():
            raise ValueError("Expected a feasible GraphState")
        if len(state.nodes) != len(self.nodes):
            raise ValueError("State size must match the registered agents")
        rounds = self.num_rounds if num_rounds is None else num_rounds
        if not isinstance(rounds, int) or rounds < 1 or max_retries < 1 or max_time <= 0:
            raise ValueError("Round count, retries and timeout must be positive")
        if not any(state.nodes):
            return [], torch.tensor(0.0)

        def isolated(node):
            result = copy.copy(node)
            result.clear_connections()
            result.inputs, result.outputs, result.raw_inputs = [], [], []
            result.last_memory = {"inputs": [], "outputs": [], "raw_inputs": []}
            return result

        nodes = [isolated(node) for node in self.nodes.values()]
        decision = isolated(self.decision_node) if self.decision_node else None
        order = [i for i in state.topological_order() if state.nodes[i]]
        for i in order:
            for j in order:
                if state.edges[i][j]:
                    nodes[i].add_successor(nodes[j], "spatial")

        async def execute(node):
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(node.async_execute(inputs), timeout=max_time)
                    return
                except Exception:
                    if attempt == max_retries - 1:
                        raise

        for round_idx in range(rounds):
            if round_idx == 1:
                for i in order:
                    for j in order:
                        if self.fixed_temporal_masks[i, j]:
                            nodes[i].add_successor(nodes[j], "temporal")
            for i in order:
                await execute(nodes[i])
            for i in order:
                nodes[i].update_memory()
        if decision:
            for i in order:
                nodes[i].add_successor(decision)
            await execute(decision)
            answers = decision.outputs
        else:
            answers = nodes[order[-1]].outputs
        return answers, torch.tensor(0.0)

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters for optimization."""
        params = []
        if self.optimized_spatial:
            params.append(self.spatial_logits)
        if self.optimized_temporal:
            params.append(self.temporal_logits)
        return params
