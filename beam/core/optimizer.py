"""Token optimizer for BEAM toolkit."""

from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
import asyncio
import copy

from beam.core.config import BEAMConfig, OptimizationStrategy
from beam.core.graph import AgentGraph


def nuclear_norm(matrix: torch.Tensor) -> torch.Tensor:
    """Compute nuclear norm of a matrix."""
    return torch.norm(matrix, p='nuc')


def frobenius_norm(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius norm of difference between two matrices."""
    return torch.norm(matrix1 - matrix2, p='fro')


class TokenOptimizer:
    """
    Main optimizer class for token-efficient multi-agent inference.
    
    Supports three optimization strategies:
    - PRUNE: Learn edge importance and prune low-weight connections
    - DROPOUT: Dynamically skip agents during inference
    - BAYESIAN: Use Bayesian optimization with optional MCMC sampling
    
    Example:
        ```python
        config = BEAMConfig(...)
        optimizer = TokenOptimizer(config)
        
        # Train on a dataset
        optimizer.train(train_data, eval_fn)
        
        # Run optimized inference
        result = await optimizer.run({"task": "..."})
        ```
    """

    def __init__(self, config: BEAMConfig, graph: Optional[AgentGraph] = None):
        """
        Initialize the optimizer.
        
        Args:
            config: BEAM configuration
            graph: Optional pre-built agent graph
        """
        self.config = config
        self.graph = graph
        self.strategy = config.optimization.strategy
        
        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training_history: List[Dict[str, Any]] = []
        self._bayesian_strategy = None

    def _bayesian(self):
        from beam.strategies.bayesian import AgentBayesian

        if self.graph is None:
            raise ValueError("Graph must be set before using the optimizer")
        if self._bayesian_strategy is None or self._bayesian_strategy.graph is not self.graph:
            self._bayesian_strategy = AgentBayesian(self.config)
            self._bayesian_strategy.set_graph(self.graph)
        return self._bayesian_strategy
        
    def _setup_optimizer(self):
        """Setup PyTorch optimizer for trainable parameters."""
        if self.graph is None:
            raise ValueError("Graph must be set before training")
        
        params = self.graph.get_trainable_parameters()
        if params:
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.optimization.learning_rate
            )

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_fn: Callable[[str, str], float],
        num_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        pruning_rate: Optional[float] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the optimizer on a dataset.
        
        Args:
            train_data: List of training examples with 'task' and 'answer' keys
            eval_fn: Function to evaluate predictions, returns utility score
            num_iterations: Number of training iterations
            batch_size: Batch size for training
            pruning_rate: Rate at which to prune edges
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        if self.strategy == OptimizationStrategy.BAYESIAN:
            result = self._bayesian().train(train_data, eval_fn, num_iterations, batch_size, verbose)
            self.training_history = self._bayesian_strategy.training_history
            return result
        num_iterations = num_iterations or self.config.optimization.num_iterations
        batch_size = batch_size or self.config.optimization.batch_size
        pruning_rate = pruning_rate or self.config.optimization.pruning_rate
        
        self._setup_optimizer()
        
        stats = {
            "total_solved": 0,
            "total_executed": 0,
            "losses": [],
            "accuracies": []
        }
        
        for iteration in range(num_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Process batches
            for batch_idx in range(0, len(train_data), batch_size):
                batch = train_data[batch_idx:batch_idx + batch_size]
                batch_stats = asyncio.get_event_loop().run_until_complete(
                    self._train_batch(batch, eval_fn)
                )
                
                stats["total_solved"] += batch_stats["solved"]
                stats["total_executed"] += batch_stats["executed"]
                stats["losses"].append(batch_stats["loss"])
                
                accuracy = stats["total_solved"] / stats["total_executed"]
                stats["accuracies"].append(accuracy)
                
                if verbose:
                    print(f"  Batch {batch_idx // batch_size + 1}: "
                          f"Accuracy = {accuracy:.4f}, Loss = {batch_stats['loss']:.4f}")
            
            # Prune after each iteration
            if self.config.optimization.optimize_spatial or self.config.optimization.optimize_temporal:
                self.graph.update_masks(pruning_rate)
        
        self.training_history.append(stats)
        return stats

    async def _train_batch(
        self,
        batch: List[Dict[str, Any]],
        eval_fn: Callable[[str, str], float]
    ) -> Dict[str, Any]:
        """Train on a single batch."""
        if self.optimizer is None:
            self._setup_optimizer()
        
        tasks = []
        answers = []
        
        for record in batch:
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            
            # Create a copy of the graph for this sample
            realized_graph = copy.deepcopy(self.graph)
            realized_graph.spatial_logits = self.graph.spatial_logits
            realized_graph.temporal_logits = self.graph.temporal_logits
            
            input_dict = {"task": task}
            tasks.append(realized_graph.arun(input_dict, self.config.num_rounds))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Compute losses
        loss_list = []
        solved = 0
        
        for (raw_answer, log_prob), true_answer in zip(results, answers):
            prediction = raw_answer[0] if raw_answer else ""
            utility = eval_fn(prediction, true_answer)
            
            if utility > 0:
                solved += 1
            
            # Policy gradient loss
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
        
        # Backprop
        if loss_list and self.optimizer:
            total_loss = torch.mean(torch.stack(loss_list))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_value = total_loss.item()
        else:
            loss_value = 0.0
        
        return {
            "solved": solved,
            "executed": len(batch),
            "loss": loss_value
        }

    async def run(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Run optimized inference.
        
        Args:
            inputs: Input dictionary with 'task' key
            num_rounds: Number of inference rounds
            
        Returns:
            Tuple of (answers, metadata)
        """
        if self.graph is None:
            raise ValueError("Graph must be set before running")
        if self.strategy == OptimizationStrategy.BAYESIAN:
            return await self._bayesian().run(inputs, num_rounds)
        
        num_rounds = num_rounds or self.config.num_rounds
        
        answers, log_probs = await self.graph.arun(inputs, num_rounds)
        
        metadata = {
            "log_probs": log_probs.item() if isinstance(log_probs, torch.Tensor) else log_probs,
            "num_edges": self.graph.num_edges,
            "num_nodes": self.graph.num_nodes
        }
        
        return answers, metadata

    def run_sync(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Synchronous version of run."""
        if self.strategy == OptimizationStrategy.BAYESIAN:
            return self._bayesian().run_sync(inputs, num_rounds)
        return asyncio.get_event_loop().run_until_complete(
            self.run(inputs, num_rounds)
        )

    def save_state(self, path: str):
        """Save optimizer state to file."""
        if self.strategy == OptimizationStrategy.BAYESIAN:
            return self._bayesian().save(path)
        state = {
            "config": self.config.to_dict(),
            "spatial_logits": self.graph.spatial_logits.data if self.graph else None,
            "temporal_logits": self.graph.temporal_logits.data if self.graph else None,
            "spatial_masks": self.graph.spatial_masks.data if self.graph else None,
            "temporal_masks": self.graph.temporal_masks.data if self.graph else None,
            "training_history": self.training_history
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load optimizer state from file."""
        if self.strategy == OptimizationStrategy.BAYESIAN:
            self._bayesian().load(path)
            self.training_history = self._bayesian_strategy.training_history
            return
        state = torch.load(path)
        
        if self.graph:
            if state["spatial_logits"] is not None:
                self.graph.spatial_logits.data = state["spatial_logits"]
            if state["temporal_logits"] is not None:
                self.graph.temporal_logits.data = state["temporal_logits"]
            if state["spatial_masks"] is not None:
                self.graph.spatial_masks.data = state["spatial_masks"]
            if state["temporal_masks"] is not None:
                self.graph.temporal_masks.data = state["temporal_masks"]
        
        self.training_history = state.get("training_history", [])

    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get statistics about current edge weights."""
        if self.graph is None:
            return {}
        if self.strategy == OptimizationStrategy.BAYESIAN:
            return self._bayesian().get_edge_statistics()
        
        spatial_probs = torch.sigmoid(self.graph.spatial_logits)
        temporal_probs = torch.sigmoid(self.graph.temporal_logits)
        
        return {
            "spatial": {
                "mean": spatial_probs.mean().item(),
                "std": spatial_probs.std().item(),
                "min": spatial_probs.min().item(),
                "max": spatial_probs.max().item(),
                "active_edges": (self.graph.spatial_masks > 0).sum().item()
            },
            "temporal": {
                "mean": temporal_probs.mean().item(),
                "std": temporal_probs.std().item(),
                "min": temporal_probs.min().item(),
                "max": temporal_probs.max().item(),
                "active_edges": (self.graph.temporal_masks > 0).sum().item()
            }
        }
