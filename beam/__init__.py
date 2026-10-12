"""
BEAM: Bayesian Energy-Aware Multi-Agent Communication
A toolkit for token-efficient multi-agent inference optimization.

Supports three optimization strategies:
- AgentPrune: Edge pruning based on learned importance
- AgentDropout: Dynamic agent dropout during inference  
- AgentBayesian: Bayesian optimization with optional MCMC sampling

Provides:
- Extensible prompt template management
- Lightweight integration utilities for LangChain and LangGraph
"""

from beam.core.config import BEAMConfig, OptimizationConfig, AgentConfig
from beam.core.optimizer import TokenOptimizer
from beam.core.graph import AgentGraph
from beam.core.node import AgentNode, create_agent_node
from beam.core.llm import BaseLLM, LLMRegistry
from beam.core.probability import GraphState
from beam.strategies import AgentPrune, AgentDropout, AgentBayesian
from beam.prompts import PromptRegistry, PromptTemplate, PromptSet

__version__ = "0.3.0"
__all__ = [
    # Config
    "BEAMConfig",
    "OptimizationConfig",
    "AgentConfig",
    # Core
    "TokenOptimizer",
    "AgentGraph",
    "GraphState",
    "AgentNode",
    "create_agent_node",
    "BaseLLM",
    "LLMRegistry",
    # Strategies
    "AgentPrune",
    "AgentDropout",
    "AgentBayesian",
    # Prompts
    "PromptRegistry",
    "PromptTemplate",
    "PromptSet",
]
