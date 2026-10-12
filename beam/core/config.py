"""Configuration management for BEAM toolkit."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any
from enum import Enum


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    PRUNE = "prune"
    DROPOUT = "dropout"
    BAYESIAN = "bayesian"


class ConnectionMode(str, Enum):
    """Agent connection topology modes."""
    FULL_CONNECTED = "full_connected"
    CHAIN = "chain"
    STAR = "star"
    LAYERED = "layered"
    DEBATE = "debate"
    RANDOM = "random"
    CUSTOM = "custom"


class DecisionMethod(str, Enum):
    """Decision aggregation methods."""
    REFER = "refer"
    DIRECT = "direct"
    MAJOR_VOTE = "major_vote"
    WEIGHTED = "weighted"


@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.2
    timeout: int = 600


@dataclass
class OptimizationConfig:
    """Optimization-specific configuration."""
    strategy: OptimizationStrategy = OptimizationStrategy.PRUNE
    
    # Spatial/Temporal optimization
    optimize_spatial: bool = True
    optimize_temporal: bool = True
    initial_spatial_probability: float = 0.5
    initial_temporal_probability: float = 0.5
    
    # Pruning parameters
    pruning_rate: float = 0.25
    pruning_iterations: int = 5
    
    # Dropout parameters
    dropout_rate: float = 0.2
    
    # Bayesian sampling and deployment
    use_bayesian: bool = False
    use_mcmc: bool = True
    mcmc_samples: int = 10
    deployment_samples: int = 10
    consensus_threshold: float = 0.5
    candidate_threshold: Optional[float] = None
    energy_initial: float = 1.0
    energy_gain: float = 0.1
    energy_cost: float = 0.1
    energy_dt: float = 1.0
    uncertainty_scale: float = 0.1
    coupling_epsilon: float = 1e-6
    edge_threshold: float = 0.5
    node_energy_weight: float = 0.1
    node_threshold: Optional[float] = None  # n - 1 when omitted
    gp_inducing_points: int = 10
    gp_length_scale: float = 1.0
    gp_scale: float = 1.0
    gp_jitter: float = 1e-6
    nuclear_weight: float = 0.01
    variance_weight: float = 0.01
    prior_weight: float = 0.005
    random_seed: Optional[int] = None

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 40
    num_iterations: int = 20


@dataclass
class AgentConfig:
    """Single agent configuration."""
    name: str
    role: Optional[str] = None
    count: int = 1
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BEAMConfig:
    """Main configuration for BEAM toolkit."""
    
    # Agent configuration
    agents: List[AgentConfig] = field(default_factory=list)
    decision_method: DecisionMethod = DecisionMethod.REFER
    
    # Graph configuration
    connection_mode: ConnectionMode = ConnectionMode.FULL_CONNECTED
    num_rounds: int = 3
    max_retries: int = 3
    
    # Custom masks (optional)
    spatial_masks: Optional[List[List[int]]] = None
    temporal_masks: Optional[List[List[int]]] = None
    
    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Domain/task specific
    domain: str = "general"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BEAMConfig":
        """Create config from dictionary."""
        config_dict = dict(config_dict)
        llm_config = LLMConfig(**config_dict.pop("llm", {}))
        opt_config = OptimizationConfig(**config_dict.pop("optimization", {}))
        
        agents = []
        for agent_data in config_dict.pop("agents", []):
            agents.append(AgentConfig(**agent_data))
        
        return cls(
            agents=agents,
            llm=llm_config,
            optimization=opt_config,
            **config_dict
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BEAMConfig":
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def get_agent_names(self) -> List[str]:
        """Get flattened list of agent names based on counts."""
        names = []
        for agent in self.agents:
            names.extend([agent.name] * agent.count)
        return names
    
    def get_num_agents(self) -> int:
        """Get total number of agents."""
        return sum(agent.count for agent in self.agents)
    
    def generate_masks(self) -> tuple:
        """Generate spatial and temporal masks based on connection mode."""
        n = self.get_num_agents()
        
        if self.spatial_masks is not None and self.temporal_masks is not None:
            return self.spatial_masks, self.temporal_masks
        
        if self.connection_mode == ConnectionMode.FULL_CONNECTED:
            spatial = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
            temporal = [[1 for _ in range(n)] for _ in range(n)]
        
        elif self.connection_mode == ConnectionMode.CHAIN:
            spatial = [[1 if j == i + 1 else 0 for j in range(n)] for i in range(n)]
            temporal = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        
        elif self.connection_mode == ConnectionMode.STAR:
            # First agent is the center
            spatial = [[0] * n for _ in range(n)]
            for i in range(1, n):
                spatial[0][i] = 1
                spatial[i][0] = 1
            temporal = [[1 for _ in range(n)] for _ in range(n)]
        
        elif self.connection_mode == ConnectionMode.LAYERED:
            # Agents in pairs, each pair connects to next
            spatial = [[0] * n for _ in range(n)]
            for i in range(0, n - 1, 2):
                if i + 1 < n:
                    spatial[i][i + 1] = 1
                    spatial[i + 1][i] = 1
                if i + 2 < n:
                    spatial[i][i + 2] = 1
                    spatial[i + 1][i + 2] = 1
            temporal = [[1 for _ in range(n)] for _ in range(n)]
        
        elif self.connection_mode == ConnectionMode.DEBATE:
            # All agents debate with each other
            spatial = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
            temporal = [[1 for _ in range(n)] for _ in range(n)]
        
        else:  # RANDOM or CUSTOM
            import random
            spatial = [[random.randint(0, 1) if i != j else 0 for j in range(n)] for i in range(n)]
            temporal = [[1 for _ in range(n)] for _ in range(n)]
        
        return spatial, temporal
