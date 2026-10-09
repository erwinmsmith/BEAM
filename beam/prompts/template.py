"""
Prompt template system for BEAM toolkit.

Provides a flexible, extensible way to define and manage prompts
for multi-agent systems.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from string import Template
import json


@dataclass
class PromptTemplate:
    """
    A single prompt template with variable substitution.
    
    Supports both simple string templates and callable generators.
    
    Example:
        ```python
        # Simple template
        template = PromptTemplate(
            name="solver",
            system="You are a {role} solving {domain} problems.",
            user="Problem: {task}\\nContext: {context}"
        )
        
        # Generate prompts
        system, user = template.render(
            role="mathematician",
            domain="algebra",
            task="Solve x + 2 = 5",
            context=""
        )
        
        # With callable
        def dynamic_system(role, domain, **kwargs):
            return f"You are an expert {role} in {domain}."
        
        template = PromptTemplate(
            name="dynamic",
            system=dynamic_system,
            user="Task: {task}"
        )
        ```
    """
    name: str
    system: Union[str, Callable] = ""
    user: Union[str, Callable] = ""
    description: str = ""
    variables: List[str] = field(default_factory=list)
    
    def render(self, **kwargs) -> tuple:
        """
        Render the template with given variables.
        
        Args:
            **kwargs: Variables to substitute
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Render system prompt
        if callable(self.system):
            system_prompt = self.system(**kwargs)
        else:
            system_prompt = self._substitute(self.system, kwargs)
        
        # Render user prompt
        if callable(self.user):
            user_prompt = self.user(**kwargs)
        else:
            user_prompt = self._substitute(self.user, kwargs)
        
        return system_prompt, user_prompt
    
    def _substitute(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in template string."""
        try:
            # Try format-style substitution first
            return template.format(**variables)
        except KeyError:
            # Fall back to partial substitution
            result = template
            for key, value in variables.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "name": self.name,
            "system": self.system if isinstance(self.system, str) else "<callable>",
            "user": self.user if isinstance(self.user, str) else "<callable>",
            "description": self.description,
            "variables": self.variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            system=data.get("system", ""),
            user=data.get("user", ""),
            description=data.get("description", ""),
            variables=data.get("variables", [])
        )


@dataclass
class PromptSet:
    """
    A collection of prompt templates for a specific domain or use case.
    
    Manages multiple roles/agents with their respective prompts.
    
    Example:
        ```python
        # Create a prompt set for math problems
        math_prompts = PromptSet(
            name="math",
            description="Prompts for mathematical problem solving"
        )
        
        # Add role templates
        math_prompts.add_role(
            role="solver",
            system="You are a math expert. Solve problems step by step.",
            user="Problem: {task}\\n\\nOther solutions: {context}"
        )
        
        math_prompts.add_role(
            role="verifier",
            system="You verify mathematical solutions for correctness.",
            user="Problem: {task}\\nSolution to verify: {context}"
        )
        
        # Add decision template
        math_prompts.set_decision_template(
            system="You synthesize multiple solutions into a final answer.",
            user="Problem: {task}\\nSolutions: {context}\\nFinal answer:"
        )
        
        # Use
        system, user = math_prompts.get_prompt("solver", task="2+2", context="")
        ```
    """
    name: str
    description: str = ""
    roles: Dict[str, PromptTemplate] = field(default_factory=dict)
    decision_template: Optional[PromptTemplate] = None
    default_variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_role(
        self,
        role: str,
        system: Union[str, Callable] = "",
        user: Union[str, Callable] = "",
        description: str = ""
    ) -> "PromptSet":
        """
        Add a role template to the set.
        
        Args:
            role: Role name
            system: System prompt template
            user: User prompt template
            description: Role description
            
        Returns:
            Self for chaining
        """
        self.roles[role] = PromptTemplate(
            name=role,
            system=system,
            user=user,
            description=description
        )
        return self
    
    def set_decision_template(
        self,
        system: Union[str, Callable] = "",
        user: Union[str, Callable] = ""
    ) -> "PromptSet":
        """
        Set the decision/aggregation template.
        
        Args:
            system: System prompt for decision agent
            user: User prompt for decision agent
            
        Returns:
            Self for chaining
        """
        self.decision_template = PromptTemplate(
            name="decision",
            system=system,
            user=user
        )
        return self
    
    def get_prompt(
        self,
        role: str,
        **kwargs
    ) -> tuple:
        """
        Get rendered prompts for a role.
        
        Args:
            role: Role name
            **kwargs: Variables to substitute
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if role not in self.roles:
            raise ValueError(f"Role '{role}' not found. Available: {list(self.roles.keys())}")
        
        # Merge default variables with provided ones
        variables = {**self.default_variables, **kwargs}
        return self.roles[role].render(**variables)
    
    def get_decision_prompt(self, **kwargs) -> tuple:
        """
        Get rendered prompts for decision agent.
        
        Args:
            **kwargs: Variables to substitute
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if self.decision_template is None:
            # Default decision template
            return (
                "Synthesize the provided information and give a final answer.",
                "Task: {task}\n\nResponses:\n{context}\n\nFinal answer:".format(**kwargs)
            )
        
        variables = {**self.default_variables, **kwargs}
        return self.decision_template.render(**variables)
    
    def get_roles(self) -> List[str]:
        """Get list of available roles."""
        return list(self.roles.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "roles": {k: v.to_dict() for k, v in self.roles.items()},
            "decision_template": self.decision_template.to_dict() if self.decision_template else None,
            "default_variables": self.default_variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptSet":
        """Create from dictionary."""
        prompt_set = cls(
            name=data["name"],
            description=data.get("description", ""),
            default_variables=data.get("default_variables", {})
        )
        
        for role, template_data in data.get("roles", {}).items():
            prompt_set.roles[role] = PromptTemplate.from_dict(template_data)
        
        if data.get("decision_template"):
            prompt_set.decision_template = PromptTemplate.from_dict(data["decision_template"])
        
        return prompt_set
    
    def save(self, path: str):
        """Save prompt set to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "PromptSet":
        """Load prompt set from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
