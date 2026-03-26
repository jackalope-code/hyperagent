import os
from abc import ABC, abstractmethod
from typing import Callable


class AgentSystem(ABC):
    """Base class for all agents in the HyperAgents system."""

    def __init__(self, model: str = None, log: Callable = print):
        self.model = model or os.getenv("MODEL", "claude-opus-4-5")
        self.log = log

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Execute the agent's primary logic."""
