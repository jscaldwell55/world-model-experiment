# environments/base.py
from abc import ABC, abstractmethod
from typing import Any
import hashlib
import inspect

class Environment(ABC):
    """
    Base environment with strict contracts for reproducibility.
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self._version_hash = self._compute_version_hash()
        
    @abstractmethod
    def reset(self, seed: int) -> dict:
        """
        Reset environment to initial state.
        Returns: Initial observation (never includes ground truth)
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Execute action and return (observation, reward, done, info).
        Observation must NEVER include ground truth.
        """
        pass
    
    @abstractmethod
    def get_ground_truth(self) -> dict:
        """
        Return hidden state for EVALUATION ONLY.
        Must never be accessible to agents.
        """
        pass
    
    @abstractmethod
    def counterfactual_query(
        self, 
        action_sequence: list[str],
        seed: int
    ) -> dict:
        """
        Simulate action_sequence WITHOUT side effects.
        
        Guarantees:
        1. Deterministic given seed
        2. Side-effect free (doesn't modify self.state)
        3. Returns final observation after sequence
        
        This is used for:
        - Evaluation of counterfactual reasoning
        - Generating ground truth for do-queries
        """
        pass
    
    @abstractmethod
    def get_time_elapsed(self) -> float:
        """Return simulation time for belief likelihood calculations"""
        pass

    def apply_shift(self, shift_type: str, **kwargs) -> dict:
        """
        Apply distribution shift to environment (optional).

        Common shift types:
        - "wiring_change": Change wiring/dynamics
        - "sensor_noise": Add observation noise
        - "parameter_change": Change physical parameters

        Args:
            shift_type: Type of shift to apply
            **kwargs: Shift-specific parameters

        Returns:
            Dict with shift info (what changed)
        """
        # Default: no shift support
        return {"supported": False, "message": "This environment does not support distribution shifts"}

    def _compute_version_hash(self) -> str:
        """Hash environment source code for versioning"""
        source = inspect.getsource(self.__class__)
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def get_version(self) -> str:
        """Return version hash for provenance logging"""
        return self._version_hash