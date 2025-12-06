"""
NeSy Bridge: Neural-Symbolic Integration for Molecular Property Prediction.

This module implements Phase 3 of the world model architecture:
1. SemanticMemory: Persistent storage of validated SAR rules with Bayesian updating
2. ConsistencyChecker: Measure alignment between neural and symbolic predictions
3. HybridPredictor: Combine neural and symbolic predictions

Core Principle: The neural model (fast, implicit) and semantic memory
(interpretable, explicit) should converge. As learning progresses,
their predictions should align.

Future Direction (Phase 3+):
- SemanticMemory rules become structured prompts for LLM
- HybridPredictor becomes the "slow path" that validates LLM "fast path"
- Disagreement between LLM and Hybrid triggers uncertainty flag
"""

from .semantic_memory import SemanticMemory
from .consistency_checker import ConsistencyChecker
from .hybrid_predictor import HybridPredictor

__all__ = ['SemanticMemory', 'ConsistencyChecker', 'HybridPredictor']
