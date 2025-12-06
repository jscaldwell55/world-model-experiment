"""
Dream State: Generative exploration layer for molecular property prediction.

This module implements Phase 2 of the world model architecture:
1. AnalogGenerator: Generate virtual molecular analogs via RDKit transformations
2. SARExtractor: Extract interpretable Structure-Activity Relationship rules
3. DreamPipeline: Orchestrate generation, prediction, filtering, and augmentation

The Dream State enables the agent to explore chemical space beyond observed data
by "imagining" plausible molecular variants and predicting their properties.
"""

from .analog_generator import AnalogGenerator
from .sar_extractor import SARExtractor
from .dream_pipeline import DreamPipeline

__all__ = ['AnalogGenerator', 'SARExtractor', 'DreamPipeline']
