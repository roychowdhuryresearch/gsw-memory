"""
GSW Reconciler - System for reconciling new GSW structures with existing global memory.

This module provides the main Reconciler class and supporting components for entity matching,
merging, and evolution tracking across chunks.
"""

from .reconciler import Reconciler
from .matching import (
    ExactMatchStrategy, 
    EmbeddingMatchStrategy,
    ExactMatchEntityIndex,
    EmbeddingEntityIndex,
    create_matching_components
)

__all__ = [
    "Reconciler",
    "ExactMatchStrategy", 
    "EmbeddingMatchStrategy",
    "ExactMatchEntityIndex",
    "EmbeddingEntityIndex",
    "create_matching_components",
]