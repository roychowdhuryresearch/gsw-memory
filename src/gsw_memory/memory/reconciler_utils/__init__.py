"""
GSW Reconciler Components - Entity matching strategies and indices.

This module provides matching strategies and entity indices for the GSW reconciler,
supporting both exact match and embedding-based approaches.
"""

from .matching import (
    ExactMatchStrategy, 
    EmbeddingMatchStrategy,
    ExactMatchEntityIndex,
    EmbeddingEntityIndex,
    create_matching_components,
    MatchingStrategy,
    EntityIndex
)

__all__ = [
    "ExactMatchStrategy", 
    "EmbeddingMatchStrategy",
    "ExactMatchEntityIndex",
    "EmbeddingEntityIndex",
    "create_matching_components",
    "MatchingStrategy",
    "EntityIndex",
]