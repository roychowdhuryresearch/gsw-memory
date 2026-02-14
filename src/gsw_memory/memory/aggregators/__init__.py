"""
GSW Memory Aggregators.

This module provides aggregation interfaces and implementations for transforming
GSW structures into query-relevant views for LLM consumption.
"""

from .base import BaseAggregator, AggregatedView
from .entity_summary import EntitySummaryAggregator

__all__ = [
    "BaseAggregator",
    "AggregatedView", 
    "EntitySummaryAggregator",
]