"""
GSW Memory Aggregators.

This module provides aggregation interfaces and implementations for transforming
GSW structures into query-relevant views for LLM consumption.
"""

from .base import BaseAggregator, AggregatedView
from .entity_summary import EntitySummaryAggregator
from .verb_summary import VerbSummaryAggregator
from .conversation_summary import ConversationSummaryAggregator

__all__ = [
    "BaseAggregator",
    "AggregatedView", 
    "EntitySummaryAggregator",
    "VerbSummaryAggregator",
    "ConversationSummaryAggregator",
]