"""
Base classes for GSW aggregation.

This module defines the abstract interfaces and data structures used by all
aggregator implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..models import GSWStructure


@dataclass
class AggregatedView:
    """
    Represents an aggregated view of GSW data for a specific query.

    This data structure contains the processed information extracted from
    a GSW structure in a format suitable for LLM consumption.
    """

    view_type: str
    """Type of aggregation (e.g., 'entity_summary', 'temporal_sequence')"""

    content: Dict[str, Any]
    """The aggregated content in a structured format"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the aggregation process"""

    relevance_scores: Optional[Dict[str, float]] = None
    """Optional relevance scores for ranking/reordering"""


class BaseAggregator(ABC):
    """
    Abstract base class for all GSW aggregators.

    Aggregators transform GSW structures into query-relevant views that can
    be consumed by LLMs or other downstream systems.
    """

    def __init__(self, gsw: GSWStructure):
        """
        Initialize the aggregator with a GSW structure.

        Args:
            gsw: The GSW structure to aggregate
        """
        self.gsw = gsw

    @abstractmethod
    def aggregate(self, query: str, **kwargs) -> AggregatedView:
        """
        Transform GSW data into a query-relevant aggregated view.

        Args:
            query: The query or question driving the aggregation
            **kwargs: Additional parameters specific to the aggregator type

        Returns:
            An AggregatedView containing the processed information
        """
        pass

    @abstractmethod
    def get_context(self, aggregated_view: AggregatedView) -> str:
        """
        Format an aggregated view into a string suitable for LLM consumption.

        Args:
            aggregated_view: The aggregated view to format

        Returns:
            A formatted string for LLM context
        """
        pass
