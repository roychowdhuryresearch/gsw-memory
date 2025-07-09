"""
GSW Memory Prompts.

This module provides prompt templates for various GSW components.
"""

from .aggregator_prompts import EntitySummaryPrompts, VerbPhraseSummaryPrompts
from .operator_prompts import CorefPrompts, ContextPrompts, EventBoundaryPrompts, ConversationAnalysisPrompts, ConversationDetectionPrompts
from .reconciler_prompts import *

__all__ = [
    "EntitySummaryPrompts",
    "VerbPhraseSummaryPrompts",
    "CorefPrompts", 
    "ContextPrompts",
    "EventBoundaryPrompts",
    "ConversationAnalysisPrompts",
    "ConversationDetectionPrompts",
]