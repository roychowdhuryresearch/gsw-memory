"""
GSW Memory Prompts.

This module provides prompt templates for various GSW components.
"""

from .aggregator_prompts import EntitySummaryPrompts
from .operator_prompts import CorefPrompts, ContextPrompts
from .reconciler_prompts import EntityVerificationPrompts, QuestionResolutionPrompts

__all__ = [
    "EntitySummaryPrompts",
    "CorefPrompts",
    "ContextPrompts",
    "EntityVerificationPrompts",
    "QuestionResolutionPrompts",
]
