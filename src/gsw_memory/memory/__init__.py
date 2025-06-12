"""GSW Memory core components."""

from .models import (
    EntityNode,
    GSWStructure,
    Question,
    Role,
    SpaceNode,
    TimeNode,
    VerbPhraseNode,
)
from .operator import GSWProcessor
from .reconciler import Reconciler, reconcile_gsw_outputs

# Import individual operators for backward compatibility
from .operator_utils import (
    CorefOperator,
    ContextGenerator,
    GSWOperator,
    SpaceTimeLinker,
    chunk_text,
    extract_json_from_output,
    parse_gsw,
)

__all__ = [
    # Core data models
    "Role",
    "EntityNode",
    "Question",
    "VerbPhraseNode",
    "SpaceNode",
    "TimeNode",
    "GSWStructure",
    # Main processor and reconciler
    "GSWProcessor",
    "Reconciler",
    "reconcile_gsw_outputs",
    # Individual operators (for advanced usage)
    "CorefOperator",
    "ContextGenerator",
    "GSWOperator",
    "SpaceTimeLinker",
    # Utility functions
    "chunk_text",
    "extract_json_from_output",
    "parse_gsw",
]
