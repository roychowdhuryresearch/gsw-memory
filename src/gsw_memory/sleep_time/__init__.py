"""
Sleep-time computation module for GSW memory.

Enables agent-driven exploration and reconciliation of GSW structures
to create explicit bridges for implicit multi-hop knowledge.
"""

from .tools import GSWTools
from .agentic_reconciler import AgenticReconciler

__all__ = ["GSWTools", "AgenticReconciler"]
