"""
Personal Memory subsystem for GSW.

Processes multi-session two-person conversations (e.g. LoCoMo benchmark)
into a three-layer hierarchical memory structure:

  Layer 1 — per-session GSW (TopicBoundaryChunker + GSWOperator CONVERSATIONAL + Reconciler)
  Layer 2 — conversation-level GSW (ConversationReconciler, speaker-filtered)
  Layer 3 — person-level global GSW (agentic cross-conversation reconciliation)
"""

from .chunker import TopicBoundaryChunker
from .data_ingestion.locomo import Conversation, LoCoMoLoader, QAPair, Session, Turn
from .models import ConversationMemory, PersonMemory
from .processor import PersonalMemoryProcessor
from .qa_agent import PersonalMemoryQAAgent
from .reconciler import ConversationReconciler

__all__ = [
    # Data ingestion
    "LoCoMoLoader",
    "Conversation",
    "Session",
    "Turn",
    "QAPair",
    # Memory containers
    "ConversationMemory",
    "PersonMemory",
    # Pipeline components
    "TopicBoundaryChunker",
    "PersonalMemoryProcessor",
    "ConversationReconciler",
    # QA
    "PersonalMemoryQAAgent",
]
