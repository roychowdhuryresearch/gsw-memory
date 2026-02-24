"""
Container models for the personal memory system.

These dataclasses hold the three-layer hierarchical memory structure built
from conversational data (e.g. LoCoMo, LongMemEval):

    Layer 1: session_gsws  — one GSWStructure per dialogue session
    Layer 2: gsw           — ConversationMemory.gsw: all sessions reconciled
    Layer 3: global_gsw    — PersonMemory.global_gsw: cross-conversation merge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gsw_memory.memory.models import GSWStructure


@dataclass
class ConversationMemory:
    """Reconciled memory for a single two-person conversation.

    Attributes:
        conversation_id: Dataset identifier (e.g. "conv-26").
        speaker_a: Name of speaker A.
        speaker_b: Name of speaker B.
        gsw: Layer-2 GSWStructure — all session GSWs reconciled within this
             conversation, with speaker attribution preserved.
        session_gsws: Layer-1 GSWStructures — one per dialogue session, in
                      chronological order.
    """

    conversation_id: str
    speaker_a: str
    speaker_b: str
    gsw: GSWStructure
    session_gsws: List[GSWStructure] = field(default_factory=list)

    def speakers(self) -> List[str]:
        """Return [speaker_a, speaker_b]."""
        return [self.speaker_a, self.speaker_b]


@dataclass
class PersonMemory:
    """Unified memory for a single person across all their conversations.

    Attributes:
        person_id: Canonical name of the person (e.g. "John").
        conversation_memories: Mapping from conversation_id to
                               ConversationMemory for every conversation this
                               person appears in.
        global_gsw: Layer-3 GSWStructure — cross-conversation entity
                    reconciliation.  None until
                    ConversationReconciler.reconcile_conversations_agentic()
                    has been called.
    """

    person_id: str
    conversation_memories: Dict[str, ConversationMemory] = field(default_factory=dict)
    global_gsw: Optional[GSWStructure] = None

    def add_conversation(self, memory: ConversationMemory) -> None:
        """Register a ConversationMemory for this person."""
        self.conversation_memories[memory.conversation_id] = memory

    def conversations_for_speaker(self, speaker_name: str) -> List[ConversationMemory]:
        """Return all ConversationMemory instances where speaker_name is one of
        the two participants."""
        return [
            cm
            for cm in self.conversation_memories.values()
            if speaker_name in (cm.speaker_a, cm.speaker_b)
        ]
