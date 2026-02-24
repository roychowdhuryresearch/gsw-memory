"""
PersonalMemoryProcessor — End-to-end pipeline for personal conversational memory.

Pipeline per conversation:
  1. Coref resolution on each session's full text (batch across sessions)
  2. Topic-boundary chunking on coref-resolved text
  3. GSW extraction (CONVERSATIONAL prompt, batch across all chunks)
  3b. SpaceTimeLinker (batch across all chunks, with session date context)
  4. Layer 1: per-session Reconciler → session_gsw
  5. Layer 2: ConversationReconciler.reconcile_sessions (speaker-filtered) → conversation_gsw
  6. Layer 3: ConversationReconciler.reconcile_conversations_agentic (always runs)
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..memory.models import GSWStructure
from ..memory.operator_utils.coref import CorefOperator
from ..memory.operator_utils.gsw_operator import GSWOperator
from ..memory.reconciler import Reconciler
from ..prompts.operator_prompts import PromptType
from .chunker import TopicBoundaryChunker
from .data_ingestion.locomo import Conversation, Session
from .models import ConversationMemory, PersonMemory

logger = logging.getLogger(__name__)


class PersonalMemoryProcessor:
    """Orchestrate the three-layer personal memory pipeline.

    Args:
        model_name: OpenAI model for coref resolution and GSW extraction.
        chunking_model: Model for topic-boundary detection (may be lighter).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunking_model: str = "gpt-4o-mini",
    ):
        self.model_name = model_name
        self.chunking_model = chunking_model
        self._chunker = TopicBoundaryChunker(model_name=chunking_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_conversation(self, conversation: Conversation) -> ConversationMemory:
        """Run the full 3-layer pipeline for a single conversation.

        Returns:
            ConversationMemory with session_gsws (Layer 1), gsw (Layer 2),
            and the global_gsw slot populated on the returned object via
            the CallerPersonMemory — NOTE: Layer 3 runs in process_person.
            For standalone use, ConversationMemory.gsw holds Layer 2 output.
        """
        sessions = conversation.sessions
        if not sessions:
            logger.warning("Conversation %s has no sessions.", conversation.sample_id)
            empty_gsw = GSWStructure(entity_nodes=[], verb_phrase_nodes=[])
            return ConversationMemory(
                conversation_id=conversation.sample_id,
                speaker_a=conversation.speaker_a,
                speaker_b=conversation.speaker_b,
                gsw=empty_gsw,
                session_gsws=[],
            )

        speaker_a = conversation.speaker_a
        speaker_b = conversation.speaker_b
        conv_id = conversation.sample_id

        # ---- Step 1: Coref on full session texts ----
        resolved_texts = self._run_coref(sessions)

        # ---- Step 2: Topic chunking per session ----
        # session_chunks[i] is a list of chunk strings for sessions[i]
        session_chunks: List[List[str]] = []
        for session, resolved_text in zip(sessions, resolved_texts):
            chunks = self._chunker.chunk_session_from_text(resolved_text, session)
            session_chunks.append(chunks)

        # ---- Step 3: GSW extraction (batch across ALL chunks) ----
        speaker_context = f"Speaker A: {speaker_a}, Speaker B: {speaker_b}"
        all_gsws = self._run_gsw_extraction(session_chunks, speaker_context)

        # ---- Step 3b: SpaceTimeLinker (batch across ALL chunks) ----
        all_gsws = self._run_spacetime_linking(all_gsws, session_chunks, sessions)

        # ---- Step 4: Layer 1 — per-session reconciliation ----
        session_gsws: List[GSWStructure] = []
        gsw_cursor = 0
        for sess_idx, chunks in enumerate(session_chunks):
            n = len(chunks)
            sess_gsws = all_gsws[gsw_cursor : gsw_cursor + n]
            gsw_cursor += n

            session_gsw = self._reconcile_session(
                chunks=chunks,
                gsws=sess_gsws,
                session_idx=sess_idx,
                conv_id=conv_id,
            )
            session_gsws.append(session_gsw)

        # ---- Step 5: Layer 2 — cross-session reconciliation ----
        from .reconciler import ConversationReconciler  # local import to avoid cycles

        conv_reconciler = ConversationReconciler()
        conversation_gsw = conv_reconciler.reconcile_sessions(
            session_gsws=session_gsws,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )

        return ConversationMemory(
            conversation_id=conv_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            gsw=conversation_gsw,
            session_gsws=session_gsws,
        )

    def process_person(
        self,
        person_id: str,
        conversations: List[Conversation],
    ) -> PersonMemory:
        """Process all conversations for a person and build their PersonMemory.

        Layer 3 (agentic cross-conversation reconciliation) always runs.

        Args:
            person_id: Identifier for the person (e.g. speaker name).
            conversations: All conversations involving this person.

        Returns:
            PersonMemory with all ConversationMemory objects and a global_gsw.
        """
        from .reconciler import ConversationReconciler  # local import to avoid cycles

        person_memory = PersonMemory(person_id=person_id)

        for conv in conversations:
            logger.info("Processing conversation %s for person %s", conv.sample_id, person_id)
            conv_memory = self.process_conversation(conv)
            person_memory.add_conversation(conv_memory)

        # Layer 3: agentic cross-conversation reconciliation (always runs)
        conv_reconciler = ConversationReconciler()
        global_gsw = conv_reconciler.reconcile_conversations_agentic(
            person_id=person_id,
            conversation_memories=person_memory.conversation_memories,
            model_name=self.model_name,
        )
        person_memory.global_gsw = global_gsw

        return person_memory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_coref(self, sessions: List[Session]) -> List[str]:
        """Run coreference resolution on each session's full text in parallel."""
        coref_model = CorefOperator(
            model_name=self.model_name,
            generation_params={"temperature": 0.0, "max_tokens": 4000},
        )
        inputs = [
            {"text": session.to_document(), "idx": i}
            for i, session in enumerate(sessions)
        ]
        responses = coref_model(inputs)

        # Build a sorted list by original idx
        idx_to_text: Dict[int, str] = {}
        for resp in responses.dataset:
            idx_to_text[resp["idx"]] = resp["text"]

        return [idx_to_text.get(i, sessions[i].to_document()) for i in range(len(sessions))]

    def _run_gsw_extraction(
        self,
        session_chunks: List[List[str]],
        speaker_context: str,
    ) -> List[GSWStructure]:
        """Batch-extract GSW structures for all chunks across all sessions.

        Returns a flat list of GSWStructures in the same order as the input
        (session 0 chunks first, then session 1, ...).
        """
        gsw_model = GSWOperator(
            model_name=self.model_name,
            generation_params={"temperature": 0.0},
            prompt_type=PromptType.CONVERSATIONAL,
            backend="openai",
            response_format=GSWStructure,
            backend_params={"require_all_responses": False},
        )

        flat_inputs = []
        global_idx = 0
        for sess_idx, chunks in enumerate(session_chunks):
            for chunk_idx, chunk_text in enumerate(chunks):
                flat_inputs.append(
                    {
                        "text": chunk_text,
                        "idx": global_idx,
                        "doc_idx": sess_idx,
                        "global_id": f"{sess_idx}_{chunk_idx}",
                        "speaker_context": speaker_context,
                        "context": "",
                    }
                )
                global_idx += 1

        if not flat_inputs:
            return []

        responses = gsw_model(flat_inputs)

        # Reconstruct in order
        idx_to_gsw: Dict[int, GSWStructure] = {}
        for resp in responses.dataset:
            try:
                if resp.get("gsw") is not None:
                    gsw = GSWStructure(**resp["gsw"])
                    self._derive_entity_speaker_ids(gsw)
                    idx_to_gsw[resp["idx"]] = gsw
            except Exception as exc:
                logger.warning("Failed to parse GSW for idx %s: %s", resp.get("idx"), exc)

        # Return flat list in original order; use empty GSW for failures
        result = []
        for i in range(len(flat_inputs)):
            result.append(
                idx_to_gsw.get(i, GSWStructure(entity_nodes=[], verb_phrase_nodes=[]))
            )
        return result

    def _run_spacetime_linking(
        self,
        all_gsws: List[GSWStructure],
        session_chunks: List[List[str]],
        sessions: List[Session],
    ) -> List[GSWStructure]:
        """Run SpaceTimeLinker on all chunks and apply results to GSWs.

        Clears any hallucinated spacetime from GSW extraction before applying
        real SpaceTimeLinker output. Passes session date as context so relative
        dates ("yesterday", "last week") can be resolved.
        """
        import json

        from ..memory.operator_utils.spacetime import SpaceTimeLinker, apply_spacetime_to_gsw

        # Clear hallucinated spacetime from GSW extraction
        for gsw in all_gsws:
            gsw.space_nodes = []
            gsw.time_nodes = []
            gsw.space_edges = []
            gsw.time_edges = []
            gsw.similarity_edges = []

        # Build SpaceTimeLinker inputs
        spacetime_inputs = []
        flat_idx = 0
        for sess_idx, chunks in enumerate(session_chunks):
            session_date = sessions[sess_idx].date_time if sess_idx < len(sessions) else ""
            session_ctx = f"Session date: {session_date}" if session_date else ""

            for chunk_idx, chunk_text in enumerate(chunks):
                gsw = all_gsws[flat_idx]
                if gsw.entity_nodes:
                    operator_output = {
                        "entity_nodes": [
                            {
                                "id": e.id,
                                "name": e.name,
                                "roles": [
                                    {"role": r.role, "states": r.states}
                                    for r in e.roles
                                ],
                            }
                            for e in gsw.entity_nodes
                        ]
                    }
                    spacetime_inputs.append(
                        {
                            "text_chunk_content": chunk_text,
                            "operator_output_json": json.dumps(operator_output, indent=2),
                            "session_context": session_ctx,
                            "chunk_id": f"s{sess_idx}_c{chunk_idx}",
                            "idx": flat_idx,
                            "doc_idx": sess_idx,
                            "global_id": f"s{sess_idx}_c{chunk_idx}",
                        }
                    )
                flat_idx += 1

        if not spacetime_inputs:
            return all_gsws

        # Run SpaceTimeLinker in batch
        spacetime_model = SpaceTimeLinker(
            model_name=self.model_name,
            generation_params={"temperature": 0.0, "max_tokens": 1000},
        )
        spacetime_responses = spacetime_model(spacetime_inputs)

        # Apply results to GSWs
        for resp in spacetime_responses.dataset:
            idx = resp.get("idx")
            links = resp.get("spatio_temporal_links", [])
            chunk_id = resp.get("chunk_id", f"unknown_{idx}")
            if idx is not None and idx < len(all_gsws) and links:
                apply_spacetime_to_gsw(all_gsws[idx], links, chunk_id=chunk_id)

        return all_gsws

    def _reconcile_session(
        self,
        chunks: List[str],
        gsws: List[GSWStructure],
        session_idx: int,
        conv_id: str,
    ) -> GSWStructure:
        """Run Layer-1 reconciliation for a single session's chunks.

        Also stamps conversation_id on every entity in the resulting GSW.
        """
        if not gsws:
            return GSWStructure(entity_nodes=[], verb_phrase_nodes=[])

        reconciler = Reconciler(matching_approach="exact")
        for chunk_idx, (chunk_text, gsw) in enumerate(zip(chunks, gsws)):
            if gsw is None:
                continue
            chunk_id = f"session{session_idx}_chunk{chunk_idx}"
            reconciler.reconcile(new_gsw=gsw, chunk_id=chunk_id, new_chunk_text=chunk_text)

        session_gsw = reconciler.global_memory
        if session_gsw is None:
            return GSWStructure(entity_nodes=[], verb_phrase_nodes=[])

        # Stamp conversation_id on entities, roles, and questions
        for entity in session_gsw.entity_nodes:
            entity.conversation_id = conv_id
            for role in entity.roles:
                role.conversation_id = conv_id

        for vp in session_gsw.verb_phrase_nodes:
            for question in vp.questions:
                if question.conversation_id is None:
                    question.conversation_id = conv_id

        return session_gsw

    @staticmethod
    def _derive_entity_speaker_ids(gsw: GSWStructure) -> None:
        """Derive EntityNode.speaker_id from its roles' speaker_ids (in-place).

        If all non-None role speaker_ids agree → set entity.speaker_id to that value.
        Otherwise (mixed speakers or all None) → leave as None.
        """
        for entity in gsw.entity_nodes:
            role_speakers = {
                r.speaker_id for r in entity.roles if r.speaker_id is not None
            }
            if len(role_speakers) == 1:
                entity.speaker_id = role_speakers.pop()
