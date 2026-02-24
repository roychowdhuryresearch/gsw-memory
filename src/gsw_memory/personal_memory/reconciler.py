"""
ConversationReconciler — Layer 2 and Layer 3 reconciliation for personal memory.

Layer 2 (reconcile_sessions):
    Merges per-session GSWStructures into a single conversation-level GSW.
    Entities are only merge candidates if they share the same speaker_id.
    This prevents Caroline's memories from being confused with Melanie's.

Layer 3 (reconcile_conversations_agentic):
    Merges conversation-level GSWs across multiple conversations for a person.
    Uses an LLM agent with lightweight tools (get_entity_timeline,
    compare_entities, detect_contradictions) to reason about entity identity
    and state evolution across conversations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..memory.models import EntityNode, GSWStructure, Role
from ..memory.reconciler import Reconciler
from ..memory.reconciler_utils.matching import (
    EntityIndex,
    ExactMatchEntityIndex,
    MatchingStrategy,
    create_matching_components,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer-3 agent prompts
# ---------------------------------------------------------------------------

_LAYER3_SYSTEM = """You are a memory reconciliation agent. Your task is to merge entity knowledge from multiple conversations a person has had into a single unified global memory.

IMPORTANT: Before every tool call, briefly explain your reasoning — which entities you are examining and why. After each tool result, state your conclusion.

Work in four phases:

## Phase 1: Entity Reconciliation
For each pair of potentially duplicate entities across conversations:
1. Explain which entities look like duplicates and why (same name, similar role, etc.)
2. Use get_entity_timeline to inspect each candidate — read the roles, VP questions, evidence
3. Use compare_entities to check naming / role overlap
4. Use detect_contradictions to surface conflicting information
5. State your decision: MERGE (same real-world entity) or KEEP_SEPARATE (different entities)

## Phase 2: Question Resolution
1. Call get_unanswered_questions for entities that have gaps
2. Read the cross-references and VP context carefully
3. For each question you can answer from the available context, call resolve_question with the answer
4. Explain your reasoning for each resolution

## Phase 3: VP Cleanup
1. Call get_duplicate_verb_phrases to see VP groups with the same phrase
2. For each group, decide: same event → call merge_verb_phrases, different instances → keep separate
3. Explain your reasoning for each decision

## Phase 4: Finalize
1. Call get_reconciliation_summary to verify the state
2. Call finish_reconciliation with your entity merge pairs
"""

_LAYER3_USER = """Reconcile the following entity list into a unified memory for person '{person_id}'.

Entities across conversations:
{entity_summary}

Work through all four phases (entity reconciliation → question resolution → VP cleanup → finalize) before calling finish_reconciliation."""

# ---------------------------------------------------------------------------
# Speaker-filtered entity index wrapper
# ---------------------------------------------------------------------------


class _SpeakerFilteredIndex(ExactMatchEntityIndex):
    """Wraps an EntityIndex to skip cross-speaker candidates.

    Inherits from ExactMatchEntityIndex so that isinstance checks in
    ExactMatchStrategy.reconcile_entities() pass. All actual storage is
    delegated to the wrapped *inner* index.
    """

    def __init__(self, inner: EntityIndex, target_speaker: str):
        super().__init__()
        self._inner = inner
        self._target_speaker = target_speaker

    def add_entities(self, entities: List[EntityNode], batch_size: int = 32) -> None:
        # Only index entities belonging to the target speaker
        speaker_entities = [
            e for e in entities
            if e.speaker_id is None or e.speaker_id == self._target_speaker
        ]
        self._inner.add_entities(speaker_entities)

    def find_candidates(self, entity: EntityNode) -> List[EntityNode]:
        if entity.speaker_id and entity.speaker_id != self._target_speaker:
            return []
        return self._inner.find_candidates(entity)

    # Delegate everything else to the inner index
    def __getattr__(self, name: str):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# ConversationReconciler
# ---------------------------------------------------------------------------


class ConversationReconciler:
    """Reconcile session and conversation GSWs with speaker awareness.

    Args:
        matching_approach: Entity matching strategy for Layer 2 ("exact" or "embedding").
    """

    def __init__(self, matching_approach: str = "exact"):
        self.matching_approach = matching_approach

    # ------------------------------------------------------------------
    # Layer 2: cross-session reconciliation
    # ------------------------------------------------------------------

    def reconcile_sessions(
        self,
        session_gsws: List[GSWStructure],
        speaker_a: str,
        speaker_b: str,
    ) -> GSWStructure:
        """Merge per-session GSWs into a single conversation-level GSW.

        Three passes ensure entities are correctly partitioned:
        1. Speaker A pass: only entities with speaker_id == speaker_a
        2. Speaker B pass: only entities with speaker_id == speaker_b
        3. Shared pass: entities with speaker_id == None (unattributed)

        This prevents cross-speaker merging and avoids duplicating shared
        entities across both speaker passes.

        Args:
            session_gsws: Layer-1 GSWs, one per session.
            speaker_a: Name of speaker A.
            speaker_b: Name of speaker B.

        Returns:
            Reconciled GSWStructure representing the full conversation.
        """
        if not session_gsws:
            return GSWStructure(entity_nodes=[], verb_phrase_nodes=[])

        if len(session_gsws) == 1:
            return session_gsws[0]

        # Three reconciliation passes — each entity belongs to exactly one pass
        gsw_a, remap_a = self._reconcile_speaker_pass(session_gsws, speaker_a)
        gsw_b, remap_b = self._reconcile_speaker_pass(session_gsws, speaker_b)
        gsw_shared, remap_shared = self._reconcile_shared_pass(
            session_gsws, speaker_a, speaker_b
        )

        merged = self._merge_gsws([gsw_a, gsw_b, gsw_shared])

        # Collect spacetime directly from session GSWs (which have Layer-1 entity IDs).
        # Spacetime is not speaker-specific, so it bypasses the speaker passes entirely.
        all_space_nodes, all_time_nodes = [], []
        all_space_edges, all_time_edges = [], []
        seen_space_ids: set = set()
        seen_time_ids: set = set()
        for sess_gsw in session_gsws:
            for sn in sess_gsw.space_nodes:
                if sn.id not in seen_space_ids:
                    seen_space_ids.add(sn.id)
                    all_space_nodes.append(sn)
            for tn in sess_gsw.time_nodes:
                if tn.id not in seen_time_ids:
                    seen_time_ids.add(tn.id)
                    all_time_nodes.append(tn)
            all_space_edges.extend(sess_gsw.space_edges)
            all_time_edges.extend(sess_gsw.time_edges)

        # Deduplicate edges
        all_space_edges = [list(e) for e in set(tuple(e) for e in all_space_edges)]
        all_time_edges = [list(e) for e in set(tuple(e) for e in all_time_edges)]

        # Unified remap: Layer-1 entity ID → Layer-2 entity ID (all passes).
        entity_remap = {**remap_a, **remap_b, **remap_shared}

        # Collect VPs once from session GSWs (VPs are not speaker-specific).
        # Deep-copy to avoid mutating originals, then remap question answers
        # from Layer-1 → Layer-2 entity IDs.
        all_vps: list = []
        seen_vp_ids: set = set()
        for sess_gsw in session_gsws:
            for vp in sess_gsw.verb_phrase_nodes:
                if vp.id not in seen_vp_ids:
                    seen_vp_ids.add(vp.id)
                    vp_copy = vp.model_copy(deep=True)
                    for q in vp_copy.questions:
                        q.answers = [entity_remap.get(ans, ans) for ans in q.answers]
                    all_vps.append(vp_copy)
        merged.verb_phrase_nodes = all_vps

        # Remap entity IDs in spacetime edges from Layer-1 IDs → Layer-2 IDs.
        all_space_edges = [
            (entity_remap.get(eid, eid), sid) for eid, sid in all_space_edges
        ]
        all_time_edges = [
            (entity_remap.get(eid, eid), tid) for eid, tid in all_time_edges
        ]

        merged.space_nodes = all_space_nodes
        merged.time_nodes = all_time_nodes
        merged.space_edges = all_space_edges
        merged.time_edges = all_time_edges
        return merged

    def _reconcile_speaker_pass(
        self, session_gsws: List[GSWStructure], speaker: str
    ) -> tuple:
        """Run Reconciler over all session GSWs for one speaker (strict).

        Only entities with speaker_id == speaker are included. Entities with
        speaker_id == None go through the shared pass instead.

        Returns:
            Tuple of (reconciled GSW, layer1_to_layer2 entity ID remap dict).
        """
        strategy, inner_index = create_matching_components(approach=self.matching_approach)
        reconciler = Reconciler.with_strategy(strategy, inner_index)

        layer1_to_layer2: Dict[str, str] = {}

        for sess_idx, sess_gsw in enumerate(session_gsws):
            # Strict filter: only entities attributed to this speaker
            speaker_gsw = self._filter_gsw_by_speaker(sess_gsw, speaker, strict=True)
            if not speaker_gsw.entity_nodes:
                continue

            # Snapshot Layer-1 entity IDs before reconcile() mutates them
            original_ids = [e.id for e in speaker_gsw.entity_nodes]

            chunk_id = f"session{sess_idx}_{speaker}"
            reconciler.reconcile(
                new_gsw=speaker_gsw,
                chunk_id=chunk_id,
            )

            # Build remap: Layer-1 ID → final Layer-2 ID
            merge_map = reconciler._last_entity_merge_map

            for orig_l1_id in original_ids:
                l2_id = f"{chunk_id}::{orig_l1_id}"
                final_l2_id = merge_map.get(l2_id, l2_id)
                layer1_to_layer2[orig_l1_id] = final_l2_id

        gsw = reconciler.global_memory or GSWStructure(entity_nodes=[], verb_phrase_nodes=[])
        return gsw, layer1_to_layer2

    def _reconcile_shared_pass(
        self,
        session_gsws: List[GSWStructure],
        speaker_a: str,
        speaker_b: str,
    ) -> tuple:
        """Reconcile entities with no speaker attribution (speaker_id is None).

        These are shared objects, dates, locations, etc. that don't belong to
        either speaker specifically. They are reconciled once to avoid duplication.

        Returns:
            Tuple of (reconciled GSW, layer1_to_layer2 entity ID remap dict).
        """
        strategy, index = create_matching_components(approach=self.matching_approach)
        reconciler = Reconciler.with_strategy(strategy, index)

        layer1_to_layer2: Dict[str, str] = {}

        for sess_idx, sess_gsw in enumerate(session_gsws):
            # Only entities with no speaker attribution (deep copy to avoid
            # cross-pass mutation from _prepare_new_gsw)
            shared_entities = [
                e.model_copy(deep=True) for e in sess_gsw.entity_nodes
                if e.speaker_id is None
            ]
            if not shared_entities:
                continue

            shared_gsw = GSWStructure(
                entity_nodes=shared_entities,
                verb_phrase_nodes=[],
                space_nodes=[],
                time_nodes=[],
                space_edges=[],
                time_edges=[],
            )

            original_ids = [e.id for e in shared_gsw.entity_nodes]

            chunk_id = f"session{sess_idx}_shared"
            reconciler.reconcile(
                new_gsw=shared_gsw,
                chunk_id=chunk_id,
            )

            merge_map = reconciler._last_entity_merge_map

            for orig_l1_id in original_ids:
                l2_id = f"{chunk_id}::{orig_l1_id}"
                final_l2_id = merge_map.get(l2_id, l2_id)
                layer1_to_layer2[orig_l1_id] = final_l2_id

        gsw = reconciler.global_memory or GSWStructure(entity_nodes=[], verb_phrase_nodes=[])
        return gsw, layer1_to_layer2

    @staticmethod
    def _filter_gsw_by_speaker(
        gsw: GSWStructure, speaker: str, strict: bool = False
    ) -> GSWStructure:
        """Return an independent copy of *gsw* containing only entities for *speaker*.

        Deep copies entities so that the base Reconciler's in-place mutations
        (_prepare_new_gsw) don't contaminate the original session_gsws or
        subsequent passes.

        VPs are excluded (empty list) because they're not speaker-specific.
        reconcile_sessions() collects and remaps VPs separately after all
        passes complete, following the same pattern as spacetime handling.

        Args:
            gsw: Source GSW structure.
            speaker: Speaker name to filter for.
            strict: If True, only include entities with speaker_id == speaker.
                    If False (legacy), also include speaker_id == None.
        """
        if strict:
            filtered_entities = [
                e.model_copy(deep=True) for e in gsw.entity_nodes
                if e.speaker_id == speaker
            ]
        else:
            filtered_entities = [
                e.model_copy(deep=True) for e in gsw.entity_nodes
                if e.speaker_id is None or e.speaker_id == speaker
            ]
        return GSWStructure(
            entity_nodes=filtered_entities,
            verb_phrase_nodes=[],
            space_nodes=[],
            time_nodes=[],
            space_edges=[],
            time_edges=[],
        )

    @staticmethod
    def _merge_gsws(gsws: List[GSWStructure]) -> GSWStructure:
        """Combine multiple non-overlapping GSWs into one.

        Entities are concatenated. VPs are deduplicated by id (they appear
        in all passes since they aren't speaker-specific).
        """
        all_entities = []
        seen_vp_ids: set = set()
        merged_vps = []

        for gsw in gsws:
            all_entities.extend(gsw.entity_nodes)
            for vp in gsw.verb_phrase_nodes:
                if vp.id not in seen_vp_ids:
                    seen_vp_ids.add(vp.id)
                    merged_vps.append(vp)

        return GSWStructure(
            entity_nodes=all_entities,
            verb_phrase_nodes=merged_vps,
        )

    # ------------------------------------------------------------------
    # Layer 3: agentic cross-conversation reconciliation
    # ------------------------------------------------------------------

    def reconcile_conversations_agentic(
        self,
        person_id: str,
        conversation_memories: Dict[str, Any],  # conv_id -> ConversationMemory
        model_name: str = "gpt-4o",
        max_iterations: int = 20,
    ) -> GSWStructure:
        """Merge conversation GSWs across all conversations for a person.

        Uses an LLM agent with tool calls to reason about entity identity
        across conversations and produce a unified global GSW.

        Args:
            person_id: Identifier for the focal person.
            conversation_memories: Dict mapping conv_id to ConversationMemory.
            model_name: Model for agentic reasoning.
            max_iterations: Max tool-call rounds.

        Returns:
            Unified GSWStructure across all conversations.
        """
        if not conversation_memories:
            return GSWStructure(entity_nodes=[], verb_phrase_nodes=[])

        # Collect all entities from all conversation GSWs
        all_entities: List[EntityNode] = []
        for conv_id, conv_mem in conversation_memories.items():
            for entity in conv_mem.gsw.entity_nodes:
                if entity.conversation_id is None:
                    entity.conversation_id = conv_id
                all_entities.append(entity)

        if not all_entities:
            return GSWStructure(entity_nodes=[], verb_phrase_nodes=[])

        # Collect all VPs from all conversations
        all_vps = []
        for conv_mem in conversation_memories.values():
            all_vps.extend(conv_mem.gsw.verb_phrase_nodes)

        # Collect all spacetime from all conversations
        all_space_nodes, all_time_nodes = [], []
        all_space_edges, all_time_edges = [], []
        for conv_mem in conversation_memories.values():
            all_space_nodes.extend(conv_mem.gsw.space_nodes)
            all_time_nodes.extend(conv_mem.gsw.time_nodes)
            all_space_edges.extend(conv_mem.gsw.space_edges)
            all_time_edges.extend(conv_mem.gsw.time_edges)

        # Build entity summary for the agent
        entity_summary = self._build_entity_summary(all_entities)

        # Run agentic loop — agent can merge entities, resolve questions, dedup VPs
        merged_entities, all_vps = self._run_agent(
            person_id=person_id,
            entity_summary=entity_summary,
            all_entities=all_entities,
            all_vps=all_vps,
            model_name=model_name,
            max_iterations=max_iterations,
        )

        # Remap entity IDs in spacetime edges for any merges
        entity_id_remap = getattr(self, '_entity_id_remap', {})
        if entity_id_remap:
            all_space_edges = [
                (entity_id_remap.get(eid, eid), sid)
                for eid, sid in all_space_edges
            ]
            all_time_edges = [
                (entity_id_remap.get(eid, eid), tid)
                for eid, tid in all_time_edges
            ]

        return GSWStructure(
            entity_nodes=merged_entities,
            verb_phrase_nodes=all_vps,
            space_nodes=all_space_nodes,
            time_nodes=all_time_nodes,
            space_edges=all_space_edges,
            time_edges=all_time_edges,
        )

    # ------------------------------------------------------------------
    # Agent helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_entity_summary(entities: List[EntityNode]) -> str:
        lines = []
        for i, e in enumerate(entities):
            role_strs = "; ".join(
                f"{r.role}" + (f" [{', '.join(r.states)}]" if r.states else "")
                for r in e.roles
            )
            speaker = e.speaker_id or "unknown"
            conv = e.conversation_id or "unknown"
            lines.append(f"{i}. [{conv} / {speaker}] {e.name}: {role_strs}")
        return "\n".join(lines)

    def _run_agent(
        self,
        person_id: str,
        entity_summary: str,
        all_entities: List[EntityNode],
        all_vps: List,
        model_name: str,
        max_iterations: int,
    ) -> tuple:
        """Run the agentic reconciliation loop.

        Returns:
            Tuple of (merged_entities, deduped_vps).
        """
        client = OpenAI()

        # Store VPs on self so _dispatch_tool can access them
        self._all_vps = all_vps

        tool_definitions = [
            # --- Inspection tools (read-only) ---
            {
                "type": "function",
                "name": "get_entity_timeline",
                "description": "Get all roles, states, and associated verb phrase questions for a specific entity by index. Includes question IDs, evidence, and speaker info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_index": {"type": "integer", "description": "0-based entity index from the summary"},
                    },
                    "required": ["entity_index"],
                },
            },
            {
                "type": "function",
                "name": "compare_entities",
                "description": "Compare two entities by index to assess if they refer to the same real-world entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_a": {"type": "integer"},
                        "index_b": {"type": "integer"},
                    },
                    "required": ["index_a", "index_b"],
                },
            },
            {
                "type": "function",
                "name": "detect_contradictions",
                "description": "Detect contradictory states between two entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_a": {"type": "integer"},
                        "index_b": {"type": "integer"},
                    },
                    "required": ["index_a", "index_b"],
                },
            },
            {
                "type": "function",
                "name": "get_unanswered_questions",
                "description": "List all unanswered (None) questions in VPs where this entity participates. Shows VP context (what IS answered), and cross-references from other VPs with the same phrase. Use this to decide what answers to fill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_index": {"type": "integer", "description": "0-based entity index"},
                    },
                    "required": ["entity_index"],
                },
            },
            {
                "type": "function",
                "name": "get_duplicate_verb_phrases",
                "description": "List groups of VPs with the same phrase text, showing each VP's questions and answers side-by-side. Use this to decide which VPs to merge.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "get_reconciliation_summary",
                "description": "Get a summary of the current reconciliation state: entity count, unanswered question count, duplicate VP count.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            # --- Action tools (write) ---
            {
                "type": "function",
                "name": "resolve_question",
                "description": "Set the answer for a specific question. Use after inspecting unanswered questions and deciding what the answer should be.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vp_id": {"type": "string", "description": "The verb phrase ID containing the question"},
                        "question_id": {"type": "string", "description": "The question ID to resolve"},
                        "new_answers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The new answers (entity IDs or TEXT:value strings)",
                        },
                    },
                    "required": ["vp_id", "question_id", "new_answers"],
                },
            },
            {
                "type": "function",
                "name": "merge_verb_phrases",
                "description": "Merge two verb phrases that represent the same event. Questions from discarded VP are added to the kept VP. Use after inspecting duplicate VPs and deciding they are the same event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keep_vp_id": {"type": "string", "description": "VP ID to keep"},
                        "discard_vp_id": {"type": "string", "description": "VP ID to discard (merged into keep)"},
                    },
                    "required": ["keep_vp_id", "discard_vp_id"],
                },
            },
            {
                "type": "function",
                "name": "finish_reconciliation",
                "description": "Complete reconciliation with a list of entity merge pairs. Call this as the final step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "merge_pairs": {
                            "type": "array",
                            "description": "List of [keep_index, discard_index] pairs to merge",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                    },
                    "required": ["merge_pairs"],
                },
            },
        ]

        messages = [
            {"role": "system", "content": _LAYER3_SYSTEM},
            {
                "role": "user",
                "content": _LAYER3_USER.format(
                    person_id=person_id,
                    entity_summary=entity_summary,
                ),
            },
        ]

        merge_pairs: List[List[int]] = []
        finished = False

        for _ in range(max_iterations):
            response = client.responses.create(
                model=model_name,
                input=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=0,
            )

            tool_calls_in_turn = []
            for item in response.output:
                if item.type == "function_call":
                    tool_calls_in_turn.append(item)

            if not tool_calls_in_turn:
                break

            messages.extend(response.output)

            tool_results = []
            for tc in tool_calls_in_turn:
                args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                result = self._dispatch_tool(tc.name, args, all_entities)

                if tc.name == "finish_reconciliation":
                    merge_pairs = args.get("merge_pairs", [])
                    finished = True

                tool_results.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps(result),
                })

            messages.extend(tool_results)

            if finished:
                break

        # Apply merges to entities and remap VP answers
        merged_entities = self._apply_merges(all_entities, merge_pairs, self._all_vps)

        return merged_entities, self._all_vps

    def _dispatch_tool(
        self,
        name: str,
        args: Dict[str, Any],
        entities: List[EntityNode],
    ) -> Any:
        """Execute a reconciliation tool call."""

        # --- Inspection tools (read-only) ---

        if name == "get_entity_timeline":
            idx = args["entity_index"]
            if not (0 <= idx < len(entities)):
                return {"error": f"Index {idx} out of range"}
            e = entities[idx]
            vp_context = []
            for vp in self._all_vps:
                entity_in_vp = any(e.id in q.answers for q in vp.questions)
                if not entity_in_vp:
                    continue
                for q in vp.questions:
                    entry = {
                        "verb_phrase": vp.phrase,
                        "verb_phrase_id": vp.id,
                        "question_id": q.id,
                        "question": q.text,
                        "answers": q.answers,
                        "speaker_id": q.speaker_id,
                        "evidence_turn_ids": q.evidence_turn_ids,
                    }
                    if q.answers == ["None"]:
                        entry["note"] = "unanswered"
                    vp_context.append(entry)
            return {
                "name": e.name,
                "id": e.id,
                "speaker_id": e.speaker_id,
                "conversation_id": e.conversation_id,
                "roles": [
                    {"role": r.role, "states": r.states, "speaker_id": r.speaker_id}
                    for r in e.roles
                ],
                "verb_phrase_questions": vp_context[:20],
            }

        elif name == "compare_entities":
            idx_a, idx_b = args["index_a"], args["index_b"]
            if not (0 <= idx_a < len(entities) and 0 <= idx_b < len(entities)):
                return {"error": "Index out of range"}
            a, b = entities[idx_a], entities[idx_b]
            name_sim = a.name.lower() == b.name.lower() or (
                a.name.lower() in b.name.lower() or b.name.lower() in a.name.lower()
            )
            roles_a = {r.role.lower() for r in a.roles}
            roles_b = {r.role.lower() for r in b.roles}
            shared_roles = roles_a & roles_b
            return {
                "entity_a": f"{a.name} (id={a.id})",
                "entity_b": f"{b.name} (id={b.id})",
                "name_similar": name_sim,
                "shared_roles": list(shared_roles),
                "same_speaker": a.speaker_id == b.speaker_id,
                "same_conversation": a.conversation_id == b.conversation_id,
            }

        elif name == "detect_contradictions":
            idx_a, idx_b = args["index_a"], args["index_b"]
            if not (0 <= idx_a < len(entities) and 0 <= idx_b < len(entities)):
                return {"error": "Index out of range"}
            a, b = entities[idx_a], entities[idx_b]
            states_a = {r.role: set(r.states) for r in a.roles}
            states_b = {r.role: set(r.states) for r in b.roles}
            contradictions = []
            for role in states_a.keys() & states_b.keys():
                if states_a[role] != states_b[role]:
                    contradictions.append({
                        "role": role,
                        "states_a": list(states_a[role]),
                        "states_b": list(states_b[role]),
                    })
            return {"contradictions": contradictions}

        elif name == "get_unanswered_questions":
            idx = args["entity_index"]
            if not (0 <= idx < len(entities)):
                return {"error": f"Index {idx} out of range"}
            e = entities[idx]
            unanswered = []
            for vp in self._all_vps:
                entity_in_vp = any(e.id in q.answers for q in vp.questions)
                if not entity_in_vp:
                    continue
                for q in vp.questions:
                    if q.answers != ["None"]:
                        continue
                    # Context: what IS answered in this VP
                    other_answers = [
                        {"question": oq.text, "answers": oq.answers}
                        for oq in vp.questions if oq.answers != ["None"]
                    ]
                    # Cross-reference: same phrase in other VPs
                    cross_refs = []
                    for other_vp in self._all_vps:
                        if other_vp is vp or other_vp.phrase != vp.phrase:
                            continue
                        for oq in other_vp.questions:
                            if oq.text == q.text:
                                cross_refs.append({
                                    "vp_id": other_vp.id,
                                    "answers": oq.answers,
                                })
                    unanswered.append({
                        "vp_id": vp.id,
                        "vp_phrase": vp.phrase,
                        "question_id": q.id,
                        "question_text": q.text,
                        "other_answers_in_vp": other_answers,
                        "cross_references": cross_refs,
                    })
            return {"entity": e.name, "unanswered": unanswered}

        elif name == "get_duplicate_verb_phrases":
            from collections import defaultdict
            phrase_groups: Dict[str, list] = defaultdict(list)
            for vp in self._all_vps:
                phrase_groups[vp.phrase].append(vp)
            duplicate_groups = []
            for phrase, vps in phrase_groups.items():
                if len(vps) < 2:
                    continue
                group = {
                    "phrase": phrase,
                    "verb_phrases": [
                        {
                            "vp_id": vp.id,
                            "questions": [
                                {
                                    "question_id": q.id,
                                    "text": q.text,
                                    "answers": q.answers,
                                    "speaker_id": q.speaker_id,
                                }
                                for q in vp.questions
                            ],
                        }
                        for vp in vps
                    ],
                }
                duplicate_groups.append(group)
            return {"duplicate_groups": duplicate_groups}

        elif name == "get_reconciliation_summary":
            none_count = 0
            total_q = 0
            for vp in self._all_vps:
                for q in vp.questions:
                    total_q += 1
                    if q.answers == ["None"]:
                        none_count += 1
            phrase_counts: Dict[str, int] = {}
            for vp in self._all_vps:
                phrase_counts[vp.phrase] = phrase_counts.get(vp.phrase, 0) + 1
            dup_phrases = {p: c for p, c in phrase_counts.items() if c > 1}
            return {
                "total_entities": len(entities),
                "total_verb_phrases": len(self._all_vps),
                "total_questions": total_q,
                "unanswered_questions": none_count,
                "duplicate_verb_phrases": dup_phrases,
            }

        # --- Action tools (write) ---

        elif name == "resolve_question":
            vp_id = args["vp_id"]
            question_id = args["question_id"]
            new_answers = args["new_answers"]
            for vp in self._all_vps:
                if vp.id != vp_id:
                    continue
                for q in vp.questions:
                    if q.id != question_id:
                        continue
                    old_answers = q.answers[:]
                    q.answers = new_answers
                    return {
                        "status": "resolved",
                        "question": q.text,
                        "old_answers": old_answers,
                        "new_answers": new_answers,
                    }
                return {"error": f"Question {question_id!r} not found in VP {vp_id!r}"}
            return {"error": f"VP {vp_id!r} not found"}

        elif name == "merge_verb_phrases":
            keep_vp_id = args["keep_vp_id"]
            discard_vp_id = args["discard_vp_id"]
            if keep_vp_id == discard_vp_id:
                return {"error": f"Cannot merge VP with itself (both IDs are {keep_vp_id!r})"}
            keep_vp = next((vp for vp in self._all_vps if vp.id == keep_vp_id), None)
            discard_vp = next((vp for vp in self._all_vps if vp.id == discard_vp_id), None)
            if not keep_vp:
                return {"error": f"VP {keep_vp_id!r} not found"}
            if not discard_vp:
                return {"error": f"VP {discard_vp_id!r} not found"}
            # Merge questions from discard into keep
            existing_q_texts = {q.text: q for q in keep_vp.questions}
            questions_added = 0
            none_filled = 0
            for q in discard_vp.questions:
                if q.text in existing_q_texts:
                    existing_q = existing_q_texts[q.text]
                    if existing_q.answers == ["None"] and q.answers != ["None"]:
                        existing_q.answers = q.answers
                        existing_q.speaker_id = q.speaker_id or existing_q.speaker_id
                        none_filled += 1
                else:
                    keep_vp.questions.append(q)
                    existing_q_texts[q.text] = q
                    questions_added += 1
            # Remove only the discard VP by identity, not by ID
            self._all_vps = [vp for vp in self._all_vps if vp is not discard_vp]
            return {
                "status": "merged",
                "kept": keep_vp_id,
                "discarded": discard_vp_id,
                "questions_added": questions_added,
                "none_filled": none_filled,
            }

        elif name == "finish_reconciliation":
            return {"status": "reconciliation_complete"}

        return {"error": f"Unknown tool: {name}"}

    def _apply_merges(
        self,
        entities: List[EntityNode],
        merge_pairs: List[List[int]],
        all_vps: list,
    ) -> List[EntityNode]:
        """Apply agent-determined merges.

        For each (keep_idx, discard_idx) pair:
        1. Absorb discard entity's roles into keep entity
        2. Remap VP question answers from discard ID → keep ID
        3. Build entity_id_remap for spacetime edge remapping
        """
        self._entity_id_remap: Dict[str, str] = {}
        discard_set: set = set()

        for pair in merge_pairs:
            if len(pair) != 2:
                continue
            keep_idx, discard_idx = pair
            if not (0 <= keep_idx < len(entities) and 0 <= discard_idx < len(entities)):
                continue
            if keep_idx == discard_idx or discard_idx in discard_set:
                continue

            keep_entity = entities[keep_idx]
            discard_entity = entities[discard_idx]

            # Absorb roles (skip exact duplicates)
            existing_role_keys = {
                (r.role, frozenset(r.states)) for r in keep_entity.roles
            }
            for role in discard_entity.roles:
                if (role.role, frozenset(role.states)) not in existing_role_keys:
                    keep_entity.roles.append(role)

            # Remap VP question answers
            for vp in all_vps:
                for q in vp.questions:
                    q.answers = [
                        keep_entity.id if ans == discard_entity.id else ans
                        for ans in q.answers
                    ]

            # Track for spacetime edge remapping
            self._entity_id_remap[discard_entity.id] = keep_entity.id
            discard_set.add(discard_idx)

        return [e for i, e in enumerate(entities) if i not in discard_set]
