"""
PersonalMemoryQAAgent — question routing and speaker-verification QA.

Architecture:
  1. Route question to the correct GSW scope:
       - If question mentions a known conversation partner → ConversationMemory.gsw
       - Otherwise → PersonMemory.global_gsw
  2. Run AgenticAnsweringAgent (reused from qa/agentic_agent.py) with
     in-memory GSW tools over the routed GSW.
  3. Speaker verification (adversarial filter): after retrieval, check
     whether the speaker_id of retrieved roles matches the question subject.
     If mismatch → set abstain=True in the return dict.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from ..memory.models import EntityNode, GSWStructure
from .models import ConversationMemory, PersonMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the QA agent
# ---------------------------------------------------------------------------

_QA_SYSTEM = """\
You answer questions about people using a structured personal memory called a \
Generative Semantic Workspace (GSW).

## Memory structure
The GSW contains **entities** (people, places, objects). Each entity has:
- **roles**: what the entity does or is (e.g. "researcher", "friend of Maria")
- **states**: how the entity changes within that role (e.g. "submitted paper", "moved to NYC")
- **speaker_id**: which conversation participant asserted this role
- **evidence_turn_ids**: dialogue turn IDs proving this role (e.g. "D1:3", "D4:7")

Entities can also be linked to **space nodes** (locations like "New York", "LGBTQ center") \
and **time nodes** (dates/times like "May 7 2023", "last summer"). These are returned by \
get_entity_context as `linked_locations` and `linked_times`.

IMPORTANT: A single entity can have roles from DIFFERENT speakers. Always check \
role.speaker_id to know who provided the information.

The memory may span multiple conversations and sessions.

## Tool usage strategy
1. Decompose the question into atomic sub-questions.
2. Use search_entities(query) to find relevant entities — this returns basic info \
(entity_id, name, speaker_id, role_count) but NOT full roles.
3. ALWAYS call get_entity_context(entity_id) next to see detailed roles, states, \
speaker_id, and evidence_turn_ids. This is where the real information is.
4. If search misses, rephrase with synonyms or try get_all_entities() as a fallback \
to scan all entity names.
5. For multi-hop questions, chain lookups: find entity A → read its roles → identify \
linked entity B → get_entity_context(B).
6. For "when" questions, check `linked_times` in get_entity_context results. \
For "where" questions, check `linked_locations`. Also check role states which may \
contain temporal/spatial info inline.

## Relationship navigation
- Relationships may be stored in one direction only (e.g. "son of" but not "parent of").
- If you don't find a direct match, search for the related entity and look at ALL its \
roles for the inverse relationship.
- Try multiple related search terms when a single query misses.

## Speaker attribution
- When the question targets a specific person (e.g. "What does Caroline do?"), verify \
that the role's speaker_id matches that person.
- If evidence comes from a different speaker than expected, still report the speaker_id \
accurately — the system handles abstention separately.

## Evidence chain
- Your final evidence_turn_ids MUST come from the role's evidence_turn_ids returned by \
get_entity_context. Do not invent turn IDs.
- Collect evidence_turn_ids from every role you used to build the answer.

## Output format
When you have the answer, respond with ONLY a JSON object:
{
    "answer": "Concise answer, no filler",
    "reasoning": "Step-by-step explanation of how you found it",
    "speaker_id": "Person this answer is about (null if unclear)",
    "evidence_turn_ids": ["D1:3", "D4:7"]
}

Example:
Question: "What job does Caroline have?"
{
    "answer": "Counselor",
    "reasoning": "Searched for Caroline → entity e1. get_entity_context showed role 'counselor' with states ['pursuing counseling certification'], speaker_id Caroline, evidence D1:9 and D1:11.",
    "speaker_id": "Caroline",
    "evidence_turn_ids": ["D1:9", "D1:11"]
}

Do NOT include phrases like "The answer is" or "Based on my search" in the answer field. \
If you cannot find sufficient evidence, set answer to "I don't know" and explain why in reasoning.
"""

_QA_USER = "Question: {question}"


# ---------------------------------------------------------------------------
# In-memory GSW search tools
# ---------------------------------------------------------------------------


class _InMemoryGSWTools:
    """Lightweight in-memory GSW search tools for the QA agent."""

    def __init__(self, gsw: GSWStructure):
        self._gsw = gsw

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name or role keyword (case-insensitive substring)."""
        q = query.lower()
        results = []
        for entity in self._gsw.entity_nodes:
            score = 0
            if q in entity.name.lower():
                score += 2
            for role in entity.roles:
                if q in role.role.lower():
                    score += 1
                for state in role.states:
                    if q in state.lower():
                        score += 1
            if score > 0:
                results.append((score, entity))
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "entity_id": e.id,
                "name": e.name,
                "speaker_id": e.speaker_id,
                "conversation_id": e.conversation_id,
                "role_count": len(e.roles),
            }
            for _, e in results[:limit]
        ]

    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """Get full role/state context for an entity by ID, including spacetime."""
        for entity in self._gsw.entity_nodes:
            if entity.id == entity_id:
                # Linked space nodes
                linked_locations = []
                for eid, sid in self._gsw.space_edges:
                    if eid == entity_id:
                        for sn in self._gsw.space_nodes:
                            if sn.id == sid:
                                linked_locations.append(sn.current_name or sn.id)
                # Linked time nodes
                linked_times = []
                for eid, tid in self._gsw.time_edges:
                    if eid == entity_id:
                        for tn in self._gsw.time_nodes:
                            if tn.id == tid:
                                linked_times.append(tn.current_name or tn.id)

                return {
                    "name": entity.name,
                    "speaker_id": entity.speaker_id,
                    "conversation_id": entity.conversation_id,
                    "roles": [
                        {
                            "role": r.role,
                            "states": r.states,
                            "speaker_id": r.speaker_id,
                            "evidence_turn_ids": r.evidence_turn_ids,
                        }
                        for r in entity.roles
                    ],
                    "linked_locations": linked_locations,
                    "linked_times": linked_times,
                }
        return {"error": f"Entity {entity_id!r} not found"}

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Return a summary of all entities in the GSW."""
        return [
            {
                "entity_id": e.id,
                "name": e.name,
                "speaker_id": e.speaker_id,
                "role_count": len(e.roles),
            }
            for e in self._gsw.entity_nodes
        ]

    def get_entity_spacetime(self, entity_id: str) -> Dict[str, Any]:
        """Get space and time nodes linked to an entity."""
        entity = next((e for e in self._gsw.entity_nodes if e.id == entity_id), None)
        if entity is None:
            return {"error": f"Entity {entity_id!r} not found"}

        locations = []
        for eid, sid in self._gsw.space_edges:
            if eid == entity_id:
                for sn in self._gsw.space_nodes:
                    if sn.id == sid:
                        locations.append({"id": sn.id, "name": sn.current_name or sn.id})
        times = []
        for eid, tid in self._gsw.time_edges:
            if eid == entity_id:
                for tn in self._gsw.time_nodes:
                    if tn.id == tid:
                        times.append({"id": tn.id, "name": tn.current_name or tn.id})

        return {
            "entity_name": entity.name,
            "locations": locations,
            "times": times,
        }

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "search_entities",
                "description": (
                    "Search entities by substring matching on names, roles, and states "
                    "(case-insensitive). Returns basic info only: entity_id, name, "
                    "speaker_id, role_count. Call get_entity_context for full details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term (name, role keyword, or state keyword)"},
                        "limit": {"type": "integer", "default": 10, "description": "Max results to return"},
                    },
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "name": "get_entity_context",
                "description": (
                    "Get the FULL detailed view of an entity: all roles with their states, "
                    "speaker_id (who asserted it), evidence_turn_ids (dialogue turns proving it), "
                    "plus linked_locations and linked_times from the spacetime graph. "
                    "Always call this after search_entities."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string", "description": "Entity ID from search results"},
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "type": "function",
                "name": "get_entity_spacetime",
                "description": (
                    "Get locations and times linked to an entity from the spacetime graph. "
                    "Use for 'when' and 'where' questions. Returns lists of location names "
                    "and time descriptions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string", "description": "Entity ID from search results"},
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "type": "function",
                "name": "get_all_entities",
                "description": (
                    "List ALL entities in memory with basic info (entity_id, name, "
                    "speaker_id, role_count). Use as a fallback when search_entities "
                    "misses — scan the full list to find what you need."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def dispatch(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "search_entities":
            return self.search_entities(**args)
        elif name == "get_entity_context":
            return self.get_entity_context(**args)
        elif name == "get_entity_spacetime":
            return self.get_entity_spacetime(**args)
        elif name == "get_all_entities":
            return self.get_all_entities()
        return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# PersonalMemoryQAAgent
# ---------------------------------------------------------------------------


class PersonalMemoryQAAgent:
    """Answer questions over a PersonMemory with speaker-verification.

    Args:
        model_name: OpenAI model for agentic QA.
        max_iterations: Max tool-call rounds per question.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_iterations: int = 10,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self._client = OpenAI()

    def answer(
        self,
        question: str,
        person_memory: PersonMemory,
        target_conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a question using the person's memory.

        Args:
            question: The question text.
            person_memory: The person's full memory.
            target_conversation_id: If set, restrict search to this conversation.

        Returns:
            Dict with keys: answer, reasoning, speaker_id, evidence_turn_ids,
                            abstain (bool), confidence (str), conversation_id.
        """
        gsw = self._route_question(question, person_memory, target_conversation_id)

        if gsw is None or not gsw.entity_nodes:
            return {
                "answer": "Insufficient memory to answer.",
                "reasoning": "No relevant GSW found for this question.",
                "speaker_id": None,
                "evidence_turn_ids": [],
                "abstain": False,
                "confidence": "low",
                "conversation_id": target_conversation_id,
            }

        raw, trace = self._run_agentic_qa(question, gsw)

        # Speaker verification for adversarial questions
        abstain = self._check_speaker_mismatch(question, raw, person_memory)

        return {
            "answer": raw.get("answer", ""),
            "reasoning": raw.get("reasoning", ""),
            "speaker_id": raw.get("speaker_id"),
            "evidence_turn_ids": raw.get("evidence_turn_ids", []),
            "abstain": abstain,
            "confidence": "high" if not abstain else "low",
            "conversation_id": target_conversation_id,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_question(
        self,
        question: str,
        person_memory: PersonMemory,
        target_conversation_id: Optional[str],
    ) -> Optional[GSWStructure]:
        """Choose which GSW to query."""
        # Explicit override
        if target_conversation_id and target_conversation_id in person_memory.conversation_memories:
            return person_memory.conversation_memories[target_conversation_id].gsw

        # Check if question names a known conversation partner
        q_lower = question.lower()
        for conv_mem in person_memory.conversation_memories.values():
            for speaker in [conv_mem.speaker_a, conv_mem.speaker_b]:
                if speaker.lower() in q_lower:
                    return conv_mem.gsw

        # Default: global GSW (Layer 3)
        if person_memory.global_gsw is not None:
            return person_memory.global_gsw

        # Fall back to first conversation GSW
        if person_memory.conversation_memories:
            return next(iter(person_memory.conversation_memories.values())).gsw

        return None

    # ------------------------------------------------------------------
    # Agentic QA loop
    # ------------------------------------------------------------------

    def _run_agentic_qa(
        self, question: str, gsw: GSWStructure,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run the agentic tool-call loop.

        Returns:
            (answer_dict, trace) where trace is a list of per-iteration dicts
            with keys ``iteration``, ``agent_text``, and ``tool_calls``.
        """
        tools = _InMemoryGSWTools(gsw)
        tool_defs = tools.tool_definitions()
        trace: List[Dict[str, Any]] = []

        messages: List[Any] = [
            {"role": "user", "content": _QA_USER.format(question=question)},
        ]

        for iteration in range(self.max_iterations):
            response = self._client.responses.create(
                model=self.model_name,
                input=messages,
                tools=tool_defs,
                instructions=_QA_SYSTEM,
                temperature=0,
            )

            # Append assistant output to conversation
            messages += response.output

            # Extract agent text from this turn
            agent_text_parts: List[str] = []
            for item in response.output:
                item_type = getattr(item, "type", None)
                if item_type == "message":
                    for part in getattr(item, "content", []):
                        if getattr(part, "type", None) == "output_text":
                            agent_text_parts.append(part.text)

            trace_entry: Dict[str, Any] = {
                "iteration": iteration + 1,
                "agent_text": "\n".join(agent_text_parts),
                "tool_calls": [],
            }

            # Collect function calls
            function_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if function_calls:
                for fc in function_calls:
                    name = getattr(fc, "name", "")
                    raw_args = getattr(fc, "arguments", "{}")
                    call_id = getattr(fc, "call_id", None)
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}
                    result = tools.dispatch(name, args)
                    trace_entry["tool_calls"].append({
                        "name": name,
                        "args": args,
                        "result": result,
                    })
                    messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    })
                trace.append(trace_entry)
                continue

            # No more tool calls → parse final answer
            trace.append(trace_entry)
            content = getattr(response, "output_text", "") or ""
            if not content:
                try:
                    msg_items = [
                        it for it in response.output
                        if getattr(it, "type", None) == "message"
                    ]
                    if msg_items:
                        content = getattr(msg_items[-1], "content", "") or ""
                except Exception:
                    content = ""

            return self._parse_answer(content), trace

        return {"answer": "Unable to answer within iteration limit.", "reasoning": ""}, trace

    @staticmethod
    def _parse_answer(content: str) -> Dict[str, Any]:
        try:
            j_start = content.find("{")
            j_end = content.rfind("}") + 1
            if j_start != -1 and j_end > j_start:
                return json.loads(content[j_start:j_end])
        except Exception:
            pass
        return {"answer": content, "reasoning": ""}

    # ------------------------------------------------------------------
    # Speaker verification
    # ------------------------------------------------------------------

    def _check_speaker_mismatch(
        self,
        question: str,
        raw_answer: Dict[str, Any],
        person_memory: PersonMemory,
    ) -> bool:
        """Return True if retrieved answer is attributed to the wrong speaker.

        Adversarial questions ask about speaker A but the correct answer is
        "I don't know / not applicable" because the information belongs to B.
        We detect this by checking: does the question mention speaker A's name,
        but the retrieved evidence has speaker_id == speaker B?
        """
        retrieved_speaker = raw_answer.get("speaker_id")
        if not retrieved_speaker:
            return False

        q_lower = question.lower()
        for conv_mem in person_memory.conversation_memories.values():
            speaker_a = conv_mem.speaker_a
            speaker_b = conv_mem.speaker_b
            # If question targets speaker_a but answer is from speaker_b (or vice versa)
            if speaker_a.lower() in q_lower and retrieved_speaker == speaker_b:
                logger.debug(
                    "Speaker mismatch: question about %s but answer from %s",
                    speaker_a,
                    retrieved_speaker,
                )
                return True
            if speaker_b.lower() in q_lower and retrieved_speaker == speaker_a:
                logger.debug(
                    "Speaker mismatch: question about %s but answer from %s",
                    speaker_b,
                    retrieved_speaker,
                )
                return True

        return False
