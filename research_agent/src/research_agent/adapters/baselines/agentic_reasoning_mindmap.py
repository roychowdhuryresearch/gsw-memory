"""Agentic Reasoning / Mind-Map Agent adapter.

Mirrors the Agentic Reasoning framework (``2502.04644``, ACL 2025) —
a ReAct-style agent whose hallmark is a dedicated **Mind-Map
sub-agent** that maintains a structured knowledge graph across the
reasoning chain. The original paper runs on DeepSeek-R1 and exposes
three peer tools (Web-Search, Coding, Mind-Map). For our FRAMES
pilot we keep the same tool surface but (a) swap Web-Search for our
BM25 retriever and (b) drop Coding (FRAMES rarely needs arithmetic).

Three tools:
- ``search(query, top_k)`` — BM25 retrieval.
- ``mind_map_update(edges)`` — add (subject, relation, object) triples
  to the persistent mind-map for this question. Triples accumulate
  across turns. The tool returns the current mind-map size.
- ``mind_map_query(focus)`` — read back all triples involving a
  focus entity (substring-match on subject or object). Returns a
  compact listing the agent can reason over.

The mind-map is per-question, not corpus-wide. It acts as the
agent's persistent scratchpad — it survives context compression
because the agent queries it explicitly rather than re-reading raw
chunks. This is the Agentic Reasoning paper's core claim.

Parsing uses OpenAI tool-calls (same style as vanilla_rag_react) so
the adapter works with any chat model that supports tools. No
training needed.
"""

from __future__ import annotations

import json
import time
from typing import Any

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import ToolCall, Trajectory
from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.corpus import load_frames_corpus


SYSTEM_PROMPT = """You are a research assistant that answers multi-hop \
questions by combining external retrieval with a structured mind-map \
of facts you accumulate.

You have three tools:
- search(query, top_k): retrieve chunks from the article corpus.
- mind_map_update(edges): add (subject, relation, object) triples to \
your persistent mind-map. Do this every time retrieval gives you new \
facts. Keep triples concrete — name real entities.
- mind_map_query(focus): look up all triples currently in the mind-map \
that mention an entity (substring match on subject or object). Use \
this as your working memory instead of re-reading chunks.

Workflow: search → read results → mind_map_update → decide if you \
need more searches (consult mind_map_query if useful) → answer.

Answer format (strict): output ONLY the answer itself — the bare \
name/year/number/yes-no. No preamble, no "the answer is", no \
markdown. Use "Yes" or "No" for yes/no questions."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Retrieve chunks from the article corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mind_map_update",
            "description": (
                "Append (subject, relation, object) triples to the mind-map. "
                "Pass an array of 3-element arrays. Example: "
                "[['13 Hours', 'directed_by', 'Michael Bay'], ...]"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "string"},
                        },
                    },
                },
                "required": ["edges"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mind_map_query",
            "description": (
                "List all mind-map triples where subject or object matches the "
                "focus (case-insensitive substring). Returns at most 30 triples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string"},
                },
                "required": ["focus"],
            },
        },
    },
]


@register_adapter
class AgenticReasoningMindMapAdapter(Adapter):
    """Agentic Reasoning with a persistent per-question mind-map."""

    system_id = "agentic_reasoning_mindmap"
    display_name = "Agentic Reasoning (Mind-Map tool)"
    description = (
        "ReAct agent with three tools (search, mind_map_update, mind_map_query). "
        "The mind-map acts as persistent scratchpad across turns."
    )

    def __init__(self, ctx: AdapterContext) -> None:
        super().__init__(ctx)
        self.corpus = ctx.extra.get("corpus") or load_frames_corpus()
        self.retriever = ctx.extra.get("retriever") or BM25Retriever(self.corpus)
        self.llm = LLMClient(
            model=ctx.model_name or ctx.model_id,
            base_url=ctx.base_url or None,
            api_key=ctx.api_key or None,
        )

    # --- tools ---------------------------------------------------------------

    def _tool_search(self, mind_map: list[tuple[str, str, str]], *, query: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, top_k=top_k)
        return {
            "results": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "title": h.chunk.title,
                    "score": round(h.score, 3),
                    "text": h.chunk.text[:800],
                }
                for h in hits
            ],
            "mind_map_size": len(mind_map),
        }

    def _tool_mind_map_update(
        self, mind_map: list[tuple[str, str, str]], *, edges: list[list[str]]
    ) -> dict[str, Any]:
        added = 0
        for row in edges or []:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            s, r, o = str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip()
            if not (s and r and o):
                continue
            triple = (s, r, o)
            if triple not in mind_map:
                mind_map.append(triple)
                added += 1
        return {"added": added, "mind_map_size": len(mind_map)}

    def _tool_mind_map_query(
        self, mind_map: list[tuple[str, str, str]], *, focus: str
    ) -> dict[str, Any]:
        needle = (focus or "").strip().lower()
        if not needle:
            return {"matches": [], "mind_map_size": len(mind_map)}
        matches = []
        for s, r, o in mind_map:
            if needle in s.lower() or needle in o.lower():
                matches.append([s, r, o])
                if len(matches) >= 30:
                    break
        return {"matches": matches, "mind_map_size": len(mind_map)}

    def _dispatch(
        self, mind_map: list[tuple[str, str, str]], name: str, args_json: str
    ) -> dict[str, Any]:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}
        try:
            if name == "search":
                return self._tool_search(mind_map, **args)
            if name == "mind_map_update":
                return self._tool_mind_map_update(mind_map, **args)
            if name == "mind_map_query":
                return self._tool_mind_map_query(mind_map, **args)
            return {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            return {"error": f"bad tool call: {exc}"}

    # --- loop ----------------------------------------------------------------

    def run_question(
        self,
        question: str,
        *,
        question_id: str,
        articles: list[dict[str, Any]] | None = None,
    ) -> Trajectory:
        traj = Trajectory(
            system_id=self.system_id,
            model_id=self.ctx.model_id,
            question_id=question_id,
        )
        traj.extra["gold_articles"] = articles or []

        mind_map: list[tuple[str, str, str]] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        start = time.time()
        stopped_reason = "finished"
        reasoning_chunks: list[str] = []

        for turn in range(1, self.ctx.max_turns + 1):
            try:
                resp = self.llm.chat(
                    messages,
                    tools=TOOLS,
                    max_tokens=self.ctx.max_completion_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                stopped_reason = "llm_error"
                traj.extra["llm_error"] = str(exc)
                break

            traj.prompt_tokens += resp.prompt_tokens
            traj.completion_tokens += resp.completion_tokens
            if resp.reasoning_content:
                reasoning_chunks.append(f"[turn {turn}] {resp.reasoning_content}")

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp.text or None}
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if not resp.tool_calls:
                traj.final_answer = (resp.text or "").strip()
                traj.reasoning = resp.text or ""
                traj.turns = turn
                break

            for tc in resp.tool_calls:
                t0 = time.time()
                result = self._dispatch(mind_map, tc["name"], tc["arguments"])
                result_json = json.dumps(result)
                traj.tool_calls.append(
                    ToolCall(
                        turn=turn,
                        name=tc["name"],
                        args=(json.loads(tc["arguments"]) if tc.get("arguments") else {}),
                        result_preview=result_json[:500],
                        result_full=result_json,
                        duration_s=round(time.time() - t0, 3),
                        error=result.get("error", "") if isinstance(result, dict) else "",
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_json,
                    }
                )

            traj.turns = turn
            if turn == self.ctx.max_turns:
                stopped_reason = "max_turns"

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["mind_map"] = [list(e) for e in mind_map]
        traj.extra["mind_map_size"] = len(mind_map)
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
