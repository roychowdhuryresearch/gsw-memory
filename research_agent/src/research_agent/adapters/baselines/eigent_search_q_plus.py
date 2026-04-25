"""EigentSearch Q+ adapter.

Ports CAMEL-AI's ``eigent_search`` QueryProcessingToolkit
(``third_party/eigent_search/eigent_search/toolkit/query_toolkit.py``)
onto our BM25 corpus. The Q+ toolkit's thesis: a base RAG agent gets
better when you add **structured query-processing tools** that force
planning, progress tracking, and evidence extraction.

Four tools are exposed (all via OpenAI-style tool-calls):

- ``plan_next_searches(search_queries)`` — propose new queries to add
  to the frontier; already-explored queries are silently filtered out.
- ``select_query_and_search(query, top_k)`` — execute one retrieval,
  mark the query as explored.
- ``extract_relevant_details(notes)`` — agent-authored extracted facts
  accumulated across turns.
- ``analyze_search_progress(summary, gaps)`` — reflection on what
  remains to be found.

Internal state: ``frontier`` (candidate queries) + ``explored`` +
``extracted_notes`` accumulators. When the agent is done it stops
calling tools; we take the final text as the answer.

Three rows in the grid: GPT-5 + Q+, gpt-oss-20B + Q+, Qwen2.5-7B + Q+.
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


SYSTEM_PROMPT = """You are a deep-research agent. Answer the user's question \
using the following structured query-processing tools:

- plan_next_searches(search_queries): propose a list of new query strings \
to add to your research frontier. Queries you have already explored will \
be silently dropped from the list.
- select_query_and_search(query, top_k): execute one retrieval on a query. \
Use this after you've planned; the system will mark the query as explored.
- extract_relevant_details(notes): record key facts you've gathered from \
search results. Call this after each useful search to consolidate notes.
- analyze_search_progress(summary, gaps): reflect on what you've learned \
vs. what's still missing. Use this to decide whether to plan more searches \
or give the final answer.

Loop structure:
1. plan_next_searches to seed the frontier.
2. Repeatedly: select_query_and_search → extract_relevant_details.
3. Periodically: analyze_search_progress; re-plan if gaps remain.
4. When confident, stop calling tools and emit the final answer as plain text.

Keep the final answer concise (one clause if possible)."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_next_searches",
            "description": "Propose new search queries to add to the research frontier. "
            "Already-explored queries are filtered out automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate queries to enqueue.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why these queries advance progress.",
                    },
                },
                "required": ["search_queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_query_and_search",
            "description": "Execute one retrieval. The query is marked as explored afterward.",
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
            "name": "extract_relevant_details",
            "description": "Record key facts gathered from the search results so they survive "
            "context pruning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One note per fact, phrased concisely.",
                    },
                },
                "required": ["notes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_search_progress",
            "description": "Reflect on progress. Summarize what you know and what's missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Aspects of the question still unanswered.",
                    },
                },
                "required": ["summary", "gaps"],
            },
        },
    },
]


@register_adapter
class EigentSearchQPlusAdapter(Adapter):
    """Q+ query-processing toolkit on BM25 + any chat LLM."""

    system_id = "eigent_search_q_plus"
    display_name = "EigentSearch Q+ (structured query-processing tools)"
    description = (
        "Ports CAMEL-AI eigent_search's QueryProcessingToolkit (plan / select / "
        "extract / analyze) onto our BM25 corpus. Maintains frontier + explored "
        "state across turns."
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
        self._top_k = int(ctx.extra.get("top_k", 5))

    # -- tool implementations ---------------------------------------------

    def _reset_state(self) -> dict[str, Any]:
        return {
            "frontier": [],
            "explored": [],
            "notes": [],
            "progress_summaries": [],
        }

    def _tool_plan(self, state, *, search_queries, rationale: str = "") -> dict[str, Any]:
        explored = set(state["explored"])
        filtered = [q for q in (search_queries or []) if q and q not in explored and q not in state["frontier"]]
        state["frontier"].extend(filtered)
        return {
            "added": filtered,
            "dropped_already_explored": [q for q in (search_queries or []) if q in explored],
            "current_frontier": list(state["frontier"]),
            "rationale": rationale,
        }

    def _tool_select_and_search(self, state, *, query: str, top_k: int = 5) -> dict[str, Any]:
        if query in state["explored"]:
            return {"error": f"query already explored: {query!r}"}
        state["explored"].append(query)
        if query in state["frontier"]:
            state["frontier"].remove(query)
        hits = self.retriever.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "title": h.chunk.title,
                    "text": h.chunk.text[:700],
                }
                for h in hits
            ],
            "frontier_remaining": list(state["frontier"]),
            "explored_count": len(state["explored"]),
        }

    def _tool_extract(self, state, *, notes) -> dict[str, Any]:
        notes = [n for n in (notes or []) if isinstance(n, str) and n.strip()]
        state["notes"].extend(notes)
        return {"added_notes": notes, "total_notes": len(state["notes"])}

    def _tool_analyze(self, state, *, summary: str, gaps) -> dict[str, Any]:
        gaps = [g for g in (gaps or []) if isinstance(g, str)]
        state["progress_summaries"].append({"summary": summary, "gaps": gaps})
        return {
            "summary": summary,
            "gaps": gaps,
            "frontier_size": len(state["frontier"]),
            "explored_size": len(state["explored"]),
            "notes_count": len(state["notes"]),
        }

    def _dispatch(self, name: str, args_json: str, state) -> dict[str, Any]:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}
        try:
            if name == "plan_next_searches":
                return self._tool_plan(state, **args)
            if name == "select_query_and_search":
                return self._tool_select_and_search(state, **args)
            if name == "extract_relevant_details":
                return self._tool_extract(state, **args)
            if name == "analyze_search_progress":
                return self._tool_analyze(state, **args)
            return {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            return {"error": f"bad tool call: {exc}"}

    # -- main loop --------------------------------------------------------

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

        state = self._reset_state()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
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
                result = self._dispatch(tc["name"], tc["arguments"], state)
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
                        "content": json.dumps(result),
                    }
                )
            traj.turns = turn
            if turn == self.ctx.max_turns:
                stopped_reason = "max_turns"

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["final_state"] = {
            "explored_queries": state["explored"],
            "frontier_remaining": state["frontier"],
            "notes": state["notes"],
            "progress_summaries": state["progress_summaries"],
        }
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
