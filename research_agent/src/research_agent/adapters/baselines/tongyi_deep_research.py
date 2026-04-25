"""Tongyi DeepResearch adapter.

Alibaba's ``Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`` (MoE, ~3.3B active)
ships a ReAct-style agent with tools {Search, Visit, Python, Scholar,
FileParser}. For the FRAMES offline Wikipedia setup we only need
Search + Visit (alias for read). Python/Scholar/FileParser are
no-ops / disabled in this adapter since they assume web access and
sandbox evaluation.

The model is run **as-shipped** (single row in the grid, no swap). The
adapter exists mainly so the harness can call it uniformly; the system
prompt matches Tongyi's ReAct mode.
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


SYSTEM_PROMPT = """You are a Deep Research agent. Answer the user's question \
by retrieving evidence from the corpus. You have two tools available:

- Search(query, top_k): find relevant document chunks.
- Visit(chunk_id): read the full article body for a chunk.

The Python, Scholar, and FileParser tools are unavailable in this run.

Think step by step. Gather evidence, then answer concisely. When you are \
ready, output the answer as plain text without wrapping tags."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Retrieve candidate chunks from the article corpus. `query` can "
                "be a single string or an array of parallel queries (batched)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Fetch the full article text for a chunk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
        },
    },
]


@register_adapter
class TongyiDeepResearchAdapter(Adapter):
    """Tongyi DeepResearch 30B-A3B, as-shipped, with Search/Visit tools only."""

    system_id = "tongyi_deep_research"
    display_name = "Tongyi DeepResearch 30B-A3B (as-shipped)"
    description = (
        "Runs the shipped Tongyi DeepResearch MoE with Search + Visit tools over "
        "our BM25 corpus. Python/Scholar/FileParser tools are disabled for the "
        "offline FRAMES setup."
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

    def _search(self, *, query: Any, top_k: int = 5, **_extra: Any) -> dict[str, Any]:
        # Tongyi's native format can emit `query` as either a single string or
        # a list of parallel queries (batch mode). Run each query and merge
        # results, deduplicating by chunk_id and keeping the best score.
        queries = query if isinstance(query, list) else [query]
        queries = [str(q) for q in queries if q]
        merged: dict[str, Any] = {}
        for q in queries:
            for h in self.retriever.search(q, top_k=top_k):
                cid = h.chunk.chunk_id
                prev = merged.get(cid)
                if prev is None or h.score > prev["score"]:
                    merged[cid] = {
                        "chunk_id": cid,
                        "title": h.chunk.title,
                        "score": round(h.score, 3),
                        "text": h.chunk.text[:600],
                    }
        # Rank by score descending; cap to top_k overall.
        results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        return {"results": results}

    def _visit(self, *, chunk_id: Any = None, url: Any = None, **_extra: Any) -> dict[str, Any]:
        # Tongyi's shipped `visit` may pass `url` (web mode) rather than
        # `chunk_id`. Accept either; url is treated as the chunk id for our
        # offline Wikipedia corpus.
        target = chunk_id if chunk_id is not None else url
        if target is None:
            return {"error": "visit requires chunk_id or url argument"}
        # Accept list forms too (some Tongyi outputs emit `url: [...]`).
        if isinstance(target, list):
            target = target[0] if target else None
            if target is None:
                return {"error": "visit received empty list"}
        target = str(target)
        chunk = self.corpus.get_chunk(target)
        if not chunk:
            return {"error": f"chunk_id {target!r} not found"}
        full = self.corpus.article_text(chunk.title)
        return {"title": chunk.title, "article_text": full[:12000]}

    def _dispatch(self, name: str, args_json: str) -> dict[str, Any]:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}
        # Tongyi's shipped prompt emits tool names in lowercase (`search`,
        # `visit`) while vLLM's hermes parser sometimes capitalizes them.
        # Normalise so both spellings work.
        name_key = (name or "").strip().lower()
        if name_key == "search":
            return self._search(**args)
        if name_key == "visit":
            return self._visit(**args)
        return {"error": f"unknown or disabled tool {name!r}"}

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
                result = self._dispatch(tc["name"], tc["arguments"])
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

        # Force-synthesis pass: if we ran out of turns without the model
        # emitting a text-only final answer, give it one more shot with
        # tools disabled so it MUST produce plain text. Tongyi's native
        # policy keeps calling tools even when confident, which otherwise
        # strands the answer in the reasoning buffer.
        if not traj.final_answer and stopped_reason in {"max_turns", "finished"}:
            try:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Based on the evidence gathered above, produce the "
                            "final answer now as plain text. Do NOT call any more "
                            "tools. Keep it concise."
                        ),
                    }
                )
                forced = self.llm.chat(
                    messages,
                    tools=None,
                    max_tokens=self.ctx.max_completion_tokens,
                )
                traj.prompt_tokens += forced.prompt_tokens
                traj.completion_tokens += forced.completion_tokens
                if forced.reasoning_content:
                    reasoning_chunks.append(f"[forced] {forced.reasoning_content}")
                final_text = (forced.text or "").strip()
                if final_text:
                    traj.final_answer = final_text
                    traj.reasoning = final_text
                    messages.append({"role": "assistant", "content": final_text})
                    stopped_reason = "forced_synthesis"
            except Exception as exc:  # noqa: BLE001
                traj.extra["forced_synthesis_error"] = str(exc)

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
