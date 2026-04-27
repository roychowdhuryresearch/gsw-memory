"""Vanilla RAG + ReAct adapter.

The minimal agentic-search baseline used as:
1. The **frontier ceiling** (GPT-5 + BGE-RAG + ReAct).
2. The **all-small control** (gpt-oss-20B or Qwen2.5-7B + BM25 + ReAct).

Two tools only:
- ``search(query, top_k=5)``: retrieve chunks from the FRAMES article corpus.
- ``read(chunk_id)``: fetch the full article body for a returned chunk's title.

The agent is told to think step-by-step, emit tool calls until it has
enough evidence, then produce a final answer. No explicit
decomposition, no graph, no scratchpad — this is the plain baseline
every other cell is measured against.
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
from research_agent.retrieval.dense import build_retriever


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's \
question using the tools provided. Before answering, gather evidence by \
calling the tools. Do not fabricate facts not supported by retrieved text.

Tools:
- search(query, top_k): returns relevant text chunks with their chunk_id and title.
- read(chunk_id): returns the full article text for the chunk's title.

When you have enough evidence, produce the final answer. Do not call tools \
after emitting the final answer.

Answer format (strict):
- Output ONLY the answer itself. No preamble, no "the answer is", no \
  explanation, no restating the question, no markdown bolding.
- If the answer is a person, output just the name: "Rob Reiner".
- If it's a year, output just the year: "1994".
- If it's a number, just the number: "12".
- If the question is yes/no, start with "Yes" or "No", then the detail \
  in the fewest words possible.
- If it's a list, comma-separated values with no surrounding prose.
- Never prefix with words like "Answer:", "Final:", "Director:", etc."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Retrieve chunks from the article corpus most relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (natural-language phrase)."},
                    "top_k": {"type": "integer", "description": "Number of chunks to return.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "2",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                },
                "required": ["chunk_id"],
            },
        },
    },
]


@register_adapter
class VanillaRAGReActAdapter(Adapter):
    """Minimal RAG + ReAct agent using BM25 retrieval."""

    system_id = "vanilla_rag_react"
    display_name = "Vanilla RAG + ReAct (BM25)"
    description = "Two-tool ReAct loop over the FRAMES BM25 index. No decomposition, no scratchpad."

    def __init__(self, ctx: AdapterContext) -> None:
        super().__init__(ctx)
        self.corpus = ctx.extra.get("corpus") or load_frames_corpus()
        retriever_override = ctx.extra.get("retriever")
        if retriever_override is not None:
            self.retriever = retriever_override
        else:
            retriever_type = ctx.extra.get("retriever_type", "bm25")
            self.retriever = build_retriever(retriever_type, self.corpus)
        self.llm = LLMClient(
            model=ctx.model_name or ctx.model_id,
            base_url=ctx.base_url or None,
            api_key=ctx.api_key or None,
        )

    # --- tool dispatch -----------------------------------------------------

    def _tool_search(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
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
            ]
        }

    def _tool_read(self, *, chunk_id: str) -> dict[str, Any]:
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not found"}
        full = self.corpus.article_text(chunk.title)
        return {
            "chunk_id": chunk_id,
            "title": chunk.title,
            "article_text": full[:12000],
        }

    def _dispatch(self, name: str, args_json: str) -> dict[str, Any]:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}
        try:
            if name == "search":
                return self._tool_search(**args)
            if name == "read":
                return self._tool_read(**args)
            return {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            return {"error": f"bad tool call: {exc}"}

    # --- agent loop --------------------------------------------------------

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
                # Model decided to answer.
                traj.final_answer = (resp.text or "").strip()
                traj.reasoning = resp.text or ""
                traj.turns = turn
                break

            # Execute each tool call.
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

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
