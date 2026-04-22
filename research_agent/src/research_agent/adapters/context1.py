"""Context-1 + reasoner adapter.

Chroma's Context-1 (`chromadb/context-1`) is a **retrieval subagent**,
not a full-stack reasoner — it runs an observe→reason→act loop using
search/read/prune tools until it has enough chunks, then hands off to a
frontier reasoner that reads the kept context and produces the final
answer. This adapter wraps the two-model pattern so the grid can vary
*which reasoner* sits on top (GPT-5, gpt-oss-20B, Qwen2.5-7B) while the
retrieval subagent stays fixed on Context-1.

Note: the HF model card for ``chromadb/context-1`` doesn't ship an
official tool schema spec. We provide a close-enough schema
(``search`` + ``read`` + ``prune`` + ``done``) that matches the
description in the Chroma tech report. Swap in a tuned schema if we
ever discover a canonical one.
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


RETRIEVAL_SYSTEM_PROMPT = """You are a retrieval subagent. Your job is to \
gather the set of documents that contain the facts needed to answer the \
user's question. You do NOT give the final answer.

Tools:
- search(query, top_k): retrieve candidate chunks.
- read(chunk_id): read the full article text for a chunk.
- prune(chunk_ids): discard chunks you no longer think are useful.
- done(kept_chunk_ids): finish retrieval. Pass the ids of chunks you \
want to keep; the downstream reasoner will see those chunks and nothing else.

Guidelines:
- Retrieve broadly first, then prune aggressively. Keep only chunks that \
directly bear on the question.
- If a chunk looks promising but too abridged, read() to get the full article.
- Stop and call done() once you have enough grounded evidence."""


REASONER_SYSTEM_PROMPT = """You answer multi-hop questions. You have \
access to a curated set of retrieved chunks (already selected by a \
retrieval specialist). Use ONLY the provided chunks to form your answer. \
Be concise — one clause if the question permits."""

REASONER_USER_TEMPLATE = """Question: {question}

Retrieved chunks (from the retrieval specialist):
{chunks_block}

Provide only the final answer, with no extra commentary."""


RETRIEVAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the corpus for chunks relevant to a query.",
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
            "name": "read",
            "description": "Read the full article text for a chunk's article.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prune",
            "description": "Drop chunk_ids from the kept set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Terminate retrieval and pass kept chunks to the reasoner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kept_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Final set of chunk_ids for the reasoner.",
                    },
                },
                "required": ["kept_chunk_ids"],
            },
        },
    },
]


@register_adapter
class Context1PlusReasonerAdapter(Adapter):
    """Two-stage: Context-1 retrieval subagent → external reasoner for answer."""

    system_id = "context1_plus_reasoner"
    display_name = "Context-1 + reasoner"
    description = (
        "Context-1 style retrieval subagent (search/read/prune/done) produces "
        "a curated chunk set, then a separate reasoner LLM answers from those chunks."
    )

    def __init__(self, ctx: AdapterContext) -> None:
        super().__init__(ctx)
        self.corpus = ctx.extra.get("corpus") or load_frames_corpus()
        self.retriever = ctx.extra.get("retriever") or BM25Retriever(self.corpus)

        # Retrieval subagent = Context-1 (or whatever the main model is).
        self.retrieval_llm = LLMClient(
            model=ctx.model_name or ctx.model_id,
            base_url=ctx.base_url or None,
            api_key=ctx.api_key or None,
        )

        # Reasoner is configured separately via ctx.extra. If model_name is
        # provided, treat the reasoner as an independent endpoint — do NOT
        # inherit the retrieval subagent's base_url (the two are usually on
        # different endpoints: Context-1 on local vLLM, reasoner on
        # OpenAI/Bedrock). Only fall back to ctx.* in the fully-degenerate
        # case where no reasoner was specified at all.
        reasoner_cfg = ctx.extra.get("reasoner") or {}
        if reasoner_cfg.get("model_name"):
            self.reasoner_name = reasoner_cfg["model_name"]
            self.reasoner_llm = LLMClient(
                model=self.reasoner_name,
                base_url=reasoner_cfg.get("base_url") or None,
                api_key=reasoner_cfg.get("api_key") or None,
            )
        else:
            self.reasoner_name = ctx.model_id
            self.reasoner_llm = LLMClient(
                model=self.reasoner_name,
                base_url=ctx.base_url or None,
                api_key=ctx.api_key or None,
            )

        self._top_k = int(ctx.extra.get("top_k", 5))
        self._max_retrieval_turns = int(ctx.extra.get("max_retrieval_turns", 12))

    # ----------------------------------------------------------------------
    # Retrieval-phase tool handlers
    # ----------------------------------------------------------------------

    def _tool_search(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, top_k=top_k)
        return {
            "results": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "title": h.chunk.title,
                    "score": round(h.score, 3),
                    "text": h.chunk.text[:600],
                }
                for h in hits
            ]
        }

    def _tool_read(self, *, chunk_id: str) -> dict[str, Any]:
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not found"}
        full = self.corpus.article_text(chunk.title)
        return {"chunk_id": chunk_id, "title": chunk.title, "article_text": full[:12000]}

    def _dispatch(self, name: str, args_json: str, kept: set[str]) -> tuple[dict[str, Any], str | None]:
        """Return (tool_result, done_signal). done_signal is the final kept chunks if done() was called."""
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}, None

        if name == "search":
            return self._tool_search(**args), None
        if name == "read":
            return self._tool_read(**args), None
        if name == "prune":
            ids = set(args.get("chunk_ids") or [])
            kept -= ids
            return {"pruned": sorted(ids), "kept_count": len(kept)}, None
        if name == "done":
            final = list(dict.fromkeys(args.get("kept_chunk_ids") or []))
            return {"kept_count": len(final)}, json.dumps(final)
        return {"error": f"unknown tool {name!r}"}, None

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

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
        traj.extra["reasoner_model"] = self.reasoner_name

        start = time.time()
        # --- Stage 1: retrieval subagent gathers chunks -------------------
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        kept_ids: set[str] = set()
        final_kept: list[str] | None = None
        retrieval_stopped = "finished"
        reasoning_chunks: list[str] = []

        for turn in range(1, self._max_retrieval_turns + 1):
            try:
                resp = self.retrieval_llm.chat(
                    messages,
                    tools=RETRIEVAL_TOOLS,
                    max_tokens=self.ctx.max_completion_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                retrieval_stopped = "llm_error"
                traj.extra["retrieval_error"] = str(exc)
                break

            traj.prompt_tokens += resp.prompt_tokens
            traj.completion_tokens += resp.completion_tokens
            if resp.reasoning_content:
                reasoning_chunks.append(f"[retrieval turn {turn}] {resp.reasoning_content}")

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
                retrieval_stopped = "no_tool_call"
                break

            for tc in resp.tool_calls:
                t0 = time.time()
                result, done_signal = self._dispatch(tc["name"], tc["arguments"], kept_ids)
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

                # Track accumulating kept set from search results so the
                # subagent has something to keep even without explicit prunes.
                if tc["name"] == "search" and isinstance(result, dict):
                    for r in result.get("results", []):
                        kept_ids.add(r["chunk_id"])

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )

                if done_signal is not None:
                    final_kept = json.loads(done_signal)
                    break

            traj.turns = turn
            if final_kept is not None:
                break
            if turn == self._max_retrieval_turns:
                retrieval_stopped = "max_retrieval_turns"

        # Fallback: if done() was never called, keep everything gathered so far.
        if final_kept is None:
            final_kept = sorted(kept_ids)[: max(self._top_k, 5)]
        traj.extra["kept_chunk_ids"] = final_kept
        traj.extra["retrieval_stopped_reason"] = retrieval_stopped

        # --- Stage 2: reasoner answers from kept chunks -------------------
        chunks_block_parts: list[str] = []
        for cid in final_kept:
            chunk = self.corpus.get_chunk(cid)
            if not chunk:
                continue
            chunks_block_parts.append(f"[{cid}] {chunk.title}\n{chunk.text[:1500]}")
        chunks_block = "\n\n".join(chunks_block_parts) or "(no chunks retained)"

        reasoner_messages = [
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REASONER_USER_TEMPLATE.format(
                    question=question, chunks_block=chunks_block
                ),
            },
        ]
        try:
            r_resp = self.reasoner_llm.chat(
                reasoner_messages,
                max_tokens=self.ctx.max_completion_tokens,
                temperature=0.1,
            )
            traj.prompt_tokens += r_resp.prompt_tokens
            traj.completion_tokens += r_resp.completion_tokens
            if r_resp.reasoning_content:
                reasoning_chunks.append(f"[reasoner] {r_resp.reasoning_content}")
            traj.final_answer = (r_resp.text or "").strip()
            traj.reasoning = r_resp.text or ""
        except Exception as exc:  # noqa: BLE001
            traj.extra["reasoner_error"] = str(exc)
            traj.final_answer = ""

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = retrieval_stopped
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages + reasoner_messages
        return traj
