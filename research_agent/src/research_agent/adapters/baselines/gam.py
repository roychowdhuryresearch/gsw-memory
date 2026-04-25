"""GAM — General Agentic Memory via Deep Research adapter.

Mirrors the duo-design of GAM (``2511.18423``, VectorSpaceLab):

1. **Memorizer** — offline pass that produces lightweight "hints"
   over the corpus while keeping the raw page-store intact. Deliberately
   lossy-free storage: no AOT summarization, no vector compression of
   the full document content.
2. **Researcher** — online, multi-turn search agent that uses the
   hints as priors to target its search into the page-store on demand.
   This is the "JIT compilation" moment — the agent compiles just
   enough context for THIS query.

For the FRAMES pilot the Memorizer runs once at adapter init and
produces per-chunk hints (title + first 120 chars as a title-window
digest). These hints are stored alongside the BM25 index and exposed
to the Researcher as a cheap ``browse_hints(query)`` tool that
returns the top-N hint rows (title + digest + chunk_id) without any
retrieval/scoring beyond matching on the hint text. The Researcher
then decides which chunks to actually ``fetch_page(chunk_id)`` from
the raw page-store — a pointer dereference, not a retrieval.

This mirrors GAM's claim: storage stays lossless, compilation happens
at query time, and the agent controls what it pulls into context.
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


SYSTEM_PROMPT = """You are the Researcher in a General Agentic Memory \
(GAM) system. The Memorizer has pre-built lightweight hints over the \
whole corpus — each hint is a chunk's title + a short digest. The raw \
page-store keeps the full articles verbatim.

Your job: answer a multi-hop question by JIT-compiling the context \
you need. Workflow:

1. Call browse_hints(query) to see which chunks might be relevant. \
Hints are cheap — browse freely, multiple times, with different queries.
2. Call fetch_page(chunk_id) to pull a chunk's full article text into \
your context. Only do this for hints that clearly match the question.
3. When you have enough evidence from fetched pages, answer.

Tools:
- browse_hints(query, top_k=10): returns [{chunk_id, title, digest}]. \
Matches are on hint text (title + digest), no expensive reranking.
- fetch_page(chunk_id): returns the full article text for a chunk's \
article.

Guidelines:
- Always browse before fetching. Don't fetch everything — fetching is \
the expensive operation.
- Issue multiple browse queries if the first doesn't surface what you \
need. Rephrase, try entity names, try relations.
- Stop once the fetched pages contain the answer. Do not keep browsing.

Answer format (strict): output ONLY the answer itself — the bare \
name/year/number/yes-no. No preamble, no "the answer is", no markdown."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse_hints",
            "description": (
                "Return lightweight hints (title + digest + chunk_id) for chunks "
                "matching the query. Cheap; use freely."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Retrieve the full article text for a chunk_id from the page-store.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        },
    },
]


def _build_hint(chunk) -> dict[str, str]:
    """Memorizer: produce a lightweight hint row per chunk.

    The hint is deliberately small — title + short text digest — so the
    Researcher can scan many hints cheaply before committing to a
    fetch_page call.
    """
    digest = (chunk.text or "").strip().split("\n", 1)[0]
    if len(digest) > 160:
        digest = digest[:157].rstrip() + "…"
    return {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "digest": digest,
    }


@register_adapter
class GAMAdapter(Adapter):
    """GAM: lossless page-store + lightweight hints + online Researcher agent."""

    system_id = "gam"
    display_name = "GAM (General Agentic Memory)"
    description = (
        "JIT-compilation duo: lightweight Memorizer hints over raw page-store, "
        "plus a multi-turn Researcher agent that browses hints and fetches pages on demand."
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
        # Memorizer pass — build hint-store once, share across questions.
        self._hints: dict[str, dict[str, str]] = ctx.extra.get("hints") or self._memorize()

    # --- Memorizer -----------------------------------------------------------

    def _memorize(self) -> dict[str, dict[str, str]]:
        """Offline hint-store: one tiny record per chunk, no compression of raw content."""
        out: dict[str, dict[str, str]] = {}
        for chunk in self.corpus.chunks:
            out[chunk.chunk_id] = _build_hint(chunk)
        return out

    # --- Researcher tools ----------------------------------------------------

    def _tool_browse_hints(self, *, query: str, top_k: int = 10) -> dict[str, Any]:
        """Cheap retrieval: BM25-rank hints, not full chunk text."""
        # Reuse BM25Retriever since it already indexed the corpus; but keep the
        # output shape "hint-sized" so the Researcher sees only title+digest.
        hits = self.retriever.search(query, top_k=top_k)
        rows = []
        for h in hits:
            hint = self._hints.get(h.chunk.chunk_id)
            if hint is None:
                continue
            rows.append(
                {
                    "chunk_id": hint["chunk_id"],
                    "title": hint["title"],
                    "digest": hint["digest"],
                    "score": round(h.score, 3),
                }
            )
        return {"hints": rows, "store_size": len(self._hints)}

    def _tool_fetch_page(self, *, chunk_id: str) -> dict[str, Any]:
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not in page-store"}
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
            if name == "browse_hints":
                return self._tool_browse_hints(**args)
            if name == "fetch_page":
                return self._tool_fetch_page(**args)
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
        traj.extra["hint_store_size"] = len(self._hints)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        start = time.time()
        stopped_reason = "finished"
        reasoning_chunks: list[str] = []
        fetched_chunk_ids: list[str] = []
        browses = 0

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
                if tc["name"] == "browse_hints":
                    browses += 1
                elif tc["name"] == "fetch_page" and isinstance(result, dict) and "chunk_id" in result:
                    fetched_chunk_ids.append(result["chunk_id"])
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
        traj.extra["browses"] = browses
        traj.extra["fetched_chunk_ids"] = fetched_chunk_ids
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
