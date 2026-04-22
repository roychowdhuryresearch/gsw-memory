"""Search-o1 adapter.

Re-implements the RUC-NLPIR/Search-o1 prompt-only framework on top of
our BM25 retriever. Key differences vs Search-R1:

- Uses tag pair ``<|begin_search_query|>``…``<|end_search_query|>`` and
  injects ``<|begin_search_result|>``…``<|end_search_result|>``.
- The Reason-in-Documents module adds a **second LLM call per
  retrieval** to compress raw docs into a reason-ready chunk before
  injection — makes Search-o1 more expensive but reduces noise.
- No explicit ``<answer>`` tag. Completion ends when the model stops
  emitting new search queries; the final block of reasoning is taken
  as the answer.

Reference: ``third_party/Search-o1/scripts/run_search_o1.py``,
``prompts.py::get_multiqa_search_o1_instruction``.
"""

from __future__ import annotations

import re
import time
from typing import Any

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import ToolCall, Trajectory
from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.corpus import load_frames_corpus


BEGIN_QUERY = "<|begin_search_query|>"
END_QUERY = "<|end_search_query|>"
BEGIN_RESULT = "<|begin_search_result|>"
END_RESULT = "<|end_search_result|>"

_QUERY_RE = re.compile(
    rf"{re.escape(BEGIN_QUERY)}\s*(.*?)\s*{re.escape(END_QUERY)}",
    re.DOTALL,
)


def _system_prompt(max_search_limit: int) -> str:
    """Search-o1 multi-hop QA system prompt (ported from prompts.py)."""
    return (
        "You are a reasoning assistant with the ability to perform web searches to "
        "help you answer the user's question accurately. You have special tools:\n\n"
        f"- To perform a search: write {BEGIN_QUERY} your query here {END_QUERY}.\n"
        "Then, the system will search and analyze relevant web pages, then provide "
        f"you with helpful information in the format {BEGIN_RESULT} ...search results... {END_RESULT}.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum "
        f"number of search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning and give "
        "a clear, concise final answer."
    )


def _reason_in_docs_prompt(prev_reasoning: str, query: str, documents: str) -> str:
    """Ported from prompts.py::get_webpage_to_reasonchain_instruction."""
    return (
        "You are given the reasoning so far, a search query, and raw search results. "
        "Extract ONLY the facts from the results that are directly useful to answer "
        "the query in light of the prior reasoning. Keep it short.\n\n"
        f"=== Prior reasoning ===\n{prev_reasoning.strip() or '(none yet)'}\n\n"
        f"=== Search query ===\n{query.strip()}\n\n"
        f"=== Raw search results ===\n{documents.strip()}\n\n"
        "=== Your distilled note ==="
    )


def _format_docs(hits) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[Doc {i}: {h.chunk.title}]\n{h.chunk.text[:700]}")
    return "\n\n".join(parts)


@register_adapter
class SearchO1Adapter(Adapter):
    """Search-o1 re-implementation over BM25 + our chat LLM."""

    system_id = "search_o1"
    display_name = "Search-o1 (special-token RAG + reason-in-docs)"
    description = (
        "Ports Search-o1's |begin_search_query| loop onto our retriever. Adds the "
        "Reason-in-Documents inner call per retrieval."
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
        self._max_search_limit = int(ctx.extra.get("max_search_limit", 10))
        self._enable_reason_in_docs = bool(ctx.extra.get("enable_reason_in_docs", True))

    # ----------------------------------------------------------------------

    def _distill_docs(
        self,
        *,
        prev_reasoning: str,
        query: str,
        documents: str,
        traj: Trajectory,
    ) -> str:
        """Second-pass LLM call to compress raw docs into a reason-ready note.

        Mirrors Search-o1's Reason-in-Documents module. Called once per
        retrieval. Falls back to the raw docs if the LLM call errors.
        """
        if not self._enable_reason_in_docs:
            return documents
        try:
            resp = self.llm.chat(
                [
                    {"role": "user", "content": _reason_in_docs_prompt(prev_reasoning, query, documents)}
                ],
                max_tokens=800,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            traj.extra.setdefault("reason_in_docs_errors", []).append(str(exc))
            return documents
        traj.prompt_tokens += resp.prompt_tokens
        traj.completion_tokens += resp.completion_tokens
        distilled = (resp.text or "").strip()
        return distilled or documents

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

        max_search_limit = min(self._max_search_limit, self.ctx.max_turns)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(max_search_limit)},
            {"role": "user", "content": f"Question: {question}\nPlease reason step by step and answer."},
        ]

        start = time.time()
        stopped_reason = "finished"
        n_searches = 0
        last_assistant_text = ""
        last_assistant_reasoning = ""
        reasoning_chunks: list[str] = []

        # Halt generation the moment the model closes a search query so the
        # adapter can inject a real retrieval result. Without this stop, the
        # model role-plays both sides of the protocol — emits fake empty
        # ``<|begin_search_result|>…<|end_search_result|>`` blocks itself
        # and keeps going, burning the whole completion budget on one turn.
        stop_seq = [END_QUERY]
        for turn in range(1, self.ctx.max_turns + 1):
            try:
                resp = self.llm.chat(
                    messages,
                    max_tokens=self.ctx.max_completion_tokens,
                    stop=stop_seq,
                )
            except TypeError:
                # Older LLMClients without a ``stop`` kwarg — fall back cleanly.
                resp = self.llm.chat(messages, max_tokens=self.ctx.max_completion_tokens)
            except Exception as exc:  # noqa: BLE001
                stopped_reason = "llm_error"
                traj.extra["llm_error"] = str(exc)
                break

            traj.prompt_tokens += resp.prompt_tokens
            traj.completion_tokens += resp.completion_tokens
            if resp.reasoning_content:
                reasoning_chunks.append(f"[turn {turn}] {resp.reasoning_content}")
            text = resp.text or ""
            reasoning_text = resp.reasoning_content or ""
            # vLLM / OpenAI truncate at the stop string and do NOT echo it,
            # but the query regex needs ``<|end_search_query|>`` present.
            # Re-append when the last visible token is an unclosed query.
            if (
                resp.finish_reason == "stop"
                and BEGIN_QUERY in text
                and not text.rstrip().endswith(END_QUERY)
            ):
                text = text + END_QUERY
            if (
                resp.finish_reason == "stop"
                and reasoning_text
                and BEGIN_QUERY in reasoning_text
                and not reasoning_text.rstrip().endswith(END_QUERY)
            ):
                reasoning_text = reasoning_text + END_QUERY
            last_assistant_text = text
            last_assistant_reasoning = reasoning_text
            messages.append({"role": "assistant", "content": text})

            # Bedrock gpt-oss (and some other reasoning models) route all
            # generated tokens into ``reasoning_content`` with nothing on
            # the visible ``text`` channel. Search-o1's query-tag protocol
            # assumes tags arrive in visible output, so we widen the scan
            # to cover both streams and record which one produced the hit.
            queries_text = _QUERY_RE.findall(text)
            queries_reasoning = _QUERY_RE.findall(reasoning_text) if reasoning_text else []
            queries = queries_text or queries_reasoning
            query_source = "text" if queries_text else ("reasoning" if queries_reasoning else "")
            if query_source:
                traj.extra.setdefault("query_sources", []).append(
                    {"turn": turn, "source": query_source}
                )
            if not queries or n_searches >= max_search_limit:
                # Model stopped requesting searches — take its final output.
                # Prefer visible text; fall back to reasoning_content when
                # the model emitted nothing visible (gpt-oss/Bedrock case).
                if text.strip():
                    traj.final_answer = text.strip()
                    traj.reasoning = text
                    traj.extra.setdefault("answer_source", "text")
                else:
                    traj.final_answer = reasoning_text.strip()
                    traj.reasoning = reasoning_text
                    traj.extra.setdefault("answer_source", "reasoning")
                traj.turns = turn
                if n_searches >= max_search_limit:
                    stopped_reason = "search_limit"
                break

            query = queries[-1].strip()
            n_searches += 1
            t0 = time.time()
            try:
                hits = self.retriever.search(query, top_k=self._top_k)
                raw_docs = _format_docs(hits) or "(no documents retrieved)"
                err = ""
            except Exception as exc:  # noqa: BLE001
                hits = []
                raw_docs = f"(retrieval error: {exc})"
                err = str(exc)
            # Feed the distill step whichever stream the model actually used.
            prev = text if text.strip() else reasoning_text
            distilled = self._distill_docs(
                prev_reasoning=prev, query=query, documents=raw_docs, traj=traj
            )

            traj.tool_calls.append(
                ToolCall(
                    turn=turn,
                    name="search",
                    args={"query": query, "top_k": self._top_k},
                    result_preview=distilled[:500],
                    result_full=distilled,
                    duration_s=round(time.time() - t0, 3),
                    error=err,
                )
            )
            traj.extra.setdefault("reason_in_docs_distilled", []).append(
                {"query": query, "raw_docs": raw_docs, "note": distilled}
            )

            messages.append(
                {
                    "role": "user",
                    "content": f"{BEGIN_RESULT}\n{distilled}\n{END_RESULT}\n",
                }
            )
            traj.turns = turn
            if turn == self.ctx.max_turns:
                stopped_reason = "max_turns"

        if not traj.final_answer:
            if last_assistant_text.strip():
                traj.final_answer = last_assistant_text.strip()
                traj.reasoning = last_assistant_text
                traj.extra.setdefault("answer_source", "text")
            else:
                traj.final_answer = last_assistant_reasoning.strip()
                traj.reasoning = last_assistant_reasoning
                traj.extra.setdefault("answer_source", "reasoning")
        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["n_searches"] = n_searches
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
