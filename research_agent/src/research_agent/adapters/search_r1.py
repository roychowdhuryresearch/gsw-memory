"""Search-R1 adapter.

Mirrors the PPO-trained ``Search-R1`` framework (``PeterGriffinJin/
Search-R1``) in inference-only mode. Unlike the tool-calls style of
Vanilla RAG+ReAct, Search-R1 uses **inline text tags**:

- ``<think>reasoning</think>`` — explicit reasoning trace.
- ``<search>query</search>`` — tells the runtime to retrieve.
- ``<information>chunks</information>`` — injected by the runtime.
- ``<answer>final</answer>`` — terminates the loop.

The substitution story: we call any chat-capable model with the
Search-R1 system prompt and parse the stream of tags. When
``<search>`` appears we pause, retrieve via our own BM25 index, and
re-issue a continuation with ``<information>…</information>``
appended. Same control flow as the reference ``infer.py`` but driven
over OpenAI-compatible chat completions so we can swap in gpt-oss-20B,
Qwen2.5-7B, Qwen2.5-3B, or the Search-R1-trained 7B checkpoint.

The agent's own ``<think>`` blocks are preserved as the reasoning
field of the resulting ``Trajectory``.
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


# Verbatim structure from Search-R1's infer.py user prompt.
SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time \
you get new information. After reasoning, if you find you lack some \
knowledge, you can call a search engine by <search> query </search> and \
it will return the top searched results between <information> and \
</information>. You can search as many times as you want. If you find \
no further external knowledge needed, you can directly provide the \
answer inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Beijing </answer>."""

USER_TEMPLATE = "Question: {question}\n"

_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_INFO_OPEN = "<information>"
_INFO_CLOSE = "</information>"


def _format_retrieval(hits) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(f"Doc {i}(Title: {h.chunk.title}) {h.chunk.text[:500]}")
    return "\n".join(parts)


@register_adapter
class SearchR1Adapter(Adapter):
    """Inference-only Search-R1 wrapper using our BM25 retriever + any LLM."""

    system_id = "search_r1"
    display_name = "Search-R1 (inline-tag ReAct, BM25)"
    description = (
        "Re-implements the Search-R1 think/search/answer tag loop on top of our "
        "BM25 retriever. Works with their trained 7B ckpt or any base LLM."
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
        self._top_k = int(ctx.extra.get("top_k", 3))

    # ----------------------------------------------------------------------

    def _extract_search_and_answer(self, text: str) -> tuple[str, str]:
        """Return (search_query, final_answer). At most one of them is non-empty."""
        answer_match = _ANSWER_RE.search(text)
        if answer_match:
            return "", answer_match.group(1).strip()
        search_match = _SEARCH_RE.findall(text)
        if search_match:
            return search_match[-1].strip(), ""
        return "", ""

    def _collect_reasoning(self, text: str) -> str:
        return "\n".join(m.strip() for m in _THINK_RE.findall(text))

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
            {"role": "user", "content": USER_TEMPLATE.format(question=question.strip().rstrip("?") + "?")},
        ]

        start = time.time()
        stopped_reason = "finished"
        accumulated_text: list[str] = []
        reasoning_chunks: list[str] = []

        for turn in range(1, self.ctx.max_turns + 1):
            try:
                resp = self.llm.chat(
                    messages,
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
            text = resp.text or ""
            accumulated_text.append(text)

            # Echo the assistant turn into the message list so the next call
            # sees the full chain.
            messages.append({"role": "assistant", "content": text})

            search_query, answer = self._extract_search_and_answer(text)

            if answer:
                traj.final_answer = answer
                traj.reasoning = self._collect_reasoning("\n".join(accumulated_text))
                traj.turns = turn
                break

            if not search_query:
                # Model produced neither a search nor an answer — treat as
                # premature stop.
                stopped_reason = "no_tag"
                traj.final_answer = text.strip()
                traj.turns = turn
                break

            t0 = time.time()
            try:
                hits = self.retriever.search(search_query, top_k=self._top_k)
                info_block = _format_retrieval(hits) or "(no documents retrieved)"
                err = ""
            except Exception as exc:  # noqa: BLE001
                info_block = f"(retrieval error: {exc})"
                err = str(exc)
                hits = []

            traj.tool_calls.append(
                ToolCall(
                    turn=turn,
                    name="search",
                    args={"query": search_query, "top_k": self._top_k},
                    result_preview=info_block[:500],
                    result_full=info_block,
                    duration_s=round(time.time() - t0, 3),
                    error=err,
                )
            )

            # Re-issue the prompt with <information>…</information> appended
            # as the next user turn, matching the upstream loop.
            messages.append(
                {
                    "role": "user",
                    "content": f"{_INFO_OPEN}\n{info_block}\n{_INFO_CLOSE}\n",
                }
            )

            traj.turns = turn
            if turn == self.ctx.max_turns:
                stopped_reason = "max_turns"

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.reasoning = traj.reasoning or self._collect_reasoning("\n".join(accumulated_text))
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj
