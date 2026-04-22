"""ASearcher adapter.

Re-implements ASearcher's inference loop (``inclusionAI/ASearcher``) in
prompt-only mode — no async RL training, just their prompt template +
tool tags applied to whatever LLM we swap in. Used for three rows of
the substitution grid (QwQ-32B / gpt-oss-20B / Qwen2.5-7B) AND for
loading the trained ASearcher-Web-7B / 14B checkpoints as-shipped
(same prompt, different weights).

Tag contract from ``third_party/ASearcher/ASearcher/train/prompts.py``:

- ``<think>reasoning</think>``
- ``<search>query</search>``            → retrieve
- ``<information>docs</information>``   → runtime-injected
- ``<answer>final_answer</answer>``     → terminates
- (We drop ``<access>`` since the FRAMES setup is an offline corpus.)

We use the SEARCH_ONLY_PROMPT_TEMPLATE (no URL-fetch tool) verbatim.
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


# Ported verbatim from ASearcher/ASearcher/train/prompts.py
# (SEARCH_ONLY_PROMPT_TEMPLATE). We present the body as a chat system+user
# pair since we're driving OpenAI-compatible chat completions.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant answers it. The Assistant analyzes the given question and "
    "information in the mind, retains important relevant information, calls a "
    "search engine to find necessary information, and provides the user with "
    "the answer. The Assistant conducts search by <search> query </search> "
    "and the top search results will be returned between <information> and "
    "</information>. The reasoning processes are enclosed within <think> "
    "</think>. Finally, the Assistant provides answer inside <answer> and "
    "</answer>, i.e. <answer> answer here </answer>. If there are multiple "
    "queries, ensure all answers are enclosed within <answer> </answer>, "
    "separated with comma. Note that when the Assistant finds the question is "
    "invalid, e.g. no answer could match all information in the question, "
    "the Assistant replies with '<answer> the question is invalid. </answer>'. "
    "You should try to find the most likely answer. The language of your "
    "answer should align with the question."
)


_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _format_docs(hits) -> str:
    if not hits:
        return "No search results are found."
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[Doc {i}] (Title: {h.chunk.title})\n{h.chunk.text[:800]}")
    return "\n\n".join(parts)


@register_adapter
class ASearcherAdapter(Adapter):
    """ASearcher search-only prompt loop over our BM25 retriever.

    Same adapter serves two cells per model scale:
    - ``system_id="asearcher_prompt"`` — untrained base LLM + ASearcher prompt
      (Tier-2 prompt-mode swap rows).
    - Loaded with ``model_id="inclusionAI/ASearcher-Web-7B"`` etc. — the
      trained checkpoint as-shipped (Tier-2 trained-ckpt rows).
    """

    system_id = "asearcher_prompt"
    display_name = "ASearcher (prompt-only, search-only tags)"
    description = (
        "Re-implements the ASearcher <think>/<search>/<answer> loop on top of "
        "our BM25 retriever. Also used for running ASearcher's trained "
        "checkpoints (Web-7B/14B) as-shipped under the same prompt."
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

    # ----------------------------------------------------------------------

    def _extract(self, text: str) -> tuple[str, str]:
        """Return (search_query, final_answer). First match wins for each tag."""
        ans_matches = _ANSWER_RE.findall(text)
        if ans_matches:
            # Multiple answers allowed; join as comma-separated per the prompt.
            return "", ", ".join(m.strip() for m in ans_matches)
        sear_matches = _SEARCH_RE.findall(text)
        if sear_matches:
            return sear_matches[-1].strip(), ""
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
            {"role": "user", "content": question.strip()},
        ]

        start = time.time()
        stopped_reason = "finished"
        acc_text: list[str] = []
        reasoning_chunks: list[str] = []
        # ASearcher's reference uses stop=["</search>", "</access>", "</answer>"].
        # Without this, the model role-plays both sides of the protocol and
        # emits fake empty <information> blocks itself — same failure mode we
        # hit on search_o1 before patching.
        stop_seq = ["</search>", "</answer>"]

        for turn in range(1, self.ctx.max_turns + 1):
            try:
                resp = self.llm.chat(
                    messages,
                    max_tokens=self.ctx.max_completion_tokens,
                    stop=stop_seq,
                )
            except TypeError:
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
            # vLLM / OpenAI truncate at the stop string and do not echo it,
            # but the extractor regex needs the closing tag present. Re-append
            # whichever tag was likely cut, using a simple balance check.
            def _reattach(t: str) -> str:
                if not t:
                    return t
                if resp.finish_reason != "stop":
                    return t
                if t.count("<search>") > t.count("</search>"):
                    return t + "</search>"
                if t.count("<answer>") > t.count("</answer>"):
                    return t + "</answer>"
                return t
            text = _reattach(text)
            reasoning_text = _reattach(reasoning_text)
            # On Bedrock gpt-oss, output lands entirely in reasoning_content;
            # fall back to that channel when the visible text is empty.
            effective = text if text.strip() else reasoning_text
            acc_text.append(effective)
            messages.append({"role": "assistant", "content": text})

            search_query, answer = self._extract(effective)

            if answer:
                traj.final_answer = answer
                traj.reasoning = self._collect_reasoning("\n".join(acc_text))
                traj.turns = turn
                break

            if not search_query:
                # No tags — the model produced free-form text as its final output.
                # Prefer visible text, fall back to reasoning_content (Bedrock gpt-oss).
                stopped_reason = "no_tag"
                traj.final_answer = (text.strip() or reasoning_text.strip())
                traj.reasoning = self._collect_reasoning("\n".join(acc_text))
                traj.turns = turn
                break

            # Retrieve and inject.
            t0 = time.time()
            try:
                hits = self.retriever.search(search_query, top_k=self._top_k)
                info_block = _format_docs(hits)
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

            messages.append(
                {
                    "role": "user",
                    "content": f"<information>\n{info_block}\n</information>",
                }
            )
            traj.turns = turn
            if turn == self.ctx.max_turns:
                stopped_reason = "max_turns"

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.reasoning = traj.reasoning or self._collect_reasoning("\n".join(acc_text))
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj


# Register a second alias for the trained checkpoints so the grid can cite
# them separately in reports (same code, different system_id → distinct
# CellResult rows).
@register_adapter
class ASearcherTrainedAdapter(ASearcherAdapter):
    system_id = "asearcher_trained"
    display_name = "ASearcher (trained ckpt, search-only tags)"
    description = (
        "Identical prompt+loop to asearcher_prompt; distinct system_id so the "
        "grid reports the trained-checkpoint runs separately."
    )
