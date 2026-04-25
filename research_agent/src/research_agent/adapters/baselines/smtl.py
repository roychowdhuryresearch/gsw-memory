"""SMTL / AFM-MHQA adapter.

Reimplements OPPO PersonalAI's MHQA agent prompt from the *Search More,
Think Less* paper (``arxiv 2602.22675``) + the broader Chain-of-Agents
AFM codebase (``arxiv 2508.13167``) on top of our BM25 retriever.

The prompt defines six function tags the agent must use:

- ``<think>`` — reasoning before each action (mandatory).
- ``<plan>`` — break the question into sub-questions.
- ``<wiki_search>query</wiki_search>`` — one retrieval call.
- ``<observation>`` — injected retrieval result.
- ``<reflection>`` — PRM-style scored reflection on progress.
- ``<answer>`` — terminal.

The "Search More, Think Less" thesis is that lighter per-step thinking +
more exploration iterations beats fewer-but-deeper steps. We encode that
thesis at inference by **setting a relatively generous max_turns** and
**trusting the <think> blocks to stay short** (governed by the prompt).

Three rows of the grid use this adapter:
1. SMTL-30B as-shipped (``PersonalAILab/SMTL-30B``) — their trained checkpoint.
2. Same prompt on ``openai/gpt-oss-20b`` — our small-model swap.
3. Same prompt on ``Qwen/Qwen2.5-7B-Instruct`` — even smaller swap.

Source prompt: ``third_party/Agent_Foundation_Models/AFM/data/mhqa_agent/sys_prompts.py``
(``MHQA_PROMPT``).
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


# Verbatim from AFM/data/mhqa_agent/sys_prompts.py::MHQA_PROMPT with one
# delta: answer delimiter softened from "|" to prose since FRAMES uses
# alias-matching exact-match, not multi-answer.
SYSTEM_PROMPT = """You can only respond to a given question using the following 6 functions: \
think, plan, wiki_search, observation, reflection and answer. Below are the descriptions \
of these functions:

1. think: Before using any of plan, wiki_search, reflection or answer, you must first use \
the think function to provide the reasoning, justification, and procedural steps for the \
function you intend to use next. Begin with <think> and end with </think>. Keep each \
<think> block concise — the guiding philosophy is "search more, think less".

2. plan: Based on the given question, break it into detailed, fine-grained sub-questions \
to facilitate execution via wiki_search. Begin with <plan> and end with </plan>.

3. wiki_search: Retrieve external information. Format: <wiki_search>search_query</wiki_search>.

4. observation: This is the runtime-injected result of wiki_search. Begin with \
<observation> and end with </observation>. You do NOT emit this tag — the system does.

5. reflection: Evaluate the current trajectory. Score three criteria — information conflict, \
tool effectiveness, trajectory monitoring — as good/average/poor. If any is poor, re-plan. \
Begin with <reflection> and end with </reflection>.

6. answer: Your response must end with the answer function. Begin with <answer> and end \
with </answer>.

Rules:
- You must use think before each plan, wiki_search, reflection, or answer.
- After reflection, you must use think then either plan, wiki_search, or answer.
- Do not emit special tags (<think>, <plan>, <wiki_search>, <observation>, <reflection>, \
<answer>) inside free text.
- Keep the final answer concise. For example, <answer> Beijing </answer>.
"""


_WIKI_SEARCH_RE = re.compile(r"<wiki_search>\s*(.*?)\s*</wiki_search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)


def _format_docs(hits) -> str:
    if not hits:
        return "No results found."
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[Doc {i}] ({h.chunk.title}) {h.chunk.text[:700]}")
    return "\n\n".join(parts)


@register_adapter
class SMTLAdapter(Adapter):
    """AFM-MHQA / SMTL prompt loop on our BM25 corpus."""

    system_id = "smtl"
    display_name = "SMTL / AFM-MHQA (6-function prompt, BM25)"
    description = (
        "OPPO PersonalAI's 6-function MHQA agent prompt (<think>/<plan>/"
        "<wiki_search>/<observation>/<reflection>/<answer>) on our BM25 corpus. "
        "Works with SMTL-30B as-shipped or any base LLM under the same prompt."
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
        answers = _ANSWER_RE.findall(text)
        if answers:
            return "", answers[-1].strip()
        searches = _WIKI_SEARCH_RE.findall(text)
        if searches:
            return searches[-1].strip(), ""
        return "", ""

    def _collect_reasoning(self, text: str) -> str:
        thinks = _THINK_RE.findall(text)
        plans = _PLAN_RE.findall(text)
        return "\n".join(s.strip() for s in thinks + plans if s.strip())

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
            {"role": "user", "content": f"Question: {question}"},
        ]

        start = time.time()
        stopped_reason = "finished"
        acc_text: list[str] = []
        reasoning_chunks: list[str] = []

        for turn in range(1, self.ctx.max_turns + 1):
            try:
                # Stop at </wiki_search> or </answer> so the server yields
                # control to the adapter for retrieval / termination.
                # Without stops, SMTL hallucinates its own <observation>
                # content and emits a bogus <answer> in the same turn.
                resp = self.llm.chat(
                    messages,
                    max_tokens=self.ctx.max_completion_tokens,
                    stop=["</wiki_search>", "</answer>"],
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
            # The stop sequence is stripped by most vLLM builds; re-append
            # the closing tag so downstream regex parsing still matches.
            # This is safe even if the text already contains it.
            if "<wiki_search>" in text and "</wiki_search>" not in text:
                text = text + "</wiki_search>"
            elif "<answer>" in text and "</answer>" not in text:
                text = text + "</answer>"
            acc_text.append(text)
            messages.append({"role": "assistant", "content": text})

            search_query, answer = self._extract(text)

            if answer:
                traj.final_answer = answer
                traj.reasoning = self._collect_reasoning("\n".join(acc_text))
                traj.turns = turn
                break

            if not search_query:
                # Model may have emitted <plan> / <reflection> / <think>
                # alone and then stopped, expecting the system to prompt
                # it to proceed. Give it a nudge and continue the loop.
                lower = text.lower()
                has_scaffold = any(
                    tag in lower
                    for tag in (
                        "<plan>",
                        "</plan>",
                        "<reflection>",
                        "</reflection>",
                        "<think>",
                        "</think>",
                    )
                )
                if has_scaffold and turn < self.ctx.max_turns:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue. Emit your next step in the workflow: "
                                "a <think> block followed by either <wiki_search>"
                                "your query</wiki_search> to retrieve evidence, "
                                "or <answer> your final answer </answer> if you "
                                "are ready to answer."
                            ),
                        }
                    )
                    traj.turns = turn
                    if turn == self.ctx.max_turns:
                        stopped_reason = "max_turns"
                    continue

                # Truly empty / off-protocol output — treat as early stop.
                stopped_reason = "no_tag"
                traj.final_answer = text.strip()
                traj.reasoning = self._collect_reasoning("\n".join(acc_text))
                traj.turns = turn
                break

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
                    name="wiki_search",
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
                    "content": f"<observation>\n{info_block}\n</observation>",
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
