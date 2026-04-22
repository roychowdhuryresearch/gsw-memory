"""Graph-R1 adapter.

Mirrors the inference-time behavior of Graph-R1 (``2507.21892``,
[LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1)) — an agentic
GraphRAG framework that interleaves four actions per step:

1. ``<think>`` — continue or terminate reasoning.
2. ``<query>`` — formulate a retrieval query.
3. ``<retrieve>`` — pull chunks from the corpus, then extract
   entity-relation n-ary facts ("hyperedges") from what came back.
4. ``<answer>`` — produce the final answer.

**Why this is prompt-mode, not trained-checkpoint.** The upstream
Graph-R1 repo releases training code (GRPO/REINFORCE++/PPO bash
scripts) + training data and a pre-built knowledge-hypergraph on
TeraBox — they do NOT publish a HuggingFace checkpoint. Their
headline Qwen2.5-3B/7B numbers come from weights you train yourself
on their HotpotQA/2Wiki/MuSiQue/NQ/PopQA/TriviaQA supplied
hypergraphs (not FRAMES). Training a Graph-R1 on FRAMES is
paper-scope work, not pilot-scope. For the pilot we run the
paper's inference-time loop against any instruction-tuned base
model (Qwen2.5-7B-Instruct, gpt-oss-20B, GPT-5) — same as how
``search_o1`` wraps QwQ / Qwen without their specific ckpt. The
trained-checkpoint alias ``graph_r1_trained`` is reserved below
for when we (or upstream) ship real weights.

Two other deviations from the paper for pragmatic FRAMES evaluation:

- **Hypergraph is built on-the-fly from retrieved chunks** rather
  than pre-indexed across the whole corpus. FRAMES has ~366 articles
  — pre-building a hypergraph over them via GPT-4o-mini extraction
  (upstream's ``script_build.py``) would add a one-off corpus-level
  pass worth ~$20. We avoid it here; each retrieve call does a small
  extract pass on only the chunks it returned, and the resulting
  hyperedges accumulate across turns in the agent's working memory.
  A fair comparison still — the agent gets graph-structured state,
  scoped to what it's actually retrieved.
- **BM25 retrieval** via our existing shared retriever rather than
  the paper's vector-search + hypergraph-walk retriever. The same
  substitution every other adapter in the grid makes.

Parsing follows the same balanced-regex approach as ``search_r1.py``.
Extracted n-ary facts are stored as simple (subject, predicate,
object, [context…]) tuples — no cross-turn entity canonicalization.
Tuples accumulate in ``traj.extra.hyperedges`` so the aggregator can
audit the graph snapshot when the agent stopped.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import ToolCall, Trajectory
from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.corpus import load_frames_corpus


SYSTEM_PROMPT = """You are a Graph-R1 agent. You answer multi-hop \
questions by alternating between reasoning, querying a corpus, \
extracting small knowledge-graph hyperedges from what you retrieve, \
and reasoning over the accumulated graph until you can answer.

Per-step format (strict):
- <think> your reasoning about what to do next </think>
- Then EXACTLY one of:
  - <query> a focused retrieval query </query>
  - <answer> final answer </answer>

When you emit <query>, the environment will return a block of the form:
  <information>
  Retrieved chunks: ...
  Extracted hyperedges so far: [(subj, rel, obj, ctx), ...]
  </information>

Guidelines:
- Issue targeted queries; do not re-query the same phrase twice.
- Use the accumulated hyperedge list as your working memory — refer to \
entities and relations you have already extracted when planning the \
next query.
- Stop as soon as the hyperedges chain together an answer. Do not \
continue retrieving once the evidence is in the graph.

Answer format: inside <answer>, output ONLY the answer itself. No \
preamble, no "the answer is". Names as plain names ("Rob Reiner"), \
years as the year ("1994"), yes/no starting with Yes or No."""


USER_TEMPLATE = "Question: {question}\n"


_QUERY_RE = re.compile(r"<query>\s*(.*?)\s*</query>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _format_information(chunks_block: str, hyperedges: list[tuple[str, str, str, str]]) -> str:
    if not hyperedges:
        edges_repr = "(none yet)"
    else:
        parts = []
        for i, (s, r, o, ctx) in enumerate(hyperedges, 1):
            snippet = (ctx[:80] + "…") if len(ctx) > 80 else ctx
            parts.append(f"  {i}. ({s!r}, {r!r}, {o!r})  // {snippet}")
        edges_repr = "\n".join(parts)
    return (
        "<information>\n"
        f"Retrieved chunks:\n{chunks_block or '(no documents retrieved)'}\n\n"
        f"Accumulated hyperedges:\n{edges_repr}\n"
        "</information>\n"
    )


def _format_chunks(hits) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(f"Doc {i} (Title: {h.chunk.title}): {h.chunk.text[:500]}")
    return "\n".join(parts)


EXTRACT_SYSTEM_PROMPT = """You extract small knowledge-graph \
hyperedges from text. A hyperedge is (subject, relation, object, \
short_context_phrase). Output STRICT JSON: {"edges": [["s","r","o", \
"ctx"], ...]}. Extract at most 6 edges per call. Focus on edges that \
name an entity and connect it to a concrete fact. Skip vague/generic \
statements."""


@register_adapter
class GraphR1Adapter(Adapter):
    """Inference-only Graph-R1 wrapper: BM25 + on-the-fly hyperedge extraction."""

    system_id = "graph_r1"
    display_name = "Graph-R1 (on-demand hyperedge scratchpad)"
    description = (
        "Think/query/retrieve/answer loop where each retrieval extracts "
        "entity-relation hyperedges that accumulate as the agent's working memory."
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

    def _extract_query_and_answer(self, text: str) -> tuple[str, str]:
        answer_match = _ANSWER_RE.search(text)
        if answer_match:
            return "", answer_match.group(1).strip()
        query_match = _QUERY_RE.findall(text)
        if query_match:
            return query_match[-1].strip(), ""
        return "", ""

    def _collect_reasoning(self, text: str) -> str:
        return "\n".join(m.strip() for m in _THINK_RE.findall(text))

    def _extract_hyperedges(self, chunks_block: str) -> list[tuple[str, str, str, str]]:
        """Second, cheap LLM call: pull hyperedges from this turn's retrieved chunks."""
        if not chunks_block.strip():
            return []
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": chunks_block},
                ],
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            payload = json.loads(resp.text or "{}")
        except Exception:  # noqa: BLE001
            return []
        raw = payload.get("edges") or []
        edges: list[tuple[str, str, str, str]] = []
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            s, r, o = (str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip())
            ctx = str(row[3]).strip() if len(row) > 3 else ""
            if s and r and o:
                edges.append((s, r, o, ctx))
        return edges

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
        traj.extra["hyperedges"] = []
        hyperedges: list[tuple[str, str, str, str]] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question.strip())},
        ]

        start = time.time()
        stopped_reason = "finished"
        accumulated_text: list[str] = []
        reasoning_chunks: list[str] = []
        seen_queries: set[str] = set()

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
            messages.append({"role": "assistant", "content": text})

            query, answer = self._extract_query_and_answer(text)

            if answer:
                traj.final_answer = answer
                traj.reasoning = self._collect_reasoning("\n".join(accumulated_text))
                traj.turns = turn
                break

            if not query:
                stopped_reason = "no_tag"
                traj.final_answer = text.strip()
                traj.turns = turn
                break

            # Dedup trivial repetition.
            norm_q = query.lower().strip()
            if norm_q in seen_queries:
                query = f"{query} (rephrase)"
            seen_queries.add(norm_q)

            # Retrieve chunks.
            t0 = time.time()
            try:
                hits = self.retriever.search(query, top_k=self._top_k)
                chunks_block = _format_chunks(hits)
                err = ""
            except Exception as exc:  # noqa: BLE001
                chunks_block = f"(retrieval error: {exc})"
                err = str(exc)
                hits = []

            # Second LLM pass: extract hyperedges from this turn's chunks.
            new_edges = self._extract_hyperedges(chunks_block)
            for edge in new_edges:
                if edge not in hyperedges:
                    hyperedges.append(edge)

            traj.tool_calls.append(
                ToolCall(
                    turn=turn,
                    name="graph_retrieve",
                    args={"query": query, "top_k": self._top_k},
                    result_preview=chunks_block[:500],
                    result_full=chunks_block + f"\n[edges extracted this turn: {len(new_edges)}]",
                    duration_s=round(time.time() - t0, 3),
                    error=err,
                )
            )

            info = _format_information(chunks_block, hyperedges)
            messages.append({"role": "user", "content": info})

            traj.turns = turn
            if turn == self.ctx.max_turns:
                stopped_reason = "max_turns"

        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["hyperedges"] = [list(e) for e in hyperedges]
        traj.extra["hyperedge_count"] = len(hyperedges)
        traj.reasoning = traj.reasoning or self._collect_reasoning("\n".join(accumulated_text))
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        return traj


# Reserved alias for a future locally-trained Graph-R1 checkpoint.
# Upstream ships training scripts (GRPO / REINFORCE++ / PPO) + HotpotQA/
# 2Wiki/MuSiQue/NQ/PopQA/TriviaQA hypergraphs on TeraBox but no released
# HF weights. When we train our own on FRAMES (or just swap in a ckpt
# someone else trains upstream), point ``--model`` at it and run under
# ``graph_r1_trained`` so the grid reports it as a distinct row.
@register_adapter
class GraphR1TrainedAdapter(GraphR1Adapter):
    system_id = "graph_r1_trained"
    display_name = "Graph-R1 (trained checkpoint, hyperedge scratchpad)"
    description = (
        "Identical prompt+loop to graph_r1; distinct system_id so the grid "
        "reports locally-trained checkpoint runs separately. Requires a "
        "Graph-R1-trained model served at --base-url; no HF weights exist "
        "upstream (training is user-run)."
    )
