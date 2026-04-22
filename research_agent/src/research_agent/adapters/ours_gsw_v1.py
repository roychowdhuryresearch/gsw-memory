"""Our system — Tier 3 v1 (zero-shot, focused GSW per sub-question).

Pipeline per question:

1. **Problem composer** — LLM decomposes the question into a flat list of
   sub-questions with explicit entity focus and hop type.
2. **Focused retrieval** — for each sub-question, BM25 pulls top-k chunks
   from the FRAMES article corpus.
3. **Focused GSW** — LLM extracts entities + binary (subject, verb, object)
   triples from the retrieved chunks of that sub-question. This is the
   scratchpad this adapter's thesis hangs on.
4. **Sub-answer** — LLM answers the sub-question using only the
   GSW triples + chunk text.
5. **Aggregator** — LLM combines sub-answers into the final answer.

Same base model (gpt-oss-20B by default) is used for all 5 stages; the
variation vs other adapters is the **structured intermediate GSW**, not
the model. No training: this is the zero-shot v1.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import ToolCall, Trajectory
from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.corpus import load_frames_corpus


# ---------------------------------------------------------------------------
# Structured outputs (Pydantic). Parsed from raw JSON the LLM returns.
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    sub_id: str
    question: str
    entity_focus: str = Field("", description="Primary entity this sub-Q is about.")
    hop_type: str = Field("", description="e.g. 'lookup', 'compare', 'aggregate', 'temporal'.")


class DecomposeOutput(BaseModel):
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class Triple(BaseModel):
    subject: str
    verb: str
    obj: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class GSWExtractionOutput(BaseModel):
    entities: list[str] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


DECOMPOSE_SYSTEM = """You are a research planner. Decompose the user's \
multi-hop question into a flat list of sub-questions. Each sub-question \
should be independently answerable by retrieving a few documents. Identify \
the main entity and the hop type for each.

Return STRICT JSON matching this schema:
{
  "sub_questions": [
    {"sub_id": "s1", "question": "...", "entity_focus": "...", "hop_type": "lookup|compare|aggregate|temporal|other"}
  ]
}
Emit 1 sub-question if the original is single-hop. Emit at most 6."""


GSW_SYSTEM = """You are an information extractor. Given a sub-question and \
a set of retrieved text chunks, extract (a) the named entities mentioned \
and (b) binary (subject, verb, object) relation triples that are directly \
supported by the chunks. Every triple must cite the chunk_ids that support \
it. Be faithful — never invent a triple.

Return STRICT JSON:
{
  "entities": ["Entity A", "Entity B", ...],
  "triples": [
    {"subject": "Entity A", "verb": "short relation phrase", "obj": "Entity B", "evidence_chunk_ids": ["<id>"]}
  ]
}
Emit at most 12 triples — include only those most relevant to the sub-question."""


SUB_ANSWER_SYSTEM = """You answer a sub-question using a small entity-relation \
graph (triples) plus the source chunks. Use ONLY the provided evidence. \
Answer in one concise phrase. If the evidence is insufficient, say "unknown"."""


AGGREGATE_SYSTEM = """You combine sub-answers into the final answer to the \
original multi-hop question. Use only the sub-answers provided. Answer in \
one concise phrase — one clause if possible. If any critical sub-answer is \
"unknown", still produce a best-effort final answer."""


# ---------------------------------------------------------------------------
# JSON-extraction helper: strips code fences, picks the outermost JSON object.
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str, model_cls) -> BaseModel | None:
    if not text:
        return None
    candidates = [text]
    block = _JSON_BLOCK_RE.search(text)
    if block:
        candidates.insert(0, block.group(0))
    for cand in candidates:
        try:
            return model_cls.model_validate_json(cand)
        except ValidationError:
            continue
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------


@register_adapter
class OursGSWv1Adapter(Adapter):
    """Zero-shot problem-composer → focused-GSW → sub-answer → aggregator."""

    system_id = "ours_gsw_v1"
    display_name = "Ours — Tier 3 v1 (focused GSW, zero-shot)"
    description = (
        "Decompose → focused retrieval → per-sub-Q GSW triple extraction → "
        "per-sub-Q answer → aggregation. No training. gpt-oss-20B by default."
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
        self._top_k = int(ctx.extra.get("top_k", 6))
        self._max_sub_questions = int(ctx.extra.get("max_sub_questions", 6))

    # ----------------------------------------------------------------------

    def _decompose(self, question: str, traj: Trajectory) -> list[SubQuestion]:
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ]
        try:
            resp = self.llm.chat(
                messages,
                max_tokens=1200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = self.llm.chat(messages, max_tokens=1200, temperature=0.0)
        traj.prompt_tokens += resp.prompt_tokens
        traj.completion_tokens += resp.completion_tokens
        if resp.reasoning_content:
            traj.extra.setdefault("_reasoning_chunks", []).append(
                f"[decompose] {resp.reasoning_content}"
            )
        out = _parse_json(resp.text, DecomposeOutput)
        if out is None or not out.sub_questions:
            # Fallback: treat the whole question as its own single sub-question.
            return [SubQuestion(sub_id="s1", question=question, entity_focus="", hop_type="lookup")]
        return out.sub_questions[: self._max_sub_questions]

    def _retrieve(self, sub_q: SubQuestion, traj: Trajectory) -> list:
        query = sub_q.question
        if sub_q.entity_focus:
            query = f"{sub_q.entity_focus} — {query}"
        t0 = time.time()
        hits = self.retriever.search(query, top_k=self._top_k)
        titles = [h.chunk.title for h in hits]
        full_payload = json.dumps(
            {
                "titles": titles,
                "hits": [
                    {"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "text": h.chunk.text}
                    for h in hits
                ],
            }
        )
        traj.tool_calls.append(
            ToolCall(
                turn=0,  # we fold per-sub-Q calls into a flat list
                name="search",
                args={"query": query, "top_k": self._top_k, "sub_id": sub_q.sub_id},
                result_preview=json.dumps(titles)[:300],
                result_full=full_payload,
                duration_s=round(time.time() - t0, 3),
            )
        )
        return hits

    def _build_gsw(
        self, sub_q: SubQuestion, hits, traj: Trajectory
    ) -> GSWExtractionOutput:
        if not hits:
            return GSWExtractionOutput()
        chunks_block = "\n\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.title}\n{h.chunk.text[:900]}" for h in hits
        )
        messages = [
            {"role": "system", "content": GSW_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Sub-question (id={sub_q.sub_id}): {sub_q.question}\n"
                    f"Entity focus: {sub_q.entity_focus or '(unspecified)'}\n\n"
                    f"Retrieved chunks:\n{chunks_block}"
                ),
            },
        ]
        try:
            resp = self.llm.chat(
                messages,
                max_tokens=1500,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = self.llm.chat(messages, max_tokens=1500, temperature=0.0)
        traj.prompt_tokens += resp.prompt_tokens
        traj.completion_tokens += resp.completion_tokens
        if resp.reasoning_content:
            traj.extra.setdefault("_reasoning_chunks", []).append(
                f"[build_gsw sub={sub_q.sub_id}] {resp.reasoning_content}"
            )
        out = _parse_json(resp.text, GSWExtractionOutput)
        return out or GSWExtractionOutput()

    def _answer_sub(
        self, sub_q: SubQuestion, gsw: GSWExtractionOutput, hits, traj: Trajectory
    ) -> str:
        triples_block = "\n".join(
            f"- ({t.subject} | {t.verb} | {t.obj})  [ev: {', '.join(t.evidence_chunk_ids)}]"
            for t in gsw.triples
        ) or "(no triples)"
        chunks_block = "\n\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.title}\n{h.chunk.text[:700]}" for h in hits
        ) or "(no chunks)"
        messages = [
            {"role": "system", "content": SUB_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Sub-question: {sub_q.question}\n\n"
                    f"Entity-relation graph for this sub-question:\n{triples_block}\n\n"
                    f"Supporting chunks:\n{chunks_block}\n\n"
                    "Answer:"
                ),
            },
        ]
        resp = self.llm.chat(messages, max_tokens=300, temperature=0.1)
        traj.prompt_tokens += resp.prompt_tokens
        traj.completion_tokens += resp.completion_tokens
        if resp.reasoning_content:
            traj.extra.setdefault("_reasoning_chunks", []).append(
                f"[answer_sub sub={sub_q.sub_id}] {resp.reasoning_content}"
            )
        return (resp.text or "").strip()

    def _aggregate(
        self, question: str, sub_pairs: list[tuple[SubQuestion, str]], traj: Trajectory
    ) -> str:
        pairs_block = "\n".join(
            f"- ({sq.sub_id}) {sq.question}\n    -> {ans or '(no answer)'}"
            for sq, ans in sub_pairs
        )
        messages = [
            {"role": "system", "content": AGGREGATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Sub-answers:\n{pairs_block}\n\n"
                    "Final answer:"
                ),
            },
        ]
        resp = self.llm.chat(messages, max_tokens=300, temperature=0.1)
        traj.prompt_tokens += resp.prompt_tokens
        traj.completion_tokens += resp.completion_tokens
        if resp.reasoning_content:
            traj.extra.setdefault("_reasoning_chunks", []).append(
                f"[aggregate] {resp.reasoning_content}"
            )
        traj.extra["aggregate_messages"] = messages
        return (resp.text or "").strip()

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

        start = time.time()

        # 1) Decompose
        sub_qs = self._decompose(question, traj)
        traj.extra["n_sub_questions"] = len(sub_qs)
        traj.extra["sub_questions"] = [sq.model_dump() for sq in sub_qs]

        # 2-4) Per-sub-Q: retrieve → build GSW → answer
        gsw_snapshot: list[dict[str, Any]] = []
        sub_pairs: list[tuple[SubQuestion, str]] = []
        for sq in sub_qs:
            hits = self._retrieve(sq, traj)
            gsw = self._build_gsw(sq, hits, traj)
            answer = self._answer_sub(sq, gsw, hits, traj)
            gsw_snapshot.append(
                {
                    "sub_id": sq.sub_id,
                    "question": sq.question,
                    "retrieved_titles": [h.chunk.title for h in hits],
                    "entities": gsw.entities,
                    "triples": [t.model_dump() for t in gsw.triples],
                    "answer": answer,
                }
            )
            sub_pairs.append((sq, answer))

        traj.extra["gsw_snapshot"] = gsw_snapshot

        # 5) Aggregate
        final = self._aggregate(question, sub_pairs, traj)
        traj.final_answer = final
        traj.reasoning = json.dumps(gsw_snapshot)
        traj.turns = len(sub_qs) + 1
        traj.wall_time_s = round(time.time() - start, 3)
        traj.extra["stopped_reason"] = "finished"
        traj.hidden_reasoning = "\n".join(traj.extra.pop("_reasoning_chunks", []))
        return traj
