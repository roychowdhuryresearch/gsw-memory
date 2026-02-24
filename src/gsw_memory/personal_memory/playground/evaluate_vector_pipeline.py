#!/usr/bin/env python3
"""
Evaluation script for vector pipeline personal memory QA.

Two modes:
  - baseline: single-pass retrieval (entity + QA indices) + batched answer generation via curator
  - agentic:  multi-turn tool-calling agent with 4 tools (search_entities, search_questions,
              get_entity_detail, search_spacetime)

Usage:
    python src/gsw_memory/personal_memory/playground/evaluate_vector_pipeline.py \
        --saved-state logs/pipeline_inspector/2026-02-23_154125_conv-26.json \
        --mode both --max-questions 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parents[3]  # gsw-memory/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

# Imports from vector_pipeline (same directory)
from vector_pipeline import (
    EmbedFn,
    VectorQAAgent,
    VectorStoreBuilder,
    make_openai_embed_fn,
    make_vllm_embed_fn,
)

# Imports from gsw_memory
from gsw_memory.personal_memory.data_ingestion.locomo import LoCoMoLoader
from gsw_memory.evaluation.hipporag_eval import calculate_exact_match, calculate_f1_score

# Curator
from bespokelabs import curator

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

CAT_LABELS = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-ended",
    5: "Adversarial",
}

_ABSTENTION_PHRASES = [
    "i don't know", "i do not know", "no answer", "unanswerable",
    "cannot answer", "can't answer", "not mentioned", "no information",
    "not enough information", "cannot be determined", "unknown",
    "not available", "n/a", "none",
]


def is_abstention(text: str) -> bool:
    """Check if a response is an abstention / refusal to answer."""
    normalized = text.strip().lower().rstrip(".")
    return any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------


class LLMJudge(curator.LLM):
    """Curator LLM for batched LLM-as-judge scoring (0 / 0.5 / 1)."""

    return_completions_object = True

    def prompt(self, input_data):
        system = (
            "You are an expert judge evaluating the accuracy of AI-generated answers "
            "against a ground truth answer. You must respond with a JSON object only."
        )

        cat = int(input_data["category"])
        gold = input_data["gold"]
        predicted = input_data["predicted"]

        if cat == 5 and not gold:
            # Adversarial: correct behavior is to refuse
            user = (
                f"Question: {input_data['question']}\n"
                f"Ground truth: This is an adversarial/trick question. The correct "
                f"response is to refuse to answer or say 'I don't know'.\n"
                f"AI answer: {predicted}\n\n"
                "Score the AI answer:\n"
                "- 1 if the AI correctly refused to answer or said it doesn't know\n"
                "- 0.5 if the AI expressed uncertainty but still attempted an answer\n"
                "- 0 if the AI confidently gave a specific (wrong) answer\n\n"
                'Respond with JSON: {"score": <0|0.5|1>, "reason": "<brief explanation>"}'
            )
        else:
            user = (
                f"Question: {input_data['question']}\n"
                f"Ground truth: {gold}\n"
                f"AI answer: {predicted}\n\n"
                "Score the AI answer:\n"
                "- 1 if the AI answer is correct (same meaning as ground truth, "
                "phrasing differences are OK)\n"
                "- 0.5 if the AI answer is partially correct (captures some but not "
                "all key information, or includes correct info mixed with errors)\n"
                "- 0 if the AI answer is wrong, irrelevant, or says 'I don't know' "
                "when the ground truth has an answer\n\n"
                'Respond with JSON: {"score": <0|0.5|1>, "reason": "<brief explanation>"}'
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse(self, input_data, response):
        raw = response["choices"][0]["message"]["content"].strip()

        # Extract JSON
        score = 0.0
        reason = ""
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    score = float(parsed.get("score", 0))
                    reason = parsed.get("reason", "")
                except (json.JSONDecodeError, ValueError):
                    reason = f"parse_error: {raw[:200]}"
            else:
                reason = f"no_json: {raw[:200]}"

        # Clamp to valid values
        if score not in (0.0, 0.5, 1.0):
            score = round(score * 2) / 2  # snap to nearest 0/0.5/1

        return [{
            "question": input_data["question"],
            "predicted": input_data["predicted"],
            "gold": input_data["gold"],
            "category": input_data["category"],
            "llm_score": score,
            "llm_reason": reason,
        }]


def llm_judge_rescore(
    per_question: List[Dict[str, Any]],
    judge_model: str,
) -> None:
    """Run LLM judge on all per-question results, writing llm_score/llm_reason in-place."""
    console.print(f"[cyan]Running LLM judge ({judge_model}) on {len(per_question)} questions...[/cyan]")

    inputs = []
    for q in per_question:
        inputs.append({
            "question": q["question"],
            "predicted": q["predicted"],
            "gold": str(q["gold"]),
            "category": str(q["category"]),
        })

    start = time.time()
    judge = LLMJudge(
        model_name=judge_model,
        generation_params={"temperature": 0},
    )
    results = judge(inputs)
    elapsed = time.time() - start
    console.print(f"[green]LLM judge completed in {elapsed:.1f}s[/green]")

    # Write scores back into per_question
    for q, judged in zip(per_question, results.dataset):
        q["llm_score"] = judged["llm_score"]
        q["llm_reason"] = judged["llm_reason"]


# ---------------------------------------------------------------------------
# Index caching
# ---------------------------------------------------------------------------


def build_or_load_index(
    gsw_dicts: Dict[str, list],
    session_chunks: Dict[str, list],
    embed_fn: EmbedFn,
    cache_dir: Path,
) -> VectorStoreBuilder:
    """Build FAISS index from GSWs, or load from cache."""
    builder = VectorStoreBuilder(embed_fn)

    if (cache_dir / "entity_index.faiss").exists():
        console.print(f"[cyan]Loading cached index from {cache_dir}[/cyan]")
        builder.load(cache_dir)
        return builder

    console.print("[cyan]Building FAISS index...[/cyan]")
    builder.build(gsw_dicts, session_chunks)
    cache_dir.mkdir(parents=True, exist_ok=True)
    builder.save(cache_dir)
    console.print(f"[green]Index saved to {cache_dir}[/green]")
    return builder


# ---------------------------------------------------------------------------
# Conversation matching
# ---------------------------------------------------------------------------


def match_qa_pairs(
    gsw_conv_ids: set,
    conversations: list,
    categories: List[int],
    max_questions: int | None = None,
) -> List[Tuple[str, Any]]:
    """Return (conv_id, QAPair) tuples for conversations present in the saved state."""
    qa_items: List[Tuple[str, Any]] = []
    for conv in conversations:
        if conv.sample_id in gsw_conv_ids:
            for qa in conv.qa_pairs:
                if qa.category in categories:
                    qa_items.append((conv.sample_id, qa))
    if max_questions:
        qa_items = qa_items[:max_questions]
    return qa_items


# ---------------------------------------------------------------------------
# Baseline text-chunk RAG
# ---------------------------------------------------------------------------

CHUNK_TASK = "Given a conversation excerpt, create an embedding for semantic retrieval."


class BaselineRetriever:
    """Text-chunk RAG: embeds raw conversation chunks into FAISS for retrieval."""

    def __init__(self, embed_fn: EmbedFn):
        self._embed_fn = embed_fn
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.index: faiss.IndexFlatIP | None = None

    def build(self, session_chunks: Dict[str, List[List[str]]]) -> None:
        """Embed raw conversation chunks and build FAISS index."""
        for conv_id, sessions in session_chunks.items():
            for sess_idx, chunks in enumerate(sessions):
                for chunk_idx, text in enumerate(chunks):
                    chunk_id = f"{conv_id}::s{sess_idx}_c{chunk_idx}"
                    self.chunk_texts.append(text)
                    self.chunk_metadata.append({
                        "chunk_id": chunk_id,
                        "session_idx": sess_idx,
                        "conv_id": conv_id,
                    })

        console.print(f"[cyan]Embedding {len(self.chunk_texts)} text chunks...[/cyan]")
        embs = self._embed_fn(self.chunk_texts, CHUNK_TASK)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieve top-K conversation chunks for a query."""
        query_emb = self._embed_fn([query], CHUNK_TASK)
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
        parts: List[str] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                meta = self.chunk_metadata[idx]
                parts.append(
                    f"[{meta['chunk_id']} | score={score:.3f}]\n{self.chunk_texts[idx]}"
                )
        return "\n\n---\n\n".join(parts)

    def save(self, directory: Path) -> None:
        """Save baseline chunk index to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "baseline_chunk_index.faiss"))
        (directory / "baseline_chunk_metadata.json").write_text(
            json.dumps({"metadata": self.chunk_metadata, "texts": self.chunk_texts}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> bool:
        """Load baseline chunk index from disk. Returns True on success."""
        path = directory / "baseline_chunk_index.faiss"
        if not path.exists():
            return False
        self.index = faiss.read_index(str(path))
        data = json.loads((directory / "baseline_chunk_metadata.json").read_text(encoding="utf-8"))
        self.chunk_metadata = data["metadata"]
        self.chunk_texts = data["texts"]
        return True


def build_or_load_baseline(
    session_chunks: Dict[str, list],
    embed_fn: EmbedFn,
    cache_dir: Path,
) -> BaselineRetriever:
    """Build baseline chunk index, or load from cache."""
    retriever = BaselineRetriever(embed_fn)
    if (cache_dir / "baseline_chunk_index.faiss").exists():
        console.print(f"[cyan]Loading cached baseline index from {cache_dir}[/cyan]")
        retriever.load(cache_dir)
        return retriever

    retriever.build(session_chunks)
    retriever.save(cache_dir)
    console.print(f"[green]Baseline index saved to {cache_dir}[/green]")
    return retriever


class BaselineChunkQA(curator.LLM):
    """Curator LLM for batched baseline QA over raw conversation chunks."""

    return_completions_object = True

    def prompt(self, input_data):
        system = (
            "You answer questions about a person's conversational memories using "
            "retrieved conversation excerpts. Each excerpt is from a specific session "
            "with a date/time header.\n\n"
            "Your response starts after 'Thought: ', where you reason step-by-step. "
            "Conclude with 'Answer: ' to present a concise, definitive answer.\n\n"
            "If you cannot answer from the excerpts, say 'Answer: I don't know'."
        )
        user = f"{input_data['context']}\n\nQuestion: {input_data['question']}\nThought: "
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse(self, input_data, response):
        answer_text = response["choices"][0]["message"]["content"]

        # Extract answer after "Answer: "
        if "Answer: " in answer_text:
            final_answer = answer_text.split("Answer: ")[-1].strip()
            if final_answer.endswith(".") and final_answer[:-1].replace(",", "").replace(" ", "").isdigit():
                final_answer = final_answer[:-1]
        else:
            final_answer = answer_text.strip()

        return [{
            "question": input_data["question"],
            "predicted": final_answer,
            "full_response": answer_text,
            "gold": input_data["gold"],
            "category": input_data["category"],
            "conv_id": input_data["conv_id"],
        }]


def run_baseline_eval(
    retriever: BaselineRetriever,
    qa_items: List[Tuple[str, Any]],
    model_name: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Run baseline text-chunk RAG evaluation using curator for batch QA."""

    # Step 1: Retrieve raw conversation chunks for each question
    console.print(f"[cyan]Retrieving chunks for {len(qa_items)} questions (top_k={top_k})...[/cyan]")
    inputs = []
    for conv_id, qa in qa_items:
        ctx = retriever.retrieve(qa.question, top_k=top_k)
        inputs.append({
            "question": qa.question,
            "context": ctx,
            "gold": str(qa.answer or ""),
            "category": str(qa.category),
            "conv_id": conv_id,
        })

    # Step 2: Batch answer generation via curator
    console.print(f"[cyan]Generating answers via curator ({model_name})...[/cyan]")
    start = time.time()
    qa_gen = BaselineChunkQA(
        model_name=model_name,
        generation_params={"temperature": 0},
    )
    dataset = qa_gen(inputs)
    elapsed = time.time() - start
    console.print(f"[green]Baseline answers generated in {elapsed:.1f}s[/green]")

    # Step 3: Compute per-question metrics
    results = []
    for item in dataset.dataset:
        gold = [item["gold"]] if item["gold"] else []
        predicted = item["predicted"]
        cat = int(item["category"])
        if not gold and cat == 5:
            em = 1.0 if is_abstention(predicted) else 0.0
            f1 = em
        else:
            em = calculate_exact_match(gold, predicted) if gold else 0.0
            f1 = calculate_f1_score(gold, predicted) if gold else 0.0
        results.append({
            "question": item["question"],
            "predicted": predicted,
            "gold": item["gold"],
            "category": int(item["category"]),
            "conv_id": item["conv_id"],
            "EM": em,
            "F1": f1,
            "full_response": item.get("full_response", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Agentic RAG
# ---------------------------------------------------------------------------


def run_agentic_eval(
    builder: VectorStoreBuilder,
    qa_items: List[Tuple[str, Any]],
    model_name: str,
) -> List[Dict[str, Any]]:
    """Run agentic 4-tool evaluation (sequential, multi-turn)."""
    agent = VectorQAAgent(builder, model_name=model_name)
    results = []

    for i, (conv_id, qa) in enumerate(qa_items):
        console.print(f"[dim]Agentic Q {i + 1}/{len(qa_items)}: {qa.question[:80]}[/dim]")
        try:
            result = agent.answer(qa.question)
            predicted = result["answer"]
        except Exception as e:
            logger.warning("Agentic QA failed for %r: %s", qa.question[:60], e)
            predicted = "Error"
            result = {"reasoning": str(e), "speaker_id": None, "evidence_turn_ids": [], "trace": []}

        gold = [str(qa.answer)] if qa.answer is not None else []
        if not gold and qa.category == 5:
            em = 1.0 if is_abstention(predicted) else 0.0
            f1 = em
        else:
            em = calculate_exact_match(gold, predicted) if gold else 0.0
            f1 = calculate_f1_score(gold, predicted) if gold else 0.0

        results.append({
            "question": qa.question,
            "predicted": predicted,
            "gold": str(qa.answer) if qa.answer is not None else "",
            "category": qa.category,
            "conv_id": conv_id,
            "EM": em,
            "F1": f1,
            "reasoning": result.get("reasoning", ""),
            "speaker_id": result.get("speaker_id"),
            "evidence_turn_ids": result.get("evidence_turn_ids", []),
            "trace": result.get("trace", []),
        })

    return results


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall and per-category EM/F1 (and LLM score if present)."""
    if not results:
        return {"overall": {"EM": 0, "F1": 0, "count": 0}, "per_category": {}, "per_question": []}

    has_llm = "llm_score" in results[0]

    overall_em = np.mean([r["EM"] for r in results])
    overall_f1 = np.mean([r["F1"] for r in results])
    overall: Dict[str, Any] = {"EM": float(overall_em), "F1": float(overall_f1), "count": len(results)}
    if has_llm:
        overall["LLM"] = float(np.mean([r["llm_score"] for r in results]))

    per_cat: Dict[int, List[Dict]] = {}
    for r in results:
        per_cat.setdefault(r["category"], []).append(r)

    per_category = {}
    for cat, items in sorted(per_cat.items()):
        cat_data: Dict[str, Any] = {
            "name": CAT_LABELS.get(cat, f"Cat-{cat}"),
            "EM": float(np.mean([r["EM"] for r in items])),
            "F1": float(np.mean([r["F1"] for r in items])),
            "count": len(items),
        }
        if has_llm:
            cat_data["LLM"] = float(np.mean([r["llm_score"] for r in items]))
        per_category[str(cat)] = cat_data

    return {
        "overall": overall,
        "per_category": per_category,
        "per_question": results,
    }


def print_summary(output: Dict[str, Any]) -> None:
    """Print a summary table to stdout."""
    for mode in ("baseline", "agentic"):
        data = output.get(mode)
        if not data:
            continue

        has_llm = "LLM" in data.get("overall", {})

        table = Table(title=f"{mode.upper()} Results")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("EM", justify="right")
        table.add_column("F1", justify="right")
        if has_llm:
            table.add_column("LLM", justify="right")

        for cat_key, cat_data in sorted(data["per_category"].items()):
            row = [
                cat_data["name"],
                str(cat_data["count"]),
                f"{cat_data['EM']:.4f}",
                f"{cat_data['F1']:.4f}",
            ]
            if has_llm:
                row.append(f"{cat_data.get('LLM', 0):.4f}")
            table.add_row(*row)

        ov = data["overall"]
        table.add_section()
        ov_row = ["Overall", str(ov["count"]), f"{ov['EM']:.4f}", f"{ov['F1']:.4f}"]
        if has_llm:
            ov_row.append(f"{ov.get('LLM', 0):.4f}")
        table.add_row(*ov_row, style="bold")

        console.print(table)
        console.print()


# ---------------------------------------------------------------------------
# Rescore existing results
# ---------------------------------------------------------------------------


def rescore_results(data: Dict[str, Any], judge_model: str | None = None) -> Dict[str, Any]:
    """Re-score per_question entries with current scoring logic + optional LLM judge."""
    per_question = data.get("per_question", [])
    for q in per_question:
        gold = [q["gold"]] if q["gold"] else []
        predicted = q["predicted"]
        cat = q["category"]
        if not gold and cat == 5:
            em = 1.0 if is_abstention(predicted) else 0.0
            f1 = em
        else:
            em = calculate_exact_match(gold, predicted) if gold else 0.0
            f1 = calculate_f1_score(gold, predicted) if gold else 0.0
        q["EM"] = em
        q["F1"] = f1

    if judge_model:
        llm_judge_rescore(per_question, judge_model)

    return compute_metrics(per_question)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vector pipeline QA on LoCoMo")
    parser.add_argument("--saved-state", default=None, help="Path to pipeline_inspector saved state JSON")
    parser.add_argument("--rescore", default=None, help="Path to existing eval JSON to re-score (no LLM calls)")
    parser.add_argument(
        "--locomo-path",
        default=str(_REPO_ROOT / "data" / "personal_memory" / "locomo" / "data" / "locomo10.json"),
        help="Path to LoCoMo JSON",
    )
    parser.add_argument("--output-dir", default="logs/vector_eval", help="Output directory")
    parser.add_argument("--embedding-backend", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--qa-model", default="gpt-4o", help="Model for QA generation")
    parser.add_argument("--mode", choices=["agentic", "baseline", "both"], default="both")
    parser.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions for testing")
    parser.add_argument("--baseline-top-k", type=int, default=20, help="Top-K for baseline retrieval")
    parser.add_argument("--cache-dir", default=None, help="Override FAISS cache directory")
    parser.add_argument("--judge-model", default=None, help="LLM model for judge scoring (e.g. gpt-4o-mini)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # --rescore: re-score an existing eval JSON without LLM calls
    if args.rescore:
        rescore_path = Path(args.rescore)
        if not rescore_path.exists():
            console.print(f"[red]Eval file not found: {rescore_path}[/red]")
            sys.exit(1)

        console.print(f"[cyan]Re-scoring: {rescore_path.name}[/cyan]")
        data = json.loads(rescore_path.read_text(encoding="utf-8"))

        output = {"metadata": data.get("metadata", {})}
        output["metadata"]["rescored_from"] = str(rescore_path)
        output["metadata"]["rescore_timestamp"] = datetime.now().isoformat()

        for mode in ("baseline", "agentic"):
            if mode in data:
                output[mode] = rescore_results(data[mode], judge_model=args.judge_model)

        # Save rescored results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rescored.json"
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Rescored results saved to {output_path}[/green]\n")
        print_summary(output)
        return

    if not args.saved_state:
        console.print("[red]--saved-state is required when not using --rescore[/red]")
        sys.exit(1)

    state_path = Path(args.saved_state)
    if not state_path.exists():
        console.print(f"[red]Saved state not found: {state_path}[/red]")
        sys.exit(1)

    # 1. Load saved state
    console.print(f"[cyan]Loading saved state: {state_path.name}[/cyan]")
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    gsw_dicts = saved_state.get("post_spacetime_gsws", {})
    session_chunks = saved_state.get("session_chunks", {})

    if not gsw_dicts:
        console.print("[red]No post_spacetime_gsws in saved state.[/red]")
        sys.exit(1)

    conv_ids = set(gsw_dicts.keys())
    console.print(f"  Conversations: {sorted(conv_ids)}")

    # 2. Embedding backend
    if args.embedding_backend == "vllm":
        embed_fn = make_vllm_embed_fn()
    else:
        embed_fn = make_openai_embed_fn()

    cache_dir = Path(args.cache_dir) if args.cache_dir else state_path.parent / f"{state_path.stem}_index"

    # Build GSW-derived FAISS index only if agentic mode is requested
    builder = None
    entity_count = qa_count = 0
    if args.mode in ("agentic", "both"):
        builder = build_or_load_index(gsw_dicts, session_chunks, embed_fn, cache_dir)
        entity_count = builder.entity_index.ntotal if builder.entity_index else 0
        qa_count = builder.qa_index.ntotal if builder.qa_index else 0
        console.print(f"  GSW Index: {entity_count} entities, {qa_count} QA pairs")

    # 3. Load LoCoMo and match conversations
    locomo_path = Path(args.locomo_path)
    if not locomo_path.exists():
        console.print(f"[red]LoCoMo file not found: {locomo_path}[/red]")
        sys.exit(1)

    conversations = LoCoMoLoader(str(locomo_path)).load()
    qa_items = match_qa_pairs(conv_ids, conversations, args.categories, args.max_questions)
    console.print(f"  Matched {len(qa_items)} QA pairs (categories: {args.categories})")

    if not qa_items:
        console.print("[red]No matching QA pairs found.[/red]")
        sys.exit(1)

    # 4. Run evaluations
    output: Dict[str, Any] = {
        "metadata": {
            "saved_state": str(state_path),
            "locomo_path": str(locomo_path),
            "conv_ids": sorted(conv_ids),
            "embedding_backend": args.embedding_backend,
            "qa_model": args.qa_model,
            "timestamp": datetime.now().isoformat(),
            "index_stats": {"entity_count": entity_count, "qa_count": qa_count},
            "categories": args.categories,
            "max_questions": args.max_questions,
        },
    }

    if args.mode in ("baseline", "both"):
        console.print("\n[bold magenta]Running Baseline Text-Chunk RAG[/bold magenta]")
        retriever = build_or_load_baseline(session_chunks, embed_fn, cache_dir)
        console.print(f"  Baseline index: {retriever.index.ntotal} chunks")
        baseline_results = run_baseline_eval(retriever, qa_items, args.qa_model, args.baseline_top_k)
        output["baseline"] = compute_metrics(baseline_results)

    if args.mode in ("agentic", "both"):
        console.print("\n[bold magenta]Running Agentic RAG (4 tools)[/bold magenta]")
        agentic_results = run_agentic_eval(builder, qa_items, args.qa_model)
        output["agentic"] = compute_metrics(agentic_results)

    # 5. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Strip traces for cleaner JSON (traces can be huge)
    save_output = json.loads(json.dumps(output, default=str))
    for mode_key in ("baseline", "agentic"):
        if mode_key in save_output:
            for q in save_output[mode_key].get("per_question", []):
                q.pop("trace", None)

    output_path.write_text(json.dumps(save_output, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    # 6. Print summary
    console.print()
    print_summary(output)


if __name__ == "__main__":
    main()
