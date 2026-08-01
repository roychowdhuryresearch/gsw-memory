#!/usr/bin/env python3
"""Run one real decomposer -> dual retrieval -> reranker -> RICR example."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Sequence


DEFAULT_QUESTION_ID = "246a2cd60bda11eba7f7acde48001122"


def release_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def select_question(package, question_id: str) -> dict:
    for question in package.questions("public"):
        if question["question_id"] == question_id:
            return question
    raise KeyError(f"Public question not found: {question_id}")


def format_qa(row: dict) -> str:
    answers = "; ".join(row.get("answer_names", []))
    roles = "; ".join(row.get("answer_role_states", []))
    return f"Question: {row['question']}\nAnswer: {answers}\nRoles: {roles}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "release"
        / "panini_2wiki_100",
    )
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument(
        "--decomposer-model",
        default="yigitturali/GSW-QA-Decomposer-Qwen3-4B",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
    )
    parser.add_argument(
        "--reranker-model",
        default="Qwen/Qwen3-Reranker-8B",
    )
    parser.add_argument("--decomposer-device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--candidates-per-hop", type=int, default=15)
    parser.add_argument("--retrieval-pool", type=int, default=60)
    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=0.5,
        help=(
            "Weight for reciprocal retrieval rank; the remainder weights the "
            "Qwen reranker probability."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "e2e_results"
        / "compositional_example.json",
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))

    from panini_course import (
        BM25Index,
        Candidate,
        CoursePackage,
        DenseIndex,
        DualRetriever,
        QueryEmbeddingStore,
        run_linear_ricr,
    )
    from panini_course.metrics import exact_match, token_f1
    from panini_course.qwen_models import (
        QwenDecomposer,
        QwenQueryEncoder,
        QwenReranker,
    )

    package = CoursePackage(args.package)
    question = select_question(package, args.question_id)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    decomposer = QwenDecomposer(
        args.decomposer_model,
        package.root / "models" / "decomposition_prompt.txt",
        quantized=False,
        device_map=args.decomposer_device,
    )
    plan = decomposer.decompose(question["question"])
    timings["decomposition_seconds"] = time.perf_counter() - started
    del decomposer
    release_memory()

    entity_index = BM25Index.load(
        package.root / "indices" / "entity_bm25.joblib",
        package.root / "indices" / "entity_ids.json",
        source="entity_bm25",
    )
    qa_index = DenseIndex.load(
        package.root / "indices" / "qa_qwen3_8b_ip.faiss",
        package.root / "indices" / "qa_ids.json",
        source="qa_dense",
    )
    qa_rows = package.qa_pairs()
    qa_by_id = {row["qa_uid"]: row for row in qa_rows}
    retriever = DualRetriever(
        entity_index=entity_index,
        qa_index=qa_index,
        entity_rows=package.entities(),
        qa_rows=qa_rows,
    )
    query_store = QueryEmbeddingStore.load(
        package.root / "embeddings" / "query_embeddings.npy",
        package.root / "embeddings" / "query_ids.json",
        package.root / "embeddings" / "queries.jsonl",
    )

    encoder = QwenQueryEncoder(
        args.embedding_model,
        quantized=False,
        device_map=args.embedding_device,
    )
    reranker = QwenReranker(
        args.reranker_model,
        quantized=False,
        device_map=args.reranker_device,
    )
    query_cache: dict[str, object] = {}
    retrieval_trace: list[dict] = []

    def query_vector(text: str):
        if text in query_cache:
            return query_cache[text]
        try:
            vector = query_store.get(text)
            source = "supplied"
        except KeyError:
            vector = encoder.encode([text])[0]
            source = "runtime_qwen"
        query_cache[text] = vector
        retrieval_trace.append(
            {"query": text, "query_embedding_source": source}
        )
        return vector

    def retrieve_and_score(text: str, top_k: int) -> Sequence[Candidate]:
        started_hop = time.perf_counter()
        hits = retriever.search(
            text,
            query_vector=query_vector(text),
            entity_top_k=20,
            qa_top_k=20,
            fused_top_k=args.retrieval_pool,
        )
        rows = [qa_by_id[hit.item_id] for hit in hits]
        documents = [format_qa(row) for row in rows]
        scores = reranker.score(text, documents, batch_size=8)
        scored_rows = []
        for hit, row, reranker_score in zip(hits, rows, scores):
            retrieval_prior = 1.0 / hit.rank
            hybrid_score = (
                args.retrieval_weight * retrieval_prior
                + (1.0 - args.retrieval_weight) * reranker_score
            )
            scored_rows.append(
                (row, reranker_score, retrieval_prior, hybrid_score)
            )
        ranked = sorted(
            scored_rows,
            key=lambda item: (-item[3], item[0]["qa_uid"]),
        )
        candidates: list[Candidate] = []
        for row, reranker_score, retrieval_prior, hybrid_score in ranked:
            for answer in row.get("answer_names", []):
                candidates.append(
                    Candidate(
                        qa_uid=row["qa_uid"],
                        answer=answer,
                        score=float(hybrid_score),
                        question=row["question"],
                        metadata={
                            "search_text": row["search_text"],
                            "reranker_score": float(reranker_score),
                            "retrieval_prior": retrieval_prior,
                        },
                    )
                )
        trace = retrieval_trace[-1]
        trace.update(
            {
                "retrieved_qa_count": len(rows),
                "reranked_candidate_count": len(candidates),
                "top_candidates": [
                    {
                        "qa_uid": candidate.qa_uid,
                        "question": candidate.question,
                        "answer": candidate.answer,
                        "score": candidate.score,
                        "reranker_score": candidate.metadata[
                            "reranker_score"
                        ],
                        "retrieval_prior": candidate.metadata[
                            "retrieval_prior"
                        ],
                    }
                    for candidate in candidates[:top_k]
                ],
                "seconds": time.perf_counter() - started_hop,
            }
        )
        return candidates[:top_k]

    retrieval_plan = [
        step for step in plan if step.get("requires_retrieval", True)
    ]
    started = time.perf_counter()
    beams = run_linear_ricr(
        retrieval_plan,
        retrieve_and_score,
        beam_width=args.beam_width,
        candidates_per_hop=args.candidates_per_hop,
    )
    timings["retrieval_and_ricr_seconds"] = time.perf_counter() - started

    prediction = beams[0].current_answer if beams else ""
    result = {
        "question_id": question["question_id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "prediction": prediction,
        "exact_match": exact_match(prediction, [question["answer"]]),
        "token_f1": token_f1(prediction, [question["answer"]]),
        "models": {
            "decomposer": args.decomposer_model,
            "embedding": args.embedding_model,
            "reranker": args.reranker_model,
        },
        "scoring": {
            "retrieval_weight": args.retrieval_weight,
            "reranker_weight": 1.0 - args.retrieval_weight,
            "retrieval_prior": "reciprocal_rank",
        },
        "decomposition": plan,
        "beams": [
            {
                "score": beam.score,
                "answers_by_step": dict(beam.answers_by_step),
                "steps": [
                    {
                        "qa_uid": step.qa_uid,
                        "question": step.question,
                        "answer": step.answer,
                        "score": step.score,
                    }
                    for step in beam.steps
                ],
            }
            for beam in beams
        ],
        "retrieval_trace": retrieval_trace,
        "timings": timings,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["exact_match"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
