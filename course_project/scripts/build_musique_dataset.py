#!/usr/bin/env python3
"""Build the deterministic 100-question MuSiQue course-project package.

The package mirrors the 2Wiki release contract while stratifying by reasoning
hop count. Each question keeps every supporting paragraph and a deterministic
sample of distractors, capped at ten context documents for Colab-scale use.
Corpus embeddings are exported by the separate export_embeddings.py stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_dataset import (
    artifact_hashes,
    flatten_gsw,
    gsw_paths_for_document,
    named_nodes,
    normalize_text,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)


HOP_QUOTAS = {2: 50, 3: 30, 4: 20}
PUBLIC_QUOTAS = {2: 40, 3: 24, 4: 16}


def corpus_pair(document: Mapping[str, Any]) -> tuple[str, str]:
    return str(document["title"]), str(document["text"])


def paragraph_pair(paragraph: Mapping[str, Any]) -> tuple[str, str]:
    return str(paragraph["title"]), str(paragraph["paragraph_text"])


def question_hops(question: Mapping[str, Any]) -> int:
    return len(question.get("question_decomposition", []))


def selected_paragraph_indices(
    question: Mapping[str, Any],
    *,
    context_size: int,
    seed: int,
) -> list[int]:
    paragraphs = list(question["paragraphs"])
    support = {
        int(item["paragraph_support_idx"])
        for item in question["question_decomposition"]
    }
    support.update(
        int(paragraph["idx"])
        for paragraph in paragraphs
        if paragraph.get("is_supporting")
    )
    if len(support) > context_size:
        raise ValueError(
            f"Question {question['id']} has {len(support)} supporting "
            f"paragraphs but context size is {context_size}"
        )

    distractors = sorted(
        int(paragraph["idx"])
        for paragraph in paragraphs
        if int(paragraph["idx"]) not in support
    )
    stable_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}:{question['id']}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = random.Random(stable_seed)
    needed = min(context_size - len(support), len(distractors))
    chosen = support | set(rng.sample(distractors, needed))
    return sorted(chosen)


def replace_musique_references(text: str) -> str:
    return re.sub(r"#(\d+)", r"<ENTITY_Q\1>", text)


def decomposition_rows(question: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "question": replace_musique_references(str(item["question"])),
            "requires_retrieval": True,
        }
        for item in question["question_decomposition"]
    ]


def question_record(
    question: Mapping[str, Any],
    *,
    context_indices: Sequence[int],
    pair_to_corpus_index: Mapping[tuple[str, str], int],
    include_gold: bool,
) -> dict[str, Any]:
    paragraph_by_index = {
        int(paragraph["idx"]): paragraph
        for paragraph in question["paragraphs"]
    }
    record: dict[str, Any] = {
        "question_id": str(question["id"]),
        "type": f"{question_hops(question)}hop",
        "hop_count": question_hops(question),
        "question": str(question["question"]),
        "context_document_ids": [
            f"doc_{pair_to_corpus_index[paragraph_pair(paragraph_by_index[i])]}"
            for i in context_indices
        ],
    }
    if include_gold:
        evidence = []
        support_document_ids = []
        for step, item in enumerate(
            question["question_decomposition"], start=1
        ):
            paragraph = paragraph_by_index[int(item["paragraph_support_idx"])]
            document_id = (
                f"doc_{pair_to_corpus_index[paragraph_pair(paragraph)]}"
            )
            support_document_ids.append(document_id)
            evidence.append(
                {
                    "step": step,
                    "question": str(item["question"]),
                    "answer": str(item["answer"]),
                    "document_id": document_id,
                }
            )
        record.update(
            {
                "answer": question.get("answer"),
                "answer_aliases": question.get("answer_aliases", []),
                "supporting_document_ids": list(
                    dict.fromkeys(support_document_ids)
                ),
                "evidences": evidence,
            }
        )
    return record


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    questions_path = args.questions.resolve()
    corpus_path = args.corpus.resolve()
    gsw_root = args.gsw_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory."
        )

    questions = read_json(questions_path)
    corpus = read_json(corpus_path)
    if not isinstance(questions, list) or not isinstance(corpus, list):
        raise TypeError("Questions and corpus inputs must be JSON lists")

    pair_to_corpus_index: dict[tuple[str, str], int] = {}
    for index, document in enumerate(corpus):
        pair = corpus_pair(document)
        if pair in pair_to_corpus_index:
            raise ValueError(f"Duplicate corpus title/text pair at row {index}")
        pair_to_corpus_index[pair] = index

    answer_value_cache: dict[int, set[str]] = {}

    def document_answer_values(corpus_index: int) -> set[str]:
        if corpus_index not in answer_value_cache:
            values: set[str] = set()
            document_id = f"doc_{corpus_index}"
            for gsw_path in gsw_paths_for_document(gsw_root, document_id):
                gsw = read_json(gsw_path)
                values.update(
                    normalize_text(node["name"])
                    for node in named_nodes(gsw).values()
                    if node.get("name")
                )
            answer_value_cache[corpus_index] = values
        return answer_value_cache[corpus_index]

    context_cache: dict[str, list[int]] = {}

    def context_for(question: Mapping[str, Any]) -> list[int]:
        question_id = str(question["id"])
        if question_id not in context_cache:
            context_cache[question_id] = selected_paragraph_indices(
                question,
                context_size=args.context_size,
                seed=args.seed,
            )
        return context_cache[question_id]

    def eligible(question: Mapping[str, Any]) -> bool:
        hops = question_hops(question)
        if hops not in HOP_QUOTAS or not question.get("answerable", True):
            return False
        paragraph_by_index = {
            int(paragraph["idx"]): paragraph
            for paragraph in question.get("paragraphs", [])
        }
        try:
            chosen = context_for(question)
            corpus_indices = [
                pair_to_corpus_index[paragraph_pair(paragraph_by_index[index])]
                for index in chosen
            ]
        except (KeyError, ValueError):
            return False
        if any(
            not gsw_paths_for_document(gsw_root, f"doc_{corpus_index}")
            for corpus_index in corpus_indices
        ):
            return False
        support_indices = {
            int(item["paragraph_support_idx"])
            for item in question["question_decomposition"]
        }
        if not support_indices.issubset(chosen):
            return False
        if args.require_gold_evidence_coverage:
            available_values = set().union(
                *(document_answer_values(index) for index in corpus_indices)
            )
            gold_values = [
                normalize_text(item["answer"])
                for item in question["question_decomposition"]
            ]
            if any(value not in available_values for value in gold_values):
                return False
        return True

    selected: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    eligibility_counts: dict[int, int] = {}
    for hops, quota in HOP_QUOTAS.items():
        candidates = sorted(
            (
                dict(question)
                for question in questions
                if question_hops(question) == hops and eligible(question)
            ),
            key=lambda question: str(question["id"]),
        )
        eligibility_counts[hops] = len(candidates)
        if len(candidates) < quota:
            raise ValueError(
                f"Only {len(candidates)} eligible {hops}-hop questions; "
                f"{quota} requested"
            )
        selected.extend(rng.sample(candidates, quota))

    public: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for hops in HOP_QUOTAS:
        group = [
            question for question in selected
            if question_hops(question) == hops
        ]
        public.extend(group[: PUBLIC_QUOTAS[hops]])
        held_out.extend(group[PUBLIC_QUOTAS[hops] :])

    selected_corpus_indices = sorted(
        {
            pair_to_corpus_index[
                paragraph_pair(
                    {
                        int(paragraph["idx"]): paragraph
                        for paragraph in question["paragraphs"]
                    }[paragraph_index]
                )
            ]
            for question in selected
            for paragraph_index in context_for(question)
        }
    )

    output.mkdir(parents=True)
    documents = [
        {
            "document_id": f"doc_{index}",
            "source_corpus_index": index,
            "title": corpus[index]["title"],
            "text": corpus[index]["text"],
        }
        for index in selected_corpus_indices
    ]
    write_jsonl(output / "documents.jsonl", documents)

    def make_record(
        question: Mapping[str, Any], include_gold: bool
    ) -> dict[str, Any]:
        return question_record(
            question,
            context_indices=context_for(question),
            pair_to_corpus_index=pair_to_corpus_index,
            include_gold=include_gold,
        )

    public_records = [make_record(question, True) for question in public]
    held_out_records = [make_record(question, False) for question in held_out]
    instructor_records = [make_record(question, True) for question in selected]
    write_jsonl(output / "questions" / "public.jsonl", public_records)
    write_jsonl(output / "questions" / "held_out.jsonl", held_out_records)
    write_jsonl(
        output / "instructor" / "questions_with_gold.jsonl",
        instructor_records,
    )

    decomposition_by_id = {
        str(question["id"]): decomposition_rows(question)
        for question in selected
    }
    instructor_decomposition_rows = [
        {
            "question_id": record["question_id"],
            "decomposed_questions": decomposition_by_id[
                record["question_id"]
            ],
        }
        for record in instructor_records
    ]
    write_jsonl(
        output / "instructor" / "reviewed_decompositions.jsonl",
        instructor_decomposition_rows,
    )
    public_ids = {record["question_id"] for record in public_records}
    write_jsonl(
        output / "questions" / "decomposition_validation.jsonl",
        (
            row for row in instructor_decomposition_rows
            if row["question_id"] in public_ids
        ),
    )

    all_entity_rows: list[dict[str, Any]] = []
    all_qa_rows: list[dict[str, Any]] = []
    gsw_count = 0
    for index in selected_corpus_indices:
        document_id = f"doc_{index}"
        target_directory = output / "gsws" / document_id
        target_directory.mkdir(parents=True, exist_ok=True)
        for source_path in gsw_paths_for_document(gsw_root, document_id):
            target_path = target_directory / source_path.name
            shutil.copy2(source_path, target_path)
            gsw = read_json(source_path)
            entity_rows, qa_rows = flatten_gsw(
                document_id, source_path.name, gsw
            )
            all_entity_rows.extend(entity_rows)
            all_qa_rows.extend(qa_rows)
            gsw_count += 1

    write_jsonl(output / "metadata" / "entities.jsonl", all_entity_rows)
    write_jsonl(output / "metadata" / "qa_pairs.jsonl", all_qa_rows)
    write_json(
        output / "metadata" / "embedding_contract.json",
        {
            "version": 1,
            "description": (
                "Embeddings are supplied by instructors and are not generated "
                "by the course-project code."
            ),
            "entity_embeddings": {
                "matrix": "embeddings/entity_embeddings.npy",
                "ids": "embeddings/entity_ids.json",
                "metadata": "metadata/entities.jsonl",
                "metadata_id_field": "entity_uid",
                "expected_rows": len(all_entity_rows),
                "dtype": "float16",
            },
            "qa_embeddings": {
                "matrix": "embeddings/qa_embeddings.npy",
                "ids": "embeddings/qa_ids.json",
                "metadata": "metadata/qa_pairs.jsonl",
                "metadata_id_field": "qa_uid",
                "expected_rows": len(all_qa_rows),
                "dtype": "float16",
            },
            "query_embeddings": {
                "matrix": "embeddings/query_embeddings.npy",
                "ids": "embeddings/query_ids.json",
                "queries": "embeddings/queries.jsonl",
                "query_id_field": "query_id",
                "query_text_field": "text",
                "dtype": "float16",
            },
        },
    )

    manifest = {
        "dataset": "MuSiQue",
        "license": "CC BY 4.0",
        "seed": args.seed,
        "context_documents_per_question": args.context_size,
        "hop_quotas": {str(key): value for key, value in HOP_QUOTAS.items()},
        "public_hop_quotas": {
            str(key): value for key, value in PUBLIC_QUOTAS.items()
        },
        "require_gold_evidence_coverage": args.require_gold_evidence_coverage,
        "eligible_by_hop": {
            str(key): value for key, value in eligibility_counts.items()
        },
        "counts": {
            "questions_total": len(selected),
            "questions_public": len(public_records),
            "questions_held_out": len(held_out_records),
            "documents": len(documents),
            "gsws": gsw_count,
            "entities": len(all_entity_rows),
            "qa_pairs": len(all_qa_rows),
        },
        "question_type_counts": dict(
            sorted(
                Counter(
                    f"{question_hops(question)}hop"
                    for question in selected
                ).items()
            )
        ),
        "source_files": {
            "questions": questions_path.name,
            "questions_sha256": sha256_file(questions_path),
            "corpus": corpus_path.name,
            "corpus_sha256": sha256_file(corpus_path),
            "gsw_root_label": gsw_root.name,
        },
        "artifacts": artifact_hashes(output),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--gsw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=232)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument(
        "--require-gold-evidence-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.context_size < 4:
        parser.error("--context-size must be at least 4")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build_dataset(parse_args(argv))
    print(json.dumps(
        {
            "counts": manifest["counts"],
            "eligible_by_hop": manifest["eligible_by_hop"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
