#!/usr/bin/env python3
"""Build the deterministic 100-question Panini course-project dataset.

This script packages questions, documents, GSWs, and flattened search metadata.
It intentionally does not generate embeddings. Instead, it writes an embedding
contract that a separate instructor export must satisfy.
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
from typing import Any, Callable, Iterable, Mapping, Sequence


QUESTION_TYPES = (
    "bridge_comparison",
    "comparison",
    "compositional",
    "inference",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def stable_question_sample(
    questions: Sequence[Mapping[str, Any]],
    *,
    questions_per_type: int,
    seed: int,
    eligible: Callable[[Mapping[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Select an equal, deterministic sample from each supported question type."""

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for question_type in QUESTION_TYPES:
        candidates = sorted(
            (
                dict(question)
                for question in questions
                if question.get("type") == question_type and eligible(question)
            ),
            key=lambda question: str(question["_id"]),
        )
        if len(candidates) < questions_per_type:
            raise ValueError(
                f"Only {len(candidates)} eligible {question_type!r} questions; "
                f"{questions_per_type} requested"
            )
        selected.extend(rng.sample(candidates, questions_per_type))
    return selected


def gsw_paths_for_document(gsw_root: Path, document_id: str) -> list[Path]:
    return sorted((gsw_root / document_id).glob("gsw_*.json"))


def normalize_text(value: Any) -> str:
    """Normalize an entity or answer string for coverage checks."""

    return " ".join(re.findall(r"\w+", str(value).casefold()))


def role_state_text(entity: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for role in entity.get("roles", []):
        role_name = str(role.get("role", "")).strip()
        states = [str(state).strip() for state in role.get("states", []) if state]
        if role_name and states:
            values.append(f"{role_name}: {', '.join(states)}")
        elif role_name:
            values.append(role_name)
    return values


def named_nodes(gsw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return all locally addressable answer nodes keyed by local node ID."""

    nodes: dict[str, dict[str, Any]] = {}
    for node_type, key in (
        ("entity", "entity_nodes"),
        ("space", "space_nodes"),
        ("time", "time_nodes"),
    ):
        for node in gsw.get(key, []):
            local_id = str(node["id"])
            name = (
                node.get("name")
                or node.get("current_name")
                or next(iter(node.get("name_history", {}).values()), None)
                or local_id
            )
            nodes[local_id] = {
                "node_type": node_type,
                "name": str(name),
                "role_states": role_state_text(node),
            }
    return nodes


def flatten_gsw(
    document_id: str,
    gsw_filename: str,
    gsw: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten one GSW into entity-index and QA-index metadata rows."""

    entity_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    nodes = named_nodes(gsw)

    for entity in gsw.get("entity_nodes", []):
        local_id = str(entity["id"])
        roles = role_state_text(entity)
        name = str(entity["name"])
        entity_uid = f"{document_id}::{gsw_filename}::{local_id}"
        entity_rows.append(
            {
                "entity_uid": entity_uid,
                "document_id": document_id,
                "gsw_file": gsw_filename,
                "local_entity_id": local_id,
                "name": name,
                "role_states": roles,
                "search_text": " - ".join(
                    part for part in (name, " | ".join(roles)) if part
                ),
            }
        )

    for verb_phrase in gsw.get("verb_phrase_nodes", []):
        verb_id = str(verb_phrase["id"])
        verb_text = str(verb_phrase.get("phrase", ""))
        for question in verb_phrase.get("questions", []):
            question_id = str(question["id"])
            answer_local_ids = [str(value) for value in question.get("answers", [])]
            answers = [nodes.get(answer_id, {}) for answer_id in answer_local_ids]
            answer_names = [
                str(answer.get("name", answer_id))
                for answer, answer_id in zip(answers, answer_local_ids)
            ]
            answer_role_states = [
                role_state
                for answer in answers
                for role_state in answer.get("role_states", [])
            ]
            qa_uid = (
                f"{document_id}::{gsw_filename}::{verb_id}::{question_id}"
            )
            question_text = str(question.get("text", ""))
            qa_rows.append(
                {
                    "qa_uid": qa_uid,
                    "document_id": document_id,
                    "gsw_file": gsw_filename,
                    "verb_phrase_id": verb_id,
                    "verb_phrase": verb_text,
                    "question_id": question_id,
                    "question": question_text,
                    "answer_local_ids": answer_local_ids,
                    "answer_names": answer_names,
                    "answer_role_states": answer_role_states,
                    "search_text": " ".join(
                        value
                        for value in (
                            question_text,
                            " ".join(answer_names),
                            " ".join(answer_role_states),
                        )
                        if value
                    ),
                }
            )

    return entity_rows, qa_rows


def load_decompositions(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = read_json(path)
    rows = payload.get("per_question_results", [])
    return {
        str(row["question_id"]): list(row.get("decomposed_questions", []))
        for row in rows
        if row.get("question_id")
    }


def public_question_record(
    question: Mapping[str, Any],
    title_to_document_id: Mapping[str, str],
    *,
    include_gold: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "question_id": str(question["_id"]),
        "type": question["type"],
        "question": question["question"],
        "context_document_ids": [
            title_to_document_id[title] for title, _ in question["context"]
        ],
    }
    if include_gold:
        record.update(
            {
                "answer": question.get("answer"),
                "answer_aliases": question.get("answer_aliases", []),
                "supporting_facts": question.get("supporting_facts", []),
                "evidences": question.get("evidences", []),
            }
        )
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(output: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[str(path.relative_to(output))] = sha256_file(path)
    return hashes


def split_selected_questions(
    selected: Sequence[Mapping[str, Any]],
    *,
    public_per_type: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for question_type in QUESTION_TYPES:
        group = [
            dict(question)
            for question in selected
            if question["type"] == question_type
        ]
        public.extend(group[:public_per_type])
        held_out.extend(group[public_per_type:])
    return public, held_out


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
        raise TypeError("Questions and corpus inputs must both be JSON lists")

    title_to_index: dict[str, int] = {}
    duplicate_titles: set[str] = set()
    for index, document in enumerate(corpus):
        title = str(document["title"])
        if title in title_to_index:
            duplicate_titles.add(title)
        title_to_index[title] = index
    if duplicate_titles:
        raise ValueError(
            f"Corpus contains duplicate titles: {sorted(duplicate_titles)[:5]}"
        )

    decomposition_map = load_decompositions(args.decompositions_log)
    answer_value_cache: dict[int, set[str]] = {}

    def document_answer_values(corpus_index: int) -> set[str]:
        if corpus_index in answer_value_cache:
            return answer_value_cache[corpus_index]
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
        return values

    def eligible(question: Mapping[str, Any]) -> bool:
        context_titles = [str(title) for title, _ in question.get("context", [])]
        support_titles = {
            str(title) for title, _ in question.get("supporting_facts", [])
        }
        if len(context_titles) != 10 or not support_titles.issubset(context_titles):
            return False
        for title in context_titles:
            corpus_index = title_to_index.get(title)
            if corpus_index is None:
                return False
            if not gsw_paths_for_document(gsw_root, f"doc_{corpus_index}"):
                return False
        if decomposition_map and str(question["_id"]) not in decomposition_map:
            return False
        if args.require_gold_evidence_coverage:
            available_values = set().union(
                *(
                    document_answer_values(title_to_index[title])
                    for title in context_titles
                )
            )
            gold_values = [
                normalize_text(evidence[2])
                for evidence in question.get("evidences", [])
                if len(evidence) >= 3
            ]
            if any(value not in available_values for value in gold_values):
                return False
        return True

    selected = stable_question_sample(
        questions,
        questions_per_type=args.questions_per_type,
        seed=args.seed,
        eligible=eligible,
    )
    public, held_out = split_selected_questions(
        selected,
        public_per_type=args.public_per_type,
    )

    selected_titles = {
        str(title)
        for question in selected
        for title, _ in question["context"]
    }
    selected_indices = sorted(title_to_index[title] for title in selected_titles)
    title_to_document_id = {
        str(corpus[index]["title"]): f"doc_{index}" for index in selected_indices
    }

    output.mkdir(parents=True)
    documents = [
        {
            "document_id": f"doc_{index}",
            "source_corpus_index": index,
            "title": corpus[index]["title"],
            "text": corpus[index]["text"],
        }
        for index in selected_indices
    ]
    write_jsonl(output / "documents.jsonl", documents)

    public_records = [
        public_question_record(
            question, title_to_document_id, include_gold=True
        )
        for question in public
    ]
    held_out_records = [
        public_question_record(
            question, title_to_document_id, include_gold=False
        )
        for question in held_out
    ]
    instructor_records = [
        public_question_record(
            question, title_to_document_id, include_gold=True
        )
        for question in selected
    ]
    write_jsonl(output / "questions" / "public.jsonl", public_records)
    write_jsonl(output / "questions" / "held_out.jsonl", held_out_records)
    write_jsonl(
        output / "instructor" / "questions_with_gold.jsonl",
        instructor_records,
    )

    if decomposition_map:
        instructor_decomposition_rows = [
            {
                "question_id": record["question_id"],
                "decomposed_questions": decomposition_map[record["question_id"]],
            }
            for record in instructor_records
            if record["question_id"] in decomposition_map
        ]
        write_jsonl(
            output / "instructor" / "reviewed_decompositions.jsonl",
            instructor_decomposition_rows,
        )
        validation_ids = {
            record["question_id"]
            for question_type in QUESTION_TYPES
            for record in [
                public_record
                for public_record in public_records
                if public_record["type"] == question_type
            ][: args.decomposition_validation_per_type]
        }
        validation_decomposition_rows = [
            row
            for row in instructor_decomposition_rows
            if row["question_id"] in validation_ids
        ]
        write_jsonl(
            output / "questions" / "decomposition_validation.jsonl",
            validation_decomposition_rows,
        )

    all_entity_rows: list[dict[str, Any]] = []
    all_qa_rows: list[dict[str, Any]] = []
    gsw_count = 0
    for index in selected_indices:
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
                "coverage": (
                    "Original questions, decomposed questions, and all "
                    "instructor-enumerated answer-instantiated sub-questions."
                ),
            },
        },
    )

    manifest = {
        "dataset": "2WikiMultiHopQA",
        "seed": args.seed,
        "question_types": list(QUESTION_TYPES),
        "questions_per_type": args.questions_per_type,
        "public_per_type": args.public_per_type,
        "require_gold_evidence_coverage": args.require_gold_evidence_coverage,
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
            sorted(Counter(question["type"] for question in selected).items())
        ),
        "source_files": {
            "questions": questions_path.name,
            "questions_sha256": sha256_file(questions_path),
            "corpus": corpus_path.name,
            "corpus_sha256": sha256_file(corpus_path),
            "gsw_root_label": gsw_root.name,
            "decompositions_log": (
                args.decompositions_log.name
                if args.decompositions_log
                else None
            ),
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
    parser.add_argument("--decompositions-log", type=Path)
    parser.add_argument("--seed", type=int, default=232)
    parser.add_argument("--questions-per-type", type=int, default=25)
    parser.add_argument("--public-per-type", type=int, default=20)
    parser.add_argument("--decomposition-validation-per-type", type=int, default=5)
    parser.add_argument(
        "--require-gold-evidence-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every object in the annotated evidence triples to appear "
            "as a named node in the selected context GSWs (default: true)."
        ),
    )
    args = parser.parse_args(argv)
    if args.public_per_type < 0:
        parser.error("--public-per-type must be non-negative")
    if args.public_per_type > args.questions_per_type:
        parser.error("--public-per-type cannot exceed --questions-per-type")
    if args.decomposition_validation_per_type < 0:
        parser.error("--decomposition-validation-per-type must be non-negative")
    if args.decomposition_validation_per_type > args.public_per_type:
        parser.error(
            "--decomposition-validation-per-type cannot exceed --public-per-type"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_dataset(args)
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
