#!/usr/bin/env python3
"""Export aligned Qwen3 embedding tables for a standalone course package.

The script reads only files inside an already-built course package. It embeds
the entity and QA metadata rows and creates a fixed query table containing the
original questions and instructor-reviewed decomposition templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ENTITY_TASK = (
    "Given an entity name and its contextual roles/states, create an embedding "
    "that captures the entity's identity for semantic search and retrieval. "
    "Use the roles and states to determine similarity."
)
QA_TASK = (
    "Given a question-answer pair, create an embedding that captures the "
    "semantic meaning for similarity comparison with user queries."
)
QUERY_TASK = (
    "Given a query, create an embedding that captures the semantic meaning for "
    "similarity comparison with QA pairs."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def detailed_instruction(task: str, text: str) -> str:
    """Match the instruction format used by the Panini embedding pipeline."""

    return f"Instruct: {task}\nQuery: {text}"


def stable_query_id(text: str) -> str:
    return f"query::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def collect_query_rows(package: Path) -> list[dict[str, Any]]:
    """Collect and deduplicate fixed queries and decomposition templates."""

    rows_by_text: dict[str, dict[str, Any]] = {}
    question_paths = (
        package / "questions" / "public.jsonl",
        package / "questions" / "held_out.jsonl",
    )
    for path in question_paths:
        for row in load_jsonl(path):
            text = str(row["question"])
            query_row = rows_by_text.setdefault(
                text,
                {
                    "query_id": stable_query_id(text),
                    "text": text,
                    "sources": [],
                },
            )
            query_row["sources"].append(
                {
                    "kind": "original_question",
                    "question_id": row["question_id"],
                }
            )

    decomposition_path = (
        package / "instructor" / "reviewed_decompositions.jsonl"
    )
    if decomposition_path.exists():
        for row in load_jsonl(decomposition_path):
            for index, item in enumerate(row["decomposed_questions"], start=1):
                text = str(item["question"])
                query_row = rows_by_text.setdefault(
                    text,
                    {
                        "query_id": stable_query_id(text),
                        "text": text,
                        "sources": [],
                    },
                )
                query_row["sources"].append(
                    {
                        "kind": "decomposition_template",
                        "question_id": row["question_id"],
                        "step": index,
                        "requires_retrieval": bool(
                            item.get("requires_retrieval", True)
                        ),
                    }
                )

    return sorted(rows_by_text.values(), key=lambda row: row["query_id"])


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def embed_texts(
    *,
    model,
    tokenizer,
    texts: Sequence[str],
    task: str,
    batch_size: int,
    max_length: int,
    dimension: int | None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    output_dimension = dimension or int(model.config.hidden_size)
    result = np.empty((len(texts), output_dimension), dtype=np.float16)
    device = next(model.parameters()).device
    started = time.time()

    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        instructed = [
            detailed_instruction(task, text) for text in texts[start:stop]
        ]
        batch = tokenizer(
            instructed,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            embeddings = last_token_pool(
                output.last_hidden_state, batch["attention_mask"]
            )
            if dimension is not None:
                embeddings = embeddings[:, :dimension]
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        result[start:stop] = embeddings.float().cpu().numpy().astype(np.float16)
        elapsed = time.time() - started
        print(
            f"embedded {stop:>6}/{len(texts)} rows "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )
    return result


def save_embedding_table(
    output: Path,
    *,
    name: str,
    ids: Sequence[str],
    matrix: np.ndarray,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{name}_embeddings.npy", matrix, allow_pickle=False)
    write_json(output / f"{name}_ids.json", list(ids))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    package = args.package.resolve()
    output = package / "embeddings"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Embedding output is not empty: {output}. Pass --overwrite to replace."
        )
    output.mkdir(parents=True, exist_ok=True)

    entity_rows = load_jsonl(package / "metadata" / "entities.jsonl")
    qa_rows = load_jsonl(package / "metadata" / "qa_pairs.jsonl")
    query_rows = collect_query_rows(package)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attention,
    ).to(args.device)
    model.eval()

    tables = (
        (
            "entity",
            [row["entity_uid"] for row in entity_rows],
            [row["search_text"] for row in entity_rows],
            ENTITY_TASK,
        ),
        (
            "qa",
            [row["qa_uid"] for row in qa_rows],
            [row["search_text"] for row in qa_rows],
            QA_TASK,
        ),
        (
            "query",
            [row["query_id"] for row in query_rows],
            [row["text"] for row in query_rows],
            QUERY_TASK,
        ),
    )

    generated: dict[str, dict[str, Any]] = {}
    for name, ids, texts, task in tables:
        print(f"Embedding {name}: {len(texts)} rows", flush=True)
        matrix = embed_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            task=task,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dimension=args.dimension,
        )
        save_embedding_table(output, name=name, ids=ids, matrix=matrix)
        generated[name] = {
            "rows": len(ids),
            "dimension": int(matrix.shape[1]),
            "dtype": str(matrix.dtype),
            "matrix_sha256": sha256_file(output / f"{name}_embeddings.npy"),
            "ids_sha256": sha256_file(output / f"{name}_ids.json"),
            "task": task,
        }
        del matrix

    write_jsonl(output / "queries.jsonl", query_rows)
    metadata = {
        "model": args.model,
        "dtype_during_inference": args.dtype,
        "stored_dtype": "float16",
        "dimension": args.dimension or int(model.config.hidden_size),
        "normalized": True,
        "instruction_format": "Instruct: {task}\\nQuery: {text}",
        "max_length": args.max_length,
        "tables": generated,
    }
    write_json(output / "embedding_manifest.json", metadata)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--dimension",
        type=int,
        help="Optional Matryoshka truncation dimension; defaults to full size.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.dimension is not None and args.dimension <= 0:
        parser.error("--dimension must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    metadata = export_embeddings(parse_args(argv))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
