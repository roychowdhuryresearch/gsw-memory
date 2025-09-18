#!/usr/bin/env python3
"""
map_title_to_entity_batched.py

For each doc_* directory, locate the entity whose name best matches the corpus
title. If the name differs, create a deep-copied clone of every gsw_*.json with
only that entity’s `name` rewritten to “Old Name (Title)”. The clones are written
to gsw_networks/normalized_networks/ in the same directory/file structure as the
original networks so you can drop them in as a replacement set.
"""

import argparse
import copy
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from bespokelabs import curator
except ImportError as exc:
    raise SystemExit("Install bespokelabs-curator (pip install bespokelabs-curator)") from exc

# Paths
CORPUS_PATH = Path("/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset/2wikimultihopqa_corpus.json")
ORIG_GSW_BASE = Path("/home/shreyas/NLP/SM/gensemworkspaces/gsw_networks/networks")
NORMALIZED_BASE = ORIG_GSW_BASE.parent / "normalized_networks"
MODEL_NAME = "gpt-4o-mini"
LEXICAL_STRIP = re.compile(r"[\\W_]+")


def normalize(text: str) -> str:
    return LEXICAL_STRIP.sub("", text.lower())


def load_corpus() -> List[Dict[str, Any]]:
    with CORPUS_PATH.open("r") as fh:
        return json.load(fh)


def load_gsws(doc_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    records = []
    for fp in sorted(doc_dir.glob("gsw_*.json")):
        if fp.name.endswith("_normalized.json"):
            continue  # skip previously generated clones
        with fp.open("r") as fh:
            records.append((fp, json.load(fh)))
    return records



@dataclass
class DocInfo:
    doc_idx: int
    doc_dir: Path
    title: str
    text: str
    gsw_records: List[Tuple[Path, Dict[str, Any]]]
    entities: List[Dict[str, Any]]


class TitleMatcher(curator.LLM):
    return_completions_object = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prompt(self, row: Dict[str, Any]):
        lines = [
            "Pick the entity whose name best refers to the article title.",
            "Return JSON: { \"entity_id\": \"eX\" } or { \"entity_id\": null } if no good match.",
            f"Article title: {row['title']}",
            f"Article: {row['text']}",
            "Entities:",
        ]
        for ent in row["entities"]:
            lines.append(f"- id: {ent.get('id')}  name: {ent.get('name')}")
        lines.append("Respond with JSON only.")
        return [
            {"role": "system", "content": "Select the entity id that matches the title best."},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def parse(self, row: Dict[str, Any], response):
        parsed = response if isinstance(response, dict) else json.loads(response)
        entity_id = parsed.get("entity_id")
        if not isinstance(entity_id, str):
            entity_id = None
        return [{
            "doc_idx": row["doc_idx"],
            "entity_id": entity_id,
        }]


def prepare_docs(limit: Optional[int], doc_ids: Optional[List[int]]) -> Tuple[List[DocInfo], List[Dict[str, Any]]]:
    corpus = load_corpus()

    doc_dirs = [ORIG_GSW_BASE / f"doc_{idx}" for idx in doc_ids] if doc_ids else sorted(ORIG_GSW_BASE.glob("doc_*"))
    if limit:
        doc_dirs = doc_dirs[:limit]

    docs: List[DocInfo] = []
    prompts: List[Dict[str, Any]] = []

    for doc_dir in doc_dirs:
        try:
            idx = int(doc_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx >= len(corpus):
            continue

        gsw_records = load_gsws(doc_dir)
        if not gsw_records:
            continue

        title = corpus[idx].get("title", "")
        text = corpus[idx].get("text", "")

        entities: List[Dict[str, Any]] = []
        for _, gsw in gsw_records:
            entities.extend(gsw.get("entity_nodes", []))

        if not entities or not title:
            continue

        doc_info = DocInfo(
            doc_idx=idx,
            doc_dir=doc_dir,
            title=title,
            text=text,
            gsw_records=gsw_records,
            entities=entities,
        )
        docs.append(doc_info)

        norm_title = normalize(title)
        lexical_hit = next((ent["id"] for ent in entities if normalize(ent.get("name", "")) == norm_title), None)
        if lexical_hit is None:
            prompts.append({
                "doc_idx": idx,
                "title": title,
                "text": text,
                "entities": entities,
            })

    return docs, prompts


def write_normalized_gsws(doc_info: DocInfo, matched_entity_id: Optional[str], dry_run: bool):
    title = doc_info.title
    norm_title = normalize(title)
    out_doc_dir = NORMALIZED_BASE / f"doc_{doc_info.doc_idx}"
    out_doc_dir.mkdir(parents=True, exist_ok=True)

    for orig_path, gsw_data in doc_info.gsw_records:
        gsw_copy = copy.deepcopy(gsw_data)
        if matched_entity_id:
            for entity in gsw_copy.get("entity_nodes", []):
                if entity.get("id") == matched_entity_id:
                    if normalize(entity.get("name", "")) != norm_title:
                        entity["name"] = f"{entity['name']} ({title})"
                    break

        out_path = out_doc_dir / orig_path.name
        if dry_run:
            print(f"\n=== dry-run: {out_path} ===")
            print(json.dumps(gsw_copy, indent=2, ensure_ascii=False))
        else:
            with out_path.open("w") as fh:
                json.dump(gsw_copy, fh, indent=2, ensure_ascii=False)
            print(f"[write] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create normalized GSW clones with title-aligned entity names.")
    parser.add_argument("--limit", type=int, help="Process only the first N doc directories.")
    parser.add_argument("--doc-ids", nargs="*", type=int, help="Explicit doc indices.")
    parser.add_argument("--dry-run", action="store_true", help="Print normalized GSWs instead of writing files.")
    parser.add_argument("--model", default=MODEL_NAME, help="Curator model to use.")
    parser.add_argument("--max-concurrency", type=int, default=10, help="Curator concurrency limit.")
    args = parser.parse_args()

    load_dotenv()

    docs, prompts = prepare_docs(limit=args.limit, doc_ids=args.doc_ids)

    lexical_matches: Dict[int, str] = {}
    for info in docs:
        norm_title = normalize(info.title)
        match = next((ent["id"] for ent in info.entities if normalize(ent["name"]) == norm_title), None)
        if match:
            lexical_matches[info.doc_idx] = match

    llm_matches: Dict[int, Optional[str]] = {}
    print(f"Running LLM calls for {len(prompts)} documents")

    if prompts:
        matcher = TitleMatcher(
            model_name=args.model,
            generation_params={"temperature": 0.0},
        )
        responses = matcher(prompts)
        for row in responses.dataset:
            llm_matches[row["doc_idx"]] = row.get("entity_id")

    for info in docs:
        matched_id = lexical_matches.get(info.doc_idx)
        if matched_id is None:
            matched_id = llm_matches.get(info.doc_idx)

        if matched_id is None:
            print(f"[skip] doc_{info.doc_idx}: could not align '{info.title}'")
            continue

        write_normalized_gsws(info, matched_id, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
