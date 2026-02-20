#!/usr/bin/env python3
"""
Extract important named entities from each document using the GSW entity list
and document text as context for an LLM call.

For each document the script:
1. Reads the raw document text from a corpus JSON file (indexed by doc position).
2. Reads the GSW entity list (names + roles) from the networks/ directory.
3. Sends both to an LLM and asks it to identify the most important/distinctive
   named entities (people, places, organisations).
4. Saves the structured result to an output JSON file.

Usage:
    python extract_document_entities.py \
        --gsw_path /mnt/SSD1/shreyas/SM_GSW/2wiki/networks \
        --corpus_path playground_data/2wikimultihopqa_corpus.json \
        --num_docs 50 \
        --output doc_entities.json

    # All docs, gpt-4o-mini (default):
    python extract_document_entities.py \
        --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks \
        --corpus_path playground_data/musique_corpus.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from bespokelabs import curator
from pydantic import BaseModel, Field

console = Console()


# ---------------------------------------------------------------------------
# Pydantic response model for structured LLM output
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    name: str = Field(description="Exact entity name as it appears in the GSW list")
    type: Literal["PERSON", "PLACE", "ORG", "OTHER"] = Field(
        description=(
            "Entity type: PERSON (people/persons), PLACE (locations/regions/countries), "
            "ORG (institutions/organisations/groups), OTHER"
        )
    )
    importance: int = Field(
        ge=1, le=5,
        description="Importance 1-5: 5=the central subject of this document, 1=minor mention"
    )


class DocEntities(BaseModel):
    entities: List[ExtractedEntity] = Field(
        description="Important named entities selected from the GSW entity list"
    )


# ---------------------------------------------------------------------------
# Curator LLM operator — follows the exact same pattern as GSWOperator
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert at identifying the most important named entities in a document.\n"
    "Given a document and its pre-extracted GSW entity list (with semantic roles), "
    "select only the entities that are named and significant: people, places, and organisations.\n\n"
    "Rules:\n"
    "- Only select entities that appear verbatim in the GSW entity list provided.\n"
    "- Exclude: bare dates/years (e.g. '11 November 875'), generic nouns, abstract concepts.\n"
    "- Focus on entities that uniquely identify THIS document's subject matter.\n"
    "- Score importance 5=main subject of the document, 1=minor supporting entity.\n"
    "- Return at most {top_k} entities."
)

USER_PROMPT = (
    "Document title: {title}\n\n"
    "Document text:\n{text}\n\n"
    "GSW entity list (name → roles):\n{entity_list}\n\n"
    "From the GSW entity list above, select the most important named entities "
    "(people, places, organisations) that best represent the unique content of this document."
)


class EntityExtractor(curator.LLM):
    """
    Curator LLM operator that extracts important entities per document.
    Follows the same pattern as GSWOperator in gsw_operator.py.
    """

    def __init__(self, top_k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._top_k = top_k

    def prompt(self, input: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(top_k=self._top_k),
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    title=input["title"],
                    text=input["text"],
                    entity_list=input["entity_list"],
                ),
            },
        ]

    def parse(self, input: Dict[str, Any], response: DocEntities) -> Dict[str, Any]:
        """
        Response is already a validated DocEntities Pydantic object
        (same as GSWOperator receives a GSWStructure object).
        """
        entities = []
        if response and response.entities:
            for e in response.entities:
                entities.append({
                    "name": e.name,
                    "type": e.type,
                    "importance": e.importance,
                })
        # Sort by importance descending
        entities.sort(key=lambda x: x["importance"], reverse=True)
        return {
            "doc_id": input["doc_id"],
            "title": input["title"],
            "entities": entities,
        }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """Load corpus JSON: list of {title, text} dicts indexed by position."""
    path = Path(corpus_path)
    if not path.exists():
        console.print(f"[red]Corpus file not found: {corpus_path}[/red]")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    console.print(f"[green]✓ Loaded corpus: {len(data)} documents[/green]")
    return data


def load_gsw_entities_per_doc(
    gsw_path: str,
    num_docs: int = -1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect entities from all GSW chunk files under each doc_* directory.
    Deduplicates by entity name, merging roles across chunks.

    Returns:
        {doc_id: [{"name": str, "roles": [str]}, ...]}
    """
    base = Path(gsw_path)
    if not base.exists():
        console.print(f"[red]GSW path not found: {gsw_path}[/red]")
        sys.exit(1)

    doc_dirs = sorted(
        base.glob("doc_*"),
        key=lambda p: int(p.name.replace("doc_", ""))
    )
    if num_docs != -1:
        doc_dirs = doc_dirs[:num_docs]

    result: Dict[str, List[Dict[str, Any]]] = {}

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        seen: Dict[str, List[str]] = {}  # name -> roles (deduplicated)

        for gsw_file in sorted(doc_dir.glob("gsw_*.json")):
            try:
                with open(gsw_file) as f:
                    data = json.load(f)
                for entity in data.get("entity_nodes", []):
                    name = entity.get("name", "").strip()
                    if not name:
                        continue
                    roles = [
                        r.get("role", "")
                        for r in entity.get("roles", [])
                        if r.get("role")
                    ]
                    if name not in seen:
                        seen[name] = roles
                    else:
                        # Merge roles across chunks
                        seen[name] = list(set(seen[name] + roles))
            except Exception as e:
                console.print(f"[yellow]Warning: could not read {gsw_file}: {e}[/yellow]")

        result[doc_id] = [{"name": n, "roles": r} for n, r in seen.items()]

    console.print(f"[green]✓ Loaded GSW entities for {len(result)} documents[/green]")
    return result


def format_entity_list(entities: List[Dict[str, Any]]) -> str:
    """Format entity list as readable text for the LLM prompt."""
    lines = []
    for e in entities:
        roles_str = ", ".join(e["roles"]) if e["roles"] else "unknown"
        lines.append(f"- {e['name']} [{roles_str}]")
    return "\n".join(lines)


def build_inputs(
    corpus: List[Dict[str, str]],
    gsw_entities: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build per-document input dicts for the curator operator."""
    inputs = []
    doc_ids = sorted(
        gsw_entities.keys(),
        key=lambda d: int(d.replace("doc_", ""))
    )

    for doc_id in doc_ids:
        idx = int(doc_id.replace("doc_", ""))
        if idx >= len(corpus):
            console.print(
                f"[yellow]Warning: {doc_id} has no corpus entry (index {idx} >= {len(corpus)}), skipping[/yellow]"
            )
            continue

        corpus_entry = corpus[idx]
        inputs.append({
            "doc_id": doc_id,
            "title": corpus_entry.get("title", ""),
            "text": corpus_entry.get("text", ""),
            "entity_list": format_entity_list(gsw_entities[doc_id]),
        })

    return inputs


# ---------------------------------------------------------------------------
# IDF reranking
# ---------------------------------------------------------------------------

def compute_idf(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute IDF score for each entity name across all documents.

    idf(entity) = log(N / df(entity))

    Entities that appear in many documents (like "Japan", "United States")
    get a low IDF; rare/distinctive entities get a high IDF.
    """
    N = len(results)
    if N == 0:
        return {}
    df: Counter = Counter()
    for entry in results.values():
        for e in entry["entities"]:
            df[e["name"]] += 1
    return {name: math.log(N / count) for name, count in df.items()}


def apply_idf_reranking(
    results: Dict[str, Any],
    idf: Dict[str, float],
    top_k: int,
) -> Dict[str, Any]:
    """
    Rerank entities within each document by final_score = importance × idf.

    Adds "idf" and "final_score" fields to each entity dict, then re-sorts
    and trims to top_k.
    """
    for entry in results.values():
        for e in entry["entities"]:
            e["idf"] = round(idf.get(e["name"], 0.0), 3)
            e["final_score"] = round(e["importance"] * e["idf"], 3)
        entry["entities"].sort(key=lambda x: x["final_score"], reverse=True)
        entry["entities"] = entry["entities"][:top_k]
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(results: Dict[str, Any]) -> None:
    """Print a Rich summary table of extracted entities (first 20 docs)."""
    table = Table(title="Extracted Entities — Sample", show_lines=True)
    table.add_column("Doc", style="cyan", width=7)
    table.add_column("Title", style="white", width=22)
    table.add_column("Total", style="green", width=6)
    table.add_column("Top People", style="yellow", width=35)
    table.add_column("Top Places", style="magenta", width=25)

    shown = 0
    for doc_id, entry in sorted(results.items(), key=lambda x: int(x[0].replace("doc_", ""))):
        if shown >= 20:
            break
        entities = entry.get("entities", [])
        people = [e["name"] for e in entities if e["type"] == "PERSON"][:3]
        places = [e["name"] for e in entities if e["type"] == "PLACE"][:2]
        table.add_row(
            doc_id,
            entry.get("title", "")[:22],
            str(len(entities)),
            ", ".join(people) or "-",
            ", ".join(places) or "-",
        )
        shown += 1

    console.print(table)
    if len(results) > 20:
        console.print(f"[dim]... and {len(results) - 20} more documents not shown[/dim]")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract important named entities per document using GSW + LLM"
    )
    parser.add_argument(
        "--gsw_path", required=True,
        help="Path to networks/ directory containing doc_*/gsw_*.json files"
    )
    parser.add_argument(
        "--corpus_path", required=True,
        help="Path to corpus JSON: list of {title, text} dicts aligned by index"
    )
    parser.add_argument(
        "--num_docs", type=int, default=-1,
        help="Number of documents to process (-1 = all, default: -1)"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Max entities to extract per document (default: 10)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", default="doc_entities.json",
        help="Output JSON file path (default: doc_entities.json)"
    )
    parser.add_argument(
        "--idf_top_k", type=int, default=None,
        help=(
            "After IDF reranking, keep only this many entities per document. "
            "Defaults to --top_k if not set."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    idf_top_k = args.idf_top_k if args.idf_top_k is not None else args.top_k

    console.print("\n[bold cyan]Document Entity Extraction[/bold cyan]")
    console.print(f"  GSW path:   {args.gsw_path}")
    console.print(f"  Corpus:     {args.corpus_path}")
    console.print(f"  Model:      {args.model}")
    console.print(f"  Top-K:      {args.top_k}  (LLM candidates)")
    console.print(f"  IDF Top-K:  {idf_top_k}  (after TF-IDF reranking)")
    console.print(f"  Num docs:   {'all' if args.num_docs == -1 else args.num_docs}")
    console.print(f"  Output:     {args.output}\n")

    # 1. Load data
    corpus = load_corpus(args.corpus_path)
    gsw_entities = load_gsw_entities_per_doc(args.gsw_path, args.num_docs)

    # 2. Build LLM inputs (one dict per document)
    inputs = build_inputs(corpus, gsw_entities)
    console.print(f"[cyan]Built {len(inputs)} document inputs[/cyan]")

    if not inputs:
        console.print(
            "[red]No documents to process. "
            "Check that --gsw_path and --corpus_path are aligned (doc_i → corpus[i]).[/red]"
        )
        sys.exit(1)

    # 3. Run LLM extraction via curator (parallelised automatically)
    console.print(f"\n[bold]Running LLM entity extraction with {args.model}...[/bold]")

    extractor = EntityExtractor(
        top_k=args.top_k,
        model_name=args.model,
        response_format=DocEntities,   # structured output — same as GSWOperator
        backend="openai",
    )

    dataset = extractor(inputs)

    # 4. Collect results from .dataset iterator
    results: Dict[str, Any] = {}
    for row in dataset.dataset:
        doc_id = row["doc_id"]
        results[doc_id] = {
            "title": row["title"],
            "entities": row["entities"],
        }

    # 5. IDF reranking — penalise entities common across many docs
    console.print("[cyan]Applying IDF reranking...[/cyan]")
    idf = compute_idf(results)
    results = apply_idf_reranking(results, idf, top_k=idf_top_k)

    # Log the most penalised (common) entities so the user can verify
    penalised = sorted(idf.items(), key=lambda x: x[1])[:10]
    console.print("  Most common entities (lowest IDF, penalised most):")
    for name, score in penalised:
        console.print(f"    idf={score:.2f}  {name}")

    # 6. Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]✓ Saved {len(results)} document entity lists → {output_path}[/bold green]")

    total_entities = sum(len(v["entities"]) for v in results.values())
    avg = total_entities / len(results) if results else 0
    console.print(f"  Total entities (after IDF reranking) : {total_entities}")
    console.print(f"  Average per document                 : {avg:.1f}")

    print_summary(results)


if __name__ == "__main__":
    main()
