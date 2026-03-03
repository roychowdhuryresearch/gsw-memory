#!/usr/bin/env python3
"""
Extract important named entities from each document using the GSW entity list
and document text as context for an LLM call.

For each document the script:
1. Reads the raw document text from a corpus JSON file (indexed by doc position)
   OR from individual article .txt files (FRAMES mode).
2. Reads the GSW entity list (names + roles) from the networks/ directory.
3. Sends both to an LLM and asks it to identify the most important/distinctive
   named entities (people, places, organisations).
4. Saves the structured result to an output JSON file.

Usage:
    # Corpus mode (2wiki, musique, etc.)
    python extract_document_entities.py \
        --gsw_path /mnt/SSD1/shreyas/SM_GSW/2wiki/networks \
        --corpus_path playground_data/2wikimultihopqa_corpus.json \
        --num_docs 50 \
        --output doc_entities.json

    # FRAMES mode (individual article .txt files + title mapping)
    python extract_document_entities.py \
        --source frames \
        --gsw_path data/sleep_time/frames/networks_output/frames_20260228_222500/networks \
        --output doc_entities_frames.json
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

# ---------------------------------------------------------------------------
# FRAMES defaults
# ---------------------------------------------------------------------------
FRAMES_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sleep_time" / "frames"
FRAMES_ARTICLES_DIR = FRAMES_DATA_DIR / "articles" / "individual"
FRAMES_TITLE_MAPPING = FRAMES_DATA_DIR / "title_to_doc_idx.json"

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


def _sanitize_filename(title: str) -> str:
    """Convert article title to a safe filename (matches generate_gsws_frames.py)."""
    name = re.sub(r'[<>:"/\\|?*]', '_', title)
    name = name.strip('. ')
    return name[:200]


def load_frames_corpus(
    gsw_path: str,
    articles_dir: Path = FRAMES_ARTICLES_DIR,
    title_mapping_path: Path = FRAMES_TITLE_MAPPING,
    num_docs: int = -1,
) -> List[Dict[str, str]]:
    """Load FRAMES articles from individual .txt files using title_to_doc_idx.json.

    Returns a list aligned by doc index (same format as load_corpus).
    """
    if not title_mapping_path.exists():
        console.print(f"[red]Title mapping not found: {title_mapping_path}[/red]")
        sys.exit(1)
    with open(title_mapping_path) as f:
        title_to_idx: Dict[str, int] = json.load(f)

    # Invert: doc_idx -> title
    idx_to_title = {v: k for k, v in title_to_idx.items()}

    # Discover which doc indices exist in the GSW path
    base = Path(gsw_path)
    doc_dirs = sorted(base.glob("doc_*"), key=lambda p: int(p.name.replace("doc_", "")))
    if num_docs != -1:
        doc_dirs = doc_dirs[:num_docs]
    max_idx = max(int(d.name.replace("doc_", "")) for d in doc_dirs) if doc_dirs else 0

    # Build corpus list aligned by index
    corpus: List[Dict[str, str]] = [{"title": "", "text": ""} for _ in range(max_idx + 1)]
    loaded = 0
    missing = 0
    for doc_dir in doc_dirs:
        idx = int(doc_dir.name.replace("doc_", ""))
        title = idx_to_title.get(idx, "")
        if not title:
            console.print(f"[yellow]Warning: no title mapping for doc_{idx}[/yellow]")
            missing += 1
            continue
        txt_path = articles_dir / (_sanitize_filename(title) + ".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8")
            corpus[idx] = {"title": title, "text": text}
            loaded += 1
        else:
            console.print(f"[yellow]Warning: article file not found for '{title}': {txt_path}[/yellow]")
            missing += 1

    console.print(f"[green]✓ Loaded FRAMES corpus: {loaded} articles (from {len(doc_dirs)} doc dirs, {missing} missing)[/green]")
    return corpus


def load_gsw_entities_per_doc(
    gsw_path: str,
    num_docs: int = -1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect entities from all GSW chunk files under each doc_* directory.
    Deduplicates by entity name, merging roles across chunks.
    Counts VP question-answer references per entity for bridge scoring.

    Returns:
        {doc_id: [{"name": str, "roles": [str], "vp_count": int}, ...]}
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
        id_to_name: Dict[str, str] = {}  # entity id -> name
        vp_counts: Dict[str, int] = {}   # entity name -> VP answer count

        for gsw_file in sorted(doc_dir.glob("gsw_*.json")):
            try:
                with open(gsw_file) as f:
                    data = json.load(f)

                # Pass 1: collect entities and build id->name map
                for entity in data.get("entity_nodes", []):
                    name = entity.get("name", "").strip()
                    if not name:
                        continue
                    eid = entity.get("id", "")
                    id_to_name[eid] = name
                    roles = [
                        r.get("role", "")
                        for r in entity.get("roles", [])
                        if r.get("role")
                    ]
                    if name not in seen:
                        seen[name] = roles
                        vp_counts[name] = 0
                    else:
                        seen[name] = list(set(seen[name] + roles))

                # Pass 2: count VP question-answer references
                for vp in data.get("verb_phrase_nodes", []):
                    for question in vp.get("questions", []):
                        for answer_id in question.get("answers", []):
                            ename = id_to_name.get(answer_id, "")
                            if ename and ename in vp_counts:
                                vp_counts[ename] += 1

            except Exception as e:
                console.print(f"[yellow]Warning: could not read {gsw_file}: {e}[/yellow]")

        result[doc_id] = [
            {"name": n, "roles": r, "vp_count": vp_counts.get(n, 0)}
            for n, r in seen.items()
        ]

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
# Bridge scoring: DF × avg VP connections
# ---------------------------------------------------------------------------

def compute_entity_df(
    gsw_entities: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, int]:
    """
    Compute document frequency for each entity name.

    Returns:
        {entity_name: number_of_documents_containing_entity}
    """
    df: Dict[str, int] = {}
    for entities in gsw_entities.values():
        for e in entities:
            name = e["name"]
            df[name] = df.get(name, 0) + 1
    return df


def apply_bridge_reranking(
    results: Dict[str, Any],
    entity_df: Dict[str, int],
    top_k: int,
    min_df: int = 2,
) -> Dict[str, Any]:
    """
    Rerank entities within each document by bridge_score = log2(DF+1) × local_vp_count.

    - Filters out entities with df < min_df (can't bridge if only in 1 doc)
    - log2(DF+1) dampens ultra-common entities (e.g. "England" df=144)
    - local_vp_count rewards entities that are semantically rich in THIS document
    """
    total_before = 0
    total_after = 0
    for entry in results.values():
        total_before += len(entry["entities"])
        for e in entry["entities"]:
            df = entity_df.get(e["name"], 0)
            local_vp = e.get("vp_count", 0)
            e["df"] = df
            e["bridge_score"] = round(math.log2(df + 1) * local_vp, 2)
        # Filter: only keep entities appearing in min_df+ documents
        entry["entities"] = [e for e in entry["entities"] if e["df"] >= min_df]
        entry["entities"].sort(key=lambda x: x["bridge_score"], reverse=True)
        entry["entities"] = entry["entities"][:top_k]
        total_after += len(entry["entities"])
    console.print(f"  Filtered df<{min_df}: {total_before} → {total_after} entities ({total_before - total_after} removed)")
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(results: Dict[str, Any]) -> None:
    """Print a Rich summary table of extracted entities (first 20 docs)."""
    table = Table(title="Extracted Entities — Sample", show_lines=True)
    table.add_column("Doc", style="cyan", width=7)
    table.add_column("Title", style="white", width=22)
    table.add_column("#", style="green", width=3)
    table.add_column("Top Entities (score | vp | df)", style="yellow", width=65)

    shown = 0
    for doc_id, entry in sorted(results.items(), key=lambda x: int(x[0].replace("doc_", ""))):
        if shown >= 20:
            break
        entities = entry.get("entities", [])
        top3 = [
            f"{e['name']} ({e.get('bridge_score', 0):.0f}|vp={e.get('vp_count', 0)}|df={e.get('df', 0)})"
            for e in entities[:3]
        ]
        table.add_row(
            doc_id,
            entry.get("title", "")[:22],
            str(len(entities)),
            ", ".join(top3) or "-",
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
        "--source", choices=["corpus", "frames"], default="corpus",
        help="Data source: 'corpus' (JSON list) or 'frames' (individual .txt articles). Default: corpus"
    )
    parser.add_argument(
        "--gsw_path", required=True,
        help="Path to networks/ directory containing doc_*/gsw_*.json files"
    )
    parser.add_argument(
        "--corpus_path", default=None,
        help="Path to corpus JSON: list of {title, text} dicts aligned by index (required for --source corpus)"
    )
    parser.add_argument(
        "--articles_dir", default=None,
        help=f"FRAMES: path to individual article .txt files (default: {FRAMES_ARTICLES_DIR})"
    )
    parser.add_argument(
        "--title_mapping", default=None,
        help=f"FRAMES: path to title_to_doc_idx.json (default: {FRAMES_TITLE_MAPPING})"
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
        "--rerank_top_k", type=int, default=None,
        help=(
            "After bridge-score reranking, keep only this many entities per document. "
            "Defaults to --top_k if not set."
        ),
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Skip LLM extraction; use all GSW entities directly (type=OTHER, importance=3)"
    )
    args = parser.parse_args()

    # Validate: corpus mode requires --corpus_path
    if args.source == "corpus" and args.corpus_path is None:
        parser.error("--corpus_path is required when --source=corpus")

    return args


def build_results_without_llm(
    gsw_entities: Dict[str, List[Dict[str, Any]]],
    corpus: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Build results directly from GSW entities without LLM classification."""
    results: Dict[str, Any] = {}
    for doc_id, entities in gsw_entities.items():
        idx = int(doc_id.replace("doc_", ""))
        title = corpus[idx]["title"] if idx < len(corpus) else ""
        results[doc_id] = {
            "title": title,
            "entities": [
                {
                    "name": e["name"],
                    "type": "OTHER",
                    "importance": 3,
                    "vp_count": e.get("vp_count", 0),
                }
                for e in entities
            ],
        }
    return results


def main():
    args = parse_args()

    rerank_top_k = args.rerank_top_k if args.rerank_top_k is not None else args.top_k

    console.print("\n[bold cyan]Document Entity Extraction[/bold cyan]")
    console.print(f"  Source:     {args.source}")
    console.print(f"  GSW path:   {args.gsw_path}")
    if args.source == "corpus":
        console.print(f"  Corpus:     {args.corpus_path}")
    else:
        articles_dir = Path(args.articles_dir) if args.articles_dir else FRAMES_ARTICLES_DIR
        title_mapping = Path(args.title_mapping) if args.title_mapping else FRAMES_TITLE_MAPPING
        console.print(f"  Articles:   {articles_dir}")
        console.print(f"  Mapping:    {title_mapping}")
    console.print(f"  Model:      {args.model}")
    console.print(f"  Top-K:      {args.top_k}  (LLM candidates)")
    console.print(f"  Rerank K:   {rerank_top_k}  (after bridge-score reranking)")
    console.print(f"  Num docs:   {'all' if args.num_docs == -1 else args.num_docs}")
    console.print(f"  Output:     {args.output}\n")

    # 1. Load data
    if args.source == "frames":
        corpus = load_frames_corpus(
            gsw_path=args.gsw_path,
            articles_dir=articles_dir,
            title_mapping_path=title_mapping,
            num_docs=args.num_docs,
        )
    else:
        corpus = load_corpus(args.corpus_path)
    gsw_entities = load_gsw_entities_per_doc(args.gsw_path, args.num_docs)

    if args.no_llm:
        # Skip LLM — use raw GSW entities directly
        console.print("[yellow]LLM disabled (--no_llm): using raw GSW entities[/yellow]")
        results = build_results_without_llm(gsw_entities, corpus)
    else:
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

    # 5. Bridge scoring — reward cross-document entities with rich local VP connections
    console.print("[cyan]Computing bridge scores: log2(DF+1) × local_vp...[/cyan]")
    entity_df = compute_entity_df(gsw_entities)
    results = apply_bridge_reranking(results, entity_df, top_k=rerank_top_k)

    # Log highest DF entities for reference
    top_df = sorted(entity_df.items(), key=lambda x: x[1], reverse=True)[:10]
    console.print("  Most cross-document entities (highest DF):")
    for name, df in top_df:
        console.print(f"    df={df:>4}  log2(df+1)={math.log2(df+1):.1f}  {name}")

    # 6. Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]✓ Saved {len(results)} document entity lists → {output_path}[/bold green]")

    total_entities = sum(len(v["entities"]) for v in results.values())
    avg = total_entities / len(results) if results else 0
    console.print(f"  Total entities (after bridge reranking) : {total_entities}")
    console.print(f"  Average per document                    : {avg:.1f}")

    print_summary(results)


if __name__ == "__main__":
    main()
