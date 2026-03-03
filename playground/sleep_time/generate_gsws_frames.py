#!/usr/bin/env python3
"""
Generate GSWs from FRAMES Wikipedia articles.

Fetches Wikipedia article text (with local caching), then runs the GSW
generation pipeline to produce per-article GSW structures.

Usage:
    # Dev subset (~355 articles) with Together AI
    python data/sleep_time/frames/generate_gsws_frames.py \
        --model together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507 --dev

    # Full dataset (2476 articles) with OpenAI
    python data/sleep_time/frames/generate_gsws_frames.py \
        --model gpt-4.1-mini

    # Skip fetching (use cached articles)
    python data/sleep_time/frames/generate_gsws_frames.py \
        --model gpt-4.1-mini --skip_fetch
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import wikipedia

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from dotenv import load_dotenv
from gsw_memory import GSWProcessor
from gsw_memory.memory.models import GSWStructure
from gsw_memory.prompts.operator_prompts import FactualExtractionPrompts, PromptType

load_dotenv()
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sleep_time" / "frames"

# ---------------------------------------------------------------------------
# Wikipedia Fetching
# ---------------------------------------------------------------------------

def _sanitize_filename(title: str) -> str:
    """Convert article title to a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', title)
    name = name.strip('. ')
    return name[:200]  # filesystem limit


def save_individual_articles(cache: dict[str, str], cache_dir: Path) -> None:
    """Save each cached article as a separate .txt file."""
    individual_dir = cache_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for title, text in cache.items():
        if not text:
            continue
        filename = _sanitize_filename(title) + ".txt"
        (individual_dir / filename).write_text(text, encoding="utf-8")
        count += 1
    print(f"  Saved {count} individual article files to: {individual_dir}")


def fetch_articles(
    article_index: dict[str, str],
    cache_dir: Path,
    delay: float = 0.5,
) -> dict[str, str]:
    """Fetch Wikipedia articles by title, with local JSON cache.

    Returns {title: text} for all successfully fetched articles.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "articles_cache.json"

    # Load existing cache
    if cache_file.exists():
        with open(cache_file) as f:
            cache: dict[str, str] = json.load(f)
    else:
        cache = {}

    titles_to_fetch = [t for t in article_index if t not in cache]
    print(f"  Cached: {len(cache)}, to fetch: {len(titles_to_fetch)}")

    for i, title in enumerate(titles_to_fetch):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            cache[title] = page.content
        except wikipedia.exceptions.DisambiguationError as e:
            # Pick first option from disambiguation
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                cache[title] = page.content
            except Exception:
                print(f"  [{i+1}/{len(titles_to_fetch)}] SKIP (disambiguation): {title}")
                cache[title] = ""
        except wikipedia.exceptions.PageError:
            print(f"  [{i+1}/{len(titles_to_fetch)}] SKIP (not found): {title}")
            cache[title] = ""
        except Exception as e:
            print(f"  [{i+1}/{len(titles_to_fetch)}] SKIP (error): {title} — {e}")
            cache[title] = ""

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(titles_to_fetch)}] fetched...")
            # Save intermediate cache
            with open(cache_file, "w") as f:
                json.dump(cache, f)

        time.sleep(delay)

    # Final save
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    # Save individual article files
    save_individual_articles(cache, cache_dir)

    # Filter out empty articles
    result = {t: text for t, text in cache.items() if t in article_index and text}
    print(f"  Articles with content: {len(result)}/{len(article_index)}")
    return result


def load_cached_articles(
    article_index: dict[str, str],
    cache_dir: Path,
) -> dict[str, str]:
    """Load previously fetched articles from cache."""
    cache_file = cache_dir / "articles_cache.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"No article cache found at {cache_file}. Run without --skip_fetch first.")

    with open(cache_file) as f:
        cache = json.load(f)

    result = {t: text for t, text in cache.items() if t in article_index and text}
    print(f"  Loaded {len(result)} articles from cache ({len(article_index) - len(result)} missing/empty)")
    return result


# ---------------------------------------------------------------------------
# Direct Together API (--direct mode)
# ---------------------------------------------------------------------------

def _init_together_client(model_name: str):
    """Initialize Together AI client."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    from together import Together
    return Together(api_key=api_key)


def generate_gsw_direct(client, model: str, text: str) -> dict | None:
    """Generate a single GSW via direct Together API call with JSON schema."""
    messages = [
        {"role": "system", "content": FactualExtractionPrompts.SYSTEM_PROMPT},
        {"role": "user", "content": FactualExtractionPrompts.USER_PROMPT_TEMPLATE.format(
            input_text=text, background_context="",
        )},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_object",
            "schema": GSWStructure.model_json_schema(),
        },
        temperature=0.0,
        # max_tokens=65536,
    )

    content = response.choices[0].message.content
    if not content:
        return None
    gsw = GSWStructure.model_validate_json(content)
    return gsw.model_dump()


def process_documents_direct(
    client,
    model: str,
    documents: list[str],
    sorted_titles: list[str],
    output_dir: Path,
) -> int:
    """Process documents via direct API calls. Returns count of successful GSWs."""
    networks_dir = output_dir / "networks"
    networks_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    for i, (title, doc) in enumerate(zip(sorted_titles, documents)):
        doc_dir = networks_dir / f"doc_{i}"
        gsw_path = doc_dir / f"gsw_{i}_0.json"

        # Skip if already processed
        if gsw_path.exists():
            print(f"  [{i+1}/{len(documents)}] SKIP (exists): {title}")
            success += 1
            continue

        try:
            gsw_dict = generate_gsw_direct(client, model, doc)
            if gsw_dict:
                doc_dir.mkdir(parents=True, exist_ok=True)
                with open(gsw_path, "w") as f:
                    json.dump(gsw_dict, f, indent=2)
                success += 1
                print(f"  [{i+1}/{len(documents)}] OK: {title}")
            else:
                print(f"  [{i+1}/{len(documents)}] EMPTY: {title}")
        except Exception as e:
            print(f"  [{i+1}/{len(documents)}] ERROR: {title} — {e}")

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate GSWs from FRAMES Wikipedia articles")
    parser.add_argument("--article_index", default=str(DATA_DIR / "article_index.json"),
                        help="Path to article_index.json")
    parser.add_argument("--output", default=str(DATA_DIR), help="Output directory")
    parser.add_argument("--model", default="together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507", help="Model name (e.g. gpt-4.1-mini, together_ai/Qwen/...)")
    parser.add_argument("--dev", default=True, action="store_true", help="Only process articles from frames_dev.json")
    parser.add_argument("--skip_fetch", action="store_true", help="Skip Wikipedia fetching, use cached articles")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N articles (0 = all)")
    parser.add_argument("--direct", action="store_true", help="Use direct Together API calls instead of Curator pipeline")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for GSW processing (Curator mode)")
    parser.add_argument("--vllm_base_url", default="None", help="vLLM base URL (for hosted_vllm models, Curator mode)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    cache_dir = output_dir / "articles"

    # Load article index
    print(f"Loading article index from: {args.article_index}")
    with open(args.article_index) as f:
        article_index: dict[str, str] = json.load(f)
    print(f"  Total articles in index: {len(article_index)}")

    # Filter to dev subset if requested
    if args.dev:
        dev_path = output_dir / "frames_dev.json"
        if not dev_path.exists():
            print(f"ERROR: --dev flag requires {dev_path}. Run prep_frames.py --dev_sample 100 first.")
            sys.exit(1)
        with open(dev_path) as f:
            dev_questions = json.load(f)
        dev_titles = set()
        for q in dev_questions:
            for a in q["articles"]:
                dev_titles.add(a["title"])
        article_index = {t: url for t, url in article_index.items() if t in dev_titles}
        print(f"  Filtered to dev subset: {len(article_index)} articles")

    # Fetch or load articles
    print("\nFetching Wikipedia articles...")
    if args.skip_fetch:
        articles = load_cached_articles(article_index, cache_dir)
    else:
        articles = fetch_articles(article_index, cache_dir)

    if not articles:
        print("ERROR: No articles to process.")
        sys.exit(1)

    # Prepare documents for GSW processing
    # Sort by title for deterministic ordering
    sorted_titles = sorted(articles.keys())
    if args.limit > 0:
        sorted_titles = sorted_titles[:args.limit]
    documents = [f"{title}\n{articles[title]}" for title in sorted_titles]

    print(f"\nPrepared {len(documents)} documents for GSW generation")
    print(f"  Model: {args.model}")

    # Save title-to-doc-index mapping for later reference
    title_to_doc_idx = {title: i for i, title in enumerate(sorted_titles)}
    mapping_path = output_dir / "title_to_doc_idx.json"
    with open(mapping_path, "w") as f:
        json.dump(title_to_doc_idx, f, indent=2)
    print(f"  Saved title→doc_idx mapping to: {mapping_path}")

    # Process documents
    gsw_output_dir = output_dir / "networks_output" / f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.now()

    if args.direct:
        # Direct Together API mode
        print(f"\nDirect mode: calling {args.model} via Together API")
        client = _init_together_client(args.model)
        print(f"Generating GSWs → {gsw_output_dir}")
        valid = process_documents_direct(client, args.model, documents, sorted_titles, gsw_output_dir)
    else:
        # Curator pipeline mode
        print("\nInitializing GSW processor (Curator)...")
        processor = GSWProcessor(
            model_name=args.model,
            vllm_base_url=args.vllm_base_url,
            generation_params={"temperature": 0.0}, #"max_tokens": 65536},
            enable_coref=False,
            enable_chunking=False,
            chunk_size=1,
            overlap=0,
            enable_context=False,
            enable_spacetime=False,
            prompt_type=PromptType.FACTUAL,
            batched=True,
            batch_size=args.batch_size,
        )
        print(f"Generating GSWs → {gsw_output_dir}")
        gsw_structures = processor.process_documents(
            documents,
            output_dir=str(gsw_output_dir),
        )
        valid = sum(1 for g in gsw_structures if g is not None)

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nDone in {duration:.1f}s")
    print(f"  Generated: {valid}/{len(documents)} GSW structures")
    print(f"  Output: {gsw_output_dir}")


if __name__ == "__main__":
    main()
