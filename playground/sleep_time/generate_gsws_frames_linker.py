#!/usr/bin/env python3
"""Generate GSWs from FRAMES Wikipedia articles using the LinkerPipeline.

Four-stage agentic pipeline:
  STEP 1: EntityListAgent — flat list of canonical entities
  STEP 2: VerbPhraseListAgent — flat list of verb phrase slots
  STEP 3: LinkerAgent — interactive tool-using agent that wires entities to
                        verb phrases until coverage is complete
  STEP 4: QuestionAgent — fan-out forward+reverse question generation
  STEP 5: deterministic assembly into GSWStructure

Usage:
    .venv/bin/python playground/sleep_time/generate_gsws_frames_linker.py \\
        --model bedrock/openai.gpt-oss-120b-1:0 \\
        --reasoning-effort medium \\
        --dev --skip-fetch --limit 1 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

from gsw_memory.memory import GSWProcessor
from gsw_memory.memory.operator_utils.agentic_linker import LinkerPipeline
from gsw_memory.memory.operator_utils.alibaba_thinking import normalize_openai_base_url

from generate_gsws_frames import fetch_articles, load_cached_articles

load_dotenv()


def default_frames_data_dir() -> Path:
    candidates = [
        REPO_ROOT / "data" / "sleep_time" / "frames",
        REPO_ROOT.parent / "data" / "sleep_time" / "frames",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_DIR = default_frames_data_dir()


def resolve_base_url(
    base_url: str | None = None,
    vllm_base_url: str | None = None,
) -> str | None:
    return normalize_openai_base_url(base_url) or normalize_openai_base_url(vllm_base_url)


def _load_article_index(article_index_path: str, output_dir: Path, dev: bool) -> dict[str, str]:
    print(f"Loading article index from: {article_index_path}")
    with open(article_index_path) as handle:
        article_index: dict[str, str] = json.load(handle)
    print(f"  Total articles in index: {len(article_index)}")

    if not dev:
        return article_index

    dev_path = output_dir / "frames_dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(
            f"--dev flag requires {dev_path}. Run prep_frames.py --dev_sample 100 first."
        )
    with open(dev_path) as handle:
        dev_questions = json.load(handle)

    dev_titles: set[str] = set()
    for question in dev_questions:
        for article in question["articles"]:
            dev_titles.add(article["title"])
    filtered = {title: url for title, url in article_index.items() if title in dev_titles}
    print(f"  Filtered to dev subset: {len(filtered)} articles")
    return filtered


def _save_outputs(
    processor: GSWProcessor,
    output_dir: Path,
    documents: list[str],
    all_documents_data: list[dict],
) -> None:
    processor._save_outputs_unified(
        output_dir=str(output_dir),
        save_intermediates=False,
        resolved_documents={idx: document for idx, document in enumerate(documents)},
        all_documents_data=all_documents_data,
        do_visualization=False,
        batch_idx=1,
    )


def _install_interrupt_handler(watchdog_seconds: float = 10.0) -> None:
    """Same shape as the agentic driver — first Ctrl-C tries clean shutdown,
    second forces exit, watchdog thread guarantees exit within
    ``watchdog_seconds``."""
    import threading

    state = {"armed": False}
    original = signal.getsignal(signal.SIGINT)

    def _force_exit_after_delay() -> None:
        import time as _time
        _time.sleep(watchdog_seconds)
        sys.stderr.write(
            f"\n[linker] shutdown watchdog fired after {watchdog_seconds:.0f}s "
            "— forcing exit(130)\n"
        )
        sys.stderr.flush()
        os._exit(130)

    def _handler(signum, frame):
        if state["armed"]:
            sys.stderr.write("\n[linker] second SIGINT — forcing exit(130)\n")
            sys.stderr.flush()
            os._exit(130)
        state["armed"] = True
        sys.stderr.write(
            f"\n[linker] SIGINT received — attempting clean shutdown. "
            f"In-flight LLM calls may take up to ~{watchdog_seconds:.0f}s to unwind. "
            "Press Ctrl-C again to force exit immediately.\n"
        )
        sys.stderr.flush()
        threading.Thread(target=_force_exit_after_delay, daemon=True).start()
        if callable(original) and original not in (signal.SIG_DFL, signal.SIG_IGN):
            original(signum, frame)
        else:
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handler)


def _setup_logging(
    log_path: Path,
    *,
    verbose: bool,
    show_reasoning: bool,
) -> None:
    """Three log densities, controlled by --verbose and --show-reasoning:

      default                  → STEP markers + orchestrator summaries.
                                 Transport noise suppressed (WARNING+).
      --show-reasoning         → above + per-call agent reasoning previews
                                 (still no full prompts/payloads).
      --verbose                → everything (DEBUG on every stream).

    The file log always captures full DEBUG regardless of console mode.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    linker_logger = logging.getLogger("gsw_memory.memory.operator_utils.agentic_linker")
    linker_logger.setLevel(logging.DEBUG)
    linker_logger.propagate = False

    transport_logger = logging.getLogger("gsw_memory.memory.operator_utils.agentic_gsw.transport_shim")
    transport_logger.setLevel(logging.DEBUG)
    transport_logger.propagate = False

    reasoning_logger = logging.getLogger("gsw_memory.agents.reasoning")
    reasoning_logger.setLevel(logging.DEBUG)
    reasoning_logger.propagate = False

    driver_logger = logging.getLogger(__name__)
    driver_logger.setLevel(logging.DEBUG)
    driver_logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Console handlers — one per logger so we can give each its own level.
    def _console(level: int) -> logging.StreamHandler:
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(fmt)
        return h

    if verbose:
        # Everything DEBUG on console.
        linker_console = _console(logging.DEBUG)
        transport_console = _console(logging.DEBUG)
        reasoning_console = _console(logging.INFO)
        driver_console = _console(logging.DEBUG)
        mode = "verbose"
    elif show_reasoning:
        # Linker INFO (STEP markers + agent summaries) + reasoning previews.
        # Transport silent except for warnings/errors.
        linker_console = _console(logging.INFO)
        transport_console = _console(logging.WARNING)
        reasoning_console = _console(logging.INFO)
        driver_console = _console(logging.INFO)
        mode = "show-reasoning"
    else:
        # Clean: just STEP markers and orchestrator summaries.
        linker_console = _console(logging.INFO)
        transport_console = _console(logging.WARNING)
        reasoning_console = _console(logging.WARNING)
        driver_console = _console(logging.INFO)
        mode = "clean"

    linker_logger.handlers = [linker_console, file_handler]
    transport_logger.handlers = [transport_console, file_handler]
    reasoning_logger.handlers = [reasoning_console, file_handler]
    driver_logger.handlers = [driver_console, file_handler]

    print(f"  Pipeline log: {log_path}")
    print(f"  Console mode: {mode}")


def _count_successful_documents(processed_documents: list[dict]) -> int:
    successful = 0
    for doc_chunks in processed_documents:
        if any(chunk_data.get("gsw") is not None for chunk_data in doc_chunks.values()):
            successful += 1
    return successful


def main() -> dict:
    parser = argparse.ArgumentParser(description="Generate GSWs from FRAMES articles using the LinkerPipeline")
    parser.add_argument("--article-index", default=str(DATA_DIR / "article_index.json"))
    parser.add_argument("--output", default=str(DATA_DIR))
    parser.add_argument("--model-name", "--model", dest="model_name", default="gpt-4.1-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument(
        "--reasoning-effort",
        "--reasoning_effort",
        dest="reasoning_effort",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sem-link", type=int, default=8)
    parser.add_argument(
        "--link-batch-size",
        type=int,
        default=8,
        help="Number of verb phrases to batch per LinkAgent / LinkVerifier call. "
        "Higher = cheaper (fewer calls, prompt text amortized) but larger output payload.",
    )
    parser.add_argument("--max-link-retries", type=int, default=2)
    parser.add_argument("--debug-traces", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show per-agent reasoning previews on console (middle ground "
        "between clean STEP markers and full DEBUG --verbose)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    cache_dir = output_dir / "articles"
    resolved_base_url = resolve_base_url(args.base_url, args.vllm_base_url)

    article_index = _load_article_index(args.article_index, output_dir, args.dev)

    print("\nFetching Wikipedia articles...")
    if args.skip_fetch:
        articles = load_cached_articles(article_index, cache_dir)
    else:
        articles = fetch_articles(article_index, cache_dir)
    if not articles:
        raise RuntimeError("No articles available to process.")

    sorted_titles = sorted(articles.keys())
    if args.limit > 0:
        sorted_titles = sorted_titles[: args.limit]
    documents = [f"{title}\n{articles[title]}" for title in sorted_titles]

    print(f"\nPrepared {len(documents)} documents for linker GSW generation")
    print(f"  Model: {args.model_name}")
    if resolved_base_url:
        print(f"  Base URL: {resolved_base_url}")

    title_to_doc_idx = {title: idx for idx, title in enumerate(sorted_titles)}
    mapping_path = output_dir / "title_to_doc_idx.json"
    with open(mapping_path, "w") as handle:
        json.dump(title_to_doc_idx, handle, indent=2)
    print(f"  Saved title→doc_idx mapping to: {mapping_path}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gsw_output_dir = output_dir / "networks_output" / f"frames_linker_{run_timestamp}"
    trace_dir = gsw_output_dir / "traces"
    gsw_output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(
        gsw_output_dir / "pipeline.log",
        verbose=args.verbose,
        show_reasoning=args.show_reasoning,
    )
    driver_logger = logging.getLogger(__name__)
    driver_logger.info(
        "run_start model=%s docs=%d output=%s",
        args.model_name,
        len(documents),
        gsw_output_dir,
    )

    pipeline = LinkerPipeline(
        model_name=args.model_name,
        base_url=resolved_base_url,
        reasoning_effort=args.reasoning_effort,
        sem_link=args.sem_link,
        link_batch_size=args.link_batch_size,
        max_link_retries=args.max_link_retries,
        debug=args.debug_traces,
        trace_dir=str(trace_dir),
    )

    saver = GSWProcessor(
        model_name=args.model_name,
        base_url=resolved_base_url,
        enable_coref=False,
        enable_chunking=False,
        enable_context=False,
        enable_spacetime=False,
    )

    def _flush_progress(doc_idx: int, results_so_far: list[dict]) -> None:
        try:
            _save_outputs(saver, gsw_output_dir, documents, results_so_far)
            driver_logger.info(
                "incremental_save doc=%d total_saved=%d output=%s",
                doc_idx,
                len(results_so_far),
                gsw_output_dir,
            )
        except Exception as exc:
            driver_logger.warning(
                "incremental_save_failed doc=%d error=%r — pipeline continues",
                doc_idx,
                exc,
            )

    _install_interrupt_handler()

    start_time = datetime.now()
    interrupted = False
    try:
        all_documents_data = pipeline.process_documents(
            documents,
            on_document_done=_flush_progress,
        )
    except KeyboardInterrupt:
        interrupted = True
        driver_logger.warning("run_interrupted signal=SIGINT — partial results already on disk")
        all_documents_data = []
    duration = (datetime.now() - start_time).total_seconds()

    if all_documents_data:
        _save_outputs(saver, gsw_output_dir, documents, all_documents_data)

    successful_docs = _count_successful_documents(all_documents_data)
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model_name": args.model_name,
        "base_url": resolved_base_url,
        "reasoning_effort": args.reasoning_effort,
        "documents_processed": len(documents),
        "successful_documents": successful_docs,
        "duration_seconds": duration,
        "article_titles": sorted_titles,
        "sem_link": args.sem_link,
        "link_batch_size": args.link_batch_size,
        "max_link_retries": args.max_link_retries,
        "debug_traces": args.debug_traces,
        "interrupted": interrupted,
        "pipeline": "linker",
    }
    with open(gsw_output_dir / "linker_run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nDone in {duration:.1f}s")
    print(f"  Generated: {successful_docs}/{len(documents)} GSW structures")
    print(f"  Output: {gsw_output_dir}")
    return {
        "output_dir": str(gsw_output_dir),
        "successful_documents": successful_docs,
        "documents_processed": len(documents),
    }


if __name__ == "__main__":
    main()
