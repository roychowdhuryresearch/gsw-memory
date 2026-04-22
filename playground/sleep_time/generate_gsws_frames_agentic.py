#!/usr/bin/env python3
"""
Generate agentic GSWs from FRAMES Wikipedia articles.

This mirrors the existing FRAMES fetch/cache flow in generate_gsws_frames.py,
but runs the staged AgenticGSWPipeline instead of the single-shot extractor.

Usage:
    # Dev subset using cached articles
    .venv/bin/python playground/sleep_time/generate_gsws_frames_agentic.py \
        --model-name gpt-4.1-mini --dev --skip-fetch

    # Full dataset with fresh Wikipedia fetch
    .venv/bin/python playground/sleep_time/generate_gsws_frames_agentic.py \
        --model-name gpt-4.1-mini
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

from gsw_memory.memory import AgenticGSWPipeline, GSWProcessor
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

    dev_titles = set()
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
    """Handle Ctrl-C reliably even when ``asyncio.to_thread`` is blocked on
    slow bedrock/litellm HTTP calls.

    Behavior:
    - First Ctrl-C: print a message, raise KeyboardInterrupt so the main()
      try/except can save partial state, AND arm a watchdog thread that
      force-exits with ``os._exit(130)`` after ``watchdog_seconds``.
    - Second Ctrl-C: immediate force-exit, skip the watchdog delay.

    The watchdog is necessary because Python 3.10's ``asyncio.run`` calls
    ``loop.shutdown_default_executor()`` which blocks waiting for every
    running thread pool worker to finish — and our workers are running
    60-second HTTP requests. Without the watchdog, one Ctrl-C appears to
    hang the process for a full minute per in-flight call.
    """
    import threading

    state = {"armed": False}
    original = signal.getsignal(signal.SIGINT)

    def _force_exit_after_delay() -> None:
        import time as _time
        _time.sleep(watchdog_seconds)
        sys.stderr.write(
            f"\n[agentic_gsw] shutdown watchdog fired after {watchdog_seconds:.0f}s "
            "— forcing exit(130)\n"
        )
        sys.stderr.flush()
        os._exit(130)

    def _handler(signum, frame):
        if state["armed"]:
            sys.stderr.write(
                "\n[agentic_gsw] second SIGINT received — forcing exit(130)\n"
            )
            sys.stderr.flush()
            os._exit(130)
        state["armed"] = True
        sys.stderr.write(
            f"\n[agentic_gsw] SIGINT received — attempting clean shutdown. "
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


def _setup_logging(log_path: Path, *, verbose: bool) -> None:
    """Configure console + file logging for the agentic pipeline.

    Console: INFO by default, DEBUG with --verbose. File: always DEBUG.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    agentic_logger = logging.getLogger("gsw_memory.memory.operator_utils.agentic_gsw")
    agentic_logger.setLevel(logging.DEBUG)
    agentic_logger.propagate = False

    root_driver_logger = logging.getLogger(__name__)
    root_driver_logger.setLevel(logging.DEBUG)
    root_driver_logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    for target in (agentic_logger, root_driver_logger):
        target.handlers = [console_handler, file_handler]

    print(f"  Pipeline log: {log_path}")


def _count_successful_documents(processed_documents: list[dict]) -> int:
    successful = 0
    for doc_chunks in processed_documents:
        if any(chunk_data.get("gsw") is not None for chunk_data in doc_chunks.values()):
            successful += 1
    return successful


def main() -> dict:
    parser = argparse.ArgumentParser(description="Generate agentic GSWs from FRAMES Wikipedia articles")
    parser.add_argument(
        "--article-index",
        default=str(DATA_DIR / "article_index.json"),
        help="Path to article_index.json",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR),
        help="Base FRAMES data directory",
    )
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        default="gpt-4.1-mini",
        help="Model name to use for all agentic stages",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help="Compatibility alias for --base-url",
    )
    parser.add_argument(
        "--reasoning-effort",
        "--reasoning_effort",
        dest="reasoning_effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort for GPT-OSS-style reasoning models",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Only process articles referenced by frames_dev.json",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip Wikipedia fetching and use cached articles",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N articles after sorting (0 = all)",
    )
    parser.add_argument("--sem-relation", type=int, default=8)
    parser.add_argument("--sem-question", type=int, default=16)
    parser.add_argument("--sem-verifier", type=int, default=3)
    parser.add_argument("--sem-coverage", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-coverage-passes", type=int, default=1)
    parser.add_argument(
        "--disable-entity-coverage",
        action="store_true",
        help="Disable answer-ID coverage repair for uncovered entities",
    )
    parser.add_argument("--window-strategy", type=str, default="paragraph")
    parser.add_argument(
        "--debug-traces",
        action="store_true",
        help="Write stage traces under the run output directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level console logging (full prompts/responses previewed)",
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

    print(f"\nPrepared {len(documents)} documents for agentic GSW generation")
    print(f"  Model: {args.model_name}")
    if resolved_base_url:
        print(f"  Base URL: {resolved_base_url}")

    title_to_doc_idx = {title: idx for idx, title in enumerate(sorted_titles)}
    mapping_path = output_dir / "title_to_doc_idx.json"
    with open(mapping_path, "w") as handle:
        json.dump(title_to_doc_idx, handle, indent=2)
    print(f"  Saved title→doc_idx mapping to: {mapping_path}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gsw_output_dir = output_dir / "networks_output" / f"frames_agentic_{run_timestamp}"
    trace_dir = gsw_output_dir / "traces"
    gsw_output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(gsw_output_dir / "pipeline.log", verbose=args.verbose)
    logging.getLogger(__name__).info(
        "run_start model=%s docs=%d output=%s",
        args.model_name,
        len(documents),
        gsw_output_dir,
    )

    pipeline = AgenticGSWPipeline(
        model_name=args.model_name,
        base_url=resolved_base_url,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
        window_strategy=args.window_strategy,
        sem_relation=args.sem_relation,
        sem_question=args.sem_question,
        sem_verifier=args.sem_verifier,
        sem_coverage=args.sem_coverage,
        enable_entity_coverage=not args.disable_entity_coverage,
        max_coverage_passes=args.max_coverage_passes,
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

    driver_logger = logging.getLogger(__name__)

    def _flush_progress(doc_idx: int, results_so_far: list[dict]) -> None:
        """Persist whatever we have so a mid-run crash doesn't lose doc_0…doc_n."""
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
        driver_logger.warning(
            "run_interrupted signal=SIGINT — saving any partial results that "
            "were already produced before the interrupt"
        )
        # Partial results live inside the orchestrator's _runner coroutine,
        # which was cancelled. Whatever was already persisted by
        # _flush_progress is on disk — fall through to the defensive save
        # with an empty list so the final-metadata path still runs.
        all_documents_data = []
    duration = (datetime.now() - start_time).total_seconds()

    # Final save in case the last flush raised or the list grew (defensive).
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
        "sem_relation": args.sem_relation,
        "sem_question": args.sem_question,
        "sem_verifier": args.sem_verifier,
        "sem_coverage": args.sem_coverage,
        "max_retries": args.max_retries,
        "max_coverage_passes": args.max_coverage_passes,
        "entity_coverage_enabled": not args.disable_entity_coverage,
        "window_strategy": args.window_strategy,
        "debug_traces": args.debug_traces,
        "agentic": True,
    }
    with open(gsw_output_dir / "agentic_run_metadata.json", "w") as handle:
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
