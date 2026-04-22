#!/usr/bin/env python3
"""
Process corpus documents through the agentic GSW pipeline.

This mirrors the main corpus-loading and output layout from generate_gsws_multihop.py
while swapping in the staged agentic builder.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from gsw_memory.memory import AgenticGSWPipeline, GSWProcessor

from generate_gsws_multihop import (
    _count_successful_documents,
    load_frames_corpus,
    load_full_corpus,
    resolve_base_url,
)

load_dotenv()


def setup_logging_agentic() -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "logs",
        f"agentic_gsw_{timestamp}",
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "gsw_output"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "processing_logs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "traces"), exist_ok=True)
    return {
        "base_dir": log_dir,
        "gsw_output_dir": os.path.join(log_dir, "gsw_output"),
        "processing_logs_dir": os.path.join(log_dir, "processing_logs"),
        "trace_dir": os.path.join(log_dir, "traces"),
        "timestamp": timestamp,
    }


def _save_outputs(
    processor: GSWProcessor,
    output_dir: str,
    documents: List[str],
    all_documents_data: List[Dict],
) -> None:
    processor._save_outputs_unified(
        output_dir=output_dir,
        save_intermediates=False,
        resolved_documents={idx: document for idx, document in enumerate(documents)},
        all_documents_data=all_documents_data,
        do_visualization=False,
        batch_idx=1,
    )


def main() -> Dict:
    parser = argparse.ArgumentParser(description="Process corpus through agentic GSW pipeline")
    parser.add_argument("--model-name", "--model", dest="model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--vllm-base-url", type=str, default=None)
    parser.add_argument(
        "--reasoning-effort",
        "--reasoning_effort",
        dest="reasoning_effort",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--corpus-type", choices=["2wiki", "frames"], default="2wiki")
    parser.add_argument("--frames-dir", type=str, default="data/sleep_time/frames")
    parser.add_argument("--dev", action="store_true", help="Filter to FRAMES dev subset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sem-relation", type=int, default=8)
    parser.add_argument("--sem-question", type=int, default=16)
    parser.add_argument("--sem-verifier", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--window-strategy", type=str, default="paragraph")
    parser.add_argument("--debug-traces", action="store_true")
    args = parser.parse_args()

    resolved_base_url = resolve_base_url(args.base_url, args.vllm_base_url)
    log_dirs = setup_logging_agentic()

    if args.corpus_type == "frames":
        documents, document_titles = load_frames_corpus(
            args.frames_dir,
            dev=args.dev,
            limit=args.limit,
        )
    else:
        documents, document_titles = load_full_corpus()
        if args.limit > 0:
            documents = documents[: args.limit]
            document_titles = document_titles[: args.limit]

    documents = [f"{title}\n{doc}" for title, doc in zip(document_titles, documents)]
    pipeline = AgenticGSWPipeline(
        model_name=args.model_name,
        base_url=resolved_base_url,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
        window_strategy=args.window_strategy,
        sem_relation=args.sem_relation,
        sem_question=args.sem_question,
        sem_verifier=args.sem_verifier,
        debug=args.debug_traces,
        trace_dir=log_dirs["trace_dir"],
    )

    start_time = datetime.now()
    all_documents_data = pipeline.process_documents(documents)
    run_duration = (datetime.now() - start_time).total_seconds()

    output_dir = os.path.join(
        log_dirs["gsw_output_dir"],
        f"full_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    saver = GSWProcessor(
        model_name=args.model_name,
        base_url=resolved_base_url,
        enable_coref=False,
        enable_chunking=False,
        enable_context=False,
        enable_spacetime=False,
    )
    _save_outputs(saver, output_dir, documents, all_documents_data)

    successful_docs = _count_successful_documents(all_documents_data)
    batch_log = {
        "batch_num": "full_corpus",
        "batch_start": "full_corpus",
        "batch_end": "full_corpus",
        "documents_processed": len(documents),
        "gsw_structures_generated": successful_docs,
        "processing_time_seconds": run_duration,
        "timestamp": datetime.now().isoformat(),
        "document_titles": document_titles,
        "agentic": True,
        "model_name": args.model_name,
        "base_url": resolved_base_url,
        "reasoning_effort": args.reasoning_effort,
        "sem_relation": args.sem_relation,
        "sem_question": args.sem_question,
        "sem_verifier": args.sem_verifier,
        "max_retries": args.max_retries,
        "window_strategy": args.window_strategy,
    }
    with open(os.path.join(log_dirs["processing_logs_dir"], "full_corpus_log.json"), "w") as handle:
        json.dump(batch_log, handle, indent=2)

    print(f"Saved agentic outputs to: {output_dir}")
    print(f"Successful documents: {successful_docs}/{len(documents)}")
    return {
        "log_dirs": log_dirs,
        "processing_stats": {
            "total_documents": len(documents),
            "gsw_structures": successful_docs,
        },
    }


if __name__ == "__main__":
    main()
