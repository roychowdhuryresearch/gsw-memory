#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gsw_memory.query_then_sleep import analyze_query_interactions
from gsw_memory.query_then_sleep.logging_utils import StageLogger, console
from gsw_memory.query_then_sleep.prompts import render_interaction_guidance_block
from gsw_memory.query_then_sleep.types import QueryInteractionTrace, QuerySubQuestionTrace, ToolCallTrace


def _load_traces(path: str) -> List[QueryInteractionTrace]:
    traces: List[QueryInteractionTrace] = []
    with open(path) as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            traces.append(
                QueryInteractionTrace(
                    question_id=str(payload.get("question_id", "") or ""),
                    question=str(payload.get("question", "") or ""),
                    doc_ids=list(payload.get("doc_ids", []) or []),
                    gold_answers=list(payload.get("gold_answers", []) or []),
                    sub_questions=[QuerySubQuestionTrace(**row) for row in payload.get("sub_questions", []) or []],
                    tool_calls=[ToolCallTrace(**row) for row in payload.get("tool_calls", []) or []],
                    final_answer=str(payload.get("final_answer", "") or ""),
                    final_reasoning=str(payload.get("final_reasoning", "") or ""),
                    exact_match=payload.get("exact_match"),
                    f1=payload.get("f1"),
                    found_relations=list(payload.get("found_relations", []) or []),
                    missing_relations=list(payload.get("missing_relations", []) or []),
                    bridge_evidence_used=bool(payload.get("bridge_evidence_used", False)),
                    bridge_evidence_injected=bool(payload.get("bridge_evidence_injected", False)),
                    bridge_evidence_used_in_answer=bool(payload.get("bridge_evidence_used_in_answer", False)),
                    bridge_ids_used=list(payload.get("bridge_ids_used", []) or []),
                    bridge_ids_seen=list(payload.get("bridge_ids_seen", []) or []),
                    bridge_ids_injected=list(payload.get("bridge_ids_injected", []) or []),
                    tool_iterations=int(payload.get("tool_iterations", 0) or 0),
                    status=str(payload.get("status", "ok") or "ok"),
                    error=str(payload.get("error", "") or ""),
                    metadata=dict(payload.get("metadata", {}) or {}),
                )
            )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze isolated query-agent traces into sleep guidance")
    parser.add_argument("--traces_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_index", type=int, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument(
        "--prior_guidance_path",
        default=None,
        help="Path to the previous batch's interaction_guidance.json. When provided, its natural_language_summary is passed to the summarizer so lessons carry forward across batches.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_logger = StageLogger(
        stage_name="analysis",
        output_dir=output_dir,
        verbose=args.verbose,
        show_thinking=args.show_thinking,
    )
    stage_logger.progress("analysis", "started", batch_index=int(args.batch_index), model=args.model or "")
    console.print(f"\nAnalyzing query interactions for batch {args.batch_index}")
    if args.model:
        console.print(f"  analysis model: {args.model}")
    if args.base_url:
        console.print(f"  base URL: {args.base_url}")

    try:
        traces = _load_traces(args.traces_path)

        # Load prior batch's guidance summary if available, to carry forward
        # persistent lessons across batches.
        prior_guidance_summary: str = ""
        if args.prior_guidance_path:
            prior_path = Path(args.prior_guidance_path)
            if prior_path.exists():
                try:
                    prior_payload = json.loads(prior_path.read_text())
                    prior_guidance_summary = str(
                        prior_payload.get("natural_language_summary", "") or ""
                    ).strip()
                except Exception as exc:
                    stage_logger.event(
                        "prior_guidance_load_failed",
                        {"prior_guidance_path": str(prior_path), "error": str(exc)},
                    )
            else:
                stage_logger.event(
                    "prior_guidance_missing",
                    {"prior_guidance_path": str(prior_path)},
                )

        stage_logger.event(
            "analysis_started",
            {
                "batch_index": int(args.batch_index),
                "trace_count": len(traces),
                "traces_path": args.traces_path,
                "prior_guidance_loaded": bool(prior_guidance_summary),
            },
        )
        summary = analyze_query_interactions(
            traces,
            batch_index=args.batch_index,
            model_name=args.model,
            base_url=args.base_url,
            cache_path=str(output_dir / "relation_classification_cache.json"),
            reasoning_effort=args.reasoning_effort,
            event_callback=stage_logger.event,
            prior_guidance_summary=prior_guidance_summary,
        )
    except Exception as exc:
        stage_logger.log_exception("Interaction analysis failed", exc)
        stage_logger.progress("analysis", "failed", batch_index=int(args.batch_index), error=str(exc))
        raise

    (output_dir / "interaction_guidance.json").write_text(json.dumps(summary.as_dict(), indent=2))
    (output_dir / "interaction_guidance_prompt.txt").write_text(render_interaction_guidance_block(summary.as_dict()))
    stage_logger.progress(
        "analysis",
        "completed",
        batch_index=int(args.batch_index),
        questions_answered=summary.questions_answered,
        avg_f1=summary.avg_f1,
    )
    stage_logger.event(
        "analysis_finished",
        {
            "batch_index": int(args.batch_index),
            "questions_answered": summary.questions_answered,
            "avg_f1": summary.avg_f1,
            "top_missing_relations": [row.get("relation_label", "") for row in summary.relation_gap_profile[:5]],
        },
    )
    stage_logger.close()


if __name__ == "__main__":
    main()
