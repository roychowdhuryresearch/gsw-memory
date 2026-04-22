import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.query_then_sleep.run_guided_sleep_batch import _build_sleep_stage_summary, _merge_bridge_dicts


def test_build_sleep_stage_summary_records_zero_bridge_reasons():
    summary = _build_sleep_stage_summary(
        report={
            "summary": {
                "rlm_metrics": {
                    "edges_explored": 5,
                    "worker_calls": 5,
                    "edges_skipped_no_path": 0,
                }
            },
            "errors": [{"error": "Empty worker response"}],
        },
        bridge_count=0,
        generation_mode="single_batch",
        generation_shards=[],
        run_output_dir="/tmp/run",
    )

    assert summary["candidate_count"] == 5
    assert summary["worker_proposals_generated"] == 5
    assert "empty_worker_outputs" in summary["likely_zero_bridge_reasons"]


def test_merge_bridge_dicts_deduplicates_shard_outputs():
    merged = _merge_bridge_dicts(
        [
            {"bridge_id": "b1", "confidence": 0.4, "source_docs": ["doc_1"]},
            {"bridge_id": "b1", "confidence": 0.8, "source_docs": ["doc_2"]},
            {"bridge_id": "b2", "confidence": 0.3, "source_docs": ["doc_9"]},
        ]
    )

    assert len(merged) == 2
    bridge_b1 = next(row for row in merged if row["bridge_id"] == "b1")
    assert bridge_b1["confidence"] == 0.8
    assert bridge_b1["source_docs"] == ["doc_1", "doc_2"]
