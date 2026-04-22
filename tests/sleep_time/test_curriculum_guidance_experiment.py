import json
import os
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.sleep_time import run_bridge_test
from playground.sleep_time import run_curriculum_guidance_experiment as guidance_experiment
from gsw_memory.sleep_time.curriculum import (
    BatchFeedbackBrief,
    BatchQueryItem,
    QueryAnswerResult,
    QueryBatchResult,
)


class _StubEntitySearcher:
    def __init__(self, *args, **kwargs):
        self.gsw_by_doc_id = {}
        self.entities = []

    def _embed_query(self, texts):
        return [[0.0, float(len(str(text)))] for text in texts]


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_fake_arm_output(arm_dir: Path, manifest_questions, *, exact_scores, bridge_ids, guidance_disabled):
    registry = []
    for family, bridge_id in bridge_ids.items():
        registry.append(
            {
                "bridge_id": bridge_id,
                "question": f"Bridge for {family}",
                "reverse_question": f"Reverse for {family}",
                "pattern_tags": [family],
                "pattern_key": family,
                "forward_pattern": {"pattern_key": family, "relation_label": family},
                "reverse_pattern": {"pattern_key": family, "relation_label": family},
            }
        )

    _write_json(
        arm_dir / "bridge_test_results.json",
        {
            "metadata": {
                "curriculum_disable_guidance": guidance_disabled,
                "curriculum_batch_size": 1,
                "curriculum_seed_batch_size": 1,
            },
            "summary": {
                "num_batches": len(manifest_questions),
                "total_queries": len(manifest_questions),
                "total_accepted_bridges": len(registry),
                "curriculum_bridge_registry": registry,
            },
        },
    )

    for batch_index, entry in enumerate(manifest_questions):
        batch_dir = arm_dir / f"batch_{batch_index}"
        bridge_id = bridge_ids[entry["family"]]
        exact_match = exact_scores[entry["question_id"]]
        bridge_used = exact_match >= 1.0
        _write_json(
            batch_dir / "query_answer_results.json",
            {
                "batch_index": batch_index,
                "query_results": [
                    {
                        "question_id": entry["question_id"],
                        "question": entry["question"],
                        "predicted_answer": f"pred-{entry['question_id']}",
                        "full_response": f"Answer: pred-{entry['question_id']}",
                        "exact_match": exact_match,
                        "f1": exact_match,
                        "bridge_evidence_used": bridge_used,
                        "retrieved_bridge_hit_count": 1,
                        "kept_bridge_hit_count": 1 if bridge_used else 0,
                        "top_evidence": [
                            {
                                "source_type": "bridge",
                                "bridge_id": bridge_id,
                                "question": f"Bridge for {entry['family']}",
                                "final_score": 0.9,
                            }
                        ],
                        "retrieval_trace": {
                            "raw_bridge_hits": [
                                {"source_type": "bridge", "bridge_id": bridge_id, "question": f"Bridge for {entry['family']}"}
                            ],
                            "kept_candidates": (
                                [{"source_type": "bridge", "bridge_id": bridge_id, "question": f"Bridge for {entry['family']}"}]
                                if bridge_used
                                else []
                            ),
                        },
                    }
                ],
                "overall_metrics": {"ExactMatch": exact_match, "F1": exact_match, "scored_queries": 1},
                "feedback_brief": {"summary_lines": []},
            },
        )
        _write_json(
            batch_dir / "bridge_registry_snapshot.json",
            {
                "batch_index": batch_index,
                "added_bridge_count": 1,
                "registry_bridge_count": len(registry),
            },
        )


def _write_progress_event(batch_dir: Path, event: dict):
    batch_dir.mkdir(parents=True, exist_ok=True)
    progress_path = batch_dir / "curriculum_progress.jsonl"
    with open(progress_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _write_shard_metadata(shard_dir: Path, payload: dict):
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "shard_metadata.json").write_text(json.dumps(payload))


def _write_tool_events(shard_dir: Path, run_id: str, events: list[dict]):
    tool_log_path = shard_dir / run_id / "logs" / "tool_calls.jsonl"
    tool_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tool_log_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def test_lookup_manifest_tracks_expected_questions():
    manifest = guidance_experiment.load_experiment_manifest(guidance_experiment.DEFAULT_MANIFEST_PATH)

    indices = [row["original_index"] for row in manifest["questions"]]
    families = [row["family"] for row in manifest["questions"]]
    splits = [row["split"] for row in manifest["questions"]]

    assert indices == [10, 9, 28, 61, 119, 277, 184, 283]
    assert families == [
        "director_birth_place",
        "director_nationality",
        "director_birth_place",
        "director_nationality",
        "director_birth_place",
        "director_nationality",
        "director_birth_place",
        "director_nationality",
    ]
    assert splits == ["train", "train", "train", "train", "test", "test", "test", "test"]


def test_run_curriculum_disables_generation_guidance(monkeypatch, tmp_path):
    captured_guidance = []

    def _fake_run_doc_batch(*, batch_index, doc_ids, args, output_dir, guidance_payload=None):
        captured_guidance.append(guidance_payload)
        return {"bridges": [], "num_bridges": 0, "doc_ids": list(doc_ids), "batch_index": batch_index}

    def _fake_answer_query_batch(batch_index, batch_items, entity_searcher, bridge_registry, args):
        query_results = [
            QueryAnswerResult(
                question_id=item.question_id,
                question=item.question,
                predicted_answer="",
                full_response="",
                exact_match=0.0,
                f1=0.0,
                bridge_evidence_used=False,
                answer_source_mix={"doc_entity": 1},
                top_evidence=[],
                patterns=["dummy.lookup.other"],
                gold_answers=[],
            )
            for item in batch_items
        ]
        return QueryBatchResult(
            batch_index=batch_index,
            query_results=query_results,
            overall_metrics={"ExactMatch": 0.0, "F1": 0.0, "scored_queries": len(query_results)},
            feedback_brief=BatchFeedbackBrief(
                high_demand_patterns=[],
                low_score_patterns=[],
                doc_dominant_patterns=[],
                bridge_helpful_patterns=[],
                summary_lines=[],
            ),
        )

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _fake_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)

    args = SimpleNamespace(
        curriculum_batch_size=1,
        curriculum_seed_batch_size=1,
        bridge_prompt_exemplar_top_k=3,
        bridge_query_top_k=5,
        curriculum_disable_guidance=True,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
    )
    items = [
        BatchQueryItem(question="What is the place of birth of the director of film A?", doc_ids=["doc_0"], question_id="q0"),
        BatchQueryItem(question="What nationality is the director of film B?", doc_ids=["doc_1"], question_id="q1"),
    ]

    result = run_bridge_test.run_curriculum(items, args)

    assert captured_guidance == [None, None]
    assert result["batch_results"][0]["generation_guidance"] is None
    assert result["batch_results"][1]["generation_guidance"] is None


def test_collect_arm_report_and_comparison_export_retrieved_family_bridges(tmp_path):
    manifest_questions = [
        {
            "order_index": 0,
            "original_index": 10,
            "question_id": "q_train_birth",
            "question": "What is the place of birth of the director of film A?",
            "family": "director_birth_place",
            "split": "train",
            "expected_pattern_hint": "place_of_birth.lookup.place",
        },
        {
            "order_index": 1,
            "original_index": 9,
            "question_id": "q_train_nat",
            "question": "What nationality is the director of film B?",
            "family": "director_nationality",
            "split": "train",
            "expected_pattern_hint": "director_nationality.lookup.region",
        },
        {
            "order_index": 2,
            "original_index": 119,
            "question_id": "q_test_birth",
            "question": "What is the place of birth of the director of film C?",
            "family": "director_birth_place",
            "split": "test",
            "expected_pattern_hint": "place_of_birth.lookup.place",
        },
        {
            "order_index": 3,
            "original_index": 277,
            "question_id": "q_test_nat",
            "question": "What nationality is the director of film D?",
            "family": "director_nationality",
            "split": "test",
            "expected_pattern_hint": "director_nationality.lookup.region",
        },
    ]
    bridge_ids = {
        "director_birth_place": "bridge_birth",
        "director_nationality": "bridge_nat",
    }
    on_dir = tmp_path / "guidance_on"
    off_dir = tmp_path / "guidance_off"
    _write_fake_arm_output(
        on_dir,
        manifest_questions,
        exact_scores={
            "q_train_birth": 1.0,
            "q_train_nat": 1.0,
            "q_test_birth": 1.0,
            "q_test_nat": 0.0,
        },
        bridge_ids=bridge_ids,
        guidance_disabled=False,
    )
    _write_fake_arm_output(
        off_dir,
        manifest_questions,
        exact_scores={
            "q_train_birth": 0.0,
            "q_train_nat": 0.0,
            "q_test_birth": 0.0,
            "q_test_nat": 0.0,
        },
        bridge_ids=bridge_ids,
        guidance_disabled=True,
    )

    on_report = guidance_experiment.collect_arm_report(
        arm_name="guidance_on",
        arm_output_dir=on_dir,
        manifest_questions=manifest_questions,
    )
    off_report = guidance_experiment.collect_arm_report(
        arm_name="guidance_off",
        arm_output_dir=off_dir,
        manifest_questions=manifest_questions,
    )

    comparison = guidance_experiment.build_comparison_report(
        manifest={
            "dataset": "synthetic",
            "family_label": "mixed_lookup_director_birth_place_and_nationality",
            "questions_path": "questions.json",
            "corpus_path": "corpus.json",
        },
        ordered_manifest_questions=manifest_questions,
        arm_reports={"guidance_on": on_report, "guidance_off": off_report},
    )

    assert on_report["overall_test_metrics"]["exact_match"] == 0.5
    assert off_report["overall_test_metrics"]["exact_match"] == 0.0
    assert comparison["delta"]["test_exact_match"] == 0.5
    assert comparison["arms"]["guidance_on"]["per_family_test_metrics"]["director_birth_place"]["exact_match"] == 1.0
    assert comparison["arms"]["guidance_off"]["per_family_test_metrics"]["director_birth_place"]["exact_match"] == 0.0
    birth_export = on_dir / "retrieved_family_bridges" / "director_birth_place.json"
    nat_export = on_dir / "retrieved_family_bridges" / "director_nationality.json"
    assert birth_export.exists()
    assert nat_export.exists()
    birth_payload = json.loads(birth_export.read_text())
    assert birth_payload["bridge_count"] == 1
    assert birth_payload["bridges"][0]["bridge_id"] == "bridge_birth"
    assert birth_payload["bridges"][0]["used_in_correct_answer_count"] == 2


def test_build_runner_command_forwards_fast_curriculum_flags(tmp_path):
    subset_questions_path = tmp_path / "selected_questions.json"
    subset_questions_path.write_text("[]")
    manifest = {
        "corpus_path": "playground_data/2wikimultihopqa_corpus.json",
        "questions": [{}, {}, {}, {}],
        "curriculum_batch_size": 1,
        "curriculum_seed_batch_size": 1,
    }
    args = SimpleNamespace(
        python_executable="python",
        gsw_path="/tmp/gsw",
        bridge_query_top_k=5,
        bridge_prompt_exemplar_top_k=3,
        pipeline_mode="hybrid",
        hybrid_scope="doc_edge",
        model="test-model",
        max_tokens=123,
        reasoning_effort="medium",
        edge_max_depth=3,
        edge_max_calls=5,
        edge_max_tokens=12000,
        max_optional_docs_per_edge=2,
        edge_parallel_workers=2,
        parallel_progress_heartbeat_seconds=10.0,
        parallel_stuck_warning_seconds=60.0,
        verifier_parallel_workers=4,
        edge_max_accepted_bridges=8,
        embedding_gpu_memory_utilization=0.5,
        base_url=None,
        root_model="root-model",
        worker_model="worker-model",
        edge_parallel_enabled=True,
        verbose=True,
        show_thinking=True,
        curriculum_batch_size=2,
        curriculum_seed_batch_size=2,
        curriculum_generation_parallel_enabled=True,
        curriculum_generation_parallel_workers=3,
    )

    command = guidance_experiment.build_runner_command(
        subset_questions_path=subset_questions_path,
        manifest=manifest,
        output_dir=tmp_path / "out",
        args=args,
        disable_guidance=True,
    )

    assert "--curriculum_batch_size" in command
    assert command[command.index("--curriculum_batch_size") + 1] == "2"
    assert "--curriculum_seed_batch_size" in command
    assert command[command.index("--curriculum_seed_batch_size") + 1] == "2"
    assert "--curriculum_generation_parallel_enabled" in command
    assert "--curriculum_generation_parallel_workers" in command
    assert command[command.index("--curriculum_generation_parallel_workers") + 1] == "3"
    assert "--curriculum_disable_guidance" in command


def test_format_progress_status_uses_curriculum_progress_event():
    snapshot = {
        "batch_index": 2,
        "phases": {
            "bridge_generation_shards": {
                "phase": "bridge_generation_shards",
                "status": "progress",
                "completed": 1,
                "total": 4,
                "unit": "shards",
            },
            "bridge_pattern_classification": {
                "phase": "bridge_pattern_classification",
                "status": "progress",
                "completed": 120,
                "total": 206,
                "unit": "bridges",
                "llm_tagged_surface_count": 240,
                "uncached_surface_count": 260,
                "failed_surface_count": 0,
            },
            "bridge_index_rebuild": {
                "phase": "bridge_index_rebuild",
                "status": "completed",
                "completed": 1,
                "total": 1,
                "unit": "rebuild",
                "surface_count": 412,
            },
            "query_answering": {
                "phase": "query_answering",
                "status": "progress",
                "completed": 1,
                "total": 2,
                "unit": "queries",
                "question_id": "q123",
                "bridge_evidence_used": True,
                "exact_match": 1.0,
                "f1": 1.0,
            },
        },
    }

    line = guidance_experiment._format_progress_status("guidance_on", snapshot)

    assert line == (
        "[guidance_on] batch 2 | shards 1/4 | patterns 120/206 llm=240/260 | "
        "index done surfaces=412 | queries 1/2 q=q123 em=1.00 f1=1.00 bridge=y"
    )


def test_latest_batch_progress_snapshot_tracks_current_batch_and_latest_phase_events(tmp_path):
    _write_progress_event(
        tmp_path / "batch_0",
        {
            "timestamp": "2026-04-02T10:00:00Z",
            "batch_index": 0,
            "phase": "bridge_generation_shards",
            "status": "completed",
            "completed": 2,
            "total": 2,
            "unit": "shards",
        },
    )
    _write_progress_event(
        tmp_path / "batch_1",
        {
            "timestamp": "2026-04-02T10:01:00Z",
            "batch_index": 1,
            "phase": "bridge_generation_shards",
            "status": "completed",
            "completed": 2,
            "total": 2,
            "unit": "shards",
            "merged_bridge_count": 7,
        },
    )
    _write_progress_event(
        tmp_path / "batch_1",
        {
            "timestamp": "2026-04-02T10:01:01Z",
            "batch_index": 1,
            "phase": "query_answering",
            "status": "progress",
            "completed": 1,
            "total": 2,
            "unit": "queries",
            "question_id": "q1",
        },
    )

    snapshot = guidance_experiment._latest_batch_progress_snapshot(tmp_path)

    assert snapshot is not None
    assert snapshot["batch_index"] == 1
    assert snapshot["phases"]["bridge_generation_shards"]["merged_bridge_count"] == 7
    assert snapshot["phases"]["query_answering"]["question_id"] == "q1"


def test_latest_batch_progress_snapshot_prefers_newer_progress_file_over_higher_batch_index(tmp_path):
    old_batch_dir = tmp_path / "batch_7"
    new_batch_dir = tmp_path / "batch_0"
    _write_progress_event(
        old_batch_dir,
        {
            "timestamp": "2026-04-01T10:00:00Z",
            "batch_index": 7,
            "phase": "query_answering",
            "status": "progress",
            "completed": 1,
            "total": 1,
            "unit": "queries",
            "question_id": "old_q",
        },
    )
    _write_progress_event(
        new_batch_dir,
        {
            "timestamp": "2026-04-02T10:00:00Z",
            "batch_index": 0,
            "phase": "bridge_generation_shards",
            "status": "progress",
            "completed": 1,
            "total": 2,
            "unit": "shards",
        },
    )

    old_progress = old_batch_dir / "curriculum_progress.jsonl"
    new_progress = new_batch_dir / "curriculum_progress.jsonl"
    old_stat = old_progress.stat()
    new_stat = new_progress.stat()
    os.utime(old_progress, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    os.utime(new_progress, ns=(new_stat.st_atime_ns, new_stat.st_mtime_ns + 1_000_000_000))

    snapshot = guidance_experiment._latest_batch_progress_snapshot(tmp_path)

    assert snapshot is not None
    assert snapshot["batch_index"] == 0
    assert "bridge_generation_shards" in snapshot["phases"]
    assert "query_answering" not in snapshot["phases"]


def test_latest_batch_progress_snapshot_includes_live_shard_status(tmp_path):
    batch_dir = tmp_path / "batch_1"
    _write_progress_event(
        batch_dir,
        {
            "timestamp": "2026-04-02T10:01:00Z",
            "batch_index": 1,
            "phase": "bridge_generation_shards",
            "status": "progress",
            "completed": 1,
            "total": 2,
            "unit": "shards",
        },
    )
    shard_dir = batch_dir / "generation_shards" / "shard_0"
    _write_shard_metadata(
        shard_dir,
        {
            "batch_index": 1,
            "shard_id": 0,
            "question_ids": ["q0"],
            "questions": ["What is the place of birth of the director of film A?"],
            "doc_ids": ["doc_0", "doc_1"],
        },
    )
    _write_tool_events(
        shard_dir,
        "run_20260402_100100",
        [
            {
                "event_type": "parallel_batch_progress",
                "timestamp": "2026-04-02T10:01:05",
                "data": {
                    "doc_id": "doc_0",
                    "completed": 1,
                    "total": 4,
                    "elapsed_seconds": 12.0,
                    "inflight_edges": [
                        {
                            "doc_id": "doc_0",
                            "entity_name": "Film A",
                            "neighbor_name": "Director A",
                            "relationship": "directed by",
                        }
                    ],
                },
            },
            {
                "event_type": "rlm_doc_summary",
                "timestamp": "2026-04-02T10:01:07",
                "data": {
                    "entity": "doc_0",
                    "bridges_created": 2,
                    "document_marked_explored": True,
                    "document_exploration": {
                        "doc_id": "doc_0",
                        "total_explored": 1,
                        "total_docs": 2,
                        "remaining_docs": ["doc_1"],
                    },
                },
            },
        ],
    )

    snapshot = guidance_experiment._latest_batch_progress_snapshot(tmp_path)

    assert snapshot is not None
    assert snapshot["batch_index"] == 1
    assert len(snapshot["shards"]) == 1
    shard_status = snapshot["shards"][0]
    assert shard_status["shard_id"] == 0
    assert shard_status["docs_total"] == 2
    assert shard_status["docs_completed"] == 1
    assert shard_status["docs_remaining"] == 1
    assert shard_status["current_doc_id"] == ""
    assert shard_status["state"] == "next_doc"
    assert shard_status["current_edge_label"] == "Film A->Director A (directed by)"


def test_load_shard_live_status_prefers_document_exploration_totals_from_mark_document_explored(tmp_path):
    shard_dir = tmp_path / "batch_0" / "generation_shards" / "shard_0"
    _write_shard_metadata(
        shard_dir,
        {
            "batch_index": 0,
            "shard_id": 0,
            "question_ids": ["q0"],
            "questions": ["What nationality is the director of film A?"],
            "doc_ids": ["doc_0", "doc_1", "doc_2"],
        },
    )
    _write_tool_events(
        shard_dir,
        "run_20260402_101500",
        [
            {
                "event_type": "parallel_batch_progress",
                "timestamp": "2026-04-02T10:15:00",
                "data": {
                    "doc_id": "doc_2",
                    "completed": 2,
                    "total": 5,
                    "elapsed_seconds": 22.0,
                    "inflight_edges": [],
                },
            },
            {
                "event_type": "rlm_doc_summary",
                "timestamp": "2026-04-02T10:15:01",
                "data": {
                    "entity": "doc_1",
                    "document_marked_explored": True,
                    "document_exploration": {
                        "doc_id": "doc_1",
                        "total_explored": 2,
                        "total_docs": 3,
                        "remaining_docs": ["doc_2"],
                    },
                },
            },
        ],
    )

    shard_status = guidance_experiment._load_shard_live_status(shard_dir)

    assert shard_status["docs_total"] == 3
    assert shard_status["docs_completed"] == 2
    assert shard_status["docs_remaining"] == 1


def test_load_shard_live_status_ignores_stale_tool_log_older_than_current_metadata(tmp_path):
    shard_dir = tmp_path / "batch_0" / "generation_shards" / "shard_0"
    old_run_dir = shard_dir / "run_20260401_100000"
    _write_tool_events(
        shard_dir,
        "run_20260401_100000",
        [
            {
                "event_type": "rlm_doc_summary",
                "timestamp": "2026-04-01T10:00:00",
                "data": {"entity": "doc_old_0"},
            },
            {
                "event_type": "rlm_doc_summary",
                "timestamp": "2026-04-01T10:00:01",
                "data": {"entity": "doc_old_1"},
            },
        ],
    )
    old_tool_log = old_run_dir / "logs" / "tool_calls.jsonl"
    old_stat = old_tool_log.stat()
    os.utime(old_tool_log, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    _write_shard_metadata(
        shard_dir,
        {
            "batch_index": 0,
            "shard_id": 0,
            "question_ids": ["q0"],
            "questions": ["What is the place of birth of the director of film A?"],
            "doc_ids": ["doc_0", "doc_1"],
        },
    )
    metadata_path = shard_dir / "shard_metadata.json"
    metadata_stat = metadata_path.stat()
    os.utime(metadata_path, ns=(metadata_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))

    shard_status = guidance_experiment._load_shard_live_status(shard_dir)

    assert shard_status["docs_total"] == 2
    assert shard_status["docs_completed"] == 0
    assert shard_status["docs_remaining"] == 2
    assert shard_status["state"] == "pending"


def test_apply_shard_status_render_cache_keeps_last_live_state_within_ttl(tmp_path):
    guidance_experiment._SHARD_STATUS_RENDER_CACHE.clear()
    shard_dir = tmp_path / "batch_0" / "generation_shards" / "shard_0"
    live_status = {
        "shard_id": 0,
        "question_ids": ["q0"],
        "questions": ["Q"],
        "doc_ids": ["doc_0"],
        "docs_total": 10,
        "docs_completed": 0,
        "docs_remaining": 10,
        "current_doc_id": "doc_86",
        "edge_completed": 0,
        "edge_total": 4,
        "inflight_edge_count": 4,
        "elapsed_seconds": 2.0,
        "worker_inflight_seconds": None,
        "current_edge_label": "1963->Michael Govan (born in)",
        "state": "running",
        "latest_timestamp": "2026-04-02T16:29:04",
        "tool_log_path": str(shard_dir / "run_x" / "logs" / "tool_calls.jsonl"),
    }
    pending_status = {
        **live_status,
        "state": "pending",
        "current_doc_id": "",
        "edge_completed": None,
        "edge_total": None,
        "inflight_edge_count": 0,
        "elapsed_seconds": None,
        "current_edge_label": "",
        "tool_log_path": "",
    }

    guidance_experiment._apply_shard_status_render_cache(shard_dir, live_status, now_monotonic=100.0)
    rendered_status = guidance_experiment._apply_shard_status_render_cache(
        shard_dir,
        pending_status,
        now_monotonic=103.0,
    )

    assert rendered_status["state"] == "running"
    assert rendered_status["state_stale"] is True
    assert rendered_status["stale_seconds"] == pytest.approx(3.0)
    assert rendered_status["current_doc_id"] == "doc_86"


def test_apply_shard_status_render_cache_uses_waiting_after_ttl(tmp_path):
    guidance_experiment._SHARD_STATUS_RENDER_CACHE.clear()
    shard_dir = tmp_path / "batch_0" / "generation_shards" / "shard_0"
    live_status = {
        "shard_id": 0,
        "question_ids": ["q0"],
        "questions": ["Q"],
        "doc_ids": ["doc_0"],
        "docs_total": 10,
        "docs_completed": 0,
        "docs_remaining": 10,
        "current_doc_id": "doc_86",
        "edge_completed": 0,
        "edge_total": 4,
        "inflight_edge_count": 4,
        "elapsed_seconds": 2.0,
        "worker_inflight_seconds": None,
        "current_edge_label": "1963->Michael Govan (born in)",
        "state": "running",
        "latest_timestamp": "2026-04-02T16:29:04",
        "tool_log_path": str(shard_dir / "run_x" / "logs" / "tool_calls.jsonl"),
    }
    pending_status = {
        **live_status,
        "state": "pending",
        "current_doc_id": "",
        "edge_completed": None,
        "edge_total": None,
        "inflight_edge_count": 0,
        "elapsed_seconds": None,
        "current_edge_label": "",
        "tool_log_path": "",
    }

    guidance_experiment._apply_shard_status_render_cache(shard_dir, live_status, now_monotonic=100.0)
    rendered_status = guidance_experiment._apply_shard_status_render_cache(
        shard_dir,
        pending_status,
        now_monotonic=107.5,
    )

    assert rendered_status["state"] == "waiting"
    assert rendered_status["state_stale"] is True
    assert rendered_status["stale_seconds"] == pytest.approx(7.5)
    assert rendered_status["current_edge_label"] == "1963->Michael Govan (born in)"


def test_print_compact_progress_uses_plain_mode_when_stdout_is_not_tty(monkeypatch):
    class _FakeStdout(StringIO):
        def isatty(self):
            return False

    fake_stdout = _FakeStdout()
    monkeypatch.setattr(guidance_experiment.sys, "stdout", fake_stdout)
    monkeypatch.delenv("GSW_EXPERIMENT_PROGRESS_MODE", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    guidance_experiment._LAST_PROGRESS_RENDERED_LINES = 0

    guidance_experiment._print_compact_progress("[guidance_on] batch 0 | shards 0/2")

    assert fake_stdout.getvalue() == "[guidance_on] batch 0 | shards 0/2\n"
    assert guidance_experiment._LAST_PROGRESS_RENDERED_LINES == 0


def test_run_experiment_arm_writes_separate_logs_and_uses_progress_files(monkeypatch, tmp_path):
    subset_questions_path = tmp_path / "selected_questions.json"
    subset_questions_path.write_text("[]")
    manifest = {
        "corpus_path": "playground_data/2wikimultihopqa_corpus.json",
        "questions": [{}, {}],
        "curriculum_batch_size": 1,
        "curriculum_seed_batch_size": 1,
    }
    args = SimpleNamespace(
        python_executable="python",
        gsw_path="/tmp/gsw",
        bridge_query_top_k=5,
        bridge_prompt_exemplar_top_k=3,
        pipeline_mode="hybrid",
        hybrid_scope="doc_edge",
        model="test-model",
        max_tokens=123,
        reasoning_effort="medium",
        edge_max_depth=3,
        edge_max_calls=5,
        edge_max_tokens=12000,
        max_optional_docs_per_edge=2,
        edge_parallel_workers=2,
        parallel_progress_heartbeat_seconds=10.0,
        parallel_stuck_warning_seconds=60.0,
        verifier_parallel_workers=4,
        edge_max_accepted_bridges=8,
        embedding_gpu_memory_utilization=0.5,
        base_url=None,
        root_model=None,
        worker_model=None,
        edge_parallel_enabled=False,
        verbose=False,
        show_thinking=False,
        curriculum_batch_size=2,
        curriculum_seed_batch_size=2,
        curriculum_generation_parallel_enabled=True,
        curriculum_generation_parallel_workers=2,
    )
    output_root = tmp_path / "experiment_out"
    captured_progress_lines = []
    launch_kwargs = {}

    def _fake_print_compact_progress(line: str):
        captured_progress_lines.append(line)

    class _FakePopen:
        def __init__(self, command, cwd=None, text=None, stdout=None, stderr=None, start_new_session=None, preexec_fn=None):
            output_dir = Path(command[command.index("--output_dir") + 1])
            stdout.write("runner stdout\n")
            stderr.write("runner stderr\n")
            self.pid = 41001
            launch_kwargs["start_new_session"] = start_new_session
            launch_kwargs["preexec_fn"] = preexec_fn
            shard_dir = output_dir / "batch_0" / "generation_shards" / "shard_0"
            _write_shard_metadata(
                shard_dir,
                {
                    "batch_index": 0,
                    "shard_id": 0,
                    "question_ids": ["q0"],
                    "questions": ["What is the place of birth of the director of film A?"],
                    "doc_ids": ["doc_0", "doc_1"],
                },
            )
            _write_tool_events(
                shard_dir,
                "run_20260402_100000",
                [
                    {
                        "event_type": "parallel_batch_progress",
                        "timestamp": "2026-04-02T10:00:00",
                        "data": {
                            "doc_id": "doc_0",
                            "completed": 1,
                            "total": 4,
                            "elapsed_seconds": 8.0,
                            "inflight_edges": [
                                {
                                    "doc_id": "doc_0",
                                    "entity_name": "Film A",
                                    "neighbor_name": "Director A",
                                    "relationship": "directed by",
                                }
                            ],
                        },
                    },
                    {
                        "event_type": "rlm_doc_summary",
                        "timestamp": "2026-04-02T09:59:59",
                        "data": {
                            "entity": "doc_0",
                            "bridges_created": 1,
                            "document_marked_explored": True,
                            "document_exploration": {
                                "doc_id": "doc_0",
                                "total_explored": 1,
                                "total_docs": 2,
                                "remaining_docs": ["doc_1"],
                            },
                        },
                    },
                ],
            )
            _write_progress_event(
                output_dir / "batch_0",
                {
                    "timestamp": "2026-04-02T10:00:00Z",
                    "batch_index": 0,
                    "phase": "bridge_generation_shards",
                    "status": "progress",
                    "completed": 1,
                    "total": 2,
                    "unit": "shards",
                    "merged_bridge_count": 9,
                },
            )
            _write_progress_event(
                output_dir / "batch_0",
                {
                    "timestamp": "2026-04-02T10:00:01Z",
                    "batch_index": 0,
                    "phase": "bridge_pattern_classification",
                    "status": "progress",
                    "completed": 10,
                    "total": 20,
                    "unit": "bridges",
                    "llm_tagged_surface_count": 18,
                    "uncached_surface_count": 20,
                    "failed_surface_count": 0,
                },
            )
            _write_fake_arm_output(
                output_dir,
                [
                    {
                        "order_index": 0,
                        "original_index": 0,
                        "question_id": "q0",
                        "question": "What is the place of birth of the director of film A?",
                        "family": "director_birth_place",
                        "split": "train",
                    },
                    {
                        "order_index": 1,
                        "original_index": 1,
                        "question_id": "q1",
                        "question": "What nationality is the director of film B?",
                        "family": "director_nationality",
                        "split": "test",
                    },
                ],
                exact_scores={"q0": 1.0, "q1": 0.0},
                bridge_ids={
                    "director_birth_place": "bridge_birth",
                    "director_nationality": "bridge_nat",
                },
                guidance_disabled=False,
            )
            self.returncode = None
            self._poll_count = 0

        def poll(self):
            self._poll_count += 1
            if self._poll_count == 1:
                return None
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(guidance_experiment, "_print_compact_progress", _fake_print_compact_progress)
    monkeypatch.setattr(guidance_experiment.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(guidance_experiment.time, "sleep", lambda _seconds: None)

    arm_output_dir = guidance_experiment.run_experiment_arm(
        arm_name="guidance_on",
        subset_questions_path=subset_questions_path,
        manifest=manifest,
        output_root=output_root,
        args=args,
    )

    assert arm_output_dir == output_root / "guidance_on"
    assert (arm_output_dir / "runner_stdout.log").read_text() == "runner stdout\n"
    assert (arm_output_dir / "runner_stderr.log").read_text() == "runner stderr\n"
    assert launch_kwargs["start_new_session"] is True
    if sys.platform.startswith("linux"):
        assert callable(launch_kwargs["preexec_fn"])
    assert any("shards 1/2 merged=9" in line for line in captured_progress_lines)
    assert any("patterns 10/20 llm=18/20" in line for line in captured_progress_lines)


def test_run_experiment_arm_raises_with_log_paths_and_stderr_tail(monkeypatch, tmp_path):
    subset_questions_path = tmp_path / "selected_questions.json"
    subset_questions_path.write_text("[]")
    manifest = {
        "corpus_path": "playground_data/2wikimultihopqa_corpus.json",
        "questions": [{}, {}],
        "curriculum_batch_size": 1,
        "curriculum_seed_batch_size": 1,
    }
    args = SimpleNamespace(
        python_executable="python",
        gsw_path="/tmp/gsw",
        bridge_query_top_k=5,
        bridge_prompt_exemplar_top_k=3,
        pipeline_mode="hybrid",
        hybrid_scope="doc_edge",
        model="test-model",
        max_tokens=123,
        reasoning_effort="medium",
        edge_max_depth=3,
        edge_max_calls=5,
        edge_max_tokens=12000,
        max_optional_docs_per_edge=2,
        edge_parallel_workers=2,
        parallel_progress_heartbeat_seconds=10.0,
        parallel_stuck_warning_seconds=60.0,
        verifier_parallel_workers=4,
        edge_max_accepted_bridges=8,
        embedding_gpu_memory_utilization=0.5,
        base_url=None,
        root_model=None,
        worker_model=None,
        edge_parallel_enabled=False,
        verbose=False,
        show_thinking=False,
        curriculum_batch_size=2,
        curriculum_seed_batch_size=2,
        curriculum_generation_parallel_enabled=True,
        curriculum_generation_parallel_workers=2,
    )
    output_root = tmp_path / "experiment_out"

    class _FailingPopen:
        def __init__(self, command, cwd=None, text=None, stdout=None, stderr=None, start_new_session=None, preexec_fn=None):
            stdout.write("runner stdout\n")
            stderr.write("first error line\nsecond error line\n")
            self.pid = 41002
            self.returncode = 2

        def poll(self):
            return self.returncode

    monkeypatch.setattr(guidance_experiment.subprocess, "Popen", _FailingPopen)
    monkeypatch.setattr(guidance_experiment, "_print_compact_progress", lambda _line: None)
    monkeypatch.setattr(guidance_experiment, "_terminate_process_group", lambda _process, grace_seconds=guidance_experiment.PROCESS_TERMINATION_GRACE_SECONDS: None)

    with pytest.raises(RuntimeError) as exc_info:
        guidance_experiment.run_experiment_arm(
            arm_name="guidance_off",
            subset_questions_path=subset_questions_path,
            manifest=manifest,
            output_root=output_root,
            args=args,
        )

    message = str(exc_info.value)
    assert "guidance_off" in message
    assert "runner_stdout.log" in message
    assert "runner_stderr.log" in message
    assert "second error line" in message


def test_terminate_process_group_escalates_to_sigkill(monkeypatch):
    kill_calls = []

    class _FakeProcess:
        pid = 55123

        def poll(self):
            return None

    monkeypatch.setattr(guidance_experiment.os, "getpgid", lambda pid: 77123)
    monkeypatch.setattr(
        guidance_experiment.os,
        "killpg",
        lambda pgid, sig: kill_calls.append((pgid, sig)),
    )
    monotonic_values = iter([0.0, 0.0, 6.0])
    monkeypatch.setattr(guidance_experiment.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(guidance_experiment.time, "sleep", lambda _seconds: None)

    guidance_experiment._terminate_process_group(_FakeProcess(), grace_seconds=5.0)

    assert kill_calls == [
        (77123, guidance_experiment.signal.SIGTERM),
        (77123, guidance_experiment.signal.SIGKILL),
    ]


def test_run_experiment_arm_terminates_process_group_on_interrupt(monkeypatch, tmp_path):
    subset_questions_path = tmp_path / "selected_questions.json"
    subset_questions_path.write_text("[]")
    manifest = {
        "corpus_path": "playground_data/2wikimultihopqa_corpus.json",
        "questions": [{}, {}],
        "curriculum_batch_size": 1,
        "curriculum_seed_batch_size": 1,
    }
    args = SimpleNamespace(
        python_executable="python",
        gsw_path="/tmp/gsw",
        bridge_query_top_k=5,
        bridge_prompt_exemplar_top_k=3,
        pipeline_mode="hybrid",
        hybrid_scope="doc_edge",
        model="test-model",
        max_tokens=123,
        reasoning_effort="medium",
        edge_max_depth=3,
        edge_max_calls=5,
        edge_max_tokens=12000,
        max_optional_docs_per_edge=2,
        edge_parallel_workers=2,
        parallel_progress_heartbeat_seconds=10.0,
        parallel_stuck_warning_seconds=60.0,
        verifier_parallel_workers=4,
        edge_max_accepted_bridges=8,
        embedding_gpu_memory_utilization=0.5,
        base_url=None,
        root_model=None,
        worker_model=None,
        edge_parallel_enabled=False,
        verbose=False,
        show_thinking=False,
        curriculum_batch_size=2,
        curriculum_seed_batch_size=2,
        curriculum_generation_parallel_enabled=True,
        curriculum_generation_parallel_workers=2,
        probe_eval=False,
    )
    output_root = tmp_path / "experiment_out"
    launch_kwargs = {}
    terminated = []

    class _InterruptingPopen:
        def __init__(self, command, cwd=None, text=None, stdout=None, stderr=None, start_new_session=None, preexec_fn=None):
            stdout.write("runner stdout\n")
            stderr.write("runner stderr\n")
            self.pid = 55124
            self.returncode = None
            launch_kwargs["start_new_session"] = start_new_session
            launch_kwargs["preexec_fn"] = preexec_fn

        def poll(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(guidance_experiment.subprocess, "Popen", _InterruptingPopen)
    monkeypatch.setattr(guidance_experiment, "_print_compact_progress", lambda _line: None)
    monkeypatch.setattr(guidance_experiment, "_install_signal_handlers", lambda _handler: {})
    monkeypatch.setattr(guidance_experiment, "_restore_signal_handlers", lambda _handlers: None)
    monkeypatch.setattr(
        guidance_experiment,
        "_terminate_process_group",
        lambda process, grace_seconds=guidance_experiment.PROCESS_TERMINATION_GRACE_SECONDS: terminated.append((getattr(process, "pid", None), grace_seconds)),
    )

    with pytest.raises(KeyboardInterrupt):
        guidance_experiment.run_experiment_arm(
            arm_name="guidance_on",
            subset_questions_path=subset_questions_path,
            manifest=manifest,
            output_root=output_root,
            args=args,
        )

    assert terminated == [(55124, guidance_experiment.PROCESS_TERMINATION_GRACE_SECONDS)]
    assert launch_kwargs["start_new_session"] is True
    if sys.platform.startswith("linux"):
        assert callable(launch_kwargs["preexec_fn"])


def test_experiment_main_runs_both_arms_and_writes_comparison(monkeypatch, tmp_path):
    questions = [
        {
            "_id": "q0",
            "question": "What is the place of birth of the director of film A?",
            "answer": "X",
            "answer_aliases": ["X"],
            "context": [["DocA", ["sent"]]],
            "supporting_facts": [["DocA", 0]],
        },
        {
            "_id": "q1",
            "question": "What nationality is the director of film B?",
            "answer": "Y",
            "answer_aliases": ["Y"],
            "context": [["DocB", ["sent"]]],
            "supporting_facts": [["DocB", 0]],
        },
        {
            "_id": "q2",
            "question": "What is the place of birth of the director of film C?",
            "answer": "Z",
            "answer_aliases": ["Z"],
            "context": [["DocC", ["sent"]]],
            "supporting_facts": [["DocC", 0]],
        },
        {
            "_id": "q3",
            "question": "What nationality is the director of film D?",
            "answer": "W",
            "answer_aliases": ["W"],
            "context": [["DocD", ["sent"]]],
            "supporting_facts": [["DocD", 0]],
        },
    ]
    questions_path = tmp_path / "questions.json"
    corpus_path = tmp_path / "corpus.json"
    questions_path.write_text(json.dumps(questions))
    corpus_path.write_text(json.dumps([{"title": "DocA"}, {"title": "DocB"}, {"title": "DocC"}, {"title": "DocD"}]))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "questions_path": str(questions_path),
                "corpus_path": str(corpus_path),
                "family_label": "mixed_lookup_director_birth_place_and_nationality",
                "curriculum_batch_size": 1,
                "curriculum_seed_batch_size": 1,
                "questions": [
                    {"original_index": 0, "question_id": "q0", "family": "director_birth_place", "split": "train"},
                    {"original_index": 1, "question_id": "q1", "family": "director_nationality", "split": "train"},
                    {"original_index": 2, "question_id": "q2", "family": "director_birth_place", "split": "test"},
                    {"original_index": 3, "question_id": "q3", "family": "director_nationality", "split": "test"},
                ],
            }
        )
    )
    output_root = tmp_path / "experiment_out"

    class _FakePopen:
        def __init__(self, command, cwd=None, text=None, stdout=None, stderr=None, start_new_session=None, preexec_fn=None):
            output_dir = Path(command[command.index("--output_dir") + 1])
            guidance_disabled = "--curriculum_disable_guidance" in command
            self.pid = 41003
            manifest_questions = [
                {
                    "order_index": 0,
                    "original_index": 0,
                    "question_id": "q0",
                    "question": questions[0]["question"],
                    "family": "director_birth_place",
                    "split": "train",
                },
                {
                    "order_index": 1,
                    "original_index": 1,
                    "question_id": "q1",
                    "question": questions[1]["question"],
                    "family": "director_nationality",
                    "split": "train",
                },
                {
                    "order_index": 2,
                    "original_index": 2,
                    "question_id": "q2",
                    "question": questions[2]["question"],
                    "family": "director_birth_place",
                    "split": "test",
                },
                {
                    "order_index": 3,
                    "original_index": 3,
                    "question_id": "q3",
                    "question": questions[3]["question"],
                    "family": "director_nationality",
                    "split": "test",
                },
            ]
            stdout.write("runner stdout\n")
            stderr.write("runner stderr\n")
            _write_fake_arm_output(
                output_dir,
                manifest_questions,
                exact_scores={
                    "q0": 1.0 if not guidance_disabled else 0.0,
                    "q1": 1.0 if not guidance_disabled else 0.0,
                    "q2": 1.0 if not guidance_disabled else 0.0,
                    "q3": 0.0,
                },
                bridge_ids={
                    "director_birth_place": "bridge_birth",
                    "director_nationality": "bridge_nat",
                },
                guidance_disabled=guidance_disabled,
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(guidance_experiment.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(guidance_experiment, "_print_compact_progress", lambda _line: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_curriculum_guidance_experiment.py",
            "--manifest",
            str(manifest_path),
            "--output_root",
            str(output_root),
            "--gsw_path",
            "/tmp/gsw",
        ],
    )

    guidance_experiment.main()

    comparison_path = output_root / "comparison.json"
    assert comparison_path.exists()
    comparison = json.loads(comparison_path.read_text())
    assert comparison["delta"]["test_exact_match"] == 0.5
    assert (output_root / "guidance_on" / "retrieved_family_bridges" / "director_birth_place.json").exists()
    assert (output_root / "guidance_off" / "retrieved_family_bridges" / "director_nationality.json").exists()
    assert (output_root / "guidance_on" / "runner_stdout.log").exists()
    assert (output_root / "guidance_off" / "runner_stderr.log").exists()
