import json
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.query_then_sleep import run_query_sleep_experiment as experiment


class _Item:
    def __init__(self, question_id: str, question: str, doc_ids):
        self.question_id = question_id
        self.question = question
        self.doc_ids = list(doc_ids)


def test_experiment_runner_writes_learning_curve_with_isolated_batch_dirs(monkeypatch, tmp_path):
    batches = [
        [_Item("q0", "Question 0", ["doc_0"])],
        [_Item("q1", "Question 1", ["doc_1"])],
    ]

    monkeypatch.setattr(experiment, "_build_batch_items", lambda *args, **kwargs: batches)
    monkeypatch.setattr(
        experiment,
        "_resolve_experiment_dataset",
        lambda **kwargs: ("dummy_questions.json", None, None),
    )

    def _fake_run_child_stage(stage_name, command, stage_output_dir):
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = stage_output_dir / "runner_stdout.log"
        stderr_path = stage_output_dir / "runner_stderr.log"
        stdout_path.write_text(f"{stage_name} ok\n")
        stderr_path.write_text("")

        if stage_name == "query":
            (stage_output_dir / "query_agent_results.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {
                                "question_id": "q0",
                                "question": "Question",
                                "doc_ids": ["doc_0"],
                                "status": "ok",
                            }
                        ],
                        "overall_metrics": {
                            "questions_answered": 1,
                            "F1": 0.5,
                            "ExactMatch": 0.0,
                            "bridge_usage_rate": 0.0,
                        },
                    }
                )
            )
            (stage_output_dir / "query_interaction_traces.jsonl").write_text("{}\n")
        elif stage_name == "analysis":
            (stage_output_dir / "interaction_guidance.json").write_text(
                json.dumps(
                    {
                        "guidance_mode": "interaction_v1",
                        "batch_index": 0,
                        "questions_answered": 1,
                        "avg_f1": 0.5,
                        "relation_gap_profile": [
                            {"relation_label": "death_place", "asked": 1, "found": 0, "missing": 1}
                        ],
                        "entity_gap_targets": [
                            {"entity": "Ermengarde of Tours", "relation_label": "death_place", "count": 1}
                        ],
                        "successful_relation_profile": [],
                        "missing_bridge_targets": [],
                        "natural_language_summary": "Focus on death_place.",
                        "supporting_examples": [],
                    }
                )
            )
            (stage_output_dir / "interaction_guidance_prompt.txt").write_text("Focus on death_place.")
        elif stage_name == "guided_sleep":
            (stage_output_dir / "sleep_guidance_payload.json").write_text("{}")
            snapshot_path = stage_output_dir / "bridge_registry_snapshot.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "batch_index": 1,
                        "registry_bridge_count": 1,
                        "all_bridge_records": [],
                    }
                )
            )
            (stage_output_dir / "guided_sleep_result.json").write_text(
                json.dumps(
                    {
                        "bridge_count": 1,
                        "snapshot_path": str(snapshot_path),
                        "generation_mode": "question_group_sharded",
                        "generation_shards": [{"shard_id": 0, "accepted_bridge_count": 1}],
                    }
                )
            )
        else:
            raise AssertionError(f"unexpected stage: {stage_name}")

        return {"stdout": stdout_path, "stderr": stderr_path}

    monkeypatch.setattr(experiment, "_run_child_stage", _fake_run_child_stage)

    args = [
        "prog",
        "--manifest",
        "dummy.json",
        "--output_root",
        str(tmp_path / "out"),
        "--gsw_path",
        str(tmp_path / "gsw"),
        "--model",
        "dummy-model",
    ]
    monkeypatch.setattr(sys, "argv", args)

    experiment.main()

    output_root = tmp_path / "out"
    learning_curve = json.loads((output_root / "learning_curve.json").read_text())
    assert len(learning_curve) == 2
    assert learning_curve[0]["avg_f1"] == 0.5
    assert learning_curve[0]["bridges_generated_for_next_batch"] == 1
    assert learning_curve[1]["bridges_before_batch"] == 1

    assert (output_root / "experiment_progress.jsonl").exists()
    assert (output_root / "batch_0" / "query" / "query_agent_results.json").exists()
    assert (output_root / "batch_0" / "query" / "runner_stdout.log").exists()
    assert (output_root / "batch_0" / "analysis" / "interaction_guidance.json").exists()
    assert (output_root / "batch_0" / "analysis" / "runner_stdout.log").exists()
    assert (output_root / "batch_0" / "sleep_for_batch_1" / "sleep_guidance_payload.json").exists()
    assert (output_root / "batch_0" / "sleep_for_batch_1" / "runner_stdout.log").exists()


def test_stage_command_builders_honor_stage_specific_model_and_base_url_flags(tmp_path):
    args = Namespace(
        questions_path="manifest.json",
        batch_size=8,
        seed_batch_size=8,
        gsw_path=str(tmp_path / "gsw"),
        model="default-model",
        base_url="http://default/v1",
        corpus_path=str(tmp_path / "corpus.json"),
        query_model="query-model",
        query_base_url="http://query/v1",
        analysis_model="analysis-model",
        analysis_base_url="http://analysis/v1",
        sleep_model="sleep-model",
        sleep_base_url="http://sleep/v1",
        root_model="root-model",
        worker_model="worker-model",
        query_max_iterations=8,
        bridge_query_top_k=5,
        embedding_gpu_memory_utilization=0.5,
        reasoning_effort="medium",
        verbose=False,
        show_thinking=False,
        pipeline_mode="hybrid",
        hybrid_scope="doc_edge",
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=3000,
        max_optional_docs_per_edge=2,
        edge_parallel_enabled=False,
        edge_parallel_workers=1,
        verifier_parallel_workers=4,
        edge_max_accepted_bridges=10,
        parallel_progress_heartbeat_seconds=10.0,
        parallel_stuck_warning_seconds=60.0,
        curriculum_generation_parallel_enabled=True,
        curriculum_generation_parallel_workers=3,
        max_tokens=500000,
        max_iterations=30,
        disable_bridge_verifier=False,
        bridge_verifier_threshold=0.7,
        bridge_verifier_fail_open=False,
        bridge_verifier_model=None,
        resolved_questions_path=str(tmp_path / "selected_questions.json"),
        resolved_corpus_path=str(tmp_path / "resolved_corpus.json"),
    )
    batch_dir = tmp_path / "batch_0"

    query_command = experiment._build_query_command(args, batch_index=0, batch_dir=batch_dir, registry_snapshot=None)
    analysis_command = experiment._build_analysis_command(args, batch_index=0, batch_dir=batch_dir)
    sleep_command = experiment._build_sleep_command(args, sleep_batch_index=1, batch_dir=batch_dir, registry_snapshot=None)

    assert "--model" in query_command and query_command[query_command.index("--model") + 1] == "query-model"
    assert "--base_url" in query_command and query_command[query_command.index("--base_url") + 1] == "http://query/v1"
    assert "--corpus_path" in query_command and query_command[query_command.index("--corpus_path") + 1] == str(tmp_path / "resolved_corpus.json")

    assert "--model" in analysis_command and analysis_command[analysis_command.index("--model") + 1] == "analysis-model"
    assert "--base_url" in analysis_command and analysis_command[analysis_command.index("--base_url") + 1] == "http://analysis/v1"

    assert "--model" in sleep_command and sleep_command[sleep_command.index("--model") + 1] == "sleep-model"
    assert "--base_url" in sleep_command and sleep_command[sleep_command.index("--base_url") + 1] == "http://sleep/v1"
    assert "--corpus_path" in sleep_command and sleep_command[sleep_command.index("--corpus_path") + 1] == str(tmp_path / "resolved_corpus.json")
    assert "--root_model" in sleep_command and sleep_command[sleep_command.index("--root_model") + 1] == "root-model"
    assert "--worker_model" in sleep_command and sleep_command[sleep_command.index("--worker_model") + 1] == "worker-model"
    assert "--curriculum_generation_parallel_enabled" in sleep_command
    assert "--curriculum_generation_parallel_workers" in sleep_command
    assert sleep_command[sleep_command.index("--curriculum_generation_parallel_workers") + 1] == "3"


def test_resolve_experiment_dataset_materializes_wrapped_manifest(tmp_path):
    source_questions = [
        {
            "id": "q0",
            "question": "Q0",
            "answer": "A0",
            "answer_aliases": [],
            "context": [["Doc A", ["sent"]]],
            "supporting_facts": [["Doc A", 0]],
        },
        {
            "id": "q1",
            "question": "Q1",
            "answer": "A1",
            "answer_aliases": [],
            "context": [["Doc B", ["sent"]]],
            "supporting_facts": [["Doc B", 0]],
        },
    ]
    source_questions_path = tmp_path / "source_questions.json"
    source_questions_path.write_text(json.dumps(source_questions))
    corpus_path = tmp_path / "source_corpus.json"
    corpus_path.write_text(json.dumps([{"title": "Doc A"}, {"title": "Doc B"}]))
    manifest_path = tmp_path / "wrapped_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "questions_path": str(source_questions_path),
                "corpus_path": str(corpus_path),
                "curriculum_batch_size": 10,
                "questions": [
                    {"original_index": 1, "question_id": "q1", "family": "spouse", "split": "train"},
                    {"original_index": 0, "question_id": "q0", "family": "mother", "split": "probe"},
                ],
            }
        )
    )

    resolved_questions_path, resolved_corpus_path, resolved_manifest = experiment._resolve_experiment_dataset(
        questions_path=str(manifest_path),
        corpus_path=None,
        output_root=tmp_path / "out",
    )

    resolved_questions = json.loads(Path(resolved_questions_path).read_text())
    assert [row["id"] for row in resolved_questions] == ["q1", "q0"]
    assert resolved_questions[0]["_experiment_family"] == "spouse"
    assert resolved_questions[1]["_experiment_split"] == "probe"
    assert resolved_corpus_path == str(corpus_path)
    assert resolved_manifest is not None
    assert int(resolved_manifest["curriculum_batch_size"]) == 10
    assert (tmp_path / "out" / "resolved_manifest.json").exists()
