from gsw_memory.query_then_sleep.logging_utils import StageLogger


def test_stage_logger_writes_expected_stage_artifacts(tmp_path):
    logger = StageLogger(stage_name="query", output_dir=tmp_path, verbose=False, show_thinking=False)
    logger.info("hello %s", "world")
    logger.event("tool_call", {"name": "search_entities", "arguments": {"query": "ermengarde"}})
    logger.progress("query_batch", "started", batch_index=0)
    logger.close()

    assert (tmp_path / "logs" / "execution.log").exists()
    assert (tmp_path / "logs" / "errors.log").exists()
    assert (tmp_path / "logs" / "agent_trace.log").exists()
    assert (tmp_path / "logs" / "tool_calls.jsonl").exists()
    assert (tmp_path / "query_progress.jsonl").exists()

    assert "hello world" in (tmp_path / "logs" / "execution.log").read_text()
    assert "search_entities" in (tmp_path / "logs" / "tool_calls.jsonl").read_text()
