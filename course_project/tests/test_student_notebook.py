from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "Panini_Course_Project.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_source(notebook) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_student_notebook_covers_every_question_and_response():
    notebook = load_notebook()
    source = notebook_source(notebook)

    assert len(notebook["cells"]) >= 45
    for number in range(1, 13):
        assert f"Question {number}" in source
        assert f"Your written response for Question {number}" in source
    assert source.count("Your written response for Question") == 12


def test_student_notebook_uses_restartable_single_model_stages():
    source = notebook_source(load_notebook())

    assert "RUN_DECOMPOSITION_STAGE = False" in source
    assert "RUN_RERANK_AND_RICR_STAGE = False" in source
    assert "RUN_ANSWER_STAGE = False" in source
    assert "QUESTION_LIMIT = 2" in source
    assert "decompositions.jsonl" in source
    assert "ricr_traces.jsonl" in source
    assert "answers.jsonl" in source
    assert "PERSISTENT_RICR" in source
    assert "PERSISTENT_TESTS" in source
    assert source.count("release_gpu()") >= 3


def test_student_notebook_keeps_solutions_empty_and_outputs_clear():
    notebook = load_notebook()
    source = notebook_source(notebook)

    assert source.count("NotImplementedError") >= 10
    assert "execute_panini_plan" not in source
    assert "Corrected student-scale run" not in source
    assert "0.4625" not in source
    assert "0.2375" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
