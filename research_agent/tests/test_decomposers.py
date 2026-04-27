"""Tests for the decomposers package.

These tests cover the *infrastructure* (schema round-trip, sandbox
guard, grader). Quality tests for specific emitted decomposer modules
live alongside the emitted .py files and are run by the eval CLI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research_agent.decomposers import (
    DecomposerTimeoutError,
    ForbiddenImportError,
    SubQuestion,
    grade_decomposer,
    load_decomposer,
    scan_source_for_forbidden_imports,
)
from research_agent.decomposers._datasets import load_musique, split_train_dev


# ----- Schema round-trip -------------------------------------------------


def test_subquestion_converts_to_both_consumer_shapes():
    sq = SubQuestion(
        sub_id="s1",
        question="Who directed X?",
        entity_focus="X",
        relation_hint="directed_by",
        hop_type="lookup",
        references=[],
    )
    trace_kwargs = sq.to_query_trace_kwargs()
    ours_kwargs = sq.to_ours_gsw_kwargs()
    assert trace_kwargs == {
        "sub_question": "Who directed X?",
        "relation_hint": "directed_by",
        "entity_hint": "X",
    }
    assert ours_kwargs == {
        "sub_id": "s1",
        "question": "Who directed X?",
        "entity_focus": "X",
        "hop_type": "lookup",
    }


# ----- Zero-LLM import guard --------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "import openai\n",
        "from anthropic import Anthropic\n",
        "import requests\n",
        "from urllib.request import urlopen\n",
        "import httpx\n",
        "from gsw_memory.memory.operator_utils.agentic_gsw.transport_shim import TransportShim\n",
        "import importlib.util\n",
    ],
)
def test_guard_rejects_banned_imports(src: str):
    assert scan_source_for_forbidden_imports(src), f"guard should flag: {src!r}"


@pytest.mark.parametrize(
    "src",
    [
        "import re\n",
        "import string\n",
        "from dataclasses import dataclass\n",
        "from typing import List\n",
        "# import openai\n",
        's = "import openai"\n',
    ],
)
def test_guard_allows_clean_sources(src: str):
    assert scan_source_for_forbidden_imports(src) == [], f"guard should allow: {src!r}"


def test_load_decomposer_rejects_rogue_module():
    src = (
        "import openai\n"
        "from research_agent.decomposers._schema import SubQuestion\n"
        "def decompose(q):\n"
        "    return [SubQuestion(sub_id='s1', question=q)]\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    with pytest.raises(ForbiddenImportError):
        load_decomposer(path)


def test_load_decomposer_requires_decompose_fn():
    src = (
        "import re\n"
        "from research_agent.decomposers._schema import SubQuestion\n"
        "def other(q): return []\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    with pytest.raises(AttributeError):
        load_decomposer(path)


def test_timeout_wrapper_fires():
    src = (
        "from research_agent.decomposers._schema import SubQuestion\n"
        "def decompose(q):\n"
        "    while True:\n"
        "        pass\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    fn = load_decomposer(path, per_call_timeout_s=0.25)
    with pytest.raises(DecomposerTimeoutError):
        fn("anything")


def test_grader_reports_coverage_and_count_match():
    src = (
        "from research_agent.decomposers._schema import SubQuestion\n"
        "def decompose(q):\n"
        "    if ' and ' in q:\n"
        "        a, b = q.split(' and ', 1)\n"
        "        return [SubQuestion(sub_id='s1', question=a),\n"
        "                SubQuestion(sub_id='s2', question=b)]\n"
        "    return []\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    fn = load_decomposer(path)
    examples = [
        {"question": "who wrote X and when was X published",
         "gold_decomposition": [{"question": "who wrote X"}, {"question": "when was X published"}]},
        {"question": "what is the capital of France",
         "gold_decomposition": [{"question": "what is the capital of France"}]},
    ]
    score = grade_decomposer(fn, examples)
    # First example: count matches (2 == 2). Second: pred is [] against gold of 1.
    assert score["n"] == 2
    assert score["coverage"] == pytest.approx(0.5)
    assert score["count_match"] == pytest.approx(0.5)


# ----- Dataset loader ---------------------------------------------------


def test_load_musique_shape():
    path = Path(__file__).resolve().parents[2] / "playground_data" / "musique_10_q.json"
    if not path.exists():
        pytest.skip(f"MuSiQue fixture not present at {path}")
    items = load_musique(path, limit=5)
    assert 0 < len(items) <= 5
    first = items[0]
    assert "question" in first and "gold_decomposition" in first
    assert isinstance(first["gold_decomposition"], list)
    if first["gold_decomposition"]:
        assert "question" in first["gold_decomposition"][0]


def test_rule_decomp_adapter_registers():
    """Importing the adapter module must register the canonical system id."""
    import research_agent.adapters.baselines.rule_decomp_gsw  # noqa: F401
    from research_agent.adapters.base import get_adapter

    cls = get_adapter("rule_decomp_gsw")
    assert cls is not None and cls.system_id == "rule_decomp_gsw"


def test_split_train_dev_is_deterministic():
    items = [{"id": f"q{i}", "question": f"q{i}?"} for i in range(40)]
    t1, d1 = split_train_dev(items, dev_fraction=0.25, seed=0)
    t2, d2 = split_train_dev(items, dev_fraction=0.25, seed=0)
    assert t1 == t2 and d1 == d2
    assert len(t1) + len(d1) == 40
    # Different seed -> different split (with high prob given n=40)
    _, d3 = split_train_dev(items, dev_fraction=0.25, seed=7)
    assert d1 != d3
