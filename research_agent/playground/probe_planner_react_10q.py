"""End-to-end probe: run `ours_gsw_planner_react_v1` on 10 curated questions.

This is the full retrieval-answerer loop (planner → topo sort → ReAct
with search/read/update_blank/finish tools), not just plan emission.
Uses the FRAMES BM25 corpus as the retrieval backend — most of the
questions land on mainstream Wikipedia topics that FRAMES already caches.

Saves a standard ``cell_result.json`` under
``logs/ours_gsw_planner_react_v1__<model>__probe10_<ts>/`` so the
Streamlit inspector (`planner_react_run_inspector.py`) can browse it.

Usage::

    .venv/bin/python playground/probe_planner_react_10q.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make src importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from research_agent.adapters.base import AdapterContext
from research_agent.adapters.ours.gsw_planner_react_v1 import (
    OursGSWPlannerReactV1Adapter,
)
from research_agent.models.trace import CellResult, QuestionResult

try:
    from dotenv import load_dotenv

    load_dotenv()
    _PARENT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
    if _PARENT_ENV.exists():
        load_dotenv(_PARENT_ENV, override=False)
except ImportError:
    pass


# Same 10 questions as the plan-only probe, so the comparison is direct.
QUESTIONS = [
    # Bucket A — collective + superlative (Pattern 2 target)
    ("qA1", "Who lived longer, one of the Brontë sisters or Jane Austen?", "Jane Austen"),
    ("qA2", "Which Jonas brother had the most Grammy nominations?", "Nick Jonas"),
    ("qA3", "Which of the three Musketeers was the youngest?", "D'Artagnan"),
    # Bucket B — collective + list/count (Pattern 1 target)
    ("qB1", "Who were the Founding Fathers of the United States?",
     "Washington, Adams, Jefferson, Madison, Franklin, Hamilton, Jay"),
    ("qB2", "Name all the Brontë sisters and their birth years.",
     "Charlotte (1816), Emily (1818), Anne (1820)"),
    ("qB3", "How many members did the Beatles have?", "4"),
    # Bucket C — non-collective baseline
    ("qC1", "Who directed The Princess Bride?", "Rob Reiner"),
    ("qC2", "What year did World War I end?", "1918"),
    ("qC3", "Who was the first person to walk on the moon?", "Neil Armstrong"),
    ("qC4", "What is the population of Tokyo as of 2024?", "about 14 million"),
]


MODEL = "bedrock/openai.gpt-oss-120b-1:0"
MAX_TURNS = 50  # Generous cap — lets complex pattern-2 argmax plans breathe.


def main(retriever: str = "bm25") -> None:
    ctx = AdapterContext(
        system_id="ours_gsw_planner_react_v1",
        model_id=MODEL,
        model_name=MODEL,
        max_turns=MAX_TURNS,
        max_completion_tokens=50000,
        extra={"retriever_type": retriever},
    )
    adapter = OursGSWPlannerReactV1Adapter(ctx)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"probe10_{retriever}_{ts}"
    out_dir = (
        Path(__file__).resolve().parent.parent
        / "logs"
        / f"ours_gsw_planner_react_v1__{MODEL.replace('/', '_')}__{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cell = CellResult(
        system_id="ours_gsw_planner_react_v1",
        model_id=MODEL,
        benchmark="probe_10q",
        subset_id="custom_collective_probe",
        config={"model": MODEL, "max_turns": MAX_TURNS, "n_questions": len(QUESTIONS)},
    )

    t_total = time.time()
    for qid, question, gold in QUESTIONS:
        print(f"\n{'=' * 80}")
        print(f"▸ {qid}: {question}")
        print(f"  gold: {gold}")
        t0 = time.time()
        traj = adapter.run_question(question, question_id=qid)
        dur = round(time.time() - t0, 1)
        extra = traj.extra
        tc_seq = [tc.name for tc in traj.tool_calls]
        tgt = plan_target_status(extra)
        print(
            f"  [{dur}s turns={traj.turns} stopped={extra.get('stopped_reason')} "
            f"updates={extra.get('blank_updates')} rej={extra.get('finish_rejections')}]"
        )
        print(f"  tools: {tc_seq}")
        print(f"  target: {tgt}")
        print(f"  pred:   {traj.final_answer!r}")

        q_res = QuestionResult(
            question_id=qid,
            question=question,
            gold_answer=gold,
            trajectory=traj,
            predicted_answer=traj.final_answer,
        )
        cell.questions.append(q_res)

    cell.n_total = len(cell.questions)
    cell.completed_at = datetime.now(timezone.utc)
    cell.mean_wall_time_s = round(
        sum(q.trajectory.wall_time_s for q in cell.questions) / max(cell.n_total, 1), 2
    )
    cell.mean_turns = round(
        sum(q.trajectory.turns for q in cell.questions) / max(cell.n_total, 1), 2
    )
    cell.mean_tool_calls = round(
        sum(len(q.trajectory.tool_calls) for q in cell.questions) / max(cell.n_total, 1), 2
    )

    cr_path = out_dir / "cell_result.json"
    cr_path.write_text(cell.model_dump_json(indent=2, exclude_none=False))
    print(f"\n\nSaved cell_result to {cr_path}")
    print(f"Total wall: {round(time.time() - t_total, 1)}s")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'qid':<6}  {'turns':>5}  {'upd':>3}  {'rej':>3}  {'target':<9}  stopped           pred")
    for q in cell.questions:
        ex = q.trajectory.extra
        tgt = plan_target_status(ex)
        print(
            f"{q.question_id:<6}  {q.trajectory.turns:>5}  "
            f"{ex.get('blank_updates', 0):>3}  {ex.get('finish_rejections', 0):>3}  "
            f"{tgt:<9}  {ex.get('stopped_reason','?'):<17} "
            f"{str(q.predicted_answer)[:60]!r}"
        )


def plan_target_status(extra: dict) -> str:
    """Find the target blank's status in executed_blanks."""
    plan = extra.get("plan_json") or {}
    target_id = None
    for e in plan.get("entities", []):
        if e.get("is_target"):
            target_id = e.get("id")
            break
    if not target_id:
        return "?"
    for b in extra.get("executed_blanks", []) or []:
        if b.get("blank_id") == target_id:
            return b.get("status", "?")
    return "?"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retriever backend (default: bm25).",
    )
    args = ap.parse_args()
    main(retriever=args.retriever)
