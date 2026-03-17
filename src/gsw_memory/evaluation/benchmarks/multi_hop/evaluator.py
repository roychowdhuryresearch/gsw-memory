"""
Batched multi-hop QA evaluation harness.

Uses ChainFollowingQA for retrieval and (optionally) bespokelabs-curator
for parallel LLM calls during decomposition and answer generation stages.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset_configs import DATASET_CONFIGS

logger = logging.getLogger(__name__)

try:
    from bespokelabs import curator

    CURATOR_AVAILABLE = True
except ImportError:
    curator = None  # type: ignore[assignment]
    CURATOR_AVAILABLE = False


@dataclass
class MultiHopEvaluationResult:
    """Container for a single question's evaluation result."""

    question_id: str
    question: str
    predicted_answer: str
    full_response: str
    gold_answers: List[str]
    processing_time: float
    decomposed_questions: List[Dict[str, Any]]
    evidence_count: int
    evidence: List[str]
    chains_info: Dict[str, Any]
    token_count: int = 0
    error: Optional[str] = None


class MultiHopEvaluator:
    """Batched evaluator for chain-following multi-hop QA.

    Supports both closed-source (OpenAI) and open-source (vLLM) models
    for decomposition and answer generation, with optional curator-based
    parallel processing.

    Example::

        from gsw_memory.qa import GSWTools, ChainFollowingQA
        from gsw_memory.evaluation.benchmarks.multi_hop import MultiHopEvaluator

        tools = GSWTools(gsw_file_paths)
        tools.build_index()

        evaluator = MultiHopEvaluator(
            data_dir="path/to/questions",
            gsw_tools=tools,
            dataset_name="2wiki",
            num_questions=100,
        )
        results = evaluator.run_evaluation()
        metrics, per_example = evaluator.compute_metrics(results)
    """

    def __init__(
        self,
        data_dir: str,
        gsw_tools: "GSWTools",  # noqa: F821
        dataset_name: str = "2wiki",
        num_questions: int = 20,
        chain_top_k: int = 15,
        beam_width: int = 5,
        entity_top_k: int = 20,
        qa_rerank_top_k: int = 15,
        scoring_mode: str = "cumulative",
        alpha: float = 0.5,
        decomposition_model: str = "gpt-4o",
        answering_model: str = "gpt-4o-mini",
        verbose: bool = False,
    ):
        from ...qa import ChainFollowingQA

        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.num_questions = num_questions
        self.verbose = verbose

        # Dataset config
        config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["2wiki"])
        self.answer_field = config["answer_field"]
        self.parse_json = config["parse_json"]
        self.allow_no_answer = config["allow_no_answer"]

        # Build QA system
        self.qa_system = ChainFollowingQA(
            gsw_tools=gsw_tools,
            decomposition_model=decomposition_model,
            answering_model=answering_model,
            beam_width=beam_width,
            chain_top_k=chain_top_k,
            entity_top_k=entity_top_k,
            qa_rerank_top_k=qa_rerank_top_k,
            scoring_mode=scoring_mode,
            alpha=alpha,
            allow_no_answer=self.allow_no_answer,
            verbose=verbose,
        )

        logger.info(
            "MultiHopEvaluator initialised: dataset=%s, questions=%d",
            dataset_name,
            num_questions,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_questions_and_answers(
        self,
    ) -> List[Tuple[str, str, List[str]]]:
        """Load questions and gold answers from the dataset JSON file.

        Returns list of (question_id, question_text, gold_answers).
        """
        questions_file = self.data_dir / f"{self.dataset_name}.json"
        if not questions_file.exists():
            # Fallback name
            questions_file = self.data_dir / "2wikimultihopqa.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        with open(questions_file) as f:
            data = json.load(f)

        results: List[Tuple[str, str, List[str]]] = []
        for i, item in enumerate(data[: self.num_questions]):
            qid = item.get("_id", f"q_{i}")
            question = item["question"]

            gold = item.get(self.answer_field, [])
            if self.parse_json and isinstance(gold, str):
                try:
                    gold = json.loads(gold)
                except Exception:
                    gold = [gold]
            if isinstance(gold, str):
                gold = [gold]

            if self.answer_field == "answer":
                gold = gold + item.get("answer_aliases", [])

            results.append((qid, question, gold))

        logger.info("Loaded %d questions from %s", len(results), questions_file)
        return results

    # ------------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------------

    def run_evaluation(self) -> List[MultiHopEvaluationResult]:
        """Run the full 3-stage evaluation pipeline.

        Stage 1: Decompose questions (sequential via QuestionDecomposer).
        Stage 2: Chain-following retrieval.
        Stage 3: Answer generation.

        Returns list of per-question results.
        """
        total_start = time.time()
        questions_data = self.load_questions_and_answers()

        results: List[MultiHopEvaluationResult] = []

        for qid, question, gold_answers in questions_data:
            q_start = time.time()
            try:
                result_obj = self.qa_system.ask(question)
                predicted = result_obj.answer
                # Parse "Answer: " if present
                if "Answer: " in predicted:
                    predicted = predicted.split("Answer: ")[-1].strip()
                    if (
                        predicted.endswith(".")
                        and predicted[:-1].replace(",", "").replace(" ", "").isdigit()
                    ):
                        predicted = predicted[:-1]

                results.append(
                    MultiHopEvaluationResult(
                        question_id=qid,
                        question=question,
                        predicted_answer=predicted,
                        full_response=result_obj.answer,
                        gold_answers=gold_answers,
                        processing_time=time.time() - q_start,
                        decomposed_questions=result_obj.decomposed_questions,
                        evidence_count=result_obj.evidence_count,
                        evidence=result_obj.evidence,
                        chains_info=result_obj.chains_info,
                    )
                )
            except Exception as e:
                logger.error("Error processing question %s: %s", qid, e)
                results.append(
                    MultiHopEvaluationResult(
                        question_id=qid,
                        question=question,
                        predicted_answer="",
                        full_response="",
                        gold_answers=gold_answers,
                        processing_time=time.time() - q_start,
                        decomposed_questions=[],
                        evidence_count=0,
                        evidence=[],
                        chains_info={},
                        error=str(e),
                    )
                )

        elapsed = time.time() - total_start
        logger.info(
            "Evaluation complete: %d questions in %.1fs (%.2fs/q)",
            len(results),
            elapsed,
            elapsed / max(len(results), 1),
        )
        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self, results: List[MultiHopEvaluationResult]
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute EM/F1 metrics via hipporag_eval.

        Returns (overall_metrics, per_example_metrics).
        """
        from ...evaluation.hipporag_eval import evaluate_qa_batch

        valid = [r for r in results if r.error is None]
        if not valid:
            return {}, []

        gold_list = [r.gold_answers for r in valid]
        pred_list = [r.predicted_answer for r in valid]

        overall, per_example = evaluate_qa_batch(gold_list, pred_list)
        return overall, per_example

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: List[MultiHopEvaluationResult],
        overall_metrics: Dict[str, float],
        per_example_metrics: List[Dict[str, Any]],
        output_dir: str = "logs",
    ) -> Path:
        """Save evaluation results to a JSON file. Returns the output path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"multihop_eval_{self.dataset_name}_{ts}.json"

        data = {
            "evaluation_info": {
                "dataset": self.dataset_name,
                "num_questions": self.num_questions,
                "timestamp": datetime.now().isoformat(),
            },
            "overall_metrics": overall_metrics,
            "per_question_results": [],
        }

        for result, metrics in zip(results, per_example_metrics):
            data["per_question_results"].append(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "predicted_answer": result.predicted_answer,
                    "gold_answers": result.gold_answers,
                    "metrics": metrics,
                    "evidence_count": result.evidence_count,
                    "evidence": result.evidence,
                    "decomposed_questions": result.decomposed_questions,
                    "chains_info": result.chains_info,
                    "error": result.error,
                }
            )

        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Results saved to %s", out_file)
        return out_file
