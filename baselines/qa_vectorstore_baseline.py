#!/usr/bin/env python3
"""
QA Vector Store Baseline for Multi-Hop QA

Instead of retrieving raw document chunks, this baseline retrieves from a
GSW question-answer pair vector store (via EntitySearcher). This isolates
the effect of GSW's QA-pair representation vs raw document chunks.

Pipeline:
1. Loads pre-built GSW structures and builds a QA-pair FAISS index (via EntitySearcher)
2. For each question, retrieves top-k most similar QA pairs from the index
3. Generates answers using GPT-4o-mini with oracle-style prompting
4. Evaluates using the same metrics as the main system
"""

import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch, format_evaluation_report

# Import EntitySearcher for QA vector store
from playground.simple_entity_search import EntitySearcher

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False

try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    CURATOR_AVAILABLE = False

console = Console()

# Dataset configurations mapping
DATASET_CONFIGS = {
    "2wiki": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": False
    },
    "2wiki_platinum": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True
    },
    "2wiki_unanswerable": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True
    },
    "musique": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": False
    },
    "musique_platinum": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True
    },
    "musique_unanswerable": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True
    },
    "hotpotqa": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": False
    },
    "popqa": {
        "answer_field": "possible_answers",
        "parse_json": True,
        "allow_no_answer": False
    },
    "nq_rear": {
        "answer_field": "reference",
        "parse_json": False,
        "allow_no_answer": False
    },
    "lveval": {
        "answer_field": "answers",
        "parse_json": False,
        "allow_no_answer": False
    }
}

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _format_qa_evidence(qa_pairs: List[Dict[str, Any]]) -> str:
    """Format QA pairs into evidence text for the answer generation prompt."""
    evidence_parts = []
    for qa in qa_pairs:
        question = qa.get('question', '')
        answer_names = qa.get('answer_names', [])
        answer_rolestates = qa.get('answer_rolestates', [])
        answers_text = ', '.join(answer_names)
        if answer_rolestates:
            answers_text += f" ({', '.join(answer_rolestates)})"
        evidence_parts.append(f"Q: {question} A: {answers_text}")
    return '\n'.join(evidence_parts)


if CURATOR_AVAILABLE:
    class QAVectorStoreAnswerGenerator(curator.LLM):
        """Curator class for generating answers from QA-pair evidence."""

        return_completions_object = True

        def __init__(self, allow_no_answer: bool = False, **kwargs):
            self.allow_no_answer = allow_no_answer
            super().__init__(**kwargs)

        def prompt(self, input):
            """Create an oracle-style answer generation prompt from QA pairs."""
            evidence_text = _format_qa_evidence(input['retrieved_qa_pairs'])

            prompt_text = f"""
{evidence_text}

Question: {input['question']}

Thought:

"""
            one_shot_docs = (
                """ Q: Who directed The Last Horse? A: Edgar Neville
                    Q: When was The Last Horse released? A: 1950
                    Q: When was the University of Southampton founded? A: 1862
                    Q: Where is the University of Southampton located? A: Southampton
                    Q: What is the population of Stanton Township? A: 505
                    Q: Where is Stanton Township? A: Champaign County, Illinois
                    Q: Who is Neville A. Stanton? A: British Professor of Human Factors and Ergonomics
                    Q: Where does Neville A. Stanton work? A: University of Southampton
                    Q: What is Neville A. Stanton's profession? A: Professor
                    Q: Who directed Finding Nemo? A: Andrew Stanton
                    Q: When was Finding Nemo released? A: 2003
                    Q: What company produced Finding Nemo? A: Pixar Animation Studios"""
            )

            rag_qa_system = (
                'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
            )

            if self.allow_no_answer:
                rag_qa_system += ' If you don\'t know the answer, say "No Answer".'

            one_shot_input = (
                f"{one_shot_docs}"
                "\n\nQuestion: "
                "When was Neville A. Stanton's employer founded?"
                '\nThought: '
            )

            one_shot_output = (
                "From the QA pairs, the employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
                "\nAnswer: 1862."
            )

            prompt_messages = [
                {"role": "system", "content": rag_qa_system},
                {"role": "user", "content": one_shot_input},
                {"role": "assistant", "content": one_shot_output},
                {"role": "user", "content": prompt_text}
            ]

            return prompt_messages

        def parse(self, input, response):
            """Parse the answer from the response."""
            answer_text = response["choices"][0]["message"]["content"]

            if 'Answer: ' in answer_text:
                final_answer = answer_text.split('Answer: ')[-1].strip()
                if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                    final_answer = final_answer[:-1]
            else:
                final_answer = answer_text.strip()

            evidence_text = _format_qa_evidence(input['retrieved_qa_pairs'])
            token_count = 0
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                token_count = len(encoding.encode(evidence_text))
            except Exception:
                token_count = max(1, len(evidence_text) // 4)

            return [{
                "question_id": input['question_id'],
                "question": input['question'],
                "predicted_answer": final_answer,
                "full_response": answer_text,
                "retrieved_qa_pairs": input['retrieved_qa_pairs'],
                "token_count": token_count
            }]


@dataclass
class QABaselineEvaluationResult:
    """Container for QA vector store baseline evaluation results."""
    question_id: str
    question: str
    predicted_answer: str
    full_response: str
    gold_answers: List[str]
    processing_time: float
    retrieved_qa_pairs: List[Dict[str, Any]]
    token_count: int = 0
    error: Optional[str] = None


class QAVectorStoreBaseline:
    """Baseline using GSW QA-pair vector store for retrieval."""

    def __init__(self, gsw_path: str, questions_path: str, num_documents: int = -1,
                 num_questions: int = 20, top_k: int = 15, cache_dir: str = ".",
                 verbose: bool = False, use_reranker: bool = False,
                 reranker_top_k: int = 30, dataset_name: str = "2wiki",
                 gpu_device: int = 1, rebuild_cache: bool = False):
        """Initialize the QA vector store baseline.

        Args:
            gsw_path: Path to GSW network directories
            questions_path: Path to questions JSON file
            num_documents: Number of GSW documents to load (-1 for all)
            num_questions: Number of questions to evaluate
            top_k: Number of QA pairs to retrieve per question
            cache_dir: Directory for EntitySearcher embedding cache
            verbose: Show detailed output
            use_reranker: Use VoyageAI to rerank retrieved QA pairs
            reranker_top_k: Number of QA pairs to retrieve before reranking
            dataset_name: Dataset name for answer field config
            gpu_device: GPU device for EntitySearcher
        """
        dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["2wiki"])
        self.dataset_name = dataset_name
        self.answer_field = dataset_config["answer_field"]
        self.parse_json = dataset_config["parse_json"]
        self.allow_no_answer = dataset_config["allow_no_answer"]

        self.questions_path = Path(questions_path)
        self.num_questions = num_questions
        self.top_k = top_k
        self.verbose = verbose
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.voyage_client = None

        console.print("[bold blue]Initializing QA Vector Store Baseline...[/bold blue]")
        console.print(f"  GSW path: {gsw_path}")
        console.print(f"  Top-k QA pairs: {top_k}")
        if use_reranker:
            console.print(f"  Reranker: retrieve {reranker_top_k}, rerank to {top_k}")

        # Initialize EntitySearcher (handles embedding model + FAISS index)
        self.entity_searcher = EntitySearcher(
            num_documents=num_documents,
            cache_dir=cache_dir,
            path_to_gsw_files=gsw_path,
            verbose=True,
            use_bm25=False,
            rebuild_cache=rebuild_cache,
            gpu_device=gpu_device,
            use_gpu_for_qa_index=True
        )

        # Initialize OpenAI client for sequential fallback
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI()
            console.print("[green]✓ OpenAI client initialized[/green]")
        else:
            raise RuntimeError("OpenAI client is required for answer generation")

        # Initialize VoyageAI reranker if needed
        if self.use_reranker:
            if VOYAGEAI_AVAILABLE:
                self.voyage_client = voyageai.Client()
                console.print("[green]✓ VoyageAI client initialized for reranking[/green]")
            else:
                raise RuntimeError("VoyageAI is required when use_reranker=True")

        # Initialize Curator answer generator
        self.answer_generator = None
        if CURATOR_AVAILABLE:
            os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"
            self.answer_generator = QAVectorStoreAnswerGenerator(
                allow_no_answer=self.allow_no_answer,
                model_name="gpt-4o-mini",
                generation_params={"temperature": 0},
            )
            console.print("[green]✓ Curator initialized for parallel answer generation[/green]")

        console.print(f"[bold green]✓ QA Vector Store Baseline initialized[/bold green]")

    def retrieve_qa_pairs(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve top-k QA pairs for a question from the GSW vector store.

        Args:
            question: The question to retrieve QA pairs for

        Returns:
            List of QA-pair dicts with question, answer_names, answer_rolestates, similarity_score
        """
        initial_k = self.reranker_top_k if self.use_reranker else self.top_k

        # Pre-embed the query (EntitySearcher._embed_query expects a list of strings)
        query_embedding = self.entity_searcher._embed_query([question])
        query_emb = np.array(query_embedding[0]) if query_embedding else None

        results = self.entity_searcher.search_qa_pairs_direct(
            query=question,
            query_embedding=query_emb,
            top_k=initial_k,
            verbose=False
        )

        if self.verbose and results:
            console.print(f"[dim]Retrieved {len(results)} QA pairs, top score: {results[0].get('similarity_score', 0):.3f}[/dim]")

        # Rerank if enabled
        if self.use_reranker and self.voyage_client and results:
            qa_texts = []
            for qa in results:
                answer_names = qa.get('answer_names', [])
                answer_rolestates = qa.get('answer_rolestates', [])
                qa_text = f"{qa['question']} {', '.join(answer_names)} {', '.join(answer_rolestates)}"
                qa_texts.append(qa_text)

            try:
                reranking = self.voyage_client.rerank(
                    question, qa_texts, model="rerank-2.5", top_k=self.top_k
                )

                reranked = []
                for r in reranking.results:
                    qa = results[r.index].copy()
                    qa['rerank_score'] = r.relevance_score
                    reranked.append(qa)

                if self.verbose:
                    console.print(f"[dim]Reranked {len(results)} -> {len(reranked)} QA pairs[/dim]")
                return reranked

            except Exception as e:
                console.print(f"[yellow]Reranking failed: {e}. Using embedding scores.[/yellow]")

        return results[:self.top_k]

    def generate_answer(self, question: str, qa_pairs: List[Dict[str, Any]]) -> Tuple[str, str, int]:
        """Generate an answer using GPT-4o-mini with oracle-style prompting (sequential fallback).

        Args:
            question: The question to answer
            qa_pairs: Retrieved QA pairs

        Returns:
            Tuple of (predicted_answer, full_response, token_count)
        """
        evidence_text = _format_qa_evidence(qa_pairs)

        prompt_text = f"""
{evidence_text}

Question: {question}

Thought:

"""

        one_shot_docs = (
            """ Q: Who directed The Last Horse? A: Edgar Neville
                Q: When was The Last Horse released? A: 1950
                Q: When was the University of Southampton founded? A: 1862
                Q: Where is the University of Southampton located? A: Southampton
                Q: What is the population of Stanton Township? A: 505
                Q: Where is Stanton Township? A: Champaign County, Illinois
                Q: Who is Neville A. Stanton? A: British Professor of Human Factors and Ergonomics
                Q: Where does Neville A. Stanton work? A: University of Southampton
                Q: What is Neville A. Stanton's profession? A: Professor
                Q: Who directed Finding Nemo? A: Andrew Stanton
                Q: When was Finding Nemo released? A: 2003
                Q: What company produced Finding Nemo? A: Pixar Animation Studios"""
        )

        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
        )

        if self.allow_no_answer:
            rag_qa_system += ' If you don\'t know the answer, say "No Answer".'

        one_shot_input = (
            f"{one_shot_docs}"
            "\n\nQuestion: "
            "When was Neville A. Stanton's employer founded?"
            '\nThought: '
        )

        one_shot_output = (
            "From the QA pairs, the employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
            "\nAnswer: 1862."
        )

        prompt_messages = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": one_shot_output},
            {"role": "user", "content": prompt_text}
        ]

        token_count = 0
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                token_count = len(encoding.encode(evidence_text))
            except Exception:
                token_count = max(1, len(evidence_text) // 4)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages,
                temperature=0.0,
                max_tokens=1000
            )

            full_response = response.choices[0].message.content

            if 'Answer: ' in full_response:
                final_answer = full_response.split('Answer: ')[-1].strip()
                if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                    final_answer = final_answer[:-1]
            else:
                final_answer = full_response.strip()

            return final_answer, full_response, token_count

        except Exception as e:
            console.print(f"[red]Error generating answer: {e}[/red]")
            return "NA", str(e), token_count

    def load_questions(self) -> List[Tuple[str, str, List[str]]]:
        """Load questions from file.

        Returns:
            List of (question_id, question, gold_answers) tuples
        """
        console.print(f"[cyan]Loading questions from {self.questions_path}...[/cyan]")

        with open(self.questions_path, 'r') as f:
            data = json.load(f)

        questions_data = []
        for i, item in enumerate(data[:self.num_questions]):
            question_id = item.get("_id", f"q_{i}")
            question = item["question"]

            gold_answers = item.get(self.answer_field, [])

            if self.parse_json and isinstance(gold_answers, str):
                try:
                    gold_answers = json.loads(gold_answers)
                except Exception:
                    gold_answers = [gold_answers]

            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            if "answer_aliases" in item:
                if isinstance(item["answer_aliases"], list):
                    gold_answers = list(gold_answers) + item["answer_aliases"]

            questions_data.append((question_id, question, gold_answers))

        console.print(f"[green]✓ Loaded {len(questions_data)} questions[/green]")
        return questions_data

    def run_batched_answer_generation(self, questions_data: List[Tuple[str, str, List[str]]],
                                      retrieved_by_question: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict]:
        """Run batched answer generation using Curator.

        Args:
            questions_data: List of (question_id, question, gold_answers)
            retrieved_by_question: Dict mapping question_id to retrieved QA pairs

        Returns:
            Dict mapping question_id to answer results
        """
        console.print("\n[bold cyan]Running Batched Answer Generation[/bold cyan]")

        answer_inputs = [
            {
                "question_id": qid,
                "question": question,
                "retrieved_qa_pairs": retrieved_by_question.get(qid, [])
            }
            for qid, question, _ in questions_data
        ]

        console.print(f"[cyan]Generating answers for {len(answer_inputs)} questions in parallel...[/cyan]")
        start_time = time.time()

        answer_dataset = self.answer_generator(answer_inputs)

        elapsed = time.time() - start_time
        console.print(f"[green]✓ Answer generation complete in {elapsed:.1f}s ({elapsed/len(answer_inputs):.2f}s per question)[/green]")

        answer_results = {item["question_id"]: item for item in answer_dataset}
        return answer_results

    def run_evaluation(self) -> List[QABaselineEvaluationResult]:
        """Run complete evaluation pipeline.

        Returns:
            List of evaluation results
        """
        console.print("\n[bold magenta]Running QA Vector Store Baseline Evaluation[/bold magenta]")
        total_start = time.time()

        questions_data = self.load_questions()

        if CURATOR_AVAILABLE and self.answer_generator is not None:
            return self._run_batched_evaluation(questions_data, total_start)
        else:
            return self._run_sequential_evaluation(questions_data, total_start)

    def _run_batched_evaluation(self, questions_data: List[Tuple[str, str, List[str]]],
                                 total_start: float) -> List[QABaselineEvaluationResult]:
        """Run evaluation with batched answer generation."""
        # Stage 1: Retrieve QA pairs for all questions
        console.print("\n[bold cyan]Stage 1: QA Pair Retrieval[/bold cyan]")
        retrieved_by_question = {}

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task = progress.add_task("Retrieving QA pairs...", total=len(questions_data))

            for question_id, question, _ in questions_data:
                qa_pairs = self.retrieve_qa_pairs(question)
                retrieved_by_question[question_id] = qa_pairs
                progress.update(task, advance=1)

        console.print(f"[green]✓ QA pair retrieval complete for {len(questions_data)} questions[/green]")

        # Stage 2: Batched answer generation
        answer_results = self.run_batched_answer_generation(questions_data, retrieved_by_question)

        # Stage 3: Compile results
        results = []
        for question_id, question, gold_answers in questions_data:
            answer_data = answer_results.get(question_id, {})
            qa_pairs = retrieved_by_question.get(question_id, [])

            result = QABaselineEvaluationResult(
                question_id=question_id,
                question=question,
                predicted_answer=answer_data.get("predicted_answer", "NA"),
                full_response=answer_data.get("full_response", ""),
                gold_answers=gold_answers,
                processing_time=0.0,
                retrieved_qa_pairs=qa_pairs,
                token_count=answer_data.get("token_count", 0),
                error=None
            )
            results.append(result)

        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")

        avg_time = total_elapsed / len(results) if results else 0
        for result in results:
            result.processing_time = avg_time

        return results

    def _run_sequential_evaluation(self, questions_data: List[Tuple[str, str, List[str]]],
                                    total_start: float) -> List[QABaselineEvaluationResult]:
        """Run evaluation with sequential processing (fallback)."""
        console.print("[yellow]Using sequential processing (curator not available)[/yellow]")

        results = []

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task = progress.add_task("Processing questions...", total=len(questions_data))

            for question_id, question, gold_answers in questions_data:
                start_time = time.time()

                try:
                    qa_pairs = self.retrieve_qa_pairs(question)
                    predicted_answer, full_response, token_count = self.generate_answer(question, qa_pairs)
                    processing_time = time.time() - start_time

                    result = QABaselineEvaluationResult(
                        question_id=question_id,
                        question=question,
                        predicted_answer=predicted_answer,
                        full_response=full_response,
                        gold_answers=gold_answers,
                        processing_time=processing_time,
                        retrieved_qa_pairs=qa_pairs,
                        token_count=token_count,
                        error=None
                    )

                except Exception as e:
                    console.print(f"[red]Error processing question {question_id}: {e}[/red]")
                    result = QABaselineEvaluationResult(
                        question_id=question_id,
                        question=question,
                        predicted_answer="NA",
                        full_response="",
                        gold_answers=gold_answers,
                        processing_time=0.0,
                        retrieved_qa_pairs=[],
                        token_count=0,
                        error=str(e)
                    )

                results.append(result)
                progress.update(task, advance=1)

        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")

        return results

    def compute_metrics(self, results: List[QABaselineEvaluationResult]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute evaluation metrics."""
        console.print("\n[cyan]Computing evaluation metrics...[/cyan]")

        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            console.print("[red]No valid results to evaluate![/red]")
            return {}, []

        gold_answers_list = [r.gold_answers for r in valid_results]
        predicted_answers = [r.predicted_answer for r in valid_results]

        overall_metrics, per_example_metrics = evaluate_qa_batch(
            gold_answers_list,
            predicted_answers
        )

        return overall_metrics, per_example_metrics

    def save_results(self, results: List[QABaselineEvaluationResult],
                    overall_metrics: Dict[str, float],
                    per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file."""
        output_file = LOG_DIR / f"qa_vectorstore_baseline_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            "evaluation_info": {
                "baseline": "qa_vectorstore",
                "dataset": self.dataset_name,
                "num_questions": self.num_questions,
                "top_k": self.top_k,
                "use_reranker": self.use_reranker,
                "reranker_top_k": self.reranker_top_k if self.use_reranker else None,
                "timestamp": datetime.now().isoformat()
            },
            "overall_metrics": overall_metrics,
            "per_question_results": []
        }

        for result, metrics in zip(results, per_example_metrics):
            question_data = {
                "question_id": result.question_id,
                "question": result.question,
                "predicted_answer": result.predicted_answer,
                "full_response": result.full_response,
                "gold_answers": result.gold_answers,
                "metrics": metrics,
                "processing_time": result.processing_time,
                "retrieved_qa_pairs": [
                    {
                        "question": qa.get("question", ""),
                        "answer_names": qa.get("answer_names", []),
                        "answer_rolestates": qa.get("answer_rolestates", []),
                        "similarity_score": qa.get("similarity_score", 0.0),
                        "rerank_score": qa.get("rerank_score", None),
                        "doc_id": qa.get("doc_id", ""),
                        "source_method": qa.get("source_method", ""),
                    }
                    for qa in result.retrieved_qa_pairs
                ],
                "token_count": result.token_count,
                "error": result.error
            }
            output_data["per_question_results"].append(question_data)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        console.print(f"[green]✓ Results saved to: {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(description="QA Vector Store Baseline for Multi-Hop QA")
    parser.add_argument("--gsw_path", type=str, required=True,
                        help="Path to GSW network directories")
    parser.add_argument("--questions_path", type=str, required=True,
                        help="Path to questions JSON file")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for EntitySearcher embeddings (defaults to <gsw_path>/../.gsw_cache)")
    parser.add_argument("--dataset_name", type=str, default="2wiki",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset name for answer field configuration")
    parser.add_argument("--num_documents", type=int, default=-1,
                        help="Number of GSW documents to load (-1 for all)")
    parser.add_argument("--num_questions", type=int, default=1000,
                        help="Number of questions to evaluate")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of QA pairs to retrieve per question")
    parser.add_argument("--use_reranker", action="store_true",
                        help="Use VoyageAI reranker on retrieved QA pairs")
    parser.add_argument("--reranker_top_k", type=int, default=30,
                        help="Number of QA pairs to retrieve before reranking")
    parser.add_argument("--gpu_device", type=int, default=1,
                        help="GPU device for FAISS index")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild EntitySearcher embedding cache")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    # Default cache_dir to <gsw_path>/../.gsw_cache (where EntitySearcher saves its caches)
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = str(Path(args.gsw_path).parent / ".gsw_cache")

    console.print("\n[bold cyan]== QA Vector Store Baseline Evaluation ==[/bold cyan]")

    try:
        baseline = QAVectorStoreBaseline(
            gsw_path=args.gsw_path,
            questions_path=args.questions_path,
            num_documents=args.num_documents,
            num_questions=args.num_questions,
            top_k=args.top_k,
            cache_dir=cache_dir,
            verbose=args.verbose,
            use_reranker=args.use_reranker,
            reranker_top_k=args.reranker_top_k,
            dataset_name=args.dataset_name,
            gpu_device=args.gpu_device,
            rebuild_cache=args.rebuild_cache,
        )

        results = baseline.run_evaluation()

        overall_metrics, per_example_metrics = baseline.compute_metrics(results)

        console.print("\n" + "=" * 60)
        console.print("[bold green]Evaluation Results:[/bold green]")
        console.print(format_evaluation_report(overall_metrics, per_example_metrics, show_examples=5))

        # Token usage summary
        console.print("\n[bold cyan]Token Usage Summary:[/bold cyan]")
        total_tokens = sum(r.token_count for r in results)
        avg_tokens = total_tokens / len(results) if results else 0
        min_tokens = min((r.token_count for r in results), default=0)
        max_tokens = max((r.token_count for r in results), default=0)

        console.print(f"Total tokens: {total_tokens:,}")
        console.print(f"Average tokens per question: {avg_tokens:.0f}")
        console.print(f"Min tokens: {min_tokens:,}")
        console.print(f"Max tokens: {max_tokens:,}")

        baseline.save_results(results, overall_metrics, per_example_metrics)

        console.print("\n[bold green]✓ QA Vector Store Baseline evaluation completed![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
