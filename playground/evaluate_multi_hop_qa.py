#!/usr/bin/env python3
"""
Multi-Hop Question Answering Evaluation Script

Evaluates the multi-hop QA system on 2WikiMultihopQA dataset:
1. Loads first 200 docs and first 20 questions from 2WikiMultihopQA
2. Runs evaluation through the multi-hop QA pipeline
3. Compares results with HippoRAG baseline using fair evaluation metrics
4. Outputs comprehensive evaluation report
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import our simplified multi-hop QA system
from playground.multi_hop_qa_simple import SimplifiedMultiHopQA
# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch, format_evaluation_report

console = Console()

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"multihop_qa_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    question_id: str
    question: str
    predicted_answer: str
    gold_answers: List[str]
    processing_time: float
    decomposed_questions: List[Dict[str, Any]]
    final_prompt: Optional[str] = None
    error: Optional[str] = None

class MultiHopQAEvaluator:
    """Evaluator for simplified multi-hop QA system on 2WikiMultihopQA dataset."""
    
    def __init__(self, num_documents: int = 200, num_questions: int = 20, verbose: bool = False, show_prompt: bool = False):
        """Initialize evaluator.
        
        Args:
            num_documents: Number of documents to load (first N from corpus)
            num_questions: Number of questions to evaluate (first N from dataset)
            verbose: Whether to show detailed output during evaluation
            show_prompt: Whether to show LLM prompts during evaluation
        """
        self.num_documents = num_documents
        self.num_questions = num_questions
        self.data_dir = Path(".data/2wiki")
        self.verbose = verbose
        self.show_prompt = show_prompt
        
        if verbose:
            console.print("[bold blue]Initializing Multi-Hop QA Evaluator...[/bold blue]")
        
        # Initialize the simplified multi-hop QA system with verbose control
        self.qa_system = SimplifiedMultiHopQA(num_documents=num_documents, verbose=verbose, show_prompt=show_prompt)
        
        if verbose:
            console.print(f"[green]✓ Evaluator ready for {num_questions} questions on {num_documents} documents[/green]")
        else:
            console.print(f"[cyan]Evaluator initialized: {num_questions} questions, {num_documents} documents[/cyan]")
    
    def load_questions_and_answers(self) -> List[Tuple[str, str, List[str]]]:
        """Load questions and gold answers from 2WikiMultihopQA dataset.
        
        Returns:
            List of tuples: (question_id, question, gold_answers)
        """
        if self.verbose:
            console.print("[cyan]Loading 2WikiMultihopQA questions...[/cyan]")
        
        questions_file = self.data_dir / "2wikimultihopqa.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
        
        # Extract first N questions
        questions_data = []
        for i, item in enumerate(data[:self.num_questions]):
            question_id = item.get("_id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("answer", [])
            
            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            
            questions_data.append((question_id, question, gold_answers))
        
        if self.verbose:
            console.print(f"[green]✓ Loaded {len(questions_data)} questions[/green]")
        return questions_data
    
    def load_baseline_results(self) -> Dict[str, Dict[str, Any]]:
        """Load HippoRAG baseline results for comparison.
        
        Returns:
            Dictionary mapping question IDs to baseline results
        """
        if self.verbose:
            console.print("[cyan]Loading HippoRAG baseline results...[/cyan]")
        
        baseline_file = self.data_dir / "2wikimultihopqa_20q_result_dict_hipporagv2.json"
        if not baseline_file.exists():
            console.print(f"[yellow]Warning: Baseline file not found: {baseline_file}[/yellow]")
            return {}
        
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        if self.verbose:
            console.print(f"[green]✓ Loaded baseline results for {len(baseline_data)} questions[/green]")
        return baseline_data
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the final answer from LLM response.
        
        Args:
            response: Full LLM response with reasoning and answer tags
            
        Returns:
            Extracted answer string
        """
        # Look for <answer> tags first
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: look for "Answer:" patterns
        answer_match = re.search(r'(?:final\s+)?answer:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Last resort: take the last line if it looks like an answer
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # Remove common prefixes
            for prefix in ['Answer:', 'Final answer:', 'The answer is:', 'A:']:
                if last_line.lower().startswith(prefix.lower()):
                    return last_line[len(prefix):].strip()
            return last_line
        
        return response.strip()
    
    def evaluate_single_question(self, question_id: str, question: str, gold_answers: List[str]) -> EvaluationResult:
        """Evaluate a single question through the multi-hop QA pipeline.
        
        Args:
            question_id: Unique identifier for the question
            question: The question text
            gold_answers: List of acceptable gold answers
            
        Returns:
            EvaluationResult with prediction and metadata
        """
        start_time = datetime.now()
        
        try:
            # Run the simplified multi-hop QA pipeline (disable verbose output for clean evaluation)
            result = self.qa_system.process_multihop_question(question)
            
            # Extract the final answer from the result
            predicted_answer = self.extract_answer_from_response(result['answer'])
            
            # Get decomposed questions from the result
            decomposed_questions = result.get('decomposed_questions', [])
            
            # Get the final prompt that was used
            final_prompt = result.get('final_prompt', None)
            
            processing_time = result.get('time_taken', (datetime.now() - start_time).total_seconds())
            
            return EvaluationResult(
                question_id=question_id,
                question=question,
                predicted_answer=predicted_answer,
                gold_answers=gold_answers,
                processing_time=processing_time,
                decomposed_questions=decomposed_questions,
                final_prompt=final_prompt,
                error=None
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            console.print(f"[red]Error processing question {question_id}: {e}[/red]")
            
            return EvaluationResult(
                question_id=question_id,
                question=question,
                predicted_answer="[ERROR]",
                gold_answers=gold_answers,
                processing_time=processing_time,
                decomposed_questions=[],
                final_prompt=None,
                error=str(e)
            )
    
    def run_evaluation(self) -> List[EvaluationResult]:
        """Run evaluation on all questions.
        
        Returns:
            List of evaluation results
        """
        console.print(f"\n[bold magenta]Running evaluation on {self.num_questions} questions...[/bold magenta]")
        
        # Load questions
        questions_data = self.load_questions_and_answers()
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing questions...", total=len(questions_data))
            
            for question_id, question, gold_answers in questions_data:
                if self.verbose:
                    console.print(f"\n[bold yellow]Question {len(results) + 1}/{len(questions_data)}:[/bold yellow] {question}")
                
                # Evaluate single question
                result = self.evaluate_single_question(question_id, question, gold_answers)
                results.append(result)
                
                # Show result based on verbose setting
                if self.verbose:
                    if result.error:
                        console.print(f"[red]✗ Error: {result.error}[/red]")
                    else:
                        console.print(f"[green]✓ Predicted:[/green] {result.predicted_answer}")
                        console.print(f"[blue]  Gold:[/blue] {result.gold_answers}")
                        console.print(f"[dim]  Time: {result.processing_time:.1f}s[/dim]")
                else:
                    # Minimal output for non-verbose mode
                    status = "✗" if result.error else "✓"
                    console.print(f"[dim]Q{len(results)}: {status}[/dim]", end=" ")
                    if len(results) % 10 == 0:  # New line every 10 questions
                        console.print()
                
                progress.update(task, advance=1)
        
        return results
    
    def compute_metrics(self, results: List[EvaluationResult]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute evaluation metrics using HippoRAG methodology.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Tuple of (overall_metrics, per_example_metrics)
        """
        if self.verbose:
            console.print("\n[cyan]Computing evaluation metrics...[/cyan]")
        
        # Filter out error cases
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            console.print("[red]No valid results to evaluate![/red]")
            return {}, []
        
        # Prepare data for HippoRAG evaluation
        gold_answers_list = [r.gold_answers for r in valid_results]
        predicted_answers = [r.predicted_answer for r in valid_results]
        
        # Compute metrics using HippoRAG evaluation
        overall_metrics, per_example_metrics = evaluate_qa_batch(gold_answers_list, predicted_answers)
        
        return overall_metrics, per_example_metrics
    
    def load_and_display_baseline_results(self, baseline_results: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Load, process and display HippoRAG baseline results.
        
        Args:
            baseline_results: Baseline results data
            
        Returns:
            Tuple of (baseline_metrics, baseline_per_example_metrics)
        """
        console.print("\n[bold blue]HippoRAG Baseline Results:[/bold blue]")
        
        if not baseline_results:
            console.print("[yellow]No baseline data available[/yellow]")
            return {}, []
        
        # Extract baseline predictions and gold answers
        baseline_predictions = []
        baseline_golds = []
        baseline_questions = []
        
        for i, (_, data) in enumerate(list(baseline_results.items())[:self.num_questions]):
            query = data.get('query', f'Question {i+1}')
            gold_answers = data.get('gold_answers', [])
            
            # Try to extract predicted answer from different possible keys
            predicted_answer = data.get('hipporag_2_predicted_answer', "")
            
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            
            baseline_predictions.append(predicted_answer)
            baseline_golds.append(gold_answers)
            baseline_questions.append(query)
        
        if not baseline_predictions:
            console.print("[yellow]Could not extract baseline predictions[/yellow]")
            return {}, []
        
        # Compute baseline metrics
        baseline_metrics, baseline_per_example = evaluate_qa_batch(baseline_golds, baseline_predictions)
        
        # Display baseline results summary
        console.print(f"[cyan]HippoRAG evaluated on {len(baseline_predictions)} questions[/cyan]")
        console.print(format_evaluation_report(baseline_metrics, baseline_per_example, show_examples=5))
        
        # Add question info to per-example metrics
        for i, (question, example_metrics) in enumerate(zip(baseline_questions, baseline_per_example)):
            example_metrics['question'] = question
            example_metrics['question_id'] = f"baseline_q_{i}"
        
        return baseline_metrics, baseline_per_example
    
    def compare_with_baseline(self, our_metrics: Dict[str, float], our_per_example: List[Dict[str, Any]], 
                            baseline_metrics: Dict[str, float], baseline_per_example: List[Dict[str, Any]], questions: List[str]) -> None:
        """Compare our results with HippoRAG baseline with detailed comparison.
        
        Args:
            our_metrics: Our overall metrics
            our_per_example: Our per-example metrics
            baseline_metrics: Baseline overall metrics
            baseline_per_example: Baseline per-example metrics
            questions: List of questions
        """
        console.print("\n[bold blue]📊 Detailed Comparison: Our System vs HippoRAG Baseline[/bold blue]")
        
        if not baseline_metrics:
            console.print("[yellow]No baseline metrics available for comparison[/yellow]")
            return
        
        # Overall comparison table
        table = Table(title="Overall Performance Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Our System", style="green")
        table.add_column("HippoRAG Baseline", style="blue")
        table.add_column("Difference", style="yellow")
        table.add_column("% Change", style="magenta")
        
        for metric in ['ExactMatch', 'F1']:
            our_score = our_metrics.get(metric, 0.0)
            baseline_score = baseline_metrics.get(metric, 0.0)
            diff = our_score - baseline_score
            pct_change = (diff / baseline_score * 100) if baseline_score > 0 else 0
            
            diff_str = f"{diff:+.4f}"
            pct_str = f"{pct_change:+.1f}%"
            
            table.add_row(
                metric,
                f"{our_score:.4f}",
                f"{baseline_score:.4f}",
                diff_str,
                pct_str
            )
        
        console.print(table)
        
        # Per-question comparison for first few examples
        console.print(f"\n[bold cyan]Per-Question Comparison (first 10 questions):[/bold cyan]")
        
        comparison_table = Table()
        comparison_table.add_column("Q#", style="dim", width=3)
        comparison_table.add_column("Question", style="white", width=40)
        comparison_table.add_column("Our Answer", style="green", width=20)
        comparison_table.add_column("HippoRAG Answer", style="blue", width=20)
        comparison_table.add_column("Gold", style="yellow", width=15)
        comparison_table.add_column("Our EM", style="green", width=6)
        comparison_table.add_column("Hip EM", style="blue", width=6)
        
        min_examples = min(len(our_per_example), len(baseline_per_example), 10)
        
        for i in range(min_examples):
            our_ex = our_per_example[i]
            baseline_ex = baseline_per_example[i]
            
            # Truncate long text
            question = questions[i]
            if len(question) > 40:
                question = question[:37] + "..."
                
            our_answer = our_ex.get('predicted_answer', 'N/A')
            if len(our_answer) > 20:
                our_answer = our_answer[:17] + "..."
                
            baseline_answer = baseline_ex.get('predicted_answer', 'N/A')
            if len(baseline_answer) > 20:
                baseline_answer = baseline_answer[:17] + "..."
                
            gold_str = str(our_ex.get('gold_answers', ['N/A'])[0])
            if len(gold_str) > 15:
                gold_str = gold_str[:12] + "..."
            
            our_em = "✅" if our_ex.get('ExactMatch', 0) == 1.0 else "❌"
            baseline_em = "✅" if baseline_ex.get('ExactMatch', 0) == 1.0 else "❌"
            
            comparison_table.add_row(
                str(i+1),
                question,
                our_answer,
                baseline_answer,
                gold_str,
                our_em,
                baseline_em
            )
        
        console.print(comparison_table)
        
        # Win/Loss summary
        our_wins = sum(1 for i in range(min_examples) 
                      if our_per_example[i].get('ExactMatch', 0) > baseline_per_example[i].get('ExactMatch', 0))
        baseline_wins = sum(1 for i in range(min_examples) 
                           if baseline_per_example[i].get('ExactMatch', 0) > our_per_example[i].get('ExactMatch', 0))
        ties = min_examples - our_wins - baseline_wins
        
        console.print(f"\n[bold]Head-to-head (first {min_examples} questions):[/bold]")
        console.print(f"[green]Our System Wins: {our_wins}[/green]")
        console.print(f"[blue]HippoRAG Wins: {baseline_wins}[/blue]")
        console.print(f"[yellow]Ties: {ties}[/yellow]")
    
    def save_results(self, results: List[EvaluationResult], overall_metrics: Dict[str, float], per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        output_file = LOG_DIR / f"multihop_qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "evaluation_info": {
                "num_documents": self.num_documents,
                "num_questions": self.num_questions,
                "timestamp": datetime.now().isoformat(),
                "total_errors": sum(1 for r in results if r.error is not None)
            },
            "overall_metrics": overall_metrics,
            "per_question_results": []
        }
        
        # Add per-question details
        for result, metrics in zip(results, per_example_metrics):
            metrics.pop('predicted_answer', None)
            metrics.pop('gold_answers', None)
            question_data = {
                "question_id": result.question_id,
                "question": result.question,
                "predicted_answer": result.predicted_answer,
                "gold_answers": result.gold_answers,
                "metrics": metrics,
                "processing_time": result.processing_time,
                "decomposed_questions": result.decomposed_questions,
                "final_prompt": result.final_prompt,
                "error": result.error
            }
            output_data["per_question_results"].append(question_data)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓ Results saved to: {output_file}[/green]")


def main(verbose: bool = False, show_prompt: bool = False):
    """Main evaluation function.
    
    Args:
        verbose: Whether to show detailed output during evaluation
        show_prompt: Whether to show LLM prompts during evaluation
    """
    console.print("\n[bold cyan]🔗 Simplified Multi-Hop QA Evaluation on 2WikiMultihopQA[/bold cyan]")
    console.print("Evaluating simplified system performance against HippoRAG baseline")
    
    try:
        # Initialize evaluator with verbose control
        evaluator = MultiHopQAEvaluator(num_documents=-1, num_questions=20, verbose=verbose, show_prompt=show_prompt)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Compute metrics
        overall_metrics, per_example_metrics = evaluator.compute_metrics(results)
        
        # Display our results
        console.print("\n" + "="*60)
        console.print("[bold green]Our Multi-Hop QA System Results:[/bold green]")
        console.print(format_evaluation_report(overall_metrics, per_example_metrics, show_examples=5))
        
        # Load, display, and compare with baseline
        baseline_results = evaluator.load_baseline_results()
        baseline_metrics, baseline_per_example = evaluator.load_and_display_baseline_results(baseline_results)
        questions = [r.question for r in results]
        # Detailed comparison
        evaluator.compare_with_baseline(overall_metrics, per_example_metrics, baseline_metrics, baseline_per_example, questions)
        
        # Save results
        evaluator.save_results(results, overall_metrics, per_example_metrics)
        
        console.print("\n[bold green]✓ Evaluation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # You can set verbose=True and show_prompt=True here for detailed output during development
    # main(verbose=True, show_prompt=True)  # For debugging prompts and detailed output
    main(verbose=False, show_prompt=False)  # Default to non-verbose for clean evaluation runs