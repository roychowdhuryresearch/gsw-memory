#!/usr/bin/env python3
"""
Simple Single Model Testing Script

Tests a specified model with the exact same retrieval context from previous evaluation results.
This isolates model reasoning performance from retrieval performance.

Usage:
    python test_single_model.py --model gpt-4o --results-file logs/multihop_qa_results_xxx.json
    python test_single_model.py --model claude-3-5-sonnet-20241022 --results-file logs/multihop_qa_results_xxx.json --num-questions 10
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import tiktoken

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch

# Import curator for LLM processing
try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    print("Error: Curator not available. Install with: pip install bespokelabs-curator")
    sys.exit(1)

console = Console()


class ModelAnswerGenerator(curator.LLM):
    """Curator class for generating answers with a specific model."""
    
    return_completions_object = True
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the answer generator with a specific model."""
        self.model_name = model_name
        super().__init__(model_name=model_name, **kwargs)
    
    def prompt(self, input_data):
        """Use the exact same prompt from the original evaluation."""
        final_prompt = input_data['final_prompt']
        
        # Store for token counting
        self._last_prompt = final_prompt
        
        return [
            {"role": "system", "content": "You are a helpful assistant that answers questions using only provided evidence."},
            {"role": "user", "content": final_prompt}
        ]
    
    def parse(self, input_data, response):
        """Parse the answer from the response."""
        answer_text = response["choices"][0]["message"]["content"]
        
        # Extract answer from tags (same logic as original evaluation)
        if '</answer>' in answer_text:
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer_text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                # If we have opening tag but no closing tag
                answer_parts = answer_text.split('<answer>')
                if len(answer_parts) > 1:
                    final_answer = answer_parts[1].strip()
                else:
                    final_answer = answer_text.strip()
        else:
            # Fallback to full response
            if '<answer>' in answer_text:
                final_answer = answer_text.split('<answer>')[1]
            else:
                final_answer = answer_text.strip()
        

        
        return [{
            "question_id": input_data['question_id'],
            "question": input_data['question'],
            "predicted_answer": final_answer,
            "gold_answers": input_data['gold_answers'],
            "full_response": answer_text,
            "token_count": input_data['token_count'],
            "final_prompt": input_data['final_prompt']
        }]


class SingleModelTester:
    """Test a single model with existing retrieval context."""
    
    def __init__(self, model_name: str, results_file: str, num_questions: Optional[int] = None):
        """Initialize the single model tester.
        
        Args:
            model_name: Name of the model to test
            results_file: Path to the original evaluation results JSON
            num_questions: Number of questions to test (None for all)
        """
        self.model_name = model_name
        self.results_file = Path(results_file)
        self.num_questions = num_questions
        
        console.print(f"[cyan]Testing model: {model_name}[/cyan]")
        console.print(f"[cyan]Loading results from: {self.results_file}[/cyan]")
        
        # Load original results
        with open(self.results_file, 'r') as f:
            self.original_results = json.load(f)
        
        # Extract questions that have final_prompt (needed for model testing)
        self.test_questions = []
        for result in self.original_results["per_question_results"]:
            if result.get("final_prompt") and not result.get("error"):
                self.test_questions.append(result)
        
        # Limit to specified number of questions
        if self.num_questions:
            self.test_questions = self.test_questions[:self.num_questions]
        
        console.print(f"[green]✓ Found {len(self.test_questions)} testable questions[/green]")
        
        # Initialize model generator
        try:
            self.generator = ModelAnswerGenerator(
                model_name=model_name,
                generation_params={"temperature": 0.0}
            )
            console.print(f"[green]✓ Initialized {model_name}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize {model_name}: {e}[/red]")
            sys.exit(1)
    
    def run_evaluation(self) -> List[Dict[str, Any]]:
        """Run the model on all test questions.
        
        Returns:
            List of result dictionaries
        """
        console.print(f"\n[bold cyan]🧪 Running {self.model_name} on {len(self.test_questions)} questions[/bold cyan]")
        
        # Prepare inputs for the model
        model_inputs = []
        for question_data in self.test_questions:

            # Compute token count of final_prompt
            encoding = tiktoken.encoding_for_model(self.model_name)
            token_count = len(encoding.encode(question_data["final_prompt"]))

            model_inputs.append({
                "question_id": question_data["question_id"],
                "question": question_data["question"],
                "final_prompt": question_data["final_prompt"],
                "gold_answers": question_data["gold_answers"],
                "token_count": token_count
            })
        
        # Run batched inference
        start_time = time.time()
        
        with Progress(SpinnerColumn(), TextColumn(f"[cyan]Processing {len(model_inputs)} questions..."), console=console) as progress:
            task = progress.add_task("Running model...", total=None)
            
            try:
                results_dataset = self.generator(model_inputs)
                elapsed = time.time() - start_time
                
                progress.update(task, completed=100, total=100)
                
            except Exception as e:
                console.print(f"[red]✗ Model evaluation failed: {e}[/red]")
                sys.exit(1)
        
        console.print(f"[green]✓ Completed in {elapsed:.1f}s ({elapsed/len(model_inputs):.2f}s per question)[/green]")
        
        # Add processing time to results
        results = []
        for result_data in results_dataset.dataset:
            result_data["processing_time"] = elapsed / len(model_inputs)
            results.append(result_data)
        
        return results
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute evaluation metrics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Tuple of (overall_metrics, per_example_metrics)
        """
        console.print(f"[cyan]Computing evaluation metrics...[/cyan]")
        
        # Prepare data for evaluation
        gold_answers_list = [r["gold_answers"] for r in results]
        predicted_answers = [r["predicted_answer"] for r in results]
        
        # Compute metrics using the same evaluation as original
        overall_metrics, per_example_metrics = evaluate_qa_batch(gold_answers_list, predicted_answers)
        
        return overall_metrics, per_example_metrics
    
    def display_results(self, overall_metrics: Dict[str, float], results: List[Dict[str, Any]]) -> None:
        """Display evaluation results.
        
        Args:
            overall_metrics: Overall performance metrics
            results: Individual question results
        """
        console.print("\n" + "="*60)
        console.print(f"[bold green]📊 Results for {self.model_name}[/bold green]")
        
        # Main metrics
        console.print(f"[cyan]Questions tested: {len(results)}[/cyan]")
        console.print(f"[green]ExactMatch: {overall_metrics.get('ExactMatch', 0):.3f}[/green]")
        console.print(f"[green]F1 Score: {overall_metrics.get('F1', 0):.3f}[/green]")
        
        # Timing and token metrics
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        avg_tokens = sum(r["token_count"] for r in results) / len(results)
        console.print(f"[yellow]Avg processing time: {avg_time:.2f}s[/yellow]")
        console.print(f"[magenta]Avg tokens per question: {avg_tokens:.0f}[/magenta]")
        
        # Compare with original if available
        if self.original_results.get("overall_metrics"):
            orig_f1 = self.original_results["overall_metrics"].get("F1", 0)
            orig_em = self.original_results["overall_metrics"].get("ExactMatch", 0)
            
            delta_f1 = overall_metrics.get("F1", 0) - orig_f1
            delta_em = overall_metrics.get("ExactMatch", 0) - orig_em
            
            console.print(f"\n[bold cyan]📈 Comparison with Original Results:[/bold cyan]")
            console.print(f"Original - F1: {orig_f1:.3f}, ExactMatch: {orig_em:.3f}")
            console.print(f"Delta - F1: {delta_f1:+.3f}, ExactMatch: {delta_em:+.3f}")
    
    def save_results(self, results: List[Dict[str, Any]], overall_metrics: Dict[str, float], 
                     per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: Individual question results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        output_file = Path("logs") / f"single_model_{self.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # Combine results with metrics
        per_question_results = []
        for result, metrics in zip(results, per_example_metrics):
            result_with_metrics = result.copy()
            result_with_metrics["metrics"] = metrics
            per_question_results.append(result_with_metrics)
        
        output_data = {
            "evaluation_info": {
                "model_name": self.model_name,
                "source_file": str(self.results_file),
                "num_questions": len(results),
                "timestamp": datetime.now().isoformat()
            },
            "overall_metrics": overall_metrics,
            "per_question_results": per_question_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓ Results saved to: {output_file}[/green]")
    
    def run_complete_evaluation(self) -> None:
        """Run the complete evaluation pipeline."""
        # Run model evaluation
        results = self.run_evaluation()
        
        # Compute metrics
        overall_metrics, per_example_metrics = self.compute_metrics(results)
        
        # Display results
        self.display_results(overall_metrics, results)
        
        # Save results
        self.save_results(results, overall_metrics, per_example_metrics)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test a single model with existing retrieval context")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to test (default: gpt-4o-mini)")
    parser.add_argument("--results-file", default="logs/multihop_qa_results_20250819_150508.json", help="Path to original evaluation results JSON")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to test (default: all)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.results_file).exists():
        console.print(f"[red]✗ Results file not found: {args.results_file}[/red]")
        sys.exit(1)
    
    console.print(f"[bold cyan]🧪 Single Model Testing[/bold cyan]")
    console.print(f"Model: {args.model}")
    console.print(f"Results file: {args.results_file}")
    if args.num_questions:
        console.print(f"Questions limit: {args.num_questions}")
    
    # Run evaluation
    tester = SingleModelTester(
        model_name=args.model,
        results_file=args.results_file,
        num_questions=args.num_questions
    )
    
    tester.run_complete_evaluation()
    
    console.print(f"\n[bold green]✅ Single model evaluation completed![/bold green]")


if __name__ == "__main__":
    main()