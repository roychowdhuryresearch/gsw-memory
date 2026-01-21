#!/usr/bin/env python3
"""
ABLATION STUDY: NO QUESTION DECOMPOSITION

This script evaluates multi-hop QA WITHOUT question decomposition.
Instead of breaking questions into sub-questions, it uses the original
question directly for retrieval.

ABLATION:
- ❌ Question decomposition (Stage 1 skipped)
- ✅ Chain-following retrieval (simplified)
- ✅ Reranking (both chain and Q&A)
- ✅ Oracle-style answer generation

Comparison baseline: evaluate_multi_hop_qa_chains_batched.py
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from dataclasses import dataclass
from datetime import datetime
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pydantic import BaseModel
from typing import List

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
# os.environ["CURATOR_DISABLE_CACHE"] = "1"
# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our chain-following multi-hop QA system for retrieval
from multi_hop_qa_chains import ChainFollowingMultiHopQA

# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch, format_evaluation_report

# Import curator for batched LLM processing
try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    print("Warning: Curator not available. Install with: pip install bespokelabs-curator")
    CURATOR_AVAILABLE = False

console = Console()

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"multihop_qa_NO_DECOMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


class ChainAnswerGenerator(curator.LLM):
    """Curator class for generating answers with oracle-style prompting in parallel."""
    
    return_completions_object = True
    
    def __init__(self, **kwargs):
        """Initialize the answer generator."""
        super().__init__(**kwargs)
    
    def prompt(self, input):
        """Create an oracle-style answer generation prompt."""
        evidence_text = '\n'.join(input['evidence'])
        
        # Build oracle-style prompt with one-shot example
        prompt_text = f"""
{evidence_text}
\n\nQuestion: {input['question']}
\n\nThought: 

"""
        
        # Store the full prompt for later reference
        self._last_prompt = prompt_text
        
        # One-shot example with Q&A pairs format (concise entity-focused answers)
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
        
        # System message for advanced reading comprehension
        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
            # 'If you don\'t know the answer, say "No Answer".'
        )
        
        # One-shot example input
        one_shot_input = (
            f"{one_shot_docs}"
            "\n\nQuestion: "
            "When was Neville A. Stanton's employer founded?"
            '\nThought: '
        )
        
        # One-shot example output
        one_shot_output = (
            "From the QA pairs, the employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
            "\nAnswer: 1862."
        )
        
        # Build the prompt template
        prompt_messages = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": one_shot_output},
            {"role": "user", "content": prompt_text}
        ]    

        return prompt_messages

    def parse(self, input, response):
        """Parse the answer from the response with oracle-style format."""
        answer_text = response["choices"][0]["message"]["content"]
        
        # Parse answer with new format (Answer: format)
        if 'Answer: ' in answer_text:
            final_answer = answer_text.split('Answer: ')[-1].strip()
            # Remove any trailing period if it's just a number/date
            if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                final_answer = final_answer[:-1]
        else:
            # Fallback to full response if no "Answer:" found
            final_answer = answer_text.strip()
        
        # Focus on evidence only for reporting token usage
        evidence_text = '\n'.join(input.get('evidence', []))
        token_count = 0
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            token_count = len(encoding.encode(evidence_text))
        except Exception:
            # Fallback rough estimate if tokenizer is unavailable
            token_count = max(1, len(evidence_text) // 4)
        
        return [{
            "question_id": input['question_id'],
            "question": input['question'],
            "predicted_answer": final_answer,
            "full_response": answer_text,
            "evidence_count": len(input['evidence']),
            "evidence": input['evidence'],
            "final_prompt": None,
            "token_count": token_count
        }]


@dataclass
class ChainBatchedEvaluationResult:
    """Container for chain-based batched evaluation results."""
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
    final_prompt: Optional[str] = None
    error: Optional[str] = None


class ChainBatchedMultiHopQAEvaluator:
    """Batched evaluator for chain-following multi-hop QA using curator for parallel LLM calls."""
    
    def __init__(self, num_documents: int = 200, num_questions: int = 20, verbose: bool = False,
                 chain_top_k: int = 15, use_bm25: bool = False):
        """Initialize batched chain evaluator.
        
        Args:
            num_documents: Number of documents to load
            num_questions: Number of questions to evaluate
            verbose: Whether to show detailed output
            chain_top_k: Number of top chains to select after reranking
        """
        self.num_documents = num_documents
        self.num_questions = num_questions
        self.data_dir = Path("/home/yigit/codebase/gsw-memory/playground_data")
        self.verbose = verbose
        self.chain_top_k = chain_top_k
        self.use_bm25 = use_bm25
        
        console.print("[bold blue]Initializing Chain-Based Batched Multi-Hop QA Evaluator...[/bold blue]")
        console.print(f"  Chain selection: Top {chain_top_k} chains")
        
        # Initialize the chain-following retrieval system (single instance for all questions)
        self.qa_system = ChainFollowingMultiHopQA(
            num_documents=num_documents, 
            verbose=False, 
            show_prompt=False,
            chain_top_k=chain_top_k,
            chain_following_mode="cumulative", 
            beam_width=5, 
            reranker_model_name="voyage",
            use_chain_reranker=True,
            use_bm25=use_bm25

        )
        # Initialize curator classes if available
        if CURATOR_AVAILABLE:
            os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"
            self.answer_generator = ChainAnswerGenerator(
                model_name="gpt-4o-mini", 
                generation_params={"temperature": 0},
                # model_name="hosted_vllm/Qwen/Qwen3-8B",
                # backend = "litellm",
                # backend_params = {
                #     "base_url": "http://127.0.0.1:6380/v1",
                #     "request_timeout": 600.0,  
                #     "max_concurrent_requests": 32,
                #     "max_requests_per_minute": 120,
                #     "max_tokens_per_minute": 200000,
                #    "seconds_to_pause_on_rate_limit": 5,
                #     "require_all_responses": False,
                # },
                # generation_params={
                #     "temperature": 0.6, 
                #     "top_p": 0.95, 
                #     "top_k": 20, 
                #     "min_p": 0, 
                #     "max_tokens": 4096, 
                #     "repetition_penalty": 1.1,
                #     "presence_penalty": 0.3,
                #     "frequency_penalty": 0.3
                #             }
            )
            console.print("[green]✓ Curator initialized for parallel processing[/green]")
        else:
            console.print("[yellow]⚠ Curator not available - will fall back to sequential processing[/yellow]")
        
        console.print(f"[cyan]Ready to evaluate {num_questions} questions on {num_documents} documents[/cyan]")
    
    def load_questions_and_answers(self) -> List[Tuple[str, str, List[str]]]:
        """Load questions and gold answers from 2WikiMultihopQA dataset.
        
        Returns:
            List of tuples: (question_id, question, gold_answers)
        """
        console.print("[cyan]Loading Musique questions...[/cyan]")
        
        questions_file = self.data_dir / "musique_multihopqa.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
        
        # Extract first N questions
        questions_data = []
        for i, item in enumerate(data[:self.num_questions]):
            # if "2hop" in item["id"] or "3hop1" in item["id"]:
            question_id = item.get("_id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("answer", [])
            # gold_answers = item.get("gold_ans", [])
            # gold_answers = item.get("reference", [])
            # gold_answers = item.get("possible_answers", [])
            # In the dataset, possible_answers is a JSON-encoded string for lists; e.g., '["cartoonist", "graphic artist", "animator", "illustrator"]'
            # Fix to always decode it if it is a string:
            # if isinstance(gold_answers, str):
            #     try:
            #         gold_answers = json.loads(gold_answers)
            #     except Exception:
            #         gold_answers = [gold_answers]
                
            # else:
            #     continue
            
            # Ensure gold_answers is a list and add aliases
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
                gold_answers.extend(item.get("answer_aliases", []))
            else:
                # answer is already a list, still need to add aliases
                answer_aliases = item.get("answer_aliases", [])
                gold_answers = gold_answers + answer_aliases
            
            questions_data.append((question_id, question, gold_answers))
        
        console.print(f"[green]✓ Loaded {len(questions_data)} questions[/green]")
        return questions_data
    
    def run_batched_decomposition(self, questions_data: List[Tuple[str, str, List[str]]]) -> Dict[str, Dict]:
        """ABLATION: Skip decomposition and use original questions directly.

        Args:
            questions_data: List of (question_id, question, gold_answers)

        Returns:
            Dictionary mapping question_id to "decomposition" results (just original question)
        """
        console.print("\n[bold yellow]Stage 1: SKIPPED - No Question Decomposition (Ablation)[/bold yellow]")
        console.print("[dim]Using original questions directly without decomposition[/dim]")

        # Create fake decomposition results with just the original question
        decomposition_results = {}
        for question_id, question, _ in questions_data:
            decomposition_results[question_id] = {
                "original_question": question,
                "decomposed_questions": [
                    {
                        "question": question,
                        "requires_retrieval": True
                    }
                ]
            }

        console.print(f"[green]✓ Prepared {len(decomposition_results)} questions (no decomposition)[/green]")
        return decomposition_results
    
    def run_chain_retrieval_stage(self, questions_data: List[Tuple[str, str, List[str]]], 
                                  decomposition_results: Dict[str, Dict]) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
        """Run chain-following information retrieval for all questions.
        
        Args:
            questions_data: Original questions data
            decomposition_results: Decomposition results from Stage 1
            
        Returns:
            Tuple of (evidence_by_question, chains_info_by_question)
        """
        console.print("\n[bold cyan]Stage 2: Chain-Following Information Retrieval[/bold cyan]")
        
        evidence_by_question = {}
        chains_info_by_question = {}
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
            task = progress.add_task("Retrieving evidence with chain following...", total=len(questions_data))
            
            for question_id, question, _ in questions_data:
                if self.verbose:
                    console.print(f"\n[dim]Processing: {question}[/dim]")
                try:
                    decomposed = decomposition_results[question_id]["decomposed_questions"]
                except:
                    print(f"Error: {question_id} not found in decomposition results") #TODO: Handle this error
                    print("Skipping question...") #TODO: Handle this error
                    continue

                # Reuse the exact retrieval logic from ChainFollowingMultiHopQA
                all_evidence, chains_info, _ = self.qa_system.collect_evidence(
                    question, decomposed=decomposed
                )
                
                evidence_by_question[question_id] = all_evidence
                chains_info_by_question[question_id] = chains_info
                
                if self.verbose:
                    console.print(f"  [green]Collected {len(all_evidence)} pieces of evidence[/green]")
                    if chains_info and not chains_info.get('fallback', False):
                        console.print(f"  [dim]Chains: {chains_info.get('total_chains', 0)} formed, {chains_info.get('selected_chains', 0)} selected[/dim]")
                    else:
                        console.print(f"  [dim]Used fallback retrieval[/dim]")
                
                # Debug: Print first few evidence items if verbose
                if self.verbose and all_evidence:
                    console.print(f"  [dim]Sample evidence: {all_evidence[0][:80]}...[/dim]")
                elif self.verbose and not all_evidence:
                    console.print(f"  [red]WARNING: No evidence collected for question![/red]")
                
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Chain retrieval complete for {len(evidence_by_question)} questions[/green]")
        return evidence_by_question, chains_info_by_question
    
    # Note: retrieval is now delegated entirely to ChainFollowingMultiHopQA.collect_evidence
    
    def run_batched_answer_generation(self, questions_data: List[Tuple[str, str, List[str]]], 
                                     evidence_by_question: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Run batched answer generation using curator with oracle-style prompting.
        
        Args:
            questions_data: Original questions data
            evidence_by_question: Evidence collected in Stage 2
            
        Returns:
            Dictionary mapping question_id to answer results
        """
        console.print("\n[bold cyan]Stage 3: Batched Answer Generation (Oracle-Style)[/bold cyan]")
        
        if not CURATOR_AVAILABLE:
            # Fallback to sequential processing
            console.print("[yellow]Using sequential answer generation (curator not available)[/yellow]")
            answer_results = {}
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
                task = progress.add_task("Generating answers...", total=len(questions_data))
                
                for question_id, question, _ in questions_data:
                    evidence = evidence_by_question.get(question_id, [])
                    answer = self.qa_system.generate_answer(question, evidence)
                    
                    # Parse answer with new format
                    if 'Answer: ' in answer:
                        final_answer = answer.split('Answer: ')[-1].strip()
                        if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                            final_answer = final_answer[:-1]
                    else:
                        final_answer = answer.strip()
                    
                    # Focus on evidence only for outputs
                    final_prompt = None
                    # Compute token count over evidence text
                    ev_text = '\n'.join(evidence)
                    try:
                        import tiktoken
                        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                        token_count = len(encoding.encode(ev_text))
                    except Exception:
                        token_count = max(1, len(ev_text) // 4)
                    
                    answer_results[question_id] = {
                        "predicted_answer": final_answer,
                        "full_response": answer,
                        "evidence_count": len(evidence),
                        "evidence": evidence,
                        "final_prompt": final_prompt,
                        "token_count": token_count
                    }
                    progress.update(task, advance=1)
            
            return answer_results
        
        # Prepare inputs for batched answer generation
        answer_inputs = [
            {
                "question_id": qid,
                "question": question,
                "evidence": evidence_by_question.get(qid, [])
            }
            for qid, question, _ in questions_data
        ]
        
        console.print(f"[cyan]Generating answers for {len(answer_inputs)} questions in parallel...[/cyan]")
        start_time = time.time()
        
        # Run batched answer generation
        answer_dataset = self.answer_generator(answer_inputs)
        
        elapsed = time.time() - start_time
        console.print(f"[green]✓ Answer generation complete in {elapsed:.1f}s ({elapsed/len(answer_inputs):.2f}s per question)[/green]")
        
        # Convert to dictionary for easy lookup
        answer_results = {item["question_id"]: item for item in answer_dataset.dataset}
        
        return answer_results
    
    def run_evaluation(self) -> List[ChainBatchedEvaluationResult]:
        """Run complete batched evaluation pipeline with chain following.
        
        Returns:
            List of evaluation results
        """
        console.print(f"\n[bold magenta]Running Chain-Based Batched Evaluation on {self.num_questions} Questions[/bold magenta]")
        total_start = time.time()
        
        # Load questions
        questions_data = self.load_questions_and_answers()
        
        # Stage 1: Batched Decomposition
        decomposition_results = self.run_batched_decomposition(questions_data)
        
        # Stage 2: Chain-Following Information Retrieval
        evidence_by_question, chains_info_by_question = self.run_chain_retrieval_stage(
            questions_data, decomposition_results
        )
        
        # Stage 3: Batched Answer Generation with Oracle-Style Prompting
        answer_results = self.run_batched_answer_generation(questions_data, evidence_by_question)
        
        # Compile results
        results = []
        for question_id, question, gold_answers in questions_data:
            try:
                decomposed = decomposition_results[question_id]["decomposed_questions"]
                answer_data = answer_results[question_id]
                chains_info = chains_info_by_question.get(question_id, {})
                
                result = ChainBatchedEvaluationResult(
                    question_id=question_id,
                    question=question,
                    predicted_answer=answer_data["predicted_answer"],
                    full_response=answer_data.get("full_response", ""),
                    gold_answers=gold_answers,
                    processing_time=0.0,  # Will be updated with total time
                    decomposed_questions=decomposed,
                    evidence_count=answer_data["evidence_count"],
                    evidence=answer_data.get("evidence", []),
                    chains_info=chains_info,
                    token_count=answer_data.get("token_count", 0),
                    final_prompt=answer_data.get("final_prompt", None),
                    error=None
                )
                results.append(result)
            except Exception as e:
                print(f"Error: {question_id} not found in answer results") #TODO: Handle this error
                print("Skipping question...") #TODO: Handle this error
                continue
        
        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")
        
        # Update processing time
        for result in results:
            result.processing_time = total_elapsed / len(results)
        
        return results
    
    def compute_metrics(self, results: List[ChainBatchedEvaluationResult]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute evaluation metrics.

        Args:
            results: List of evaluation results

        Returns:
            Tuple of (overall_metrics, per_example_metrics)
        """
        console.print("\n[cyan]Computing evaluation metrics...[/cyan]")

        # Filter out error cases
        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            console.print("[red]No valid results to evaluate![/red]")
            return {}, []

        # Prepare data for evaluation
        gold_answers_list = [r.gold_answers for r in valid_results]
        predicted_answers = [r.predicted_answer for r in valid_results]

        # Compute metrics with substring matching enabled
        overall_metrics, per_example_metrics = evaluate_qa_batch(
            gold_answers_list,
            predicted_answers,
        )

        return overall_metrics, per_example_metrics
    
    def save_results(self, results: List[ChainBatchedEvaluationResult], overall_metrics: Dict[str, float], 
                     per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        output_file = LOG_DIR / f"multihop_qa_NO_DECOMP_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "evaluation_info": {
                "num_documents": self.num_documents,
                "num_questions": self.num_questions,
                "chain_top_k": self.chain_top_k,
                "timestamp": datetime.now().isoformat(),
                "batched": True,
                "chain_following": True,
                "oracle_prompting": True,
                "curator_available": CURATOR_AVAILABLE,
                "ablation": "NO_DECOMPOSITION",
                "ablation_description": "Questions are NOT decomposed; original question used directly"
            },
            "overall_metrics": overall_metrics,
            "per_question_results": []
        }
        
        # Add per-question details
        for result, metrics in zip(results, per_example_metrics):
            question_data = {
                "question_id": result.question_id,
                "question": result.question,
                "predicted_answer": result.predicted_answer,
                "full_response": result.full_response,
                "gold_answers": result.gold_answers,
                "metrics": metrics,
                "processing_time": result.processing_time,
                "decomposed_questions": result.decomposed_questions,
                "evidence_count": result.evidence_count,
                "evidence": result.evidence,
                "chains_info": result.chains_info,
                "token_count": result.token_count,
                "final_prompt": result.final_prompt,
                "error": result.error
            }
            output_data["per_question_results"].append(question_data)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓ Results saved to: {output_file}[/green]")
    
    def display_chain_statistics(self, results: List[ChainBatchedEvaluationResult]) -> None:
        """Display statistics about chain formation and selection.
        
        Args:
            results: List of evaluation results
        """
        console.print("\n[bold cyan]Chain Statistics:[/bold cyan]")
        
        chain_results = [r for r in results if r.chains_info and not r.chains_info.get('fallback', False)]
        fallback_results = [r for r in results if r.chains_info and r.chains_info.get('fallback', False)]
        
        if chain_results:
            total_chains = sum(r.chains_info.get('total_chains', 0) for r in chain_results)
            selected_chains = sum(r.chains_info.get('selected_chains', 0) for r in chain_results)
            avg_chains = total_chains / len(chain_results) if chain_results else 0
            avg_selected = selected_chains / len(chain_results) if chain_results else 0
            
            console.print(f"Questions using chain following: {len(chain_results)}")
            console.print(f"Questions using fallback: {len(fallback_results)}")
            console.print(f"Total chains formed: {total_chains}")
            console.print(f"Total chains selected: {selected_chains}")
            console.print(f"Average chains per question: {avg_chains:.1f}")
            console.print(f"Average selected chains: {avg_selected:.1f}")
        else:
            console.print("No chain-following results found (all used fallback)")


def main(verbose: bool = False):
    """Main evaluation function.

    Args:
        verbose: Whether to show detailed output
    """
    console.print("\n[bold cyan]🚀 ABLATION: Multi-Hop QA WITHOUT Question Decomposition[/bold cyan]")
    console.print("[yellow]ABLATION: Questions are used directly without decomposition[/yellow]")
    console.print("Using simplified retrieval with oracle-style prompting")
    
    try:
        # Initialize evaluator
        evaluator = ChainBatchedMultiHopQAEvaluator(
            num_documents=-1, 
            num_questions=1000, 
            verbose=verbose,
            use_bm25=True
       )

        
        # Run batched evaluation
        results = evaluator.run_evaluation()
        
        # Compute metrics
        overall_metrics, per_example_metrics = evaluator.compute_metrics(results)
        
        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]Evaluation Results:[/bold green]")
        console.print(format_evaluation_report(overall_metrics, per_example_metrics, show_examples=5))
        
        # Display chain statistics
        evaluator.display_chain_statistics(results)
        
        # Display token usage summary
        console.print("\n[bold cyan]Token Usage Summary:[/bold cyan]")
        total_tokens = sum(r.token_count for r in results)
        avg_tokens = total_tokens / len(results) if results else 0
        min_tokens = min((r.token_count for r in results), default=0)
        max_tokens = max((r.token_count for r in results), default=0)
        
        console.print(f"Total tokens: {total_tokens:,}")
        console.print(f"Average tokens per question: {avg_tokens:.0f}")
        console.print(f"Min tokens: {min_tokens:,}")
        console.print(f"Max tokens: {max_tokens:,}")
        
        # Save results
        evaluator.save_results(results, overall_metrics, per_example_metrics)
        
        console.print("\n[bold green]✓ Chain-based batched evaluation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # Set verbose=True for detailed output during development
    main(verbose=False)
