#!/usr/bin/env python3
"""
Oracle Retriever Evaluation Script

Tests the theoretical upper bound performance by providing models with 
the exact ground truth documents from 2wikimultihopqa.json.

This bypasses all retrieval and gives the LLM perfect information to
isolate reasoning performance from retrieval performance.

Usage:
    python oracle_eval.py --model gpt-4o --num-questions 20
    python oracle_eval.py --model gpt-4o --include-titles False
    python oracle_eval.py --model claude-3-5-sonnet-20241022 --include-titles True
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


class OracleAnswerGenerator(curator.LLM):
    """Curator class for generating answers with oracle ground truth documents."""
    
    return_completions_object = True
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the oracle answer generator."""
        self.model_name = model_name
        super().__init__(model_name=model_name, **kwargs)
    
    def prompt(self, input_data):
        """Create an oracle answer generation prompt with ground truth documents."""
        evidence_text = input_data['oracle_context']
        
        prompt_text = f"""
        {evidence_text}
        \n\nQuestion: " {input_data['question']}
        \n\nThought: 

"""
        # print(prompt_text)
        
        # Store for token counting
        self._last_prompt = prompt_text

        one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)



        one_shot_ircot_demo = (
            f'{one_shot_rag_qa_docs}'
            '\n\nQuestion: '
            f"When was Neville A. Stanton's employer founded?"
            '\nThought: '
            f"The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
            '\n\n'
        )


        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
            'DO NOT USE ANY EXTERNAL SOURCES. ONLY USE THE PROVIDED DOCUMENTS.'
        )

        one_shot_rag_qa_input = (
            f"{one_shot_rag_qa_docs}"
            "\n\nQuestion: "
            "When was Neville A. Stanton's employer founded?"
            '\nThought: '
        )

        one_shot_rag_qa_output = (
            "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
            "\nAnswer: 1862."
        )


        prompt_template = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_rag_qa_input},
            {"role": "assistant", "content": one_shot_rag_qa_output},
            {"role": "user", "content": prompt_text}
        ]

        # print(prompt_template)

        return prompt_template

        
        return [
            # {"role": "system", "content": "You are a helpful assistant that answers questions using only provided documents."},
            {"role": "system", "content": "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after 'Thought: ', where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with 'Answer: ' to present a concise, definitive response, devoid of additional elaborations."},
            # {"role": "system", "content": "You are a helpful assistant that answers questions using only provided documents."},
            {"role": "user", "content": prompt_text}
        ]
    
    def parse(self, input_data, response):
        """Parse the answer from the response."""
        answer_text = response["choices"][0]["message"]["content"]

        if 'Answer: ' in answer_text:
            final_answer = answer_text.split('Answer: ')[1].strip()
        else:
            final_answer = answer_text.strip()
        
        # Extract answer from tags (same logic as other evaluation scripts)
        # if '</answer>' in answer_text:
        #     answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer_text, re.DOTALL | re.IGNORECASE)
        #     if answer_match:
        #         final_answer = answer_match.group(1).strip()
        #     else:
        #         # If we have opening tag but no closing tag
        #         answer_parts = answer_text.split('<answer>')
        #         if len(answer_parts) > 1:
        #             final_answer = answer_parts[1].strip()
        #         else:
        #             final_answer = answer_text.strip()
        # else:
        #     # Fallback to full response
        #     if '<answer>' in answer_text:
        #         final_answer = answer_text.split('<answer>')[1].strip()
        #     else:
        #         final_answer = answer_text.strip()
        
        
        return [{
            "question_id": input_data['question_id'],
            "question": input_data['question'],
            "predicted_answer": final_answer,
            "gold_answers": input_data['gold_answers'],
            "full_response": answer_text,
            "token_count": input_data['token_count'],
            "oracle_context": input_data['oracle_context']
        }]


class OracleEvaluator:
    """Oracle evaluator using ground truth documents from 2wikimultihopqa."""
    
    def __init__(self, model_name: str, num_questions: Optional[int] = None, include_titles: bool = True, 
                 supporting_only: bool = True):
        """Initialize the oracle evaluator.
        
        Args:
            model_name: Name of the model to test
            num_questions: Number of questions to test (None for all)
            include_titles: Whether to include document titles in context
            supporting_only: Whether to use only supporting facts (True) or full context (False)
        """
        self.model_name = model_name
        self.num_questions = num_questions
        self.include_titles = include_titles
        self.supporting_only = supporting_only
        self.dataset_file = Path("/home/yigit/codebase/gsw-memory/playground_data/nq_rear.json")
        
        console.print(f"[cyan]Oracle Evaluation with model: {model_name}[/cyan]")
        console.print(f"[cyan]Include document titles: {include_titles}[/cyan]")
        console.print(f"[cyan]Supporting facts only: {supporting_only}[/cyan]")
        
        # Validate dataset file
        if not self.dataset_file.exists():
            console.print(f"[red]✗ Dataset file not found: {self.dataset_file}[/red]")
            sys.exit(1)
        
        # Initialize model generator
        try:
            self.generator = OracleAnswerGenerator(
                model_name=model_name,
                generation_params={"temperature": 0.0}
            )
            console.print(f"[green]✓ Initialized {model_name}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize {model_name}: {e}[/red]")
            sys.exit(1)
    
    def load_2wiki_dataset(self) -> List[Dict[str, Any]]:
        """Load questions and ground truth context from 2wikimultihopqa.json.
        
        Returns:
            List of question dictionaries with oracle context
        """
        console.print(f"[cyan]Loading 2wikimultihopqa dataset...[/cyan]")
        
        with open(self.dataset_file, 'r') as f:
            data = json.load(f)
        
        # Process questions and extract oracle context
        questions_data = []
        for i, item in enumerate(data):
            if self.num_questions and i >= self.num_questions:
                break
                
            question_id = item.get("_id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("", [])
            context = item.get("contexts", [])
            supporting_facts = item.get("supporting_facts", [])
            
            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            
            # Format oracle context based on supporting_only flag
            if self.supporting_only:
                oracle_context = self._format_supporting_context(context, supporting_facts)
                num_docs = len(supporting_facts)
            else:
                oracle_context = self._format_oracle_context(context)
                num_docs = len(context)
            
            questions_data.append({
                "question_id": question_id,
                "question": question,
                "gold_answers": gold_answers,
                "oracle_context": oracle_context,
                "num_documents": num_docs
            })
        
        console.print(f"[green]✓ Loaded {len(questions_data)} questions with oracle context[/green]")
        return questions_data

    def load_musique_dataset(self) -> List[Dict[str, Any]]:
        """Load questions and ground truth context from musique.json.

        Returns:
            List of question dictionaries with oracle context
        """
        console.print(f"[cyan]Loading Musique dataset...[/cyan]")

        with open(self.dataset_file, 'r') as f:
            data = json.load(f)

        # Process questions and extract oracle context
        questions_data = []
        for i, item in enumerate(data):
            if self.num_questions and i >= self.num_questions:
                break

            question_id = item.get("id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("reference", [])
            paragraphs = item.get("contexts", [])

            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            # Format oracle context based on supporting_only flag
            if self.supporting_only:
                # Filter to only supporting paragraphs
                supporting_paragraphs = [p for p in paragraphs if p.get("is_supporting", False)]
                oracle_context = self._format_musique_context(supporting_paragraphs)
                num_docs = len(supporting_paragraphs)
            else:
                oracle_context = self._format_musique_context(paragraphs)
                num_docs = len(paragraphs)

            questions_data.append({
                "question_id": question_id,
                "question": question,
                "gold_answers": gold_answers,
                "oracle_context": oracle_context,
                "num_documents": num_docs
            })

        console.print(f"[green]✓ Loaded {len(questions_data)} questions with oracle context[/green]")
        return questions_data

    def _format_oracle_context(self, context: List[List[Any]]) -> str:
        """Format the ground truth context documents.
        
        Args:
            context: List of [title, sentences] pairs from 2wiki dataset
            
        Returns:
            Formatted context string
        """
        formatted_docs = []
        
        for doc_data in context:
            if len(doc_data) != 2:
                continue
                
            title, sentences = doc_data
            
            # Join sentences into a single text
            if isinstance(sentences, list):
                doc_text = " ".join(sentences)
            else:
                doc_text = str(sentences)
            
            # Format with or without title based on flag
            if self.include_titles:
                formatted_doc = f"Document: {title}\n{doc_text}"
            else:
                formatted_doc = doc_text
            
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)

    def _format_musique_context(self, paragraphs: List[Dict[str, Any]]) -> str:
        """Format the ground truth context from Musique paragraphs.

        Args:
            paragraphs: List of paragraph dictionaries from Musique dataset

        Returns:
            Formatted context string
        """
        formatted_docs = []

        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("text", "")

            # Format with or without title based on flag
            if self.include_titles and title:
                formatted_doc = f"Title: {title}\n{text}"
            else:
                formatted_doc = text

            formatted_docs.append(formatted_doc)

        return "\n\n".join(formatted_docs)

    def _format_supporting_context(self, context: List[List[Any]], supporting_facts: List[List[Any]]) -> str:
        """Format only the supporting facts context.
        
        Args:
            context: List of [title, sentences] pairs from 2wiki dataset
            supporting_facts: List of [title, sentence_index] pairs indicating supporting facts
            
        Returns:
            Formatted context string with only supporting documents/sentences
        """
        # Create a mapping from title to document content
        context_map = {}
        for doc_data in context:
            if len(doc_data) != 2:
                continue
            title, sentences = doc_data
            context_map[title] = sentences
        
        formatted_docs = []
        
        for fact in supporting_facts:
            if len(fact) != 2:
                continue
                
            title, sentence_idx = fact
            
            if title not in context_map:
                continue
            
            sentences = context_map[title]
            if isinstance(sentences, list) and 0 <= sentence_idx < len(sentences):
                # Use specific sentence
                sentence_text = sentences[sentence_idx]
            elif isinstance(sentences, list):
                # If index is out of range, use all sentences
                sentence_text = " ".join(sentences)
            else:
                # If sentences is not a list, use as-is
                sentence_text = str(sentences)
            
            # Format with or without title based on flag
            if self.include_titles:
                formatted_doc = f"Document: {title}\n{sentence_text}"
            else:
                formatted_doc = sentence_text
            
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def run_oracle_evaluation(self) -> List[Dict[str, Any]]:
        """Run oracle evaluation on all questions.
        
        Returns:
            List of result dictionaries
        """
        # Load dataset
        questions_data = self.load_musique_dataset()
        
        console.print(f"\n[bold cyan]🔮 Running Oracle Evaluation on {len(questions_data)} questions[/bold cyan]")
        
        # Prepare inputs for the model
        model_inputs = []
        for question_data in questions_data:
            # Calculate token count
            encoding = tiktoken.encoding_for_model(self.model_name)
            token_count = len(encoding.encode(question_data["oracle_context"]))

            model_inputs.append({
                "question_id": question_data["question_id"],
                "question": question_data["question"],
                "oracle_context": question_data["oracle_context"],
                "gold_answers": question_data["gold_answers"],
                "token_count": token_count
            })
        
        # Run batched inference
        start_time = time.time()
        
        with Progress(SpinnerColumn(), TextColumn(f"[cyan]Processing {len(model_inputs)} questions with oracle context..."), console=console) as progress:
            task = progress.add_task("Running model...", total=None)
            
            try:
                results_dataset = self.generator(model_inputs)
                elapsed = time.time() - start_time
                
                progress.update(task, completed=100, total=100)
                
            except Exception as e:
                console.print(f"[red]✗ Oracle evaluation failed: {e}[/red]")
                sys.exit(1)
        
        console.print(f"[green]✓ Completed in {elapsed:.1f}s ({elapsed/len(model_inputs):.2f}s per question)[/green]")
        
        # Add processing time and document info to results
        results = []
        for i, result_data in enumerate(results_dataset.dataset):
            result_data["processing_time"] = elapsed / len(model_inputs)
            result_data["num_documents"] = questions_data[i]["num_documents"]
            results.append(result_data)
        
        return results
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute evaluation metrics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Tuple of (overall_metrics, per_example_metrics)
        """
        console.print(f"[cyan]Computing oracle evaluation metrics...[/cyan]")
        
        # Prepare data for evaluation
        gold_answers_list = [r["gold_answers"] for r in results]
        predicted_answers = [r["predicted_answer"] for r in results]
        
        # Compute metrics using the same evaluation as other scripts
        overall_metrics, per_example_metrics = evaluate_qa_batch(gold_answers_list, predicted_answers)
        
        return overall_metrics, per_example_metrics
    
    def display_results(self, overall_metrics: Dict[str, float], results: List[Dict[str, Any]]) -> None:
        """Display oracle evaluation results.
        
        Args:
            overall_metrics: Overall performance metrics
            results: Individual question results
        """
        console.print("\n" + "="*60)
        console.print(f"[bold green]🔮 Oracle Results for {self.model_name}[/bold green]")
        
        # Configuration info
        console.print(f"[cyan]Include document titles: {self.include_titles}[/cyan]")
        console.print(f"[cyan]Supporting facts only: {self.supporting_only}[/cyan]")
        console.print(f"[cyan]Questions tested: {len(results)}[/cyan]")
        
        # Main metrics
        console.print(f"[green]ExactMatch: {overall_metrics.get('ExactMatch', 0):.3f}[/green]")
        console.print(f"[green]F1 Score: {overall_metrics.get('F1', 0):.3f}[/green]")
        
        # Timing and document metrics
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        avg_tokens = sum(r["token_count"] for r in results) / len(results)
        avg_docs = sum(r["num_documents"] for r in results) / len(results)
        
        console.print(f"[yellow]Avg processing time: {avg_time:.2f}s[/yellow]")
        console.print(f"[magenta]Avg tokens per question: {avg_tokens:.0f}[/magenta]")
        console.print(f"[blue]Avg documents per question: {avg_docs:.1f}[/blue]")
        
        # Oracle interpretation
        console.print(f"\n[bold cyan]📈 Oracle Performance Insights:[/bold cyan]")
        console.print(f"This represents the theoretical upper bound with perfect retrieval.")
        console.print(f"Gap between oracle and actual retrieval shows retrieval system limitations.")
    
    def save_results(self, results: List[Dict[str, Any]], overall_metrics: Dict[str, float], 
                     per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save oracle evaluation results to JSON file.
        
        Args:
            results: Individual question results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        output_file = Path("logs") / f"oracle_eval_{self.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                "evaluation_type": "oracle",
                "include_titles": self.include_titles,
                "supporting_only": self.supporting_only,
                "num_questions": len(results),
                "timestamp": datetime.now().isoformat(),
                "dataset_source": str(self.dataset_file)
            },
            "overall_metrics": overall_metrics,
            "per_question_results": per_question_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓ Oracle results saved to: {output_file}[/green]")
    
    def run_complete_evaluation(self) -> None:
        """Run the complete oracle evaluation pipeline."""
        # Run oracle evaluation
        results = self.run_oracle_evaluation()
        
        # Compute metrics
        overall_metrics, per_example_metrics = self.compute_metrics(results)
        
        # Display results
        self.display_results(overall_metrics, results)
        
        # Save results
        self.save_results(results, overall_metrics, per_example_metrics)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Oracle retriever evaluation with ground truth documents")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to test (default: gpt-4o-mini)")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--include-titles", type=bool, default=True, help="Include document titles in context (default: True)")
    parser.add_argument("--supporting-only", type=bool, default=True, help="Use only supporting facts (True) or full context (False) (default: True)")
    
    args = parser.parse_args()
    
    console.print(f"[bold cyan]🔮 Oracle Retriever Evaluation[/bold cyan]")
    console.print(f"Model: {args.model}")
    console.print(f"Include titles: {args.include_titles}")
    console.print(f"Supporting facts only: {args.supporting_only}")
    if args.num_questions:
        console.print(f"Questions limit: {args.num_questions}")
    
    # Run oracle evaluation
    evaluator = OracleEvaluator(
        model_name=args.model,
        num_questions=args.num_questions,
        include_titles=args.include_titles,
        supporting_only=args.supporting_only
    )
    
    evaluator.run_complete_evaluation()
    
    console.print(f"\n[bold green]✅ Oracle evaluation completed![/bold green]")


if __name__ == "__main__":
    main()