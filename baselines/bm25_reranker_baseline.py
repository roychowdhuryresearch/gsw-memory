#!/usr/bin/env python3
"""
BM25 + Reranker Baseline for Multi-Hop QA

A simple baseline that:
1. Indexes documents from corpus using BM25
2. For each question, retrieves top-k most similar documents using BM25
3. Optionally reranks using VoyageAI reranker (can be disabled)
4. Generates answers using GPT-4o-mini with the retrieved context
5. Evaluates using the same metrics as the main system
"""

import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['OPENAI_API_KEY'] = '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'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch, format_evaluation_report

# Check for dependencies
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")
    BM25_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("Warning: tiktoken not available. Install with: pip install tiktoken")
    TIKTOKEN_AVAILABLE = False

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    print("Warning: VoyageAI not available. Install with: pip install voyageai")
    VOYAGEAI_AVAILABLE = False

try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    print("Warning: Curator not available. Install with: pip install bespokelabs-curator")
    CURATOR_AVAILABLE = False

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


if CURATOR_AVAILABLE:
    class BM25AnswerGenerator(curator.LLM):
        """Curator class for generating answers with oracle-style prompting in parallel."""

        return_completions_object = True

        def __init__(self, **kwargs):
            """Initialize the answer generator."""
            super().__init__(**kwargs)

        def prompt(self, input):
            """Create an oracle-style answer generation prompt."""
            # Build evidence in Q&A format from retrieved docs
            evidence_parts = []
            for doc in input['retrieved_docs']:
                evidence_parts.append(f"Q: What is {doc['title']}? A: {doc['text']}")

            evidence_text = '\n'.join(evidence_parts)

            # Build oracle-style prompt
            prompt_text = f"""
{evidence_text}

Question: {input['question']}

Thought:

"""

            # One-shot example with Q&A pairs format
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

            # System message
            rag_qa_system = (
                'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
                'If you don\'t know the answer, say "No Answer".'
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

            # Build message structure for OpenAI API
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
                # Remove trailing period if it's just a number/date
                if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                    final_answer = final_answer[:-1]
            else:
                # Fallback to full response if no "Answer:" found
                final_answer = answer_text.strip()

            # Compute token count over evidence text
            evidence_parts = []
            for doc in input['retrieved_docs']:
                evidence_parts.append(f"Q: What is {doc['title']}? A: {doc['text']}")
            evidence_text = '\n'.join(evidence_parts)

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
                "retrieved_docs": input['retrieved_docs'],
                "token_count": token_count
            }]


@dataclass
class BaselineEvaluationResult:
    """Container for baseline evaluation results."""
    question_id: str
    question: str
    predicted_answer: str
    full_response: str
    gold_answers: List[str]
    processing_time: float
    retrieved_docs: List[Dict[str, Any]]
    token_count: int = 0
    error: Optional[str] = None


class BM25RerankerBaseline:
    """Simple baseline using BM25 retrieval with optional reranking."""

    def __init__(self, corpus_path: str, questions_path: str, num_questions: int = 20,
                 top_k: int = 5, cache_dir: str = ".", rebuild_cache: bool = False,
                 verbose: bool = False, use_reranker: bool = False, reranker_top_k: int = 20):
        """Initialize the baseline system.

        Args:
            corpus_path: Path to corpus.json
            questions_path: Path to questions.json
            num_questions: Number of questions to evaluate
            top_k: Number of documents to retrieve per question (after reranking if enabled)
            cache_dir: Directory to store BM25 index cache
            rebuild_cache: Force rebuild of BM25 index cache
            verbose: Show detailed output
            use_reranker: If True, use VoyageAI reranker to rerank top-k documents
            reranker_top_k: Number of documents to retrieve before reranking (only used if use_reranker=True)
        """
        self.corpus_path = Path(corpus_path)
        self.questions_path = Path(questions_path)
        self.num_questions = num_questions
        self.top_k = top_k
        self.verbose = verbose
        self.cache_dir = Path(cache_dir)
        self.rebuild_cache = rebuild_cache
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k

        # Data structures
        self.documents = []
        self.bm25_index = None
        self.tokenized_corpus = []
        self.openai_client = None
        self.voyage_client = None

        # Cache files
        self.bm25_cache_file = self.cache_dir / "bm25_index_baseline.pkl"
        self.corpus_cache_file = self.cache_dir / "bm25_corpus_baseline.pkl"

        console.print("[bold blue]Initializing BM25 + Reranker Baseline...[/bold blue]")
        if use_reranker:
            console.print(f"[cyan]Reranker mode enabled: Retrieve top-{reranker_top_k}, rerank to top-{top_k}[/cyan]")
        else:
            console.print(f"[cyan]Pure BM25 mode: Retrieve top-{top_k}[/cyan]")

        # Load documents
        self._load_documents()

        # Initialize BM25
        if BM25_AVAILABLE:
            self._load_or_build_bm25_index()
        else:
            raise RuntimeError("rank-bm25 is required for this baseline")

        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI()
            console.print("[green]✓ OpenAI client initialized[/green]")
        else:
            raise RuntimeError("OpenAI client is required for answer generation")

        # Initialize VoyageAI client if reranker is enabled
        if self.use_reranker:
            if VOYAGEAI_AVAILABLE:
                self.voyage_client = voyageai.Client()
                console.print("[green]✓ VoyageAI client initialized for reranking[/green]")
            else:
                raise RuntimeError("VoyageAI is required when use_reranker=True. Install with: pip install voyageai")

        # Initialize curator answer generator for batched processing
        self.answer_generator = None
        if CURATOR_AVAILABLE:
            os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"
            self.answer_generator = BM25AnswerGenerator(
                # model_name="gpt-4o-mini",
                # generation_params={"temperature": 0},
                model_name="hosted_vllm/Qwen/Qwen3-8B",
                backend = "litellm",
                backend_params = {
                    "base_url": "http://127.0.0.1:6379/v1",
                    "request_timeout": 600.0,  
                    "max_concurrent_requests": 32,
                    "max_requests_per_minute": 120,
                    "max_tokens_per_minute": 200000,
                    "seconds_to_pause_on_rate_limit": 5,
                    "require_all_responses": False,
                },
                generation_params={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "max_tokens": 1000}
            )
            console.print("[green]✓ Curator initialized for parallel answer generation[/green]")
        else:
            console.print("[yellow]⚠ Curator not available - will use sequential processing[/yellow]")

        console.print(f"[bold green]✓ Baseline initialized with {len(self.documents)} documents[/bold green]")

    def _load_documents(self):
        """Load documents from corpus file."""
        console.print(f"[cyan]Loading documents from {self.corpus_path}...[/cyan]")

        with open(self.corpus_path, 'r') as f:
            corpus_data = json.load(f)

        # Each document has "title" and "text" fields
        for i, doc in enumerate(corpus_data):
            self.documents.append({
                "id": i,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "combined": f"{doc.get('title', '')} {doc.get('text', '')}"
            })

        console.print(f"[green]✓ Loaded {len(self.documents)} documents[/green]")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 (simple lowercase split)."""
        return text.lower().split()

    def _load_or_build_bm25_index(self):
        """Load BM25 index from cache or build it."""
        # Try to load from cache
        if not self.rebuild_cache and self.bm25_cache_file.exists() and self.corpus_cache_file.exists():
            console.print("[cyan]Loading BM25 index from cache...[/cyan]")
            try:
                with open(self.bm25_cache_file, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                with open(self.corpus_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.tokenized_corpus = cached_data['tokenized_corpus']
                    cached_doc_count = cached_data['doc_count']

                if cached_doc_count == len(self.documents):
                    console.print(f"[green]✓ Loaded BM25 index with {len(self.tokenized_corpus)} documents[/green]")
                    return
                else:
                    console.print("[yellow]Cache size mismatch, rebuilding BM25 index[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Could not load cache: {e}, rebuilding[/yellow]")

        # Build BM25 index
        self._build_bm25_index()
        self._save_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from documents."""
        console.print("[cyan]Building BM25 index from documents...[/cyan]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task_id = progress.add_task("Tokenizing documents...", total=len(self.documents))

            for doc in self.documents:
                tokenized = self._tokenize_text(doc['combined'])
                self.tokenized_corpus.append(tokenized)
                progress.update(task_id, advance=1)

        # Build BM25 index
        console.print("[cyan]Building BM25Okapi index...[/cyan]")
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        console.print(f"[green]✓ Built BM25 index with {len(self.tokenized_corpus)} documents[/green]")

    def _save_bm25_index(self):
        """Save BM25 index to cache."""
        try:
            with open(self.bm25_cache_file, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            console.print(f"[green]✓ Saved BM25 index to {self.bm25_cache_file}[/green]")

            # Save tokenized corpus separately
            cached_data = {
                'tokenized_corpus': self.tokenized_corpus,
                'doc_count': len(self.documents)
            }
            with open(self.corpus_cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            console.print(f"[green]✓ Saved corpus data to {self.corpus_cache_file}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save BM25 cache: {e}[/yellow]")

    def retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a question using BM25.

        Args:
            question: The question to retrieve documents for

        Returns:
            List of top-k documents with scores
        """
        # Tokenize the question
        tokenized_query = self._tokenize_text(question)

        # Determine how many to retrieve initially
        initial_k = self.reranker_top_k if self.use_reranker else self.top_k

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:initial_k]

        # Build initial candidate list
        candidates = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['bm25_score'] = float(scores[idx])
            candidates.append(doc)

        # If reranker is enabled, rerank the candidates
        if self.use_reranker and self.voyage_client:
            # Prepare documents for reranking
            doc_texts = [doc['combined'] for doc in candidates]

            # Call VoyageAI rerank API
            try:
                reranking = self.voyage_client.rerank(
                    question,
                    doc_texts,
                    model="rerank-2.5",
                    top_k=self.top_k  # Get top-k after reranking
                )

                # Create reranked list
                retrieved = []
                for result in reranking.results:
                    doc = candidates[result.index].copy()
                    doc['rerank_score'] = result.relevance_score
                    doc['similarity_score'] = doc['bm25_score']  # Keep both scores
                    retrieved.append(doc)

                if self.verbose:
                    console.print(f"[dim]Reranked {len(candidates)} -> {len(retrieved)} documents[/dim]")

                return retrieved

            except Exception as e:
                console.print(f"[yellow]Warning: Reranking failed: {e}. Falling back to BM25-only retrieval.[/yellow]")
                # Fall through to return candidates based on BM25 scores

        # Return documents with BM25 scores (no reranking)
        retrieved = []
        for doc in candidates[:self.top_k]:
            doc['similarity_score'] = doc['bm25_score']
            retrieved.append(doc)

        return retrieved

    def _count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens in text."""
        if not TIKTOKEN_AVAILABLE:
            return max(1, len(text) // 4)

        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def generate_answer(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, str, int]:
        """Generate an answer using GPT-4o-mini with oracle-style prompting.

        Args:
            question: The question to answer
            retrieved_docs: Retrieved documents with context

        Returns:
            Tuple of (predicted_answer, full_response, token_count)
        """
        # Build evidence in Q&A format similar to the evaluation script
        # For baseline, we convert documents to simple Q&A format
        evidence_parts = []
        for i, doc in enumerate(retrieved_docs):
            # Format: Q: What is [title] about? A: [text]
            evidence_parts.append(f"Q: What is {doc['title']}? A: {doc['text']}")

        evidence_text = '\n'.join(evidence_parts)

        # Build oracle-style prompt (same as ChainAnswerGenerator)
        prompt_text = f"""
{evidence_text}

Question: {question}

Thought:

"""

        # One-shot example with Q&A pairs format (same as evaluation script)
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

        # System message (same as evaluation script)
        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
            'If you don\'t know the answer, say "No Answer".'
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

        # Build the prompt messages
        prompt_messages = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": one_shot_output},
            {"role": "user", "content": prompt_text}
        ]

        # Count tokens (only evidence text)
        token_count = self._count_tokens(evidence_text)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages,
                temperature=0.0,
                max_tokens=1000
            )

            full_response = response.choices[0].message.content

            # Parse answer (same logic as ChainAnswerGenerator)
            if 'Answer: ' in full_response:
                final_answer = full_response.split('Answer: ')[-1].strip()
                # Remove trailing period if it's just a number/date
                if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                    final_answer = final_answer[:-1]
            else:
                # Fallback to full response if no "Answer:" found
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
            gold_answers = item.get("answer", [])
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

            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
                gold_answers.extend(item.get("answer_aliases", []))
            else:
                gold_answers = gold_answers + item.get("answer_aliases", [])

            questions_data.append((question_id, question, gold_answers))

        console.print(f"[green]✓ Loaded {len(questions_data)} questions[/green]")
        return questions_data

    def run_batched_answer_generation(self, questions_data: List[Tuple[str, str, List[str]]],
                                      retrieved_docs_by_question: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict]:
        """Run batched answer generation using curator.

        Args:
            questions_data: List of (question_id, question, gold_answers) tuples
            retrieved_docs_by_question: Dictionary mapping question_id to retrieved docs

        Returns:
            Dictionary mapping question_id to answer results
        """
        console.print("\n[bold cyan]Running Batched Answer Generation[/bold cyan]")

        # Prepare inputs for batched answer generation
        answer_inputs = [
            {
                "question_id": qid,
                "question": question,
                "retrieved_docs": retrieved_docs_by_question.get(qid, [])
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

    def run_evaluation(self) -> List[BaselineEvaluationResult]:
        """Run complete evaluation pipeline.

        Returns:
            List of evaluation results
        """
        console.print("\n[bold magenta]Running Baseline Evaluation[/bold magenta]")
        total_start = time.time()

        # Load questions
        questions_data = self.load_questions()

        # Use batched processing if curator is available
        if CURATOR_AVAILABLE and self.answer_generator is not None:
            return self._run_batched_evaluation(questions_data, total_start)
        else:
            return self._run_sequential_evaluation(questions_data, total_start)

    def _run_batched_evaluation(self, questions_data: List[Tuple[str, str, List[str]]],
                                 total_start: float) -> List[BaselineEvaluationResult]:
        """Run evaluation with batched answer generation.

        Args:
            questions_data: List of (question_id, question, gold_answers) tuples
            total_start: Start time for total elapsed calculation

        Returns:
            List of evaluation results
        """
        # Stage 1: Retrieve documents for all questions (sequential - BM25 is fast)
        console.print("\n[bold cyan]Stage 1: Document Retrieval[/bold cyan]")
        retrieved_docs_by_question = {}

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task = progress.add_task("Retrieving documents...", total=len(questions_data))

            for question_id, question, _ in questions_data:
                retrieved_docs = self.retrieve_documents(question)
                retrieved_docs_by_question[question_id] = retrieved_docs
                progress.update(task, advance=1)

        console.print(f"[green]✓ Document retrieval complete for {len(questions_data)} questions[/green]")

        # Stage 2: Batched answer generation
        answer_results = self.run_batched_answer_generation(questions_data, retrieved_docs_by_question)

        # Stage 3: Compile results
        results = []
        for question_id, question, gold_answers in questions_data:
            answer_data = answer_results.get(question_id, {})
            retrieved_docs = retrieved_docs_by_question.get(question_id, [])

            result = BaselineEvaluationResult(
                question_id=question_id,
                question=question,
                predicted_answer=answer_data.get("predicted_answer", "NA"),
                full_response=answer_data.get("full_response", ""),
                gold_answers=gold_answers,
                processing_time=0.0,  # Will be updated with average
                retrieved_docs=retrieved_docs,
                token_count=answer_data.get("token_count", 0),
                error=None
            )
            results.append(result)

        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")

        # Update processing time with average
        avg_time = total_elapsed / len(results) if results else 0
        for result in results:
            result.processing_time = avg_time

        return results

    def _run_sequential_evaluation(self, questions_data: List[Tuple[str, str, List[str]]],
                                    total_start: float) -> List[BaselineEvaluationResult]:
        """Run evaluation with sequential processing (fallback when curator not available).

        Args:
            questions_data: List of (question_id, question, gold_answers) tuples
            total_start: Start time for total elapsed calculation

        Returns:
            List of evaluation results
        """
        console.print("[yellow]Using sequential processing (curator not available)[/yellow]")

        # Process each question
        results = []

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task = progress.add_task("Processing questions...", total=len(questions_data))

            for question_id, question, gold_answers in questions_data:
                start_time = time.time()

                # Retrieve documents
                retrieved_docs = self.retrieve_documents(question)

                # Generate answer
                predicted_answer, full_response, token_count = self.generate_answer(
                    question, retrieved_docs
                )

                processing_time = time.time() - start_time

                result = BaselineEvaluationResult(
                    question_id=question_id,
                    question=question,
                    predicted_answer=predicted_answer,
                    full_response=full_response,
                    gold_answers=gold_answers,
                    processing_time=processing_time,
                    retrieved_docs=retrieved_docs,
                    token_count=token_count,
                    error=None
                )

                results.append(result)
                progress.update(task, advance=1)

                if self.verbose and len(results) % 10 == 0:
                    console.print(f"[dim]Processed {len(results)} questions...[/dim]")

        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")

        return results

    def compute_metrics(self, results: List[BaselineEvaluationResult]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
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

        # Compute metrics
        overall_metrics, per_example_metrics = evaluate_qa_batch(
            gold_answers_list,
            predicted_answers
        )

        return overall_metrics, per_example_metrics

    def save_results(self, results: List[BaselineEvaluationResult],
                    overall_metrics: Dict[str, float],
                    per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file.

        Args:
            results: List of evaluation results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        reranker_suffix = "_with_reranker" if self.use_reranker else ""
        output_file = LOG_DIR / f"bm25_baseline_results{reranker_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            "evaluation_info": {
                "baseline": "bm25_okapi" + ("_with_reranker" if self.use_reranker else ""),
                "num_questions": self.num_questions,
                "top_k": self.top_k,
                "use_reranker": self.use_reranker,
                "reranker_top_k": self.reranker_top_k if self.use_reranker else None,
                "timestamp": datetime.now().isoformat()
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
                "retrieved_docs": [
                    {
                        "title": doc["title"],
                        "text": doc["text"][:200] + "...",  # Truncate for readability
                        "bm25_score": doc.get("bm25_score", doc.get("similarity_score", 0.0)),
                        "rerank_score": doc.get("rerank_score", None)
                    }
                    for doc in result.retrieved_docs
                ],
                "token_count": result.token_count,
                "error": result.error
            }
            output_data["per_question_results"].append(question_data)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        console.print(f"[green]✓ Results saved to: {output_file}[/green]")


def main(verbose: bool = False):
    """Main evaluation function."""
    console.print("\n[bold cyan]=== BM25 + Reranker Baseline Evaluation ===[/bold cyan]")

    try:
        # Initialize baseline
        baseline = BM25RerankerBaseline(
            corpus_path="/home/yigit/codebase/gsw-memory/playground_data/musique_corpus.json",
            questions_path="/home/yigit/codebase/gsw-memory/playground_data/musique_platinum.json",
            num_questions=1000,
            top_k=5,
            cache_dir=".",
            rebuild_cache=True,
            verbose=verbose,
            use_reranker=False,  # Set to True to enable reranking
            reranker_top_k=20    # Number of docs to retrieve before reranking (when use_reranker=True)
        )

        # Run evaluation
        results = baseline.run_evaluation()

        # Compute metrics
        overall_metrics, per_example_metrics = baseline.compute_metrics(results)

        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]Evaluation Results:[/bold green]")
        console.print(format_evaluation_report(overall_metrics, per_example_metrics, show_examples=5))

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
        baseline.save_results(results, overall_metrics, per_example_metrics)

        console.print("\n[bold green]✓ Baseline evaluation completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # Set verbose=True for detailed output during development
    main(verbose=False)
