#!/usr/bin/env python3
"""
Qwen-3 Embedding Baseline for Multi-Hop QA

A simple baseline that:
1. Embeds documents from musique_corpus.json using Qwen3-Embedding-8B
2. For each question, retrieves top-5 most similar documents
3. Generates answers using GPT-4o-mini with the retrieved context
4. Evaluates using the same metrics as the main system
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch, format_evaluation_report

# Check for dependencies
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Install with: pip install vllm>=0.8.5")
    VLLM_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    SIMILARITY_AVAILABLE = False

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
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
    FAISS_AVAILABLE = False

console = Console()

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


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


class Qwen3EmbeddingBaseline:
    """Simple baseline using Qwen-3 embeddings and top-k retrieval."""

    def __init__(self, corpus_path: str, questions_path: str, num_questions: int = 20,
                 top_k: int = 5, cache_dir: str = ".", rebuild_cache: bool = False,
                 verbose: bool = False, use_reranker: bool = False, reranker_top_k: int = 20,
                 gpu_device: int = 3):
        """Initialize the baseline system.

        Args:
            corpus_path: Path to musique_corpus.json
            questions_path: Path to musique.json
            num_questions: Number of questions to evaluate
            top_k: Number of documents to retrieve per question (after reranking if enabled)
            cache_dir: Directory to store embedding caches
            rebuild_cache: Force rebuild of embedding cache
            verbose: Show detailed output
            use_reranker: If True, use VoyageAI reranker to rerank top-k documents
            reranker_top_k: Number of documents to retrieve before reranking (only used if use_reranker=True)
            gpu_device: GPU device ID for FAISS GPU acceleration (default: 3)
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
        self.gpu_device = gpu_device

        # Data structures
        self.documents = []
        self.doc_embeddings = None
        self.embedding_model = None
        self.openai_client = None
        self.voyage_client = None
        self.doc_faiss_index = None  # FAISS GPU index for document embeddings

        # Initialize GPU resources for FAISS
        self.gpu_resources = None
        if FAISS_AVAILABLE:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not initialize GPU resources: {e}[/yellow]")

        # Cache files (GPU format)
        self.doc_embedding_cache_file = self.cache_dir / "doc_embeddings_baseline_gpu.faiss"
        self.doc_metadata_cache_file = self.cache_dir / "doc_metadata_baseline.json"
        # Legacy HNSW cache path for backward compatibility
        self.doc_embedding_hnsw_cache = self.cache_dir / "doc_embeddings_baseline_hnsw.faiss"

        console.print("[bold blue]Initializing Qwen-3 Embedding Baseline...[/bold blue]")
        if use_reranker:
            console.print(f"[cyan]Reranker mode enabled: Retrieve top-{reranker_top_k}, rerank to top-{top_k}[/cyan]")

        # Load documents
        self._load_documents()

        # Initialize models
        if VLLM_AVAILABLE:
            self._initialize_embedding_model()
            self._load_or_generate_embeddings()
        else:
            raise RuntimeError("VLLM is required for this baseline")

        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI()
            console.print("[green] OpenAI client initialized[/green]")
        else:
            raise RuntimeError("OpenAI client is required for answer generation")

        # Initialize VoyageAI client if reranker is enabled
        if self.use_reranker:
            if VOYAGEAI_AVAILABLE:
                self.voyage_client = voyageai.Client()
                console.print("[green]✓ VoyageAI client initialized for reranking[/green]")
            else:
                raise RuntimeError("VoyageAI is required when use_reranker=True. Install with: pip install voyageai")

        console.print(f"[bold green] Baseline initialized with {len(self.documents)} documents[/bold green]")

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

        console.print(f"[green] Loaded {len(self.documents)} documents[/green]")

    def _initialize_embedding_model(self):
        """Initialize the Qwen embedding model."""
        console.print("[cyan]Initializing Qwen3-Embedding-8B model...[/cyan]")
        try:
            self.embedding_model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
            console.print("[green] Qwen embedding model initialized[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing embedding model: {e}[/red]")
            raise

    def _load_or_generate_embeddings(self):
        """Load embeddings from cache or generate them."""
        # Try to load from FAISS GPU cache first
        if not self.rebuild_cache and FAISS_AVAILABLE and self.gpu_resources is not None and self.doc_embedding_cache_file.exists() and self.doc_metadata_cache_file.exists():
            console.print("[cyan]Loading embeddings from FAISS GPU cache...[/cyan]")
            try:
                # Load metadata
                with open(self.doc_metadata_cache_file, 'r') as f:
                    metadata = json.load(f)
                cached_doc_count = metadata.get('doc_count', 0)

                if cached_doc_count == len(self.documents):
                    # Load CPU index from disk
                    cpu_index = faiss.read_index(str(self.doc_embedding_cache_file))

                    # Transfer to GPU
                    self.doc_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

                    # Reconstruct embeddings from CPU index for compatibility
                    num_vectors = cpu_index.ntotal
                    self.doc_embeddings = faiss.vector_to_array(cpu_index.reconstruct_n(0, num_vectors))
                    self.doc_embeddings = self.doc_embeddings.reshape(num_vectors, -1)

                    console.print(f"[green]✓ Loaded FAISS GPU index with {num_vectors} document embeddings[/green]")
                    return
                else:
                    console.print("[yellow]Cache size mismatch, regenerating embeddings[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Could not load GPU cache: {e}, trying HNSW fallback[/yellow]")

        # Fallback: Try to load from legacy HNSW cache and convert to GPU
        if not self.rebuild_cache and FAISS_AVAILABLE and self.gpu_resources is not None and self.doc_embedding_hnsw_cache.exists():
            console.print("[cyan]Loading embeddings from legacy HNSW cache...[/cyan]")
            try:
                # Load HNSW index
                hnsw_index = faiss.read_index(str(self.doc_embedding_hnsw_cache))

                # Reconstruct embeddings from HNSW index
                num_vectors = hnsw_index.ntotal
                self.doc_embeddings = faiss.vector_to_array(hnsw_index.reconstruct_n(0, num_vectors))
                self.doc_embeddings = self.doc_embeddings.reshape(num_vectors, -1)

                # Build GPU flat index from embeddings
                embedding_dim = self.doc_embeddings.shape[1]
                cpu_index = faiss.IndexFlatIP(embedding_dim)
                embeddings_normalized = self.doc_embeddings.copy()
                faiss.normalize_L2(embeddings_normalized)
                cpu_index.add(embeddings_normalized)

                # Transfer to GPU
                self.doc_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

                console.print(f"[green]✓ Converted HNSW cache to GPU index with {num_vectors} embeddings[/green]")
                return
            except Exception as e:
                console.print(f"[yellow]Could not load HNSW cache: {e}, trying npz fallback[/yellow]")

        # Fallback: Try to load from NPZ cache and convert to GPU
        npz_cache_file = self.cache_dir / "doc_embeddings_baseline.npz"
        if not self.rebuild_cache and npz_cache_file.exists():
            console.print("[cyan]Loading embeddings from npz cache...[/cyan]")
            try:
                data = np.load(npz_cache_file, allow_pickle=True)
                cached_doc_count = data['doc_count']

                if cached_doc_count == len(self.documents):
                    self.doc_embeddings = data['embeddings'].astype(np.float32)

                    # Build FAISS GPU index from loaded embeddings if FAISS is available
                    if FAISS_AVAILABLE and self.gpu_resources is not None:
                        embedding_dim = self.doc_embeddings.shape[1]
                        cpu_index = faiss.IndexFlatIP(embedding_dim)
                        embeddings_normalized = self.doc_embeddings.copy()
                        faiss.normalize_L2(embeddings_normalized)
                        cpu_index.add(embeddings_normalized)

                        # Transfer to GPU
                        self.doc_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

                        console.print(f"[green]✓ Loaded npz cache and built FAISS GPU index with {len(self.doc_embeddings)} embeddings[/green]")
                    else:
                        console.print(f"[green]✓ Loaded {len(self.doc_embeddings)} document embeddings from npz cache[/green]")
                    return
                else:
                    console.print("[yellow]Cache size mismatch, regenerating embeddings[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Could not load npz cache: {e}[/yellow]")

        # Generate embeddings
        self._generate_embeddings()
        self._save_embeddings()

    def _generate_embeddings(self):
        """Generate embeddings for all documents."""
        console.print("[cyan]Generating embeddings for documents...[/cyan]")

        # Prepare document texts with instructions
        task = 'Given a document with title and text, create an embedding that captures the semantic meaning for retrieval.'

        doc_texts = []
        for doc in self.documents:
            instructed_text = get_detailed_instruct(task, doc['combined'])
            doc_texts.append(instructed_text)

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task_id = progress.add_task("Embedding documents...", total=len(doc_texts))

            for i in range(0, len(doc_texts), batch_size):
                batch = doc_texts[i:i+batch_size]
                outputs = self.embedding_model.embed(batch)
                batch_embeddings = [o.outputs.embedding for o in outputs]
                all_embeddings.extend(batch_embeddings)
                progress.update(task_id, advance=len(batch))

        self.doc_embeddings = np.array(all_embeddings, dtype=np.float32)
        console.print(f"[green]✓ Generated {len(self.doc_embeddings)} embeddings with shape {self.doc_embeddings.shape}[/green]")

        # Build FAISS GPU index
        if FAISS_AVAILABLE and self.gpu_resources is not None:
            embedding_dim = self.doc_embeddings.shape[1]
            # Create CPU flat index for exact nearest neighbor search
            cpu_index = faiss.IndexFlatIP(embedding_dim)
            # Normalize embeddings for cosine similarity
            embeddings_normalized = self.doc_embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            cpu_index.add(embeddings_normalized)

            # Transfer to GPU
            self.doc_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

            console.print(f"[green]✓ Built FAISS GPU index with {self.doc_faiss_index.ntotal} vectors[/green]")


    def _save_embeddings(self):
        """Save embeddings to cache using FAISS."""
        try:
            # Use FAISS if available
            if FAISS_AVAILABLE and self.doc_faiss_index is not None:
                # Transfer from GPU to CPU for saving
                if self.gpu_resources is not None:
                    cpu_index = faiss.index_gpu_to_cpu(self.doc_faiss_index)
                else:
                    cpu_index = self.doc_faiss_index

                # Save CPU FAISS index
                faiss.write_index(cpu_index, str(self.doc_embedding_cache_file))
                console.print(f"[green]✓ Saved FAISS index to {self.doc_embedding_cache_file}[/green]")

                # Save metadata separately
                metadata = {
                    "doc_count": len(self.documents),
                    "embedding_dim": self.doc_embeddings.shape[1]
                }
                with open(self.doc_metadata_cache_file, 'w') as f:
                    json.dump(metadata, f)
                console.print(f"[green]✓ Saved metadata to {self.doc_metadata_cache_file}[/green]")
            else:
                # Fallback to npz if FAISS not available
                npz_file = self.cache_dir / "doc_embeddings_baseline.npz"
                np.savez_compressed(
                    npz_file,
                    embeddings=self.doc_embeddings,
                    doc_count=len(self.documents)
                )
                console.print(f"[green]✓ Saved embeddings to {npz_file} (fallback)[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save embeddings: {e}[/yellow]")


    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query using the Qwen model."""
        task = 'Given a question, create an embedding to retrieve relevant documents.'
        instructed_query = get_detailed_instruct(task, query)

        try:
            outputs = self.embedding_model.embed([instructed_query])
            embedding = np.array(outputs[0].outputs.embedding)
            return embedding
        except Exception as e:
            console.print(f"[red]Error embedding query: {e}[/red]")
            raise

    def retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a question using FAISS GPU search.

        Args:
            question: The question to retrieve documents for

        Returns:
            List of top-k documents with scores
        """
        # Embed the question
        query_embedding = self._embed_query(question)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Determine how many to retrieve initially
        initial_k = self.reranker_top_k if self.use_reranker else self.top_k

        # Use FAISS search if available, otherwise fall back to cosine_similarity
        if FAISS_AVAILABLE and self.doc_faiss_index is not None:
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
            # Search FAISS GPU index - returns (similarities, indices)
            similarities, indices = self.doc_faiss_index.search(query_embedding, initial_k)

            # Build initial candidate list from FAISS results
            candidates = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx].copy()
                    doc['embedding_similarity'] = float(similarity)
                    candidates.append(doc)
        else:
            # Fallback to cosine_similarity if FAISS not available
            if not SIMILARITY_AVAILABLE:
                raise RuntimeError("Neither FAISS nor scikit-learn available for similarity search")

            similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:initial_k]

            candidates = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc['embedding_similarity'] = float(similarities[idx])
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
                    doc['similarity_score'] = doc['embedding_similarity']  # Keep both scores
                    retrieved.append(doc)

                if self.verbose:
                    console.print(f"[dim]Reranked {len(candidates)} -> {len(retrieved)} documents[/dim]")

                return retrieved

            except Exception as e:
                console.print(f"[yellow]Warning: Reranking failed: {e}. Falling back to embedding-only retrieval.[/yellow]")
                # Fall through to return candidates based on embedding similarity

        # Return documents with embedding similarity scores (no reranking)
        retrieved = []
        for doc in candidates[:self.top_k]:
            doc['similarity_score'] = doc['embedding_similarity']
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
        for item in data[:self.num_questions]:
            question_id = item.get("_id", "unknown")
            question = item["question"]
            gold_answers = item.get("answer", [])

            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
                gold_answers.extend(item.get("answer_aliases", []))
            else:
                gold_answers = gold_answers + item.get("answer_aliases", [])

            questions_data.append((question_id, question, gold_answers))

        console.print(f"[green] Loaded {len(questions_data)} questions[/green]")
        return questions_data

    def run_evaluation(self) -> List[BaselineEvaluationResult]:
        """Run complete evaluation pipeline.

        Returns:
            List of evaluation results
        """
        console.print("\n[bold magenta]Running Baseline Evaluation[/bold magenta]")
        total_start = time.time()

        # Load questions
        questions_data = self.load_questions()

        # Process each question
        results = []

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console) as progress:
            task = progress.add_task("Processing questions...", total=len(questions_data))

            for question_id, question, gold_answers in questions_data:
                start_time = time.time()

                try:
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

                except Exception as e:
                    console.print(f"[red]Error processing question {question_id}: {e}[/red]")
                    result = BaselineEvaluationResult(
                        question_id=question_id,
                        question=question,
                        predicted_answer="NA",
                        full_response="",
                        gold_answers=gold_answers,
                        processing_time=0.0,
                        retrieved_docs=[],
                        token_count=0,
                        error=str(e)
                    )

                results.append(result)
                progress.update(task, advance=1)

                if self.verbose and len(results) % 10 == 0:
                    console.print(f"[dim]Processed {len(results)} questions...[/dim]")

        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green] Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")

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
        output_file = LOG_DIR / f"qwen3_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            "evaluation_info": {
                "baseline": "qwen3_embedding_top_k",
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
                        "embedding_similarity": doc.get("embedding_similarity", doc.get("similarity_score", 0.0)),
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

        console.print(f"[green] Results saved to: {output_file}[/green]")


def main(verbose: bool = False):
    """Main evaluation function."""
    console.print("\n[bold cyan]=� Qwen-3 Embedding Baseline Evaluation[/bold cyan]")

    try:
        # Initialize baseline
        baseline = Qwen3EmbeddingBaseline(
            corpus_path="/home/yigit/codebase/gsw-memory/playground_data/2wikimultihopqa_corpus.json",
            questions_path="/home/yigit/codebase/gsw-memory/playground_data/2wikimultihopqa.json",
            num_questions=1000,
            top_k=5,
            cache_dir=".",
            rebuild_cache=False,
            verbose=verbose,
            use_reranker=True,  # Set to True to enable reranking
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

        console.print("\n[bold green] Baseline evaluation completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # Set verbose=True for detailed output during development
    main(verbose=False)
