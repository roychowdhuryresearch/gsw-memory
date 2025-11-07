#!/usr/bin/env python3
"""
Simple Entity Search Script

Extracts entities (with roles/states) from GSW files and performs semantic search.
Returns top-k entities based on query similarity using Qwen-3 embeddings.
"""

import json
import glob
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys
import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
import re

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Similarity computation imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    SIMILARITY_AVAILABLE = False

# FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
    FAISS_AVAILABLE = False

# VLLM for embeddings
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Install with: pip install vllm>=0.8.5")
    VLLM_AVAILABLE = False

# BM25 for entity search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("Warning: BM25 not available. Install with: pip install rank-bm25")
    BM25_AVAILABLE = False

# OpenAI for answer generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("Warning: tiktoken not available. Install with: pip install tiktoken")
    TIKTOKEN_AVAILABLE = False

from src.gsw_memory.memory.models import GSWStructure

console = Console()


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'

def load_gsw_files(num_documents: int = 50, path_to_gsw_files: str = None) -> Tuple[List[GSWStructure], List[str]]:
    """Load GSW structures from JSON files."""
    print(f"Loading first {num_documents} GSW files...")
    
    # base_dir = "/mnt/SSD1/shreyas/SM_GSW/2wiki/networks"
    if not path_to_gsw_files:
        base_dir = "/mnt/SSD1/shreyas/SM_GSW/musique/normalized_networks"
    else:
        base_dir = path_to_gsw_files
    
    if num_documents == -1:
        doc_dirs = sorted(glob.glob(os.path.join(base_dir, "doc_*")), 
                          key=lambda x: int(Path(x).name.replace("doc_", "")))
    else:
        doc_dirs = sorted(glob.glob(os.path.join(base_dir, "doc_*")), 
                          key=lambda x: int(Path(x).name.replace("doc_", "")))[:num_documents]
    
    gsw_structures = []
    doc_ids = []
    
    for doc_dir in doc_dirs:
        gsw_files = glob.glob(os.path.join(doc_dir, "gsw_*.json"))
        for file_path in gsw_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    gsw_structures.append(GSWStructure(**data))
                    doc_ids.append(Path(file_path).parent.name)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    print(f"Loaded {len(gsw_structures)} GSW structures from {len(doc_dirs)} documents")
    return gsw_structures, doc_ids


class EntitySearcher:
    """Simple entity searcher that extracts entities from GSW files and performs semantic search."""
    
    def __init__(self, num_documents: int = 50, cache_dir: str = None, path_to_gsw_files: str = None, rebuild_cache: bool = False, verbose: bool = True, use_bm25: bool = False, gpu_device: int = 2, use_gpu_for_qa_index: bool = True):
        """Initialize the entity searcher.

        Args:
            num_documents: Number of documents to load from GSW corpus
            path_to_gsw_files: Path to the GSW files
            cache_dir: Directory to store/load embedding caches (default: current dir)
            rebuild_cache: If True, force regenerate all embeddings even if cache exists
            verbose: If True, show initialization messages
            use_bm25: If True, use BM25 for entity search instead of embeddings
            gpu_device: GPU device ID to use (0-3 for RTX A6000s, default: 0)
            use_gpu_for_qa_index: If True, transfer Q&A FAISS index to GPU (requires significant GPU memory, default: False)
        """
        self.verbose_init = verbose
        self.gpu_device = gpu_device
        self.use_gpu_for_qa_index = use_gpu_for_qa_index

        # Initialize GPU resources for FAISS
        if FAISS_AVAILABLE:
            self.gpu_resources = faiss.StandardGpuResources()
            if self.verbose_init:
                console.print(f"[cyan]Initialized FAISS GPU resources on device {gpu_device}[/cyan]")
                if not use_gpu_for_qa_index:
                    console.print(f"[cyan]Q&A embeddings will use CPU (set use_gpu_for_qa_index=True to use GPU)[/cyan]")
        else:
            self.gpu_resources = None
        self.entities = []
        self.entity_texts = []
        self.embeddings = None
        self.embedding_model = None
        self.gsw_by_doc_id = {}  # Store GSW structures by doc_id for QA lookup
        self.openai_client = None  # For answer generation
        self.show_llm_prompt = True  # Toggle for debugging LLM prompts
        self.path_to_gsw_files = path_to_gsw_files # Path to the GSW files
        # Embedding cache attributes
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".")
        self.rebuild_cache = rebuild_cache
        self.qa_embedding_cache = {}  # In-memory cache for Q&A embeddings: {qa_text_hash: embedding}
        self.qa_metadata_cache = {}  # Maps qa_text_hash to Q&A metadata for direct search
        self.qa_embeddings_matrix = None  # Numpy matrix of all Q&A embeddings for fast similarity search
        self.qa_hash_to_idx = {}  # Maps qa_text_hash to index in embeddings matrix
        # FAISS GPU indexes for vector search
        self.entity_faiss_index = None  # FAISS GPU index for entity embeddings
        self.qa_faiss_index = None  # FAISS GPU index for Q&A embeddings
        self.entity_embedding_cache_file = self.cache_dir / f"entity_embeddings_{num_documents}docs_gpu.faiss"
        self.qa_embedding_cache_file = self.cache_dir / f"qa_embeddings_{num_documents}docs_gpu.faiss"
        self.cache_metadata_file = self.cache_dir / f"embedding_metadata_{num_documents}docs.json"
        self.qa_metadata_file = self.cache_dir / f"qa_metadata_{num_documents}docs.json"
        # Legacy HNSW cache path for backward compatibility
        self.entity_embedding_hnsw_cache = self.cache_dir / f"entity_embeddings_{num_documents}docs_hnsw.faiss"
        self.qa_embedding_hnsw_cache = self.cache_dir / f"qa_embeddings_{num_documents}docs_hnsw.faiss"
        self.cache_hits = 0
        self.cache_misses = 0
        self.use_bm25 = use_bm25

        
        if self.verbose_init:
            console.print("[bold blue]Loading GSW entities...[/bold blue]")
        gsw_structures, doc_ids = load_gsw_files(num_documents, self.path_to_gsw_files)
        
        # Store GSW structures by doc_id for later QA lookup
        for gsw, doc_id in zip(gsw_structures, doc_ids):
            self.gsw_by_doc_id[doc_id] = gsw
        
        self._extract_entities_from_gsw(gsw_structures, doc_ids)
        
        if VLLM_AVAILABLE:
            self._initialize_embedding_model()

            # BM25 index is only for the entity search
            if self.embedding_model:
                # Try to load cached embeddings first

                if self.use_bm25:
                    self._build_bm25_index()

                if not self.rebuild_cache and self._load_entity_embeddings_cache():
                    if self.verbose_init:
                        console.print("[green]✓ Loaded entity embeddings from cache[/green]")
                else:
                    self._generate_embeddings()
                    self._save_entity_embeddings_cache()
                
                # Load Q&A embedding cache if it exists, or precompute all Q&A embeddings
                if not self.rebuild_cache and self._load_qa_embeddings_cache():
                    if self.verbose_init:
                        console.print(f"[green]✓ Loaded {len(self.qa_embedding_cache)} Q&A embeddings from cache[/green]")
                    # Build embeddings matrix for loaded cache
                    self._build_qa_embeddings_matrix()

                # Precompute any missing Q&A embeddings
                self._precompute_qa_embeddings()

            else:
                raise ValueError("No embedding model or BM25 index found")
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                if self.verbose_init:
                    console.print("[green]✓ OpenAI client initialized for answer generation[/green]")
            except Exception as e:
                if self.verbose_init:
                    console.print(f"[yellow]Warning: Could not initialize OpenAI client: {e}[/yellow]")
                self.openai_client = None
        
        if self.verbose_init:
            console.print(f"[bold green]✓ Loaded {len(self.entities)} entities[/bold green]")
            console.print(f"[bold green]✓ Embeddings: {'Available' if self.embeddings is not None else 'Text search only'}[/bold green]")
    
    def _extract_entities_from_gsw(self, gsw_structures: List[GSWStructure], doc_ids: List[str]):
        """Extract entities from GSW structures."""
        if self.verbose_init:
            console.print("Extracting entity data...")
        
        entity_count = 0
        
        for gsw, doc_id in zip(gsw_structures, doc_ids):
            if not gsw.entity_nodes:
                continue
                
            for entity in gsw.entity_nodes:
                # Extract role descriptions with states
                role_descriptions = []
                for role in entity.roles:
                    if role.states:
                        # Include both role and states
                        states_text = ', '.join(role.states)
                        role_desc = f"{role.role}: {states_text}"
                    else:
                        # Just the role if no states
                        role_desc = role.role
                    role_descriptions.append(role_desc)
                
                # Create entity description
                if role_descriptions:
                    roles_text = ' | '.join(role_descriptions)
                    search_text = f"{entity.name} - Roles: {roles_text}"
                else:
                    search_text = f"{entity.name} - No specific roles"
                
                entity_info = {
                    "id": entity.id,
                    "name": entity.name,
                    "roles": [{
                        "role": role.role,
                        "states": role.states
                    } for role in entity.roles],
                    "all_states": [state for role in entity.roles for state in role.states],
                    "search_text": search_text,
                    "doc_id": doc_id,
                    "chunk_id": "0",  # GSW structures don't have chunk_id
                    "summary": "",  # No summary in GSW structures
                    "role_descriptions": role_descriptions,
                    "qa_pairs": []  # Will be populated during search
                }
                
                self.entities.append(entity_info)
                self.entity_texts.append(search_text)
                entity_count += 1
        
        if self.verbose_init:
            console.print(f"Extracted {entity_count} entities from {len(set(doc_ids))} documents")
    
    def _resolve_entity_ids_to_names(self, entity_ids: List[str], doc_id: str) -> List[str]:
        """Resolve entity IDs to their names using the GSW structure.
        
        Args:
            entity_ids: List of entity IDs to resolve
            doc_id: Document ID to look up the GSW structure
            
        Returns:
            List of entity names corresponding to the IDs
        """
        if doc_id not in self.gsw_by_doc_id:
            return entity_ids  # Return IDs if we can't resolve
        
        gsw = self.gsw_by_doc_id[doc_id]
        
        # Create ID to name mapping
        id_to_name = {}
        id_to_rolestate = {}
        for entity_node in gsw.entity_nodes:
            id_to_name[entity_node.id] = entity_node.name
            # create a rolestate string for each entity
            id_to_rolestate[entity_node.id] = ' | '.join([f"{role.role}: {', '.join(role.states)}" for role in entity_node.roles])
        
        # Resolve each ID
        resolved_names = []
        resolved_rolestates = []
        for entity_id in entity_ids:
            resolved_names.append(id_to_name.get(entity_id, entity_id))
            resolved_rolestates.append(id_to_rolestate.get(entity_id, ''))
        return resolved_names, resolved_rolestates
    
    def _find_qa_pairs_for_entity(self, entity_id: str, doc_id: str) -> List[Dict[str, Any]]:
        """Find QA pairs where this entity ID appears as an answer, 
        plus all other questions from the same verb phrase."""
        qa_pairs = []
        
        if doc_id not in self.gsw_by_doc_id:
            return qa_pairs
        
        gsw = self.gsw_by_doc_id[doc_id]
        
        # Look through verb phrase nodes for QA pairs
        for verb_node in gsw.verb_phrase_nodes:
            # Check if this verb node has questions
            if hasattr(verb_node, 'questions') and verb_node.questions:
                # First check if our entity appears in any answer in this verb phrase
                entity_found_in_verb = False
                for question in verb_node.questions:
                    if hasattr(question, 'answers') and question.answers:
                        if entity_id in question.answers:
                            entity_found_in_verb = True
                            break
                
                # If entity was found in this verb phrase, add ALL questions from it
                if entity_found_in_verb:
                    for question in verb_node.questions:
                        if hasattr(question, 'answers') and question.answers:
                            # Resolve answer IDs to names
                            answer_names, answer_rolestates = self._resolve_entity_ids_to_names(question.answers, doc_id)
                            
                            qa_pairs.append({
                                "question": question.text if hasattr(question, 'text') else "Unknown question",
                                "answer_ids": question.answers,
                                "answer_names": answer_names,
                                "answer_rolestates": answer_rolestates,
                                "question_id": question.id if hasattr(question, 'id') else None,
                                "verb_phrase": verb_node.phrase if hasattr(verb_node, 'phrase') else "Unknown verb phrase"
                            })
        
        return qa_pairs
    
    
    def _initialize_embedding_model(self):
        """Initialize the Qwen embedding model."""
        try:
            if self.verbose_init:
                console.print("[cyan]Initializing Qwen3-Embedding-8B model...[/cyan]")
            self.embedding_model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
            if self.verbose_init:
                console.print("[green]✓ Qwen embedding model initialized[/green]")
        except Exception as e:
            if self.verbose_init:
                console.print(f"[red]Error initializing embedding model: {e}[/red]")
            self.embedding_model = None

    def _build_bm25_index(self) -> None:
        """Build BM25 index for entity search."""
        if not self.entity_texts:
            return
        
        if self.verbose_init:
            console.print("Building BM25 index for entity search...")

        token_pattern = re.compile(r"\w+")
        self.tokenized_entities = [
            token_pattern.findall(text.lower()) for text in self.entity_texts
        ]
        self.bm25 = BM25Okapi(self.tokenized_entities)

    
    def _generate_embeddings(self):
        """Generate embeddings for all entity texts."""
        if not self.entity_texts or not self.embedding_model:
            return
        
        if self.verbose_init:
            console.print("Generating embeddings for entity search...")
        
        try:
            # Use task description similar to hypernode clustering
            task = 'Given an entity name and its contextual roles/states, create an embedding that captures the entity\'s identity for semantic search and retrieval. Use the roles and states to determine similarity.'
            
            # Prepare input texts with instructions
            input_texts = []
            for entity_text in self.entity_texts:
                instructed_query = get_detailed_instruct(task, entity_text)
                input_texts.append(instructed_query)
            
            # Generate embeddings in batches
            batch_size = 50
            all_embeddings = []
            
            for batch_start in range(0, len(input_texts), batch_size):
                batch_texts = input_texts[batch_start:batch_start+batch_size]
                batch_num = batch_start//batch_size + 1
                total_batches = (len(input_texts) + batch_size - 1)//batch_size
                if self.verbose_init:
                    console.print(f"Processing batch {batch_num}/{total_batches}")
                
                outputs = self.embedding_model.embed(batch_texts)
                batch_embeddings = [o.outputs.embedding for o in outputs]
                all_embeddings.extend(batch_embeddings)
            
            self.embeddings = np.array(all_embeddings, dtype=np.float32)
            if self.verbose_init:
                console.print(f"Generated embeddings with shape: {self.embeddings.shape}")

            # Build FAISS GPU index
            if FAISS_AVAILABLE and self.gpu_resources is not None:
                embedding_dim = self.embeddings.shape[1]
                # Create CPU index first
                cpu_index = faiss.IndexFlatIP(embedding_dim)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)
                cpu_index.add(self.embeddings)
                # Transfer to GPU
                self.entity_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)
                if self.verbose_init:
                    console.print(f"Built FAISS GPU index with {self.entity_faiss_index.ntotal} vectors on GPU {self.gpu_device}")

        except Exception as e:
            if self.verbose_init:
                console.print(f"[red]Error generating embeddings: {e}[/red]")
            self.embeddings = None
            self.entity_faiss_index = None
    
    def _precompute_qa_embeddings(self):
        """Precompute embeddings for all Q&A pairs in the GSW structures."""
        if not self.embedding_model:
            return

        if self.verbose_init:
            console.print("Precomputing Q&A pair embeddings...")

        # Collect all unique Q&A pairs from all GSW structures
        all_qa_data = {}  # qa_text -> metadata dict
        qa_count = 0

        for doc_id, gsw in self.gsw_by_doc_id.items():
            # Look through verb phrase nodes for QA pairs
            for verb_node in gsw.verb_phrase_nodes:
                if hasattr(verb_node, 'questions') and verb_node.questions:
                    for question in verb_node.questions:
                        if hasattr(question, 'answers') and question.answers:
                            # Resolve answer IDs to names for embedding
                            answer_names, answer_rolestates = self._resolve_entity_ids_to_names(question.answers, doc_id)
                            answers_text = ', '.join(answer_names)

                            # Create Q&A text representation (same format as in _rerank_qa_pairs)
                            qa_text = f"{question.text} {answers_text}"

                            # Store metadata for this Q&A pair
                            if qa_text not in all_qa_data:
                                all_qa_data[qa_text] = {
                                    'question': question.text if hasattr(question, 'text') else "Unknown question",
                                    'answer_ids': question.answers,
                                    'answer_names': answer_names,
                                    'answer_rolestates': answer_rolestates,
                                    'doc_id': doc_id,
                                    'verb_phrase': verb_node.phrase if hasattr(verb_node, 'phrase') else "Unknown verb phrase"
                                }
                            qa_count += 1

        if self.verbose_init:
            console.print(f"Found {len(all_qa_data)} unique Q&A pairs from {qa_count} total pairs")

        if not all_qa_data:
            return

        # Store metadata for all Q&A pairs
        for qa_text, metadata in all_qa_data.items():
            qa_hash = self._get_qa_text_hash(qa_text)
            self.qa_metadata_cache[qa_hash] = metadata

        # Check how many are already cached
        uncached_texts = []
        for qa_text in all_qa_data.keys():
            qa_hash = self._get_qa_text_hash(qa_text)
            if qa_hash not in self.qa_embedding_cache:
                uncached_texts.append(qa_text)
        
        if not uncached_texts:
            if self.verbose_init:
                console.print(f"[green]✓ All {len(all_qa_data)} Q&A pairs already cached[/green]")
            # Build embeddings matrix even if all are cached
            self._build_qa_embeddings_matrix()
            return
        
        if self.verbose_init:
            console.print(f"Generating embeddings for {len(uncached_texts)} uncached Q&A pairs...")
        
        try:
            # Task for Q&A embeddings
            task = 'Given a question-answer pair, create an embedding that captures the semantic meaning for similarity comparison with user queries.'
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i+batch_size]
                if self.verbose_init:
                    console.print(f"  Processing batch {i//batch_size + 1}/{(len(uncached_texts) + batch_size - 1)//batch_size}...")
                
                # Create instructed texts for this batch
                instructed_batch = [get_detailed_instruct(task, qa_text) for qa_text in batch_texts]
                
                # Generate embeddings
                outputs = self.embedding_model.embed(instructed_batch)
                
                # Store in cache
                for qa_text, output in zip(batch_texts, outputs):
                    qa_hash = self._get_qa_text_hash(qa_text)
                    embedding = np.array(output.outputs.embedding)
                    self.qa_embedding_cache[qa_hash] = embedding
            
            if self.verbose_init:
                console.print(f"[green]✓ Precomputed {len(uncached_texts)} Q&A embeddings (total cached: {len(self.qa_embedding_cache)})[/green]")
            
            # Save the Q&A cache after precomputation
            self._save_qa_embeddings_cache()

            # Build embeddings matrix for fast similarity search
            self._build_qa_embeddings_matrix()

        except Exception as e:
            if self.verbose_init:
                console.print(f"[red]Error precomputing Q&A embeddings: {e}[/red]")
    
    def _embed_query(self, queries: List[str]) -> Optional[np.ndarray]:
        """Embed a query using the Qwen model."""
        if not self.embedding_model:
            return None
        
        # Use the same task description as for entity embeddings
        task = 'Given an query, create an embedding that captures the semantic meaning for similarity comparison with QA pairs.'
        instructed_queries = [get_detailed_instruct(task, query) for query in queries]
        
        try:
            outputs = self.embedding_model.embed(instructed_queries)
            embeddings = [np.array(output.outputs.embedding) for output in outputs]
            return embeddings
        except Exception as e:
            console.print(f"[red]Error embedding query: {e}[/red]")
            return None

    def _embed_chain(self, chain: str) -> Optional[np.ndarray]:
        """Embed a chain using the Qwen model."""
        if not self.embedding_model:
            return None
        
        task = 'Given a chain of questions and answer pairs, create an embedding that captures the semantic meaning captured across all question-answer pairs in the chain for similarity comparison with user queries.'
        instructed_chain = get_detailed_instruct(task, chain)

        try:
            outputs = self.embedding_model.embed([instructed_chain])
            embedding = np.array(outputs[0].outputs.embedding)
            return embedding
        except Exception as e:
            console.print(f"[red]Error embedding chain: {e}[/red]")
            return None
    
    def _get_qa_text_hash(self, qa_text: str) -> str:
        """Generate a hash for Q&A text to use as cache key."""
        return hashlib.md5(qa_text.encode()).hexdigest()
    
    def _save_entity_embeddings_cache(self):
        """Save entity embeddings to disk using FAISS (transfer from GPU to CPU first)."""
        if self.embeddings is None:
            return

        try:
            # Use FAISS if available, otherwise fall back to npz
            if FAISS_AVAILABLE and self.entity_faiss_index is not None:
                # Transfer from GPU to CPU
                cpu_index = faiss.index_gpu_to_cpu(self.entity_faiss_index)
                # Save CPU index
                faiss.write_index(cpu_index, str(self.entity_embedding_cache_file))
                if self.verbose_init:
                    console.print(f"[green]✓ Saved entity FAISS GPU index to {self.entity_embedding_cache_file}[/green]")

                # Save metadata separately
                metadata = {
                    "entity_texts": self.entity_texts,
                    "num_entities": len(self.entity_texts)
                }
                with open(self.cache_metadata_file, 'w') as f:
                    json.dump(metadata, f)
                if self.verbose_init:
                    console.print(f"[green]✓ Saved entity metadata to {self.cache_metadata_file}[/green]")
            else:
                # Fallback to old npz format if FAISS not available
                npz_file = self.cache_dir / f"entity_embeddings_-1docs.npz"
                np.savez_compressed(
                    npz_file,
                    embeddings=self.embeddings,
                    entity_texts=self.entity_texts
                )
                if self.verbose_init:
                    console.print(f"[green]✓ Saved entity embeddings to {npz_file} (fallback)[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save entity embeddings cache: {e}[/yellow]")
    
    def _load_entity_embeddings_cache(self) -> bool:
        """Load entity embeddings from disk cache and transfer to GPU.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        # Try loading GPU FAISS cache first
        if FAISS_AVAILABLE and self.gpu_resources and self.entity_embedding_cache_file.exists() and self.cache_metadata_file.exists():
            try:
                # Load metadata
                with open(self.cache_metadata_file, 'r') as f:
                    metadata = json.load(f)
                cached_texts = metadata.get('entity_texts', [])

                # Verify that cached texts match current entity texts
                if len(cached_texts) != len(self.entity_texts):
                    if self.verbose_init:
                        console.print("[yellow]Cache size mismatch, regenerating embeddings[/yellow]")
                    return False

                # Check if texts are the same (order matters)
                if not all(ct == et for ct, et in zip(cached_texts, self.entity_texts)):
                    if self.verbose_init:
                        console.print("[yellow]Entity texts changed, regenerating embeddings[/yellow]")
                    return False

                # Load CPU index
                cpu_index = faiss.read_index(str(self.entity_embedding_cache_file))

                # Transfer to GPU
                self.entity_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

                # Reconstruct embeddings array for compatibility
                self.embeddings = cpu_index.reconstruct_n(0, cpu_index.ntotal)
                self.embeddings = self.embeddings.reshape(cpu_index.ntotal, -1)

                if self.verbose_init:
                    console.print(f"[green]✓ Loaded GPU FAISS index with {self.entity_faiss_index.ntotal} vectors on GPU {self.gpu_device}[/green]")
                return True
            except Exception as e:
                if self.verbose_init:
                    console.print(f"[yellow]Could not load GPU cache: {e}, trying HNSW fallback[/yellow]")

        # Try loading legacy HNSW cache and convert to GPU
        if FAISS_AVAILABLE and self.gpu_resources and self.entity_embedding_hnsw_cache.exists() and self.cache_metadata_file.exists():
            try:
                if self.verbose_init:
                    console.print("[cyan]Found legacy HNSW cache, converting to GPU...[/cyan]")

                # Load metadata
                with open(self.cache_metadata_file, 'r') as f:
                    metadata = json.load(f)
                cached_texts = metadata.get('entity_texts', [])

                if len(cached_texts) == len(self.entity_texts) and all(ct == et for ct, et in zip(cached_texts, self.entity_texts)):
                    # Load HNSW index
                    cpu_index = faiss.read_index(str(self.entity_embedding_hnsw_cache))

                    # Reconstruct embeddings
                    self.embeddings = cpu_index.reconstruct_n(0, cpu_index.ntotal)
                    self.embeddings = self.embeddings.reshape(cpu_index.ntotal, -1).astype(np.float32)

                    # Create new flat index and transfer to GPU
                    embedding_dim = self.embeddings.shape[1]
                    new_cpu_index = faiss.IndexFlatIP(embedding_dim)
                    # Embeddings from HNSW should already be normalized
                    faiss.normalize_L2(self.embeddings)
                    new_cpu_index.add(self.embeddings)
                    self.entity_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, new_cpu_index)

                    if self.verbose_init:
                        console.print(f"[green]✓ Converted HNSW to GPU with {self.entity_faiss_index.ntotal} vectors[/green]")

                    # Save as GPU cache for next time
                    self._save_entity_embeddings_cache()
                    return True
            except Exception as e:
                if self.verbose_init:
                    console.print(f"[yellow]Could not convert HNSW cache: {e}[/yellow]")

        # Fallback: Try loading NPZ format and convert to GPU
        npz_file = self.cache_dir / f"entity_embeddings_-1docs.npz"
        if npz_file.exists():
            try:
                data = np.load(npz_file, allow_pickle=True)
                cached_texts = data['entity_texts']

                # Verify that cached texts match current entity texts
                if len(cached_texts) != len(self.entity_texts):
                    console.print("[yellow]Cache size mismatch, regenerating embeddings[/yellow]")
                    return False

                # Check if texts are the same (order matters)
                if not all(ct == et for ct, et in zip(cached_texts, self.entity_texts)):
                    console.print("[yellow]Entity texts changed, regenerating embeddings[/yellow]")
                    return False

                self.embeddings = data['embeddings'].astype(np.float32)

                # Build FAISS GPU index from loaded embeddings if FAISS is available
                if FAISS_AVAILABLE and self.gpu_resources:
                    embedding_dim = self.embeddings.shape[1]
                    cpu_index = faiss.IndexFlatIP(embedding_dim)
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(self.embeddings)
                    cpu_index.add(self.embeddings)
                    # Transfer to GPU
                    self.entity_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)
                    if self.verbose_init:
                        console.print(f"[green]✓ Loaded npz cache and built FAISS GPU index[/green]")
                return True
            except Exception as e:
                console.print(f"[yellow]Could not load entity embeddings cache: {e}[/yellow]")
                return False

        return False
    
    def _save_qa_embeddings_cache(self):
        """Save Q&A embeddings cache to disk using FAISS."""
        if not self.qa_embedding_cache:
            return

        try:
            # Use FAISS if available
            if FAISS_AVAILABLE and self.qa_faiss_index is not None:
                # Transfer from GPU to CPU first (matches entity embeddings pattern)
                cpu_index = faiss.index_gpu_to_cpu(self.qa_faiss_index)
                # Save CPU index
                faiss.write_index(cpu_index, str(self.qa_embedding_cache_file))
                if self.verbose_init:
                    console.print(f"[green]✓ Saved Q&A FAISS index to {self.qa_embedding_cache_file}[/green]")

                # Save metadata separately (hash to index mapping and metadata)
                metadata = {
                    "qa_hash_to_idx": self.qa_hash_to_idx,
                    "num_qa_pairs": len(self.qa_embedding_cache)
                }
                with open(self.qa_metadata_file, 'w') as f:
                    json.dump(metadata, f)
                if self.verbose_init:
                    console.print(f"[green]✓ Saved {len(self.qa_embedding_cache)} Q&A metadata[/green]")
            else:
                # Fallback to old npz format if FAISS not available
                qa_hashes = []
                qa_embeddings = []

                for qa_hash, embedding in self.qa_embedding_cache.items():
                    qa_hashes.append(qa_hash)
                    qa_embeddings.append(embedding)

                npz_file = self.cache_dir / f"qa_embeddings_-1docs.npz"
                np.savez_compressed(
                    npz_file,
                    hashes=qa_hashes,
                    embeddings=np.array(qa_embeddings)
                )
                if self.verbose_init:
                    console.print(f"[green]✓ Saved {len(self.qa_embedding_cache)} Q&A embeddings to {npz_file} (fallback)[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save Q&A embeddings cache: {e}[/yellow]")
    
    def _load_qa_embeddings_cache(self) -> bool:
        """Load Q&A embeddings from disk cache (FAISS or npz fallback).

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        # Try loading FAISS index first
        if FAISS_AVAILABLE and self.qa_embedding_cache_file.exists() and self.qa_metadata_file.exists():
            try:
                # Load metadata
                with open(self.qa_metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.qa_hash_to_idx = metadata.get('qa_hash_to_idx', {})

                # Load FAISS index
                self.qa_faiss_index = faiss.read_index(str(self.qa_embedding_cache_file))

                # Reconstruct embeddings from FAISS index
                num_vectors = self.qa_faiss_index.ntotal
                embeddings = self.qa_faiss_index.reconstruct_n(0, num_vectors)
                embeddings = embeddings.reshape(num_vectors, -1)

                # Rebuild the cache dictionary
                self.qa_embedding_cache = {}
                for qa_hash, idx in self.qa_hash_to_idx.items():
                    self.qa_embedding_cache[qa_hash] = embeddings[idx]

                if self.verbose_init:
                    console.print(f"[green]✓ Loaded FAISS index with {num_vectors} Q&A embeddings[/green]")
                return True
            except Exception as e:
                if self.verbose_init:
                    console.print(f"[yellow]Could not load Q&A FAISS cache: {e}, trying npz fallback[/yellow]")

        # Fallback: Try loading old npz format
        npz_file = self.cache_dir / f"qa_embeddings_-1docs.npz"
        if npz_file.exists():
            try:
                data = np.load(npz_file, allow_pickle=True)
                qa_hashes = data['hashes']
                qa_embeddings = data['embeddings']

                # Rebuild the cache dictionary
                self.qa_embedding_cache = {}
                for qa_hash, embedding in zip(qa_hashes, qa_embeddings):
                    self.qa_embedding_cache[qa_hash] = embedding

                if self.verbose_init:
                    console.print(f"[green]✓ Loaded {len(self.qa_embedding_cache)} Q&A embeddings from npz cache[/green]")
                return True
            except Exception as e:
                console.print(f"[yellow]Could not load Q&A embeddings cache: {e}[/yellow]")
                return False

        return False

    def _build_qa_embeddings_matrix(self):
        """Build a matrix of all Q&A embeddings for fast similarity search."""
        if not self.qa_embedding_cache:
            return

        # Create ordered lists of hashes and embeddings
        qa_hashes = []
        qa_embeddings = []

        for qa_hash, embedding in self.qa_embedding_cache.items():
            qa_hashes.append(qa_hash)
            qa_embeddings.append(embedding)

        # Convert to numpy array for fast similarity computation
        self.qa_embeddings_matrix = np.array(qa_embeddings, dtype=np.float32)

        # Create hash to index mapping
        self.qa_hash_to_idx = {qa_hash: idx for idx, qa_hash in enumerate(qa_hashes)}

        # Build FAISS index (GPU or CPU based on configuration)
        if FAISS_AVAILABLE and len(qa_embeddings) > 0:
            embedding_dim = self.qa_embeddings_matrix.shape[1]
            # Create CPU index for Q&A embeddings
            cpu_index = faiss.IndexFlatIP(embedding_dim)
            # Normalize embeddings for cosine similarity
            embeddings_normalized = self.qa_embeddings_matrix.copy()
            faiss.normalize_L2(embeddings_normalized)
            cpu_index.add(embeddings_normalized)

            # Transfer to GPU only if enabled and GPU resources available
            if self.use_gpu_for_qa_index and self.gpu_resources:
                try:
                    self.qa_faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)
                    if self.verbose_init:
                        console.print(f"[green]✓ Built Q&A FAISS GPU index with {self.qa_faiss_index.ntotal} vectors on GPU {self.gpu_device}[/green]")
                except Exception as e:
                    if self.verbose_init:
                        console.print(f"[yellow]Warning: Could not transfer Q&A index to GPU: {e}[/yellow]")
                        console.print(f"[yellow]Falling back to CPU index[/yellow]")
                    self.qa_faiss_index = cpu_index
            else:
                # Use CPU index
                self.qa_faiss_index = cpu_index
                if self.verbose_init:
                    console.print(f"[green]✓ Built Q&A FAISS CPU index with {self.qa_faiss_index.ntotal} vectors[/green]")
        elif self.verbose_init:
            console.print(f"[green]✓ Built Q&A embeddings matrix with shape {self.qa_embeddings_matrix.shape}[/green]")

    def search_qa_pairs_direct(self, query: str, query_embedding: np.ndarray = None, top_k: int = 10, verbose: bool = False) -> List[Dict[str, Any]]:
        """Search directly for Q&A pairs matching the query using FAISS GPU.

        Args:
            query: Search query
            top_k: Number of top Q&A pairs to return
            verbose: Whether to display search results

        Returns:
            List of Q&A pair dictionaries with similarity scores
        """
        if not self.embedding_model or self.qa_embeddings_matrix is None:
            if verbose:
                console.print("[yellow]Q&A embeddings not available for direct search[/yellow]")
            return []

        # Embed the query
        if query_embedding is not None:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        else:
            embeddings = self._embed_query(query)
            if embeddings is None:
                return []
            query_embedding = embeddings[0].reshape(1, -1).astype(np.float32)

        # Use FAISS GPU search if available, otherwise fall back to cosine_similarity
        if FAISS_AVAILABLE and self.qa_faiss_index is not None:
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
            # Search FAISS GPU index - exact search, no efSearch parameter
            similarities, indices = self.qa_faiss_index.search(query_embedding, top_k)

            # Build results from FAISS search
            results = []
            # Create reverse mapping from index to hash
            idx_to_hash = {idx: qa_hash for qa_hash, idx in self.qa_hash_to_idx.items()}

            for idx, similarity in zip(indices[0], similarities[0]):
                if idx >= 0 and idx in idx_to_hash:
                    qa_hash = idx_to_hash[idx]
                    if qa_hash in self.qa_metadata_cache:
                        metadata = self.qa_metadata_cache[qa_hash].copy()
                        metadata['similarity_score'] = float(similarity)
                        metadata['source_method'] = 'direct_qa_search_gpu'
                        results.append(metadata)
        else:
            # Fallback to cosine_similarity
            if not SIMILARITY_AVAILABLE:
                if verbose:
                    console.print("[yellow]Neither FAISS nor scikit-learn available[/yellow]")
                return []

            similarities = cosine_similarity(query_embedding, self.qa_embeddings_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                # Find the hash for this index
                qa_hash = None
                for h, i in self.qa_hash_to_idx.items():
                    if i == idx:
                        qa_hash = h
                        break

                if qa_hash and qa_hash in self.qa_metadata_cache:
                    metadata = self.qa_metadata_cache[qa_hash].copy()
                    metadata['similarity_score'] = float(similarities[idx])
                    metadata['source_method'] = 'direct_qa_search'
                    results.append(metadata)

        if verbose:
            search_method = "GPU" if (FAISS_AVAILABLE and self.qa_faiss_index is not None) else "cosine similarity"
            console.print(f"[cyan]Direct Q&A search ({search_method}) found {len(results)} pairs[/cyan]")
            if results:
                console.print(f"[dim]Top score: {results[0]['similarity_score']:.3f}[/dim]")

        return results

    def search(self, query: str, query_embedding: np.ndarray = None, top_k: int = 5, verbose: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """Search for entities matching the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            verbose: Whether to display search results
            
        Returns:
            List of (entity_info, score) tuples sorted by relevance
        """
        if verbose:
            console.print(f"[cyan]Searching for: '{query}'[/cyan]")

        if self.use_bm25 and self.bm25 is not None:
            results = self._search_with_bm25(query=query, top_k=top_k, verbose=verbose)
        elif self.embeddings is not None and SIMILARITY_AVAILABLE:
            results = self._search_with_embeddings(query=query, query_embedding=query_embedding, top_k=top_k, verbose=verbose)
        else:
            results = self._search_with_text(query=query, top_k=top_k, verbose=verbose)
        
        # Enrich results with QA pairs
        enriched_results = []
        for entity, score in results:
            # Find QA pairs for this entity
            qa_pairs = self._find_qa_pairs_for_entity(entity["id"], entity["doc_id"])
            entity["qa_pairs"] = qa_pairs
            enriched_results.append((entity, score))
        
        # Display Q&A pairs table
        if verbose:
            self._display_qa_pairs_table(enriched_results)
        

        return enriched_results
    
    def _search_with_embeddings(self, query: str, top_k: int, verbose: bool, query_embedding: np.ndarray = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search using FAISS GPU index for exact nearest neighbor search."""
        # Get query embedding
        if query_embedding is not None:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        else:
            embeddings = self._embed_query(query)
            if embeddings is None:
                console.print("[yellow]Warning: Could not embed query, using text search[/yellow]")
                return self._search_with_text(query, top_k, verbose)
            query_embedding = embeddings[0].reshape(1, -1).astype(np.float32)

        # Use FAISS GPU search if available, otherwise fall back to cosine_similarity
        if FAISS_AVAILABLE and self.entity_faiss_index is not None:
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
            # Search FAISS GPU index - returns (similarities, indices) - exact search, no efSearch needed
            similarities, indices = self.entity_faiss_index.search(query_embedding, top_k)

            # Build results from FAISS search
            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx >= 0 and idx < len(self.entities):  # Valid index
                    results.append((self.entities[idx], float(similarity)))
        else:
            # Fallback to cosine_similarity if FAISS not available
            if not SIMILARITY_AVAILABLE:
                console.print("[yellow]Warning: Neither FAISS nor scikit-learn available, using text search[/yellow]")
                return self._search_with_text(query, top_k, verbose)

            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(self.entities[idx], float(similarities[idx])) for idx in top_indices]

        if verbose:
            search_method = "FAISS GPU Search" if (FAISS_AVAILABLE and self.entity_faiss_index is not None) else "Embedding-based Search"
            self._display_search_results(results, search_method)

        return results
    
    def _search_with_text(self, query: str, top_k: int, verbose: bool) -> List[Tuple[Dict[str, Any], float]]:
        """Search using simple text matching."""
        query_lower = query.lower()
        
        scored_entities = []
        for entity in self.entities:
            score = 0.0
            text_lower = entity["search_text"].lower()
            
            # Simple scoring: exact matches, partial matches
            if query_lower in text_lower:
                score += 1.0
            
            # Boost for name matches
            if query_lower in entity["name"].lower():
                score += 2.0
            
            # Boost for role matches
            for role_info in entity["roles"]:
                if query_lower in role_info["role"].lower():
                    score += 1.5
                for state in role_info["states"]:
                    if query_lower in state.lower():
                        score += 1.0
            
            if score > 0:
                scored_entities.append((entity, score))
        
        # Sort by score and return top-k
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        results = scored_entities[:top_k]
        
        if verbose:
            self._display_search_results(results, "Text-based Search")
        
        return results


    def _search_with_bm25(self, query: str, top_k: int, verbose: bool):
        if not self.bm25:
            console.print("[yellow]BM25 index not built, using text search[/yellow]")
            return self._search_with_text(query, top_k, verbose)

        token_pattern = re.compile(r"\w+")
        query_tokens = token_pattern.findall(query.lower())
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.entities[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        if not results:
            return self._search_with_text(query, top_k, verbose)

        if verbose:
            self._display_search_results(results, "BM25 Search")
        return results

    
    def _display_search_results(self, results: List[Tuple[Dict[str, Any], float]], search_type: str):
        """Display search results in a formatted table."""
        if not results:
            console.print("[yellow]No search results found.[/yellow]")
            return
        
        console.print(f"\n[bold magenta]{search_type} Results:[/bold magenta]")
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Entity Name", style="green", width=20)
        table.add_column("Roles", style="white", width=25)
        table.add_column("Questions Answered", style="yellow", width=35)
        table.add_column("Location", style="dim", width=15)
        
        for entity, score in results:
            score_str = f"{score:.3f}"
            name = entity['name']
            
            # Show first few roles
            role_list = [r["role"] for r in entity['roles'][:2]]
            roles_text = ", ".join(role_list)
            if len(entity['roles']) > 2:
                roles_text += f" (+{len(entity['roles']) - 2})"
            
            # Show QA pairs
            qa_pairs = entity.get('qa_pairs', [])
            if qa_pairs:
                # Show first few questions
                questions = [qa["question"][:50] + "..." if len(qa["question"]) > 50 else qa["question"] for qa in qa_pairs[:2]]
                qa_text = "\n".join(questions)
                if len(qa_pairs) > 2:
                    qa_text += f"\n(+{len(qa_pairs) - 2} more)"
            else:
                qa_text = "[dim]No questions found[/dim]"
            
            location = entity['doc_id']
            
            table.add_row(score_str, name, roles_text, qa_text, location)
        
        console.print(table)
    
    def _display_qa_pairs_table(self, results: List[Tuple[Dict[str, Any], float]]):
        """Display all Q&A pairs from the top-k entities in a separate table."""
        if not results:
            return
        
        # Collect all Q&A pairs with entity info
        all_qa_pairs = []
        for entity, score in results:
            qa_pairs = entity.get('qa_pairs', [])
            for qa in qa_pairs:
                # Use resolved answer names if available
                answer_text = ', '.join(qa.get('answer_names', qa.get('answer_ids', [])))
                all_qa_pairs.append({
                    'entity_name': entity['name'],
                    'entity_score': score,
                    'question': qa['question'],
                    'verb_phrase': qa.get('verb_phrase', 'Unknown'),
                    'answer_ids': qa.get('answer_ids', []),
                    'answer_names': qa.get('answer_names', []),
                    'answers': answer_text,
                    'doc_id': entity['doc_id']
                })
        
        if not all_qa_pairs:
            console.print("\n[yellow]No Q&A pairs found for the retrieved entities.[/yellow]")
            return
        
        console.print(f"\n[bold magenta]All Q&A Pairs from Top-{len(results)} Entities ({len(all_qa_pairs)} total):[/bold magenta]")
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("#", style="dim", width=4)
        table.add_column("Question", style="white", width=45)
        table.add_column("Answer", style="green", width=25)
        table.add_column("Verb Context", style="cyan", width=20)
        table.add_column("Found Via", style="blue", width=15)
        table.add_column("Source", style="dim", width=10)
        
        for i, qa_info in enumerate(all_qa_pairs, 1):
            entity_name = qa_info['entity_name']
            question = qa_info['question']
            if len(question) > 60:
                question = question[:57] + "..."
            answers = qa_info['answers']
            if len(answers) > 30:
                answers = answers[:27] + "..."
            verb_phrase = qa_info['verb_phrase']
            if len(verb_phrase) > 25:
                verb_phrase = verb_phrase[:22] + "..."
            doc_id = qa_info['doc_id']
            
            table.add_row(str(i), question, answers, verb_phrase, entity_name, doc_id)
        
        console.print(table)
    
    def _rerank_qa_pairs(self, query: str, qa_pairs: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank Q&A pairs based on semantic similarity to the query.
        
        Args:
            query: The user's query
            qa_pairs: List of Q&A pair dictionaries
            top_k: Number of top Q&A pairs to return
            
        Returns:
            Top-k most relevant Q&A pairs
        """
        if not qa_pairs:
            return []
        
        # If we don't have embeddings capability, return first top_k pairs
        if not self.embedding_model or not SIMILARITY_AVAILABLE:
            console.print("[yellow]Embedding model not available, using first {} Q&A pairs[/yellow]".format(min(top_k, len(qa_pairs))))
            return qa_pairs[:top_k]
        
        # Embed the query
        # Removed debug print
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return qa_pairs[:top_k]
        
        # Removed verbose print
        
        # Create text representations for each Q&A pair (question + answer)
        qa_texts = []
        for qa in qa_pairs:
            # Combine question and answer for embedding
            answer_names = qa.get('answer_names', qa.get('answers', []))
            qa_text = f"{qa['question']} {', '.join(answer_names)}"
            qa_texts.append(qa_text)

        # Removed debug print
        
        # Get embeddings from cache (should all be precomputed)
        qa_embeddings = []
        missing_count = 0

        for qa_text in qa_texts:
            qa_hash = self._get_qa_text_hash(qa_text)
            
            if qa_hash in self.qa_embedding_cache:
                # Use cached embedding
                qa_embeddings.append(self.qa_embedding_cache[qa_hash])
                self.cache_hits += 1
            else:
                # This shouldn't happen if precomputation worked correctly
                missing_count += 1
                self.cache_misses += 1
                # Use zero embedding as fallback
                qa_embeddings.append(np.zeros(self.embeddings.shape[1]))  # Use same dimension as entity embeddings
        
        if missing_count > 0:
            console.print(f"[yellow]Warning: {missing_count} Q&A pairs not found in cache[/yellow]")
        
        qa_embeddings = np.array(qa_embeddings)
        
        # Calculate similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, qa_embeddings)[0]

        # Removed debug prints
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]


        # Return reranked Q&A pairs with scores
        reranked_pairs = []
        for idx in top_indices:
            qa_with_score = qa_pairs[idx].copy()
            qa_with_score['similarity_score'] = float(similarities[idx])
            reranked_pairs.append(qa_with_score)
        
        # Removed verbose output about selected Q&A pairs
        
        return reranked_pairs
    
    def _count_tokens(self, text: str, model: str = "gpt-4o-mini") -> Optional[int]:
        """Count the number of tokens in the text for a given model."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            # Get the encoding for the specific model
            encoding = tiktoken.encoding_for_model(model)
            
            # Count tokens
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not count tokens: {e}[/yellow]")
            return None
    
    def _generate_llm_answer(self, query: str, results: List[Tuple[Dict[str, Any], float]], qa_top_k: int = 15) -> str:
        """Generate an answer using LLM with reranked Q&A pairs as context.
        
        Args:
            query: The user's query
            results: Retrieved entity results
            qa_top_k: Number of top Q&A pairs to pass to LLM after reranking
        """
        if not self.openai_client:
            return "[yellow]LLM answer generation not available (OpenAI client not initialized)[/yellow]"
        
        if not results:
            return "[yellow]No entities found to generate answer from.[/yellow]"
        
        # Collect all Q&A pairs
        all_qa_pairs = []
        for entity, _ in results:
            qa_pairs = entity.get('qa_pairs', [])
            for qa in qa_pairs:
                # Use resolved answer names
                answer_text = ', '.join(qa.get('answer_names', qa.get('answer_ids', ['Unknown'])))
                all_qa_pairs.append({
                    'entity': entity['name'],
                    'question': qa['question'],
                    'answers': answer_text,
                    'verb_phrase': qa.get('verb_phrase', ''),
                    'doc_id': entity['doc_id']
                })
        
        if not all_qa_pairs:
            return "[yellow]No Q&A pairs found in the retrieved entities.[/yellow]"
        
        # Rerank Q&A pairs based on similarity to the query
        reranked_qa_pairs = self._rerank_qa_pairs(query, all_qa_pairs, top_k=qa_top_k)
        
        # Prepare context from reranked Q&A pairs
        context_parts = []
        context_parts.append("RELEVANT Q&A PAIRS FROM KNOWLEDGE BASE:")
        for i, qa in enumerate(reranked_qa_pairs, 1):
            context_parts.append(f"\n{i}. Question: {qa['question']}")
            context_parts.append(f"   Answer: {qa['answers']}")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following Q&A pairs retrieved from a knowledge base, please answer this question:

Question: {query}

{context}

It is likely that the answer to your question is present in the question, for example, 

1. Question: What is Leo Messi Top Scorer of? 
Context: Who is the top scorer of World Cup 2022?
Answer: Lionel Messi

In such cases, you should infer the answer from the question.

Please provide your answer in the following format:

<reasoning>
Reasoning to arrive at the answer based on the provided QA pairs. 
</reasoning>

<answer>
Only final answer, NA if you cannot find the answer.
</answer>
"""
        # Count tokens if tiktoken is available
        token_count = self._count_tokens(prompt, "gpt-4o-mini")
        
        # Display the context being sent to LLM (for debugging)
        if self.show_llm_prompt:
            console.print("\n[dim]Context being sent to LLM:[/dim]")
            console.print(Panel(prompt, title="LLM Prompt", expand=False, border_style="dim"))
        
        # Display token count
        if token_count:
            console.print(f"\n[cyan]Token count: {token_count} tokens[/cyan]")
        
        try:
            # Removed verbose generating message
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided Q&A pairs from a knowledge base."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"[red]Error generating answer: {str(e)}[/red]"
    
    def show_entity_details(self, entity: Dict[str, Any]):
        """Display detailed information about an entity."""
        console.print(f"\n[bold yellow]Entity: {entity['name']}[/bold yellow]")
        console.print(f"[dim]ID: {entity['id']} | Location: {entity['doc_id']}/{entity['chunk_id']}[/dim]")
        
        if entity['roles']:
            console.print(f"\n[cyan]Roles and States:[/cyan]")
            for role_info in entity['roles']:
                states_text = ", ".join(role_info['states']) if role_info['states'] else "None"
                console.print(f"  • {role_info['role']}: {states_text}")
        
        if entity['summary']:
            console.print(f"\n[cyan]Summary:[/cyan]")
            console.print(Panel(entity['summary'], expand=False))
        
        console.print(f"\n[cyan]Search Text:[/cyan]")
        console.print(Panel(entity['search_text'], expand=False))
        
        # Show QA pairs if available
        qa_pairs = entity.get('qa_pairs', [])
        if qa_pairs:
            console.print(f"\n[cyan]Questions This Entity Answers ({len(qa_pairs)}):[/cyan]")
            for i, qa in enumerate(qa_pairs[:5], 1):  # Show first 5
                console.print(f"  {i}. {qa['question']}")
            if len(qa_pairs) > 5:
                console.print(f"  ... and {len(qa_pairs) - 5} more questions")
        else:
            console.print(f"\n[dim]No questions found where this entity is an answer.[/dim]")
    
    def run_interactive_mode(self):
        """Run interactive search mode."""
        console.print("\n[bold cyan]🔍 Interactive Entity Search[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a query to search entities")
        console.print("  - 'show <entity_name>' - Show detailed info about an entity")
        console.print("  - 'topk <number>' - Set number of results to retrieve")
        console.print("  - 'qatopk <number>' - Set number of Q&A pairs for LLM (after reranking)")
        console.print("  - 'answer on/off' - Toggle LLM answer generation")
        console.print("  - 'debug on/off' - Toggle showing LLM prompts")
        console.print("  - 'stats' - Show entity statistics")
        console.print("  - 'test' - Run predefined test queries")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' or 'exit' - Exit")
        
        top_k = 10
        qa_top_k = 5
        generate_answer = True
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]Query[/bold yellow]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif user_input.lower() == 'help':
                    console.print("Commands: query, show <name>, topk <n>, qatopk <n>, answer on/off, debug on/off, stats, test, quit")
                
                elif user_input.lower().startswith('debug'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        if parts[1].lower() == 'on':
                            self.show_llm_prompt = True
                            console.print("[green]Debug mode enabled - will show LLM prompts[/green]")
                        elif parts[1].lower() == 'off':
                            self.show_llm_prompt = False
                            console.print("[yellow]Debug mode disabled[/yellow]")
                        else:
                            console.print("[red]Use 'debug on' or 'debug off'[/red]")
                
                elif user_input.lower().startswith('answer'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        if parts[1].lower() == 'off':
                            generate_answer = False
                            console.print("[yellow]LLM answer generation disabled[/yellow]")
                        elif parts[1].lower() == 'on':
                            generate_answer = True
                            console.print("[green]LLM answer generation enabled[/green]")
                        else:
                            console.print("[red]Use 'answer on' or 'answer off'[/red]")
                
                elif user_input.lower().startswith('qatopk'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            qa_top_k = int(parts[1])
                            console.print(f"[green]Q&A top-k set to: {qa_top_k}[/green]")
                        except ValueError:
                            console.print("[red]Invalid number for Q&A top-k[/red]")
                
                elif user_input.lower().startswith('topk'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            top_k = int(parts[1])
                            console.print(f"[green]Top-k set to: {top_k}[/green]")
                        except ValueError:
                            console.print("[red]Invalid number for top-k[/red]")
                
                elif user_input.lower().startswith('show'):
                    name = user_input[4:].strip()
                    if name:
                        # Find entity by name
                        found = False
                        for entity in self.entities:
                            if name.lower() in entity['name'].lower():
                                # Add QA pairs before showing details
                                qa_pairs = self._find_qa_pairs_for_entity(entity["id"], entity["doc_id"])
                                entity["qa_pairs"] = qa_pairs
                                self.show_entity_details(entity)
                                found = True
                                break
                        if not found:
                            console.print(f"[yellow]No entity found matching: {name}[/yellow]")
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                
                elif user_input.lower() == 'test':
                    self._run_test_queries(top_k)
                
                else:
                    # Process as a search query
                    self.search(user_input, top_k, verbose=True, generate_answer=generate_answer, qa_top_k=qa_top_k)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def _show_stats(self):
        """Show statistics about the loaded entities."""
        if not self.entities:
            console.print("[yellow]No entities loaded.[/yellow]")
            return
        
        total_entities = len(self.entities)
        entities_with_roles = len([e for e in self.entities if e['roles']])
        avg_roles = np.mean([len(e['roles']) for e in self.entities])
        
        # Document statistics
        unique_docs = len(set(e['doc_id'] for e in self.entities))
        unique_chunks = len(set(f"{e['doc_id']}/{e['chunk_id']}" for e in self.entities))
        
        console.print(f"[bold cyan]Entity Statistics:[/bold cyan]")
        console.print(f"  Total entities: {total_entities}")
        console.print(f"  Entities with roles: {entities_with_roles}")
        console.print(f"  Average roles per entity: {avg_roles:.1f}")
        console.print(f"  Unique documents: {unique_docs}")
        console.print(f"  Unique chunks: {unique_chunks}")
        console.print(f"  Embeddings available: {'Yes' if self.embeddings is not None else 'No'}")
    
    def save_cache(self):
        """Save all caches to disk."""
        self._save_entity_embeddings_cache()
        self._save_qa_embeddings_cache()
        console.print(f"[green]✓ Cache saved (hits: {self.cache_hits}, misses: {self.cache_misses})[/green]")
    
    def _run_test_queries(self, top_k: int):
        """Run predefined test queries."""
        test_queries = [
            "participant",
            "workshop",
            "investigation",
            "character",
            "event"
        ]
        
        console.print(f"\n[bold cyan]Running {len(test_queries)} test queries...[/bold cyan]")
        
        for i, query in enumerate(test_queries, 1):
            console.print(f"\n[bold]Test {i}/{len(test_queries)}: '{query}'[/bold]")
            self.search(query, top_k, verbose=True, generate_answer=True)


def main():
    """Main function - directly start interactive mode."""
    console.print("\n[bold cyan]🔍 GSW Entity Search System[/bold cyan]")
    console.print("Loading entities from GSW corpus...")
    
    # Configuration
    num_documents = 200  # Load first 200 documents
    
    # Initialize searcher
    try:
        searcher = EntitySearcher(num_documents)
    except Exception as e:
        console.print(f"[red]Error initializing searcher: {e}[/red]")
        return
    
    if not searcher.entities:
        console.print("[red]No entities found in the GSW files.[/red]")
        return
    
    # Start interactive mode
    searcher.run_interactive_mode()


if __name__ == "__main__":
    main()