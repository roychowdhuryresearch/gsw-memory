#!/usr/bin/env python3
"""
Multi-Hop Question Answering System

Uses the enhanced EntitySearcher for multi-hop reasoning through:
1. LLM-based question decomposition with entity placeholders
2. Chain generation from Q&A pair answer entities
3. Sequential chain execution using semantic search and reranking
4. Final LLM reasoning over accumulated evidence chains
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from dataclasses import dataclass
from datetime import datetime

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import our enhanced entity searcher
from playground.simple_entity_search import EntitySearcher

# OpenAI for question decomposition and final reasoning
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"multihop_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_handle = open(LOG_FILE, "w", buffering=1)  # Line buffered

class LoggingConsole(Console):
    """Console that also logs to file."""
    def print(self, *args, **kwargs):
        # Capture plain text
        with self.capture() as capture:
            super().print(*args, **kwargs)
        
        # Write to log file
        text = capture.get()
        if text:
            log_handle.write(text)
            log_handle.flush()
        
        # Print to terminal
        super().print(*args, **kwargs)

console = LoggingConsole()
print(f"📝 Logging to: {LOG_FILE}")


@dataclass
class ReasoningChain:
    """Represents a single reasoning chain through multiple hops."""
    chain_id: str
    steps: List[Dict[str, Any]]
    final_entities: List[str]
    accumulated_evidence: Dict[str, List[Dict[str, Any]]]  # Dict mapping questions to their evidences
    

class MultiHopQA:
    """Simple multi-hop QA system using enhanced entity search."""
    
    def __init__(self, num_documents: int = 200, verbose: bool = True):
        """Initialize multi-hop QA system.
        
        Args:
            num_documents: Number of documents to load for entity search
            verbose: Whether to show detailed output during initialization and processing
        """
        self.verbose = verbose
        
        if verbose:
            console.print("[bold blue]Initializing Multi-Hop QA System...[/bold blue]")
        
        # Initialize the enhanced entity searcher
        self.entity_searcher = EntitySearcher(num_documents, cache_dir="/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/.gsw_cache", verbose=self.verbose)
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                if verbose:
                    console.print("[green]✓ OpenAI client initialized for decomposition and reasoning[/green]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not initialize OpenAI client: {e}[/yellow]")
        
        if verbose:
            console.print("[bold green]✓ Multi-Hop QA System ready[/bold green]")
    
    def decompose_question(self, question: str) -> List[Dict[str, Any]]:
        """Decompose a multi-hop question into single-hop questions with <ENTITY> placeholders and retrieval flags.
        
        Args:
            question: The multi-hop question to decompose
            
        Returns:
            List of dictionaries with 'question' and 'requires_retrieval' keys.
            First question is specific and subsequent use <ENTITY> placeholder.
        """
        if not self.openai_client:
            if self.verbose:
                console.print("[red]OpenAI client not available for question decomposition[/red]")
            return [{"question": question, "requires_retrieval": True}]  # Fallback to original question
        
        decomposition_prompt = f"""Break down this multi-hop question into a sequence of single-hop questions.
The FIRST question should keep the original specific entity/information from the question.
SUBSEQUENT questions should use <ENTITY> as a placeholder if it requires answers from the previous step to form the question.

IMPORTANT: Avoid over-decomposition. Each question should extract meaningful entities (proper nouns like names, places), not single-word descriptors. Keep questions at an appropriate granularity level.

For each question, indicate whether it requires retrieval from the knowledge base (requires_retrieval: true) 
or can be answered directly from the previous step's result (requires_retrieval: false).

Format each decomposed question as:
Question: [the question text]
Requires retrieval: [true/false]

Examples:

Question: "What is the birth year of the spouse of the director of Casablanca?"
Decomposition:
1. Question: Who directed Casablanca?
   Requires retrieval: true
2. Question: Who was <ENTITY_Q1>'s spouse?
   Requires retrieval: true
3. Question: What is <ENTITY_Q2>'s birth year?
   Requires retrieval: true

Question: "Which film has the director who is older, Dune or The Dark Knight?"
Decomposition:
1. Question: Who directed Dune?
   Requires retrieval: true
2. Question: Who directed The Dark Knight?
   Requires retrieval: true
3. Question: When was <ENTITY_Q1> born?
   Requires retrieval: true
4. Question: When was <ENTITY_Q2> born?
   Requires retrieval: true
5. Question: Who is older, <ENTITY_Q3> or <ENTITY_Q4>?
   Requires retrieval: false


IMPORTANT:
    AVOID over-decomposition like this:
    DON'T break "Who is John Doe?" into:
    1. Who is John Doe? → "English"
    2. When was <ENTITY_Q1> born? → "When was English born?"

    DO ask directly: "When was John Doe born?"

Now decompose this question:
Question: "{question}"
Decomposition:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            decomposition_text = response.choices[0].message.content
            
            # Parse the structured response with questions and retrieval flags
            questions = []
            lines = decomposition_text.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for question lines (starting with number or "Question:")
                if re.match(r'^[\d]+[\.\)]\s*Question:', line) or line.startswith('Question:'):
                    # Extract the question text
                    question_match = re.search(r'Question:\s*(.+)', line)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        
                        # Look for the requires_retrieval flag in the next line or same line
                        requires_retrieval = True  # Default to true
                        
                        # Check if it's on the same line
                        if 'Requires retrieval:' in line:
                            retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', line, re.IGNORECASE)
                            if retrieval_match:
                                requires_retrieval = retrieval_match.group(1).lower() == 'true'
                        # Check the next line
                        elif i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if 'Requires retrieval:' in next_line:
                                retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', next_line, re.IGNORECASE)
                                if retrieval_match:
                                    requires_retrieval = retrieval_match.group(1).lower() == 'true'
                                i += 1  # Skip the retrieval line
                        
                        questions.append({
                            "question": question_text,
                            "requires_retrieval": requires_retrieval
                        })
                
                i += 1
            
            if not questions:
                # Fallback: treat the whole response as a single question if parsing fails
                if self.verbose:
                    console.print("[yellow]Warning: Could not parse structured decomposition, using fallback[/yellow]")
                return [{"question": question, "requires_retrieval": True}]
            
            if self.verbose:
                console.print(f"[cyan]Decomposed into {len(questions)} questions:[/cyan]")
                for i, q in enumerate(questions, 1):
                    retrieval_str = "✓ retrieval" if q["requires_retrieval"] else "✗ no retrieval"
                    console.print(f"  {i}. {q['question']} [{retrieval_str}]")
            
            return questions if questions else [{"question": question, "requires_retrieval": True}]
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]Error in question decomposition: {e}[/red]")
            return [{"question": question, "requires_retrieval": True}]  # Fallback to original question
    
    def parse_entity_references(self, decomposed_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse entity references in decomposed questions to understand dependencies.
        
        Args:
            decomposed_questions: List of decomposed question dictionaries
            
        Returns:
            Dictionary with entity mapping information
        """
        entity_map = {}
        question_entities = {}  # Maps question index to entities it produces
        
        for i, q_info in enumerate(decomposed_questions):
            question = q_info["question"]
            
            # Check if this question references previous entities
            entity_refs = re.findall(r'<ENTITY_Q(\d+)>', question)
            if entity_refs:
                q_info["entity_refs"] = [f"Q{ref}" for ref in entity_refs]
            
            # Check if this question will produce entities (usually first questions or entity-finding questions)
            if i == 0 or (i > 0 and "Who" in question and "<ENTITY" not in question):
                question_entities[f"Q{i+1}"] = True
        
        return {
            "entity_producers": question_entities,
            "decomposed_with_refs": decomposed_questions
        }
    
    def extract_global_entity_ids_from_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[str]:
        """Extract unique global entity IDs from reranked Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A pair dictionaries with answer_ids and source doc info
            
        Returns:
            List of unique global entity IDs (format: "doc_id::entity_id") to use for next hop
        """
        global_entity_ids = set()
        entity_names = []
        for qa in qa_pairs:
            answer_ids = qa.get('answer_ids', [])
            source_doc_id = qa.get('doc_id', '')  # Get the document this Q&A came from
            
            for entity_id in answer_ids:
                if entity_id and source_doc_id:
                    # Create global entity ID: doc_id::entity_id
                    global_id = f"{source_doc_id}::{entity_id}"
                    global_entity_ids.add(global_id)
                    entity_names.append(qa.get('answer_names', ''))
        
        return list(global_entity_ids)
    
    def lookup_entity_by_global_id(self, global_entity_id: str) -> Optional[Dict[str, Any]]:
        """Look up a specific entity by its global ID.
        
        Args:
            global_entity_id: Global entity ID in format "doc_id::entity_id"
            
        Returns:
            Entity information dict or None if not found
        """
        if "::" not in global_entity_id:
            return None
        
        doc_id, entity_id = global_entity_id.split("::", 1)
        
        # Check if we have the GSW structure for this document
        if doc_id not in self.entity_searcher.gsw_by_doc_id:
            if self.verbose:
                console.print(f"[red]Document {doc_id} not found in GSW structures[/red]")
            return None
        
        gsw = self.entity_searcher.gsw_by_doc_id[doc_id]
        
        # Find the entity in the GSW structure
        for entity_node in gsw.entity_nodes:
            if entity_node.id == entity_id:
                # Create entity info similar to what entity_searcher returns
                role_descriptions = []
                for role in entity_node.roles:
                    if role.states:
                        states_text = ', '.join(role.states)
                        role_desc = f"{role.role}: {states_text}"
                    else:
                        role_desc = role.role
                    role_descriptions.append(role_desc)
                
                if role_descriptions:
                    roles_text = ' | '.join(role_descriptions)
                    search_text = f"{entity_node.name} - Roles: {roles_text}"
                else:
                    search_text = f"{entity_node.name} - No specific roles"
                
                return {
                    "id": entity_node.id,
                    "global_id": global_entity_id,
                    "name": entity_node.name,
                    "roles": [{
                        "role": role.role,
                        "states": role.states
                    } for role in entity_node.roles],
                    "all_states": [state for role in entity_node.roles for state in role.states],
                    "search_text": search_text,
                    "doc_id": doc_id,
                    "chunk_id": "0",
                    "summary": "",
                    "role_descriptions": role_descriptions
                }
        
        if self.verbose:
            console.print(f"[red]Entity {entity_id} not found in document {doc_id}[/red]")
        return None
    
    def find_qa_pairs_for_global_entity(self, global_entity_id: str) -> List[Dict[str, Any]]:
        """Find Q&A pairs involving a specific global entity ID.
        
        Args:
            global_entity_id: Global entity ID in format "doc_id::entity_id"
            
        Returns:
            List of Q&A pairs involving this entity
        """
        if "::" not in global_entity_id:
            return []
        
        doc_id, entity_id = global_entity_id.split("::", 1)
        
        # Use the entity searcher's method to find Q&A pairs for this specific entity
        qa_pairs = self.entity_searcher._find_qa_pairs_for_entity(entity_id, doc_id)
        
        # Add document context to each Q&A pair
        for qa in qa_pairs:
            qa['doc_id'] = doc_id
            qa['global_entity_id'] = global_entity_id
        
        return qa_pairs
    
    def _get_entity_name_from_global_id(self, global_entity_id: str) -> Optional[str]:
        """Get entity name from global entity ID.
        
        Args:
            global_entity_id: Global entity ID in format "doc_id::entity_id"
            
        Returns:
            Entity name or None if not found
        """
        entity = self.lookup_entity_by_global_id(global_entity_id)
        return entity['name'] if entity else None
    
    def _rerank_qa_pairs_for_multihop(self, query: str, qa_pairs: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """Rerank Q&A pairs based on semantic similarity to the query for multi-hop reasoning.
        
        Args:
            query: The query to rank Q&A pairs against
            qa_pairs: List of Q&A pair dictionaries
            top_k: Number of top Q&A pairs to return
            
        Returns:
            Top-k most relevant Q&A pairs
        """
        if not qa_pairs:
            return []
        
        # Check if we have embeddings capability
        if not hasattr(self.entity_searcher, 'embedding_model') or not self.entity_searcher.embedding_model:
            if self.verbose:
                console.print(f"[yellow]Embedding model not available, using first {min(top_k, len(qa_pairs))} Q&A pairs[/yellow]")
            return qa_pairs[:top_k]
        
        try:
            # Import required modules for embeddings
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Embed the query using entity_searcher's method
            query_embedding = self.entity_searcher._embed_query(query)
            if query_embedding is None:
                return qa_pairs[:top_k]
            
            if self.verbose:
                console.print(f"[cyan]Reranking {len(qa_pairs)} Q&A pairs for multi-hop reasoning...[/cyan]")
            
            # Create text representations and get cached embeddings
            qa_embeddings = []
            missing_count = 0
            
            for qa in qa_pairs:
                # Use answer_names if available, otherwise use answers
                answer_text = qa.get('answer_names', qa.get('answers', ''))
                if isinstance(answer_text, list):
                    answer_text = ', '.join(str(a) for a in answer_text)
                qa_text = f"{qa['question']} {answer_text}"
                
                # Get embedding from cache
                qa_hash = self.entity_searcher._get_qa_text_hash(qa_text)
                
                if qa_hash in self.entity_searcher.qa_embedding_cache:
                    # Use cached embedding
                    qa_embeddings.append(self.entity_searcher.qa_embedding_cache[qa_hash])
                else:
                    # This shouldn't happen if precomputation worked correctly
                    missing_count += 1
                    # Use zero embedding as fallback
                    qa_embeddings.append(np.zeros_like(query_embedding))
            
            if missing_count > 0 and self.verbose:
                console.print(f"[yellow]Warning: {missing_count} Q&A pairs not found in cache[/yellow]")
            
            qa_embeddings = np.array(qa_embeddings)
            
            # Calculate similarities
            query_embedding = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding, qa_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return reranked Q&A pairs with scores
            reranked_pairs = []
            for idx in top_indices:
                qa_with_score = qa_pairs[idx].copy()
                qa_with_score['similarity_score'] = float(similarities[idx])
                reranked_pairs.append(qa_with_score)
            
            if reranked_pairs and self.verbose:
                console.print(f"[green]Selected top {len(reranked_pairs)} Q&A pairs (scores: {reranked_pairs[0]['similarity_score']:.3f} - {reranked_pairs[-1]['similarity_score']:.3f})[/green]")
            
            return reranked_pairs
            
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Warning: Q&A reranking failed: {e}. Using original order.[/yellow]")
            return qa_pairs[:top_k]
    
    def _extract_unique_entities_from_reranked_qa(self, reranked_qa_pairs: List[Dict[str, Any]], max_entities: int = 5) -> List[tuple]:
        """Extract unique global entity IDs from reranked Q&A pairs along with the Q&A pair that justified each selection.
        
        Args:
            reranked_qa_pairs: List of reranked Q&A pairs (most relevant first)
            max_entities: Maximum number of unique entities to extract
            
        Returns:
            List of tuples: (global_entity_id, qa_pair_that_contains_this_entity)
        """
        unique_entities = []
        seen_entities = set()
        
        for qa in reranked_qa_pairs:
            answer_ids = qa.get('answer_ids', [])
            source_doc_id = qa.get('doc_id', '')
            
            for entity_id in answer_ids:
                if entity_id and source_doc_id:
                    # Create global entity ID
                    global_id = f"{source_doc_id}::{entity_id}"
                    
                    # Only add if we haven't seen this entity before
                    if global_id not in seen_entities:
                        unique_entities.append((global_id, qa))  # Store both entity and the Q&A pair
                        seen_entities.add(global_id)
                        
                        # Stop once we have enough unique entities
                        if len(unique_entities) >= max_entities:
                            if self.verbose:
                                console.print(f"[dim]Extracted {len(unique_entities)} unique entities from top-ranked Q&A pairs[/dim]")
                            return unique_entities
        
        if self.verbose:
            console.print(f"[dim]Extracted {len(unique_entities)} unique entities from {len(reranked_qa_pairs)} Q&A pairs[/dim]")
        return unique_entities
    
    def execute_single_hop(self, question: str, step_num: int, chain_id: str = "", is_final_step: bool = False, final_top_k: int = 15) -> Dict[str, Any]:
        """Execute a single hop query using semantic search across all documents.
        
        Args:
            question: The question to search for (with entities already substituted)
            step_num: Step number in the reasoning chain
            chain_id: ID of the chain this hop belongs to
            is_final_step: Whether this is the final step (for answer generation, not entity extraction)
            final_top_k: Number of Q&A pairs to keep for final answer generation
            
        Returns:
            Dictionary with search results, reranked Q&A pairs, and extracted global entity IDs (empty for final step)
        """
        if self.verbose:
            console.print(f"\n[bold cyan]Step {step_num} - Chain {chain_id}:[/bold cyan] {question}")
            console.print(f"[dim]Semantic search across all documents: {question}[/dim]")
        
        # Always use semantic search to find diverse information across documents
        search_results = self.entity_searcher.search(
            query=question,
            top_k=10,  # Get top 10 entities
            verbose=False,  # Don't show tables for intermediate steps
        )
        
        # Extract Q&A pairs from semantic search results
        all_qa_pairs = []
        for entity, score in search_results:
            qa_pairs = entity.get('qa_pairs', [])
            for qa in qa_pairs:
                qa_with_context = qa.copy()
                qa_with_context['source_entity'] = entity['name']
                qa_with_context['source_entity_id'] = entity['id']
                qa_with_context['entity_score'] = score
                qa_with_context['doc_id'] = entity['doc_id']
                all_qa_pairs.append(qa_with_context)
        
        # Rerank Q&A pairs by relevance to the question
        if is_final_step:
            # For final step: Keep more Q&A pairs for answer generation
            reranked_qa_pairs = self._rerank_qa_pairs_for_multihop(question, all_qa_pairs, top_k=final_top_k)
            global_entity_ids_for_next_hop = []  # No next hop needed for final step
            if self.verbose:
                console.print(f"[green]Final step: Selected top {len(reranked_qa_pairs)} Q&A pairs for answer generation[/green]")
        else:
            # For intermediate steps: Rerank all Q&A pairs, then extract entities for next hop
            reranked_qa_pairs = self._rerank_qa_pairs_for_multihop(question, all_qa_pairs, top_k=-1)
            entities_with_qa = self._extract_unique_entities_from_reranked_qa(reranked_qa_pairs, max_entities=5)
            # Split into separate lists for backward compatibility
            global_entity_ids_for_next_hop = [entity_id for entity_id, _ in entities_with_qa]
            if self.verbose:
                console.print(f"[green]Intermediate step: Found {len(global_entity_ids_for_next_hop)} global entity IDs for next hop: {global_entity_ids_for_next_hop[:3]}{'...' if len(global_entity_ids_for_next_hop) > 3 else ''}[/green]")
        
        return {
            'question': question,
            'search_results': search_results,
            'qa_pairs': reranked_qa_pairs,  # Use reranked Q&A pairs for better quality
            'next_hop_global_entity_ids': global_entity_ids_for_next_hop,
            'next_hop_entities_with_qa': entities_with_qa if not is_final_step else [],  # Include Q&A pairs for each entity
            'step_num': step_num,
            'chain_id': chain_id
        }
    
    def generate_chains(self, decomposed_questions: List[Dict[str, Any]], show_intermediate_qa: bool = True) -> List[ReasoningChain]:
        """Generate and execute reasoning chains for multi-hop questions.
        
        Args:
            decomposed_questions: List of dicts with 'question' and 'requires_retrieval' keys
            show_intermediate_qa: Whether to store detailed Q&A pairs for intermediate steps
            
        Returns:
            List of completed reasoning chains
        """
        if not decomposed_questions:
            return []
        
        if self.verbose:
            console.print(f"\n[bold magenta]Executing {len(decomposed_questions)} reasoning steps...[/bold magenta]")
        
        # Start with the first question - no entity focus needed
        first_q_info = decomposed_questions[0]
        first_question = first_q_info["question"]
        
        # Only execute retrieval if needed
        if first_q_info["requires_retrieval"]:
            first_hop_result = self.execute_single_hop(first_question, step_num=1, chain_id="init")
        else:
            # Skip retrieval for questions that don't need it
            if self.verbose:
                console.print(f"\n[bold cyan]Step 1 - Chain init:[/bold cyan] {first_question}")
                console.print(f"[dim]Skipping retrieval (requires_retrieval=false)[/dim]")
            first_hop_result = {
                'question': first_question,
                'search_results': [],
                'qa_pairs': [],
                'next_hop_global_entity_ids': [],
                'step_num': 1,
                'chain_id': "init"
            }
        
        # Initialize chains with global entity IDs from first hop
        chains = []
        entities_with_qa = first_hop_result.get('next_hop_entities_with_qa', [])
        
        if entities_with_qa:
            # We have both entities and their justifying Q&A pairs
            for i, (global_entity_id, qa_pair) in enumerate(entities_with_qa):
                # Initialize accumulated_evidence as dict with first question
                # For intermediate steps, only store the Q&A pair that justified this entity selection
                if not show_intermediate_qa:
                    initial_evidence = {first_hop_result['question']: self._get_entity_name_from_global_id(global_entity_id)}
                else:
                    # Store only the Q&A pair that led to selecting this entity
                    initial_evidence = {first_hop_result['question']: [qa_pair]}
                
                chain = ReasoningChain(
                    chain_id=f"chain_{i+1}",
                    steps=[first_hop_result],
                    final_entities=[global_entity_id],  # Store global entity IDs
                    accumulated_evidence=initial_evidence
                )
                chains.append(chain)
        elif first_hop_result['next_hop_global_entity_ids']:
            # Fallback: find the Q&A pair for each entity manually
            qa_pairs = first_hop_result.get('qa_pairs', [])
            for i, global_entity_id in enumerate(first_hop_result['next_hop_global_entity_ids']):
                # Find the Q&A pair that contains this entity
                justifying_qa = None
                if "::" in global_entity_id:
                    doc_id, entity_id = global_entity_id.split("::", 1)
                    for qa in qa_pairs:
                        if qa.get('doc_id') == doc_id and entity_id in qa.get('answer_ids', []):
                            justifying_qa = qa
                            break
                
                if not show_intermediate_qa:
                    initial_evidence = {first_hop_result['question']: self._get_entity_name_from_global_id(global_entity_id)}
                else:
                    # Store only the justifying Q&A pair, not all of them
                    initial_evidence = {first_hop_result['question']: [justifying_qa] if justifying_qa else []}
                
                chain = ReasoningChain(
                    chain_id=f"chain_{i+1}",
                    steps=[first_hop_result],
                    final_entities=[global_entity_id],  # Store global entity IDs
                    accumulated_evidence=initial_evidence
                )
                chains.append(chain)
        else:
            # Special case: first question didn't need retrieval or didn't find entities
            # Create a single chain with no entities
            initial_evidence = {first_hop_result['question']: first_hop_result['qa_pairs'] if show_intermediate_qa else "No retrieval needed"}
            chain = ReasoningChain(
                chain_id="chain_1",
                steps=[first_hop_result],
                final_entities=[],  # No entities yet
                accumulated_evidence=initial_evidence
            )
            chains.append(chain)
        
        if self.verbose:
            console.print(f"[yellow]Generated {len(chains)} initial chains[/yellow]")
        
        # Execute remaining questions for each chain
        for step_idx in range(1, len(decomposed_questions)):
            q_info = decomposed_questions[step_idx]
            question_template = q_info["question"]
            requires_retrieval = q_info["requires_retrieval"]
            if not requires_retrieval:
                continue
            is_final_step_global = (step_idx + 1) == len(decomposed_questions)
            
            if self.verbose:
                if is_final_step_global:
                    console.print(f"\n[bold red]FINAL Step {step_idx + 1} Template:[/bold red] {question_template} [retrieval: {requires_retrieval}]")
                    console.print(f"[dim]This is the final step - collecting Q&A pairs for answer generation[/dim]")
                else:
                    console.print(f"\n[bold magenta]Step {step_idx + 1} Template:[/bold magenta] {question_template} [retrieval: {requires_retrieval}]")
            
            new_chains = []
            
            for chain in chains:
                # Handle chains with no entities (e.g., from non-retrieval questions)
                if not chain.final_entities:
                    # Process question without entity substitution
                    if self.verbose:
                        console.print(f"[bold yellow]Chain {chain.chain_id} - No entities to substitute[/bold yellow]")
                    
                    # For questions with <ENTITY> but no entities, we might need to skip or handle specially
                    if "<ENTITY>" in question_template:
                        if self.verbose:
                            console.print(f"[yellow]Warning: Question template contains <ENTITY> but no entities available from previous step[/yellow]")
                        continue  # Skip this chain for this question
                    
                    # Process the question as-is without substitution
                    substituted_question = question_template
                    is_final_step = (step_idx + 1) == len(decomposed_questions)
                    
                    # Execute based on retrieval flag
                    if requires_retrieval:
                        hop_result = self.execute_single_hop(
                            substituted_question, 
                            step_num=step_idx + 1, 
                            chain_id=chain.chain_id,
                            is_final_step=is_final_step,
                            final_top_k=15
                        )
                    else:
                        hop_result = {
                            'question': substituted_question,
                            'search_results': [],
                            'qa_pairs': [],
                            'next_hop_global_entity_ids': [],
                            'step_num': step_idx + 1,
                            'chain_id': chain.chain_id
                        }
                    
                    # Create new chain
                    new_accumulated_evidence = chain.accumulated_evidence.copy()
                    if not is_final_step:
                        if show_intermediate_qa:
                            new_accumulated_evidence[substituted_question] = hop_result['qa_pairs']
                        else:
                            new_accumulated_evidence[substituted_question] = "No entities"
                    else:
                        new_accumulated_evidence[substituted_question] = hop_result['qa_pairs']
                    
                    new_chain = ReasoningChain(
                        chain_id=chain.chain_id,
                        steps=chain.steps + [hop_result],
                        final_entities=hop_result['next_hop_global_entity_ids'] if not is_final_step else [],
                        accumulated_evidence=new_accumulated_evidence
                    )
                    new_chains.append(new_chain)
                    continue
                
                # For each global entity ID in the chain's final entities, create a new branch
                for global_entity_id in chain.final_entities:
                    # Get entity name from the global entity ID for substitution
                    entity_name = self._get_entity_name_from_global_id(global_entity_id)
                    if not entity_name:
                        if self.verbose:
                            console.print(f"[yellow]Could not get entity name for {global_entity_id}, skipping[/yellow]")
                        continue
                    
                    # Substitute <ENTITY> placeholder with actual entity name
                    substituted_question = question_template.replace("<ENTITY>", entity_name)
                    if self.verbose:
                        console.print(f"[bold yellow]Chain {chain.chain_id} - Substituted Question:[/bold yellow] {substituted_question}")
                        console.print(f"[dim]  Template: {question_template}[/dim]")
                        console.print(f"[dim]  Entity: {entity_name} (ID: {global_entity_id})[/dim]")
                    
                    # Check if this is the final step
                    is_final_step = (step_idx + 1) == len(decomposed_questions)
                    
                    # Execute this hop with semantic search only if retrieval is needed
                    if requires_retrieval:
                        hop_result = self.execute_single_hop(
                            substituted_question, 
                            step_num=step_idx + 1, 
                            chain_id=chain.chain_id,
                            is_final_step=is_final_step,
                            final_top_k=15  # Keep top 15 Q&A pairs for final answer generation
                        )
                    else:
                        # Skip retrieval - just pass through entities and create a simple result
                        if self.verbose:
                            console.print(f"\n[bold cyan]Step {step_idx + 1} - Chain {chain.chain_id}:[/bold cyan] {substituted_question}")
                            console.print(f"[dim]Skipping retrieval (requires_retrieval=false) - reasoning question[/dim]")
                        
                        # For non-retrieval questions, we still want to maintain the chain structure
                        # For the final step with no retrieval, we don't need to propagate entities
                        # For intermediate steps, only pass the current entity (not all chain.final_entities)
                        hop_result = {
                            'question': substituted_question,
                            'search_results': [],
                            'qa_pairs': [],  # No Q&A pairs for reasoning-only questions
                            'next_hop_global_entity_ids': [] if is_final_step else [global_entity_id],  # Only pass current entity, not all
                            'step_num': step_idx + 1,
                            'chain_id': chain.chain_id
                        }
                    
                    # Create new chain with this hop's results
                    # Use shortened entity ID for chain naming to avoid overly long IDs
                    entity_short_id = global_entity_id.split("::")[-1] if "::" in global_entity_id else global_entity_id
                    
                    # For final steps, no more entities needed; for intermediate steps
                    if is_final_step:
                        final_entities_for_chain = []  # No more hops needed
                    else:
                        final_entities_for_chain = hop_result['next_hop_global_entity_ids'] 
                    
                    # Create new accumulated evidence dict by copying existing and adding new question
                    new_accumulated_evidence = chain.accumulated_evidence.copy()
                    
                    # Update the previous question's evidence with the entity we're actually using
                    # This happens when we're beyond the first hop\
                    if not show_intermediate_qa:
                        if step_idx > 0:
                            prev_questions = list(new_accumulated_evidence.keys())
                            if prev_questions:
                                last_question = prev_questions[-1]
                                # Update it to show which entity we're continuing with
                                new_accumulated_evidence[last_question] = entity_name
                    
                    if not is_final_step:
                        # For intermediate steps, store based on mode
                        if show_intermediate_qa:
                            # Store Q&A pairs for detailed evidence display
                            new_accumulated_evidence[substituted_question] = hop_result['qa_pairs']
                        else:
                            # Store placeholder - will be updated with entity name when we continue
                            new_accumulated_evidence[substituted_question] = "[To be determined]"
                    else:
                        # For final step, always store the Q&A pairs for reasoning
                        new_accumulated_evidence[substituted_question] = hop_result['qa_pairs']
                    
                    new_chain = ReasoningChain(
                        chain_id=f"{chain.chain_id}_{entity_short_id}",
                        steps=chain.steps + [hop_result],
                        final_entities=final_entities_for_chain,
                        accumulated_evidence=new_accumulated_evidence
                    )
                    new_chains.append(new_chain)
            
            chains = new_chains
            if self.verbose:
                console.print(f"[yellow]Step {step_idx + 1}: {len(chains)} active chains[/yellow]")
        
        return chains
    
    def generate_chains_with_entity_tracking(self, decomposed_questions: List[Dict[str, Any]], show_intermediate_qa: bool = True) -> List[ReasoningChain]:
        """Generate reasoning chains with proper entity tracking for indexed placeholders.
        
        Args:
            decomposed_questions: List of dicts with 'question', 'requires_retrieval', and potentially 'entity_refs' keys
            show_intermediate_qa: Whether to store detailed Q&A pairs for intermediate steps
            
        Returns:
            List of completed reasoning chains with proper entity substitution
        """
        if not decomposed_questions:
            return []
        
        # Parse entity references to understand question dependencies (modifies decomposed_questions in-place)
        self.parse_entity_references(decomposed_questions)
        
        if self.verbose:
            console.print(f"\n[bold magenta]Executing {len(decomposed_questions)} reasoning steps with entity tracking...[/bold magenta]")
        
        # Keep track of entities produced by each question for each chain
        chains = []
        
        # Execute first question(s) that don't depend on other entities
        for i, q_info in enumerate(decomposed_questions):
            question = q_info["question"]
            requires_retrieval = q_info["requires_retrieval"]
            
            # Check if this question has entity references
            if "entity_refs" in q_info:
                # This question depends on previous results, skip for now
                continue
            
            # This is an independent question, execute it
            if requires_retrieval:
                hop_result = self.execute_single_hop(question, step_num=i+1, chain_id=f"init_q{i+1}")
                entity_ids = hop_result['next_hop_global_entity_ids']
            else:
                hop_result = {
                    'question': question,
                    'search_results': [],
                    'qa_pairs': [],
                    'next_hop_global_entity_ids': [],
                    'step_num': i+1,
                    'chain_id': f"init_q{i+1}"
                }
                entity_ids = []
            
            # Create a chain for each entity found (or one chain if no entities)
            if entity_ids:
                for j, entity_id in enumerate(entity_ids):
                    entity_context = {f"Q{i+1}": entity_id}  # Track which entity this chain uses for Q1, Q2, etc.
                    chain = ReasoningChain(
                        chain_id=f"chain_q{i+1}_{j+1}",
                        steps=[hop_result],
                        final_entities=[entity_id],
                        accumulated_evidence={question: hop_result['next_hop_entities_with_qa'][j][1]}
                    )
                    chain.entity_context = entity_context  # Add entity tracking
                    chains.append(chain)
            else:
                # No entities found, create empty chain
                entity_context = {f"Q{i+1}": None}
                chain = ReasoningChain(
                    chain_id=f"chain_q{i+1}_1", 
                    steps=[hop_result],
                    final_entities=[],
                    accumulated_evidence={question: hop_result['qa_pairs']}
                )
                chain.entity_context = entity_context
                chains.append(chain)
        
        if self.verbose:
            console.print(f"[yellow]Generated {len(chains)} initial chains from independent questions[/yellow]")
        
        # Now process dependent questions in order
        new_chains = []
        for i, q_info in enumerate(decomposed_questions):
            question_template = q_info["question"]
            requires_retrieval = q_info["requires_retrieval"]
            
            # Skip if this question was already processed (no entity references)
            if "entity_refs" not in q_info:
                continue
                
            if not requires_retrieval:
                continue
                
            entity_refs = q_info.get("entity_refs", [])
            is_final_step = (i + 1) == len(decomposed_questions)
            
            if self.verbose:
                console.print(f"\n[bold magenta]Step {i + 1} Template:[/bold magenta] {question_template}")
                console.print(f"[dim]Entity references: {entity_refs}[/dim]")
            
            for chain in chains:
                # Check if we have all required entities for this question
                missing_entities = []
                for ref in entity_refs:
                    if ref not in getattr(chain, 'entity_context', {}):
                        missing_entities.append(ref)
                
                if missing_entities:
                    if self.verbose:
                        console.print(f"[yellow]Chain {chain.chain_id} missing entities {missing_entities}, skipping[/yellow]")
                    continue
                
                # Substitute entity placeholders with actual entity names
                substituted_question = question_template
                for ref in entity_refs:
                    entity_id = getattr(chain, 'entity_context', {}).get(ref)
                    if entity_id:
                        entity_name = self._get_entity_name_from_global_id(entity_id)
                        if entity_name:
                            placeholder = f"<ENTITY_{ref}>"
                            substituted_question = substituted_question.replace(placeholder, entity_name)
                        else:
                            if self.verbose:
                                console.print(f"[yellow]Could not get entity name for {entity_id}[/yellow]")
                            continue
                
                if self.verbose:
                    console.print(f"[bold yellow]Chain {chain.chain_id} - Substituted:[/bold yellow] {substituted_question}")
                
                # Execute this hop
                hop_result = self.execute_single_hop(
                    substituted_question,
                    step_num=i+1,
                    chain_id=chain.chain_id,
                    is_final_step=is_final_step,
                    final_top_k=5
                )
                
                # Handle chain branching based on entities found
                if hop_result['next_hop_global_entity_ids'] and not is_final_step:
                    # Create a separate chain for each entity found (branching)
                    for j, entity_id in enumerate(hop_result['next_hop_global_entity_ids']):
                        new_entity_context = getattr(chain, 'entity_context', {}).copy()
                        new_entity_context[f"Q{i+1}"] = entity_id
                        
                        new_accumulated_evidence = chain.accumulated_evidence.copy()
                        new_accumulated_evidence[substituted_question] = hop_result['qa_pairs'] if show_intermediate_qa else "Entity tracked"
                        
                        # Create unique chain ID for this branch
                        entity_short_id = entity_id.split("::")[-1] if "::" in entity_id else entity_id
                        new_chain_id = f"{chain.chain_id}_e{j+1}_{entity_short_id}"
                        
                        new_chain = ReasoningChain(
                            chain_id=new_chain_id,
                            steps=chain.steps + [hop_result],
                            final_entities=[entity_id],  # Each chain tracks its specific entity
                            accumulated_evidence=new_accumulated_evidence
                        )
                        new_chain.entity_context = new_entity_context
                        new_chains.append(new_chain)
                else:
                    # Final step or no entities found - create single chain
                    new_entity_context = getattr(chain, 'entity_context', {}).copy()
                    new_accumulated_evidence = chain.accumulated_evidence.copy()
                    new_accumulated_evidence[substituted_question] = hop_result['qa_pairs'] if  is_final_step else ""
             
                    
                    new_chain = ReasoningChain(
                        chain_id=chain.chain_id,
                        steps=chain.steps + [hop_result],
                        final_entities=hop_result['next_hop_global_entity_ids'] if not is_final_step else [],
                        accumulated_evidence=new_accumulated_evidence
                    )
                    new_chain.entity_context = new_entity_context
                    new_chains.append(new_chain)
            
        chains = new_chains if len(new_chains) > 0 else chains
        if self.verbose:
            console.print(f"[yellow]Step {i + 1}: {len(chains)} active chains[/yellow]")
        
        return chains

    def reason_over_chains(self, original_question: str, chains: List[ReasoningChain], decomposed_questions: List[Dict[str, Any]] = None, show_prompt: bool = True, show_intermediate_qa: bool = True) -> str:
        """Use LLM to reason over accumulated chains and generate final answer.
        
        Args:
            original_question: The original multi-hop question
            chains: List of completed reasoning chains with accumulated evidence
            decomposed_questions: The decomposed questions (with retrieval flags) used to generate chains
            show_prompt: Whether to display the full prompt sent to LLM
            show_intermediate_qa: Whether to show Q&A pairs for intermediate steps (instead of just entity names)
            
        Returns:
            Final answer generated by LLM
        """
        if not self.openai_client:
            return "[red]OpenAI client not available for final reasoning[/red]"
        
        if not chains:
            return "[yellow]No reasoning chains available for final answer[/yellow]"
        
        # Prepare context from all chains - simple approach
        context_parts = []
        context_parts.append(f"Question: {original_question}\n")
        context_parts.append("Here are different evidence chains to answer this question:\n")
        
        for i, chain in enumerate(chains, 1):
            # context_parts.append(f"EVIDENCE CHAIN {i} (followed entity path: {chain.chain_id}):")
            context_parts.append(f"EVIDENCE CHAIN {i}:")
            
            # # Debug: Show what questions were asked in this chain
            # if decomposed_questions:
            #     context_parts.append("  Questions asked:")
            #     for step in chain.steps:
            #         context_parts.append(f"    - {step['question']}")
            
            context_parts.append("  Evidence collected:")
            # Show evidences organized by decomposed questions using the new dict structure
            accumulated_items = list(chain.accumulated_evidence.items())
            
            # For intermediate questions (all except last): show ONLY the Q&A pair that justified the entity selection
            for idx, (question_asked, evidence_data) in enumerate(accumulated_items[:-1]):
                context_parts.append(f"    Question asked: {question_asked}")
                
                # Check if evidence_data is Q&A pairs (list) or entity name (string)
                if isinstance(evidence_data, dict) and evidence_data:
                    # Evidence is stored as one QA pair with metadata
                    context_parts.append(f"    QA pair from knowledge base that led to next entity:")
                    # Show only the Q&A pair(s) that justified this chain - should be 1 or very few
                    question_text = evidence_data['question']
                    answer_text = ", ".join(evidence_data.get('answer_names'))
                    context_parts.append(f"      Q: {question_text}")
                    context_parts.append(f"      A: {answer_text}")
                    context_parts.append(f"    → Following entity: {answer_text}")
                else:
                    # Evidence is stored as entity name (fast mode was used)
                    if isinstance(evidence_data, str) and evidence_data != "[To be determined]":
                        context_parts.append(f"    → Following entity: {evidence_data}")
                    else:
                        context_parts.append(f"    → Following entity: {evidence_data}")
                
                context_parts.append("")  # Blank line between questions within chain
            
            # For the last question: show top k evidences for final reasoning
            if accumulated_items:
                last_question, last_qa_pairs = accumulated_items[-1]
                context_parts.append(f"    Question asked: {last_question}")
                context_parts.append(f"    Evidence collected:")
                 
                if isinstance(last_qa_pairs, dict):
                    last_qa_pairs = [last_qa_pairs]

                # Show top 5 Q&A pairs for the last question
                for j, qa in enumerate(last_qa_pairs, 1):
                    question_text = qa['question']
                    answer_text = qa.get('answer_names', qa.get('answers', ''))
                    if isinstance(answer_text, list):
                        answer_text = ', '.join(str(a) for a in answer_text)
                    
                    context_parts.append(f"      {j}. Q: {question_text}")
                    context_parts.append(f"         A: {answer_text}")
                
                if not last_qa_pairs:
                    context_parts.append(" No evidence found for this question.")
                context_parts.append("")  # Blank line between questions within chain
            
            context_parts.append("")  # Blank line between chains
        
        context = "\n".join(context_parts)
        
        # Create final reasoning prompt with proper system explanation
        reasoning_prompt = f"""You are an agent designed to answer multi-hop questions using structured knowledge.

SYSTEM OVERVIEW:
The knowledge base has been processed into semantic workspaces containing question-answer pairs. To answer multi-hop questions, we decompose them into steps and follow entity paths through the knowledge.

METHODOLOGY:
1. We break the multi-hop question into single steps
2. For each step, we identify relevant entities and explore them
3. Each "Evidence Chain" below represents following a specific entity path
4. Most chains will be irrelevant - only some lead to the correct answer

YOUR TASK:
Identify which evidence chain(s) logically answer the question and extract the answer.


{context}

EVALUATION PROCESS:
1. Identify what information the question seeks
2. Determine which chains follow entities relevant to that information
3. Discard chains following irrelevant entities (e.g., for "mother's death", ignore chains about spouse/children)
4. Extract the answer from relevant chain(s)

RULES:
- Use ONLY the provided Q&A evidence
- Do NOT use external knowledge
- If no chain contains the answer, state "No answer found"

Please respond in the following format:

<reasoning>
Please reason step by step to arrive at the answer.
</reasoning>

<answer>
[Provide only the answer based solely on evidence given. Provide the answer in a single word or phrase.]
</answer>"""

        # Display the prompt if requested
        if show_prompt:
            console.print("\n[bold blue]Final LLM Prompt:[/bold blue]")
            console.print(Panel(reasoning_prompt, expand=False, title="Prompt to LLM", border_style="blue"))
            
            # Also count tokens if tiktoken is available
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                token_count = len(encoding.encode(reasoning_prompt))
                console.print(f"[dim]Prompt tokens: {token_count}[/dim]")
            except Exception:
                pass
        
        try:
            if self.verbose:
                console.print(f"\n[cyan]Performing final reasoning over {len(chains)} chains...[/cyan]")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided evidence. You MUST use ONLY the Q&A evidence given. Do NOT use any external knowledge."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[red]Error in final reasoning: {e}[/red]"
    
    def display_chains_summary(self, chains: List[ReasoningChain]):
        """Display a summary of the reasoning chains in a formatted table."""
        if not chains:
            console.print("[yellow]No reasoning chains to display[/yellow]")
            return
        
        console.print(f"\n[bold magenta]Generated {len(chains)} Reasoning Chains:[/bold magenta]")
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Chain ID", style="cyan", width=15)
        table.add_column("Steps", style="white", width=10)
        table.add_column("Final Entities", style="green", width=20)
        table.add_column("Evidence Count", style="blue", width=15)
        table.add_column("Path Summary", style="dim", width=50)
        
        for chain in chains:
            steps_count = str(len(chain.steps))
            final_entities = str(len(chain.final_entities))
            # Count total evidences across all questions in the dict
            total_evidence_count = sum(len(qa_pairs) for qa_pairs in chain.accumulated_evidence.values())
            evidence_count = str(total_evidence_count)
            
            # Create path summary
            path_parts = []
            for step in chain.steps:
                question = step['question']
                if len(question) > 25:
                    question = question[:22] + "..."
                path_parts.append(question)
            path_summary = " → ".join(path_parts)
            
            if len(path_summary) > 50:
                path_summary = path_summary[:47] + "..."
            
            table.add_row(
                chain.chain_id,
                steps_count,
                final_entities,
                evidence_count,
                path_summary
            )
        
        console.print(table)
    
    def ask_multihop_question(self, question: str, show_intermediate_qa: bool = True, verbose: bool = None) -> str:
        """Execute the complete multi-hop QA pipeline.
        
        Args:
            question: The multi-hop question to answer
            show_intermediate_qa: Whether to show Q&A pairs for intermediate steps (richer context)
            verbose: Whether to show detailed output (if None, uses instance default)
            
        Returns:
            Final answer after multi-hop reasoning
        """
        import time
        
        # Use method parameter or fall back to instance default
        show_verbose = verbose if verbose is not None else self.verbose
        
        if show_verbose:
            console.print(f"\n[bold blue]Multi-Hop Question:[/bold blue] {question}")
        
        # Start timing from the beginning of retrieval
        start_time = time.time()
        
        # Step 1: Decompose the question
        decomposed_questions = self.decompose_question(question)
        
        # Step 2: Check if we need entity tracking (indexed placeholders)
        has_indexed_entities = any("<ENTITY_Q" in q["question"] for q in decomposed_questions)
        
        if has_indexed_entities:
            if show_verbose:
                console.print("[cyan]Using entity tracking mode for indexed placeholders[/cyan]")
            chains = self.generate_chains_with_entity_tracking(decomposed_questions, show_intermediate_qa=show_intermediate_qa)
        else:
            if show_verbose:
                console.print("[cyan]Using standard chain generation[/cyan]")
            chains = self.generate_chains(decomposed_questions, show_intermediate_qa=show_intermediate_qa)
        
        # Step 3: Display chains summary (only if verbose)
        if show_verbose:
            self.display_chains_summary(chains)
        
        # Step 4: Final LLM reasoning
        final_answer = self.reason_over_chains(question, chains, decomposed_questions=decomposed_questions, show_prompt=show_verbose, show_intermediate_qa=show_intermediate_qa)
        
        # Calculate total time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Display timing information
        console.print(f"\n[bold yellow]⏱️ Time taken: {elapsed_time:.2f} seconds[/bold yellow]")
        
        return final_answer
    
    def run_interactive_mode(self):
        """Run interactive multi-hop QA mode."""
        console.print("\n[bold cyan]🔗 Multi-Hop Question Answering System[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a multi-hop question to get an answer")
        console.print("  - 'help' - Show this help")
        console.print("  - 'mode' - Toggle between entity names only (fast) and full Q&A pairs (detailed)")
        console.print("  - 'quit' or 'exit' - Exit")
        
        # Default mode: show only entity names for faster processing
        show_intermediate_qa = True
        console.print(f"[dim]Current mode: {'Detailed Q&A pairs' if show_intermediate_qa else 'Entity names only'}[/dim]")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]Multi-Hop Question[/bold yellow]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif user_input.lower() == 'help':
                    console.print("Ask a multi-hop question like:")
                    console.print("  - 'What is the birth year of the spouse of the director of Casablanca?'")
                    console.print("  - 'Where was the author of To Kill a Mockingbird born?'")
                    console.print("  - 'Which film was released first, Casablanca or The Godfather?'")
                
                elif user_input.lower() == 'mode':
                    show_intermediate_qa = not show_intermediate_qa
                    mode_desc = 'Detailed Q&A pairs' if show_intermediate_qa else 'Entity names only'
                    console.print(f"[green]Switched to: {mode_desc}[/green]")
                    if show_intermediate_qa:
                        console.print("[dim]Will show full Q&A evidence for all steps (more context, slower)[/dim]")
                    else:
                        console.print("[dim]Will show only entity names for intermediate steps (faster)[/dim]")
                
                else:
                    # Process as a multi-hop question
                    answer = self.ask_multihop_question(user_input, show_intermediate_qa=show_intermediate_qa)
                    console.print("\n[bold green]Final Answer:[/bold green]")
                    console.print(Panel(answer, expand=False))
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")


def main():
    """Main function - start interactive multi-hop QA mode."""
    console.print("\n[bold cyan]🔗 Multi-Hop Question Answering System[/bold cyan]")
    console.print("Initializing system...")
    
    # Initialize multi-hop QA system
    try:
        multihop_qa = MultiHopQA(num_documents=-1, verbose=True)  # Load 200 documents
    except Exception as e:
        console.print(f"[red]Error initializing system: {e}[/red]")
        return
    
    # Start interactive mode
    multihop_qa.run_interactive_mode()


if __name__ == "__main__":
    main()