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

console = Console()


@dataclass
class ReasoningChain:
    """Represents a single reasoning chain through multiple hops."""
    chain_id: str
    steps: List[Dict[str, Any]]
    final_entities: List[str]
    accumulated_evidence: Dict[str, List[Dict[str, Any]]]  # Dict mapping questions to their evidences
    

class MultiHopQA:
    """Simple multi-hop QA system using enhanced entity search."""
    
    def __init__(self, num_documents: int = 200):
        """Initialize multi-hop QA system.
        
        Args:
            num_documents: Number of documents to load for entity search
        """
        console.print("[bold blue]Initializing Multi-Hop QA System...[/bold blue]")
        
        # Initialize the enhanced entity searcher
        self.entity_searcher = EntitySearcher(num_documents)
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                console.print("[green]✓ OpenAI client initialized for decomposition and reasoning[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not initialize OpenAI client: {e}[/yellow]")
        
        console.print("[bold green]✓ Multi-Hop QA System ready[/bold green]")
    
    def decompose_question(self, question: str) -> List[str]:
        """Decompose a multi-hop question into single-hop questions with <ENTITY> placeholders.
        
        Args:
            question: The multi-hop question to decompose
            
        Returns:
            List of single-hop questions where first question is specific and subsequent use <ENTITY> placeholder
        """
        if not self.openai_client:
            console.print("[red]OpenAI client not available for question decomposition[/red]")
            return [question]  # Fallback to original question
        
        decomposition_prompt = f"""Break down this multi-hop question into a sequence of single-hop questions.
The FIRST question should keep the original specific entity/information from the question.
SUBSEQUENT questions should use <ENTITY> as a placeholder that will be replaced with entities found from previous steps.

Examples:

Question: "What is the birth year of the spouse of the director of Casablanca?"
Decomposition:
1. Who directed Casablanca?
2. Who was <ENTITY>'s spouse?
3. What is <ENTITY>'s birth year?

Question: "In which city was the university founded where the author of 'To Kill a Mockingbird' studied?"
Decomposition:
1. Who wrote 'To Kill a Mockingbird'?
2. Where did <ENTITY> study?
3. In which city was <ENTITY> founded?

Question: "What is the capital of the country where the author of '1984' was born?"
Decomposition:
1. Who wrote '1984'?
2. Where was <ENTITY> born?
3. What is the capital of <ENTITY>?

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
                temperature=0.3,
                max_tokens=300
            )
            
            decomposition_text = response.choices[0].message.content
            
            # Extract numbered questions from the response
            questions = []
            lines = decomposition_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Match patterns like "1. Question?" or "1) Question?" or "- Question?"
                if re.match(r'^[\d]+[\.\)]\s+', line) or line.startswith('- '):
                    # Remove the numbering/bullets
                    question = re.sub(r'^[\d]+[\.\)]\s+', '', line)
                    question = re.sub(r'^-\s+', '', question)
                    if question:
                        questions.append(question)
            
            if not questions:
                # Fallback: split by lines and take non-empty ones
                questions = [line.strip() for line in lines if line.strip() and not line.strip().endswith(':')]
            
            console.print(f"[cyan]Decomposed into {len(questions)} questions:[/cyan]")
            for i, q in enumerate(questions, 1):
                console.print(f"  {i}. {q}")
            
            return questions if questions else [question]
            
        except Exception as e:
            console.print(f"[red]Error in question decomposition: {e}[/red]")
            return [question]  # Fallback to original question
    
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
            console.print(f"[yellow]Embedding model not available, using first {min(top_k, len(qa_pairs))} Q&A pairs[/yellow]")
            return qa_pairs[:top_k]
        
        try:
            # Import required modules for embeddings (same as simple_entity_search)
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            def get_detailed_instruct(task_description: str, query: str) -> str:
                """Create instruction for Qwen embedding model."""
                return f'Instruct: {task_description}\nQuery: {query}'
            
            # Embed the query using the same approach as simple_entity_search
            task = 'Given a user question, create an embedding optimized for finding relevant question-answer pairs'
            instructed_query = get_detailed_instruct(task, query)
            query_outputs = self.entity_searcher.embedding_model.embed([instructed_query])
            query_embedding = np.array(query_outputs[0].outputs.embedding)
            
            console.print(f"[cyan]Reranking {len(qa_pairs)} Q&A pairs for multi-hop reasoning...[/cyan]")
            
            # Create text representations for each Q&A pair (question + answer)
            qa_texts = []
            for qa in qa_pairs:
                # Use answer_names if available, otherwise use answers
                answer_text = qa.get('answer_names', qa.get('answers', ''))
                if isinstance(answer_text, list):
                    answer_text = ', '.join(str(a) for a in answer_text)
                qa_text = f"{qa['question']} {answer_text}"
                qa_texts.append(qa_text)
            
            # Embed all Q&A pairs using the same task as simple_entity_search
            task = 'Given a question-answer pair, create an embedding that captures the semantic meaning for similarity comparison with user queries.'
            qa_embeddings = []
            
            for qa_text in qa_texts:
                instructed_qa = get_detailed_instruct(task, qa_text)
                try:
                    outputs = self.entity_searcher.embedding_model.embed([instructed_qa])
                    embedding = np.array(outputs[0].outputs.embedding)
                    qa_embeddings.append(embedding)
                except Exception as e:
                    console.print(f"[red]Error embedding Q&A pair: {e}[/red]")
                    # Use zero embedding as fallback
                    qa_embeddings.append(np.zeros_like(query_embedding))
            
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
            
            console.print(f"[green]Selected top {len(reranked_pairs)} Q&A pairs (scores: {reranked_pairs[0]['similarity_score']:.3f} - {reranked_pairs[-1]['similarity_score']:.3f})[/green]")
            
            return reranked_pairs
            
        except Exception as e:
            console.print(f"[yellow]Warning: Q&A reranking failed: {e}. Using original order.[/yellow]")
            return qa_pairs[:top_k]
    
    def _extract_unique_entities_from_reranked_qa(self, reranked_qa_pairs: List[Dict[str, Any]], max_entities: int = 5) -> List[str]:
        """Extract unique global entity IDs from reranked Q&A pairs.
        
        Args:
            reranked_qa_pairs: List of reranked Q&A pairs (most relevant first)
            max_entities: Maximum number of unique entities to extract
            
        Returns:
            List of unique global entity IDs from top-ranked Q&A pairs
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
                        unique_entities.append(global_id)
                        seen_entities.add(global_id)
                        
                        # Stop once we have enough unique entities
                        if len(unique_entities) >= max_entities:
                            console.print(f"[dim]Extracted {len(unique_entities)} unique entities from top-ranked Q&A pairs[/dim]")
                            return unique_entities
        
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
        console.print(f"\n[bold cyan]Step {step_num} - Chain {chain_id}:[/bold cyan] {question}")
        console.print(f"[dim]Semantic search across all documents: {question}[/dim]")
        
        # Always use semantic search to find diverse information across documents
        search_results = self.entity_searcher.search(
            query=question,
            top_k=10,  # Get top 10 entities
            verbose=False,  # Don't show tables for intermediate steps
            generate_answer=False,  # No LLM answer for intermediate steps
            qa_top_k=15  # Get top 15 Q&A pairs for reranking
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
            console.print(f"[green]Final step: Selected top {len(reranked_qa_pairs)} Q&A pairs for answer generation[/green]")
        else:
            # For intermediate steps: Rerank all Q&A pairs, then extract entities for next hop
            reranked_qa_pairs = self._rerank_qa_pairs_for_multihop(question, all_qa_pairs, top_k=-1)
            global_entity_ids_for_next_hop = self._extract_unique_entities_from_reranked_qa(reranked_qa_pairs, max_entities=5)
            console.print(f"[green]Intermediate step: Found {len(global_entity_ids_for_next_hop)} global entity IDs for next hop: {global_entity_ids_for_next_hop[:3]}{'...' if len(global_entity_ids_for_next_hop) > 3 else ''}[/green]")
        
        return {
            'question': question,
            'search_results': search_results,
            'qa_pairs': reranked_qa_pairs,  # Use reranked Q&A pairs for better quality
            'next_hop_global_entity_ids': global_entity_ids_for_next_hop,
            'step_num': step_num,
            'chain_id': chain_id
        }
    
    def generate_chains(self, decomposed_questions: List[str], show_intermediate_qa: bool = False) -> List[ReasoningChain]:
        """Generate and execute reasoning chains for multi-hop questions.
        
        Args:
            decomposed_questions: List of entity-agnostic decomposed questions
            show_intermediate_qa: Whether to store detailed Q&A pairs for intermediate steps
            
        Returns:
            List of completed reasoning chains
        """
        if not decomposed_questions:
            return []
        
        console.print(f"\n[bold magenta]Executing {len(decomposed_questions)} reasoning steps...[/bold magenta]")
        
        # Start with the first question - no entity focus needed
        first_question = decomposed_questions[0]
        first_hop_result = self.execute_single_hop(first_question, step_num=1, chain_id="init")
        
        # Initialize chains with global entity IDs from first hop
        chains = []
        for i, global_entity_id in enumerate(first_hop_result['next_hop_global_entity_ids']):  # Limit to top 5 entities
            # Initialize accumulated_evidence as dict with first question
            if not show_intermediate_qa:
                initial_evidence = {first_hop_result['question']: self._get_entity_name_from_global_id(global_entity_id)}
            else:
                initial_evidence = {first_hop_result['question']: first_hop_result['qa_pairs']}
            
            chain = ReasoningChain(
                chain_id=f"chain_{i+1}",
                steps=[first_hop_result],
                final_entities=[global_entity_id],  # Store global entity IDs
                accumulated_evidence=initial_evidence
            )
            chains.append(chain)
        
        console.print(f"[yellow]Generated {len(chains)} initial chains[/yellow]")
        
        # Execute remaining questions for each chain
        for step_idx in range(1, len(decomposed_questions)):
            question_template = decomposed_questions[step_idx]
            is_final_step_global = (step_idx + 1) == len(decomposed_questions)
            
            if is_final_step_global:
                console.print(f"\n[bold red]FINAL Step {step_idx + 1} Template:[/bold red] {question_template}")
                console.print(f"[dim]This is the final step - collecting Q&A pairs for answer generation[/dim]")
            else:
                console.print(f"\n[bold magenta]Step {step_idx + 1} Template:[/bold magenta] {question_template}")
            
            new_chains = []
            
            for chain in chains:
                # For each global entity ID in the chain's final entities, create a new branch
                for global_entity_id in chain.final_entities:
                    # Get entity name from the global entity ID for substitution
                    entity_name = self._get_entity_name_from_global_id(global_entity_id)
                    if not entity_name:
                        console.print(f"[yellow]Could not get entity name for {global_entity_id}, skipping[/yellow]")
                        continue
                    
                    # Substitute <ENTITY> placeholder with actual entity name
                    substituted_question = question_template.replace("<ENTITY>", entity_name)
                    console.print(f"[bold yellow]Chain {chain.chain_id} - Substituted Question:[/bold yellow] {substituted_question}")
                    console.print(f"[dim]  Template: {question_template}[/dim]")
                    console.print(f"[dim]  Entity: {entity_name} (ID: {global_entity_id})[/dim]")
                    
                    # Check if this is the final step
                    is_final_step = (step_idx + 1) == len(decomposed_questions)
                    
                    # Execute this hop with semantic search
                    hop_result = self.execute_single_hop(
                        substituted_question, 
                        step_num=step_idx + 1, 
                        chain_id=chain.chain_id,
                        is_final_step=is_final_step,
                        final_top_k=15  # Keep top 15 Q&A pairs for final answer generation
                    )
                    
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
            console.print(f"[yellow]Step {step_idx + 1}: {len(chains)} active chains[/yellow]")
        
        return chains
    
    def reason_over_chains(self, original_question: str, chains: List[ReasoningChain], decomposed_questions: List[str] = None, show_prompt: bool = True, show_intermediate_qa: bool = False) -> str:
        """Use LLM to reason over accumulated chains and generate final answer.
        
        Args:
            original_question: The original multi-hop question
            chains: List of completed reasoning chains with accumulated evidence
            decomposed_questions: The decomposed questions used to generate chains
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
            
            # For intermediate questions (all except last): show entity names or Q&A pairs based on mode and storage
            for idx, (question_asked, evidence_data) in enumerate(accumulated_items[:-1]):
                context_parts.append(f"    Question asked: {question_asked}")
                
                # Check if evidence_data is Q&A pairs (list) or entity name (string)
                if isinstance(evidence_data, list) and evidence_data:
                    # Evidence is stored as Q&A pairs (detailed mode was used during generation)
                    context_parts.append(f"    Evidence collected:")
                    # Show top 5 Q&A pairs for this intermediate step
                    for j, qa in enumerate(evidence_data[:5], 1):
                        question_text = qa['question']
                        answer_text = qa.get('answer_names', qa.get('answers', ''))
                        if isinstance(answer_text, list):
                            answer_text = ', '.join(str(a) for a in answer_text)
                        
                        context_parts.append(f"      {j}. Q: {question_text}")
                        context_parts.append(f"         A: {answer_text}")
                    
                    # For Q&A mode, we need to show which entity was used for continuation
                    # We can get this from the next question in the chain or from chain structure
                    context_parts.append(f"    → Evidence from this step used for continuation")
                else:
                    # Evidence is stored as entity name (fast mode was used)
                    if show_intermediate_qa and isinstance(evidence_data, str) and evidence_data != "[To be determined]":
                        # User wants detailed view but we only have entity names
                        context_parts.append(f"    → Continued with: {evidence_data}")
                    else:
                        # Show entity name or placeholder
                        context_parts.append(f"    → Continued with: {evidence_data}")
                
                context_parts.append("")  # Blank line between questions within chain
            
            # For the last question: show top 5 evidences for final reasoning
            if accumulated_items:
                last_question, last_qa_pairs = accumulated_items[-1]
                context_parts.append(f"    Question asked: {last_question}")
                context_parts.append(f"    Evidence collected:")
                
                # Show top 5 Q&A pairs for the last question
                for j, qa in enumerate(last_qa_pairs[:5], 1):
                    question_text = qa['question']
                    answer_text = qa.get('answer_names', qa.get('answers', ''))
                    if isinstance(answer_text, list):
                        answer_text = ', '.join(str(a) for a in answer_text)
                    
                    context_parts.append(f"      {j}. Q: {question_text}")
                    context_parts.append(f"         A: {answer_text}")
                
                if not last_qa_pairs:
                    context_parts.append("      No evidence found for this question.")
                context_parts.append("")  # Blank line between questions within chain
            
            context_parts.append("")  # Blank line between chains
        
        context = "\n".join(context_parts)
        
        # Create final reasoning prompt - simple approach
        reasoning_prompt = f"""{context}
Based on the evidence chains above, please answer: {original_question}

IMPORTANT: 
- Use ONLY the Q&A evidence provided above
- Do NOT use any external knowledge
- If the evidence doesn't contain the answer, say so

<answer>
Provide your answer based solely on the Q&A evidence shown above without any extra words, just the answer in few words.
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
    
    def ask_multihop_question(self, question: str, show_intermediate_qa: bool = False) -> str:
        """Execute the complete multi-hop QA pipeline.
        
        Args:
            question: The multi-hop question to answer
            show_intermediate_qa: Whether to show Q&A pairs for intermediate steps (richer context)
            
        Returns:
            Final answer after multi-hop reasoning
        """
        console.print(f"\n[bold blue]Multi-Hop Question:[/bold blue] {question}")
        
        # Step 1: Decompose the question
        decomposed_questions = self.decompose_question(question)
        
        # Step 2: Generate and execute reasoning chains
        chains = self.generate_chains(decomposed_questions, show_intermediate_qa=show_intermediate_qa)
        
        # Step 3: Display chains summary
        self.display_chains_summary(chains)
        
        # Step 4: Final LLM reasoning (with prompt display and decomposition context)
        final_answer = self.reason_over_chains(question, chains, decomposed_questions=decomposed_questions, show_prompt=True, show_intermediate_qa=show_intermediate_qa)
        
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
        show_intermediate_qa = False
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
        multihop_qa = MultiHopQA(num_documents=200)  # Load 200 documents
    except Exception as e:
        console.print(f"[red]Error initializing system: {e}[/red]")
        return
    
    # Start interactive mode
    multihop_qa.run_interactive_mode()


if __name__ == "__main__":
    main()