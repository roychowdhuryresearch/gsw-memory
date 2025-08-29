#!/usr/bin/env python3
"""
Beam Search Multi-Hop Question Answering System

An enhanced version that uses beam search to:
1. Maintain multiple hypothesis paths (beam_width=5)
2. Filter irrelevant Q&A pairs using embedding similarity
3. Score and prune paths based on relevance
4. Select the best path for final answer generation

This implementation explores multiple reasoning paths simultaneously
and uses similarity scoring to focus on relevant evidence.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
import numpy as np
import time

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from rich.console import Console
from rich.prompt import Prompt
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

# For similarity computation
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

console = Console()


class SimplifiedMultiHopQA:
    """Simplified multi-hop QA system with similarity filtering."""
    
    def __init__(
        self,
        num_documents: int = 200,
        verbose: bool = True,
        show_prompt: bool = False,
        similarity_threshold: float = 0.7
    ):
        """Initialize the simplified multi-hop QA system with similarity filtering.
        
        Args:
            num_documents: Number of documents to load
            verbose: Whether to show detailed output
            show_prompt: Whether to show the full LLM prompt
            similarity_threshold: Minimum similarity score for Q&A pairs
        """
        self.verbose = verbose
        self.show_prompt = show_prompt
        self.similarity_threshold = similarity_threshold
        
        if verbose:
            console.print("[bold blue]Initializing Simplified Multi-Hop QA System with Filtering...[/bold blue]")
            console.print(f"  Similarity threshold: {similarity_threshold}")
        
        # Initialize entity searcher (handles embeddings)
        self.entity_searcher = EntitySearcher(
            num_documents, 
            cache_dir="/home/yigit/codebase/gsw-memory/.gsw_cache",
            verbose=False  # Keep entity searcher quiet
        )
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                if verbose:
                    console.print("[green]✓ OpenAI client initialized[/green]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not initialize OpenAI: {e}[/yellow]")
        
        # Cache for question embeddings
        self.question_embedding_cache = {}
        
        if verbose:
            console.print("[bold green]✓ System ready[/bold green]")
    
    def _get_question_embedding(self, question: str) -> Optional[np.ndarray]:
        """Get embedding for a question, using cache when possible.
        
        Args:
            question: The question to embed
            
        Returns:
            Numpy array of the embedding or None if failed
        """
        # Check cache first
        if question in self.question_embedding_cache:
            return self.question_embedding_cache[question]
        
        # Use entity searcher's embedding capability
        embedding = self.entity_searcher._embed_query(question)
        
        if embedding is not None:
            self.question_embedding_cache[question] = embedding
        
        return embedding
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not SKLEARN_AVAILABLE:
            # Fallback to manual cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        # Use sklearn's cosine similarity
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        return float(similarity)
    
    def filter_qa_pairs_by_similarity(
        self, 
        qa_pairs: List[Dict[str, Any]], 
        query: str, 
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Filter Q&A pairs based on similarity to query.
        
        Args:
            qa_pairs: List of Q&A pair dictionaries
            query: Query to compare against
            threshold: Minimum similarity score (uses self.similarity_threshold if None)
            
        Returns:
            Filtered list of Q&A pairs with similarity scores added
        """
        if not qa_pairs:
            return []
        
        threshold = threshold or self.similarity_threshold
        
        # Check if all QA pairs already have similarity scores
        all_have_scores = all('similarity_score' in qa for qa in qa_pairs)
        
        # Only get query embedding if we need to calculate new similarities
        query_embedding = None
        if not all_have_scores:
            query_embedding = self._get_question_embedding(query)
            if query_embedding is None:
                if self.verbose:
                    console.print("[yellow]Could not embed query for filtering, returning all Q&A pairs[/yellow]")
                return qa_pairs
        
        # Calculate similarities for each Q&A pair
        scored_pairs = []
        for qa in qa_pairs:
            # Check if similarity score already exists (from reranking)
            if 'similarity_score' in qa:
                similarity = qa['similarity_score']
                # Use existing score directly
                if similarity >= threshold:
                    scored_pairs.append(qa)
            else:
                # Only calculate if score doesn't exist
                # Create text representation of Q&A pair
                qa_text = qa.get('question', '')
                if 'answer_names' in qa:
                    qa_text += ' ' + ', '.join(qa['answer_names'])
                elif 'answers' in qa:
                    qa_text += ' ' + str(qa['answers'])
                
                # Get embedding for Q&A pair
                qa_embedding = self._get_question_embedding(qa_text)
                if qa_embedding is None:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, qa_embedding)
                
                # Add similarity score to Q&A pair
                qa_with_score = qa.copy()
                qa_with_score['similarity_score'] = similarity
                
                if similarity >= threshold:
                    scored_pairs.append(qa_with_score)
        
        # Sort by similarity
        scored_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        if self.verbose:
            filtered_count = len(qa_pairs) - len(scored_pairs)
            if filtered_count > 0:
                console.print(f"[dim]Filtered out {filtered_count} Q&A pairs below threshold {threshold:.2f}[/dim]")
        
        return scored_pairs
    
    def score_question_relevance(self, question: str, original_query: str) -> float:
        """Score how relevant a question is to the original query.
        
        Args:
            question: The question to score
            original_query: The original user query
            
        Returns:
            Relevance score between 0 and 1
        """
        q_embedding = self._get_question_embedding(question)
        orig_embedding = self._get_question_embedding(original_query)
        
        if q_embedding is None or orig_embedding is None:
            return 0.5  # Default middle score if embeddings fail
        
        return self._calculate_similarity(q_embedding, orig_embedding)
    
    def substitute_entities(self, question_template: str, entities_by_question: Dict[str, List[str]]) -> List[str]:
        """Substitute entity placeholders with actual entity names.
        
        Args:
            question_template: Question with placeholders like <ENTITY> or <ENTITY_Q1>
            entities_by_question: Dict mapping Q1, Q2, etc. to entity names
            
        Returns:
            List of substituted questions (one per entity combination)
        """
        if "<ENTITY" not in question_template:
            return [question_template]
        
        substituted_questions = []
        
        # Handle simple <ENTITY> placeholder (use most recent question's entities)
        if "<ENTITY>" in question_template:
            # Find the most recent question number
            if entities_by_question:
                last_q = max(entities_by_question.keys(), key=lambda x: int(x[1:]))
                entities = entities_by_question.get(last_q, [])
                
                for entity_name in entities:
                    q = question_template.replace("<ENTITY>", entity_name)
                    substituted_questions.append(q)
        
        # Handle indexed placeholders like <ENTITY_Q1>, <ENTITY_Q2>
        else:
            # Find all entity references
            refs = re.findall(r'<ENTITY_Q(\d+)>', question_template)
            if refs:
                # For now, just substitute with the first entity from each referenced question
                # (In a more complex version, we could do cartesian product)
                q = question_template
                for ref in refs:
                    q_key = f"Q{ref}"
                    if q_key in entities_by_question and entities_by_question[q_key]:
                        # Use the first entity from this question
                        for entity in entities_by_question[q_key]:
                            q = question_template.replace(f"<ENTITY_Q{ref}>", entity)
                            substituted_questions.append(q)
        
        return substituted_questions if substituted_questions else [question_template]
    
    def is_question_referenced_in_future(self, current_index: int, decomposed: List[Dict[str, Any]]) -> bool:
        """Check if the current question index is referenced in any future questions.
        
        Args:
            current_index: Current question index (0-based)
            decomposed: List of all decomposed questions
            
        Returns:
            True if any future question references this one with <ENTITY_Q{}>
        """
        current_q_ref = f"<ENTITY_Q{current_index + 1}>"  # Q1 is index 0, so add 1
        
        # Check all subsequent questions (only those that require retrieval)
        for future_index in range(current_index + 1, len(decomposed)):
            future_q_info = decomposed[future_index]
            # Skip non-retrieval questions when checking for references
            if not future_q_info.get("requires_retrieval", True):
                continue
            
            future_question = future_q_info.get("question", "")
            if current_q_ref in future_question:
                return True
        
        return False
    
    def extract_entities_from_qa_pairs(self, qa_pairs: List[Dict[str, Any]], max_entities: int = 5) -> Tuple[List[str], List[str]]:
        """Extract unique entity names from Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A pairs with answer information
            max_entities: Maximum number of unique entities to extract
            
        Returns:
            Tuple of (unique entity names, evidence strings used)
        """
        unique_entities = []
        qa_pair_used = []
        seen = set()
        
        for qa in qa_pairs:
            # Get answer names (could be a list or string)
            answer_names = qa.get('answer_names', qa.get('answers', []))
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            
            for name in answer_names:
                if name and name not in seen:
                    unique_entities.append(name)
                    answer_text = ', '.join(qa.get('answer_names', qa.get('answer_ids', [])))
                    answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                    if answer_rolestates:
                        qa_pair_used.append(f"Q: {qa['question']} A: {answer_text} {answer_rolestates}")
                    else:
                        qa_pair_used.append(f"Q: {qa['question']} A: {answer_text}")
                    seen.add(name)
                    if len(unique_entities) >= max_entities:
                        return unique_entities, qa_pair_used
        
        return unique_entities, qa_pair_used
    
    def decompose_question(self, question: str) -> List[Dict[str, Any]]:
        """Decompose a multi-hop question into single-hop questions.
        
        Reuses the decomposition logic from the original implementation.
        """
        if not self.openai_client:
            return [{"question": question, "requires_retrieval": True}]
        
        decomposition_prompt = f"""Break down this multi-hop question into a sequence of single-hop questions.
The FIRST question should keep the original specific entity/information from the question.
SUBSEQUENT questions should use <ENTITY> as a placeholder if it requires answers from the previous step to form the question.

IMPORTANT: Avoid over-decomposition. Each question should extract meaningful entities (proper nouns like names, places), not single-word descriptors. Keep questions at an appropriate granularity level.

For each question, indicate whether it requires retrieval from the knowledge base, any question that requires factual information MUST require retrieval.
The only case where retreival is not required is if the question just requires comparison of responses from previous questions.

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
            
            # Parse the response
            questions = []
            lines = decomposition_text.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if re.match(r'^[\d]+[\.)\s]*Question:', line) or line.startswith('Question:'):
                    question_match = re.search(r'Question:\s*(.+)', line)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        
                        # Check for requires_retrieval flag
                        requires_retrieval = True  # Default
                        
                        if 'Requires retrieval:' in line:
                            retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', line, re.IGNORECASE)
                            if retrieval_match:
                                requires_retrieval = retrieval_match.group(1).lower() == 'true'
                        elif i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if 'Requires retrieval:' in next_line:
                                retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', next_line, re.IGNORECASE)
                                if retrieval_match:
                                    requires_retrieval = retrieval_match.group(1).lower() == 'true'
                                i += 1
                        
                        questions.append({
                            "question": question_text,
                            "requires_retrieval": requires_retrieval
                        })
                
                i += 1
            
            if self.verbose and questions:
                console.print(f"[cyan]Decomposed into {len(questions)} questions:[/cyan]")
                for i, q in enumerate(questions, 1):
                    retrieval_str = "✓" if q["requires_retrieval"] else "✗"
                    console.print(f"  {i}. {q['question']} [{retrieval_str}]")
            
            return questions if questions else [{"question": question, "requires_retrieval": True}]
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]Error in decomposition: {e}[/red]")
            return [{"question": question, "requires_retrieval": True}]
    
    def process_multihop_question(self, question: str, final_topk: int = 10) -> Dict[str, Any]:
        """Process a multi-hop question with simplified approach and similarity filtering.
        
        Args:
            question: The multi-hop question to answer
            final_topk: Maximum evidence items for final questions
            
        Returns:
            Dictionary containing the answer and collected evidence
        """
        start_time = time.time()
        
        if self.verbose:
            console.print(f"\n[bold blue]Processing: {question}[/bold blue]")
            console.print(f"[dim]Similarity threshold: {self.similarity_threshold}[/dim]")
        
        # Step 1: Decompose the question
        decomposed = self.decompose_question(question)
        
        if not decomposed:
            return {
                "question": question,
                "answer": "Failed to decompose question",
                "time_taken": time.time() - start_time
            }
        
        # Step 2: Initialize simple storage
        all_evidence = []  # Flat list of all Q&A pairs
        entities_by_question = {}  # Q1 -> ["Michael Curtiz"], Q2 -> ["Mildred Lewis"]
        
        # Step 3: Process each question sequentially
        for i, q_info in enumerate(decomposed):
            question_template = q_info["question"]
            requires_retrieval = q_info["requires_retrieval"]
            
            if not requires_retrieval:
                if self.verbose:
                    console.print(f"[dim]Skipping Q{i+1} (no retrieval needed): {question_template}[/dim]")
                continue
            
            # Check if this question is referenced in future questions
            is_referenced = self.is_question_referenced_in_future(i, decomposed)
            
            # Substitute entities from PREVIOUS questions if needed
            actual_questions = self.substitute_entities(question_template, entities_by_question)
            
            if self.verbose:
                mode = "entity extraction" if is_referenced else "Q&A collection"
                console.print(f"\n[cyan]Q{i+1}: Processing {len(actual_questions)} question(s) [{mode}][/cyan]")
            
            # Collect evidence for this question level
            question_evidence = []
            question_entities = []
            
            for actual_q in actual_questions:
                if self.verbose:
                    console.print(f"  → {actual_q}")
                
                # Search and collect evidence
                qa_pairs = self.search_and_collect_evidence(actual_q)
                
                if is_referenced:
                    # First filter Q&A pairs by similarity
                    filtered_qa = self.filter_qa_pairs_by_similarity(
                        qa_pairs,
                        actual_q,
                        threshold=self.similarity_threshold
                    )
                    
                    # Extract entities from FILTERED results only
                    entities, qa_pair_used = self.extract_entities_from_qa_pairs(filtered_qa)
                    question_entities.extend(entities)
                    
                    # Use the same qa_pair_used as evidence (already formatted by extract_entities_from_qa_pairs)
                    question_evidence.extend(qa_pair_used)
                else:
                    # Apply similarity filtering for final evidence
                    filtered_qa = self.filter_qa_pairs_by_similarity(
                        qa_pairs,
                        actual_q,
                        threshold=self.similarity_threshold
                    )
                    
                    # Format the Q&A pairs as evidence strings
                    final_question_evidence = set()
                    for qa in filtered_qa:
                        if len(final_question_evidence) >= final_topk:
                            break
                        q_text = qa.get('question', '')
                        answer_names = qa.get('answer_names', qa.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        answer_text = ', '.join(str(name) for name in answer_names if name)
                        answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                        if q_text and answer_text:
                            evidence_str = f"Q: {q_text} A: {answer_text}"
                            if answer_rolestates:
                                evidence_str += f" {answer_rolestates}"
                            final_question_evidence.add(evidence_str)
                    question_evidence.extend(list(final_question_evidence))
            
            # Store results
            all_evidence.extend(question_evidence)
            # Only store entities if they'll be referenced in future
            if is_referenced:
                entities_by_question[f"Q{i+1}"] = list(set(question_entities))  # Unique entities
            
            if self.verbose:
                if is_referenced:
                    console.print(f"  [green]Found {len(question_evidence)} Q&A pairs, {len(entities_by_question[f'Q{i+1}'])} unique entities[/green]")
                else:
                    console.print(f"  [green]Collected {len(question_evidence)} Q&A pairs (terminal question)[/green]")
        
        # Final deduplication of evidence
        all_evidence = list(set(all_evidence))
        
        # Step 4: Generate final answer
        answer = self.generate_answer(question, all_evidence, decomposed)
        
        elapsed_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "evidence_count": len(all_evidence),
            "time_taken": elapsed_time,
            "decomposed_questions": decomposed,
            "entities_found": entities_by_question,
            "final_prompt": getattr(self, '_last_prompt', None)  # Include the last prompt used
        }
    
    
    
    def search_and_collect_evidence(self, question: str, top_k_entities: int = 10, top_k_qa: int = 15) -> List[Dict[str, Any]]:
        """Search for evidence using the entity searcher.
        
        Args:
            question: Question to search for
            top_k_entities: Number of entities to retrieve
            top_k_qa: Number of Q&A pairs after reranking
            
        Returns:
            List of Q&A pairs with metadata
        """
        # Search for relevant entities
        search_results = self.entity_searcher.search(
            query=question,
            top_k=top_k_entities,
            verbose=False
        )
        
        # Extract Q&A pairs
        all_qa_pairs = []
        for entity, score in search_results:
            qa_pairs = entity.get('qa_pairs', [])
            for qa in qa_pairs:
                qa_with_context = qa.copy()
                qa_with_context['source_entity'] = entity['name']
                qa_with_context['source_entity_id'] = entity['id']
                qa_with_context['entity_score'] = score
                all_qa_pairs.append(qa_with_context)
        
        # Rerank Q&A pairs
        if hasattr(self.entity_searcher, '_rerank_qa_pairs') and all_qa_pairs:
            reranked = self.entity_searcher._rerank_qa_pairs(question, all_qa_pairs, top_k=top_k_qa)
            return reranked
        
        return all_qa_pairs[:top_k_qa]
    
    
    
    
    
    def generate_answer(
        self, 
        original_question: str,
        all_evidence: List[str],
        decomposed_questions: List[Dict[str, Any]] = None
    ) -> str:
        """Generate final answer using collected evidence.
        
        Args:
            original_question: Original user question
            evidence: List of evidence strings
            decomposed_questions: The decomposed questions (for context)
            
        Returns:
            Generated answer string
        """
        if not self.openai_client:
            return "OpenAI client not available for answer generation"
        
        if not all_evidence:
            return "No evidence found to answer the question"
        
        # Deduplicate evidence  
        unique_evidence = list(set(all_evidence))
        
        # Format evidence
        evidence_text = '\n'.join(unique_evidence)
        
        # Create prompt
        prompt = f"""Answer the following multi-hop question using ONLY the provided evidence.

Question: {original_question}

Available Evidence (Q&A pairs from knowledge base):
{evidence_text}

Instructions:
1. Use ONLY the Q&A pairs provided above
2. Be sure to check all the Q&A pairs for the answer
3. Do NOT use any external knowledge
4. If the evidence doesn't contain the answer, say "Cannot determine from available evidence"
5. Be concise and direct

Please respond in the following format:

<reasoning>
Reasoning about the question and the evidence.
</reasoning>
<answer>
Only the final answer, respond with a single word or phrase only.
</answer>
"""
        
        # Store prompt for debugging
        self._last_prompt = prompt
        
        # Show prompt if requested
        if self.show_prompt:
            console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
            console.print("[bold cyan]FULL LLM PROMPT:[/bold cyan]")
            console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")
            console.print(Panel(prompt, expand=False, border_style="blue"))
            
            # Count tokens if tiktoken available
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                token_count = len(encoding.encode(prompt))
                console.print(f"\n[bold yellow]📊 Prompt tokens: {token_count}[/bold yellow]")
            except:
                pass
            
            console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")
        
        try:
            if self.verbose:
                console.print(f"\n[cyan]Generating answer from {len(all_evidence)} Q&A pairs...[/cyan]")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions using only provided evidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def run_interactive_mode(self):
        """Run interactive question-answering mode."""
        console.print("\n[bold cyan]🔍 Multi-Hop QA System[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a multi-hop question to get an answer")
        console.print("  - 'threshold <value>' - Set similarity threshold (current: {:.2f})".format(self.similarity_threshold))
        console.print("  - 'prompt on/off' - Toggle showing the full LLM prompt")
        console.print("  - 'verbose on/off' - Toggle detailed output")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' or 'exit' - Exit\n")
        
        console.print(f"[dim]Current settings: verbose={self.verbose}, show_prompt={self.show_prompt}[/dim]")
        console.print(f"[dim]Similarity threshold: {self.similarity_threshold}[/dim]")
        
        while True:
            try:
                user_input = Prompt.ask("[bold yellow]Question[/bold yellow]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif user_input.lower() == 'help':
                    console.print("Ask multi-hop questions like:")
                    console.print("  - 'What is the birth year of the spouse of the director of Casablanca?'")
                    console.print("  - 'Where was the author of To Kill a Mockingbird born?'")
                    console.print("  - 'Which film was released first, Dune or The Dark Knight?'")
                
                elif user_input.lower().startswith('threshold '):
                    try:
                        self.similarity_threshold = float(user_input.split()[1])
                        console.print(f"[green]✓ Similarity threshold set to {self.similarity_threshold}[/green]")
                    except:
                        console.print("[yellow]Invalid threshold. Use 'threshold <float>'[/yellow]")
                
                elif user_input.lower().startswith('prompt '):
                    setting = user_input.lower().split()[1]
                    if setting == 'on':
                        self.show_prompt = True
                        console.print("[green]✓ Prompt display enabled[/green]")
                    elif setting == 'off':
                        self.show_prompt = False
                        console.print("[green]✓ Prompt display disabled[/green]")
                    else:
                        console.print("[yellow]Use 'prompt on' or 'prompt off'[/yellow]")
                
                elif user_input.lower().startswith('verbose '):
                    setting = user_input.lower().split()[1]
                    if setting == 'on':
                        self.verbose = True
                        console.print("[green]✓ Verbose mode enabled[/green]")
                    elif setting == 'off':
                        self.verbose = False
                        console.print("[green]✓ Verbose mode disabled[/green]")
                    else:
                        console.print("[yellow]Use 'verbose on' or 'verbose off'[/yellow]")
                
                else:
                    # Process the question using sequential similarity filtering
                    result = self.process_multihop_question(user_input)
                    
                    # Display results
                    console.print(f"\n[bold green]Answer:[/bold green]")
                    console.print(Panel(result['answer'], expand=False))
                    
                    console.print(f"\n[dim]Statistics:[/dim]")
                    console.print(f"  Evidence used: {result['evidence_count']} Q&A pairs")
                    console.print(f"  Time taken: {result['time_taken']:.2f} seconds")
                    console.print(f"  Questions processed: {len(result.get('questions_processed', []))}")
                    
                    if self.verbose and result.get('questions_processed'):
                        console.print(f"\n[dim]Processing path:[/dim]")
                        for i, q in enumerate(result['questions_processed'][:5], 1):
                            console.print(f"  {i}. {q[:80]}...")
                    
                    if self.verbose and result.get('entities_found'):
                        console.print(f"\n[dim]Entities discovered:[/dim]")
                        for q_num, entities in result['entities_found'].items():
                            if entities:
                                console.print(f"  {q_num}: {', '.join(entities[:3])}{'...' if len(entities) > 3 else ''}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    console.print("\n[bold cyan]🔍 Multi-Hop QA System[/bold cyan]")
    console.print("Initializing...")
    
    try:
        qa_system = SimplifiedMultiHopQA(num_documents=-1, verbose=True)
        qa_system.run_interactive_mode()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())