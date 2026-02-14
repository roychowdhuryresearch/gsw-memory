"""
Entity matching strategies and indices for the GSW reconciler.

This module provides different approaches for indexing and matching entities
during reconciliation, with each strategy paired with its corresponding index.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from bespokelabs import curator

try:
    import faiss
    import numpy as np
    from langchain_voyageai import VoyageAIEmbeddings

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    faiss = None
    np = None
    VoyageAIEmbeddings = None

from ...prompts.reconciler_prompts import (
    EntityVerificationPrompts,
    QuestionResolutionPrompts,
)
from ..models import EntityNode, GSWStructure


def format_entity_string(entity: EntityNode) -> str:
    """Format entity as a string for LLM verification."""
    entity_string = f"Entity: {entity.name}"
    for role in entity.roles:
        entity_string += f" Role: {role.role} States: {', '.join(role.states)}"
    return entity_string


# Abstract Base Classes


class EntityIndex(ABC):
    """Abstract base class for entity indices."""

    @abstractmethod
    def add_entities(self, entities: List[EntityNode], batch_size: int = 32):
        """Add new entities to the index."""
        pass


class MatchingStrategy(ABC):
    """Abstract base class for entity matching strategies."""

    @abstractmethod
    def reconcile_entities(
        self, new_entities: List[EntityNode], entity_index: EntityIndex
    ) -> List[Tuple[EntityNode, EntityNode]]:
        """Reconcile entities between new entities and existing entities."""
        pass

    @abstractmethod
    def reconcile_verb_phrases(
        self, new_gsw: GSWStructure, global_memory: GSWStructure
    ) -> Set[str]:
        """Reconcile verb phrases between new GSW and global memory."""
        pass

    @abstractmethod
    def resolve_questions(
        self,
        new_chunk_text: Optional[str],
        matched_old_entity_ids: Set[str],
        global_memory: GSWStructure,
        new_gsw_entities: List[EntityNode],
        chunk_id: Optional[str] = None,
    ) -> None:
        """Resolve unanswered questions using new chunk text."""
        pass


# LLM Components for Question Resolution


class QuestionResolver(curator.LLM):
    """Curator class for resolving unanswered questions."""

    return_completions_object = True

    def prompt(self, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create prompt for resolving questions using LLM."""
        new_chunk_text = input_data.get("new_chunk_text", "[Error: Missing chunk text]")
        candidate_list_str = input_data.get(
            "candidate_list_str", "[Error: Missing candidate questions string]"
        )
        new_entity_manifest = input_data.get(
            "new_entity_manifest", "[Error: Missing new entity manifest]"
        )

        user_prompt = QuestionResolutionPrompts.USER_PROMPT_TEMPLATE.format(
            new_chunk_text=new_chunk_text,
            candidate_list_str=candidate_list_str,
            new_entity_manifest=new_entity_manifest,
        )

        return [
            {"role": "system", "content": QuestionResolutionPrompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """Parse the LLM response to extract the JSON list of answers."""
        try:
            content = response["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed_response = json.loads(content)
            return {"parsed_response": parsed_response}
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"parsed_response": []}


class EntityVerifier(curator.LLM):
    """Curator class for verifying entity similarity."""

    return_completions_object = True

    def prompt(self, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create prompt for verifying entity similarity."""
        pair_descriptions = input_data.get("pair_descriptions", [])

        user_prompt = EntityVerificationPrompts.USER_PROMPT_TEMPLATE.format(
            pair_descriptions="\n".join(pair_descriptions)
        )

        return [
            {"role": "system", "content": EntityVerificationPrompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """Parse LLM response to extract verification results."""
        try:
            content = response["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            verification_results = json.loads(content)
            return {"parsed_response": verification_results}
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return {"parsed_response": {}}


# Exact Match Strategy + Index


class ExactMatchEntityIndex(EntityIndex):
    """Maintains entities and looks up by exact name match."""

    def __init__(self):
        self.entity_by_name: Dict[str, List[EntityNode]] = {}
        self.entity_by_id: Dict[str, EntityNode] = {}

    def add_entities(self, entities: List[EntityNode], batch_size: int = 32):
        """Add new entities to the index."""
        if not entities:
            return

        for entity in entities:
            # Store by ID
            self.entity_by_id[entity.id] = entity

            # Clean entity name for matching
            cleaned_entity_to_check = re.sub(r"[^a-zA-Z0-9\s]", "", entity.name).lower()

            # Store by name (multiple entities might have the same name)
            if cleaned_entity_to_check not in self.entity_by_name:
                self.entity_by_name[cleaned_entity_to_check] = []
            self.entity_by_name[cleaned_entity_to_check].append(entity)

    def get_entities_by_name(self, name: str) -> List[EntityNode]:
        """Get all entities with exactly the given name."""
        return self.entity_by_name.get(name, [])

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityNode]:
        """Get entity by its ID."""
        return self.entity_by_id.get(entity_id)


class ExactMatchStrategy(MatchingStrategy):
    """Strategy for matching entities by exact name and verifying roles/states."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict[str, float]] = None,
    ):
        """Initialize with model name and generation parameters."""
        self.model_name = model_name
        self.generation_params = generation_params or {
            "temperature": 0.0,
        }

        # Initialize LLM components
        self.question_resolver = QuestionResolver(
            model_name=model_name, generation_params=self.generation_params
        )

    def reconcile_entities(
        self, new_entities: List[EntityNode], entity_index: EntityIndex
    ) -> List[Tuple[EntityNode, EntityNode]]:
        """Find entities with exact name matches."""
        if not isinstance(entity_index, ExactMatchEntityIndex):
            raise TypeError("ExactMatchStrategy requires an ExactMatchEntityIndex")

        candidate_pairs = []

        # Find exact name matches
        for new_entity in new_entities:
            cleaned_entity_to_check = re.sub(
                r"[^a-zA-Z0-9\s]", "", new_entity.name
            ).lower()
            old_entities = entity_index.get_entities_by_name(cleaned_entity_to_check)

            # Don't match entities from the same chunk
            for old_entity in old_entities:
                if old_entity.chunk_id != new_entity.chunk_id:
                    candidate_pairs.append((new_entity, old_entity))

        return candidate_pairs

    def reconcile_verb_phrases(
        self, new_gsw: GSWStructure, global_memory: GSWStructure
    ) -> Set[str]:
        """Reconcile verb phrases based on exact verb + argument match."""

        def _get_first_word(text):
            """Helper to safely get the first word of a string."""
            return text.split(maxsplit=1)[0] if text and isinstance(text, str) else ""

        matched_new_verb_ids = set()
        verb_matches_to_process = []

        # Phase 1: Find potential matches
        for new_vp in new_gsw.verb_phrase_nodes:
            best_match_old_vp = None
            for old_vp in global_memory.verb_phrase_nodes:
                # Condition 1: Exact verb/phrase match
                if new_vp.phrase == old_vp.phrase:
                    # Condition 2: Argument match (first word + shared entity)
                    found_arg_match = False
                    for new_q in new_vp.questions:
                        new_q_first_word = _get_first_word(new_q.text)
                        if not new_q_first_word:
                            continue

                        for old_q in old_vp.questions:
                            old_q_first_word = _get_first_word(old_q.text)
                            if new_q_first_word == old_q_first_word:
                                # Check for shared non-"None" answers
                                new_answers_set = {
                                    a for a in new_q.answers if a != "None"
                                }
                                old_answers_set = {
                                    a for a in old_q.answers if a != "None"
                                }
                                if new_answers_set & old_answers_set:
                                    found_arg_match = True
                                    break
                        if found_arg_match:
                            break

                    if found_arg_match:
                        best_match_old_vp = old_vp
                        if best_match_old_vp.id != new_vp.id:
                            break
                        else:
                            best_match_old_vp = None

            if best_match_old_vp:
                matched_new_verb_ids.add(new_vp.id)
                verb_matches_to_process.append((new_vp, best_match_old_vp))

        # Phase 2: Execute merges
        for new_vp, old_vp in verb_matches_to_process:
            existing_question_tuples = {
                (q.text, tuple(sorted(q.answers))) for q in old_vp.questions
            }
            questions_to_add = []

            for new_q in new_vp.questions:
                matching_old_qs = [q for q in old_vp.questions if q.text == new_q.text]

                if matching_old_qs:
                    updated_existing_none = False
                    needs_adding_as_new = True

                    for old_q in matching_old_qs:
                        is_old_none = old_q.answers == ["None"]
                        is_new_none = new_q.answers == ["None"]

                        if (
                            is_old_none
                            and not is_new_none
                            and not updated_existing_none
                        ):
                            # Update the first found "None" answer
                            old_q.answers = new_q.answers
                            old_q.chunk_id = new_q.chunk_id
                            updated_existing_none = True
                            needs_adding_as_new = False
                            existing_question_tuples.add(
                                (old_q.text, tuple(sorted(old_q.answers)))
                            )
                            break

                    # Check if we need to add as new question
                    if not updated_existing_none:
                        is_new_none = new_q.answers == ["None"]
                        is_different_from_all = True

                        if not is_new_none:
                            for old_q in matching_old_qs:
                                if old_q.answers != ["None"] and sorted(
                                    old_q.answers
                                ) == sorted(new_q.answers):
                                    is_different_from_all = False
                                    needs_adding_as_new = False
                                    break
                        else:
                            needs_adding_as_new = False

                        if (
                            needs_adding_as_new
                            and not is_new_none
                            and is_different_from_all
                        ):
                            new_q_tuple = (new_q.text, tuple(sorted(new_q.answers)))
                            if new_q_tuple not in existing_question_tuples and not any(
                                q.text == new_q.text
                                and tuple(sorted(q.answers)) == new_q_tuple[1]
                                for q in questions_to_add
                            ):
                                questions_to_add.append(new_q)
                                existing_question_tuples.add(new_q_tuple)
                else:
                    # Question text is new, add it
                    new_q_tuple = (new_q.text, tuple(sorted(new_q.answers)))
                    if not any(
                        q.text == new_q.text
                        and tuple(sorted(q.answers)) == new_q_tuple[1]
                        for q in questions_to_add
                    ):
                        questions_to_add.append(new_q)
                        existing_question_tuples.add(new_q_tuple)

            # Add all collected new questions to the old verb phrase
            old_vp.questions.extend(questions_to_add)

        return matched_new_verb_ids

    def resolve_questions(
        self,
        new_chunk_text: Optional[str],
        matched_old_entity_ids: Set[str],
        global_memory: GSWStructure,
        new_gsw_entities: List[EntityNode],
        chunk_id: Optional[str] = None,
    ) -> None:
        """Identify and resolve unanswered questions linked to matched entities."""
        import os

        if not new_chunk_text or not matched_old_entity_ids:
            return

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Build reverse index: entity_id -> list of VP indices containing it
        entity_to_vp_indices = {}
        for vp_idx, vp in enumerate(global_memory.verb_phrase_nodes):
            vp_entities = set()
            for q in vp.questions:
                vp_entities.update(a for a in q.answers if a != "None")
            for entity_id in vp_entities:
                if entity_id not in entity_to_vp_indices:
                    entity_to_vp_indices[entity_id] = []
                entity_to_vp_indices[entity_id].append(vp_idx)

        # Find relevant VPs and collect unanswered questions
        vps_with_unanswered = {}
        for entity_id in matched_old_entity_ids:
            if entity_id in entity_to_vp_indices:
                for vp_idx in entity_to_vp_indices[entity_id]:
                    vp = global_memory.verb_phrase_nodes[vp_idx]

                    answered_q_list = []
                    unanswered_q_list = []
                    has_unanswered = False

                    for q in vp.questions:
                        if q.answers == ["None"]:
                            unanswered_q_list.append(q)
                            has_unanswered = True
                        else:
                            answered_q_list.append(q)

                    if has_unanswered:
                        vps_with_unanswered[vp.id] = {
                            "phrase": vp.phrase,
                            "answered": answered_q_list,
                            "unanswered": unanswered_q_list,
                        }

        if not vps_with_unanswered:
            return

        # Format data for LLM call
        entity_id_to_name_map = {e.id: e.name for e in global_memory.entity_nodes}

        def format_prompt_answer_list(answers):
            formatted = []
            for ans in answers:
                name = entity_id_to_name_map.get(ans)
                if name:
                    formatted.append(name)
                else:
                    formatted.append(ans)
            return formatted

        # Format unanswered questions
        unanswered_prompt_list = []
        for vp_id, data in vps_with_unanswered.items():
            vp_context_str = ""
            if data["answered"]:
                answered_qas = []
                for ans_q in data["answered"]:
                    formatted_ans = format_prompt_answer_list(ans_q.answers)
                    answered_qas.append(f"    - {ans_q.text} -> {formatted_ans}")
                if answered_qas:
                    vp_context_str = (
                        f"  Context from VP '{data['phrase']}':\n"
                        + "\n".join(answered_qas)
                    )

            for q in data["unanswered"]:
                q_context = (
                    f"Question ID: {q.id}\n  Question Text: {q.text}\n{vp_context_str}"
                )
                unanswered_prompt_list.append(q_context)

        candidate_list_str = "\n---\n".join(unanswered_prompt_list)

        # Format new entity manifest
        new_entity_manifest_parts = [
            f"- {e.name} (ID: {e.id})" for e in new_gsw_entities
        ]
        new_entity_manifest = "\n".join(new_entity_manifest_parts)
        if not new_entity_manifest:
            new_entity_manifest = "(None)"

        # Prepare input for LLM
        input_data = {
            "new_chunk_text": new_chunk_text,
            "candidate_list_str": candidate_list_str,
            "new_entity_manifest": new_entity_manifest,
        }

        # Log the prompt
        log_file = f"logs/prompt_log_{chunk_id}.txt"
        prompt_messages = self.question_resolver.prompt(input_data)
        with open(log_file, "w") as f:
            f.write("=== PROMPT ===\n")
            for message in prompt_messages:
                f.write(f"Role: {message['role']}\n")
                f.write(f"Content: {message['content']}\n\n")

        try:
            # Call the LLM
            llm_results = self.question_resolver([input_data])

            # Log the response
            with open(log_file, "a") as f:
                f.write("=== RESPONSE ===\n")
                f.write(f"Parsed: {llm_results.dataset[0]['parsed_response']}\n\n")

            # Process the results and update global memory
            resolved_answers_list = llm_results.dataset[0]["parsed_response"]

            if resolved_answers_list and isinstance(resolved_answers_list, list):

                def find_question_by_id(question_id):
                    """Find a question by its ID in the global memory."""
                    for vp in global_memory.verb_phrase_nodes:
                        for q in vp.questions:
                            if q.id == question_id:
                                return vp, q
                    return None, None

                questions_updated = 0
                for answer_dict in resolved_answers_list:
                    question_id = answer_dict.get("question_id")
                    answer_text = answer_dict.get("answer_text")
                    answer_entity_id = answer_dict.get("answer_entity_id")

                    if not question_id:
                        continue

                    vp_node, question_node = find_question_by_id(question_id)
                    if question_node:
                        if answer_entity_id:
                            # Verify the entity exists in global memory
                            entity_exists = any(
                                e.id == answer_entity_id
                                for e in global_memory.entity_nodes
                            )
                            if entity_exists:
                                question_node.answers = [answer_entity_id]
                            else:
                                question_node.answers = [f"TEXT:{answer_text}"]
                        elif answer_text:
                            question_node.answers = [f"TEXT:{answer_text}"]
                        else:
                            question_node.answers = ["None"]

                        if chunk_id:
                            question_node.chunk_id = chunk_id

                        questions_updated += 1

                if questions_updated > 0:
                    print(f"Updated {questions_updated} questions with new answers.")

        except Exception as e:
            print(f"Error during question resolution: {e}")
            with open(log_file, "a") as f:
                f.write("=== ERROR ===\n")
                f.write(f"{str(e)}\n")


# Embedding Match Strategy + Index


class EmbeddingEntityIndex(EntityIndex):
    """Maintains an index of entity embeddings for efficient similarity search."""

    def __init__(self, embedding_dim: int = 1024, gpu_device: int = 0):
        """Initialize the embedding index.

        Args:
            embedding_dim: Dimension of the embeddings (default: 1024)
            gpu_device: GPU device ID for FAISS GPU acceleration (default: 0)
        """
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not available. Install with: "
                "pip install faiss-gpu langchain-voyageai"
            )

        self.embedding_dim = embedding_dim
        self.gpu_device = gpu_device
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, EntityNode] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.embedding_model = VoyageAIEmbeddings(model="voyage-3")

        # Initialize GPU resources for FAISS
        self.gpu_resources = None
        try:
            self.gpu_resources = faiss.StandardGpuResources()
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize GPU resources: {e}")

    def add_entities(self, entities: List[EntityNode], batch_size: int = 32):
        """Add new entities to the index."""
        if not entities:
            return

        # Create entity strings for embedding
        entity_strings = []
        for entity in entities:
            roles_states = []
            for role in entity.roles:
                role_state = f"Role: {role.role} States: {', '.join(role.states)}"
                roles_states.append(role_state)

            entity_string = f"Entity: {entity.name} {' '.join(roles_states)}"
            entity_strings.append(entity_string)

        # Get embeddings for new entities
        new_embeddings = np.zeros((len(entities), self.embedding_dim), dtype=np.float32)
        for i in range(0, len(entities), batch_size):
            batch = entity_strings[i : i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch)
            new_embeddings[i : i + len(batch)] = batch_embeddings

        # Normalize embeddings
        faiss.normalize_L2(new_embeddings)

        # Update mappings
        start_idx = len(self.entity_to_id)
        for i, entity in enumerate(entities):
            idx = start_idx + i
            self.entity_to_id[entity.id] = idx
            self.id_to_entity[idx] = entity

        # Update FAISS GPU index
        if self.index is None:
            if self.gpu_resources is not None:
                # Create CPU flat index for exact nearest neighbor search
                cpu_index = faiss.IndexFlatIP(self.embedding_dim)
                if self.embeddings is not None:
                    cpu_index.add(self.embeddings)
                cpu_index.add(new_embeddings)

                # Transfer to GPU
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)
            else:
                # Fallback to CPU index if GPU not available
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                if self.embeddings is not None:
                    self.index.add(self.embeddings)
                self.index.add(new_embeddings)
        else:
            # For existing GPU index, we need to transfer to CPU, add, then transfer back
            if self.gpu_resources is not None:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                cpu_index.add(new_embeddings)
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)
            else:
                # CPU index, just add directly
                self.index.add(new_embeddings)

        # Update embeddings array
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def find_similar(
        self, query_entity: EntityNode, k: int = 5
    ) -> List[Tuple[EntityNode, float]]:
        """Find k most similar entities to the query entity."""
        if not self.index:
            return []

        # Format query entity string
        query_string = f"Entity: {query_entity.name}"
        for role in query_entity.roles:
            query_string += f" Role: {role.role} States: {', '.join(role.states)}"

        # Get query embedding
        query_embedding = np.array(self.embedding_model.embed_query(query_string))
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        # Search index
        similarities, indices = self.index.search(query_embedding, k)

        # Convert to list of (entity, similarity) tuples
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.id_to_entity) and idx >= 0:
                entity = self.id_to_entity[idx]
                results.append((entity, float(similarity)))

        return results


class EmbeddingMatchStrategy(MatchingStrategy):
    """Strategy for matching entities by embedding similarity."""

    def __init__(
        self,
        k: int = 5,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict[str, float]] = None,
    ):
        """Initialize the strategy."""
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not available. Install with: "
                "pip install faiss-cpu langchain-voyageai"
            )

        self.k = k
        self.model_name = model_name
        self.generation_params = generation_params or {"temperature": 0.0}

        # Initialize verifier
        self.entity_verifier = EntityVerifier(
            model_name=model_name, generation_params=self.generation_params
        )

    def format_candidate_pairs(
        self, candidate_pairs: List[Tuple[EntityNode, EntityNode]]
    ) -> List[str]:
        """Format candidate pairs for verification."""
        pair_descriptions = []
        for i, (new_entity, old_entity) in enumerate(candidate_pairs):
            new_entity_string = format_entity_string(new_entity)
            old_entity_string = format_entity_string(old_entity)
            pair_descriptions.append(
                f"pair_{i}: New entity '{new_entity_string}' and old entity '{old_entity_string}'"
            )
        return pair_descriptions

    def verify_pairs(
        self, candidate_pairs: List[Tuple[EntityNode, EntityNode]]
    ) -> Dict[str, bool]:
        """Verify if entities are semantically similar."""
        if not candidate_pairs:
            return {}

        pair_descriptions = self.format_candidate_pairs(candidate_pairs)
        input_data = {"pair_descriptions": pair_descriptions}

        verification_results = self.entity_verifier([input_data])
        return verification_results[0]["parsed_response"]

    def reconcile_entities(
        self, new_entities: List[EntityNode], entity_index: EntityIndex
    ) -> List[Tuple[EntityNode, EntityNode]]:
        """Find entities with similar embeddings and verify them."""
        if not isinstance(entity_index, EmbeddingEntityIndex):
            raise TypeError("EmbeddingMatchStrategy requires an EmbeddingEntityIndex")

        candidate_pairs = []

        # Find similar entities using embedding similarity
        for new_entity in new_entities:
            similar_entities = entity_index.find_similar(new_entity, k=self.k)
            for old_entity, similarity in similar_entities:
                # Don't match entities from the same chunk
                if old_entity.chunk_id != new_entity.chunk_id:
                    candidate_pairs.append((new_entity, old_entity))

        # Verify entity similarity using LLM
        verification_results = self.verify_pairs(candidate_pairs)

        # Extract verified pairs
        verified_pairs = []
        for i, (new_entity, old_entity) in enumerate(candidate_pairs):
            pair_id = f"pair_{i}"
            if verification_results.get(pair_id, False):
                verified_pairs.append((new_entity, old_entity))

        return verified_pairs

    def reconcile_verb_phrases(
        self, new_gsw: GSWStructure, global_memory: GSWStructure
    ) -> Set[str]:
        """Default implementation - no verb phrase reconciliation."""
        return set()

    def resolve_questions(
        self,
        new_chunk_text: Optional[str],
        matched_old_entity_ids: Set[str],
        global_memory: GSWStructure,
        new_gsw_entities: List[EntityNode],
        chunk_id: Optional[str] = None,
    ) -> None:
        """Default implementation - no question resolution."""
        if matched_old_entity_ids:
            print(
                "Warning: Question resolution not implemented for EmbeddingMatchStrategy."
            )


# Factory Function


def create_matching_components(
    approach: str = "exact", **kwargs
) -> Tuple[MatchingStrategy, EntityIndex]:
    """
    Factory function to create matching strategy and index pairs.

    Args:
        approach: Matching approach ("exact" or "embedding")
        **kwargs: Additional parameters for strategy/index initialization

    Returns:
        Tuple of (strategy, index) instances
    """
    if approach == "exact":
        strategy = ExactMatchStrategy(
            model_name=kwargs.get("model_name", "gpt-4o"),
            generation_params=kwargs.get(
                "generation_params", {"temperature": 0.0, "max_tokens": 500}
            ),
        )
        index = ExactMatchEntityIndex()
        return strategy, index

    elif approach == "embedding":
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not available. Install with: "
                "pip install faiss-cpu langchain-voyageai"
            )

        strategy = EmbeddingMatchStrategy(
            k=kwargs.get("k", 5),
            model_name=kwargs.get("model_name", "gpt-4o"),
            generation_params=kwargs.get("generation_params", {"temperature": 0.0}),
        )
        index = EmbeddingEntityIndex(embedding_dim=kwargs.get("embedding_dim", 1024))
        return strategy, index

    else:
        raise ValueError(
            f"Unknown matching approach: {approach}. Use 'exact' or 'embedding'."
        )
