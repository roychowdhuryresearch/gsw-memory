"""
Entity indexing strategies for the GSW reconciler.

This module provides different approaches for indexing and retrieving entities
for matching during reconciliation.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

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

from ..models import EntityNode


class EntityIndex(ABC):
    """Abstract base class for entity indices."""

    @abstractmethod
    def add_entities(self, entities: List[EntityNode], batch_size: int = 32):
        """Add new entities to the index."""
        pass


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


class EmbeddingEntityIndex(EntityIndex):
    """Maintains an index of entity embeddings for efficient similarity search."""

    def __init__(self, embedding_dim: int = 1024):
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not available. Install with: "
                "pip install faiss-cpu langchain-voyageai"
            )
        
        self.embedding_dim = embedding_dim
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, EntityNode] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.embedding_model = VoyageAIEmbeddings(model="voyage-3")

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

        # Update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if self.embeddings is not None:
                self.index.add(self.embeddings)

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