"""
Core data models for GSW Memory System.

This module contains the primary data structures that represent the semantic
workspace, including entities, roles, states, and their relationships.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Role(BaseModel):
    """Class to represent the roles and states associated with an entity."""

    role: str = Field(description="The role identifier (e.g., 'Suspect', 'Officer')")
    states: List[str] = Field(
        default_factory=list,
        description="List of states for this role (e.g., 'apprehended', 'charged')",
    )
    chunk_id: Optional[str] = Field(
        default=None, description="A global identifier for the chunk"
    )


class EntityNode(BaseModel):
    """Class to represent an entity in the GSW."""

    id: str = Field(description="Unique identifier for the entity")
    name: str = Field(description="Entity name (e.g., 'Jonathan Miller')")
    roles: List[Role] = Field(
        default_factory=list, description="List of roles, states this entity has"
    )
    chunk_id: Optional[str] = Field(
        default=None, description="A global identifier for the chunk"
    )
    summary: Optional[str] = Field(
        default=None,
        description="An entity focussed summary detailing the entity's involvement.",
    )


class Question(BaseModel):
    """Class to represent questions associated with a verb phrase in the GSW, note that these"""

    id: str = Field(description="Unique identifier for the question")
    text: str = Field(description="The question text")
    answers: List[str] = Field(
        default_factory=list, description="List of answers to this question"
    )
    chunk_id: Optional[str] = Field(
        default=None, description="A global identifier for the chunk"
    )


class VerbPhraseNode(BaseModel):
    """Class to represent a verb phrase and its associated semantic questions."""

    id: str = Field(description="Unique identifier for the verb phrase")
    phrase: str = Field(description="The verb phrase (e.g., 'apprehended', 'charged')")
    questions: List[Question] = Field(
        default_factory=list, description="Questions associated with this verb"
    )
    chunk_id: Optional[str] = Field(
        default=None, description="A global identifier for the chunk"
    )


class SpaceNode(BaseModel):
    """Space node class to represent a spatial location in the GSW."""

    id: str = Field(description="Unique identifier for the space node")
    name_history: Dict[str, str] = Field(
        default_factory=dict, description="chunk_id -> location name mapping"
    )
    current_name: Optional[str] = Field(
        default=None, description="Current/canonical name"
    )
    chunk_id: Optional[str] = Field(
        default=None, description="When this location was first identified"
    )
    type: str = Field(default="space", description="Node type identifier")

    @property
    def formatted_history(self) -> str:
        """Returns the formatted history string for display."""
        return ", ".join(
            [f"Chunk{k}:{v}" for k, v in sorted(self.name_history.items())]
        )


class TimeNode(BaseModel):
    """Time node class to represent a temporal context in the GSW."""

    id: str = Field(description="Unique identifier for the time node")
    name_history: Dict[str, str] = Field(
        default_factory=dict, description="chunk_id -> time description mapping"
    )
    current_name: Optional[str] = Field(
        default=None, description="Current/canonical name"
    )
    chunk_id: Optional[str] = Field(
        default=None, description="When this time was first identified"
    )
    type: str = Field(default="time", description="Node type identifier")

    @property
    def formatted_history(self) -> str:
        """Returns the formatted history string for display."""
        return ", ".join(
            [f"Chunk{k}:{v}" for k, v in sorted(self.name_history.items())]
        )


class GSWStructure(BaseModel):
    """
    The core Generative Semantic Workspace structure.

    This represents the complete semantic memory state, including entities,
    verb phrases, spatial and temporal nodes, and their relationships.
    """

    entity_nodes: List[EntityNode] = Field(
        default_factory=list, description="All entities in the workspace"
    )
    verb_phrase_nodes: List[VerbPhraseNode] = Field(
        default_factory=list, description="All verb phrases"
    )
    space_nodes: List[SpaceNode] = Field(
        default_factory=list, description="All spatial locations"
    )
    time_nodes: List[TimeNode] = Field(
        default_factory=list, description="All temporal contexts"
    )

    # Relationship edges
    similarity_edges: List[Tuple[str, str]] = Field(
        default_factory=list, description="Entity similarity connections"
    )
    space_edges: List[Tuple[str, str]] = Field(
        default_factory=list, description="Entity-space connections"
    )
    time_edges: List[Tuple[str, str]] = Field(
        default_factory=list, description="Entity-time connections"
    )

    @classmethod
    def from_json(cls, json_data: dict) -> "GSWStructure":
        """Create a GSWStructure from JSON data."""
        return cls(**json_data)

    def copy(self) -> "GSWStructure":
        """Create a deep copy of the GSW structure."""
        return GSWStructure(
            entity_nodes=self.entity_nodes.copy(),
            verb_phrase_nodes=self.verb_phrase_nodes.copy(),
            space_nodes=self.space_nodes.copy(),
            time_nodes=self.time_nodes.copy(),
            similarity_edges=self.similarity_edges.copy(),
            space_edges=self.space_edges.copy(),
            time_edges=self.time_edges.copy(),
        )

    # Entity management methods
    def add_entity(self, entity: EntityNode) -> None:
        """Add a new entity to the GSW structure."""
        if any(e.id == entity.id for e in self.entity_nodes):
            return  # Entity already exists
        self.entity_nodes.append(entity)

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityNode]:
        """Find an entity node by its global ID."""
        for entity in self.entity_nodes:
            if entity.id == entity_id:
                return entity
        return None

    # Verb phrase management
    def add_verb_phrase(self, verb_phrase: VerbPhraseNode) -> None:
        """Add a new verb phrase to the GSW structure."""
        composite_id = (
            f"{verb_phrase.chunk_id}_{verb_phrase.id}"
            if verb_phrase.chunk_id
            else verb_phrase.id
        )

        existing_composite_ids = [
            f"{v.chunk_id}_{v.id}" if v.chunk_id else v.id
            for v in self.verb_phrase_nodes
        ]

        if composite_id in existing_composite_ids:
            return  # Verb phrase already exists
        self.verb_phrase_nodes.append(verb_phrase)

    # Get verb phrase by id
    def get_verb_phrase_by_id(self, verb_phrase_id: str) -> Optional[VerbPhraseNode]:
        """Find a verb phrase node by its global ID."""
        for vp in self.verb_phrase_nodes:
            if vp.id == verb_phrase_id:
                return vp
        return None

    def get_question_by_id(
        self, question_id: str
    ) -> Optional[Tuple[VerbPhraseNode, Question]]:
        """Find a question node (and its parent VP) by its global ID."""
        for vp in self.verb_phrase_nodes:
            for question in vp.questions:
                if question.id == question_id:
                    return vp, question
        return None

    # Space-time management
    def add_space_node(self, space_node: SpaceNode) -> None:
        """Add a new space node to the GSW structure."""
        if any(s.id == space_node.id for s in self.space_nodes):
            return  # Space node already exists
        self.space_nodes.append(space_node)

    def add_time_node(self, time_node: TimeNode) -> None:
        """Add a new time node to the GSW structure."""
        if any(t.id == time_node.id for t in self.time_nodes):
            return  # Time node already exists
        self.time_nodes.append(time_node)

    def get_space_node_by_id(self, space_id: str) -> Optional[SpaceNode]:
        """Find a space node by its ID."""
        for space in self.space_nodes:
            if space.id == space_id:
                return space
        return None

    def get_time_node_by_id(self, time_id: str) -> Optional[TimeNode]:
        """Find a time node by its ID."""
        for time in self.time_nodes:
            if time.id == time_id:
                return time
        return None

    # Edge management
    def add_space_edge(self, entity_node_id: str, space_node_id: str) -> None:
        """Add a space edge between a space node and an entity node."""
        edge = (entity_node_id, space_node_id)
        if edge not in self.space_edges:
            self.space_edges.append(edge)

    def add_time_edge(self, entity_node_id: str, time_node_id: str) -> None:
        """Add a time edge between a time node and an entity node."""
        edge = (entity_node_id, time_node_id)
        if edge not in self.time_edges:
            self.time_edges.append(edge)

    def add_similarity_edge(self, entity_id1: str, entity_id2: str) -> None:
        """Add a similarity edge between two entities."""
        entity1_exists = any(e.id == entity_id1 for e in self.entity_nodes)
        entity2_exists = any(e.id == entity_id2 for e in self.entity_nodes)

        if not entity1_exists or not entity2_exists:
            return  # One or both entities don't exist

        edge = (entity_id1, entity_id2)
        reverse_edge = (entity_id2, entity_id1)

        if (
            edge not in self.similarity_edges
            and reverse_edge not in self.similarity_edges
        ):
            self.similarity_edges.append(edge)

    # Merge entities usually during reconciliation of entities
    def merge_external_entity(
        self, target_entity_id: str, external_entity: EntityNode
    ) -> None:
        """Merge an external entity into a target entity within this structure."""
        target_entity = self.get_entity_by_id(target_entity_id)
        if not target_entity:
            print(
                f"Warning: Target entity {target_entity_id} not found for merging external entity {external_entity.id}"
            )
            return

        # Combine roles from external_entity into target_entity
        for role in external_entity.roles:
            if not any(
                r.role == role.role and r.chunk_id == role.chunk_id
                for r in target_entity.roles
            ):
                target_entity.roles.append(role)

        # Update verb phrase references
        for verb in self.verb_phrase_nodes:
            for question in verb.questions:
                question.answers = [
                    target_entity_id if answer == external_entity.id else answer
                    for answer in question.answers
                ]

        # Remove similarity edges referencing external entity
        self.similarity_edges = [
            edge for edge in self.similarity_edges if external_entity.id not in edge
        ]

    # Merge space nodes, usually when one entity connected to another entity gets tagged with a new space
    def merge_space_nodes(
        self, target_id: str, source_id: str, chunk_id: Optional[str] = None
    ) -> None:
        """Merge source space node into target space node."""
        target_node = self.get_space_node_by_id(target_id)
        source_node = self.get_space_node_by_id(source_id)

        if not target_node or not source_node:
            print(
                f"Warning: Could not merge space nodes. Target: {target_id}, Source: {source_id}"
            )
            return

        # Merge name history
        for chunk, name in source_node.name_history.items():
            if chunk not in target_node.name_history:
                target_node.name_history[chunk] = name

        # Update current name if source is newer
        if (
            source_node.chunk_id
            and target_node.chunk_id
            and source_node.chunk_id > target_node.chunk_id
        ):
            target_node.current_name = source_node.current_name
            target_node.chunk_id = chunk_id

        # Redirect edges from source to target
        self.space_edges = [
            (edge[0], target_id) if edge[1] == source_id else edge
            for edge in self.space_edges
        ]

        # Remove source node
        self.space_nodes = [node for node in self.space_nodes if node.id != source_id]

    # Merge time nodes, usually when one entity connected to another entity gets tagged with a new time
    def merge_time_nodes(
        self, target_id: str, source_id: str, chunk_id: Optional[str] = None
    ) -> None:
        """Merge source time node into target time node."""
        target_node = self.get_time_node_by_id(target_id)
        source_node = self.get_time_node_by_id(source_id)

        if not target_node or not source_node:
            print(
                f"Warning: Could not merge time nodes. Target: {target_id}, Source: {source_id}"
            )
            return

        # Merge name history
        for chunk, name in source_node.name_history.items():
            if chunk not in target_node.name_history:
                target_node.name_history[chunk] = name

        # Update current name if source is newer
        if (
            source_node.chunk_id
            and target_node.chunk_id
            and source_node.chunk_id > target_node.chunk_id
        ):
            target_node.current_name = source_node.current_name
            target_node.chunk_id = chunk_id

        # Redirect edges from source to target
        self.time_edges = [
            (edge[0], target_id) if edge[1] == source_id else edge
            for edge in self.time_edges
        ]

        # Remove source node
        self.time_nodes = [node for node in self.time_nodes if node.id != source_id]

    # Get the most connected entities in the GSW structure
    def get_most_connected_entities(self, top_n: int = 5) -> str:
        """
        Identifies the most connected entities in the GSW structure.
        Returns a formatted string of the top N most connected entities.
        This could be used to provide an LLM with information about the structure of the GSW.

        """
        entity_connections = {entity.id: 0 for entity in self.entity_nodes}

        # Count connections from verb phrase questions
        for vp in self.verb_phrase_nodes:
            for question in vp.questions:
                for answer in question.answers:
                    if (
                        answer in entity_connections
                        and answer != "None"
                        and not (isinstance(answer, str) and answer.startswith("TEXT:"))
                    ):
                        entity_connections[answer] += 1

        # Count connections from similarity edges
        for entity1_id, entity2_id in self.similarity_edges:
            if entity1_id in entity_connections:
                entity_connections[entity1_id] += 1
            if entity2_id in entity_connections:
                entity_connections[entity2_id] += 1

        # Count space and time connections
        for entity_id, _ in self.space_edges:
            if entity_id in entity_connections:
                entity_connections[entity_id] += 1

        for entity_id, _ in self.time_edges:
            if entity_id in entity_connections:
                entity_connections[entity_id] += 1

        # Get top connected entities
        connected_entities = []
        for entity_id, count in entity_connections.items():
            entity = self.get_entity_by_id(entity_id)
            if entity:
                connected_entities.append((entity, count))

        connected_entities.sort(key=lambda x: x[1], reverse=True)
        return "\n".join(
            [
                f"{entity.name} ({count} connections)"
                for entity, count in connected_entities[:top_n]
            ]
        )
