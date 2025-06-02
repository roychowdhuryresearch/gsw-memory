"""
Bipartite graph creation and visualization for GSW structures.

This module provides functionality to convert GSW structures into NetworkX graphs
and export them to Cytoscape format for interactive visualization.
"""

import colorsys
import json
from typing import Dict, Optional

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from ..memory.models import GSWStructure


def get_color_gradient(num_chunks: int, min_hue: float = 0.1, max_hue: float = 0.3) -> str:
    """Generate a color based on number of chunks contributing to an entity's roles.

    Args:
        num_chunks: Number of different chunks in the entity's roles
        min_hue: Hue value for minimum number of chunks (0.1 = orange)
        max_hue: Hue value for maximum number of chunks (0.3 = green)

    Returns:
        Hex color string
    """
    # Cap the number of chunks at 10 for the color scaling
    capped_num = min(num_chunks, 10)

    # Calculate hue based on number of chunks (1 chunk = min_hue, 10+ chunks = max_hue)
    if capped_num <= 1:
        hue = min_hue
    else:
        hue = min_hue + (max_hue - min_hue) * (capped_num - 1) / 9

    # Convert HSV to RGB (S and V are fixed)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)

    # Convert to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    return hex_color


def create_bipartite_graph(semantic_rep: GSWStructure):
    """Create a NetworkX bipartite graph from a GSW structure.
    
    Args:
        semantic_rep: GSWStructure object to convert to graph
        
    Returns:
        NetworkX Graph object
        
    Raises:
        ImportError: If NetworkX is not available
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX is required for graph visualization. Install with: "
            "pip install networkx"
        )
    
    G = nx.Graph()

    # First, create a lookup of entity IDs to actual entity objects for quick reference
    entity_lookup = {entity.id: entity for entity in semantic_rep.entity_nodes}

    # Add entity nodes (set bipartite attribute to 0)
    for entity in semantic_rep.entity_nodes:
        G.add_node(
            entity.id,
            bipartite=0,
            type="entity",
            name=entity.name,
            roles=entity.roles,
            chunk_id=entity.chunk_id,
        )

    # Add verb phrase nodes (set bipartite attribute to 1)
    for verb in semantic_rep.verb_phrase_nodes:
        G.add_node(
            verb.id,
            bipartite=1,
            type="verb_phrase",
            phrase=verb.phrase,
            chunk_id=verb.chunk_id,
        )

        # Add edges based on questions
        for question in verb.questions:
            for answer_id in question.answers:
                # Handle 'None' or empty answers by creating special nodes
                if answer_id is None or answer_id == "None" or not answer_id:
                    # Create a unique ID for this unresolved answer
                    unresolved_id = f"unresolved_{verb.id}_{question.id}"

                    # Add a special node for this unresolved answer
                    G.add_node(
                        unresolved_id,
                        bipartite=0,  # Same side as entity nodes
                        type="unresolved",
                        name="Unknown",
                        chunk_id=verb.chunk_id,
                        is_unresolved=True,
                    )

                    # Add an edge to this unresolved node
                    G.add_edge(
                        verb.id,
                        unresolved_id,
                        question_id=question.id,
                        question_text=question.text,
                        chunk_id=verb.chunk_id,
                        is_unresolved=True,
                    )
                    continue

                # If answer_id doesn't exist as a node, we need to handle it
                if answer_id not in G:
                    # Check if this entity exists in our entity lookup
                    if answer_id in entity_lookup:
                        # Entity exists in our lookup but wasn't added to graph yet
                        entity = entity_lookup[answer_id]
                        G.add_node(
                            entity.id,
                            bipartite=0,
                            type="entity",
                            name=entity.name,
                            roles=entity.roles,
                            chunk_id=entity.chunk_id,
                        )
                    else:
                        # This might be a reference to an entity that was merged
                        # For now, just add a placeholder node
                        G.add_node(
                            answer_id,
                            bipartite=0,
                            type="entity",
                            name=str(answer_id),
                            roles=[],
                            is_placeholder=True,
                        )

                # Now add the edge with question information
                G.add_edge(
                    verb.id,
                    answer_id,
                    question_id=question.id,
                    question_text=question.text,
                    chunk_id=verb.chunk_id,
                )

    # Add similarity edges if they exist
    if hasattr(semantic_rep, "similarity_edges") and semantic_rep.similarity_edges:
        for entity1_id, entity2_id in semantic_rep.similarity_edges:
            # Add entities if they don't exist (unlikely but just to be safe)
            for entity_id in [entity1_id, entity2_id]:
                if entity_id not in G:
                    # Try to get actual entity data if available
                    if entity_id in entity_lookup:
                        entity = entity_lookup[entity_id]
                        G.add_node(
                            entity.id,
                            bipartite=0,
                            type="entity",
                            name=entity.name,
                            roles=entity.roles,
                            chunk_id=entity.chunk_id,
                        )
                    else:
                        G.add_node(
                            entity_id,
                            bipartite=0,
                            type="entity",
                            name=str(entity_id),
                            roles=[],
                        )
            # Add the similarity edge
            G.add_edge(
                entity1_id,
                entity2_id,
                relationship="similar",
            )

    # Add space nodes if they exist
    if hasattr(semantic_rep, "space_nodes") and semantic_rep.space_nodes:
        for space_node in semantic_rep.space_nodes:
            short_label = space_node.current_name or space_node.id
            short_label = short_label[:20] + "..." if len(short_label) > 20 else short_label
            G.add_node(
                space_node.id,
                bipartite=2,
                type="space",
                label=short_label,
                full_name=space_node.current_name or "N/A",
                history=space_node.formatted_history,
                chunk_id=space_node.chunk_id,
            )

    # Add time nodes if they exist
    if hasattr(semantic_rep, "time_nodes") and semantic_rep.time_nodes:
        for time_node in semantic_rep.time_nodes:
            short_label = time_node.current_name or time_node.id
            short_label = short_label[:20] + "..." if len(short_label) > 20 else short_label
            G.add_node(
                time_node.id,
                bipartite=2,
                type="time",
                label=short_label,
                full_name=time_node.current_name or "N/A",
                history=time_node.formatted_history,
                chunk_id=time_node.chunk_id,
            )

    # Add space edges if they exist
    if hasattr(semantic_rep, "space_edges") and semantic_rep.space_edges:
        for entity_id, space_id in semantic_rep.space_edges:
            if G.has_node(space_id) and G.has_node(entity_id):
                G.add_edge(entity_id, space_id, label="located_at")

    # Add time edges if they exist
    if hasattr(semantic_rep, "time_edges") and semantic_rep.time_edges:
        for entity_id, time_id in semantic_rep.time_edges:
            if G.has_node(time_id) and G.has_node(entity_id):
                G.add_edge(entity_id, time_id, label="occurred_at")

    return G


def visualize_bipartite_graph(G) -> Dict:
    """Convert networkx bipartite graph to cytoscape JSON format.
    
    Args:
        G: NetworkX Graph object
        
    Returns:
        Dictionary in Cytoscape JSON format
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX is required for graph visualization. Install with: "
            "pip install networkx"
        )

    # Get unique chunk IDs for coloring and grouping
    all_chunk_ids = set()
    for _, attr in G.nodes(data=True):
        chunk_id = attr.get("chunk_id")
        if chunk_id:
            all_chunk_ids.add(chunk_id)

    # Sort chunk IDs to create a consistent order
    chunk_id_list = sorted(list(all_chunk_ids))
    chunk_id_to_index = {chunk_id: i for i, chunk_id in enumerate(chunk_id_list)}

    # Create initial cytoscape data structure with enhanced metadata
    cyjs_data = {
        "elements": {"nodes": [], "edges": []},
        "data": {
            "title": "GSW Visualization",
            "description": "Graph visualization of GSW structure with temporal evolution",
            "chunks": list(all_chunk_ids),
            "generated_by": "GSW Memory Package",
        },
        "style": [
            # Base styles for all nodes
            {
                "selector": "node",
                "css": {
                    "content": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "background-color": "#DDD",
                    "text-wrap": "wrap",
                    "text-max-width": "200px",
                    "font-family": "Helvetica Neue, Helvetica, sans-serif",
                    "text-outline-width": "1px",
                    "text-outline-color": "#ffffff",
                    "text-outline-opacity": 0.5,
                    "color": "#333333",
                    "border-width": "1px",
                    "border-color": "#777777",
                },
            },
            # Entity nodes
            {
                "selector": "node[type='entity']",
                "css": {
                    "shape": "roundrectangle",
                    "background-color": "#6FB1FC",
                    "width": "160px",
                    "height": "80px",
                    "font-size": "13px",
                    "font-weight": "bold",
                },
            },
            # Verb phrase nodes
            {
                "selector": "node[type='verb_phrase']",
                "css": {
                    "shape": "triangle",
                    "background-color": "#FF6347",
                    "width": "120px",
                    "height": "80px",
                    "font-size": "11px",
                    "font-weight": "normal",
                },
            },
            # Space nodes
            {
                "selector": "node[type='space']",
                "css": {
                    "shape": "diamond",
                    "background-color": "#32CD32",
                    "width": "100px",
                    "height": "60px",
                    "font-size": "10px",
                },
            },
            # Time nodes
            {
                "selector": "node[type='time']",
                "css": {
                    "shape": "pentagon",
                    "background-color": "#FFD700",
                    "width": "100px",
                    "height": "60px",
                    "font-size": "10px",
                },
            },
            # Edges
            {
                "selector": "edge",
                "css": {
                    "width": "2px",
                    "line-color": "#666",
                    "curve-style": "bezier",
                    "label": "data(question_text)",
                    "font-size": "8px",
                    "text-rotation": "autorotate",
                    "text-margin-y": "-10px",
                },
            },
        ],
    }

    # Convert nodes
    for node_id, attr in G.nodes(data=True):
        node_type = attr.get("type", "unknown")
        
        # Determine node label based on type
        if node_type == "entity":
            label = attr.get("name", node_id)
            # Check for temporal evolution
            roles = attr.get("roles", [])
            unique_chunks = set()
            for role in roles:
                if hasattr(role, 'chunk_id') and role.chunk_id:
                    unique_chunks.add(role.chunk_id)
            has_temporal_evolution = len(unique_chunks) > 1
            
            # Create detailed tooltip information
            roles_info = []
            for role in roles:
                role_text = f"Role: {role.role if hasattr(role, 'role') else 'N/A'}"
                if hasattr(role, 'states') and role.states:
                    role_text += f" | States: {', '.join(role.states)}"
                if hasattr(role, 'chunk_id') and role.chunk_id:
                    role_text += f" | Chunk: {role.chunk_id}"
                roles_info.append(role_text)
            
            tooltip = f"Entity: {label}\n" + "\n".join(roles_info)
            
        elif node_type == "verb_phrase":
            label = attr.get("phrase", node_id)
            tooltip = f"Verb Phrase: {label}\nChunk: {attr.get('chunk_id', 'N/A')}"
            has_temporal_evolution = False
            
        elif node_type in ["space", "time"]:
            label = attr.get("label", attr.get("full_name", node_id))
            tooltip = f"{node_type.title()}: {attr.get('full_name', 'N/A')}\nHistory: {attr.get('history', 'N/A')}"
            has_temporal_evolution = False
            
        else:
            label = attr.get("name", node_id)
            tooltip = f"Node: {label}"
            has_temporal_evolution = False

        # Create node data
        node_data = {
            "data": {
                "id": node_id,
                "label": label,
                "type": node_type,
                "tooltip": tooltip,
                "chunk_id": attr.get("chunk_id"),
                "has_temporal_evolution": has_temporal_evolution,
                "is_placeholder": attr.get("is_placeholder", False),
                "is_unresolved": attr.get("is_unresolved", False),
            }
        }
        
        cyjs_data["elements"]["nodes"].append(node_data)

    # Convert edges
    for source, target, attr in G.edges(data=True):
        edge_data = {
            "data": {
                "id": f"{source}_{target}",
                "source": source,
                "target": target,
                "question_text": attr.get("question_text", ""),
                "question_id": attr.get("question_id", ""),
                "relationship": attr.get("relationship", ""),
                "label": attr.get("label", ""),
                "chunk_id": attr.get("chunk_id"),
                "is_unresolved": attr.get("is_unresolved", False),
            }
        }
        
        cyjs_data["elements"]["edges"].append(edge_data)

    return cyjs_data


def create_and_save_gsw_visualization(gsw: GSWStructure, output_path: str) -> None:
    """Create a GSW visualization and save it to a file.
    
    Args:
        gsw: GSWStructure to visualize
        output_path: Path to save the Cytoscape JSON file
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX is required for graph visualization. Install with: "
            "pip install networkx"
        )
    
    # Create bipartite graph
    graph = create_bipartite_graph(gsw)
    
    # Convert to Cytoscape format
    cytoscape_data = visualize_bipartite_graph(graph)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(cytoscape_data, f, indent=2)