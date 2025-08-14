#!/usr/bin/env python3
"""
Explore entity hypernode clustering using Qwen embeddings.
Tests whether entities can be clustered into meaningful groups for efficient reconciliation.
"""

import json
import glob
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass

# Similarity computation imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp
    SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn or scipy not available. Install with: pip install scikit-learn scipy")
    SIMILARITY_AVAILABLE = False

# VLLM for embeddings
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Install with: pip install vllm>=0.8.5")
    VLLM_AVAILABLE = False

from src.gsw_memory.memory.models import GSWStructure


@dataclass
class EntityData:
    """Container for entity data used in clustering."""
    entity_id: str
    name: str
    roles: List[str]
    doc_id: str
    description: str
    embedding: np.ndarray = None


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


def generate_hypernode_name(entities: List[EntityData]) -> str:
    """Generate a name for a hypernode based on entity names."""
    if not entities:
        return "Empty Hypernode"
    
    if len(entities) == 1:
        return entities[0].name
    
    # Get all entity names
    names = [entity.name for entity in entities]
    
    # Count occurrences of each name
    name_counts = Counter(names)
    
    # If there are repeated names, use the most common one
    if name_counts.most_common(1)[0][1] > 1:
        return name_counts.most_common(1)[0][0]
    
    # If all names are unique, pick the shortest one (often the most canonical)
    # or fall back to the first one
    shortest_name = min(names, key=len)
    return shortest_name


def load_gsw_files(num_documents: int = 50) -> Tuple[List[GSWStructure], List[str]]:
    """Load GSW structures from JSON files."""
    print(f"Loading first {num_documents} GSW files...")
    
    base_dir = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/logs/full_2wiki_corpus_20250710_202211/gsw_output_global_ids/networks"
    
    # Get first N document directories
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


def extract_entity_data(gsw_structures: List[GSWStructure], doc_ids: List[str]) -> List[EntityData]:
    """Extract entities with their role descriptions."""
    print("Extracting entity data...")
    
    entity_data = []
    
    for gsw, doc_id in zip(gsw_structures, doc_ids):
        if not gsw.entity_nodes:
            continue
            
        for entity in gsw.entity_nodes:
            # Extract role descriptions with states
            role_descriptions = []
            for role in entity.roles:  # Limit to first 10 roles to avoid too much text
                if role.states:
                    # Include both role and states
                    states_text = ', '.join(role.states)  # Limit states too
                    role_desc = f"{role.role}: {states_text}"
                else:
                    # Just the role if no states
                    role_desc = role.role
                role_descriptions.append(role_desc)
            
            # Create entity description
            if role_descriptions:
                roles_text = ' | '.join(role_descriptions)
                description = f"{entity.name} - Roles: {roles_text}"
            else:
                description = f"{entity.name} - No specific roles"
            
            entity_data.append(EntityData(
                entity_id=entity.id,
                name=entity.name,
                roles=role_descriptions,
                doc_id=doc_id,
                description=description
            ))
    
    print(f"Extracted {len(entity_data)} entities")
    return entity_data


def generate_embeddings(entity_data: List[EntityData]) -> List[EntityData]:
    """Generate embeddings for entities using Qwen model."""
    if not VLLM_AVAILABLE:
        print("VLLM not available, skipping embedding generation")
        return entity_data
        
    print("Generating embeddings with Qwen3-Embedding-8B...")
    
    # Custom task for entity identity matching
    task = 'Given an entity name and its contextual roles/states, create an embedding that captures the entity\'s unique identity to match the same entity appearing in different documents, even with slight name variations. You can further use the roles and states to determine their similarity.'

    
    # Prepare input texts with instructions
    input_texts = []
    for entity in entity_data:
        instructed_query = get_detailed_instruct(task, entity.description)
        input_texts.append(instructed_query)
    
    try:
        # Initialize model
        model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 50
        all_embeddings = []
        
        for batch_start in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[batch_start:batch_start+batch_size]
            batch_num = batch_start//batch_size + 1
            total_batches = (len(input_texts) + batch_size - 1)//batch_size
            print(f"Processing batch {batch_num}/{total_batches}")
            
            outputs = model.embed(batch_texts)
            batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Add embeddings to entity data
        for entity, embedding in zip(entity_data, embeddings):
            entity.embedding = embedding.numpy()
        
        print(f"Generated embeddings for {len(entity_data)} entities")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return entity_data
    
    return entity_data


def find_entity_similarities(entity_data: List[EntityData], similarity_threshold: float = 0.85) -> Dict[str, Any]:
    """Find pairwise similarities between entities to identify same entities across documents."""
    if not SIMILARITY_AVAILABLE:
        print("Similarity computation libraries not available")
        return {}
    
    print(f"Computing pairwise similarities with threshold {similarity_threshold}...")
    
    # Extract embeddings and create mapping
    embeddings_list = []
    entity_indices = {}
    valid_entities = []
    
    for entity in entity_data:
        if entity.embedding is not None:
            embeddings_list.append(entity.embedding)
            entity_indices[len(embeddings_list) - 1] = entity
            valid_entities.append(entity)
    
    if len(embeddings_list) == 0:
        print("No embeddings available for similarity computation")
        return {}
    
    embeddings = np.array(embeddings_list)
    print(f"Computing similarities for {len(embeddings)} entities with {embeddings.shape[1]} dimensions")
    
    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find entity pairs above threshold
    entity_pairs = []
    similarity_connections = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            entity_a = valid_entities[i]
            entity_b = valid_entities[j]
            
            # Only compare entities from different documents
            if entity_a.doc_id == entity_b.doc_id:
                continue
                
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                entity_pairs.append({
                    'entity_a': {
                        'name': entity_a.name,
                        'id': entity_a.entity_id,
                        'doc_id': entity_a.doc_id,
                        'roles': entity_a.roles[:5]  # Limit for display
                    },
                    'entity_b': {
                        'name': entity_b.name,
                        'id': entity_b.entity_id,
                        'doc_id': entity_b.doc_id,
                        'roles': entity_b.roles[:5]  # Limit for display
                    },
                    'similarity': float(similarity)
                })
                similarity_connections.append((i, j))
    
    print(f"Found {len(entity_pairs)} entity pairs above similarity threshold {similarity_threshold}")
    
    # Create hypernodes using connected components
    if similarity_connections:
        # Build adjacency matrix
        n_entities = len(valid_entities)
        adjacency_matrix = sp.lil_matrix((n_entities, n_entities))
        
        for i, j in similarity_connections:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
        
        # Find connected components (hypernodes)
        n_components, labels = connected_components(adjacency_matrix, directed=False)
        
        # Group entities by hypernode
        hypernodes = defaultdict(list)
        for entity_idx, hypernode_id in enumerate(labels):
            entity = valid_entities[entity_idx]
            hypernodes[hypernode_id].append(entity)
        
        print(f"Created {n_components} hypernodes from connected components")
    else:
        hypernodes = {}
        labels = []
    
    # Convert hypernodes to JSON serializable format with names
    hypernodes_serializable = {}
    hypernode_names = {}
    
    for hypernode_id, entities in hypernodes.items():
        hypernode_name = generate_hypernode_name(entities)
        hypernode_names[str(hypernode_id)] = hypernode_name
        
        hypernodes_serializable[str(hypernode_id)] = {
            'hypernode_name': hypernode_name,
            'entities': [
                {
                    'entity_id': entity.entity_id,
                    'name': entity.name,
                    'roles': entity.roles,
                    'doc_id': entity.doc_id,
                    'description': entity.description
                }
                for entity in entities
            ]
        }
    
    return {
        'entity_pairs': entity_pairs,
        'similarity_threshold': similarity_threshold,
        'hypernodes': dict(hypernodes),  # Keep original EntityData objects for analysis
        'hypernodes_serializable': hypernodes_serializable,  # JSON-ready version for saving
        'hypernode_names': hypernode_names,  # Names for each hypernode
        'hypernode_labels': labels.tolist() if similarity_connections else [],
        'n_hypernodes': len(hypernodes),
        'similarity_matrix': similarity_matrix.tolist()  # For detailed analysis
    }


def analyze_similarities(entity_data: List[EntityData], similarity_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze similarity results and generate insights about same-entity matches."""
    print("Analyzing entity similarity results...")
    
    if not similarity_results:
        return {}
    
    entity_pairs = similarity_results.get('entity_pairs', [])
    hypernodes = similarity_results.get('hypernodes', {})
    
    analysis = {
        'total_entities': len([e for e in entity_data if e.embedding is not None]),
        'entity_pairs_found': len(entity_pairs),
        'similarity_threshold': similarity_results.get('similarity_threshold', 0.85),
        'hypernodes_created': len(hypernodes),
        'entities_in_hypernodes': sum(len(entities) for entities in hypernodes.values()),
        'singleton_entities': 0,
        'hypernode_details': {},
        'top_matches': [],
        'cross_document_matches': 0
    }
    
    # Analyze hypernodes
    hypernode_sizes = []
    for hypernode_id, entities in hypernodes.items():
        hypernode_sizes.append(len(entities))
        
        # Check for cross-document matches
        doc_ids = set(entity.doc_id for entity in entities)
        is_cross_document = len(doc_ids) > 1
        if is_cross_document:
            analysis['cross_document_matches'] += 1
        
        # Sample entities from this hypernode
        sample_entities = []
        for entity in entities[:10]:  # Limit to 10 for display
            sample_entities.append({
                'name': entity.name,
                'doc_id': entity.doc_id,
                'roles': entity.roles[:3]  # Show first 3 roles
            })
        
        analysis['hypernode_details'][str(hypernode_id)] = {
            'size': len(entities),
            'cross_document': is_cross_document,
            'document_ids': list(doc_ids),
            'sample_entities': sample_entities
        }
    
    # Calculate singleton entities (entities not in any hypernode)
    entities_in_hypernodes = sum(len(entities) for entities in hypernodes.values())
    analysis['singleton_entities'] = analysis['total_entities'] - entities_in_hypernodes
    
    # Get top similarity matches
    if entity_pairs:
        sorted_pairs = sorted(entity_pairs, key=lambda x: x['similarity'], reverse=True)
        analysis['top_matches'] = sorted_pairs[:10]  # Top 10 matches
    
    # Hypernode size statistics
    if hypernode_sizes:
        analysis['hypernode_size_stats'] = {
            'min_size': min(hypernode_sizes),
            'max_size': max(hypernode_sizes),
            'avg_size': np.mean(hypernode_sizes),
            'median_size': np.median(hypernode_sizes)
        }
    
    return analysis


def print_similarity_summary(analysis: Dict[str, Any], similarity_results: Dict[str, Any] = None) -> None:
    """Print a summary of entity similarity results."""
    print("\n" + "="*80)
    print("ENTITY SIMILARITY ANALYSIS SUMMARY")
    print("="*80)
    
    if not analysis:
        print("No similarity analysis results found")
        return
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"{'='*40}")
    print(f"Total entities analyzed: {analysis['total_entities']}")
    print(f"Similarity threshold: {analysis['similarity_threshold']}")
    print(f"Entity pairs found: {analysis['entity_pairs_found']}")
    print(f"Hypernodes created: {analysis['hypernodes_created']}")
    print(f"Entities in hypernodes: {analysis['entities_in_hypernodes']}")
    print(f"Singleton entities: {analysis['singleton_entities']}")
    print(f"Cross-document hypernodes: {analysis['cross_document_matches']}")
    
    # Hypernode size statistics
    if 'hypernode_size_stats' in analysis:
        stats = analysis['hypernode_size_stats']
        print(f"\nHYPERNODE SIZE STATISTICS:")
        print(f"{'='*40}")
        print(f"Min hypernode size: {stats['min_size']}")
        print(f"Max hypernode size: {stats['max_size']}")
        print(f"Average hypernode size: {stats['avg_size']:.1f}")
        print(f"Median hypernode size: {stats['median_size']:.1f}")
    
    # Top similarity matches
    if analysis['top_matches']:
        print(f"\nTOP ENTITY MATCHES:")
        print(f"{'='*40}")
        for i, match in enumerate(analysis['top_matches'][:5], 1):
            print(f"\n{i}. Similarity: {match['similarity']:.3f}")
            print(f"   Entity A: {match['entity_a']['name']} (doc: {match['entity_a']['doc_id']})")
            print(f"   Entity B: {match['entity_b']['name']} (doc: {match['entity_b']['doc_id']})")
            if match['entity_a']['roles']:
                print(f"   A Roles: {', '.join(match['entity_a']['roles'][:3])}")
            if match['entity_b']['roles']:
                print(f"   B Roles: {', '.join(match['entity_b']['roles'][:3])}")
    
    # Cross-document hypernodes
    cross_doc_hypernodes = [
        (hid, details) for hid, details in analysis['hypernode_details'].items()
        if details['cross_document']
    ]
    
    if cross_doc_hypernodes:
        print(f"\nCROSS-DOCUMENT HYPERNODES:")
        print(f"{'='*40}")
        hypernode_names = similarity_results.get('hypernode_names', {}) if similarity_results else {}
        for hid, details in sorted(cross_doc_hypernodes, key=lambda x: x[1]['size'], reverse=True)[:5]:
            hypernode_name = hypernode_names.get(str(hid), f"Hypernode {hid}")
            print(f"\n'{hypernode_name}' (Hypernode {hid}) - {details['size']} entities across {len(details['document_ids'])} documents:")
            print(f"  Documents: {', '.join(details['document_ids'])}")
            print(f"  Sample entities:")
            for entity in details['sample_entities'][:5]:
                roles_str = ', '.join(entity['roles'][:2]) if entity['roles'] else "no roles"
                print(f"    - {entity['name']} (doc: {entity['doc_id']}) - {roles_str}")
    
    # Reconciliation impact estimate
    if analysis['hypernodes_created'] > 0:
        print(f"\nRECONCILATION IMPACT ESTIMATE:")
        print(f"{'='*40}")
        original_comparisons = analysis['total_entities'] * (analysis['total_entities'] - 1) // 2
        
        # Calculate comparisons within hypernodes
        hypernode_comparisons = 0
        for details in analysis['hypernode_details'].values():
            size = details['size']
            hypernode_comparisons += size * (size - 1) // 2
        
        # Add singleton entities (no comparisons needed for them)
        reduction_ratio = 1 - (hypernode_comparisons / original_comparisons) if original_comparisons > 0 else 0
        
        print(f"Original pairwise comparisons: {original_comparisons:,}")
        print(f"Hypernode pairwise comparisons: {hypernode_comparisons:,}")
        print(f"Complexity reduction: {reduction_ratio:.1%}")
        print(f"Speedup factor: {original_comparisons/hypernode_comparisons if hypernode_comparisons > 0 else float('inf'):.1f}x")


def save_results(entity_data: List[EntityData], similarity_results: Dict[str, Any], 
                analysis: Dict[str, Any], output_dir: str) -> None:
    """Save similarity results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save entity data (without embeddings to reduce file size)
    entity_data_serializable = []
    for entity in entity_data:
        entity_dict = {
            'entity_id': entity.entity_id,
            'name': entity.name,
            'roles': entity.roles,
            'doc_id': entity.doc_id,
            'description': entity.description,
            'has_embedding': entity.embedding is not None
        }
        entity_data_serializable.append(entity_dict)
    
    with open(output_path / "entity_data.json", 'w') as f:
        json.dump(entity_data_serializable, f, indent=2)
    
    # Save similarity results (excluding full similarity matrix to reduce size)
    similarity_results_serializable = similarity_results.copy()
    if 'similarity_matrix' in similarity_results_serializable:
        del similarity_results_serializable['similarity_matrix']  # Too large for JSON
    
    # Use the JSON-serializable version of hypernodes
    if 'hypernodes_serializable' in similarity_results_serializable:
        similarity_results_serializable['hypernodes'] = similarity_results_serializable['hypernodes_serializable']
        del similarity_results_serializable['hypernodes_serializable']
    
    with open(output_path / "similarity_results.json", 'w') as f:
        json.dump(similarity_results_serializable, f, indent=2)
    
    # Save analysis
    with open(output_path / "similarity_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run hypernode similarity exploration."""
    print("HYPERNODE ENTITY SIMILARITY EXPLORATION")
    print("=" * 80)
    
    # Configuration
    num_documents = 200  # Start with 50 documents
    similarity_threshold = 0.85  # High threshold for same-entity detection
    output_dir = f"/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_results_{int(time.time())}"
    
    # Step 1: Load GSW files
    gsw_structures, doc_ids = load_gsw_files(num_documents)
    if not gsw_structures:
        print("No GSW files loaded. Exiting.")
        return
    
    # Step 2: Extract entity data
    entity_data = extract_entity_data(gsw_structures, doc_ids)
    if not entity_data:
        print("No entities extracted. Exiting.")
        return
    
    print(f"Entity statistics:")
    print(f"  Total entities: {len(entity_data)}")
    print(f"  Entities with roles: {len([e for e in entity_data if e.roles])}")
    print(f"  Average roles per entity: {np.mean([len(e.roles) for e in entity_data]):.1f}")
    
    # Step 3: Generate embeddings
    entity_data = generate_embeddings(entity_data)
    entities_with_embeddings = [e for e in entity_data if e.embedding is not None]
    
    if not entities_with_embeddings:
        print("No embeddings generated. Cannot proceed with similarity detection.")
        return
    
    print(f"Successfully generated embeddings for {len(entities_with_embeddings)} entities")
    
    # Step 4: Find entity similarities
    similarity_results = find_entity_similarities(entity_data, similarity_threshold)
    if not similarity_results:
        print("No similarity results. Exiting.")
        return
    
    # Step 5: Analyze similarities
    analysis = analyze_similarities(entity_data, similarity_results)
    
    # Step 6: Display results
    print_similarity_summary(analysis, similarity_results)
    
    # Step 7: Save results
    save_results(entity_data, similarity_results, analysis, output_dir)
    
    print(f"\n{'='*80}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(entity_data)} entities from {num_documents} documents")
    print(f"Found {analysis.get('entity_pairs_found', 0)} similar entity pairs")
    print(f"Created {analysis.get('hypernodes_created', 0)} hypernodes")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()