#!/usr/bin/env python3
"""
Apply connected components clustering to hypernodes using their summary embeddings.
This follows the same approach as test_hypernode_clustering.py but at the hypernode level.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys
import numpy as np
from collections import defaultdict

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Similarity computation imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp
    SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn or scipy not available. Install with: pip install scikit-learn scipy")
    SIMILARITY_AVAILABLE = False


def load_hypernode_summaries(summaries_dir: str) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    """Load hypernode summaries and embeddings from directory."""
    summaries_path = Path(summaries_dir) / "hypernode_summaries.json"
    embeddings_path = Path(summaries_dir) / "hypernode_embeddings.json"
    
    print(f"Loading summaries from: {summaries_path}")
    
    # Load summaries
    with open(summaries_path, 'r') as f:
        summaries = json.load(f)
    
    # Convert to dict indexed by hypernode_id
    summaries_dict = {s['hypernode_id']: s for s in summaries}
    
    # Load embeddings if available
    embeddings_dict = None
    if embeddings_path.exists():
        print(f"Loading embeddings from: {embeddings_path}")
        with open(embeddings_path, 'r') as f:
            embeddings_data = json.load(f)
        
        # Convert to dict
        embeddings_dict = {}
        for hid, embedding in zip(embeddings_data['hypernode_ids'], embeddings_data['embeddings']):
            embeddings_dict[hid] = np.array(embedding)
    else:
        print("No embeddings file found")
    
    return summaries_dict, embeddings_dict


def find_hypernode_connected_components(summaries: Dict[str, Any],
                                      embeddings: Dict[str, np.ndarray],
                                      similarity_threshold: float = 0.75) -> Dict[str, Any]:
    """Find connected components of hypernodes using similarity threshold."""
    if not SIMILARITY_AVAILABLE or not embeddings:
        print("Cannot compute similarities without embeddings")
        return {}
    
    print(f"\nFinding hypernode similarities with threshold {similarity_threshold}...")
    
    # Prepare data for similarity computation
    hypernode_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[hid] for hid in hypernode_ids])
    
    print(f"Computing similarities for {len(embedding_matrix)} hypernode summaries")
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    # Find similarity connections above threshold
    similarity_connections = []
    similarity_pairs = []
    
    for i in range(len(hypernode_ids)):
        for j in range(i + 1, len(hypernode_ids)):
            similarity = similarity_matrix[i, j]
            
            if similarity >= similarity_threshold:
                hid_a = hypernode_ids[i]
                hid_b = hypernode_ids[j]
                
                similarity_connections.append((i, j))
                similarity_pairs.append({
                    'hypernode_a': {
                        'id': hid_a,
                        'name': summaries[hid_a]['name'],
                        'summary': summaries[hid_a]['summary'][:100] + '...'
                    },
                    'hypernode_b': {
                        'id': hid_b,
                        'name': summaries[hid_b]['name'],
                        'summary': summaries[hid_b]['summary'][:100] + '...'
                    },
                    'similarity': float(similarity)
                })
    
    print(f"Found {len(similarity_pairs)} hypernode pairs above threshold {similarity_threshold}")
    
    # Create super-hypernodes using connected components
    super_hypernodes = {}
    super_hypernode_labels = []
    
    if similarity_connections:
        # Build adjacency matrix (following exact pattern from test_hypernode_clustering.py)
        n_hypernodes = len(hypernode_ids)
        adjacency_matrix = sp.lil_matrix((n_hypernodes, n_hypernodes))
        
        for i, j in similarity_connections:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
        
        # Find connected components (super-hypernodes)
        n_components, labels = connected_components(adjacency_matrix, directed=False)
        super_hypernode_labels = labels.tolist()
        
        # Group hypernodes by super-hypernode
        super_hypernodes_temp = defaultdict(list)
        for hypernode_idx, super_hypernode_id in enumerate(labels):
            hid = hypernode_ids[hypernode_idx]
            super_hypernodes_temp[super_hypernode_id].append(hid)
        
        print(f"Created {n_components} super-hypernodes from connected components")
        
        # Convert to final format with names
        for super_hid, hid_list in super_hypernodes_temp.items():
            # Collect all data from constituent hypernodes
            all_entities = []
            all_names = []
            all_variations = []
            all_documents = set()
            all_summaries = []
            
            for hid in hid_list:
                summary = summaries[hid]
                all_names.append(summary['name'])
                all_variations.extend(summary['entity_variations'])
                all_documents.update(summary['source_documents'])
                all_entities.extend(summary['entities_included'])
                all_summaries.append(summary['summary'])
            
            # Choose primary name (shortest as per original logic)
            primary_name = min(all_names, key=len) if all_names else f"SuperHypernode_{super_hid}"
            
            super_hypernodes[str(super_hid)] = {
                'super_hypernode_name': primary_name,
                'constituent_hypernodes': hid_list,
                'constituent_names': list(set(all_names)),
                'all_entity_variations': list(set(all_variations)),
                'source_documents': list(all_documents),
                'entities_included': all_entities,
                'individual_summaries': all_summaries,
                'hypernode_count': len(hid_list),
                'entity_count': len(all_entities)
            }
    else:
        super_hypernodes = {}
        super_hypernode_labels = [i for i in range(len(hypernode_ids))]  # Each hypernode is its own component
    
    return {
        'similarity_pairs': similarity_pairs,
        'similarity_threshold': similarity_threshold,
        'super_hypernodes': super_hypernodes,
        'super_hypernode_labels': super_hypernode_labels,
        'n_super_hypernodes': len(set(super_hypernode_labels)),
        'original_hypernode_count': len(hypernode_ids)
    }


def analyze_super_hypernodes(results: Dict[str, Any]) -> None:
    """Analyze the super-hypernode results (following pattern from test_hypernode_clustering.py)."""
    print("\n" + "="*80)
    print("SUPER-HYPERNODE ANALYSIS SUMMARY")
    print("="*80)
    
    if not results:
        print("No super-hypernode results found")
        return
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"{'='*40}")
    print(f"Original hypernodes: {results['original_hypernode_count']}")
    print(f"Similarity threshold: {results['similarity_threshold']}")
    print(f"Hypernode pairs found: {len(results['similarity_pairs'])}")
    print(f"Super-hypernodes created: {results['n_super_hypernodes']}")
    print(f"Multi-hypernode clusters: {len(results['super_hypernodes'])}")
    
    super_hypernodes = results['super_hypernodes']
    
    # Super-hypernode size statistics
    if super_hypernodes:
        sizes = [sh['hypernode_count'] for sh in super_hypernodes.values()]
        print(f"\nSUPER-HYPERNODE SIZE STATISTICS:")
        print(f"{'='*40}")
        print(f"Min hypernode count: {min(sizes)}")
        print(f"Max hypernode count: {max(sizes)}")
        print(f"Average hypernode count: {np.mean(sizes):.1f}")
        print(f"Median hypernode count: {np.median(sizes):.1f}")
    
    # Top similarity matches
    if results['similarity_pairs']:
        print(f"\nTOP HYPERNODE MATCHES:")
        print(f"{'='*40}")
        sorted_pairs = sorted(results['similarity_pairs'], key=lambda x: x['similarity'], reverse=True)
        for i, match in enumerate(sorted_pairs[:5], 1):
            print(f"\n{i}. Similarity: {match['similarity']:.3f}")
            print(f"   Hypernode A: {match['hypernode_a']['name']}")
            print(f"   Hypernode B: {match['hypernode_b']['name']}")
            print(f"   A Summary: {match['hypernode_a']['summary']}")
            print(f"   B Summary: {match['hypernode_b']['summary']}")
    
    # Show largest super-hypernodes
    if super_hypernodes:
        print(f"\nLARGEST SUPER-HYPERNODES:")
        print(f"{'='*40}")
        sorted_super = sorted(super_hypernodes.items(), 
                            key=lambda x: x[1]['hypernode_count'], 
                            reverse=True)
        
        for sid, sdata in sorted_super[:5]:
            print(f"\nSuper-hypernode {sid}: '{sdata['super_hypernode_name']}'")
            print(f"  Constituent hypernodes: {sdata['hypernode_count']} ({sdata['constituent_hypernodes']})")
            print(f"  Constituent names: {', '.join(sdata['constituent_names'][:5])}")
            print(f"  Total entities: {sdata['entity_count']}")
            print(f"  Documents: {len(sdata['source_documents'])} documents")
            
            # Show entity variations for key entities
            variations = sdata['all_entity_variations'][:10]
            print(f"  Key entity variations: {', '.join(variations)}")
    
    # Check for specific entities (like Nicki Minaj)
    print(f"\nSPECIFIC ENTITY CHECKS:")
    print(f"{'='*40}")
    target_entities = ["Nicki Minaj", "Lothair II", "British"]
    
    for target in target_entities:
        found = False
        for sid, sdata in super_hypernodes.items():
            if any(target.lower() in var.lower() for var in sdata['all_entity_variations']):
                found = True
                print(f"\n✓ Found '{target}' in super-hypernode {sid} ('{sdata['super_hypernode_name']}')")
                print(f"  Combined from {sdata['hypernode_count']} hypernodes: {sdata['constituent_hypernodes']}")
                print(f"  All variations: {', '.join([v for v in sdata['all_entity_variations'] if target.lower() in v.lower()])}")
        
        if not found:
            print(f"\n✗ '{target}' not found in any super-hypernode (may be singleton)")


def main():
    """Main function to create super-hypernodes using connected components."""
    print("HYPERNODE CONNECTED COMPONENTS CLUSTERING")
    print("=" * 60)
    
    # Configuration
    summaries_dir = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_summaries_1755045125"
    similarity_threshold = 0.60  # Can experiment with different thresholds
    output_dir = f"/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/super_hypernodes_{int(time.time())}"
    
    # Step 1: Load hypernode summaries and embeddings
    summaries, embeddings = load_hypernode_summaries(summaries_dir)
    print(f"Loaded {len(summaries)} hypernode summaries")
    
    if not embeddings:
        print("ERROR: No embeddings found. Cannot perform similarity-based clustering.")
        return
    
    # Step 2: Find connected components
    results = find_hypernode_connected_components(
        summaries, embeddings, similarity_threshold
    )
    
    # Step 3: Analyze results
    analyze_super_hypernodes(results)
    
    # Step 4: Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save super-hypernodes
    with open(Path(output_dir) / "super_hypernodes.json", 'w') as f:
        json.dump(results['super_hypernodes'], f, indent=2)
    
    # Save similarity pairs
    with open(Path(output_dir) / "similarity_pairs.json", 'w') as f:
        json.dump(results['similarity_pairs'], f, indent=2)
    
    # Save complete results
    results_to_save = results.copy()
    with open(Path(output_dir) / "complete_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CLUSTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Original hypernodes: {results['original_hypernode_count']}")
    print(f"Super-hypernodes: {results['n_super_hypernodes']}")
    print(f"Multi-hypernode clusters: {len(results['super_hypernodes'])}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()