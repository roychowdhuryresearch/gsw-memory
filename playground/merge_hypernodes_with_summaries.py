#!/usr/bin/env python3
"""
Merge similar hypernodes using their generated summaries and embeddings.
This leverages the rich summaries created by GPT-4o for better clustering.
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


def find_similar_hypernodes_by_summary(summaries: Dict[str, Any],
                                      embeddings: Dict[str, np.ndarray],
                                      similarity_threshold: float = 0.75) -> Dict[str, Any]:
    """Find similar hypernodes based on their summary embeddings."""
    if not SIMILARITY_AVAILABLE or not embeddings:
        print("Cannot compute similarities without embeddings")
        return {}
    
    print(f"\nFinding similar hypernodes with threshold {similarity_threshold}...")
    
    # Prepare data
    hypernode_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[hid] for hid in hypernode_ids])
    
    print(f"Computing similarities for {len(embedding_matrix)} hypernode summaries")
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    # Find pairs above threshold
    merge_candidates = []
    similarity_connections = []
    
    for i in range(len(hypernode_ids)):
        for j in range(i + 1, len(hypernode_ids)):
            similarity = similarity_matrix[i, j]
            
            if similarity >= similarity_threshold:
                hid_a = hypernode_ids[i]
                hid_b = hypernode_ids[j]
                
                summary_a = summaries[hid_a]
                summary_b = summaries[hid_b]
                
                merge_candidates.append({
                    'hypernode_a': {
                        'id': hid_a,
                        'name': summary_a['name'],
                        'entity_variations': summary_a['entity_variations'],
                        'summary': summary_a['summary'][:200] + '...'  # Truncate for display
                    },
                    'hypernode_b': {
                        'id': hid_b,
                        'name': summary_b['name'],
                        'entity_variations': summary_b['entity_variations'],
                        'summary': summary_b['summary'][:200] + '...'
                    },
                    'similarity': float(similarity)
                })
                similarity_connections.append((i, j))
    
    print(f"Found {len(merge_candidates)} hypernode pairs above threshold")
    
    # Create merged hypernodes using connected components
    merged_hypernodes = {}
    
    if similarity_connections:
        # Build adjacency matrix
        n_hypernodes = len(hypernode_ids)
        adjacency_matrix = sp.lil_matrix((n_hypernodes, n_hypernodes))
        
        for i, j in similarity_connections:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
        
        # Find connected components
        n_components, labels = connected_components(adjacency_matrix, directed=False)
        
        # Group hypernodes by merged cluster
        merge_groups = defaultdict(list)
        for idx, merged_id in enumerate(labels):
            hid = hypernode_ids[idx]
            merge_groups[merged_id].append(hid)
        
        # Create merged hypernodes
        for merged_id, hid_list in merge_groups.items():
            if len(hid_list) > 1:  # Only include actual merges
                # Collect all data from summaries
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
                
                # Choose primary name (shortest or most common)
                primary_name = min(all_names, key=len)
                
                # Create merged summary by combining individual summaries
                merged_summary = f"{primary_name} is an entity that appears across multiple contexts. "
                merged_summary += " ".join(all_summaries)
                
                merged_hypernodes[f"merged_{merged_id}"] = {
                    'merged_hypernode_name': primary_name,
                    'original_hypernodes': hid_list,
                    'original_names': list(set(all_names)),
                    'all_variations': list(set(all_variations)),
                    'source_documents': list(all_documents),
                    'entities_included': all_entities,
                    'merged_summary': merged_summary,
                    'individual_summaries': all_summaries
                }
        
        print(f"Created {len(merged_hypernodes)} merged hypernodes")
    
    return {
        'merge_candidates': merge_candidates,
        'similarity_threshold': similarity_threshold,
        'merged_hypernodes': merged_hypernodes,
        'n_original': len(hypernode_ids)
    }


def analyze_merges_for_specific_entities(merged_hypernodes: Dict[str, Any], 
                                        target_names: List[str] = ["Nicki Minaj"]) -> None:
    """Check if specific entities were successfully merged."""
    print("\n" + "="*60)
    print("CHECKING TARGET ENTITY MERGES")
    print("="*60)
    
    for target_name in target_names:
        found = False
        for mid, mdata in merged_hypernodes.items():
            # Check in all variations
            if any(target_name.lower() in var.lower() for var in mdata['all_variations']):
                found = True
                print(f"\n✓ Found '{target_name}' in merged hypernode {mid}:")
                print(f"  Primary name: {mdata['merged_hypernode_name']}")
                print(f"  Merged from {len(mdata['original_hypernodes'])} hypernodes: {mdata['original_hypernodes']}")
                print(f"  All name variations: {', '.join(mdata['all_variations'][:5])}")
                print(f"  Source documents: {', '.join(mdata['source_documents'][:5])}")
                print(f"  Merged summary preview: {mdata['merged_summary'][:300]}...")
                
        if not found:
            print(f"\n✗ '{target_name}' was NOT merged (may still be in separate hypernodes)")


def main():
    """Main function to merge hypernodes using summaries."""
    print("HYPERNODE MERGING USING SUMMARIES")
    print("=" * 60)
    
    # Configuration
    summaries_dir = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_summaries_1754877299"
    similarity_threshold = 0.75  # Lower threshold for summary-based merging
    output_dir = f"/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/merged_hypernodes_{int(time.time())}"
    
    # Step 1: Load hypernode summaries and embeddings
    summaries, embeddings = load_hypernode_summaries(summaries_dir)
    print(f"Loaded {len(summaries)} hypernode summaries")
    
    if not embeddings:
        print("ERROR: No embeddings found. Cannot perform similarity-based merging.")
        print("Please ensure hypernode_embeddings.json exists in the summaries directory.")
        return
    
    # Step 2: Find and merge similar hypernodes
    merge_results = find_similar_hypernodes_by_summary(
        summaries, embeddings, similarity_threshold
    )
    
    # Step 3: Analyze results
    merged = merge_results['merged_hypernodes']
    
    print("\n" + "="*60)
    print("MERGE STATISTICS")
    print("="*60)
    print(f"Original hypernodes: {merge_results['n_original']}")
    print(f"Merged clusters created: {len(merged)}")
    print(f"Hypernodes involved in merges: {sum(len(m['original_hypernodes']) for m in merged.values())}")
    
    # Show top merges
    print("\n" + "="*60)
    print("TOP MERGED HYPERNODES")
    print("="*60)
    
    # Sort by number of original hypernodes merged
    sorted_merges = sorted(merged.items(), 
                          key=lambda x: len(x[1]['original_hypernodes']), 
                          reverse=True)
    
    for mid, mdata in sorted_merges[:5]:
        print(f"\n{mid}: {mdata['merged_hypernode_name']}")
        print(f"  Merged {len(mdata['original_hypernodes'])} hypernodes: {mdata['original_hypernodes']}")
        print(f"  Name variations: {', '.join(mdata['original_names'])}")
        print(f"  Documents: {len(mdata['source_documents'])} documents")
        print(f"  Summary preview: {mdata['merged_summary'][:200]}...")
    
    # Step 4: Check specific entities
    analyze_merges_for_specific_entities(merged, ["Nicki Minaj", "Lothair II", "British"])
    
    # Step 5: Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save merged hypernodes
    with open(Path(output_dir) / "merged_hypernodes.json", 'w') as f:
        json.dump(merge_results['merged_hypernodes'], f, indent=2)
    
    # Save merge candidates for analysis
    with open(Path(output_dir) / "merge_candidates.json", 'w') as f:
        json.dump(merge_results['merge_candidates'], f, indent=2)
    
    # Create final consolidated summary list
    final_summaries = []
    
    # Add unmerged summaries
    merged_hids = set()
    for mdata in merged.values():
        merged_hids.update(mdata['original_hypernodes'])
    
    for hid, summary in summaries.items():
        if hid not in merged_hids:
            final_summaries.append(summary)
    
    # Add merged summaries
    for mid, mdata in merged.items():
        final_summaries.append({
            'hypernode_id': mid,
            'name': mdata['merged_hypernode_name'],
            'summary': mdata['merged_summary'],
            'entity_variations': mdata['all_variations'],
            'source_documents': mdata['source_documents'],
            'entities_included': mdata['entities_included'],
            'is_merged': True,
            'original_hypernodes': mdata['original_hypernodes']
        })
    
    with open(Path(output_dir) / "final_summaries.json", 'w') as f:
        json.dump(final_summaries, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Final hypernode count: {len(final_summaries)}")
    print(f"  Unmerged: {len(final_summaries) - len(merged)}")
    print(f"  Merged: {len(merged)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()