#!/usr/bin/env python3
"""
Test global reconciliation on a subset of GSW documents.
Evaluates performance and compression ratios at different scales.
"""

import json
import time
import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import sys

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.gsw_memory.memory.reconciler import reconcile_gsw_outputs
from src.gsw_memory.memory.models import GSWStructure


def load_gsw_files(file_paths: List[str]) -> List[Dict[str, Dict]]:
    """Load GSW structures from JSON files and format as processor outputs."""
    processor_outputs = []
    
    for i, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract document ID from path (e.g., "doc_123" from path)
                doc_id = Path(file_path).parent.name
                
                # Format as processor output with chunks
                processor_output = {
                    f"doc_{doc_id}": {
                        "chunks": [
                            {
                                "chunk_id": f"{doc_id}_0",
                                "text": "",  # We don't have the original text here
                                "gsw": GSWStructure(**data)
                            }
                        ]
                    }
                }
                processor_outputs.append(processor_output)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return processor_outputs


def analyze_reconciliation_results(
    reconciled_gsw: Union[GSWStructure, List[GSWStructure]], 
    input_entities_count: int,
    elapsed_time: float,
    strategy: str
) -> Dict[str, Any]:
    """Analyze reconciliation results and compute metrics."""
    
    # Handle both single GSW and list of GSWs
    if isinstance(reconciled_gsw, list):
        # For local strategy, aggregate all entities
        all_entities = []
        for gsw in reconciled_gsw:
            if gsw and gsw.entities:
                all_entities.extend(gsw.entities)
        output_entities_count = len(all_entities)
        entities_for_analysis = all_entities
    else:
        # For global strategy, single GSW
        output_entities_count = len(reconciled_gsw.entities) if reconciled_gsw and reconciled_gsw.entities else 0
        entities_for_analysis = reconciled_gsw.entities if reconciled_gsw else []
    
    # Compute compression ratio
    compression_ratio = input_entities_count / output_entities_count if output_entities_count > 0 else 0
    
    # Analyze entity distribution
    entity_roles_distribution = {}
    for entity in entities_for_analysis:
        num_roles = len(entity.roles) if entity.roles else 0
        bucket = f"{(num_roles // 10) * 10}-{((num_roles // 10) + 1) * 10 - 1}"
        entity_roles_distribution[bucket] = entity_roles_distribution.get(bucket, 0) + 1
    
    # Find entities with most roles (potential over-merging)
    top_entities = sorted(
        entities_for_analysis, 
        key=lambda e: len(e.roles) if e.roles else 0, 
        reverse=True
    )[:5]
    
    top_entities_info = []
    for entity in top_entities:
        verb_phrases = []
        if entity.roles:
            for role in entity.roles[:10]:
                if role.verb_phrase and role.verb_phrase.verb:
                    verb_phrases.append(role.verb_phrase.verb)
        
        top_entities_info.append({
            "name": entity.name,
            "id": entity.entity_id,
            "num_roles": len(entity.roles) if entity.roles else 0,
            "role_types": list(set(verb_phrases))
        })
    
    return {
        "input_entities": input_entities_count,
        "output_entities": output_entities_count,
        "compression_ratio": round(compression_ratio, 2),
        "elapsed_time": round(elapsed_time, 2),
        "entities_per_second": round(input_entities_count / elapsed_time if elapsed_time > 0 else 0, 2),
        "entity_roles_distribution": entity_roles_distribution,
        "top_entities_by_roles": top_entities_info
    }


def count_input_entities(processor_outputs: List[Dict[str, Dict]]) -> int:
    """Count total number of input entities from processor outputs."""
    total = 0
    for output in processor_outputs:
        for doc_id, doc_data in output.items():
            if "chunks" in doc_data:
                for chunk in doc_data["chunks"]:
                    if "gsw" in chunk and chunk["gsw"].entities:
                        total += len(chunk["gsw"].entities)
    return total


def test_reconciliation_at_scale(num_documents: int, strategy: str = "global"):
    """Test reconciliation with a specific number of documents."""
    
    print(f"\n{'='*60}")
    print(f"Testing reconciliation with {num_documents} documents")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    # Get file paths from the corrected global IDs directory
    base_dir = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/logs/full_2wiki_corpus_20250710_202211/gsw_output_global_ids/networks"
    
    # Get first N document directories
    doc_dirs = sorted(glob.glob(os.path.join(base_dir, "doc_*")), 
                      key=lambda x: int(Path(x).name.replace("doc_", "")))[:num_documents]
    
    # Get GSW files from these directories
    file_paths = []
    for doc_dir in doc_dirs:
        gsw_files = glob.glob(os.path.join(doc_dir, "gsw_*.json"))
        file_paths.extend(gsw_files)
    
    print(f"Loading {len(file_paths)} GSW files from {num_documents} documents...")
    
    # Load GSW structures as processor outputs
    processor_outputs = load_gsw_files(file_paths)
    print(f"Loaded {len(processor_outputs)} processor outputs")
    
    # Count input entities
    input_entities_count = count_input_entities(processor_outputs)
    print(f"Input entities: {input_entities_count}")
    
    # Create output directory for this run
    output_dir = Path(f"/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/reconciliation_tests/{strategy}_{num_documents}docs_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run reconciliation
    print(f"Running {strategy} reconciliation...")
    start_time = time.time()
    
    try:
        reconciled_gsw = reconcile_gsw_outputs(
            processor_outputs=processor_outputs,
            strategy=strategy,
            matching_approach="exact",
            output_dir=str(output_dir),
            save_statistics=True,
            enable_visualization=False
        )
        elapsed_time = time.time() - start_time
        
        # Analyze results
        metrics = analyze_reconciliation_results(reconciled_gsw, input_entities_count, elapsed_time, strategy)
        
        # Print results
        print(f"\n{'-'*40}")
        print("RECONCILIATION RESULTS")
        print(f"{'-'*40}")
        print(f"Input entities:     {metrics['input_entities']}")
        print(f"Output entities:    {metrics['output_entities']}")
        print(f"Compression ratio:  {metrics['compression_ratio']}:1")
        print(f"Elapsed time:       {metrics['elapsed_time']} seconds")
        print(f"Processing speed:   {metrics['entities_per_second']} entities/second")
        
        print(f"\n{'-'*40}")
        print("ENTITY ROLES DISTRIBUTION")
        print(f"{'-'*40}")
        for bucket, count in sorted(metrics['entity_roles_distribution'].items()):
            print(f"  {bucket} roles: {count} entities")
        
        print(f"\n{'-'*40}")
        print("TOP ENTITIES BY ROLE COUNT")
        print(f"{'-'*40}")
        for entity_info in metrics['top_entities_by_roles']:
            print(f"  {entity_info['name'][:50]}: {entity_info['num_roles']} roles")
            print(f"    Sample verbs: {', '.join(entity_info['role_types'][:5])}")
        
        # Save detailed metrics
        metrics_file = output_dir / "reconciliation_metrics.json"
        
        save_data = {
            "metrics": metrics,
            "num_documents": num_documents,
            "strategy": strategy,
            "timestamp": time.time()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        return metrics
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nERROR: Reconciliation failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to test reconciliation at different scales."""
    
    # Test with progressively larger document sets
    test_sizes = [10]
    
    results = []
    
    for size in test_sizes:
        print(f"\n{'#'*60}")
        print(f"TESTING WITH {size} DOCUMENTS")
        print(f"{'#'*60}")
        
        # Test global strategy
        metrics = test_reconciliation_at_scale(size, strategy="global")
        if metrics:
            results.append({"size": size, "strategy": "global", "metrics": metrics})
        else:
            print(f"Global strategy failed at {size} documents")
            # Try local strategy as fallback
            print(f"\nTrying local strategy...")
            local_metrics = test_reconciliation_at_scale(size, strategy="local")
            if local_metrics:
                results.append({"size": size, "strategy": "local", "metrics": local_metrics})
            else:
                print(f"Both strategies failed at {size} documents. Stopping tests.")
                break
        
        # If global takes too long, also test local strategy
        if metrics and metrics['elapsed_time'] > 30:
            print(f"\nAlso testing local strategy for comparison...")
            local_metrics = test_reconciliation_at_scale(size, strategy="local")
            if local_metrics:
                results.append({"size": size, "strategy": "local", "metrics": local_metrics})
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*60}")
    
    for result in results:
        m = result['metrics']
        print(f"\n{result['size']} documents ({result['strategy']} strategy):")
        print(f"  - Compression: {m['compression_ratio']}:1")
        print(f"  - Time: {m['elapsed_time']}s")
        print(f"  - Speed: {m['entities_per_second']} entities/s")
    
    # Save summary
    summary_dir = Path("/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/reconciliation_tests")
    summary_dir.mkdir(exist_ok=True)
    summary_file = summary_dir / f"summary_{int(time.time())}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()