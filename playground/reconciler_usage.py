#!/usr/bin/env python3
"""
Example: Using the Reconciler Separately

This example demonstrates how to:
1. Run GSWProcessor and save outputs to local
2. Load the saved outputs later
3. Run the reconciler separately with different strategies
"""

import os
from gsw_memory import GSWProcessor, reconcile_gsw_outputs
from gsw_memory.utils.loaders import load_operator_outputs


def main():
    # Load your saved GSW outputs
    loaded_outputs = load_operator_outputs("/mnt/SSD1/chenda/gsw-memory/test_output")
    output_path = "/mnt/SSD1/chenda/gsw-memory/test_output/reconciled_local"
    os.makedirs(output_path, exist_ok=True)
    print(f"✅ Loaded {len(loaded_outputs)} documents from saved outputs")
    print()
    
    # Run reconciler with local strategy and exact matching
    print("=== Running reconciler with LOCAL strategy and EXACT matching ===")
    local_reconciled = reconcile_gsw_outputs(
        processor_outputs=loaded_outputs,
        strategy="local",
        matching_approach="exact",
        output_dir=output_path,
        save_statistics=True,
        enable_visualization=True
    )
    
    print(f"✅ Local reconciliation complete: {len(local_reconciled)} reconciled documents")
    print()
    
    # Show results for each document
    print("=== Reconciliation Results ===")
    for i, doc_gsw in enumerate(local_reconciled):
        print(f"Document {i}:")
        print(f"  - Entities: {len(doc_gsw.entity_nodes)}")
        print(f"  - Verb phrases: {len(doc_gsw.verb_phrase_nodes)}")
        print(f"  - Questions: {sum(len(vp.questions) for vp in doc_gsw.verb_phrase_nodes)}")
        print(f"  - Conversation nodes: {len(doc_gsw.conversation_nodes)}")
        print(f"  - Conversation participant edges: {len(doc_gsw.conversation_participant_edges)}")
        print(f"  - Conversation topic edges: {len(doc_gsw.conversation_topic_edges)}")
        print(f"  - Conversation space edges: {len(doc_gsw.conversation_space_edges)}")
        print(f"  - Conversation time edges: {len(doc_gsw.conversation_time_edges)}")
        print()
    
    print("=== Summary ===")
    print("📁 Original outputs: /mnt/SSD1/chenda/gsw-memory/test_output")
    print("📁 Local reconciliation: reconciled_local/")
    print()
    print("The reconciler has:")
    print("- Merged entities across chunks within each document")
    print("- Reconciled conversation nodes and edges")
    print("- Updated entity references in conversation edges")
    print("- Preserved document separation (local strategy)")
    print()
    print("You can now use these reconciled GSW structures for:")
    print("- Entity summary aggregation")
    print("- Question answering")
    print("- Knowledge graph analysis")
    print("- Conversation analysis")


if __name__ == "__main__":
    main()
    