#!/usr/bin/env python3
"""
Simple test script for the GSW operator functionality.
Run this script to test the operator implementation.
"""

from gsw_memory.memory import GSWProcessor

processor = GSWProcessor(
    model_name="gpt-4o",
    enable_coref=False,
    enable_chunking=False,
    enable_context=False,
    chunk_size=3,
    overlap=1,
    coref_chunk_size=20
)

test_document = """
John walked into the coffee shop. He ordered a large latte from the barista. 
The barista, whose name was Sarah, smiled at him. She prepared the drink carefully. 
John paid for his coffee and sat down at a table near the window. 
He opened his laptop and began working on his presentation.
"""

print("Processing single document...")
gsw_structures = processor.process_documents([test_document])

# import sys
# import os
# import json

# # Add the src directory to the path
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# from gsw_memory.memory import GSWProcessor

# def test_basic_processing():
#     """Test basic single document processing."""
#     print("=" * 50)
#     print("Test 1: Basic Single Document Processing")
#     print("=" * 50)
    
#     processor = GSWProcessor(
#         model_name="gpt-4o",
#         enable_coref=True,
#         enable_chunking=True,
#         enable_context=True,
#         chunk_size=3,
#         overlap=1,
#         coref_chunk_size=20
#     )
    
#     test_document = """
#     John walked into the coffee shop. He ordered a large latte from the barista. 
#     The barista, whose name was Sarah, smiled at him. She prepared the drink carefully. 
#     John paid for his coffee and sat down at a table near the window. 
#     He opened his laptop and began working on his presentation.
#     """
    
#     print("Processing single document...")
#     gsw_structures = processor.process_documents([test_document])
    
#     print(f"Generated {len(gsw_structures)} GSW structures")
#     for i, gsw in enumerate(gsw_structures):
#         print(f"\nChunk {i}:")
#         print(f"  Entities: {len(gsw.entity_nodes)}")
#         print(f"  Verb phrases: {len(gsw.verb_phrase_nodes)}")
        
#         # Show entity details
#         if gsw.entity_nodes:
#             print("  Entity names:", [e.name for e in gsw.entity_nodes])
    
#     return gsw_structures

# def test_multiple_documents():
#     """Test multiple documents processing."""
#     print("\n" + "=" * 50)
#     print("Test 2: Multiple Documents Processing")
#     print("=" * 50)
    
#     processor = GSWProcessor(
#         model_name="gpt-4o",
#         enable_coref=True,
#         enable_chunking=True,
#         enable_context=True,
#         chunk_size=3,
#         overlap=1
#     )
    
#     documents = [
#         "Alice went to the library. She checked out three books about history. The librarian helped her find additional resources.",
#         "Bob visited the same library later. He was looking for science fiction novels. Alice was still there reading.",
#         "The librarian organized the returned books. She noticed that many students were interested in history topics."
#     ]
    
#     print("Processing multiple documents...")
#     multi_gsw_structures = processor.process_documents(documents)
    
#     print(f"Generated {len(multi_gsw_structures)} GSW structures from {len(documents)} documents")
#     for i, gsw in enumerate(multi_gsw_structures):
#         print(f"\nChunk {i}:")
#         print(f"  Entities: {len(gsw.entity_nodes)}")
#         if gsw.entity_nodes:
#             print(f"  Entity names: {[e.name for e in gsw.entity_nodes]}")
#         print(f"  Verb phrases: {len(gsw.verb_phrase_nodes)}")
    
#     return multi_gsw_structures

# def test_save_outputs():
#     """Test saving outputs to disk."""
#     print("\n" + "=" * 50)
#     print("Test 3: Save Outputs to Disk")
#     print("=" * 50)
    
#     processor = GSWProcessor(
#         model_name="gpt-4o",
#         enable_coref=True,
#         enable_chunking=True,
#         enable_context=True
#     )
    
#     documents = [
#         "The scientist conducted an experiment. She recorded the results carefully.",
#         "Her colleague reviewed the data. He found some interesting patterns."
#     ]
    
#     output_dir = "test_outputs"
    
#     print("Processing with save outputs enabled...")
#     saved_gsw_structures = processor.process_documents(
#         documents,
#         output_dir=output_dir,
#         save_intermediates=True
#     )
    
#     print(f"Outputs saved to: {output_dir}")
#     print("Check the following directories:")
#     print("  - networks/: Parsed GSW structures")
#     print("  - networks_raw/: Raw LLM responses")
#     print("  - coref/: Coreference resolved texts")
#     print("  - chunks/: Individual chunks")
#     print("  - context/: Generated contexts")
    
#     return saved_gsw_structures

# def test_configuration_options():
#     """Test different configuration options."""
#     print("\n" + "=" * 50)
#     print("Test 4: Different Configuration Options")
#     print("=" * 50)
    
#     processor = GSWProcessor(model_name="gpt-4o")
    
#     test_document = """
#     The teacher explained the lesson. Students took notes carefully. 
#     They asked questions about the material. The teacher answered thoroughly.
#     """
    
#     print("Test 4a: No coreference resolution")
#     no_coref_gsw = processor.process_documents(
#         [test_document],
#         enable_coref=False
#     )
#     print(f"Generated {len(no_coref_gsw)} structures")
    
#     print("\nTest 4b: No chunking")
#     no_chunking_gsw = processor.process_documents(
#         [test_document],
#         enable_chunking=False
#     )
#     print(f"Generated {len(no_chunking_gsw)} structures")
    
#     print("\nTest 4c: No context generation")
#     no_context_gsw = processor.process_documents(
#         [test_document],
#         enable_context=False
#     )
#     print(f"Generated {len(no_context_gsw)} structures")

# def inspect_gsw_structure(gsw_structures):
#     """Detailed inspection of a GSW structure."""
#     print("\n" + "=" * 50)
#     print("Test 5: Inspect Generated GSW Structure")
#     print("=" * 50)
    
#     if gsw_structures:
#         sample_gsw = gsw_structures[0]
#         print("Sample GSW Structure:")
#         print("\nEntities:")
#         for entity in sample_gsw.entity_nodes:
#             print(f"  - {entity.name} (ID: {entity.id})")
#             for role in entity.roles:
#                 print(f"    Role: {role.role}")
#                 print(f"    States: {role.states}")
        
#         print("\nVerb Phrases:")
#         for vp in sample_gsw.verb_phrase_nodes:
#             print(f"  - {vp.phrase} (ID: {vp.id})")
#             for question in vp.questions:
#                 print(f"    Q: {question.text}")
#                 print(f"    A: {question.answers}")
        
#         # Convert to JSON for inspection
#         gsw_dict = sample_gsw.model_dump(mode="json")
#         print("\nJSON representation (first 500 chars):")
#         json_str = json.dumps(gsw_dict, indent=2)
#         print(json_str[:500] + ("..." if len(json_str) > 500 else ""))

# def test_error_handling():
#     """Test error handling."""
#     print("\n" + "=" * 50)
#     print("Test 6: Error Handling")
#     print("=" * 50)
    
#     processor = GSWProcessor(model_name="gpt-4o")
    
#     print("Test 6a: Empty document list")
#     try:
#         empty_result = processor.process_documents([])
#         print(f"Result: {len(empty_result)} structures")
#     except Exception as e:
#         print(f"Error: {e}")
    
#     print("\nTest 6b: Very short document")
#     try:
#         short_result = processor.process_documents(["Hi."])
#         print(f"Result: {len(short_result)} structures")
#     except Exception as e:
#         print(f"Error: {e}")

# def main():
#     """Run all tests."""
#     print("GSW Operator Test Suite")
#     print("=====================")
    
#     try:
#         # Test basic functionality
#         gsw_structures = test_basic_processing()
        
#         # Test multiple documents
#         multi_gsw_structures = test_multiple_documents()
        
#         # Test saving outputs
#         saved_gsw_structures = test_save_outputs()
        
#         # Test configuration options
#         test_configuration_options()
        
#         # Inspect structure details
#         inspect_gsw_structure(gsw_structures if gsw_structures else multi_gsw_structures)
        
#         # Test error handling
#         test_error_handling()
        
#         print("\n" + "=" * 50)
#         print("All tests completed successfully!")
#         print("=" * 50)
        
#     except Exception as e:
#         print(f"\nTest failed with error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()