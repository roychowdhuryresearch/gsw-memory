#!/usr/bin/env python3
"""
Test script for new factual extraction prompt using GSWProcessor.

Tests the new FactualExtractionPrompts integrated with the actual GSWProcessor
to evaluate extraction quality before full integration.
"""

import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable cache for testing
os.environ["CURATOR_DISABLE_CACHE"] = "true"

# Import after setting up the path
from gsw_memory import GSWProcessor
from gsw_memory.prompts.operator_prompts import FactualExtractionPrompts
from bespokelabs import curator


class FactualGSWOperator(curator.LLM):
    """Modified GSW Operator that uses FactualExtractionPrompts instead of OperatorPrompts."""
    
    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to generate a GSW using factual extraction prompts."""
        return [
            {"role": "system", "content": FactualExtractionPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FactualExtractionPrompts.USER_PROMPT_TEMPLATE.format(
                    input_text=input["text"],
                    background_context=input.get("context", "Wikipedia article content.")
                ),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract text and graph (same as original GSWOperator)."""
        parsed_response = {
            "text": input["text"],
            "idx": input["idx"],
            "graph": response["choices"][0]["message"]["content"],
            "context": input.get("context", ""),
            "doc_idx": input.get("doc_idx", input["idx"]),
            "global_id": input.get("global_id", "unknown"),
        }
        
        # Include sentence indices if available
        if "start_sentence" in input:
            parsed_response["start_sentence"] = input["start_sentence"]
        if "end_sentence" in input:
            parsed_response["end_sentence"] = input["end_sentence"]

        return [parsed_response]


def monkey_patch_gsw_operator():
    """Temporarily replace GSWOperator with FactualGSWOperator for testing."""
    import gsw_memory.memory.operator_utils as operator_utils
    import gsw_memory.memory.operator as operator_module
    
    # Replace the GSWOperator in both modules
    operator_utils.GSWOperator = FactualGSWOperator
    operator_module.GSWOperator = FactualGSWOperator
    
    print("🔄 Monkey-patched GSWOperator to use FactualExtractionPrompts")


def load_first_n_documents(corpus_path: str, n: int = 5):
    """Load the first n documents from the 2wiki corpus."""
    print(f"Loading first {n} documents from {corpus_path}")
    
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    
    documents = corpus[:n]
    print(f"Loaded {len(documents)} documents:")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc['title']}")
    
    return documents


def test_factual_extraction(documents):
    """Test the new factual extraction prompt using GSWProcessor."""
    print("\n=== Testing Factual Extraction Prompt with GSWProcessor ===")
    
    # Monkey patch the GSWOperator to use our factual prompts
    monkey_patch_gsw_operator()
    
    # Initialize GSWProcessor with the same settings as 2wiki
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=False,
        enable_chunking=False,
        chunk_size=1,  # Smaller chunks for factual content
        overlap=0,
        enable_context=False,
        enable_spacetime=True,
    )
    
    # Extract just the document texts
    document_texts = [doc["text"] for doc in documents]
    
    print(f"Processing {len(document_texts)} documents...")
    
    # Process documents through the full GSWProcessor pipeline
    gsw_structures = processor.process_documents(document_texts)
    
    # Convert results to the expected format for compatibility with save_results
    results = []
    for i, doc in enumerate(documents):
        try:
            # Get the GSW structure for this document (should be one chunk since chunking is disabled)
            doc_data = gsw_structures[i]
            if doc_data:
                # Get the first (and only) chunk
                chunk_key = list(doc_data.keys())[0]
                chunk_data = doc_data[chunk_key]
                
                if chunk_data.get("gsw"):
                    # Convert GSWStructure to dict format for saving
                    gsw_dict = {
                        "entity_nodes": [
                            {
                                "id": entity.id,
                                "name": entity.name,
                                "roles": [
                                    {
                                        "role": role.role,
                                        "states": role.states
                                    } for role in entity.roles
                                ]
                            } for entity in chunk_data["gsw"].entity_nodes
                        ],
                        "verb_phrase_nodes": [
                            {
                                "id": vp.id,
                                "phrase": vp.phrase,
                                "questions": [
                                    {
                                        "id": q.id,
                                        "text": q.text,
                                        "answers": q.answers
                                    } for q in vp.questions
                                ]
                            } for vp in chunk_data["gsw"].verb_phrase_nodes
                        ]
                    }
                    
                    results.append({
                        "title": doc["title"],
                        "gsw_data": gsw_dict,
                        "raw_response": "GSW processed successfully",
                        "success": True
                    })
                else:
                    results.append({
                        "title": doc["title"],
                        "gsw_data": None,
                        "raw_response": "No GSW generated",
                        "error": "GSW parsing failed",
                        "success": False
                    })
            else:
                results.append({
                    "title": doc["title"],
                    "gsw_data": None,
                    "raw_response": "No document data",
                    "error": "No document data generated",
                    "success": False
                })
        except Exception as e:
            results.append({
                "title": doc["title"],
                "gsw_data": None,
                "raw_response": f"Error: {str(e)}",
                "error": str(e),
                "success": False
            })
    
    return results


def save_results(results, output_dir: str):
    """Save the extraction results for manual evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    for i, result in enumerate(results):
        title = result["title"].replace(" ", "_").replace("/", "_")
        
        # Save the GSW data if extraction succeeded
        if result["success"] and result["gsw_data"]:
            gsw_file = os.path.join(output_dir, f"{i+1}_{title}_gsw.json")
            with open(gsw_file, "w") as f:
                json.dump(result["gsw_data"], f, indent=2)
            print(f"✅ Saved GSW for '{result['title']}' to {gsw_file}")
        else:
            print(f"❌ Failed to extract GSW for '{result['title']}': {result.get('error', 'Unknown error')}")
        
        # Save raw response for debugging
        raw_file = os.path.join(output_dir, f"{i+1}_{title}_raw.txt")
        with open(raw_file, "w") as f:
            f.write(f"Title: {result['title']}\n")
            f.write(f"Success: {result['success']}\n")
            if not result["success"]:
                f.write(f"Error: {result.get('error', 'Unknown')}\n")
            f.write(f"\nRaw Response:\n{result['raw_response']}")
        
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(results),
        "successful_extractions": sum(1 for r in results if r["success"]),
        "failed_extractions": sum(1 for r in results if not r["success"]),
        "results": [
            {
                "title": r["title"],
                "success": r["success"], 
                "error": r.get("error", None)
            } for r in results
        ]
    }
    
    summary_file = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to {summary_file}")
    print(f"Success rate: {summary['successful_extractions']}/{summary['total_documents']} ({summary['successful_extractions']/summary['total_documents']*100:.1f}%)")


def main():
    """Main function to test the factual extraction prompt."""
    print("🧪 Testing Factual Extraction Prompt on 2wiki Corpus")
    
    # Configuration
    corpus_path = "/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset/2wikimultihopqa_corpus.json"
    output_dir = "factual_prompt_test_results"
    
    # Load documents
    documents = load_first_n_documents(corpus_path, n=5)
    
    # Test extraction
    results = test_factual_extraction(documents)
    
    # Save results
    save_results(results, output_dir)
    
    print(f"\n✅ Testing complete! Check {output_dir}/ for results.")
    print("\nNext steps:")
    print("1. Manually review the GSW extractions")
    print("2. Check if key factual relationships are captured")
    print("3. Identify any missing or incorrect extractions")
    print("4. Iterate on the prompt if needed")


if __name__ == "__main__":
    main()