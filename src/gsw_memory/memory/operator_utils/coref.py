"""
Coreference Resolution Operator.

This module contains the CorefOperator class for performing coreference resolution
on text documents using LLM-based processing.
"""

from bespokelabs import curator

from ...prompts.operator_prompts import CorefPrompts
from .chunk import chunk_text, chunk_text_tokencount
from .utils import estimate_tokens


class CorefOperator(curator.LLM):
    """Curator class for performing coreference resolution."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to perform coreference resolution."""
        return [
            {"role": "system", "content": CorefPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CorefPrompts.USER_PROMPT_TEMPLATE.format(text=input["text"]),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract resolved text."""
        result = {
            "text": response["choices"][0]["message"]["content"],
            "idx": input["idx"],  # idx is always doc_idx
        }
        
        # If chunk_idx is present in input, include it in output
        if "chunk_idx" in input:
            result["chunk_idx"] = input["chunk_idx"]
            
        return [result]


def process_single_coref_batch(documents: list, model_name: str = "gpt-4o") -> dict:
    """Process coreference resolution for multiple short documents (<= 3000 tokens each) in parallel."""
    coref_model = CorefOperator(
        model_name=model_name,
        generation_params={"temperature": 0.0, "max_tokens": 4000},
    )
    
    # Prepare inputs for all documents
    coref_inputs = [
        {"text": doc["text"], "idx": doc["idx"]}
        for doc in documents
    ]
    
    # Process all documents in parallel
    coref_responses = coref_model(coref_inputs)
    
    # Create resolved documents mapping
    resolved_documents = {
        resp["idx"]: resp["text"] for resp in coref_responses.dataset
    }
    
    return resolved_documents


def process_chunked_coref_batch(documents: list, model_name: str = "gpt-4o") -> dict:
    """Process coreference resolution for multiple long documents (> 3000 tokens each) by chunking each."""
    coref_model = CorefOperator(
        model_name=model_name,
        generation_params={"temperature": 0.0, "max_tokens": 4000},
    )
    
    # Prepare inputs for all chunks from all documents
    all_coref_inputs = []
    
    for doc in documents:
        doc_idx = doc["idx"]
        text = doc["text"]
        
        # Chunk the text with chunk_size=3000 (approximate token count)
        chunks = chunk_text_tokencount(text, max_tokens=3000, overlap_tokens=500)
        
        # Prepare inputs for each chunk of this document
        for chunk in chunks:
            chunk_idx = f"{doc_idx}_{chunk['idx']}"
            all_coref_inputs.append({
                "text": chunk["text"], 
                "idx": doc_idx,  # idx is always doc_idx
                "chunk_idx": chunk_idx  # chunk_idx for chunk identifier
            })
    print(f"Processing {len(all_coref_inputs)} chunks in parallel")
    # Process all chunks in parallel
    coref_responses = coref_model(all_coref_inputs)
    
    # Group resolved chunks back by document
    resolved_documents = {}
    for doc in documents:
        doc_idx = doc["idx"]
        resolved_chunks = []
        
        # Collect and sort chunks for this document
        chunk_responses = [resp for resp in coref_responses.dataset if resp["idx"] == doc_idx]
        def extract_chunk_number(chunk_idx):
            return int(chunk_idx.split("_")[1])
        chunk_responses.sort(key=lambda x: extract_chunk_number(x["chunk_idx"]))
        resolved_chunks = [resp["text"] for resp in chunk_responses]
        
        # Combine chunks back into document
        resolved_documents[doc_idx] = " ".join(resolved_chunks)
    
    return resolved_documents
