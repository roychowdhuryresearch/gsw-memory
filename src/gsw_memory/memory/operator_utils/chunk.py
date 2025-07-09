"""
Chunking utilities for GSW operators.

This module contains various text chunking methods including sentence-based
and event boundary-based chunking.
"""

import json
import re
from typing import Dict, List, Any

from bespokelabs import curator

from ...prompts.operator_prompts import EventBoundaryPrompts
from .utils import estimate_tokens


class EventBoundaryDetector(curator.LLM):
    """Curator class for detecting event boundaries in text"""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to detect event boundaries"""
        return [
            {
                "role": "system", 
                "content": EventBoundaryPrompts.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": EventBoundaryPrompts.USER_PROMPT_TEMPLATE.format(
                    text=input["numbered_text"]
                ),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract boundary indices and event summaries"""
        try:
            content = response["choices"][0]["message"]["content"].strip()
            
            if content.upper() == "NONE":
                boundaries = []
                event_summaries = {}
            else:
                # Try to parse JSON format first
                try:
                    # Extract JSON if wrapped in markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    # Try to fix truncated JSON by adding missing closing brackets
                    if not content.strip().endswith(']'):
                        # Find the last complete entry
                        last_complete_entry = content.rfind('"}')
                        if last_complete_entry != -1:
                            content = content[:last_complete_entry + 2] + ']'
                    
                    events_data = json.loads(content)
                    boundaries = []
                    event_summaries = {}
                    
                    window_start = input.get("window_start", 0)
                    window_end = input.get("window_end", 0)
                    
                    for event in events_data:
                        if isinstance(event, dict) and "index" in event:
                            global_index = event["index"]
                            event_summary = event.get("event_summary", "")
                            
                            # Check if this index is within our window
                            if window_start <= global_index < window_end:
                                # Convert to window-relative index for boundaries list
                                relative_index = global_index - window_start
                                boundaries.append(relative_index)
                                # Store with relative index as key for consistency
                                event_summaries[str(relative_index)] = event_summary
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # Fallback: try to extract partial data using regex
                    try:
                        # Use regex to find index numbers, but validate they're in window range
                        index_pattern = r'"index":\s*(\d+)'
                        matches = re.findall(index_pattern, content)
                        window_start = input.get("window_start", 0)
                        window_end = input.get("window_end", 0)
                        
                        valid_boundaries = []
                        for match in matches:
                            global_idx = int(match)
                            if window_start <= global_idx < window_end:
                                relative_idx = global_idx - window_start
                                valid_boundaries.append(relative_idx)
                        
                        boundaries = valid_boundaries
                        event_summaries = {}
                    except Exception as regex_error:
                        boundaries = []
                        event_summaries = {}
                    
        except Exception as e:
            boundaries = []
            event_summaries = {}
        
        return [{
            "boundaries": boundaries,
            "event_summaries": event_summaries,
            "window_start": input["window_start"],
            "window_end": input["window_end"]
        }]


def split_into_numbered_sentences(text: str) -> List[str]:
    """Split text into sentences and return as list"""
    # More sophisticated sentence splitting
    # Handle common abbreviations and edge cases
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Simple sentence splitting - you might want to use spacy or nltk for better results
    sentences = []
    for sent in re.split(r'[.!?]+', text):
        sent = sent.strip()
        if sent and len(sent) > 3:  # Filter out very short fragments
            sentences.append(sent)
    
    return sentences


def create_numbered_text_explicit(sentences: List[str], start_idx: int = 0) -> str:
    """Create explicitly numbered text that's very clear for LLM"""
    numbered_lines = []
    for i, sentence in enumerate(sentences):
        global_idx = start_idx + i
        # Use clear bracket notation with index
        numbered_lines.append(f"[{global_idx}] {sentence.strip()}.")
    
    return "\n".join(numbered_lines)


def create_windowed_text(sentences: List[str], window_size: int = 5000, overlap: int = 1000, model_name: str = "gpt-4o") -> List[Dict[str, Any]]:
    """Create overlapping windows of text based on token count with explicit indexing"""
    windows = []
    start_idx = 0
    
    while start_idx < len(sentences):
        current_tokens = 0
        end_idx = start_idx
        
        # Add sentences until we reach window_size tokens
        while end_idx < len(sentences) and current_tokens < window_size:
            sentence_tokens = estimate_tokens(sentences[end_idx], model_name)
            if current_tokens + sentence_tokens > window_size:
                break
            current_tokens += sentence_tokens
            end_idx += 1
        
        # Create explicitly numbered text for this window
        window_sentences = sentences[start_idx:end_idx]
        numbered_text = create_numbered_text_explicit(window_sentences, start_idx)
        
        windows.append({
            "numbered_text": numbered_text,
            "window_start": start_idx,
            "window_end": end_idx,
            "token_count": current_tokens
        })
        
        # Move to next window with overlap
        if end_idx >= len(sentences):
            break
            
        # Calculate overlap in sentences based on token count
        overlap_tokens = 0
        overlap_start = end_idx - 1
        while overlap_start >= start_idx and overlap_tokens < overlap:
            sentence_tokens = estimate_tokens(sentences[overlap_start], model_name)
            if overlap_tokens + sentence_tokens <= overlap:
                overlap_tokens += sentence_tokens
                overlap_start -= 1
            else:
                break
        
        start_idx = overlap_start + 1
    
    return windows


def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> List[Dict]:
    """Split text into overlapping chunks using sentence-based chunking.

    Args:
        text: The input text to chunk
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks

    Returns:
        List of dictionaries containing chunked text and indices
    """
    # Split into sentences - basic split on ., ! and ?
    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    chunks = []
    i = 0
    chunk_id = 0

    while i < len(sentences):
        # Get chunk_size sentences starting from i
        chunk_sentences = sentences[i : i + chunk_size]
        if chunk_sentences:  # Only add if we have sentences
            chunks.append(
                {
                    "text": ". ".join(chunk_sentences) + ".",
                    "idx": chunk_id,
                    "start_sentence": i,
                    "end_sentence": i + len(chunk_sentences),
                }
            )
            chunk_id += 1

        # Move forward by chunk_size - overlap sentences
        i += chunk_size - overlap

    return chunks


def chunk_text_tokencount(
    text: str, 
    max_tokens: int = 3000, 
    overlap_tokens: int = 1000,
    model_name: str = "gpt-4o"
) -> List[Dict]:
    """Split text into overlapping chunks using token-count-based chunking.

    This function chunks text based on token count while respecting sentence boundaries.
    If adding a sentence would exceed the token limit, it stops at the previous sentence.

    Args:
        text: The input text to chunk
        max_tokens: Maximum number of tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        model_name: Model name for token estimation

    Returns:
        List of dictionaries containing chunked text and indices
    """
    # Split into sentences - basic split on ., ! and ?
    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    if not sentences:
        return [{"text": text, "idx": 0, "start_sentence": 0, "end_sentence": 0, "token_count": 0}]

    chunks = []
    chunk_id = 0
    i = 0

    while i < len(sentences):
        # Start building current chunk
        current_chunk_sentences = []
        current_tokens = 0
        j = i
        
        # Add sentences until we reach max_tokens
        while j < len(sentences):
            # Estimate tokens for the next sentence
            next_sentence = sentences[j]
            sentence_tokens = estimate_tokens(next_sentence, model_name)
            
            # Check if adding this sentence would exceed the limit
            if current_tokens + sentence_tokens > max_tokens:
                break
                
            # Add the sentence to current chunk
            current_chunk_sentences.append(next_sentence)
            current_tokens += sentence_tokens
            j += 1
        
        # If we couldn't add any sentences (single sentence too long), add it anyway
        if not current_chunk_sentences and i < len(sentences):
            current_chunk_sentences = [sentences[i]]
            current_tokens = estimate_tokens(sentences[i], model_name)
            j = i + 1
        
        # Create the chunk
        if current_chunk_sentences:
            chunk_text = ". ".join(current_chunk_sentences) + "."
            chunks.append({
                "text": chunk_text,
                "idx": chunk_id,
                "start_sentence": i,
                "end_sentence": j,
                "token_count": current_tokens,
                "sentence_count": len(current_chunk_sentences)
            })
            chunk_id += 1
        
        # Calculate overlap for next chunk
        if j >= len(sentences):
            break
            
        # Find overlap sentences that fit within overlap_tokens
        overlap_sentences = []
        overlap_tokens_used = 0
        overlap_start = j - 1
        
        while overlap_start >= i and overlap_tokens_used < overlap_tokens:
            sentence_tokens = estimate_tokens(sentences[overlap_start], model_name)
            if overlap_tokens_used + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentences[overlap_start])
                overlap_tokens_used += sentence_tokens
                overlap_start -= 1
            else:
                break
        
        # Move to next starting position (after overlap)
        i = overlap_start + 1 if overlap_sentences else j

    # Add any remaining sentences as a final chunk
    if i < len(sentences):
        current_chunk_sentences = sentences[i:]
        chunk_text = ". ".join(current_chunk_sentences) + "."
        current_tokens = sum(estimate_tokens(s, model_name) for s in current_chunk_sentences)
        chunks.append({
            "text": chunk_text,
            "idx": chunk_id,
            "start_sentence": i,
            "end_sentence": len(sentences),
            "token_count": current_tokens,
            "sentence_count": len(current_chunk_sentences)
        })

    return chunks


def chunk_by_event_boundaries(
    text: str, 
    model_name: str = "gpt-4o", 
    window_size: int = 5000, 
    overlap: int = 1000
) -> List[Dict[str, Any]]:
    """
    Chunk text based on LLM-detected event boundaries using sliding windows
    
    Args:
        text: Input text to chunk
        model_name: LLM model to use for boundary detection
        window_size: Size of sliding windows in tokens
        overlap: Overlap between windows in tokens
        
    Returns:
        List of chunk dictionaries with text, indices, and metadata
    """
    
    # Step 1: Split into sentences
    sentences = split_into_numbered_sentences(text)
    
    if len(sentences) == 0:
        return [{"text": text, "idx": 0, "start_sentence": 0, "end_sentence": 0, "sentence_count": 0}]
    
    # Check if document is smaller than window size
    total_tokens = sum(estimate_tokens(sentence, model_name) for sentence in sentences)
    
    if total_tokens <= window_size:
        # Use sentence-based chunking as fallback for small documents
        return chunk_text_unified(text, method="sentence", chunk_size=5, overlap=1)
    
    # Step 2: Create overlapping windows with explicit indexing
    windows = create_windowed_text(sentences, window_size, overlap, model_name)
    
    # Step 3: Detect boundaries in each window
    boundary_detector = EventBoundaryDetector(
        model_name=model_name,
        generation_params={"temperature": 0.0, "max_tokens": 500}
    )
    
    boundary_responses = boundary_detector(windows)
    
    # Step 4: Consolidate boundaries from all windows
    global_boundaries = set()
    global_event_summaries = {}  # Store event summaries
    
    for resp in boundary_responses.dataset:
        window_start = resp["window_start"]
        boundaries_found = resp["boundaries"]
        event_summaries = resp.get("event_summaries", {})
        
        for local_boundary in boundaries_found:
            global_boundary = window_start + local_boundary
            if 0 < global_boundary < len(sentences):  # Valid boundary (not at start/end)
                global_boundaries.add(global_boundary)
                # Store event summary if available (convert to string key)
                if str(local_boundary) in event_summaries:
                    global_event_summaries[global_boundary] = event_summaries[str(local_boundary)]
    
    # If no boundaries found, use sentence-based chunking as fallback
    if len(global_boundaries) == 0:
        return chunk_text_unified(text, method="sentence", chunk_size=5, overlap=1)
    
    # Step 5: Create chunks based on boundaries
    boundary_list = sorted(list(global_boundaries))
    
    chunks = []
    chunk_id = 0
    
    start_idx = 0
    for boundary_idx in boundary_list + [len(sentences)]:  # Include end
        if boundary_idx > start_idx:
            chunk_sentences = sentences[start_idx:boundary_idx]
            chunk_text = ". ".join(chunk_sentences) + "."
            
            # Get event summary for this chunk (from the boundary that starts it)
            event_summary = global_event_summaries.get(boundary_idx, "") if boundary_idx in boundary_list else ""
            
            chunks.append({
                "text": chunk_text,
                "idx": chunk_id,
                "start_sentence": start_idx,
                "end_sentence": boundary_idx,
                "sentence_count": len(chunk_sentences),
                "word_count": len(chunk_text.split()),
                "token_count": estimate_tokens(chunk_text, model_name),
                "event_summary": event_summary
            })
            chunk_id += 1
            start_idx = boundary_idx
    
    return chunks


def chunk_text_unified(
    text: str,
    method: str = "sentence",
    **kwargs
) -> List[Dict]:
    """
    Unified chunking interface that supports multiple chunking methods.
    
    Args:
        text: The input text to chunk
        method: Chunking method to use ("sentence", "event_boundary", or "token_count")
        **kwargs: Method-specific parameters
        
    Returns:
        List of dictionaries containing chunked text and indices
    """
    if method == "sentence":
        chunk_size = kwargs.get("chunk_size", 3)
        overlap = kwargs.get("overlap", 1)
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    elif method == "event_boundary":
        model_name = kwargs.get("model_name", "gpt-4o")
        window_size = kwargs.get("window_size", 5000)
        overlap = kwargs.get("overlap", 1000)
        return chunk_by_event_boundaries(
            text, 
            model_name=model_name,
            window_size=window_size,
            overlap=overlap
        )
    
    elif method == "token_count":
        max_tokens = kwargs.get("max_tokens", 3000)
        overlap_tokens = kwargs.get("overlap_tokens", 200)
        model_name = kwargs.get("model_name", "gpt-4o")
        return chunk_text_tokencount(
            text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            model_name=model_name
        )
    
    else:
        raise ValueError(f"Unknown chunking method: {method}")