"""
Simple loaders for existing GSW-Memory evaluation log data.

This module provides utilities to load already-processed data from evaluation
log directories, enabling iteration on downstream components without re-running
expensive LLM operations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..memory.models import GSWStructure


class LoaderError(Exception):
    """Exception raised during data loading."""
    pass


def load_operator_outputs(log_dir: Union[str, Path]) -> List[Dict[str, Dict]]:
    """
    Load operator outputs from gsw_results_combined.json.
    
    Args:
        log_dir: Path to the evaluation log directory
        
    Returns:
        List of document dictionaries in the format expected by reconciler
        Structure: [{"0_0": {gsw, text, ...}, "0_1": {...}}, {"1_0": {...}}]
        
    Raises:
        LoaderError: If loading fails
    """
    try:
        log_dir = Path(log_dir)
        combined_file = log_dir / "gsw_output" / "gsw_results_combined.json"
        
        if not combined_file.exists():
            raise FileNotFoundError(f"GSW combined results not found: {combined_file}")
            
        with open(combined_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Reconstruct the expected format: List[Dict[str, Dict]]
        all_documents_data = []
        
        # Sort documents by index to maintain order
        doc_keys = sorted(data["documents"].keys(), key=lambda x: int(x.split("_")[1]))
        
        for doc_key in doc_keys:
            doc_chunks = data["documents"][doc_key]
            
            # Reconstruct GSW structures from JSON
            reconstructed_chunks = {}
            for chunk_id, chunk_data in doc_chunks.items():
                chunk_copy = chunk_data.copy()
                
                # Convert GSW JSON back to GSWStructure object
                if chunk_copy["gsw"] is not None:
                    chunk_copy["gsw"] = GSWStructure.model_validate(chunk_copy["gsw"])
                    
                reconstructed_chunks[chunk_id] = chunk_copy
                
            all_documents_data.append(reconstructed_chunks)
            
        print(f"✅ Loaded operator outputs: {len(all_documents_data)} documents, "
              f"{sum(len(doc) for doc in all_documents_data)} chunks")
        
        return all_documents_data
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise LoaderError(f"Failed to load operator outputs from {log_dir}: {str(e)}") from e


def load_reconciled_gsw(log_dir: Union[str, Path]) -> GSWStructure:
    """
    Load reconciled GSW from global_reconciled.json.
    
    Args:
        log_dir: Path to the evaluation log directory
        
    Returns:
        Reconciled GSWStructure object
        
    Raises:
        LoaderError: If loading fails
    """
    try:
        log_dir = Path(log_dir)
        reconciled_file = log_dir / "reconciled_output" / "reconciled" / "global_reconciled.json"
        
        if not reconciled_file.exists():
            raise FileNotFoundError(f"Reconciled GSW not found: {reconciled_file}")
            
        with open(reconciled_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert JSON back to GSWStructure
        reconciled_gsw = GSWStructure.model_validate(data)
        
        print(f"✅ Loaded reconciled GSW: {len(reconciled_gsw.entity_nodes)} entities, "
              f"{len(reconciled_gsw.verb_phrase_nodes)} verb phrases")
        
        return reconciled_gsw
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise LoaderError(f"Failed to load reconciled GSW from {log_dir}: {str(e)}") from e


def load_entity_summaries(log_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Load entity summaries from all_entity_summaries.json.
    
    Args:
        log_dir: Path to the evaluation log directory
        
    Returns:
        Dictionary mapping entity IDs to summary text
        
    Raises:
        LoaderError: If loading fails
    """
    try:
        log_dir = Path(log_dir)
        summaries_file = log_dir / "results" / "all_entity_summaries.json"
        
        if not summaries_file.exists():
            raise FileNotFoundError(f"Entity summaries not found: {summaries_file}")
            
        with open(summaries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Keep the full structure as expected by EntitySummaryAggregator
        summaries = {}
        for entity_id, entity_data in data.items():
            if isinstance(entity_data, dict) and "summary" in entity_data:
                # Keep the full structure with entity_name, summary, entity_id
                summaries[entity_id] = entity_data
            else:
                # Handle case where it's just the summary text (legacy format)
                summaries[entity_id] = {"summary": str(entity_data)}
                
        print(f"✅ Loaded entity summaries: {len(summaries)} entities")
        
        return summaries
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise LoaderError(f"Failed to load entity summaries from {log_dir}: {str(e)}") from e


def load_from_logs(log_dir: Union[str, Path]) -> Dict[str, any]:
    """
    Load all available data from an evaluation log directory.
    
    Args:
        log_dir: Path to the evaluation log directory
        
    Returns:
        Dictionary containing all loaded data:
        {
            "operator_outputs": List[Dict[str, Dict]] or None,
            "reconciled_gsw": GSWStructure or None,
            "entity_summaries": Dict[str, str] or None,
            "available_stages": List[str]
        }
        
    Raises:
        LoaderError: If log directory doesn't exist
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        raise LoaderError(f"Log directory not found: {log_dir}")
        
    result = {
        "operator_outputs": None,
        "reconciled_gsw": None,
        "entity_summaries": None,
        "available_stages": []
    }
    
    # Try to load each stage
    try:
        result["operator_outputs"] = load_operator_outputs(log_dir)
        result["available_stages"].append("operator")
    except (FileNotFoundError, LoaderError):
        pass
        
    try:
        result["reconciled_gsw"] = load_reconciled_gsw(log_dir)
        result["available_stages"].append("reconciler")
    except (FileNotFoundError, LoaderError):
        pass
        
    try:
        result["entity_summaries"] = load_entity_summaries(log_dir)
        result["available_stages"].append("aggregator")
    except (FileNotFoundError, LoaderError):
        pass
        
    if not result["available_stages"]:
        raise LoaderError(f"No valid evaluation data found in {log_dir}")
        
    print(f"📁 Loaded from {log_dir}")
    print(f"   Available stages: {', '.join(result['available_stages'])}")
    
    return result


def get_available_logs(logs_base_dir: Union[str, Path] = None) -> List[Path]:
    """
    Get list of available evaluation log directories.
    
    Args:
        logs_base_dir: Base directory containing logs (defaults to gsw-memory/logs)
        
    Returns:
        List of Path objects for available log directories
    """
    if logs_base_dir is None:
        # Default to logs directory relative to this file
        current_file = Path(__file__)
        gsw_memory_root = current_file.parents[2]  # Go up to gsw-memory/
        logs_base_dir = gsw_memory_root / "logs"
    else:
        logs_base_dir = Path(logs_base_dir)
        
    if not logs_base_dir.exists():
        return []
        
    # Find directories that match evaluation log pattern
    log_dirs = []
    for item in logs_base_dir.iterdir():
        if item.is_dir() and ("eval" in item.name or "test" in item.name):
            log_dirs.append(item)
            
    return sorted(log_dirs, reverse=True)  # Most recent first


def print_log_summary(log_dir: Union[str, Path]) -> None:
    """
    Print a summary of what's available in a log directory.
    
    Args:
        log_dir: Path to the evaluation log directory
    """
    log_dir = Path(log_dir)
    
    print(f"\n📊 Log Summary: {log_dir.name}")
    print("=" * 60)
    
    # Check each stage
    stages = {
        "Operator Output": log_dir / "gsw_output" / "gsw_results_combined.json",
        "Reconciled GSW": log_dir / "reconciled_output" / "reconciled" / "global_reconciled.json", 
        "Entity Summaries": log_dir / "results" / "all_entity_summaries.json",
        "Detailed Results": log_dir / "results" / "detailed_results.json",
        "Summary Metrics": log_dir / "results" / "summary_metrics.json"
    }
    
    for stage_name, file_path in stages.items():
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✅ {stage_name:<20} {file_size:>10,} bytes")
        else:
            print(f"❌ {stage_name:<20} {'Not found':>15}")
            
    # Read README if available
    readme_file = log_dir / "README.md"
    if readme_file.exists():
        print(f"\n📖 README.md available")
    
    print()


if __name__ == "__main__":
    # Demo usage
    logs = get_available_logs()
    
    if logs:
        print(f"Found {len(logs)} evaluation logs:")
        for log_dir in logs[:5]:  # Show first 5
            print_log_summary(log_dir)
    else:
        print("No evaluation logs found")