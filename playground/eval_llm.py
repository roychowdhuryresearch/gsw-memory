#!/usr/bin/env python3
"""
Example usage of the LLM-based QA evaluation script

This script demonstrates how to run the evaluation on your QA result files.
"""

import subprocess
import os
import sys

def run_evaluation():
    """Run the evaluation script with the specified files"""
    
    # Define the file paths
    files_to_evaluate = [
        "/home/yigit/codebase/gsw-memory/logs/baseline_musique_20250808_121101/baseline_embedding_results.json",
        "/home/yigit/codebase/gsw-memory/logs/agentic_2wiki_20250808_172940/agentic_multi_file_results.json",
        "/home/yigit/codebase/gsw-memory/logs/musique_50_q_result_dict_hipporagv2.json"
    ]
    
    # Check if files exist
    missing_files = [f for f in files_to_evaluate if not os.path.exists(f)]
    if missing_files:
        print("Warning: The following files do not exist:")
        for f in missing_files:
            print(f"  - {f}")
        print()
    
    # Filter to existing files
    existing_files = [f for f in files_to_evaluate if os.path.exists(f)]
    
    if not existing_files:
        print("No evaluation files found. Please check the file paths.")
        return
    
    print(f"Running evaluation on {len(existing_files)} files:")
    for f in existing_files:
        print(f"  - {f}")
    print()
    
    # Prepare command
    cmd = [
        sys.executable,
        "eval_llm_comparison.py",
        "--files"
    ] + existing_files + [
        "--output-dir", "evaluation_results",
        "--model", "gpt-4o"
    ]
    
    print("Command to be executed:")
    print(" ".join(cmd))
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Or provide it as argument: --api-key your-api-key")
        print()
        return
    
    try:
        # Run the evaluation
        print("Starting evaluation...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Evaluation completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        print("\nError output:")
        print(e.stderr)
        print("\nStandard output:")
        print(e.stdout)
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("LLM-BASED QA EVALUATION EXAMPLE")
    print("=" * 60)
    print()
    
    # Check if evaluation script exists
    if not os.path.exists("eval_llm_comparison.py"):
        print("ERROR: eval_llm_comparison.py not found in current directory!")
        print("Please make sure you're running this from the correct directory.")
        sys.exit(1)
    
    run_evaluation()