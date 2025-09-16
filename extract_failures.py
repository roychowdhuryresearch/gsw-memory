#!/usr/bin/env python3
"""
Extract failure cases from 2wiki evaluation results for further analysis.

This script identifies incorrect predictions and saves them to separate files
organized by question type for detailed investigation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip whitespace)."""
    return answer.lower().strip()

def is_correct_answer(predicted: str, gold_answers: List[str]) -> bool:
    """Check if predicted answer matches any gold answer."""
    predicted_norm = normalize_answer(predicted)
    gold_norm = [normalize_answer(g) for g in gold_answers]
    
    # Exact match
    if predicted_norm in gold_norm:
        return True
    
    # Check if predicted is substring of gold or vice versa
    for gold in gold_norm:
        if predicted_norm in gold or gold in predicted_norm:
            return True
    
    return False

def extract_failures(results_file: str, output_dir: str) -> Dict[str, int]:
    """Extract failure cases and save to organized files."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total': len(results),
        'correct': 0,
        'failures': 0,
        'by_type': {}
    }
    
    # Collect failures by question type
    failures_by_type = {}
    all_failures = []
    
    for result in results:
        question_type = result['question_type']
        predicted = result['predicted_answer']
        gold_answers = result['gold_answers']
        
        # Track statistics
        if question_type not in stats['by_type']:
            stats['by_type'][question_type] = {'total': 0, 'correct': 0, 'failures': 0}
        
        stats['by_type'][question_type]['total'] += 1
        
        # Check if answer is correct
        is_correct = is_correct_answer(predicted, gold_answers)
        
        if is_correct:
            stats['correct'] += 1
            stats['by_type'][question_type]['correct'] += 1
        else:
            stats['failures'] += 1
            stats['by_type'][question_type]['failures'] += 1
            
            # Add to failure collections
            failure_case = {
                'question_id': result['question_id'],
                'question': result['question'],
                'question_type': question_type,
                'predicted_answer': predicted,
                'gold_answers': gold_answers,
                'reasoning': result['reasoning'],
                'tool_calls': result['tool_calls'],
                'num_tool_calls': result['num_tool_calls'],
                'approach': result['approach']
            }
            
            all_failures.append(failure_case)
            
            if question_type not in failures_by_type:
                failures_by_type[question_type] = []
            failures_by_type[question_type].append(failure_case)
    
    # Save all failures
    with open(output_path / 'all_failures.json', 'w') as f:
        json.dump(all_failures, f, indent=2)
    
    # Save failures by question type
    for question_type, failures in failures_by_type.items():
        filename = f'failures_{question_type}.json'
        with open(output_path / filename, 'w') as f:
            json.dump(failures, f, indent=2)
    
    # Save analysis summary
    analysis = {
        'summary': stats,
        'accuracy_by_type': {
            qtype: data['correct'] / data['total'] 
            for qtype, data in stats['by_type'].items()
        },
        'failure_patterns': {
            'compositional_failures': len(failures_by_type.get('compositional', [])),
            'comparison_failures': len(failures_by_type.get('comparison', [])),
            'bridge_comparison_failures': len(failures_by_type.get('bridge_comparison', [])),
            'inference_failures': len(failures_by_type.get('inference', []))
        }
    }
    
    with open(output_path / 'failure_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"Results Analysis:")
    print(f"  Total questions: {stats['total']}")
    print(f"  Correct answers: {stats['correct']} ({stats['correct']/stats['total']:.1%})")
    print(f"  Failed answers: {stats['failures']} ({stats['failures']/stats['total']:.1%})")
    print()
    
    print("By question type:")
    for qtype, data in stats['by_type'].items():
        accuracy = data['correct'] / data['total']
        print(f"  {qtype}: {data['correct']}/{data['total']} correct ({accuracy:.1%}), {data['failures']} failures")
    
    print(f"\nFailure cases saved to: {output_path}")
    print("Files created:")
    print(f"  - all_failures.json ({len(all_failures)} cases)")
    for qtype, failures in failures_by_type.items():
        print(f"  - failures_{qtype}.json ({len(failures)} cases)")
    print(f"  - failure_analysis.json (summary statistics)")
    
    return stats

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract failure cases from 2wiki evaluation results')
    parser.add_argument('--results-file', '-r', 
                       default='logs/agentic_2wiki_20250715_163600/agentic_multi_file_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', '-o',
                       default='logs/agentic_2wiki_20250715_163600/failure_analysis',
                       help='Output directory for failure cases')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    try:
        stats = extract_failures(args.results_file, args.output_dir)
        return 0
    except Exception as e:
        print(f"Error processing results: {e}")
        return 1

if __name__ == '__main__':
    exit(main())