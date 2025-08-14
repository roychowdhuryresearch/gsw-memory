#!/usr/bin/env python3
"""
LLM-based Evaluation Script for QA Results

This script evaluates multiple QA result files by comparing predicted answers 
with gold/reference answers using an LLM for semantic similarity assessment.

Supports:
- Baseline embedding results
- Agentic multi-file results  
- HippoRAG results

Usage:
    python eval_llm_comparison.py --files file1.json file2.json file3.json
"""

import json
import argparse
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import openai
import time
from tqdm import tqdm
import pandas as pd

@dataclass
class QAResult:
    """Standardized QA result structure"""
    question_id: str
    question: str
    predicted_answer: str
    gold_answers: List[str]
    file_source: str
    additional_info: Dict[str, Any] = None

class LLMEvaluator:
    """LLM-based evaluator for comparing predicted vs gold answers"""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        if api_key:
            self.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            print("Warning: No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")
            
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def evaluate_answer_pair(self, predicted: str, gold_answers: List[str], question: str) -> Dict[str, Any]:
        """
        Evaluate a predicted answer against gold answers using LLM
        
        Returns:
            Dict containing semantic_match_score (0-1), reasoning, and best_gold_match
        """
        
        gold_answers_text = "\n".join([f"- {ans}" for ans in gold_answers])
        
        prompt = f"""You are an expert evaluator for question-answering systems. Your task is to assess whether a predicted answer semantically matches any of the reference (gold) answers, even if the wording is different.

Question: {question}

Reference Answers:
{gold_answers_text}

Predicted Answer: {predicted}

Please evaluate on a scale of 0.0 to 1.0 how well the predicted answer matches the semantic meaning of the reference answers:
- 1.0: Perfect semantic match (same meaning, may have different wording)  
- 0.8-0.9: Very close match (minor differences in specificity or phrasing)
- 0.6-0.7: Good match (correct core information, some missing details)
- 0.4-0.5: Partial match (some correct information, significant gaps)
- 0.2-0.3: Poor match (minimal overlap in meaning)
- 0.0-0.1: No match (completely different or incorrect information)

Respond with a JSON object in this exact format:
{{
    "semantic_match_score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your evaluation>",
    "best_gold_match": "<which reference answer matches best, or 'none' if no good match>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith("```"):
                result_text = result_text[3:-3].strip()
            
            result = json.loads(result_text)
            
            # Validate result structure
            if not all(key in result for key in ["semantic_match_score", "reasoning", "best_gold_match"]):
                raise ValueError("Missing required keys in LLM response")
            
            # Clamp score to valid range
            result["semantic_match_score"] = max(0.0, min(1.0, float(result["semantic_match_score"])))
            
            return result
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            # Fallback to simple string matching
            predicted_lower = predicted.lower().strip()
            max_score = 0.0
            best_match = "none"
            
            for gold in gold_answers:
                gold_lower = gold.lower().strip()
                if predicted_lower == gold_lower:
                    max_score = 1.0
                    best_match = gold
                    break
                elif predicted_lower in gold_lower or gold_lower in predicted_lower:
                    max_score = max(max_score, 0.7)
                    best_match = gold
            
            return {
                "semantic_match_score": max_score,
                "reasoning": f"Fallback evaluation due to error: {e}",
                "best_gold_match": best_match
            }

class ResultLoader:
    """Utility class to load and standardize different QA result file formats"""
    
    @staticmethod
    def load_baseline_results(file_path: str) -> List[QAResult]:
        """Load baseline embedding results format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            result = QAResult(
                question_id=str(item.get('question_id', '')),
                question=item.get('question', ''),
                predicted_answer=item.get('predicted_answer', ''),
                gold_answers=item.get('gold_answers', []),
                file_source=Path(file_path).name,
                additional_info={
                    'question_type': item.get('question_type'),
                    'reasoning': item.get('reasoning'),
                    'tool_calls': len(item.get('tool_calls', []))
                }
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def load_agentic_results(file_path: str) -> List[QAResult]:
        """Load agentic multi-file results format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            result = QAResult(
                question_id=str(item.get('question_id', '')),
                question=item.get('question', ''),
                predicted_answer=item.get('predicted_answer', ''),
                gold_answers=item.get('gold_answers', []),
                file_source=Path(file_path).name,
                additional_info={
                    'question_type': item.get('question_type'),
                    'supporting_doc_ids': item.get('supporting_doc_ids', [])
                }
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def load_hipporag_results(file_path: str) -> List[QAResult]:
        """Load HippoRAG results format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        # HippoRAG format is a dict of dicts where each key is a question ID
        for question_id, question_data in data.items():
            # Skip non-question entries (metadata keys)
            if not isinstance(question_data, dict):
                continue
                
            question = question_data.get('query', '')
            gold_answers = question_data.get('gold_answers', [])
            predicted_answer = question_data.get('hipporag_2_predicted_answer', '')
            question_type = question_data.get('question_type', 'unknown')
            
            result = QAResult(
                question_id=question_id,
                question=question,
                predicted_answer=predicted_answer,
                gold_answers=gold_answers,
                file_source=Path(file_path).name,
                additional_info={
                    'question_type': question_type,
                    'exact_match_score': question_data.get('exact_match_score', {}),
                    'f1_score': question_data.get('f1_score', {}),
                    'num_documents': len([k for k in question_data.keys() if k.isdigit()])
                }
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def detect_file_format(file_path: str) -> str:
        """Detect the format of a result file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and data:
                first_item = data[0]
                if 'question_id' in first_item and 'predicted_answer' in first_item:
                    if 'tool_calls' in first_item:
                        return 'baseline'
                    else:
                        return 'agentic'
            elif isinstance(data, dict):
                if isinstance(data, dict) and "0" in data and isinstance(data["0"], dict) and ('hipporag_2_predicted_answer' in data["0"] or ('query' in data["0"] and 'gold_answers' in data["0"])):
                    return 'hipporag'
                    
            return 'unknown'
        except:
            return 'unknown'
    
    @staticmethod
    def load_results(file_path: str) -> List[QAResult]:
        """Auto-detect format and load results"""
        format_type = ResultLoader.detect_file_format(file_path)
        
        if format_type == 'baseline':
            return ResultLoader.load_baseline_results(file_path)
        elif format_type == 'agentic':
            return ResultLoader.load_agentic_results(file_path)
        elif format_type == 'hipporag':
            return ResultLoader.load_hipporag_results(file_path)
        else:
            raise ValueError(f"Unknown file format: {file_path}")

def evaluate_results(results: List[QAResult], evaluator: LLMEvaluator) -> pd.DataFrame:
    """Evaluate a list of QA results and return detailed metrics"""
    
    evaluation_data = []
    
    for result in tqdm(results, desc="Evaluating answers"):
        # Skip results without gold answers (like HippoRAG format)
        if not result.gold_answers:
            eval_result = {
                "semantic_match_score": None,
                "reasoning": "No gold answers available for evaluation",
                "best_gold_match": "none"
            }
        else:
            eval_result = evaluator.evaluate_answer_pair(
                result.predicted_answer, 
                result.gold_answers, 
                result.question
            )
        
        # Add some delay to avoid rate limiting
        time.sleep(0.1)
        
        row = {
            'question_id': result.question_id,
            'file_source': result.file_source,
            'question': result.question[:100] + "..." if len(result.question) > 100 else result.question,
            'predicted_answer': result.predicted_answer,
            'gold_answers': "; ".join(result.gold_answers),
            'semantic_match_score': eval_result["semantic_match_score"],
            'reasoning': eval_result["reasoning"],
            'best_gold_match': eval_result["best_gold_match"],
            'question_type': result.additional_info.get('question_type', 'unknown') if result.additional_info else 'unknown'
        }
        
        evaluation_data.append(row)
    
    return pd.DataFrame(evaluation_data)

def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate a summary report from evaluation results"""
    
    report = []
    report.append("=" * 80)
    report.append("LLM-BASED QA EVALUATION SUMMARY REPORT")
    report.append("=" * 80)
    
    # Overall statistics
    total_questions = len(df)
    files_evaluated = df['file_source'].unique()
    
    report.append(f"\nOVERALL STATISTICS:")
    report.append(f"  Total questions evaluated: {total_questions}")
    report.append(f"  Files evaluated: {len(files_evaluated)}")
    for file in files_evaluated:
        count = len(df[df['file_source'] == file])
        report.append(f"    - {file}: {count} questions")
    
    # Performance by file
    report.append(f"\nPERFORMANCE BY FILE:")
    
    for file in files_evaluated:
        file_df = df[df['file_source'] == file]
        valid_scores = file_df['semantic_match_score'].dropna()
        
        if len(valid_scores) > 0:
            avg_score = valid_scores.mean()
            high_quality = (valid_scores >= 0.8).sum()
            medium_quality = ((valid_scores >= 0.5) & (valid_scores < 0.8)).sum()
            low_quality = (valid_scores < 0.5).sum()
            
            report.append(f"\n  {file}:")
            report.append(f"    Average semantic match score: {avg_score:.3f}")
            report.append(f"    High quality answers (≥0.8): {high_quality} ({high_quality/len(valid_scores)*100:.1f}%)")
            report.append(f"    Medium quality answers (0.5-0.8): {medium_quality} ({medium_quality/len(valid_scores)*100:.1f}%)")
            report.append(f"    Low quality answers (<0.5): {low_quality} ({low_quality/len(valid_scores)*100:.1f}%)")
        else:
            report.append(f"\n  {file}:")
            report.append(f"    No valid scores available for evaluation")
    
    # Question type analysis (if available)
    question_types = df['question_type'].value_counts()
    if len(question_types) > 1:
        report.append(f"\nPERFORMANCE BY QUESTION TYPE:")
        for q_type, count in question_types.items():
            if q_type != 'unknown':
                type_df = df[df['question_type'] == q_type]
                valid_scores = type_df['semantic_match_score'].dropna()
                if len(valid_scores) > 0:
                    avg_score = valid_scores.mean()
                    report.append(f"    {q_type}: {avg_score:.3f} (n={count})")
    
    # Examples of high and low scoring answers
    valid_df = df[df['semantic_match_score'].notna()]
    
    if len(valid_df) > 0:
        report.append(f"\nEXAMPLE HIGH-SCORING ANSWERS:")
        high_scoring = valid_df[valid_df['semantic_match_score'] >= 0.9].head(3)
        for idx, row in high_scoring.iterrows():
            report.append(f"  Score: {row['semantic_match_score']:.2f}")
            report.append(f"  Question: {row['question']}")
            report.append(f"  Predicted: {row['predicted_answer']}")
            report.append(f"  Gold: {row['gold_answers']}")
            report.append(f"  Reasoning: {row['reasoning']}")
            report.append("")
        
        report.append(f"EXAMPLE LOW-SCORING ANSWERS:")
        low_scoring = valid_df[valid_df['semantic_match_score'] <= 0.3].head(3)
        for idx, row in low_scoring.iterrows():
            report.append(f"  Score: {row['semantic_match_score']:.2f}")
            report.append(f"  Question: {row['question']}")
            report.append(f"  Predicted: {row['predicted_answer']}")
            report.append(f"  Gold: {row['gold_answers']}")
            report.append(f"  Reasoning: {row['reasoning']}")
            report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate QA results using LLM-based comparison')
    parser.add_argument('--files', nargs='+', required=True, help='QA result files to evaluate')
    parser.add_argument('--output-dir', default='./eval_output', help='Output directory for results')
    parser.add_argument('--model', default='gpt-4o', help='LLM model to use for evaluation')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = LLMEvaluator(model=args.model, api_key=args.api_key)
    
    # Load all results
    all_results = []
    for file_path in args.files:
        print(f"Loading {file_path}...")
        try:
            results = ResultLoader.load_results(file_path)
            all_results.extend(results)
            print(f"  Loaded {len(results)} results")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    if not all_results:
        print("No results loaded. Exiting.")
        return
    
    print(f"\nTotal results to evaluate: {len(all_results)}")
    
    # Evaluate results
    print("Starting LLM-based evaluation...")
    df = evaluate_results(all_results, evaluator)
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, 'detailed_evaluation_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    # Generate and save summary report
    summary_report = generate_summary_report(df)
    report_file = os.path.join(args.output_dir, 'evaluation_summary_report.txt')
    with open(report_file, 'w') as f:
        f.write(summary_report)
    print(f"Summary report saved to: {report_file}")
    
    # Print summary to console
    print("\n" + summary_report)

if __name__ == "__main__":
    main()