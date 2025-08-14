#!/usr/bin/env python3
"""
Generate comprehensive summaries for hypernodes using GSW structure information.
Creates retrieval-ready summaries that consolidate entity information across documents.
"""

import json
import glob
import os

# Disable cache for testing
os.environ["CURATOR_DISABLE_CACHE"] = "true"
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Bespoke curator imports
try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    print("Warning: Bespoke curator not available. Install with: pip install bespokelabs-curator")
    CURATOR_AVAILABLE = False

# Embedding imports
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Install with: pip install vllm>=0.8.5")
    VLLM_AVAILABLE = False

import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from src.gsw_memory.memory.models import GSWStructure


class HypernodeSummarizer(curator.LLM):
    """Curator class for generating hypernode summaries in parallel."""
    
    return_completions_object = True
    
    def __init__(self, **kwargs):
        """Initialize the summarizer."""
        super().__init__(**kwargs)
    
    def prompt(self, input_data):
        """Create a prompt for summarizing a hypernode."""
        consolidated_info = input_data
        
        prompt = f"""You are tasked with creating a comprehensive summary for an entity that appears across multiple documents. You will be given information about the entity including its name variations, roles, states, and question-answer pairs.

Entity Name: {consolidated_info['hypernode_name']}
Name Variations: {', '.join(consolidated_info['entity_variations'])}

ROLES AND STATES:
"""
        
        for i, role in enumerate(consolidated_info['roles_and_states'][:15], 1):  # Limit to avoid too long prompts
            prompt += f"{i}. {role}\n"
        
        if consolidated_info['questions_and_answers']:
            prompt += f"\nRELEVANT QUESTIONS AND ANSWERS:\n"
            for i, qa in enumerate(consolidated_info['questions_and_answers'][:10], 1):  # Limit to avoid too long prompts
                answers_str = ', '.join([str(a) for a in qa['answers'] if str(a) != 'None'])
                prompt += f"{i}. Q: {qa['question']}\n"
                prompt += f"   A: {answers_str}\n"
                if qa.get('verb_phrase'):
                    prompt += f"   Context: {qa['verb_phrase']}\n"
                prompt += "\n"
        
        prompt += f"""
TASK:
Create a comprehensive, factual summary of this entity that consolidates all the information provided. The summary should:

1. Start with the most common/canonical name
2. Include key identifying information (titles, roles, relationships)
3. Mention important facts, dates, or achievements
4. Consolidate information from different documents into a coherent narrative
5. Be clear and informative
6. Focus on the most important and distinguishing characteristics
7. If there are entities that do not match the narrative and style of other entities, feel free to discard them from the summary generation process, entity combinations can be noisy. 

Do not include speculative information or details not supported by the provided context.

SUMMARY:"""
        
        return [
            {"role": "system", "content": "You are an expert at creating concise, factual summaries from structured data."},
            {"role": "user", "content": prompt}
        ]
    
    def parse(self, input_data, response):
        """Parse the response to extract the summary."""
        # summary_text = response.choices[0].message.content.strip()
        return {
            'hypernode_id': input_data['hypernode_id'],
            'hypernode_name': input_data['hypernode_name'], 
            'summary': response["choices"][0]["message"]["content"],
            'entity_variations': input_data['entity_variations'],
            'source_documents': input_data['source_documents'],
            'entities_included': input_data['entities_included'],
            'confidence_score': 0.9  # High confidence for GPT-4o generated summaries
        }


@dataclass
class HypernodeSummary:
    """Container for hypernode summary data."""
    hypernode_id: str
    name: str
    summary: str
    entity_variations: List[str]
    source_documents: List[str]
    entities_included: List[str]
    confidence_score: float = 0.0
    embedding: np.ndarray = None


def load_hypernode_results(results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load existing hypernode results from similarity analysis."""
    print(f"Loading hypernode results from {results_dir}")
    
    # Load similarity results
    similarity_file = Path(results_dir) / "similarity_results.json"
    if not similarity_file.exists():
        raise FileNotFoundError(f"Similarity results not found: {similarity_file}")
    
    with open(similarity_file, 'r') as f:
        similarity_results = json.load(f)
    
    # Load analysis results  
    analysis_file = Path(results_dir) / "similarity_analysis.json"
    if not analysis_file.exists():
        raise FileNotFoundError(f"Analysis results not found: {analysis_file}")
    
    with open(analysis_file, 'r') as f:
        analysis_results = json.load(f)
    
    print(f"Loaded {len(similarity_results.get('hypernodes', {}))} hypernodes")
    return similarity_results, analysis_results


def load_original_gsw_structures(entity_doc_ids: List[str]) -> Dict[str, GSWStructure]:
    """Load original GSW structures for the documents that contain hypernode entities."""
    print(f"Loading GSW structures for {len(set(entity_doc_ids))} unique documents")
    
    base_dir = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/logs/full_2wiki_corpus_20250710_202211/gsw_output_global_ids/networks"
    
    gsw_structures = {}
    unique_doc_ids = set(entity_doc_ids)
    
    for doc_id in unique_doc_ids:
        doc_dir = os.path.join(base_dir, doc_id)
        gsw_files = glob.glob(os.path.join(doc_dir, "gsw_*.json"))
        
        for file_path in gsw_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    gsw_structures[doc_id] = GSWStructure(**data)
                break  # Only need one GSW file per document
            except Exception as e:
                print(f"Warning: Error loading GSW for {doc_id}: {e}")
                continue
    
    print(f"Successfully loaded {len(gsw_structures)} GSW structures")
    return gsw_structures


def extract_entity_context(entity_data: Dict[str, Any], gsw_structure: GSWStructure) -> Dict[str, Any]:
    """Extract comprehensive context for an entity from its GSW structure."""
    entity_id = entity_data['entity_id']
    entity_name = entity_data['name']
    
    # Find the entity in the GSW structure
    target_entity = None
    for entity in gsw_structure.entity_nodes:
        if entity.id == entity_id or entity.name == entity_name:
            target_entity = entity
            break
    
    if not target_entity:
        return {
            'roles_and_states': entity_data.get('roles', []),
            'questions_and_answers': [],
            'related_entities': []
        }
    
    # Extract roles and states
    roles_and_states = []
    for role in target_entity.roles:
        if role.states:
            states_text = ', '.join(role.states)
            roles_and_states.append(f"{role.role}: {states_text}")
        else:
            roles_and_states.append(role.role)
    
    # Find questions and answers involving this entity
    questions_and_answers = []
    related_entities = set()
    
    for verb_phrase in gsw_structure.verb_phrase_nodes:
        for question in verb_phrase.questions:
            # Check if this entity is mentioned in answers
            if entity_id in question.answers or any(entity_id in str(answer) for answer in question.answers):
                qa_pair = {
                    'question': question.text,
                    'answers': "Entity: " + entity_name + "\t" + "description: " + "".join(roles_and_states),
                    'verb_phrase': verb_phrase.phrase
                }
                questions_and_answers.append(qa_pair)
                
                # Collect related entities from the same questions
                for answer in question.answers:
                    if answer != entity_id and answer != "None":
                        if str(answer).startswith("TEXT:"):
                            related_entities.add(str(answer).split("TEXT:")[1].strip())
                        else:
                            related_entities.add(answer)
    
    return {
        'roles_and_states': roles_and_states,
        'questions_and_answers': questions_and_answers,
        'related_entities': list(related_entities)
    }


def consolidate_hypernode_information(hypernode_data: Dict[str, Any], 
                                    gsw_structures: Dict[str, GSWStructure]) -> Dict[str, Any]:
    """Consolidate all information for a hypernode across its entities."""
    hypernode_id = hypernode_data.get('hypernode_id', 'unknown')
    hypernode_name = hypernode_data.get('hypernode_name', 'Unknown Entity')
    entities = hypernode_data.get('entities', [])

    # Collect all information
    all_entity_names = set()
    all_roles_and_states = []
    all_questions_and_answers = []
    all_related_entities = set()
    source_documents = set()
    entities_included = []
    
    for entity_data in entities:
        doc_id = entity_data['doc_id']
        source_documents.add(doc_id)
        entities_included.append(entity_data['entity_id'])
        all_entity_names.add(entity_data['name'])
        
        # Get GSW structure for this document
        if doc_id in gsw_structures:
            context = extract_entity_context(entity_data, gsw_structures[doc_id])
            all_roles_and_states.extend(context['roles_and_states'])
            all_questions_and_answers.extend(context['questions_and_answers'])
            all_related_entities.update(context['related_entities'])
        else:
            # Fallback to basic role information
            all_roles_and_states.extend(entity_data.get('roles', []))
    
    return {
        'hypernode_id': hypernode_id,
        'hypernode_name': hypernode_name,
        'entity_variations': list(all_entity_names),
        'roles_and_states': list(set(all_roles_and_states)),  # Remove duplicates
        'questions_and_answers': all_questions_and_answers,
        'related_entities': list(all_related_entities),
        'source_documents': list(source_documents),
        'entities_included': entities_included,
        'entity_count': len(entities)
    }


def create_summarization_prompt(consolidated_info: Dict[str, Any]) -> str:
    """Create a prompt for LLM to generate hypernode summary."""
    
    prompt = f"""You are tasked with creating a comprehensive summary for an entity that appears across multiple documents. You will be given information about the entity including its name variations, roles, states, and question-answer pairs.

Entity Name: {consolidated_info['hypernode_name']}
Name Variations: {', '.join(consolidated_info['entity_variations'])}

ROLES AND STATES:
"""
    
    for i, role in enumerate(consolidated_info['roles_and_states'][:15], 1):  # Limit to avoid too long prompts
        prompt += f"{i}. {role}\n"
    
    if consolidated_info['questions_and_answers']:
        prompt += f"\nRELEVANT QUESTIONS AND ANSWERS:\n"
        for i, qa in enumerate(consolidated_info['questions_and_answers'][:10], 1):  # Limit to avoid too long prompts
            answers_str = ', '.join([str(a) for a in qa['answers'] if str(a) != 'None'])
            prompt += f"{i}. Q: {qa['question']}\n"
            prompt += f"   A: {answers_str}\n"
            if qa.get('verb_phrase'):
                prompt += f"   Context: {qa['verb_phrase']}\n"
            prompt += "\n"
    
    prompt += f"""
TASK:
Create a comprehensive, factual summary of this entity that consolidates all the information provided. The summary should:

1. Start with the most common/canonical name
2. Include key identifying information (titles, roles, relationships)
3. Mention important facts, dates, or achievements
4. Consolidate information from different documents into a coherent narrative
5. Be 2-4 sentences long, clear and informative
6. Focus on the most important and distinguishing characteristics

Do not include speculative information or details not supported by the provided context.

SUMMARY:"""
    
    return prompt


def generate_summary_with_llm(consolidated_info: Dict[str, Any]) -> Tuple[str, float]:
    """Generate summary using OpenAI GPT-4."""
    if not OPENAI_AVAILABLE:
        return f"Summary for {consolidated_info['hypernode_name']}: No LLM available for summary generation.", 0.5
    
    try:
        client = OpenAI()
        prompt = create_summarization_prompt(consolidated_info)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at creating concise, factual summaries from structured data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        summary = response.choices[0].message.content.strip()
        confidence_score = 0.9  # High confidence for GPT-4o generated summaries
        
        return summary, confidence_score
        
    except Exception as e:
        print(f"Error generating summary for {consolidated_info['hypernode_name']}: {e}")
        # Fallback summary
        entity_names = ', '.join(consolidated_info['entity_variations'])
        roles = ', '.join(consolidated_info['roles_and_states'][:3])
        fallback_summary = f"{consolidated_info['hypernode_name']} ({entity_names}): {roles}"
        return fallback_summary, 0.3


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


def generate_summary_embeddings(summaries: List[HypernodeSummary]) -> List[HypernodeSummary]:
    """Generate embeddings for hypernode summaries using Qwen model."""
    if not VLLM_AVAILABLE:
        print("VLLM not available, skipping embedding generation")
        return summaries
        
    print("Generating embeddings for hypernode summaries...")
    
    # Custom task for summary retrieval
    task = 'Given a comprehensive entity summary, create an embedding optimized for semantic search and question answering retrieval'
    
    # Prepare input texts with instructions
    input_texts = []
    for summary in summaries:
        # Combine summary with entity variations for better matching
        full_text = f"{summary.name}: {summary.summary}"
        if len(summary.entity_variations) > 1:
            full_text += f" (Also known as: {', '.join(summary.entity_variations)})"
        
        instructed_query = get_detailed_instruct(task, full_text)
        input_texts.append(instructed_query)
    
    try:
        # Initialize model
        model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
        
        # Generate embeddings
        outputs = model.embed(input_texts)
        embeddings = [output.outputs.embedding for output in outputs]
        
        # Add embeddings to summaries
        for summary, embedding in zip(summaries, embeddings):
            summary.embedding = np.array(embedding)
        
        print(f"Generated embeddings for {len(summaries)} hypernode summaries")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
    
    return summaries


def main():
    """Main function to generate hypernode summaries."""
    print("HYPERNODE SUMMARY GENERATION")
    print("=" * 60)
    
    # Configuration
    results_dir_pattern = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_results_*"
    
    # Find the most recent hypernode results directory
    results_dirs = glob.glob(results_dir_pattern)
    if not results_dirs:
        print(f"No hypernode results found matching pattern: {results_dir_pattern}")
        return
    
    latest_results_dir = max(results_dirs, key=os.path.getctime)
    print(f"Using hypernode results from: {latest_results_dir}")
    
    output_dir = f"/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_summaries_{int(time.time())}"
    
    # Step 1: Load hypernode results
    similarity_results, analysis_results = load_hypernode_results(latest_results_dir)
    
    hypernodes = similarity_results.get('hypernodes', {})
    if not hypernodes:
        print("No hypernodes found in results")
        return
    
    print(f"Processing {len(hypernodes)} hypernodes")
    
    # Step 2: Get all document IDs that contain hypernode entities
    all_doc_ids = set()
    for hypernode_data in hypernodes.values():
        for entity in hypernode_data.get('entities', []):
            all_doc_ids.add(entity['doc_id'])
    
    # Step 3: Load original GSW structures
    gsw_structures = load_original_gsw_structures(list(all_doc_ids))
    
    # Step 4: Consolidate information for all hypernodes
    print("\nConsolidating information for all hypernodes...")
    consolidated_inputs = []
    
    for hypernode_id, hypernode_data in hypernodes.items():
        print(f"  Processing hypernode {hypernode_id}: {hypernode_data.get('hypernode_name', 'Unknown')}")
        
        # Add hypernode_id to the data for processing
        hypernode_data['hypernode_id'] = hypernode_id
        
        # Consolidate information
        consolidated_info = consolidate_hypernode_information(hypernode_data, gsw_structures)
        consolidated_inputs.append(consolidated_info)
    
    # Step 5: Generate all summaries in parallel using curator
    print(f"\nGenerating summaries for {len(consolidated_inputs)} hypernodes in parallel...")
    
    if not CURATOR_AVAILABLE:
        print("Warning: Curator not available, falling back to sequential processing")
        hypernode_summaries = []
        for consolidated_info in consolidated_inputs:
            summary_text, confidence = generate_summary_with_llm(consolidated_info)
            hypernode_summary = HypernodeSummary(
                hypernode_id=consolidated_info['hypernode_id'],
                name=consolidated_info['hypernode_name'],
                summary=summary_text,
                entity_variations=consolidated_info['entity_variations'],
                source_documents=consolidated_info['source_documents'],
                entities_included=consolidated_info['entities_included'],
                confidence_score=confidence
            )
            hypernode_summaries.append(hypernode_summary)
    else:
        # Use curator for parallel processing
        summarizer = HypernodeSummarizer(
            model_name="gpt-4o-mini",
            generation_params={"temperature": 0.1, "max_tokens": 400}
        )
        
        # Generate summaries in parallel
        summary_responses = summarizer(consolidated_inputs)
        
        # Convert responses to HypernodeSummary objects
        hypernode_summaries = []
        for response in summary_responses.dataset:
            hypernode_summary = HypernodeSummary(
                hypernode_id=response['hypernode_id'],
                name=response['hypernode_name'],
                summary=response['summary'],
                entity_variations=response['entity_variations'],
                source_documents=response['source_documents'],
                entities_included=response['entities_included'],
                confidence_score=response['confidence_score']
            )
            hypernode_summaries.append(hypernode_summary)
        
        print(f"Generated {len(hypernode_summaries)} summaries in parallel")
    
    # Step 6: Generate embeddings
    hypernode_summaries = generate_summary_embeddings(hypernode_summaries)
    
    # Step 7: Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save summaries in JSON format
    summaries_data = []
    for summary in hypernode_summaries:
        summary_dict = {
            'hypernode_id': summary.hypernode_id,
            'name': summary.name,
            'summary': summary.summary,
            'entity_variations': summary.entity_variations,
            'source_documents': summary.source_documents,
            'entities_included': summary.entities_included,
            'confidence_score': summary.confidence_score,
            'has_embedding': summary.embedding is not None
        }
        summaries_data.append(summary_dict)
    
    with open(Path(output_dir) / "hypernode_summaries.json", 'w') as f:
        json.dump(summaries_data, f, indent=2)
    
    # Save embeddings separately (if available)
    if hypernode_summaries and hypernode_summaries[0].embedding is not None:
        embeddings_data = {
            'embeddings': [summary.embedding.tolist() for summary in hypernode_summaries],
            'hypernode_ids': [summary.hypernode_id for summary in hypernode_summaries]
        }
        with open(Path(output_dir) / "hypernode_embeddings.json", 'w') as f:
            json.dump(embeddings_data, f)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {len(hypernode_summaries)} hypernode summaries")
    print(f"Average confidence score: {np.mean([s.confidence_score for s in hypernode_summaries]):.2f}")
    print(f"Results saved to: {output_dir}")
    
    # Show sample summaries
    print(f"\nSAMPLE SUMMARIES:")
    print("-" * 40)
    for i, summary in enumerate(hypernode_summaries[:3], 1):
        print(f"\n{i}. {summary.name} (Hypernode {summary.hypernode_id})")
        print(f"   Variations: {', '.join(summary.entity_variations)}")
        print(f"   Documents: {', '.join(summary.source_documents)}")
        print(f"   Summary: {summary.summary}")


if __name__ == "__main__":
    main()