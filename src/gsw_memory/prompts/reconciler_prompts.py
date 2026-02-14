"""
Prompt templates for GSW Reconciler components.
"""


class QuestionResolutionPrompts:
    """Prompts for resolving unanswered questions using new chunk text."""
    
    SYSTEM_PROMPT = """You are an expert question answering system. Analyze the provided text and answer the specified questions based ONLY on the text. Provide answers in the specified JSON format."""
    
    USER_PROMPT_TEMPLATE = """Given the following text chunk, the entities introduced within it, and a list of currently unanswered questions from a knowledge graph:

Text Chunk:
--- START TEXT ---
{new_chunk_text}
--- END TEXT ---

Entities Introduced in This Chunk:
--- START ENTITIES ---
{new_entity_manifest}
--- END ENTITIES ---

Currently Unanswered Questions (Answer is 'None'):
--- START QUESTIONS ---
{candidate_list_str}
--- END QUESTIONS ---

Your task is to determine if the Text Chunk provides a specific answer for any of these unanswered questions. Base your answers ONLY on the provided Text Chunk.

Respond ONLY with a JSON list containing objects for the questions you can now answer. Each object should have:
- "question_id": The ID of the question being answered.
- "answer_text": The specific text snippet from the Text Chunk that answers the question.
- "answer_entity_id": (Optional) If the answer corresponds exactly to one of the Entities Introduced in This Chunk, provide its ID. Otherwise, omit this field or set it to null.

Example Response Format:
[
  {{"question_id": "chunk_X::q_Y", "answer_text": "Some Text Snippet", "answer_entity_id": "chunk_X::ent_Z"}},
  {{"question_id": "chunk_A::q_B", "answer_text": "Another text answer"}}
]

If no questions can be answered from the text, respond with an empty JSON list: []"""


class EntityVerificationPrompts:
    """Prompts for verifying entity similarity."""
    
    SYSTEM_PROMPT = """You are a semantic verification system. Respond only with the required JSON format."""
    
    USER_PROMPT_TEMPLATE = """Determine if each of the following entity pairs refers to the same real-world entity or strongly related entities.
For each pair, consider:
1. Are they referring to the same real-world entity?
2. Are they different entities but strongly related (e.g., person and their role, same person in different contexts)?
3. Are they completely unrelated or just coincidentally similar?

Entity pairs:
{pair_descriptions}

Respond with JSON in the following format:
{{
    "pair_0": true/false,
    "pair_1": true/false,
    ...
}}
Where "true" means the entities are the same or strongly related, and "false" means they are unrelated."""