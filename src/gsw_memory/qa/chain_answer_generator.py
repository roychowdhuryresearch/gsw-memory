"""
Oracle-style answer generation from retrieved evidence.

Generates concise answers from QA-pair evidence using one-shot prompting
with chain-of-thought reasoning.
"""

import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze "
    "precise QA pairs extracted from the documents and corresponding questions "
    'meticulously. Your response start after "Thought: ", where you will '
    "methodically break down the reasoning process, illustrating how you arrive "
    'at conclusions. Conclude with "Answer: " to present only a concise, '
    "definitive response, devoid of additional elaborations."
)

ANSWER_SYSTEM_PROMPT_NO_ANSWER = (
    ANSWER_SYSTEM_PROMPT + ' If you don\'t know the answer, say "No Answer".'
)

ONE_SHOT_INPUT = (
    """ Q: Who directed The Last Horse? A: Edgar Neville
                Q: When was The Last Horse released? A: 1950
                Q: When was the University of Southampton founded? A: 1862
                Q: Where is the University of Southampton located? A: Southampton
                Q: What is the population of Stanton Township? A: 505
                Q: Where is Stanton Township? A: Champaign County, Illinois
                Q: Who is Neville A. Stanton? A: British Professor of Human Factors and Ergonomics
                Q: Where does Neville A. Stanton work? A: University of Southampton
                Q: What is Neville A. Stanton's profession? A: Professor
                Q: Who directed Finding Nemo? A: Andrew Stanton
                Q: When was Finding Nemo released? A: 2003
                Q: What company produced Finding Nemo? A: Pixar Animation Studios"""
    "\n\nQuestion: When was Neville A. Stanton's employer founded?"
    "\nThought: "
)

ONE_SHOT_OUTPUT = (
    "From the QA pairs, the employer of Neville A. Stanton is University of "
    "Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


class ChainAnswerGenerator:
    """Generate answers from evidence using oracle-style one-shot prompting."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        allow_no_answer: bool = False,
    ):
        self.model_name = model_name
        self.allow_no_answer = allow_no_answer
        self.client = OpenAI()

    def generate(self, question: str, evidence: List[str]) -> str:
        """Generate an answer for *question* given *evidence* strings.

        Returns the full LLM response (including reasoning). Falls back to an
        error message if the API call fails.
        """
        if not evidence:
            return "No evidence found to answer the question"

        evidence_text = "\n".join(evidence)
        prompt_text = f"\n{evidence_text}\n\n\nQuestion: {question}\n\n\nThought:\n\n"

        system_prompt = (
            ANSWER_SYSTEM_PROMPT_NO_ANSWER
            if self.allow_no_answer
            else ANSWER_SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ONE_SHOT_INPUT},
            {"role": "assistant", "content": ONE_SHOT_OUTPUT},
            {"role": "user", "content": prompt_text},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            return f"Error generating answer: {e}"
