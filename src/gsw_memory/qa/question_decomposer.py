"""
Question decomposition for multi-hop QA.

Breaks complex multi-hop questions into atomic single-hop sub-questions
using LLM-based structured output.
"""

import logging
from typing import Any, Dict, List

from openai import OpenAI

from .chain_models import DecomposedQuestionList

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """Your task is to break down a complex multi-hop question into the most efficient sequence of single-hop, **atomic** questions.

## Your Main Goal: Build Smart Bridges, Don't Just Collect Nouns
The most critical skill is to convert complex logical clauses (like "despite," "the country where," "the year before") into a single, powerful **bridging question**. This question should use a known entity as context to find the next one. Avoid finding all the entities separately and then trying to figure out how they connect.

---
## A Simple Analogy for Efficiency

**Question:** "What is the phone number of the mother of the tallest player on the Lakers?"

** Inefficient Path:**
1.  Who are the players on the Lakers?
2.  What are all their heights?
3.  Who is the mother of the tallest player? *(This step is a logical leap)*

** Efficient Path:**
1.  Who is the tallest player on the Lakers?
2.  Who is the mother of `<ENTITY_Q1>`?
3.  What is the phone number of `<ENTITY_Q2>`?

---
## How to Decompose a Question
This process follows a logical flow from high-level analysis to the fine-tuning of your question chain.

### 1. Analyze the Query's Components
First, break down the original question into its fundamental building blocks. Identify the core **entities** (people, places, organizations), their **properties** (attributes like rank, location, date), and the **relationships** that connect them.

### 2. Construct an Atomic Chain
Next, formulate a sequence of questions where each question retrieves a single fact.
* **Isolate Comparisons:** Don't ask "who is faster?" Ask for the specific rank or time of each person involved.
* **Link with Placeholders:** Use `<ENTITY_Qn>` to pass the answer from a previous question (`Qn`) into the next one.

### 3. Optimize for Efficiency and Precision
Your final goal is the **shortest and most direct path** to the answer.
* **Embed Constraints to Build Bridges:** If a piece of information is only a filter (like a date or location), embed it as a constraint in the next question instead of asking for it directly.
  **Important note for bridges:** There can be no `<ENTITY_Qn>` in the first question if the nth question DOES NOT require retrieval.

## Formatting
Format each decomposed question as follows:


Question: [the question text]
Requires retrieval: [True/False]

And provide the response in the following JSON format:
{{
  "questions": [
    {{
      "question": "the decomposed question text",
      "requires_retrieval": "True/False"
    }}
  ]
}}

Examples:

Input: "What is the birth year of the spouse of the director of Casablanca?"
Output:
{{
    "questions": [
        {{
            "question": "Who directed Casablanca?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who was <ENTITY_Q1>'s spouse?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "What is <ENTITY_Q2>'s birth year?",
            "requires_retrieval": "true"
        }}
    ]
}}

Input: "Which film has the director who is older, Dune or The Dark Knight?"
Output:
{{
    "questions": [
        {{
            "question": "Who directed Dune?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who directed The Dark Knight?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": "false"
        }}
    ]
}}


IMPORTANT:
    AVOID over-decomposition like this:
    DON'T break "Who is John Doe?" into:
    1. Who is John Doe? → "English"
    2. When was <ENTITY_Q1> born? → "When was English born?"

    DO ask directly: "When was John Doe born?"

Now decompose this question:
Input: "{question}"
Output:
"""


class QuestionDecomposer:
    """Decompose multi-hop questions into atomic sub-questions via OpenAI structured output."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI()

    def decompose(self, question: str) -> List[Dict[str, Any]]:
        """Decompose *question* into a list of sub-question dicts.

        Each dict has keys ``question`` (str) and ``requires_retrieval`` (bool).
        Falls back to the original question as a single sub-question on error.
        """
        prompt = DECOMPOSITION_PROMPT.format(question=question)
        try:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that breaks down complex questions into simple steps.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=DecomposedQuestionList,
            )
            parsed = response.choices[0].message.parsed
            questions = [
                {"question": q.question, "requires_retrieval": q.requires_retrieval}
                for q in parsed.questions
            ]
            return (
                questions
                if questions
                else [{"question": question, "requires_retrieval": True}]
            )
        except Exception as e:
            logger.warning("Decomposition failed: %s", e)
            return [{"question": question, "requires_retrieval": True}]
