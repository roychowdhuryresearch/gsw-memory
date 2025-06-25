import json
from typing import Dict, List

from bespokelabs import curator


class AnsweringAgent(curator.LLM):
    """
    Curator class for answering questions using pre-computed GSW entity summaries.
    """

    return_completions_object = True

    def prompt(self, input_data):
        """Create a prompt for the LLM to answer a question using entity summaries."""

        system_prompt = """You are a question answering agent that only uses the provided information to answer questions.

            Your task is to answer questions based exclusively on the chapter by chapter narrative summaries provided. Do not use any external knowledge.

            You are provided summaries ranked by relevance to the question, earlier summaries are more relevant than later ones.

            """

        user_prompt = f"""Please answer the following question 

            QUESTION: {input_data["question"]}

            KNOWLEDGE BASE INFORMATION (Chapter by Chapter Narrative Summaries):
            {input_data["context"]}

            Go through each chapter provided to you step by step and reason about whether the information in the chapter is relevant to the question.

            Provide your answer in the following JSON format:

            ```json
            {{
            "question": "Question text",
            "reasoning": "chapter by chapter reasoning",
            "answer": "Comma seperated answers to the query"
            }}
            ```
            """

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse the LLM response and return as an answer string."""
        answer_text = response["choices"][0]["message"]["content"].strip()

        # Attempt to parse the JSON block
        json_text = answer_text  # Default to full text
        if "```json" in answer_text:
            json_text = answer_text.split("```json")[1].split("```")[0].strip()
        elif "```" in answer_text and answer_text.count("```") >= 2:
            # Handle cases where ```json might be missing but ``` is present
            potential_json = answer_text.split("```", 2)
            if len(potential_json) > 1:
                json_text = potential_json[1].strip()

        try:
            model_response = json.loads(json_text)
            # Handle potential list answers from the model
            if isinstance(model_response.get("answer"), list):
                answer = ", ".join(str(item) for item in model_response["answer"])
            else:
                answer = model_response.get(
                    "answer", json_text
                )  # Fallback if 'answer' key is missing

            reasoning = model_response.get("reasoning", "")  # Fallback for reasoning
        except (json.JSONDecodeError, AttributeError):
            # Handle cases where JSON is malformed or not found, or model_response isn't a dict
            print(
                f"Warning: Could not parse JSON from response. Raw text: {answer_text}"
            )
            answer = answer_text  # Use the raw response as answer if parsing fails
            reasoning = "Could not parse reasoning from response."

        # Return a structure similar to the original parser for consistency
        return {
            "question": input_data["question"],
            "correct_answer": input_data.get(
                "correct_answer", "N/A"
            ),  # Use .get for safety
            "answer": answer,
            "chapters_hit": input_data.get("chapters_hit", []),  # Use .get for safety
            "correct_chapters": input_data.get(
                "correct_answer_chapters", []
            ),  # Use .get for safety
            "retrieval_type": input_data.get(
                "retrieval_type", "summary_based"
            ),  # Indicate retrieval type
            "reasoning": reasoning,
            "context_used": input_data["context"],  # The summary context passed in
        }

    def answer_question(self, reranked_results: List[Dict]) -> str:
        """Answer a question using the ranked summaries."""

        questions_with_context = [
            {
                "question": result["question"],
                "context": result["context_to_answering_agent"],
            }
            for result in reranked_results
        ]

        answers = self(questions_with_context)

        return answers.dataset
