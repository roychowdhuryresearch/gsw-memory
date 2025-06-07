"""
GSW Generation Operator.

This module contains the GSWOperator class for generating Generative Semantic
Workspaces from text using sophisticated semantic role extraction.
"""

from bespokelabs import curator

from ...prompts.operator_prompts import OperatorPrompts


class GSWOperator(curator.LLM):
    """Curator class for generating GSWs using sophisticated semantic role extraction."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to generate a GSW."""
        return [
            {"role": "system", "content": OperatorPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": OperatorPrompts.USER_PROMPT_TEMPLATE.replace(
                    "{{input_text}}", input["text"]
                ).replace(
                    "{{background_context}}", input.get("context", "")
                ),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract text and graph."""
        parsed_response = {
            "text": input["text"],
            "idx": input["idx"],
            "graph": response["choices"][0]["message"]["content"],
            "context": input.get("context", ""),
            "doc_idx": input.get("doc_idx", input["idx"]),
            "global_id": input.get("global_id", "unknown"),
        }
        
        # Include sentence indices if available
        if "start_sentence" in input:
            parsed_response["start_sentence"] = input["start_sentence"]
        if "end_sentence" in input:
            parsed_response["end_sentence"] = input["end_sentence"]

        return [parsed_response]