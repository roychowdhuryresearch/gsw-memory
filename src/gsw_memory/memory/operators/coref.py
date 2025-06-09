"""
Coreference Resolution Operator.

This module contains the CorefOperator class for performing coreference resolution
on text documents using LLM-based processing.
"""

from bespokelabs import curator

from ...prompts.operator_prompts import CorefPrompts


class CorefOperator(curator.LLM):
    """Curator class for performing coreference resolution."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to perform coreference resolution."""
        return [
            {"role": "system", "content": CorefPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CorefPrompts.USER_PROMPT_TEMPLATE.format(text=input["text"]),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract resolved text."""
        return [
            {
                "text": response["choices"][0]["message"]["content"],
                "idx": input["idx"],
            }
        ]
