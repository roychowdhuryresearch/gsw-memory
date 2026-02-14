"""
Context Generation Operator.

This module contains the ContextGenerator class for generating chunk-specific
context within document processing pipelines.
"""

from bespokelabs import curator

from ...prompts.operator_prompts import ContextPrompts


class ContextGenerator(curator.LLM):
    """Curator class for generating chunk-specific context."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to generate context."""
        return [
            {"role": "system", "content": ContextPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ContextPrompts.USER_PROMPT_TEMPLATE.format(
                    doc_text=input["doc_text"], chunk_text=input["chunk_text"]
                ),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract the generated context."""
        return [
            {
                "context": response["choices"][0]["message"]["content"].strip(),
                "doc_idx": input.get("doc_idx", 0),
                "chunk_idx": input.get("chunk_idx", 0),
                "global_id": input.get("global_id", "unknown"),
            }
        ]
