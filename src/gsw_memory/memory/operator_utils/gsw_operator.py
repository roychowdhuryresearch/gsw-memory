"""
GSW Generation Operator.

This module contains the GSWOperator class for generating Generative Semantic
Workspaces from text using sophisticated semantic role extraction.
"""

from bespokelabs import curator


from ...prompts.operator_prompts import FactualExtractionPrompts, OperatorPrompts, PromptType
from ..models import GSWStructure


# class GSWOperator(curator.LLM):
#     """Curator class for generating GSWs using sophisticated semantic role extraction."""

#     return_completions_object = True

#     def __init__(self, prompt_type: PromptType = PromptType.EPISODIC, **kwargs):
#         """Initialize GSWOperator with specified prompt type.
        
#         Args:
#             prompt_type: Type of prompts to use (EPISODIC or FACTUAL)
#             **kwargs: Additional arguments passed to curator.LLM
#         """
#         super().__init__(**kwargs)
#         self.prompt_type = prompt_type
        
#         # Select appropriate prompt class based on type
#         if prompt_type == PromptType.EPISODIC:
#             self.prompt_class = OperatorPrompts
#         elif prompt_type == PromptType.FACTUAL:
#             self.prompt_class = FactualExtractionPrompts
#         else:
#             raise ValueError(f"Unsupported prompt type: {prompt_type}")

#     def prompt(self, input):
#         """Create a prompt for the LLM to generate a GSW."""
#         return [
#             {"role": "system", "content": self.prompt_class.SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": self.prompt_class.USER_PROMPT_TEMPLATE.format(
#                     input_text=input["text"],
#                     background_context=input.get("context", "")
#                 ),
#             },
#         ]

#     def parse(self, input, response):
#         """Parse the LLM response to extract text and graph."""
#         parsed_response = {
#             "text": input["text"],
#             "idx": input["idx"],
#             "graph": response["choices"][0]["message"]["content"],
#             "context": input.get("context", ""),
#             "doc_idx": input.get("doc_idx", input["idx"]),
#             "global_id": input.get("global_id", "unknown"),
#         }
        
#         # Include sentence indices if available
#         if "start_sentence" in input:
#             parsed_response["start_sentence"] = input["start_sentence"]
#         if "end_sentence" in input:
#             parsed_response["end_sentence"] = input["end_sentence"]

#         return parsed_response


#Uncomment for structured outputs
class GSWOperator(curator.LLM):
    """Curator class for generating GSWs using sophisticated semantic role extraction."""

    # response_format = GSWStructure  # Use constrained decoding

    def __init__(self, prompt_type: PromptType = PromptType.EPISODIC, **kwargs):
        """Initialize GSWOperator with specified prompt type.

        Args:
            prompt_type: Type of prompts to use (EPISODIC or FACTUAL)
            **kwargs: Additional arguments passed to curator.LLM
        """
        super().__init__(**kwargs)
        self.prompt_type = prompt_type

        # Select appropriate prompt class based on type
        if prompt_type == PromptType.EPISODIC:
            self.prompt_class = OperatorPrompts
        elif prompt_type == PromptType.FACTUAL:
            self.prompt_class = FactualExtractionPrompts
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

    def prompt(self, input):
        """Create a prompt for the LLM to generate a GSW."""
        return [
            {"role": "system", "content": self.prompt_class.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.prompt_class.USER_PROMPT_TEMPLATE.format(
                    input_text=input["text"],
                    background_context=input.get("context", "")
                ),
            },
        ]

    def parse(self, input, response: GSWStructure):
        """Parse the LLM response - already a validated GSWStructure Pydantic object!"""
        # Convert Pydantic object to dict for PyArrow serialization
        gsw_dict = response.model_dump() if response else None

        parsed_response = {
            "text": input["text"],
            "idx": input.get("idx", 0),
            "gsw": gsw_dict,  # Convert to dict for PyArrow compatibility
            "context": input.get("context", ""),
            "doc_idx": input.get("doc_idx", input.get("idx", 0)),
            "global_id": input.get("global_id", "unknown"),
        }

        # Include sentence indices if available
        if "start_sentence" in input:
            parsed_response["start_sentence"] = input["start_sentence"]
        if "end_sentence" in input:
            parsed_response["end_sentence"] = input["end_sentence"]

        return parsed_response