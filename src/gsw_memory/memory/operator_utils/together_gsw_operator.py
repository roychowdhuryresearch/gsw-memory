"""
GSW Operator using Together API directly (bypasses curator async issues).

This provides a drop-in replacement for the curator-based GSWOperator
that uses the Together API client directly with json_schema structured output.
"""

import json
import together
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ...prompts.operator_prompts import (
    FactualExtractionPrompts,
    OperatorPrompts,
    PromptType,
)
from ..models import GSWStructure


class TogetherGSWOperator:
    """GSW operator using Together API directly with batch processing."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct-Turbo",
        api_key: Optional[str] = None,
        prompt_type: PromptType = PromptType.FACTUAL,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        max_retries: int = 3,
        show_progress: bool = True,
    ):
        """Initialize Together GSW operator.

        Args:
            model_name: Together AI model name (e.g., "Qwen/Qwen2.5-72B-Instruct-Turbo")
            api_key: Together API key (uses TOGETHER_API_KEY env var if not provided)
            prompt_type: Type of prompts to use (FACTUAL or EPISODIC)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retries on failure
            show_progress: Whether to show progress bar
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.show_progress = show_progress

        # Select prompt class
        if prompt_type == PromptType.EPISODIC:
            self.prompt_class = OperatorPrompts
        elif prompt_type == PromptType.FACTUAL:
            self.prompt_class = FactualExtractionPrompts
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

        # Initialize Together client
        if api_key:
            self.client = together.Together(api_key=api_key)
        else:
            self.client = together.Together()  # Uses TOGETHER_API_KEY from env

    def process_single(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document.

        Args:
            doc: Document dict with 'text', 'idx', 'doc_idx', 'global_id', 'context'

        Returns:
            Dict with GSW result and metadata matching curator output format
        """
        messages = [
            {"role": "system", "content": self.prompt_class.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.prompt_class.USER_PROMPT_TEMPLATE.format(
                    input_text=doc["text"], background_context=doc.get("context", "")
                ),
            },
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    response_format={
                        "type": "json_schema",
                        "schema": GSWStructure.model_json_schema(),
                    },
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Parse JSON response
                output = json.loads(response.choices[0].message.content)

                # Validate by creating GSWStructure
                gsw = GSWStructure(**output)

                # Return in curator-compatible format
                return {
                    "text": doc["text"],
                    "idx": doc.get("idx", 0),
                    "gsw": gsw.model_dump(),  # Convert to dict
                    "context": doc.get("context", ""),
                    "doc_idx": doc.get("doc_idx", doc.get("idx", 0)),
                    "global_id": doc.get("global_id", "unknown"),
                    "start_sentence": doc.get("start_sentence"),
                    "end_sentence": doc.get("end_sentence"),
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    if self.show_progress:
                        print(
                            f"  ⚠️  Retry {attempt + 1}/{self.max_retries} for {doc.get('global_id', 'unknown')}: {str(e)[:80]}"
                        )
                    continue
                else:
                    if self.show_progress:
                        print(
                            f"  ❌ Failed after {self.max_retries} attempts: {str(e)[:80]}"
                        )

                    # Return None for failed documents (matching curator behavior)
                    return {
                        "text": doc["text"],
                        "idx": doc.get("idx", 0),
                        "gsw": None,
                        "context": doc.get("context", ""),
                        "doc_idx": doc.get("doc_idx", doc.get("idx", 0)),
                        "global_id": doc.get("global_id", "unknown"),
                        "start_sentence": doc.get("start_sentence"),
                        "end_sentence": doc.get("end_sentence"),
                        "error": str(e),
                    }

    def __call__(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents.

        Args:
            documents: List of document dicts

        Returns:
            List of GSW results in curator-compatible format
        """
        results = []

        iterator = tqdm(
            documents, desc="Generating GSWs", disable=not self.show_progress
        )

        for doc in iterator:
            result = self.process_single(doc)
            results.append(result)

        # Print summary
        if self.show_progress:
            successes = sum(1 for r in results if r.get("gsw") is not None)
            failures = len(results) - successes
            print(f"\n✅ Generated {successes}/{len(results)} GSWs ({failures} failed)")

        return results
