"""Standalone wrappers for the supplied Qwen decomposer, encoder, and reranker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def _model_kwargs(*, dtype: str, quantized: bool, device_map: str):
    import torch

    kwargs = {
        "dtype": torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        "device_map": device_map,
    }
    if quantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=kwargs["dtype"],
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    stop = text.rfind("}")
    if start < 0 or stop < start:
        raise ValueError(f"Model did not return a JSON object: {text[:200]}")
    return json.loads(text[start : stop + 1])


class QwenDecomposer:
    def __init__(
        self,
        model_name: str,
        prompt_path: str | Path,
        *,
        quantized: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **_model_kwargs(
                dtype=dtype, quantized=quantized, device_map=device_map
            ),
        ).eval()

    def decompose(self, question: str, *, max_new_tokens: int = 768) -> list[dict]:
        import torch

        prompt = self.prompt_template.format(question=question)
        messages = [
            {
                "role": "system",
                "content": (
                    "You break complex questions into efficient atomic "
                    "retrieval steps and return valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        response = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        payload = _extract_json_object(response)
        questions = payload.get("questions")
        if not isinstance(questions, list):
            raise ValueError("Decomposer JSON must contain a questions list")
        return [
            {
                "question": str(item["question"]),
                "requires_retrieval": str(
                    item.get("requires_retrieval", "true")
                ).casefold()
                in {"true", "1", "yes"},
            }
            for item in questions
        ]


class QwenQueryEncoder:
    TASK = (
        "Given a query, create an embedding that captures the semantic meaning "
        "for similarity comparison with QA pairs."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        *,
        quantized: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            **_model_kwargs(
                dtype=dtype,
                quantized=quantized,
                device_map=device_map,
            ),
        ).eval()

    def encode(self, texts: Sequence[str], max_length: int = 512):
        import torch
        import torch.nn.functional as functional

        instructed = [
            f"Instruct: {self.TASK}\nQuery: {text}" for text in texts
        ]
        batch = self.tokenizer(
            instructed,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model(**batch)
            embeddings = output.last_hidden_state[:, -1]
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().numpy()


class QwenReranker:
    INSTRUCTION = (
        "Given a decomposed question, score a GSW QA pair based on how likely "
        "it is to resolve that retrieval hop."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8B",
        *,
        quantized: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_length: int = 2048,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **_model_kwargs(
                dtype=dtype, quantized=quantized, device_map=device_map
            ),
        ).eval()
        self.max_length = max_length
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            "requirements based on the Query and the Instruct provided. The "
            'answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = (
            "<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(
            prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            suffix, add_special_tokens=False
        )

    def _format(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.INSTRUCTION}\n"
            f"<Query>: {query}\n<Document>: {document}"
        )

    def score(self, query: str, documents: Sequence[str], batch_size: int = 8):
        import torch

        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            texts = [
                self._format(query, document)
                for document in documents[start : start + batch_size]
            ]
            encoded = self.tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=(
                    self.max_length
                    - len(self.prefix_tokens)
                    - len(self.suffix_tokens)
                ),
            )
            encoded["input_ids"] = [
                self.prefix_tokens + ids + self.suffix_tokens
                for ids in encoded["input_ids"]
            ]
            batch = self.tokenizer.pad(
                encoded, padding=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits[:, -1, :]
                binary = torch.stack(
                    [logits[:, self.false_id], logits[:, self.true_id]], dim=1
                )
                probabilities = torch.softmax(binary, dim=1)[:, 1]
            scores.extend(probabilities.float().cpu().tolist())
        return scores
