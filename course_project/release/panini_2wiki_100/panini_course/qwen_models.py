"""Standalone wrappers for decomposition, encoding, reranking, and answering."""

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

    def generate_raw(self, question: str, *, max_new_tokens: int = 768) -> str:
        """Generate the model response without parsing it.

        Keeping generation separate from parsing lets long-running Colab jobs
        persist the original response before validating its JSON structure.
        """

        import torch

        messages = self.build_messages(question)
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
        return self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def build_messages(self, question: str) -> list[dict[str, str]]:
        """Build the research decomposer's two-message prompt."""

        prompt = self.prompt_template.format(question=question)
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that breaks down complex "
                    "questions into simple steps."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        """Parse and normalize a raw decomposer response."""

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

    def decompose(self, question: str, *, max_new_tokens: int = 768) -> list[dict]:
        """Generate and parse a decomposition in one call."""

        return self.parse_response(
            self.generate_raw(question, max_new_tokens=max_new_tokens)
        )


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


class QwenAnswerer:
    """Course-scale local wrapper for PANINI's research answer prompt."""

    SYSTEM_PROMPT = (
        "As an advanced reading comprehension assistant, your task is to analyze "
        "precise QA pairs extracted from the documents and corresponding questions "
        "meticulously. Your response start after \"Thought: \", where you will "
        "methodically break down the reasoning process, illustrating how you arrive "
        "at conclusions. Conclude with \"Answer: \" to present only a concise, "
        "definitive response, devoid of additional elaborations."
    )
    ONE_SHOT_EVIDENCE = """ Q: Who directed The Last Horse? A: Edgar Neville
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
    ONE_SHOT_QUESTION = "When was Neville A. Stanton's employer founded?"
    ONE_SHOT_ANSWER = (
        "From the QA pairs, the employer of Neville A. Stanton is University of "
        "Southampton. The University of Southampton was founded in 1862. "
        "\nAnswer: 1862."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        *,
        quantized: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
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

    def answer(
        self,
        question: str,
        evidence: Sequence[str],
        *,
        max_new_tokens: int = 512,
    ) -> str:
        return self.answer_with_trace(
            question, evidence, max_new_tokens=max_new_tokens
        )["answer"]

    def answer_with_trace(
        self,
        question: str,
        evidence: Sequence[str],
        *,
        max_new_tokens: int = 512,
    ) -> dict[str, object]:
        import torch

        messages = self.build_messages(question, evidence)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # The research prompt explicitly requests a visible Thought and
            # Answer. Qwen's separate hidden-thinking mode is unnecessary and
            # costs substantial free-tier Colab memory/time.
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
        ).strip()
        return {
            "answer": self.parse_response(response),
            "response": response,
            "messages": messages,
        }

    @classmethod
    def build_messages(
        cls, question: str, evidence: Sequence[str], *, allow_no_answer: bool = False
    ) -> list[dict[str, str]]:
        """Build the same four-message, one-shot prompt as evaluate_panini.py."""

        system = cls.SYSTEM_PROMPT
        if allow_no_answer:
            system += ' If you don\'t know the answer, say "No Answer".'
        evidence_text = "\n".join(evidence)
        prompt_text = f"\n{evidence_text}\n\n\nQuestion: {question}\n\n\nThought: \n\n"
        one_shot_input = (
            cls.ONE_SHOT_EVIDENCE
            + "\n\nQuestion: "
            + cls.ONE_SHOT_QUESTION
            + "\nThought: "
        )
        return [
            {
                "role": "system",
                "content": system,
            },
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": cls.ONE_SHOT_ANSWER},
            {
                "role": "user",
                "content": prompt_text,
            },
        ]

    @staticmethod
    def parse_response(response: str) -> str:
        answer = response.split("Answer: ")[-1].strip() if "Answer: " in response else response.strip()
        if answer.endswith(".") and answer[:-1].replace(",", "").replace(" ", "").isdigit():
            answer = answer[:-1]
        return answer
