#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Question Decomposition

This script fine-tunes a language model using LoRA (Low-Rank Adaptation) to decompose
complex multi-hop questions into sequences of simpler single-hop questions.

Usage:
    python question_decomp_lora_ft.py --model_id Qwen/Qwen3-8B --num_train_epochs 3

    # Specify custom data paths
    python question_decomp_lora_ft.py --test_data_path ./musique.json --train_data_path ./musique_full_v1.0_train.jsonl

    # Use specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 python question_decomp_lora_ft.py --model_id Qwen/Qwen3-8B
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from bespokelabs import curator


# =============================================================================
# Pydantic Models for Question Decomposition
# =============================================================================

class DecomposedQuestion(BaseModel):
    question: str
    requires_retrieval: bool


class DecomposedQuestionList(BaseModel):
    questions: List[DecomposedQuestion]


# =============================================================================
# Curator Class for Question Decomposition
# =============================================================================

class ChainQuestionDecomposer(curator.LLM):
    """Curator class for decomposing multi-hop questions in parallel."""

    def __init__(self, **kwargs):
        """Initialize the question decomposer."""
        super().__init__(**kwargs)

    def prompt(self, input):
        """Create a decomposition prompt for each question."""
        decomposition_prompt = f"""Your task is to break down a complex multi-hop question into the most efficient sequence of single-hop, **atomic** questions.

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

<decomposition>
Question: [the question text]
Requires retrieval: [true/false]

And provide the response in the following json format:
{{
  "questions": [
    {{
      "question": "the decomposed question text",
      "requires_retrieval": "true/false"
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
Input: "{input['question']}"
Output:
"""

        return [
            {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
            {"role": "user", "content": decomposition_prompt}
        ]

    def parse(self, input, response: DecomposedQuestionList):
        """Parse the decomposition response."""
        questions = [{"question": q.question, "requires_retrieval": q.requires_retrieval} for q in response.questions]

        return [{
            "question_id": input['question_id'],
            "original_question": input['question'],
            "decomposed_questions": questions,
        }]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_musique_data(test_data_path: str, train_data_path: str, samples_per_type: int = 135):
    """
    Load and prepare Musique dataset for training.

    Args:
        test_data_path: Path to test JSON file
        train_data_path: Path to train JSONL file
        samples_per_type: Number of samples to take from each question type

    Returns:
        List of dicts with 'question_id' and 'question' keys
    """
    print(f"Loading test data from: {test_data_path}")
    test_musique = json.load(open(test_data_path))

    print(f"Loading train data from: {train_data_path}")
    train_musique = [json.loads(line) for line in open(train_data_path)]

    # Extract questions
    test_musique_questions = {q["id"]: q["question"] for q in test_musique}
    train_musique_questions = {q["id"]: q["question"] for q in train_musique}

    # Check for overlap
    overlap = set(train_musique_questions) & set(test_musique_questions)
    if overlap:
        print(f"Warning: Found {len(overlap)} overlapping questions between train and test")

    print(f"Total training questions: {len(train_musique_questions)}")

    # Count questions by type
    datapoint_type_counts = {}
    for qid in train_musique_questions:
        q_type = qid.split("_")[0]
        datapoint_type_counts[q_type] = datapoint_type_counts.get(q_type, 0) + 1

    print("Question type distribution:")
    for q_type, count in datapoint_type_counts.items():
        print(f"  {q_type}: {count}")

    # Sample from each type
    q_type_keys = list(datapoint_type_counts.keys())
    train_musique_questions_by_type = {q_type: [] for q_type in q_type_keys}

    for q_type in q_type_keys:
        count = 0
        for qid in train_musique_questions:
            if qid.split("_")[0] == q_type:
                train_musique_questions_by_type[q_type].append(qid)
                count += 1
                if count == samples_per_type:
                    break

    print(f"\nSampled questions per type (max {samples_per_type}):")
    for q_type, qids in train_musique_questions_by_type.items():
        print(f"  {q_type}: {len(qids)}")

    # Flatten to list
    train_musique_questions_by_type_list = [
        q for q_type in q_type_keys for q in train_musique_questions_by_type[q_type]
    ]

    # Create input format for decomposition
    decompose_inputs = [
        {"question_id": qid, "question": train_musique_questions[qid]}
        for qid in train_musique_questions_by_type_list
    ]

    print(f"\nTotal questions prepared for decomposition: {len(decompose_inputs)}")

    return decompose_inputs


def generate_decompositions(decompose_inputs, model_name="gpt-5", output_path=None):
    """
    Generate question decompositions using the ChainQuestionDecomposer.

    Args:
        decompose_inputs: List of dicts with 'question_id' and 'question'
        model_name: Model to use for decomposition
        output_path: Optional path to save decomposition results

    Returns:
        Dict mapping question_id to decomposition results
    """
    print(f"\nGenerating decompositions using {model_name}...")
    print(f"Processing {len(decompose_inputs)} questions...")

    # golden_question_decomposer = ChainQuestionDecomposer(
    #     model_name=model_name,
    #     # generation_params={"temperature": 0.0},
    #     response_format=DecomposedQuestionList
    # )

    # decomposition_dataset = golden_question_decomposer(decompose_inputs)

    # decomposition_results = {
    #     item["question_id"]: item
    #     for item in decomposition_dataset.dataset
    # }

    # print(f"Generated {len(decomposition_results)} decompositions")

    # # Save if path provided
    # if output_path:
    #     print(f"Saving decompositions to: {output_path}")
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(decomposition_results, f, indent=4, ensure_ascii=False)
    
    # load the decompositions from the output path
    with open("/home/yigit/codebase/gsw-memory/playground/question_decomp_local/q_decomp_training_5.json", 'r', encoding='utf-8') as f:
        decomposition_results = json.load(f)

    print(f"Loaded {len(decomposition_results)} decompositions")

    return decomposition_results


# =============================================================================
# Training Dataset Creation
# =============================================================================

def create_chat_messages(example):
    """
    Convert a single example into chat format for training.

    Args:
        example: Dict with 'original_question' and 'decomposed_questions' keys

    Returns:
        Dict with 'messages' key containing the chat-formatted data
    """
    original_question = example['original_question']
    decomposed_questions = example['decomposed_questions']

    # Serialize the decomposed questions to JSON format
    assistant_response = json.dumps(
        {"questions": decomposed_questions},
        indent=4,
        ensure_ascii=False
    )

    # Create the instruction prompt
    user_prompt = f"""Your task is to break down a complex multi-hop question into the most efficient sequence of single-hop, **atomic** questions.

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

<decomposition>
Question: [the question text]
Requires retrieval: [true/false]

And provide the response in the following json format:
{{
  "questions": [
    {{
      "question": "the decomposed question text",
      "requires_retrieval": "true/false"
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
Input: "{original_question}"
Output:
"""

    # Create chat messages
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    return {"messages": messages}


def create_training_dataset(decomposition_results):
    """
    Create HuggingFace Dataset from decomposition results.

    Args:
        decomposition_results: Dict mapping question_id to decomposition

    Returns:
        Dataset with 'messages' column
    """
    print("\nCreating training dataset...")

    # Convert to list
    decomposition_list = list(decomposition_results.values())
    print(f"Total examples: {len(decomposition_list)}")

    # Create HuggingFace Dataset
    raw_dataset = Dataset.from_list(decomposition_list)

    # Apply chat formatting
    training_dataset = raw_dataset.map(
        create_chat_messages,
        remove_columns=raw_dataset.column_names,
        desc="Creating chat-formatted training data"
    )

    print(f"Training dataset created with {len(training_dataset)} examples")
    print(f"Column names: {training_dataset.column_names}")

    return training_dataset


def load_non_thinking_template(template_path="playground/question_decomp_local/qwen3_nonthinking.jinja"):
    """
    Load and fix the non-thinking chat template.

    Args:
        template_path: Path to the Jinja2 template file

    Returns:
        Modified template string with <think> tags removed
    """
    print(f"\nLoading non-thinking template from: {template_path}")

    with open(template_path, 'r') as f:
        template_content = f.read()

    print("✓ No Thinking Template Loaded")

    return template_content


def test_chat_template(tokenizer, training_dataset, chat_template=None):
    """
    Test that the chat template works correctly before training.

    Args:
        tokenizer: Tokenizer instance
        training_dataset: Dataset with 'messages' column
        chat_template: Optional custom chat template string
    """
    print("\nTesting chat template compatibility...")

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("✓ Chat template found!")
        print(f"  Template preview (first 200 chars): {str(tokenizer.chat_template)[:200]}...")
    else:
        print("✗ WARNING: No chat template found! This may cause errors.")
        print("  Consider using an instruct-tuned model variant.")

    # Test with sample
    print("\nTesting formatting with sample data...")
    try:
        sample = training_dataset[0]

        # Apply chat template (use custom template if provided)
        if chat_template:
            formatted = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template
            )
        else:
            formatted = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False
            )

        print("✓ Formatting successful!")
        print(f"  Original message length: {len(str(sample['messages']))}")
        print(f"  Formatted text length: {len(formatted)}")
        print(f"\nFormatted output preview (first 500 chars):")
        print(formatted[:500])
        print("\n... [truncated] ...")
        print(f"\nLast 200 chars:")
        print(formatted[-200:])

        # Check if <think> tags are present
        if '<think>' in formatted:
            print("\n⚠ WARNING: <think> tags found in formatted output!")
            print("  This may cause issues during inference.")
        else:
            print("\n✓ No <think> tags in formatted output")

    except Exception as e:
        print(f"\n✗ ERROR during formatting: {e}")
        print("  You may need to adjust the formatting function or use a different model.")

    print("\n" + "="*60)
    print("Test complete! Review the output above before training.")


# =============================================================================
# LoRA Training Functions
# =============================================================================

def train(model_id, tokenizer, dataset, training_args, chat_template=None):
    """
    Train a model with LoRA on local GPU.

    Args:
        model_id: HuggingFace model identifier
        tokenizer: Tokenizer instance
        dataset: Training dataset with 'messages' column
        training_args: TrainingArguments instance
        chat_template: Optional custom chat template string

    Returns:
        Trained SFTTrainer instance
    """
    print(f"\nLoading model: {model_id}")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    print("Configuring LoRA...")
    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    def formatting_function(example):
        """Format a single example using the tokenizer's chat template."""
        if chat_template:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template
            )
        else:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )

    print("Initializing trainer...")
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_function,
        # max_seq_length=4096,
        # packing=True,
    )

    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    return trainer


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )
    test_data_path: str = field(
        default="/home/yigit/codebase/gsw-memory/playground_data/musique.json",
        metadata={"help": "Path to test data JSON file."},
    )
    train_data_path: str = field(
        default="/home/yigit/codebase/gsw-memory/playground_data/musique_full_v1.0_train.jsonl",
        metadata={"help": "Path to train data JSONL file."},
    )
    output_dir: str = field(
        default="./question_decomp_lora",
        metadata={"help": "Directory to save the trained model."},
    )
    decomposition_output_path: str = field(
        default="./q_decomp_training_5.json",
        metadata={"help": "Path to save decomposition results."},
    )
    samples_per_type: int = field(
        default=135,
        metadata={"help": "Number of samples to take from each question type."},
    )
    decomposition_model: str = field(
        default="gpt-5",
        metadata={"help": "Model to use for question decomposition."},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs."},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU for training."},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps."},
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate for training."},
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps."},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps."},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps."},
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep."},
    )
    test_template_only: bool = field(
        default=False,
        metadata={"help": "Only test the chat template without training."},
    )
    use_non_thinking_template: bool = field(
        default=True,
        metadata={"help": "Use the non-thinking chat template (removes <think> tags)."},
    )
    chat_template_path: str = field(
        default="playground/question_decomp_local/qwen3_nonthinking.jinja",
        metadata={"help": "Path to custom chat template file."},
    )


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Question Decomposition")

    # Add arguments from ScriptArguments
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model ID from HuggingFace hub")
    parser.add_argument("--test_data_path", type=str,
                        default="/home/yigit/codebase/gsw-memory/playground_data/musique.json",
                        help="Path to test data JSON")
    parser.add_argument("--train_data_path", type=str,
                        default="/home/yigit/codebase/gsw-memory/playground_data/musique_full_v1.0_train.jsonl",
                        help="Path to train data JSONL")
    parser.add_argument("--output_dir", type=str, default="./question_decomp_lora",
                        help="Output directory for trained model")
    parser.add_argument("--decomposition_output_path", type=str, default="./q_decomp_training_5.json",
                        help="Path to save decomposition results")
    parser.add_argument("--samples_per_type", type=int, default=135,
                        help="Samples per question type")
    parser.add_argument("--decomposition_model", type=str, default="gpt-5",
                        help="Model for decomposition generation")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum checkpoints to keep")
    parser.add_argument("--test_template_only", action="store_true",
                        help="Only test chat template without training")
    parser.add_argument("--skip_decomposition", action="store_true",
                        help="Skip decomposition generation and load from file")

    args = parser.parse_args()

    # Print GPU info
    print("="*60)
    print("GPU Configuration")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Step 1: Load and prepare data
    if not args.skip_decomposition:
        print("="*60)
        print("Step 1: Loading and Preparing Data")
        print("="*60)
        decompose_inputs = load_musique_data(
            args.test_data_path,
            args.train_data_path,
            args.samples_per_type
        )

        # Step 2: Generate decompositions
        print("\n" + "="*60)
        print("Step 2: Generating Question Decompositions")
        print("="*60)
        decomposition_results = generate_decompositions(
            decompose_inputs,
            model_name=args.decomposition_model,
            output_path=args.decomposition_output_path
        )
    else:
        print("="*60)
        print("Loading existing decomposition results")
        print("="*60)
        print(f"Loading from: {args.decomposition_output_path}")
        with open(args.decomposition_output_path, 'r', encoding='utf-8') as f:
            decomposition_results = json.load(f)
        print(f"Loaded {len(decomposition_results)} decompositions")

    # Step 3: Create training dataset
    print("\n" + "="*60)
    print("Step 3: Creating Training Dataset")
    print("="*60)
    training_dataset = create_training_dataset(decomposition_results)

    # Step 4: Configure tokenizer
    print("\n" + "="*60)
    print("Step 4: Configuring Tokenizer")
    print("="*60)
    print(f"Loading tokenizer for: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Configure padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = 'right'

    print(f"\nTokenizer configured:")
    print(f"  Model: {args.model_id}")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  Padding side: {tokenizer.padding_side}")

    # Step 4.5: Load non-thinking chat template
    print("\n" + "="*60)
    print("Step 4.5: Loading Non-Thinking Chat Template")
    print("="*60)
    chat_template = load_non_thinking_template()

    # Step 5: Test chat template
    print("\n" + "="*60)
    print("Step 5: Testing Chat Template")
    print("="*60)
    test_chat_template(tokenizer, training_dataset, chat_template=chat_template)

    if args.test_template_only:
        print("\nTemplate test complete. Exiting without training.")
        return

    # Step 6: Configure training
    print("\n" + "="*60)
    print("Step 6: Configuring Training Arguments")
    print("="*60)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="wandb",
    )

    print(f"Training configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")

    # Step 7: Train
    print("\n" + "="*60)
    print("Step 7: Training Model")
    print("="*60)
    trainer = train(args.model_id, tokenizer, training_dataset, training_args, chat_template=chat_template)

    # Step 8: Save final model
    final_output_path = os.path.join(args.output_dir, "final")
    print("\n" + "="*60)
    print("Step 8: Saving Final Model")
    print("="*60)
    print(f"Saving to: {final_output_path}")
    trainer.save_model(final_output_path)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {final_output_path}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Decomposition results saved to: {args.decomposition_output_path}")


if __name__ == "__main__":
    main()
