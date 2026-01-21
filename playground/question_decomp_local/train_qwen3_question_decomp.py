#!/usr/bin/env python3
"""
Question Decomposition Training Script
Trains Qwen3-0.6B model for multi-hop question decomposition with LLM judge evaluation.
"""

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import Dataset, ClassLabel
from dotenv import load_dotenv
from openai import OpenAI
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


# Load environment variables
load_dotenv()


class QuestionDecompositionJudge:
    """GPT-4 based judge for evaluating question decomposition quality."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def create_judge_prompt(self, original_question: str, decomposed_questions: List[Dict]) -> str:
        """Create evaluation prompt for the judge."""
        decomp_str = json.dumps({"questions": decomposed_questions}, indent=2)

        return f"""You are an expert evaluator of question decomposition quality for multi-hop QA systems.

**Original Question:** {original_question}

**Generated Decomposition:**
{decomp_str}

**Evaluation Criteria** (Score each decomposed question 1-5):

1. **Atomicity**: Is this a single-hop question retrieving only one piece of information?
2. **Bridge Building**: Proper use of <ENTITY_Qn> placeholders to reference previous answers?
3. **Efficiency**: Most direct path, avoiding over-decomposition?
4. **Correctness**: Logically sound and contributes to answering the original question?
5. **Retrieval Flag**: Is requires_retrieval set correctly?

**Response Format (JSON):**
{{
  "evaluations": [
    {{
      "question_index": 0,
      "question_text": "...",
      "scores": {{
        "atomicity": 5,
        "bridge_building": 5,
        "efficiency": 5,
        "correctness": 5,
        "retrieval_flag": 5
      }},
      "average": 5.0,
      "feedback": "Brief explanation"
    }}
  ],
  "overall_average": 5.0,
  "overall_feedback": "Brief summary"
}}"""

    def judge_decomposition(self, original_question: str, decomposed_questions: List[Dict]) -> Dict:
        """Evaluate a single decomposition."""
        prompt = self.create_judge_prompt(original_question, decomposed_questions)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of question decomposition quality. Provide detailed, fair evaluations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result["original_question"] = original_question
            result["decomposed_questions"] = decomposed_questions

            return result
        except Exception as e:
            return {
                "error": str(e),
                "original_question": original_question,
                "evaluations": [],
                "overall_average": 0.0
            }

    def compute_aggregate_metrics(self, evaluation_results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics across evaluations."""
        valid_results = [r for r in evaluation_results if "error" not in r]

        if not valid_results:
            return {"error_rate": 1.0}

        overall_scores = [r["overall_average"] for r in valid_results]

        # Aggregate per-criterion scores
        all_scores = {
            "atomicity": [],
            "bridge_building": [],
            "efficiency": [],
            "correctness": [],
            "retrieval_flag": []
        }

        for result in valid_results:
            for eval_item in result.get("evaluations", []):
                scores = eval_item.get("scores", {})
                for criterion in all_scores:
                    if criterion in scores:
                        all_scores[criterion].append(scores[criterion])

        metrics = {
            "overall_average": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            "num_evaluated": len(valid_results),
            "error_rate": (len(evaluation_results) - len(valid_results)) / len(evaluation_results)
        }

        for criterion, values in all_scores.items():
            if values:
                metrics[f"{criterion}_avg"] = sum(values) / len(values)

        return metrics


class LLMJudgeEvaluationCallback(TrainerCallback):
    """
    Callback that evaluates model outputs using LLM judge during evaluation.
    """

    def __init__(
        self,
        eval_dataset,
        judge: QuestionDecompositionJudge,
        tokenizer,
        chat_template: str,
        num_samples: int = 20,
        logs_dir: str = "./judge_logs"
    ):
        self.eval_dataset = eval_dataset
        self.judge = judge
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.num_samples = num_samples
        self.logs_dir = logs_dir

        # Create logs directory
        os.makedirs(logs_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called when trainer runs evaluation."""
        current_step = state.global_step

        print(f"\n{'='*60}")
        print(f"🔍 Running LLM Judge Evaluation at step {current_step}")
        print(f"{'='*60}")

        # Get the actual training model from kwargs
        model = kwargs.get('model')
        if model is None:
            print("⚠️  Model not found in kwargs, skipping evaluation")
            return

        # Sample examples from eval set
        eval_indices = random.sample(range(len(self.eval_dataset)), min(self.num_samples, len(self.eval_dataset)))
        eval_samples = [self.eval_dataset[i] for i in eval_indices]

        # Model is already in eval mode by Trainer, no need to set it
        # Generate decompositions with the model
        evaluation_results = []

        for i, sample in enumerate(eval_samples):
            # Get the original question directly from the dataset
            # eval_raw has: question_id, original_question, decomposed_questions, question_type
            original_question = sample["original_question"]

            # Create the user prompt using the same format as training
            user_prompt = self._create_user_prompt(original_question)

            # Generate decomposition
            messages = [{"role": "user", "content": user_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # Use BF16 autocast to match training environment's dtype handling
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # Parse JSON from generated text
            try:
                # Extract JSON from the output
                json_start = generated_text.find("{")
                json_end = generated_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    generated_decomp = json.loads(json_str)
                    decomposed_questions = generated_decomp.get("questions", [])
                else:
                    decomposed_questions = []
            except:
                decomposed_questions = []

            # Judge the decomposition
            if decomposed_questions:
                result = self.judge.judge_decomposition(original_question, decomposed_questions)
                evaluation_results.append(result)

            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{len(eval_samples)} samples...")

        # Compute aggregate metrics
        if evaluation_results:
            judge_metrics = self.judge.compute_aggregate_metrics(evaluation_results)

            print(f"\n📊 Evaluation Results (Step {current_step}):")
            print(f"  Overall Average: {judge_metrics.get('overall_average', 0):.2f}/5.0")
            print(f"  Atomicity: {judge_metrics.get('atomicity_avg', 0):.2f}/5.0")
            print(f"  Bridge Building: {judge_metrics.get('bridge_building_avg', 0):.2f}/5.0")
            print(f"  Efficiency: {judge_metrics.get('efficiency_avg', 0):.2f}/5.0")
            print(f"  Correctness: {judge_metrics.get('correctness_avg', 0):.2f}/5.0")
            print(f"  Retrieval Flag: {judge_metrics.get('retrieval_flag_avg', 0):.2f}/5.0")
            print(f"  Evaluated: {judge_metrics.get('num_evaluated', 0)} samples")

            # Log to WandB if available
            if state.is_world_process_zero:
                try:
                    import wandb
                    # Don't specify step - let WandB auto-increment
                    wandb.log({
                        "judge/overall_average": judge_metrics.get("overall_average", 0),
                        "judge/atomicity": judge_metrics.get("atomicity_avg", 0),
                        "judge/bridge_building": judge_metrics.get("bridge_building_avg", 0),
                        "judge/efficiency": judge_metrics.get("efficiency_avg", 0),
                        "judge/correctness": judge_metrics.get("correctness_avg", 0),
                        "judge/retrieval_flag": judge_metrics.get("retrieval_flag_avg", 0),
                        "judge/num_evaluated": judge_metrics.get("num_evaluated", 0),
                    })
                except:
                    pass

            # Save detailed results to JSON
            log_file = os.path.join(self.logs_dir, f"judge_eval_step_{current_step}.json")
            with open(log_file, "w") as f:
                json.dump({
                    "step": current_step,
                    "metrics": judge_metrics,
                    "detailed_results": evaluation_results
                }, f, indent=2)

            print(f"  💾 Detailed results saved to: {log_file}")
        else:
            print(f"\n⚠️  No valid evaluations generated at step {current_step}")

        print(f"{'='*60}\n")

    def _create_user_prompt(self, original_question: str) -> str:
        """Create the question decomposition prompt."""
        return f"""Your task is to break down a complex multi-hop question into the most efficient sequence of single-hop, **atomic** questions.

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
{{{{
  "questions": [
    {{{{
      "question": "the decomposed question text",
      "requires_retrieval": "true/false"
    }}}}
  ]
}}}}

Examples:

Input: "What is the birth year of the spouse of the director of Casablanca?"
Output:
{{{{
    "questions": [
        {{{{
            "question": "Who directed Casablanca?",
            "requires_retrieval": "true"
        }}}},
        {{{{
            "question": "Who was <ENTITY_Q1>'s spouse?",
            "requires_retrieval": "true"
        }}}},
        {{{{
            "question": "What is <ENTITY_Q2>'s birth year?",
            "requires_retrieval": "true"
        }}}}
    ]
}}}}

Input: "Which film has the director who is older, Dune or The Dark Knight?"
Output:
{{{{
    "questions": [
        {{{{
            "question": "Who directed Dune?",
            "requires_retrieval": "true"
        }}}},
        {{{{
            "question": "Who directed The Dark Knight?",
            "requires_retrieval": "true"
        }}}},
        {{{{
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": "true"
        }}}},
        {{{{
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": "false"
        }}}}
    ]
}}}}


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
    callback = LLMJudgeEvaluationCallback(None, None, None, None)
    user_prompt = callback._create_user_prompt(original_question)

    # Create the chat messages
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    return {"messages": messages}


def get_question_type(question_id: str) -> str:
    """Extract question type from ID for stratification."""
    return question_id.split("__")[0]


def load_and_split_dataset(dataset_path: str, train_size: int = 600, test_size: int = 200, seed: int = 42):
    """Load dataset and perform stratified train/test split."""
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        decomposition_results = json.load(f)

    print(f"Loaded {len(decomposition_results)} question decompositions")

    # Convert to list format
    decomposition_list = list(decomposition_results.values())

    # Add question types for stratification
    decomposition_list_with_types = [
        {**item, "question_type": get_question_type(item["question_id"])}
        for item in decomposition_list
    ]

    # Count distribution
    type_counts = Counter([item["question_type"] for item in decomposition_list_with_types])
    print("\nQuestion type distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    # Create dataset with question types
    full_dataset = Dataset.from_list(decomposition_list_with_types)

    # Convert question_type to ClassLabel for stratification
    unique_types = sorted(list(set([item["question_type"] for item in decomposition_list_with_types])))
    full_dataset = full_dataset.cast_column(
        "question_type",
        ClassLabel(names=unique_types)
    )

    # Stratified split
    split_dataset = full_dataset.train_test_split(
        test_size=test_size,
        train_size=train_size,
        stratify_by_column="question_type",
        seed=seed
    )

    train_raw = split_dataset["train"]
    eval_raw = split_dataset["test"]

    print(f"\n✓ Split complete!")
    print(f"  Training set: {len(train_raw)} examples")
    print(f"  Evaluation set: {len(eval_raw)} examples")

    # Apply chat formatting
    training_dataset = train_raw.map(
        create_chat_messages,
        remove_columns=train_raw.column_names,
        desc="Creating chat-formatted training data"
    )

    eval_dataset = eval_raw.map(
        create_chat_messages,
        remove_columns=eval_raw.column_names,
        desc="Creating chat-formatted evaluation data"
    )

    print(f"\n✓ Chat-formatted datasets ready:")
    print(f"  Training: {len(training_dataset)} examples")
    print(f"  Evaluation: {len(eval_dataset)} examples")

    return training_dataset, eval_dataset, eval_raw


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 for question decomposition")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to question decomposition JSON dataset")
    parser.add_argument("--chat_template_path", type=str, required=True, help="Path to Jinja chat template file")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./qwen3_0.6b_question_decomp_full_ft", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")

    # Evaluation arguments
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=3, help="Evaluation frequency in steps")
    parser.add_argument("--num_eval_samples", type=int, default=20, help="Number of samples to evaluate with LLM judge")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="LLM judge model")
    parser.add_argument("--logs_dir", type=str, default="./judge_logs", help="Directory for judge evaluation logs")

    # Dataset split arguments
    parser.add_argument("--train_size", type=int, default=600, help="Training set size")
    parser.add_argument("--test_size", type=int, default=200, help="Test set size")

    # WandB arguments
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name (if None, no WandB logging)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")

    args = parser.parse_args()

    # Load chat template
    print(f"Loading chat template from: {args.chat_template_path}")
    with open(args.chat_template_path, 'r') as f:
        chat_template = f.read()

    # Load dataset
    training_dataset, eval_dataset, eval_raw = load_and_split_dataset(
        args.dataset_path,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,
    )

    print(f"Model loaded with full finetuning enabled.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Define formatting function
    def formatting_function(example):
        """Format examples for training."""
        msgs = example["messages"]

        if isinstance(msgs, list) and msgs and isinstance(msgs[0], list):
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
                for m in msgs
            ]
        else:
            texts = [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, chat_template=chat_template
                )
            ]

        return [t for t in texts if isinstance(t, str) and t.strip()]

    # Setup training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb_project else "none",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
    )

    # Initialize WandB if specified
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"qwen3-question-decomp-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )

    # Initialize LLM Judge
    print(f"\nInitializing LLM Judge with model: {args.judge_model}")
    judge = QuestionDecompositionJudge(
        model=args.judge_model,
        temperature=0.0
    )

    # Create evaluation callback
    judge_callback = LLMJudgeEvaluationCallback(
        eval_dataset=eval_raw,
        judge=judge,
        tokenizer=tokenizer,
        chat_template=chat_template,
        num_samples=args.num_eval_samples,
        logs_dir=args.logs_dir
    )

    # Create trainer
    print("\n" + "="*60)
    print("Creating trainer...")
    print("="*60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_function,
        args=training_args,
        callbacks=[judge_callback],
    )

    print(f"\n✓ Trainer configured with:")
    print(f"  - Training examples: {len(training_dataset)}")
    print(f"  - Evaluation examples: {len(eval_dataset)}")
    print(f"  - Evaluation strategy: every {training_args.eval_steps} steps")
    print(f"  - LLM judge samples per evaluation: {judge_callback.num_samples}")

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer_stats = trainer.train()

    # Print training stats
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Training runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training runtime: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")

    # Save model
    print(f"\nSaving model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
