#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for GSW Creation

This script fine-tunes a language model using LoRA (Low-Rank Adaptation) to generate
Generative Semantic Workspace (GSW) structures from text documents.

Usage:
    python gsw_lora_ft.py --model_id Qwen/Qwen3-8B --num_train_epochs 3

    # Specify custom data paths
    python gsw_lora_ft.py --golden_gsw_dir /path/to/golden/gsws --num_train_samples 500

    # Use DoRA instead of LoRA
    python gsw_lora_ft.py --model_id Qwen/Qwen3-8B --use_dora

    # Use specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 python gsw_lora_ft.py --model_id Qwen/Qwen3-8B

    # Push to HuggingFace Hub after training
    python gsw_lora_ft.py \
        --model_id Qwen/Qwen3-8B \
        --push_to_hub \
        --hub_model_id username/qwen3-gsw-creation

    # Test template only without training
    python gsw_lora_ft.py --model_id Qwen/Qwen3-8B --test_template_only
"""

import argparse
import glob as glob_module
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Import GSW models and prompts
import sys
sys.path.append("/home/yigit/codebase/gsw-memory/src")
from gsw_memory.memory.models import GSWStructure
from gsw_memory.prompts.operator_prompts import FactualExtractionPrompts


# =============================================================================
# Data Loading Functions
# =============================================================================

def sort_natural_key(s):
    """Extract numeric part from path string for natural sorting."""
    match = re.search(r'doc_(\d+)', s)
    return int(match.group(1)) if match else s


def load_musique_corpus(test_data_path: str, train_data_path: str, num_samples: Optional[int] = None):
    """
    Load Musique dataset corpus.

    Args:
        test_data_path: Path to test JSON file
        train_data_path: Path to train JSONL file
        num_samples: Number of samples to load from training set (None = all)

    Returns:
        Tuple of (train_corpus_dict, test_corpus_dict)
    """
    print(f"Loading test data from: {test_data_path}")
    test_musique = json.load(open(test_data_path))

    print(f"Loading train data from: {train_data_path}")
    train_musique = [json.loads(line) for line in open(train_data_path)]

    # Build corpus dictionaries
    train_corpus = {}
    for data in train_musique:
        paragraphs = data["paragraphs"]
        for paragraph in paragraphs:
            global_id = f"{data['id']}_{paragraph['idx']}"
            train_corpus[global_id] = {
                "global_id": global_id,
                "title": paragraph["title"],
                "text": paragraph["title"] + "\n" + paragraph["paragraph_text"],
                "id": data["id"],
                "idx": paragraph["idx"]
            }

    test_corpus = {}
    for data in test_musique:
        paragraphs = data["paragraphs"]
        for paragraph in paragraphs:
            global_id = f"{data['id']}_{paragraph['idx']}"
            test_corpus[global_id] = {
                "global_id": global_id,
                "title": paragraph["title"],
                "text": paragraph["title"] + "\n" + paragraph["paragraph_text"],
                "id": data["id"],
                "idx": paragraph["idx"]
            }

    print(f"Total training documents: {len(train_corpus)}")
    print(f"Total test documents: {len(test_corpus)}")

    # Sample if requested
    if num_samples and num_samples < len(train_corpus):
        import random
        sampled_keys = random.sample(list(train_corpus.keys()), num_samples)
        train_corpus = {k: train_corpus[k] for k in sampled_keys}
        print(f"Sampled {num_samples} documents from training set")

    return train_corpus, test_corpus


def load_golden_gsws(golden_gsw_dir: str, max_docs: Optional[int] = None):
    """
    Load golden GSW structures from directory.

    Args:
        golden_gsw_dir: Directory containing doc_* subdirectories with JSON files
        max_docs: Maximum number of document directories to load (None = all)

    Returns:
        Dict mapping document identifiers to GSWStructure objects
    """
    print(f"\nLoading golden GSWs from: {golden_gsw_dir}")

    # Get all doc_* directories
    doc_dirs = sorted(
        glob_module.glob(f"{golden_gsw_dir}/doc_*"),
        key=sort_natural_key
    )

    if max_docs:
        doc_dirs = doc_dirs[:max_docs]
        print(f"Loading first {max_docs} document directories")

    golden_gsws = {}
    total_loaded = 0
    total_errors = 0

    for doc_dir in doc_dirs:
        if not os.path.isdir(doc_dir):
            continue

        # Load all JSON files in this directory
        json_files = sorted(glob_module.glob(os.path.join(doc_dir, "*.json")))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    doc_data = json.load(f)
                    gsw = GSWStructure(**doc_data)

                    # Extract identifier from filename or use doc_dir name + filename
                    # Assuming filename pattern or we use full path as key
                    file_key = os.path.basename(json_file).replace('.json', '')
                    doc_key = os.path.basename(doc_dir)

                    # Create a global_id key (you may need to adjust this based on your data)
                    # For now, using doc_dir/filename pattern
                    gsw_key = f"{doc_key}/{file_key}"

                    golden_gsws[gsw_key] = gsw
                    total_loaded += 1

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                total_errors += 1

    print(f"Loaded {total_loaded} golden GSWs")
    if total_errors > 0:
        print(f"Errors: {total_errors}")

    return golden_gsws


def match_corpus_to_gsws(corpus: Dict, golden_gsws: Dict):
    """
    Match corpus documents to golden GSWs.

    This function attempts to match documents from the corpus to their corresponding
    GSW structures. The matching strategy depends on the key formats.

    Args:
        corpus: Dictionary of corpus documents (key = global_id)
        golden_gsws: Dictionary of GSW structures

    Returns:
        List of dicts with 'document' and 'gsw' keys
    """
    print("\nMatching corpus documents to golden GSWs...")

    matched_pairs = []
    unmatched_docs = []

    # Strategy 1: Try direct key matching
    for global_id, doc in corpus.items():
        if global_id in golden_gsws:
            matched_pairs.append({
                "global_id": global_id,
                "document": doc,
                "gsw": golden_gsws[global_id]
            })
        else:
            unmatched_docs.append(global_id)

    # If direct matching doesn't work well, try alternate strategies
    if len(matched_pairs) < len(corpus) * 0.1:  # Less than 10% matched
        print("Direct matching unsuccessful, trying alternate strategies...")
        matched_pairs = []

        # Strategy 2: Try matching by extracting doc numbers
        gsw_keys_by_doc = {}
        for gsw_key in golden_gsws.keys():
            # Extract doc number from keys like "doc_0/chunk_0"
            match = re.search(r'doc_(\d+)', gsw_key)
            if match:
                doc_num = int(match.group(1))
                if doc_num not in gsw_keys_by_doc:
                    gsw_keys_by_doc[doc_num] = []
                gsw_keys_by_doc[doc_num].append(gsw_key)

        # Try to match corpus docs
        for global_id, doc in corpus.items():
            # Try to extract doc number from global_id
            # Assuming format like "2hop__12345_0" -> extract first number
            match = re.search(r'(\d+)', global_id)
            if match:
                doc_num = int(match.group(1))
                if doc_num in gsw_keys_by_doc and gsw_keys_by_doc[doc_num]:
                    # Use first GSW for this doc number
                    gsw_key = gsw_keys_by_doc[doc_num][0]
                    matched_pairs.append({
                        "global_id": global_id,
                        "document": doc,
                        "gsw": golden_gsws[gsw_key]
                    })

    print(f"Matched {len(matched_pairs)} document-GSW pairs")
    if unmatched_docs and len(matched_pairs) > 0:
        print(f"Unmatched documents: {len(corpus) - len(matched_pairs)}")

    return matched_pairs


# =============================================================================
# Training Dataset Creation
# =============================================================================

def create_chat_messages(example):
    """
    Convert a single example into chat format for training.

    Args:
        example: Dict with 'document' and 'gsw' keys

    Returns:
        Dict with 'messages' key containing the chat-formatted data
    """
    document = example['document']
    gsw = example['gsw']

    # Serialize GSW to JSON format
    if isinstance(gsw, GSWStructure):
        assistant_response = gsw.model_dump_json(indent=4)
    else:
        assistant_response = json.dumps(gsw, indent=4, ensure_ascii=False)

    # Create the system and user prompts using FactualExtractionPrompts
    system_prompt = FactualExtractionPrompts.SYSTEM_PROMPT
    user_prompt = FactualExtractionPrompts.USER_PROMPT_TEMPLATE.format(
        input_text=document['text'],
        background_context=""  # Empty context for now
    )

    # Create chat messages
    # Model will automatically add <think> tags when generating
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    return {"messages": messages}


def create_training_dataset(matched_pairs):
    """
    Create HuggingFace Dataset from matched document-GSW pairs.

    Args:
        matched_pairs: List of dicts with 'document' and 'gsw' keys

    Returns:
        Dataset with 'messages' column
    """
    print("\nCreating training dataset...")
    print(f"Total examples: {len(matched_pairs)}")

    # Create HuggingFace Dataset
    raw_dataset = Dataset.from_list(matched_pairs)

    # Apply chat formatting
    training_dataset = raw_dataset.map(
        create_chat_messages,
        remove_columns=raw_dataset.column_names,
        desc="Creating chat-formatted training data"
    )

    print(f"Training dataset created with {len(training_dataset)} examples")
    print(f"Column names: {training_dataset.column_names}")

    return training_dataset


def test_chat_template(tokenizer, training_dataset):
    """
    Test that the chat template works correctly before training.

    Args:
        tokenizer: Tokenizer instance
        training_dataset: Dataset with 'messages' column
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

        # Apply chat template
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

        # Check if <think> tags are present (should be for thinking models)
        if '<think>' in formatted or 'add_generation_prompt' in str(tokenizer.chat_template):
            print("\n✓ Thinking mode detected in template")
        else:
            print("\n⚠ Note: Thinking tags not detected, model may add them during generation")

    except Exception as e:
        print(f"\n✗ ERROR during formatting: {e}")
        print("  You may need to adjust the formatting function or use a different model.")

    print("\n" + "="*60)
    print("Test complete! Review the output above before training.")


# =============================================================================
# LoRA Training Functions
# =============================================================================

def train(model_id, tokenizer, dataset, training_args, use_dora=False):
    """
    Train a model with LoRA/DoRA on local GPU.

    Args:
        model_id: HuggingFace model identifier
        tokenizer: Tokenizer instance
        dataset: Training dataset with 'messages' column
        training_args: TrainingArguments instance
        use_dora: Whether to use DoRA instead of LoRA

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

    adapter_type = "DoRA" if use_dora else "LoRA"
    print(f"Configuring {adapter_type}...")

    # LoRA/DoRA configuration
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,  # Enable DoRA if requested
    )

    def formatting_function(example):
        """Format a single example using the tokenizer's chat template."""
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
    )

    # Start training
    print("\n" + "="*60)
    print(f"Starting {adapter_type} training...")
    print("="*60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    return trainer


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="LoRA/DoRA Fine-Tuning for GSW Creation")

    # Add arguments
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B",
                        help="Model ID from HuggingFace hub")
    parser.add_argument("--test_data_path", type=str,
                        default="/home/yigit/codebase/gsw-memory/playground_data/musique.json",
                        help="Path to test data JSON")
    parser.add_argument("--train_data_path", type=str,
                        default="/home/yigit/codebase/gsw-memory/playground_data/musique_full_v1.0_train.jsonl",
                        help="Path to train data JSONL")
    parser.add_argument("--golden_gsw_dir", type=str,
                        default="/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini",
                        help="Directory containing golden GSW structures")
    parser.add_argument("--output_dir", type=str, default="./gsw_creation_lora",
                        help="Output directory for trained model")
    parser.add_argument("--num_train_samples", type=int, default=None,
                        help="Number of training samples (None = all)")
    parser.add_argument("--max_golden_docs", type=int, default=100,
                        help="Maximum golden GSW documents to load")
    parser.add_argument("--num_train_epochs", type=int, default=3,
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
    parser.add_argument("--use_dora", action="store_true",
                        help="Use DoRA instead of LoRA")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push the trained model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID on HuggingFace Hub (e.g., 'username/model-name')")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="HuggingFace API token (optional, can use HF_TOKEN env var)")
    parser.add_argument("--hub_private_repo", action="store_true",
                        help="Create a private repository on HuggingFace Hub")
    parser.add_argument("--skip_data_load", action="store_true",
                        help="Skip data loading and use cached dataset")
    parser.add_argument("--cached_dataset_path", type=str, default="./gsw_training_dataset.json",
                        help="Path to cached dataset file")

    args = parser.parse_args()

    # Validate arguments
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is enabled")

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

    # Step 1: Load data or use cached
    if not args.skip_data_load:
        print("="*60)
        print("Step 1: Loading Musique Corpus")
        print("="*60)
        train_corpus, test_corpus = load_musique_corpus(
            args.test_data_path,
            args.train_data_path,
            args.num_train_samples
        )

        # Step 2: Load golden GSWs
        print("\n" + "="*60)
        print("Step 2: Loading Golden GSWs")
        print("="*60)
        golden_gsws = load_golden_gsws(
            args.golden_gsw_dir,
            args.max_golden_docs
        )

        # Step 3: Match corpus to GSWs
        print("\n" + "="*60)
        print("Step 3: Matching Documents to GSWs")
        print("="*60)
        matched_pairs = match_corpus_to_gsws(train_corpus, golden_gsws)

        if len(matched_pairs) == 0:
            print("\n✗ ERROR: No matched document-GSW pairs found!")
            print("  Please check that your corpus and golden GSW keys match.")
            print(f"  Sample corpus key: {list(train_corpus.keys())[0] if train_corpus else 'N/A'}")
            print(f"  Sample GSW key: {list(golden_gsws.keys())[0] if golden_gsws else 'N/A'}")
            return

        # Save matched pairs if requested
        if args.cached_dataset_path:
            print(f"\nSaving matched pairs to: {args.cached_dataset_path}")
            with open(args.cached_dataset_path, 'w') as f:
                # Convert GSWStructure to dict for JSON serialization
                serializable_pairs = []
                for pair in matched_pairs:
                    gsw_dict = pair['gsw'].model_dump() if isinstance(pair['gsw'], GSWStructure) else pair['gsw']
                    serializable_pairs.append({
                        'global_id': pair['global_id'],
                        'document': pair['document'],
                        'gsw': gsw_dict
                    })
                json.dump(serializable_pairs, f, indent=2)

    else:
        print("="*60)
        print("Loading cached dataset")
        print("="*60)
        print(f"Loading from: {args.cached_dataset_path}")
        with open(args.cached_dataset_path, 'r') as f:
            data = json.load(f)

        # Check if this is training data with thinking traces or matched pairs
        if isinstance(data, dict) and "training_examples" in data:
            # This is LoRA training data with thinking traces (from gsw_playground.py)
            print("  Detected: LoRA training data with thinking traces")
            if "metadata" in data:
                print(f"  Metadata: {json.dumps(data['metadata'], indent=2)}")

            # Training examples already have "messages" field, so create dataset directly
            training_dataset = Dataset.from_list(data["training_examples"])
            print(f"  Loaded {len(training_dataset)} training examples with thinking traces")

            # Skip to tokenizer configuration (no need to create training dataset)
            matched_pairs = None

        elif isinstance(data, list):
            # This is matched pairs format (old format)
            print("  Detected: Matched pairs format (converting to training dataset)")
            matched_pairs = data
            # Convert dict back to GSWStructure
            for pair in matched_pairs:
                pair['gsw'] = GSWStructure(**pair['gsw'])
            print(f"  Loaded {len(matched_pairs)} matched pairs")

        else:
            raise ValueError(f"Unknown cached dataset format: {args.cached_dataset_path}")

    # Step 4: Create training dataset (if needed)
    if matched_pairs is not None:
        print("\n" + "="*60)
        print("Step 4: Creating Training Dataset")
        print("="*60)
        training_dataset = create_training_dataset(matched_pairs)
    else:
        print("\n" + "="*60)
        print("Step 4: Training Dataset Already Loaded")
        print("="*60)
        print(f"  Dataset size: {len(training_dataset)}")
        print(f"  Columns: {training_dataset.column_names}")

    # Step 5: Configure tokenizer
    print("\n" + "="*60)
    print("Step 5: Configuring Tokenizer")
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

    # Step 6: Test chat template
    print("\n" + "="*60)
    print("Step 6: Testing Chat Template")
    print("="*60)
    test_chat_template(tokenizer, training_dataset)

    if args.test_template_only:
        print("\nTemplate test complete. Exiting without training.")
        return

    # Step 7: Configure training
    print("\n" + "="*60)
    print("Step 7: Configuring Training Arguments")
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
        report_to="none",  # Change to "wandb" if you want W&B logging
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_token=args.hub_token if args.push_to_hub else None,
        hub_private_repo=args.hub_private_repo,
    )

    adapter_type = "DoRA" if args.use_dora else "LoRA"
    print(f"Training configuration ({adapter_type}):")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    if args.push_to_hub:
        print(f"\nHuggingFace Hub configuration:")
        print(f"  Push to hub: Enabled")
        print(f"  Hub model ID: {args.hub_model_id}")
        print(f"  Repository type: {'Private' if args.hub_private_repo else 'Public'}")

    # Step 8: Train
    print("\n" + "="*60)
    print(f"Step 8: Training Model with {adapter_type}")
    print("="*60)
    trainer = train(args.model_id, tokenizer, training_dataset, training_args, use_dora=args.use_dora)

    # Step 9: Save final model
    final_output_path = os.path.join(args.output_dir, "final")
    print("\n" + "="*60)
    print("Step 9: Saving Final Model")
    print("="*60)
    print(f"Saving to: {final_output_path}")
    trainer.save_model(final_output_path)

    # Step 10: Push to HuggingFace Hub (if requested)
    if args.push_to_hub:
        print("\n" + "="*60)
        print("Step 10: Pushing Model to HuggingFace Hub")
        print("="*60)
        print(f"Target repository: {args.hub_model_id}")
        print(f"Repository type: {'Private' if args.hub_private_repo else 'Public'}")

        try:
            print("\nPushing model... This may take a few minutes.")
            trainer.push_to_hub(
                commit_message=f"Training complete - {adapter_type} fine-tuned model for GSW creation",
                blocking=True,
            )

            print("\n✓ Model successfully pushed to HuggingFace Hub!")
            print(f"  View at: https://huggingface.co/{args.hub_model_id}")

        except Exception as e:
            print(f"\n✗ Error pushing to HuggingFace Hub: {e}")
            print("\nTroubleshooting tips:")
            print("  1. Make sure you're logged in: huggingface-cli login")
            print("  2. Check your HuggingFace token has write permissions")
            print("  3. Verify the repository name is valid and available")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {final_output_path}")
    print(f"Checkpoints saved to: {args.output_dir}")
    if args.push_to_hub:
        print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
