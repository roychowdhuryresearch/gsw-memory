"""
RLVR training script: LoRA + GRPO on Qwen3-30B-A3B-Thinking.

Trains the sleep-time GSW exploration agent using Reinforcement Learning
with Verifiable Rewards (RLVR). The agent learns to explore GSW structures
and create high-quality multi-hop bridge QA pairs.

Architecture:
    - Policy: Qwen3-30B-A3B-Thinking + LoRA (rank 256)
    - Algorithm: GRPO (Group Relative Policy Optimization)
    - Framework: veRL with FSDP2 backend
    - Rollout: vLLM (4-way tensor parallel)
    - Reward: token F1 + bridge quality bonuses (see reward.py)

Token masking (Search-R1 style):
    Tool response tokens are masked from the policy loss.
    Only the agent's own reasoning + tool call tokens are trained on.

Usage:
    # Single node, 8 GPUs
    torchrun --nproc_per_node=8 \\
        src/gsw_memory/sleep_time/train.py \\
        conf/rl/grpo_lora_qwen3.yaml

    # Or with veRL launcher:
    python -m verl.trainer.main_ppo \\
        algorithm=grpo \\
        --config-path conf/rl \\
        --config-name grpo_lora_qwen3
"""

import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# ---------------------------------------------------------------------------
# Lazy imports — veRL and vLLM are optional at import time so the file can
# be read without them installed (e.g., during data_prep or reward testing).
# ---------------------------------------------------------------------------

def _require_verl():
    try:
        import verl  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "veRL is required for training. Install with:\n"
            "  pip install verl\n"
            "See: https://github.com/verl-project/verl"
        ) from e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Parsed training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B-Thinking"
    enable_thinking: bool = True

    # LoRA
    lora_rank: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"

    # GRPO
    group_size: int = 8
    kl_coef: float = 0.001
    clip_ratio: float = 0.2

    # Training
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    total_steps: int = 2000
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    save_every: int = 200
    eval_every: int = 100

    # Environment
    max_turns: int = 30
    reward_f1_threshold: float = 0.5
    reward_main_scale: float = 10.0

    # Data
    train_index: str = "data/rl_training/index.json"
    min_hops: int = 2
    shuffle: bool = True

    # Rollout
    vllm_tensor_parallel: int = 4
    vllm_gpu_memory_utilization: float = 0.85
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.9
    max_model_len: int = 32768

    # Logging
    output_dir: str = "logs/rl_training"
    wandb_project: str = "gsw-rlvr"
    log_every: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        # Flatten nested YAML into flat config fields
        model = raw.get("model", {})
        cfg.model_name = model.get("name", cfg.model_name)
        cfg.enable_thinking = model.get("enable_thinking", cfg.enable_thinking)
        cfg.max_model_len = model.get("max_position_embeddings", cfg.max_model_len)

        lora = raw.get("lora", {})
        cfg.lora_rank = lora.get("rank", cfg.lora_rank)
        cfg.lora_alpha = lora.get("alpha", cfg.lora_alpha)
        cfg.lora_dropout = lora.get("dropout", cfg.lora_dropout)
        cfg.target_modules = lora.get("target_modules", cfg.target_modules)

        grpo = raw.get("grpo", {})
        cfg.group_size = grpo.get("group_size", cfg.group_size)
        cfg.kl_coef = grpo.get("kl_coef", cfg.kl_coef)
        cfg.clip_ratio = grpo.get("clip_ratio", cfg.clip_ratio)

        tr = raw.get("training", {})
        cfg.learning_rate = tr.get("learning_rate", cfg.learning_rate)
        cfg.warmup_steps = tr.get("warmup_steps", cfg.warmup_steps)
        cfg.total_steps = tr.get("total_steps", cfg.total_steps)
        cfg.batch_size = tr.get("batch_size", cfg.batch_size)
        cfg.gradient_accumulation = tr.get("gradient_accumulation", cfg.gradient_accumulation)
        cfg.max_grad_norm = tr.get("max_grad_norm", cfg.max_grad_norm)
        cfg.save_every = tr.get("save_every", cfg.save_every)
        cfg.eval_every = tr.get("eval_every", cfg.eval_every)

        env = raw.get("environment", {})
        cfg.max_turns = env.get("max_turns", cfg.max_turns)
        cfg.reward_f1_threshold = env.get("reward_f1_threshold", cfg.reward_f1_threshold)
        cfg.reward_main_scale = env.get("reward_main_scale", cfg.reward_main_scale)

        data = raw.get("data", {})
        cfg.train_index = data.get("train_index", cfg.train_index)
        cfg.min_hops = data.get("min_hops", cfg.min_hops)
        cfg.shuffle = data.get("shuffle", cfg.shuffle)

        rollout = raw.get("rollout", {})
        cfg.vllm_tensor_parallel = rollout.get("tensor_parallel_size", cfg.vllm_tensor_parallel)
        cfg.vllm_gpu_memory_utilization = rollout.get("gpu_memory_utilization", cfg.vllm_gpu_memory_utilization)
        cfg.rollout_temperature = rollout.get("temperature", cfg.rollout_temperature)
        cfg.rollout_top_p = rollout.get("top_p", cfg.rollout_top_p)

        log = raw.get("logging", {})
        cfg.output_dir = log.get("output_dir", cfg.output_dir)
        cfg.wandb_project = log.get("wandb_project", cfg.wandb_project)
        cfg.log_every = log.get("log_every", cfg.log_every)

        return cfg


# ---------------------------------------------------------------------------
# Tool response masking (Search-R1 style)
# ---------------------------------------------------------------------------

# Tool response tokens are enclosed in <tool_response>...</tool_response>
# We zero out their loss contribution so the model is only trained on its
# own reasoning and tool call tokens.
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def mask_tool_response_tokens(
    input_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Build an attention mask that zeroes out tool response tokens.

    Args:
        input_ids: (seq_len,) token id tensor for one sequence.
        tokenizer: HuggingFace tokenizer.

    Returns:
        loss_mask: (seq_len,) float tensor, 0 for tool response tokens, 1 elsewhere.
    """
    text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    loss_mask = torch.ones(len(input_ids), dtype=torch.float)

    # Find all <tool_response>...</tool_response> spans by character position
    pattern = re.compile(
        re.escape(TOOL_RESPONSE_OPEN) + r".*?" + re.escape(TOOL_RESPONSE_CLOSE),
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        # Convert character span → token span (approximate via re-tokenization)
        prefix = text[: match.start()]
        span_text = match.group(0)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        span_ids = tokenizer.encode(span_text, add_special_tokens=False)
        start_tok = len(prefix_ids)
        end_tok = start_tok + len(span_ids)
        loss_mask[start_tok:end_tok] = 0.0

    return loss_mask


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def build_prompt(question: str, system_prompt: str) -> str:
    """Format the initial prompt for one training episode."""
    from .prompts import SLEEP_TIME_SYSTEM_PROMPT
    sys_p = system_prompt or SLEEP_TIME_SYSTEM_PROMPT
    return (
        f"<|im_start|>system\n{sys_p}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_episode(
    entry: Dict[str, Any],
    policy_fn,
    entity_searcher_cls,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    Run one complete episode for a training example.

    Args:
        entry:             One entry from training index.
        policy_fn:         Callable(prompt: str) → (response: str, token_ids: list).
        entity_searcher_cls: EntitySearcher class.
        cfg:               Training config.

    Returns:
        Dict with: prompt, response, token_ids, reward, bridges, trajectory.
    """
    from .environment import GSWEnvironment
    from .prompts import SLEEP_TIME_SYSTEM_PROMPT

    entity_searcher = entity_searcher_cls(
        path_to_gsw_files=entry["gsw_dir"],
        verbose=False,
    )
    env = GSWEnvironment(
        entity_searcher=entity_searcher,
        question=entry["question"],
        gold_answer=entry["answer"],
        gold_decomposition=entry.get("decomposition", []),
        max_turns=cfg.max_turns,
    )

    obs = env.reset()
    full_prompt = build_prompt(entry["question"], SLEEP_TIME_SYSTEM_PROMPT)

    # The policy generates the full tool-calling trajectory in one pass
    # (agentic / multi-turn generation via vLLM with tool call parsing)
    response, token_ids = policy_fn(full_prompt)

    # Parse tool calls from the response and step the environment
    # to materialize tool results and build the actual bridge set.
    _replay_tool_calls(response, env)

    reward = env.get_reward()

    return {
        "id": entry["id"],
        "prompt": full_prompt,
        "response": response,
        "token_ids": token_ids,
        "reward": reward,
        "bridges": env.get_bridges(),
        "trajectory": env.get_trajectory(),
        "num_turns": env.turn,
    }


def _replay_tool_calls(response: str, env) -> None:
    """
    Parse tool calls from the model's response text and step the environment.
    This materializes the bridge set so rewards can be computed.

    Tool calls are expected in the format:
        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )
    for match in pattern.finditer(response):
        if env.done:
            break
        try:
            call = json.loads(match.group(1))
            tool_name = call.get("name", "")
            tool_args = call.get("arguments", {})
            env.step(tool_name, tool_args)
        except (json.JSONDecodeError, Exception):
            continue


# ---------------------------------------------------------------------------
# GRPO advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO: normalize rewards within the group to compute advantages.
    advantages = (r - mean(r)) / (std(r) + 1e-8)
    """
    if len(rewards) == 0:
        return []
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = (var_r + 1e-8) ** 0.5
    return [(r - mean_r) / std_r for r in rewards]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    """Main training loop."""
    _require_verl()

    import verl
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Lazy import EntitySearcher
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "playground"))
    from simple_entity_search import EntitySearcher  # type: ignore

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training index ---
    with open(cfg.train_index) as f:
        train_index: List[Dict[str, Any]] = json.load(f)

    train_index = [e for e in train_index if e.get("num_hops", 2) >= cfg.min_hops]
    print(f"Training examples: {len(train_index)}")

    if cfg.shuffle:
        random.shuffle(train_index)

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    # --- Load model + LoRA ---
    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        # Exclude MoE router from LoRA
        modules_to_save=[],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Optimizer ---
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.total_steps,
    )

    # --- vLLM policy function ---
    # TODO: Replace with actual vLLM server call.
    # The policy_fn below is a stub — wire it to the vLLM server running
    # the current LoRA-merged model weights.
    def policy_fn(prompt: str) -> Tuple[str, List[int]]:
        """
        Call the vLLM server to generate a response.

        In production:
            1. Merge current LoRA weights into base model
            2. Serve merged model via vLLM
            3. Send prompt to vLLM OpenAI-compatible endpoint
            4. Return (response_text, token_ids)

        TODO: Implement vLLM server synchronization with veRL's
              async weight update mechanism.
        """
        raise NotImplementedError(
            "policy_fn must be connected to the vLLM server. "
            "See veRL documentation for actor_rollout_ref worker setup."
        )

    # --- Training loop ---
    step = 0
    example_idx = 0
    all_rewards = []

    print(f"Starting GRPO training for {cfg.total_steps} steps...")

    while step < cfg.total_steps:
        # Collect a batch of episodes
        batch_episodes = []
        for _ in range(cfg.batch_size * cfg.group_size):
            entry = train_index[example_idx % len(train_index)]
            example_idx += 1

            try:
                episode = run_episode(entry, policy_fn, EntitySearcher, cfg)
                batch_episodes.append(episode)
            except NotImplementedError:
                raise  # re-raise: policy_fn not implemented
            except Exception as e:
                print(f"  Episode error ({entry['id']}): {e}")
                continue

        if not batch_episodes:
            continue

        # Group episodes by prompt (cfg.group_size per prompt for GRPO)
        # Here we treat each consecutive group_size episodes as one GRPO group
        rewards = [ep["reward"] for ep in batch_episodes]
        advantages = compute_advantages(rewards)
        all_rewards.extend(rewards)

        # --- Policy gradient update ---
        # TODO: Replace with veRL GRPO trainer call:
        #   trainer.update(batch_episodes, advantages, cfg)
        #
        # The update should:
        #   1. Tokenize (prompt + response) pairs
        #   2. Apply mask_tool_response_tokens to zero out tool outputs
        #   3. Compute policy gradient loss with GRPO clipping
        #   4. Add KL penalty vs reference model
        #   5. Backward + clip gradients + optimizer step
        #
        # Reference implementation:
        #   verl/trainer/ppo/core_algos.py :: compute_grpo_loss()

        optimizer.zero_grad()
        # [veRL GRPO loss computation goes here]
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

        step += 1

        # --- Logging ---
        if step % cfg.log_every == 0:
            recent_rewards = all_rewards[-100:]
            mean_r = sum(recent_rewards) / len(recent_rewards)
            nonzero = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            print(
                f"Step {step}/{cfg.total_steps} | "
                f"mean_reward={mean_r:.3f} | "
                f"nonzero_frac={nonzero:.2%}"
            )

        # --- Checkpoint ---
        if step % cfg.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint_step_{step}"
            ckpt_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"Saved checkpoint: {ckpt_dir}")

    print("Training complete.")
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final LoRA adapter saved to: {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLVR training: LoRA + GRPO + Qwen3")
    parser.add_argument(
        "config",
        nargs="?",
        default="conf/rl/grpo_lora_qwen3.yaml",
        help="Path to YAML config (default: conf/rl/grpo_lora_qwen3.yaml)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    print(f"Config loaded from: {args.config}")
    print(f"  Model:      {cfg.model_name}")
    print(f"  LoRA rank:  {cfg.lora_rank}  alpha: {cfg.lora_alpha}")
    print(f"  GRPO group: {cfg.group_size}  KL: {cfg.kl_coef}")
    print(f"  Steps:      {cfg.total_steps}  LR: {cfg.learning_rate}")
    print(f"  Data:       {cfg.train_index}")

    train(cfg)
