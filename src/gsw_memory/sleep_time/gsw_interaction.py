"""
veRL Interaction for multi-turn GSW exploration.

Implements verl.interactions.base.BaseInteraction so that veRL's tool_agent_loop
can execute real GSW tool calls during rollout and assign a reward at episode end.

How it fits into veRL's agent loop
-------------------------------------
When interaction_config_path is set in rollout.multi_turn, after each assistant
turn the agent loop transitions to INTERACTING state and calls:

    generate_response(instance_id, messages, **interaction_kwargs)

We parse the last assistant message for a tool call, run it against the live
GSWEnvironment, and return the tool result as a user message so the model can
continue reasoning. When the agent calls mark_entity_explored (or max_turns is
hit), we return should_terminate=True and the final bridge-F1 reward.

Parquet / extra_info wiring
-----------------------------
make_parquet.py must store interaction_kwargs under extra_info:

    extra_info = {
        "interaction_kwargs": {
            "name":         "gsw",          # must match interaction name in YAML
            "ground_truth": "...",          # gold answer
            "gsw_dirs":     [...],          # list of GSW directory paths
            "answer_aliases": [...],
            "decomposition":  [...],
        },
        ...
    }

veRL's tool_agent_loop reads extra_info["interaction_kwargs"] and passes it
verbatim to start_interaction / generate_response.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

logger = logging.getLogger(__name__)

try:
    from verl.interactions.base import BaseInteraction
except ImportError:
    # Fallback stub for testing outside veRL
    class BaseInteraction:  # type: ignore
        def __init__(self, config: dict[str, Any]):
            self.config = config
            self.name: str = config.get("name", "gsw")


# Tool call regex — matches <tool_call>{ ... }</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_last_tool_call(messages: list[dict[str, Any]]) -> Optional[tuple[str, dict]]:
    """
    Walk messages in reverse to find the last assistant message containing
    a <tool_call>...</tool_call> block.

    Returns (tool_name, tool_args) or None if not found.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        match = _TOOL_CALL_RE.search(content)
        if match:
            try:
                call = json.loads(match.group(1))
                name = call.get("name", "")
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return name, args
            except (json.JSONDecodeError, KeyError):
                continue
    return None


class GSWInteraction(BaseInteraction):
    """
    Multi-turn interaction that executes real GSW tool calls during veRL rollout.

    Lifecycle per episode:
        start_interaction  → create GSWEnvironment, call env.reset()
        generate_response  → parse tool call, step env, return result
        ...repeat...
        generate_response  → env.done → return should_terminate=True + reward
        calculate_score    → final bridge-F1 reward (also available from last generate_response)
        finalize_interaction → cleanup
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._episodes: dict[str, dict] = {}  # instance_id → episode state

    # ------------------------------------------------------------------
    # BaseInteraction interface
    # ------------------------------------------------------------------

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        gsw_dirs: Optional[list[str]] = None,
        answer_aliases: Optional[list[str]] = None,
        decomposition: Optional[list[dict]] = None,
        **kwargs,
    ) -> str:
        """
        Create a new GSWEnvironment for this episode.

        Args are passed from extra_info["interaction_kwargs"] in the parquet.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        gsw_dirs = gsw_dirs or []
        answer_aliases = answer_aliases or []
        decomposition = decomposition or []
        gold_answer = ground_truth or ""

        try:
            env = self._build_env(
                gsw_dirs=gsw_dirs,
                gold_answer=gold_answer,
                gold_decomposition=decomposition,
                max_turns=self.config.get("max_turns", 30),
            )
            initial_obs = env.reset()
        except Exception as exc:
            logger.warning(f"[GSWInteraction] Failed to build env for {instance_id}: {exc}")
            env = None
            initial_obs = json.dumps({"error": str(exc)})

        self._episodes[instance_id] = {
            "env": env,
            "gold_answer": gold_answer,
            "answer_aliases": answer_aliases,
            "decomposition": decomposition,
            "last_reward": 0.0,
            "initial_obs": initial_obs,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Parse the last tool call from messages, execute it, return result.

        Returns:
            (should_terminate, response_str, turn_reward, metadata)
        """
        ep = self._episodes.get(instance_id)
        if ep is None:
            return True, json.dumps({"error": "unknown instance_id"}), 0.0, {}

        env = ep["env"]
        if env is None:
            return True, json.dumps({"error": "env failed to initialise"}), 0.0, {}

        if env.done:
            reward = await self.calculate_score(instance_id)
            ep["last_reward"] = reward
            return True, json.dumps({"type": "episode_done", "reward": reward}), reward, {}

        # Parse tool call from last assistant message
        parsed = _parse_last_tool_call(messages)
        if parsed is None:
            # No tool call found — agent produced freeform text; end episode
            reward = await self.calculate_score(instance_id)
            ep["last_reward"] = reward
            return True, json.dumps({
                "type": "no_tool_call",
                "message": "No <tool_call> found in response. Episode ended.",
                "reward": reward,
            }), reward, {}

        tool_name, tool_args = parsed

        # Execute tool in environment
        try:
            obs, done = env.step(tool_name, tool_args)
        except Exception as exc:
            obs = json.dumps({"error": f"Tool execution failed: {exc}"})
            done = False

        if done:
            reward = await self.calculate_score(instance_id)
            ep["last_reward"] = reward
            # Append reward info to the observation
            obs_data = _try_parse_json(obs)
            if isinstance(obs_data, dict):
                obs_data["episode_reward"] = reward
                obs = json.dumps(obs_data)
            return True, obs, reward, {}

        # Episode continues — turn reward is 0 (reward only at end)
        return False, obs, 0.0, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Compute final bridge-F1 reward over all bridges created in the episode."""
        ep = self._episodes.get(instance_id)
        if ep is None:
            return 0.0
        env = ep.get("env")
        if env is None:
            return 0.0

        from .reward import compute_reward

        bridges = env.get_bridges()
        return compute_reward(
            bridges=bridges,
            gold_answer=ep["gold_answer"],
            gold_decomposition=ep["decomposition"],
            gold_aliases=ep["answer_aliases"],
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Release episode state."""
        self._episodes.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(
        self,
        gsw_dirs: list[str],
        gold_answer: str,
        gold_decomposition: list[dict],
        max_turns: int,
    ):
        """Build a GSWEnvironment from a list of GSW directory paths."""
        from .environment import GSWEnvironment

        from gsw_memory.sleep_time.entity_search import EntitySearcher

        # Use the first gsw_dir as the primary search path.
        # GSWTools / EntitySearcher can be pointed at a parent directory
        # that contains multiple doc_N sub-directories.
        if not gsw_dirs:
            raise ValueError("gsw_dirs must be non-empty")

        # Find the common parent — EntitySearcher usually takes the network root
        gsw_root = str(Path(gsw_dirs[0]).parent) if len(gsw_dirs) == 1 else str(
            Path(gsw_dirs[0]).parent
        )

        entity_searcher = EntitySearcher(
            path_to_gsw_files=gsw_root,
            verbose=False,
        )

        # Extract the question from the initial observation (not available here)
        # so we use an empty string; it's only used for env logging anyway.
        return GSWEnvironment(
            entity_searcher=entity_searcher,
            question="",
            gold_answer=gold_answer,
            gold_decomposition=gold_decomposition,
            max_turns=max_turns,
        )


def _try_parse_json(s: str) -> Any:
    """Try to parse a JSON string; return original string on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s
