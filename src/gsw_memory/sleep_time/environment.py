"""
GSW Exploration Environment for RLVR training.

Wraps the existing 16 GSWTools methods in a gym-like interface.
An episode corresponds to exploring one target entity from the GSW corpus.
The episode ends when the agent calls mark_entity_explored or hits max_turns.

Usage (standalone test):
    from gsw_memory.sleep_time.environment import GSWEnvironment
    env = GSWEnvironment(gsw_dir, question, gold_answer, gold_decomposition)
    obs = env.reset()
    while not env.done:
        obs, done = env.step("reconcile_entity_across_docs",
                             {"entity_name": "Houston Baptist University"})
    reward = env.get_reward()
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from .tools import GSWTools


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GSWEnvironment:
    """
    Gym-like environment for agentic GSW exploration.

    State:
        A string representation of the current observation history
        (system prompt + tool call/result pairs so far).

    Actions:
        Any of the 16 GSWTools methods, identified by name.
        Calling mark_entity_explored ends the episode.

    Reward:
        Computed lazily by reward.py over bridges created during the episode.
        Call get_reward() after the episode is done.
    """

    # Names of the 16 available tool methods
    TOOL_NAMES = [
        "browse_entities",
        "get_entity_documents",
        "get_document_entities",
        "find_entity_neighbors",
        "get_entity_context",
        "reconcile_entity_across_docs",
        "search_qa_pairs",
        "trace_relationship_chain",
        "create_bridge_qa",
        "validate_bridge",
        "get_bridge_statistics",
        "suggest_next_entity",
        "mark_entity_explored",
        "plan_entity_exploration",
        "mark_relationship_explored",
        "get_exploration_status",
        # note: get_exploration_progress is available but not listed as an
        # action since it gives global stats rather than entity-specific info
    ]

    def __init__(
        self,
        entity_searcher,
        question: str,
        gold_answer: str,
        gold_decomposition: List[Dict[str, Any]],
        max_turns: int = 30,
    ):
        """
        Args:
            entity_searcher: Initialized EntitySearcher with GSWs loaded.
            question:         The MuSiQue multi-hop question for this episode.
            gold_answer:      Gold final answer string.
            gold_decomposition: MuSiQue decomposition steps (list of sub-QA dicts).
            max_turns:        Max tool calls before episode is force-ended.
        """
        self.entity_searcher = entity_searcher
        self.question = question
        self.gold_answer = gold_answer
        self.gold_decomposition = gold_decomposition
        self.max_turns = max_turns

        self.tools: Optional[GSWTools] = None
        self.turn: int = 0
        self.done: bool = False
        self._obs_history: List[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """
        Reset the environment for a new episode.
        Returns the initial observation string (system context).
        """
        self.tools = GSWTools(self.entity_searcher)
        self.turn = 0
        self.done = False
        self._obs_history = []

        initial_obs = self._format_initial_obs()
        self._obs_history.append(initial_obs)
        return initial_obs

    def step(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """
        Execute one tool call.

        Args:
            tool_name: Name of one of the 16 TOOL_NAMES.
            tool_args: Keyword arguments for the tool.

        Returns:
            (observation_str, done)
            observation_str: JSON-serialized tool result.
            done: True if the episode has ended.
        """
        if self.done:
            return self._obs("Episode already done.", terminal=True), True

        if tool_name not in self.TOOL_NAMES:
            obs = self._obs(
                f"ERROR: unknown tool '{tool_name}'. "
                f"Available: {self.TOOL_NAMES}",
                terminal=False,
            )
            self.turn += 1
            self._check_turn_limit()
            self._obs_history.append(obs)
            return obs, self.done

        self.turn += 1

        # Dispatch
        try:
            result = self._dispatch(tool_name, tool_args)
        except Exception as e:
            result = {"error": str(e)}

        obs = self._obs(result, terminal=False)
        self._obs_history.append(obs)

        # Episode ends when the agent marks an entity as explored
        if tool_name == "mark_entity_explored":
            self.done = True

        self._check_turn_limit()
        return obs, self.done

    def get_reward(self) -> float:
        """
        Compute reward after the episode.
        Delegates to reward.compute_reward.
        """
        from .reward import compute_reward

        bridges = self.tools.bridges_created if self.tools else []
        return compute_reward(
            bridges=bridges,
            gold_answer=self.gold_answer,
            gold_decomposition=self.gold_decomposition,
        )

    def get_trajectory(self) -> List[str]:
        """Return full observation history for trajectory logging."""
        return list(self._obs_history)

    def get_bridges(self) -> List[Dict[str, Any]]:
        """Return all bridge QA pairs created during this episode."""
        return self.tools.bridges_created if self.tools else []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Call the named tool method on self.tools."""
        fn = getattr(self.tools, tool_name)
        return fn(**tool_args)

    def _obs(self, result: Any, terminal: bool) -> str:
        """Serialize a tool result into an observation string."""
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            return str(result)

    def _format_initial_obs(self) -> str:
        """Build the initial observation that seeds the episode."""
        return json.dumps(
            {
                "type": "episode_start",
                "question": self.question,
                "instruction": (
                    "Explore the GSW corpus to find multi-hop connections. "
                    "Use reconcile_entity_across_docs to understand entities, "
                    "create_bridge_qa to create bridge QA pairs, and "
                    "mark_entity_explored when you have exhausted all connections."
                ),
            },
            ensure_ascii=False,
        )

    def _check_turn_limit(self):
        """Force-end the episode if max_turns is reached."""
        if self.turn >= self.max_turns and not self.done:
            self.done = True
            timeout_obs = self._obs(
                {"type": "timeout", "message": f"Reached max turns ({self.max_turns})."},
                terminal=True,
            )
            self._obs_history.append(timeout_obs)


# ---------------------------------------------------------------------------
# Convenience: build environment from index entry
# ---------------------------------------------------------------------------

def env_from_index_entry(
    entry: Dict[str, Any],
    entity_searcher_cls,
    max_turns: int = 30,
) -> GSWEnvironment:
    """
    Build a GSWEnvironment from a training index entry (as produced by data_prep.py).

    Args:
        entry:              One entry from data/rl_training/index.json.
        entity_searcher_cls: The EntitySearcher class (passed to avoid circular import).
        max_turns:          Episode turn limit.

    Returns:
        Initialized (but not reset) GSWEnvironment.
    """
    entity_searcher = entity_searcher_cls(
        path_to_gsw_files=entry["gsw_dir"],
        verbose=False,
    )
    return GSWEnvironment(
        entity_searcher=entity_searcher,
        question=entry["question"],
        gold_answer=entry["answer"],
        gold_decomposition=entry.get("decomposition", []),
        max_turns=max_turns,
    )
