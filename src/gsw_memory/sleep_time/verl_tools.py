"""
veRL native tool wrapper for GSW exploration tools.

All 10 GSW tools share the same GSWBaseTool class. The tool name is
determined by tool_schema.function.name (set in gsw_tool_config.yaml).
A shared GSWEnvironment is lazily created per episode and stored in
agent_data.extra_fields["gsw_env"].

Per-episode data (gsw_dirs, ground_truth, etc.) comes from the parquet
extra_info["interaction_kwargs"], accessible via agent_data.interaction_kwargs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class GSWBaseTool(BaseTool):
    """
    Generic veRL native tool that dispatches to GSWEnvironment.step().

    One instance of this class is created per tool name (10 total).
    All instances for the same episode share a single GSWEnvironment
    stored in agent_data.extra_fields["gsw_env"].
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            return ToolResponse(text='{"error": "no agent_data"}'), 0.0, {}

        env = self._get_or_create_env(agent_data)
        tool_name = self.tool_schema.function.name

        obs, done = env.step(tool_name, parameters)
        reward = env.get_reward() if done else 0.0

        return ToolResponse(text=obs), reward, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_or_create_env(agent_data):
        """Lazily build a shared GSWEnvironment for this episode."""
        if "gsw_env" not in agent_data.extra_fields:
            from .environment import GSWEnvironment

            from gsw_memory.sleep_time.entity_search import EntitySearcher

            ik = agent_data.interaction_kwargs or {}
            gsw_dirs = ik.get("gsw_dirs", [])
            if not gsw_dirs:
                logger.warning("No gsw_dirs in interaction_kwargs; env will be empty")
                agent_data.extra_fields["gsw_env"] = None
                return None

            gsw_root = str(Path(gsw_dirs[0]).parent)
            searcher = EntitySearcher(
                path_to_gsw_files=gsw_root,
                verbose=False,
            )
            env = GSWEnvironment(
                entity_searcher=searcher,
                question="",
                gold_answer=ik.get("ground_truth", ""),
                gold_decomposition=ik.get("decomposition", []),
                max_turns=30,
            )
            env.reset()
            agent_data.extra_fields["gsw_env"] = env

        return agent_data.extra_fields["gsw_env"]
