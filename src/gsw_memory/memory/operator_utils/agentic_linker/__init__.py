"""Linker-based agentic GSW pipeline.

A four-stage replacement for the legacy ``agentic_gsw`` package.

Architecture: extract entities and verb phrases as flat lists in parallel,
then hand both lists + the source text to a multi-turn ``LinkerAgent`` whose
termination contract requires every entity and every verb phrase to be wired
into at least one ``LinkedRelation`` before it can stop. The result is a
GSWStructure assembled deterministically from the linked relations and the
question pairs generated for each.
"""

from .orchestrator import LinkerPipeline

__all__ = ["LinkerPipeline"]
