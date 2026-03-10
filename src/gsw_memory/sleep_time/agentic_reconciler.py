"""
Agentic reconciler for sleep-time GSW exploration.

Uses Together AI's models to drive autonomous exploration and bridge creation.
"""

import json
import os
import logging
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple

from .tools import GSWTools
from .prompts import SLEEP_TIME_SYSTEM_PROMPT, SLEEP_TIME_BRIDGE_VERIFIER_PROMPT

# Setup logger for this module
logger = logging.getLogger(__name__)


class AgenticReconciler:
    """
    Orchestrates agentic sleep-time exploration of GSW corpus.

    Uses Together AI for LLM inference with tool calling.
    """

    def __init__(
        self,
        entity_searcher,
        model_name: str = "gpt-4o",
        budget: Optional[Dict[str, int]] = None,
        verbose: bool = True,
        output_callback: Optional[Any] = None,
        reasoning_effort: str = "high",
        base_url: Optional[str] = None,
        bridge_verifier_enabled: bool = True,
        bridge_verifier_threshold: float = 0.75,
        bridge_verifier_fail_open: bool = False,
        bridge_verifier_model: Optional[str] = None,
        pipeline_mode: str = "legacy",
        hybrid_scope: str = "doc_edge",
        root_model: Optional[str] = None,
        worker_model: Optional[str] = None,
        edge_max_depth: int = 1,
        edge_max_calls: int = 2,
        edge_max_tokens: int = 3000,
        max_optional_docs_per_edge: int = 2,
    ):
        """
        Initialize agentic reconciler.

        Args:
            entity_searcher: Initialized EntitySearcher with loaded GSWs
            model_name: Model name - OpenAI (gpt-4o, gpt-4o-mini), Together AI (openai/gpt-oss-120b, meta-llama/...),
                        or any model name when base_url is provided (local vllm server)
            budget: Dict with "max_entities" and/or "max_tokens" limits
            verbose: Print progress messages
            output_callback: Optional callback function for interactive display
                            Format: callback(event_type: str, data: Dict[str, Any])
            reasoning_effort: Reasoning effort for Together AI models: "low", "medium", or "high" (default: "high")
            base_url: Base URL for OpenAI-compatible API server (e.g. "http://127.0.0.1:6379/v1" for vllm).
                      When set, the OpenAI client is used regardless of model name format.
            bridge_verifier_enabled: Enable post-creation verifier gate for create_bridge_qa.
            bridge_verifier_threshold: Minimum verifier score (0-1) required to keep a created bridge.
            bridge_verifier_fail_open: If True, verifier API errors do not block bridge creation.
            bridge_verifier_model: Optional model for verifier subagent (defaults to model_name).
            pipeline_mode: Exploration mode. "legacy" uses tool-calling chat loop, "rlm" uses deterministic root + edge workers,
                           "hybrid" uses planner-guided self-controller within hard guards.
            hybrid_scope: Hybrid autonomy scope. "edge" | "doc_edge" | "corpus_doc_edge".
            root_model: Optional model for root-stage calls (defaults to model_name).
            worker_model: Optional model for recursive edge workers (defaults to model_name).
            edge_max_depth: Maximum recursion depth per edge in RLM mode.
            edge_max_calls: Maximum worker calls per edge in RLM mode.
            edge_max_tokens: Token budget per edge in RLM mode.
            max_optional_docs_per_edge: Optional fuzzy docs to include per edge in RLM mode.
        """
        self.entity_searcher = entity_searcher
        self.tools = GSWTools(entity_searcher)
        self.model_name = model_name
        self.verbose = verbose
        self.output_callback = output_callback  # Callback for interactive display
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.bridge_verifier_enabled = bridge_verifier_enabled
        self.bridge_verifier_threshold = max(0.0, min(1.0, bridge_verifier_threshold))
        self.bridge_verifier_fail_open = bridge_verifier_fail_open
        self.bridge_verifier_model = bridge_verifier_model or model_name
        self.pipeline_mode = pipeline_mode if pipeline_mode in {"legacy", "rlm", "hybrid"} else "legacy"
        self.hybrid_scope = hybrid_scope if hybrid_scope in {"edge", "doc_edge", "corpus_doc_edge"} else "doc_edge"
        self.root_model = root_model or model_name
        self.worker_model = worker_model or model_name
        self.edge_max_depth = max(0, int(edge_max_depth))
        self.edge_max_calls = max(1, int(edge_max_calls))
        self.edge_max_tokens = max(512, int(edge_max_tokens))
        self.max_optional_docs_per_edge = max(0, int(max_optional_docs_per_edge))

        # Budget tracking
        self.budget = budget or {"max_entities": 20, "max_tokens": 1_000_000}
        self.tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.entities_explored = 0
        self._current_doc_exploration_doc_id: Optional[str] = None
        self.rlm_metrics: Dict[str, Any] = {
            "root_calls": 0,
            "worker_calls": 0,
            "verifier_calls": 0,
            "root_input_tokens": 0,
            "root_output_tokens": 0,
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
            "verifier_input_tokens": 0,
            "verifier_output_tokens": 0,
            "edges_explored": 0,
            "edges_with_bridges": 0,
            "recursive_invocations": 0,
            "low_signal_edges": 0,
            "repeated_failure_stops": 0,
            "edges_skipped_no_path": 0,
            "planner_actions_overridden": 0,
            "prefilter_reject_count": 0,
            "alias_resolution_hits": 0,
            "calls_blocked_no_progress": 0,
            "planner_decisions": 0,
            "planner_fallbacks": 0,
            "planner_stop_actions": 0,
            "docs_attempted": 0,
            "docs_completed": 0,
        }

        # Safety: Cache valid document IDs for reference
        self._valid_doc_ids = set(entity_searcher.gsw_by_doc_id.keys())

        if self.verbose and not output_callback:  # Don't log if using callback (callback handles display)
            logger.info(f"Valid document range: doc_0 to doc_{max([int(d.split('_')[1]) for d in self._valid_doc_ids if d.startswith('doc_')])}")

        # Initialize appropriate client based on model name
        self._initialize_client(model_name, base_url)

        # Tool definitions in OpenAI format
        self.tool_definitions = self._create_tool_definitions()

        if self.verbose:
            print(f"✓ Initialized AgenticReconciler")
            print(f"  Model: {model_name}")
            print(f"  Provider: {self.provider}")
            print(f"  Budget: {self.budget}")
            print(f"  Pipeline mode: {self.pipeline_mode}")
            print(
                f"  Bridge verifier: {'enabled' if self.bridge_verifier_enabled else 'disabled'} "
                f"(model={self.bridge_verifier_model}, threshold={self.bridge_verifier_threshold:.2f})"
            )
            if self.pipeline_mode in {"rlm", "hybrid"}:
                print(
                    "  Pipeline config: "
                    f"root_model={self.root_model}, worker_model={self.worker_model}, "
                    f"depth={self.edge_max_depth}, calls={self.edge_max_calls}, "
                    f"edge_tokens={self.edge_max_tokens}, optional_docs={self.max_optional_docs_per_edge}, "
                    f"hybrid_scope={self.hybrid_scope}"
                )
            print(f"  GSWs loaded: {len(self.entity_searcher.gsw_by_doc_id)}")

    def _initialize_client(self, model_name: str, base_url: Optional[str] = None):
        """
        Initialize the appropriate LLM client based on model name or base_url.

        OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
        Together AI models: Contains '/' like openai/gpt-oss-120b, meta-llama/...
        Bedrock models: bedrock/qwen.qwen3-235b-... (via LiteLLM)
        vllm / local OpenAI-compatible: any model name when base_url is provided
        """
        self.litellm = None  # Only set for bedrock provider

        # Detect provider — vllm/local takes priority if base_url is given
        if base_url is not None:
            self.provider = "vllm"
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key="EMPTY",   # vllm does not require a real key
                    base_url=base_url,
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        elif model_name.startswith("gpt-"):
            self.provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            # Import OpenAI client
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        elif model_name.startswith("bedrock/"):
            self.provider = "bedrock"
            # Ensure converse API is used (required for tool calling)
            if not model_name.startswith("bedrock/converse/"):
                self.model_name = model_name.replace("bedrock/", "bedrock/converse/", 1)
            try:
                import litellm
                self.litellm = litellm
            except ImportError:
                raise ImportError("LiteLLM package not installed. Run: pip install litellm")

        elif "/" in model_name:
            self.provider = "together"
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")

            # Import Together client
            try:
                from together import Together
                self.client = Together(api_key=api_key)
            except ImportError:
                raise ImportError("Together package not installed. Run: pip install together")
        else:
            raise ValueError(
                f"Unknown model provider for '{model_name}'. "
                "Expected OpenAI model (gpt-4o), Together AI model (contains '/'), "
                "Bedrock model (bedrock/...), or provide --base_url for vllm."
            )

    def _create_tool_definitions(self) -> List[Dict[str, Any]]:
        """Create OpenAI-format tool definitions for graph-walk exploration."""
        return [
            # Discovery tools
            {
                "type": "function",
                "function": {
                    "name": "get_document_entities",
                    "description": "Get entities in a document with their relationships and neighbor info. Returns entities sorted by total neighbors (most connected first). Each entity includes roles, states, neighbors with QA pairs, and counts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID (e.g., 'doc_3')"
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_entity_documents",
                    "description": "Get list of document IDs that mention this entity (exact name match). Use to check if a neighbor entity appears in other documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity to search for"
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_entities",
                    "description": "Search for entities across all documents using BM25/semantic similarity. Use when exact name matching (get_entity_documents) returns empty — discovers fuzzy matches like 'Rare' → 'Rare Ltd' in other documents. Returns matches with relevance scores.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Entity name or description to search for"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default 10)",
                                "default": 10
                            },
                            "exclude_doc_id": {
                                "type": "string",
                                "description": "Optional doc_id to exclude from results (e.g., the current document)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            # Graph walk tool
            {
                "type": "function",
                "function": {
                    "name": "find_entity_neighbors",
                    "description": "Find all entities connected to this entity via verb phrases (relationships). Returns neighbor name, relationship type, and doc_id. Use this to see an entity's relationship edges and decide which to follow to other documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity to find neighbors for"
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            # Context tool
            {
                "type": "function",
                "function": {
                    "name": "get_entity_context",
                    "description": "Get all QA pairs, roles, states, and relationships for an entity. Supports single doc, multiple docs (batch), or all docs (merged).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity to get context for"
                            },
                            "doc_id": {
                                "description": "Optional: Single doc ID (string), list of doc IDs (array), or omit for merged context from all docs. Use array for batch retrieval across multiple docs in one call.",
                                "oneOf": [
                                    {
                                        "type": "string",
                                        "description": "Single document ID (e.g., 'doc_4')"
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of document IDs for batch retrieval (e.g., ['doc_0', 'doc_4', 'doc_6'])"
                                    }
                                ]
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            # Bridge tools
            {
                "type": "function",
                "function": {
                    "name": "create_bridge_qa",
                    "description": "Create one or more two-ended bridge QA pairs (QA inversion). Each bridge has a forward question and a reverse question that swap entity roles. Can be used in two modes: (1) Single bridge: pass question, answers, reverse_question, reverse_answers, source_docs, reasoning. (2) Multiple bridges: pass bridges array. Validation is performed automatically.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Forward bridge question (single bridge mode)"
                            },
                            "answers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Entity references in doc_x::ey format that answer the forward question (e.g. ['doc_21::e5'])"
                            },
                            "reverse_question": {
                                "type": "string",
                                "description": "Reverse question viewing the relationship from the answer's perspective (single bridge mode)"
                            },
                            "reverse_answers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Entity references in doc_x::ey format that answer the reverse question (e.g. ['doc_23::e2'])"
                            },
                            "source_docs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of source document IDs (single bridge mode)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "How this bridge was derived (single bridge mode)"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score (0-1, default 0.9)",
                                "default": 0.9
                            },
                            "entities_involved": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Entities mentioned in bridge (optional)"
                            },
                            "bridges": {
                                "type": "array",
                                "description": "Array of two-ended bridge specifications for batch creation. Each bridge must have: question, answers, reverse_question, reverse_answers, source_docs, reasoning. Maximum 20 bridges per call.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "answers": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "reverse_question": {"type": "string"},
                                        "reverse_answers": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "source_docs": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "reasoning": {"type": "string"},
                                        "confidence": {"type": "number", "default": 0.9},
                                        "entities_involved": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["question", "answers", "reverse_question", "reverse_answers", "source_docs", "reasoning"]
                                },
                                "minItems": 1,
                                "maxItems": 20
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bridge_statistics",
                    "description": "Get statistics on bridges created so far (total, coverage, quality).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            # Tracking
            {
                "type": "function",
                "function": {
                    "name": "mark_document_explored",
                    "description": "Mark a document as fully explored. Returns remaining unexplored documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to mark as explored"
                            },
                            "num_bridges_created": {
                                "type": "integer",
                                "description": "Number of bridges created from this document",
                                "default": 0
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            },
            # Document-level planning tools
            {
                "type": "function",
                "function": {
                    "name": "plan_document_exploration",
                    "description": "Auto-build an exploration checklist for a document. Lists all entities and their neighbor edges with pending status. Call this after reading a document's entities to create a structured TODO.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID (e.g., 'doc_5')"
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "begin_neighbor_focus",
                    "description": "Acquire a strict focus lock for one pending edge (source entity -> neighbor). Must be called before neighbor exploration in document mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID the exploration plan belongs to"
                            },
                            "entity_name": {
                                "type": "string",
                                "description": "Source entity in the current document"
                            },
                            "neighbor_name": {
                                "type": "string",
                                "description": "Neighbor entity to focus on"
                            },
                            "relationship": {
                                "type": "string",
                                "description": "Exact relationship phrase connecting source entity to neighbor"
                            }
                        },
                        "required": ["doc_id", "entity_name", "neighbor_name", "relationship"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_neighbor_doc_coverage",
                    "description": "Build per-edge doc coverage plan for the active focused edge. Mandatory docs come from exact entity matches; fuzzy docs are optional hints.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID for the active edge"
                            },
                            "entity_name": {
                                "type": "string",
                                "description": "Source entity for the active edge"
                            },
                            "neighbor_name": {
                                "type": "string",
                                "description": "Focused neighbor entity"
                            },
                            "relationship": {
                                "type": "string",
                                "description": "Exact relationship phrase for this edge"
                            },
                            "include_fuzzy": {
                                "type": "boolean",
                                "description": "Include optional fuzzy document hints from search_entities",
                                "default": True
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum fuzzy hits to inspect when include_fuzzy=true",
                                "default": 10
                            }
                        },
                        "required": ["doc_id", "entity_name", "neighbor_name", "relationship"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_corpus_exploration",
                    "description": "Build global exploration queue across documents using existing plan/completion state. Use between documents to pick what to explore next.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "description": "Queue strategy (default max_pending_neighbors)",
                                "default": "max_pending_neighbors"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum queue entries to return",
                                "default": 20
                            },
                            "include_completed": {
                                "type": "boolean",
                                "description": "Include completed docs in returned queue",
                                "default": False
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_neighbor_explored",
                    "description": "Mark one or more neighbor edges as explored in a document's plan. Supports relationship disambiguation when a neighbor appears under multiple relationships.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID the plan belongs to"
                            },
                            "entity_name": {
                                "type": "string",
                                "description": "Entity in the document whose neighbor was explored"
                            },
                            "neighbor_name": {
                                "description": "Single neighbor name (string) or list of names (batch mode)",
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}}
                                ]
                            },
                            "relationship": {
                                "description": "Optional relationship (string) or list (batch) to disambiguate repeated neighbor names.",
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}}
                                ]
                            },
                            "bridges_created": {
                                "description": "Number of bridges created. Int for single/uniform, list for individual counts per neighbor.",
                                "oneOf": [
                                    {"type": "integer", "default": 0},
                                    {"type": "array", "items": {"type": "integer"}}
                                ]
                            }
                        },
                        "required": ["doc_id", "entity_name", "neighbor_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doc_exploration_status",
                    "description": "Get current exploration status for a document's plan. Shows which neighbor edges are explored vs pending per entity, with overall counts and active focus lock.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to check"
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            }
        ]

    def _record_stage_call(self, stage: str) -> None:
        """Track stage-level call counts for RLM/verification metrics."""
        if stage == "root":
            self.rlm_metrics["root_calls"] += 1
        elif stage == "worker":
            self.rlm_metrics["worker_calls"] += 1
        elif stage == "verifier":
            self.rlm_metrics["verifier_calls"] += 1

    def _consume_usage(self, response: Any, stage: str = "legacy") -> None:
        """Aggregate token usage into global and stage-level counters."""
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        total_tokens = int(total_tokens or 0)

        self.tokens_used += total_tokens
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens

        if stage == "root":
            self.rlm_metrics["root_input_tokens"] += prompt_tokens
            self.rlm_metrics["root_output_tokens"] += completion_tokens
        elif stage == "worker":
            self.rlm_metrics["worker_input_tokens"] += prompt_tokens
            self.rlm_metrics["worker_output_tokens"] += completion_tokens
        elif stage == "verifier":
            self.rlm_metrics["verifier_input_tokens"] += prompt_tokens
            self.rlm_metrics["verifier_output_tokens"] += completion_tokens

    def _call_model_for_stage(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stage: str = "worker",
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Any:
        """
        Unified provider call wrapper for stage-routed model calls.

        Returns:
            Assistant message object from the first choice.
        """
        selected_model = model_name or self.model_name
        if self.provider == "bedrock" and selected_model.startswith("bedrock/") and not selected_model.startswith("bedrock/converse/"):
            selected_model = selected_model.replace("bedrock/", "bedrock/converse/", 1)
        self._record_stage_call(stage)

        api_params: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(128, int(max_tokens)),
        }
        if tools is not None:
            api_params["tools"] = tools
        if tool_choice is not None:
            api_params["tool_choice"] = tool_choice
        if response_format is not None and self.provider in {"openai", "vllm"}:
            api_params["response_format"] = response_format

        # Provider-specific knobs.
        if self.provider == "vllm":
            api_params["presence_penalty"] = 0.2
            api_params["extra_body"] = {"top_k": 20, "min_p": 0.0}
        elif self.provider == "bedrock":
            api_params["reasoning_effort"] = self.reasoning_effort
            allowed_params = ["reasoning_effort"]
            if tools is not None:
                allowed_params.extend(["tools", "tool_choice"])
            api_params["allowed_openai_params"] = allowed_params
            # Bedrock computes dynamic limits; removing hard max avoids overflow on smaller contexts.
            api_params.pop("max_tokens", None)
        elif self.provider == "together":
            api_params["top_k"] = 20
            api_params["presence_penalty"] = 0.6
            if "gpt-oss" in selected_model.lower():
                api_params["reasoning_effort"] = self.reasoning_effort

        try:
            try:
                if self.provider == "bedrock":
                    response = self.litellm.completion(**api_params)
                else:
                    response = self.client.chat.completions.create(**api_params)
            except Exception:
                if "response_format" in api_params:
                    fallback_params = dict(api_params)
                    fallback_params.pop("response_format", None)
                    if self.provider == "bedrock":
                        response = self.litellm.completion(**fallback_params)
                    else:
                        response = self.client.chat.completions.create(**fallback_params)
                else:
                    raise
        except Exception:
            raise

        self._consume_usage(response, stage=stage)
        return response.choices[0].message

    def _coerce_to_text(self, value: Any) -> str:
        """Convert provider-specific text payloads to a plain string."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for block in value:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif block is not None:
                    parts.append(str(block))
            return "\n".join(parts).strip()
        return str(value)

    def _extract_message_content_with_source(self, message: Any) -> Tuple[str, str]:
        """
        Extract verifier text with source metadata.

        Checks content, reasoning, and reasoning_content in that order.
        """
        fields = [
            ("content", getattr(message, "content", None)),
            ("reasoning", getattr(message, "reasoning", None)),
            ("reasoning_content", getattr(message, "reasoning_content", None)),
        ]
        for source, value in fields:
            text = self._coerce_to_text(value).strip()
            if text:
                return text, source
        return "", "empty"

    def _normalize_verifier_text(self, text: str) -> str:
        """Normalize raw verifier output before JSON parsing."""
        normalized = (text or "").strip()
        if not normalized:
            return ""

        # Remove think tags used by reasoning models (closed and unclosed).
        normalized = re.sub(r"<think>.*?</think>", " ", normalized, flags=re.DOTALL | re.IGNORECASE).strip()
        normalized = re.sub(r"<think>.*", " ", normalized, flags=re.DOTALL | re.IGNORECASE).strip()

        # Unwrap a full markdown fenced block if present.
        fenced_full = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", normalized, flags=re.DOTALL | re.IGNORECASE)
        if fenced_full:
            normalized = fenced_full.group(1).strip()

        return normalized

    def _preview_text(self, text: str, max_len: int = 240) -> str:
        """Compact a raw text payload for safe logs."""
        compact = re.sub(r"\s+", " ", (text or "")).strip()
        if len(compact) > max_len:
            return compact[:max_len] + "..."
        return compact

    def _extract_first_balanced_json_object(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON object from arbitrary text."""
        start_idx: Optional[int] = None
        depth = 0
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if start_idx is None:
                if ch == "{":
                    start_idx = idx
                    depth = 1
                    in_string = False
                    escape = False
                continue

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx:idx + 1]

        return None

    def _coerce_reasoning_content(self, rc) -> Optional[str]:
        """Normalize reasoning_content (string, list of blocks, or pydantic objects)."""
        if isinstance(rc, str):
            return rc.strip() or None
        if isinstance(rc, list):
            parts = []
            for block in rc:
                text = None
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or block.get("thinking")
                else:
                    # Pydantic object
                    text = getattr(block, "text", None) or getattr(block, "content", None) or getattr(block, "thinking", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip() or None
        return self._coerce_to_text(rc) or None

    def _extract_think_tags(self, content: Optional[str]) -> Optional[str]:
        """Extract <think> content from text. Handles closed AND unclosed tags."""
        if not content:
            return None
        # Try closed tag first
        m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if m:
            return m.group(1).strip() or None
        # Try unclosed tag (model transitioned to tool call before closing)
        m = re.search(r'<think>(.*)', content, re.DOTALL)
        if m:
            text = m.group(1).strip()
            # Strip partial closing tag if present
            text = re.sub(r'</think\s*$', '', text).strip()
            return text or None
        return None

    def _extract_reasoning(self, message) -> Optional[str]:
        """Extract reasoning/thinking from assistant message, per provider."""
        if self.provider == "together":
            # Together AI GPT-OSS: .reasoning field (string)
            if hasattr(message, 'reasoning') and message.reasoning:
                return self._coerce_to_text(message.reasoning)
            # Together Qwen3: <think> tags in content
            return self._extract_think_tags(message.content)

        elif self.provider == "vllm":
            # vLLM with --enable-reasoning: .reasoning_content (string or list)
            rc = getattr(message, 'reasoning_content', None)
            if rc:
                return self._coerce_reasoning_content(rc)
            # vLLM without --enable-reasoning: <think> tags in content
            return self._extract_think_tags(message.content)

        elif self.provider == "bedrock":
            # Bedrock via LiteLLM: .reasoning_content (string or list of content blocks)
            rc = getattr(message, 'reasoning_content', None)
            if rc:
                return self._coerce_reasoning_content(rc)
            return self._extract_think_tags(message.content)

        elif self.provider == "openai":
            # OpenAI o-series: .reasoning (undocumented) or content fallback
            if hasattr(message, 'reasoning') and message.reasoning:
                return self._coerce_to_text(message.reasoning)
            # Standard OpenAI: no reasoning field, use content only when no tool calls
            if message.content and not message.tool_calls:
                return message.content
            return None

        return None

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Parse JSON object from raw model output."""
        stripped = self._normalize_verifier_text(text)
        if not stripped:
            raise ValueError("Empty verifier response")

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Parse fenced JSON snippets if direct parse fails.
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        for block in fenced_blocks:
            candidate = block.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        # Last chance: parse first balanced JSON object embedded in prose.
        embedded = self._extract_first_balanced_json_object(stripped)
        if embedded:
            return json.loads(embedded)

        raise ValueError("No JSON object found in verifier output")

    def _normalize_verifier_output(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize verifier output into a stable shape."""
        raw_score = raw.get("score", raw.get("confidence", 0.0))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        raw_codes = raw.get("failure_codes", [])
        if isinstance(raw_codes, str):
            raw_codes = [raw_codes]
        if not isinstance(raw_codes, list):
            raw_codes = []
        failure_codes = [str(code).strip().upper() for code in raw_codes if str(code).strip()]

        pass_flag = raw.get("pass")
        if pass_flag is None:
            pass_flag = raw.get("valid", False)

        notes = raw.get("notes") or raw.get("reasoning") or raw.get("explanation") or ""

        return {
            "pass": bool(pass_flag),
            "score": score,
            "failure_codes": failure_codes,
            "notes": str(notes),
            "raw": raw,
        }

    def _call_verifier_model(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """Call verifier model once and return raw text with source field."""
        verifier_message = self._call_model_for_stage(
            messages=messages,
            model_name=self.bridge_verifier_model,
            max_tokens=900,
            temperature=0.0,
            stage="verifier",
            response_format={"type": "json_object"},
        )
        raw_text, source_field = self._extract_message_content_with_source(verifier_message)
        return raw_text, source_field

    def _verify_bridge_candidate(self, bridge_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM verifier subagent for a proposed bridge candidate."""
        if not self.bridge_verifier_enabled:
            return {
                "pass": True,
                "score": 1.0,
                "failure_codes": [],
                "notes": "Verifier disabled.",
                "raw": {},
            }

        candidate_payload = {
            "question": bridge_spec.get("question"),
            "answers": bridge_spec.get("answers", []),
            "reverse_question": bridge_spec.get("reverse_question"),
            "reverse_answers": bridge_spec.get("reverse_answers", []),
            "source_docs": bridge_spec.get("source_docs", []),
            "reasoning": bridge_spec.get("reasoning", ""),
            "entities_involved": bridge_spec.get("entities_involved", []),
            "similarity_signal": bridge_spec.get("similarity_signal", []),
        }

        messages = [
            {"role": "system", "content": SLEEP_TIME_BRIDGE_VERIFIER_PROMPT},
            {"role": "user", "content": json.dumps(candidate_payload, ensure_ascii=False, indent=2)},
        ]

        try:
            raw_text, source_field = self._call_verifier_model(messages)
        except Exception as e:
            logger.error(f"Bridge verifier API call failed: {e}", exc_info=True)
            if self.bridge_verifier_fail_open:
                return {
                    "pass": True,
                    "score": 1.0,
                    "failure_codes": ["VERIFIER_ERROR_FAIL_OPEN"],
                    "notes": f"Verifier error ignored due to fail-open policy: {e}",
                    "raw": {},
                }
            return {
                "pass": False,
                "score": 0.0,
                "failure_codes": ["VERIFIER_ERROR"],
                "notes": f"Verifier API error: {e}",
                "raw": {},
            }

        try:
            parsed = self._extract_json_from_text(raw_text)
            verdict = self._normalize_verifier_output(parsed)
            verdict["source_field"] = source_field
            verdict["parse_stage"] = "initial"
            return verdict
        except Exception as initial_error:
            logger.warning(
                "Verifier parse failed [stage=initial provider=%s model=%s source=%s] error=%s preview=%s",
                self.provider,
                self.bridge_verifier_model,
                source_field,
                initial_error,
                self._preview_text(raw_text),
            )

        # One strict repair attempt with the same model.
        repair_messages = messages + [
            {"role": "assistant", "content": raw_text or "[empty_response]"},
            {
                "role": "user",
                "content": (
                    "Your previous reply was not parseable JSON. "
                    "Re-output ONLY one valid JSON object with keys: "
                    "pass (bool), score (0..1 number), failure_codes (array of strings), notes (string). "
                    "Do not include markdown, prose, or extra keys."
                ),
            },
        ]

        try:
            repaired_text, repaired_source = self._call_verifier_model(repair_messages)
            repaired_parsed = self._extract_json_from_text(repaired_text)
            repaired_verdict = self._normalize_verifier_output(repaired_parsed)
            repaired_verdict["source_field"] = repaired_source
            repaired_verdict["parse_stage"] = "repair_retry"
            repaired_verdict["raw"] = {
                "initial_text": raw_text,
                "repaired_text": repaired_text,
            }
            return repaired_verdict
        except Exception as retry_error:
            logger.error(
                "Verifier parse failed [stage=repair_retry provider=%s model=%s source=%s] error=%s preview=%s",
                self.provider,
                self.bridge_verifier_model,
                repaired_source if "repaired_source" in locals() else "unknown",
                retry_error,
                self._preview_text(repaired_text if "repaired_text" in locals() else ""),
            )

            if self.bridge_verifier_fail_open:
                return {
                    "pass": True,
                    "score": 1.0,
                    "failure_codes": ["VERIFIER_PARSE_ERROR_FAIL_OPEN"],
                    "notes": f"Verifier parse error ignored due to fail-open policy: {retry_error}",
                    "raw": {
                        "initial_text": raw_text,
                        "retry_text": repaired_text if "repaired_text" in locals() else "",
                    },
                    "source_field": source_field,
                    "parse_stage": "repair_retry",
                }

            return {
                "pass": False,
                "score": 0.0,
                "failure_codes": ["VERIFIER_PARSE_ERROR"],
                "notes": f"Verifier parse error: {retry_error}",
                "raw": {
                    "initial_text": raw_text,
                    "retry_text": repaired_text if "repaired_text" in locals() else "",
                },
                "source_field": source_field,
                "parse_stage": "repair_retry",
            }

    def _rollback_bridge(self, bridge_id: Optional[str]) -> bool:
        """Remove a bridge by id from in-memory storage."""
        if not bridge_id:
            return False
        before = len(self.tools.bridges_created)
        self.tools.bridges_created = [
            b for b in self.tools.bridges_created if b.get("bridge_id") != bridge_id
        ]
        return len(self.tools.bridges_created) < before

    def _apply_verifier_to_bridge_result(self, result: Dict[str, Any], bridge_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Gate successful bridge creations with verifier output."""
        if not isinstance(result, dict) or not result.get("success") or not self.bridge_verifier_enabled:
            return result

        verifier_bridge_spec = dict(bridge_spec or {})
        validation = result.get("validation", {}) if isinstance(result, dict) else {}
        if isinstance(validation, dict) and validation.get("similarity_signal"):
            verifier_bridge_spec["similarity_signal"] = validation.get("similarity_signal", [])

        verdict = self._verify_bridge_candidate(verifier_bridge_spec)
        validation = result.setdefault("validation", {})
        validation["verifier_score"] = verdict["score"]
        validation["verifier_failure_codes"] = verdict["failure_codes"]
        validation["verifier_notes"] = verdict["notes"]
        validation["verifier_model"] = self.bridge_verifier_model
        validation["verifier_threshold"] = self.bridge_verifier_threshold
        validation["verifier_raw_pass"] = verdict["pass"]
        if verdict.get("source_field"):
            validation["verifier_source_field"] = verdict["source_field"]
        if verdict.get("parse_stage"):
            validation["verifier_parse_stage"] = verdict["parse_stage"]

        passes_gate = verdict["pass"] and verdict["score"] >= self.bridge_verifier_threshold
        validation["verifier_pass"] = passes_gate

        if passes_gate:
            return result

        bridge_id = result.get("bridge_id")
        rolled_back = self._rollback_bridge(bridge_id)
        logger.info(
            "Bridge %s rejected by verifier (score=%.2f, threshold=%.2f, codes=%s, rolled_back=%s)",
            bridge_id,
            verdict["score"],
            self.bridge_verifier_threshold,
            verdict["failure_codes"],
            rolled_back,
        )

        return {
            "success": False,
            "error": (
                "Bridge rejected by verifier. "
                f"score={verdict['score']:.2f}, threshold={self.bridge_verifier_threshold:.2f}"
            ),
            "bridge_id": bridge_id,
            "rolled_back": rolled_back,
            "validation": {
                "valid": False,
                "confidence": verdict["score"],
                "reasoning": verdict["notes"] or "Bridge rejected by verifier",
                "failure_codes": verdict["failure_codes"] or ["VERIFIER_REJECTED"],
                "stage": "verifier",
                "verifier_model": self.bridge_verifier_model,
                "verifier_threshold": self.bridge_verifier_threshold,
                "verifier_source_field": verdict.get("source_field"),
                "verifier_parse_stage": verdict.get("parse_stage"),
            }
        }

    def _emit_create_bridge_callbacks(self, result: Any, arguments: Dict[str, Any]) -> None:
        """Emit callback events for bridge creation and verifier validation."""
        if not self.output_callback:
            return

        if isinstance(result, list):
            bridge_specs = arguments.get("bridges", []) if isinstance(arguments, dict) else []
            for idx, item in enumerate(result):
                spec = bridge_specs[idx] if idx < len(bridge_specs) else {}
                self._emit_create_bridge_callbacks(item, spec)
            return

        if not isinstance(result, dict):
            return

        if result.get("success"):
            self.output_callback("bridge_created", {
                "bridge_id": result.get("bridge_id"),
                "question": arguments.get("question", ""),
                "answers": arguments.get("answers", []),
                "validation": result.get("validation", {})
            })
            return

        validation = result.get("validation", {})
        if not validation:
            validation = {
                "valid": False,
                "confidence": 0.0,
                "reasoning": result.get("error", "Bridge creation failed."),
            }
        validation_payload = dict(validation)
        validation_payload.setdefault("question", arguments.get("question", ""))
        validation_payload.setdefault("reverse_question", arguments.get("reverse_question", ""))
        self.output_callback("validation", validation_payload)

    def _execute_create_bridge_tool(self, tool_method: Any, arguments: Dict[str, Any]) -> Any:
        """Execute create_bridge_qa with post-create verifier gating."""
        bridges_arg = arguments.get("bridges")

        # Single bridge mode
        if bridges_arg is None:
            result = tool_method(**arguments)
            result = self._apply_verifier_to_bridge_result(result, arguments)
            self._emit_create_bridge_callbacks(result, arguments)
            return result

        # Preserve existing tool-level batch validation behavior
        if not isinstance(bridges_arg, list) or len(bridges_arg) == 0 or len(bridges_arg) > 20:
            result = tool_method(**arguments)
            self._emit_create_bridge_callbacks(result, arguments)
            return result

        # Process each batch item independently so verifier can reject selectively
        final_results = []
        for bridge_spec in bridges_arg:
            raw_result = tool_method(bridges=[bridge_spec])
            if isinstance(raw_result, list):
                single_result = raw_result[0] if raw_result else {
                    "success": False,
                    "error": "Empty result returned for batch bridge item.",
                    "validation": {"valid": False, "confidence": 0.0}
                }
            else:
                single_result = raw_result

            if isinstance(single_result, dict):
                single_result = self._apply_verifier_to_bridge_result(single_result, bridge_spec)
            final_results.append(single_result)
            self._emit_create_bridge_callbacks(single_result, bridge_spec)

        return final_results

    def _focus_guard_error(
        self,
        tool_name: str,
        message: str,
        hint: str,
        active_focus: Optional[Dict[str, Any]] = None,
        stage: str = "focus_guard",
    ) -> Dict[str, Any]:
        """Build a structured focus-guard error payload."""
        payload = {
            "error": message,
            "tool": tool_name,
            "stage": stage,
            "hint": hint,
        }
        if active_focus is not None:
            payload["active_focus"] = active_focus
        return payload

    @staticmethod
    def _normalize_name(value: Any) -> str:
        """Normalize user/model text for strict equality checks."""
        if value is None:
            return ""
        return str(value).strip().lower()

    def _check_neighbor_focus_guard(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enforce strict neighbor-focus protocol for document exploration mode.

        Returns:
            None if call is allowed, otherwise a structured focus_guard error payload.
        """
        current_doc = self._current_doc_exploration_doc_id
        if not current_doc:
            return None

        active_focus = self.tools.active_neighbor_focus
        requires_focus = {
            "search_entities",
            "get_entity_context",
            "create_bridge_qa",
            "mark_neighbor_explored",
            "plan_neighbor_doc_coverage",
        }

        if active_focus is None:
            if tool_name in requires_focus:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message=(
                        f"Tool '{tool_name}' requires an active neighbor focus lock "
                        "in document exploration mode."
                    ),
                    hint=(
                        "Call begin_neighbor_focus(doc_id, entity_name, neighbor_name, relationship) "
                        "for a pending edge before neighbor exploration."
                    ),
                    active_focus=None,
                )
            if tool_name == "mark_document_explored":
                arg_doc = arguments.get("doc_id")
                if arg_doc is not None and self._normalize_name(arg_doc) != self._normalize_name(current_doc):
                    return self._focus_guard_error(
                        tool_name=tool_name,
                        message="mark_document_explored doc_id must match the current exploration document.",
                        hint=f"Use doc_id='{current_doc}'.",
                        active_focus=None,
                    )

                plan = self.tools.doc_exploration_plans.get(current_doc)
                pending = int(plan.get("pending_count", 0)) if isinstance(plan, dict) else 0
                if pending > 0:
                    return self._focus_guard_error(
                        tool_name=tool_name,
                        message="Cannot mark document explored while pending edges remain.",
                        hint="Call get_doc_exploration_status and finish all pending edges first.",
                        active_focus=None,
                        stage="coverage_guard",
                    )
            return None

        active_doc = active_focus.get("doc_id")
        if active_doc != current_doc:
            return self._focus_guard_error(
                tool_name=tool_name,
                message=(
                    f"Active neighbor focus belongs to '{active_doc}', but current exploration doc "
                    f"is '{current_doc}'."
                ),
                hint="Resolve or clear the active focus in the current document before proceeding.",
                active_focus=active_focus,
            )

        locked_allowed = {
            "search_entities",
            "get_entity_context",
            "create_bridge_qa",
            "plan_neighbor_doc_coverage",
            "mark_neighbor_explored",
            "get_bridge_statistics",
            "get_doc_exploration_status",
        }
        if tool_name not in locked_allowed:
            return self._focus_guard_error(
                tool_name=tool_name,
                message=(
                    f"Tool '{tool_name}' is blocked while neighbor focus lock is active."
                ),
                hint=(
                    "Allowed while locked: search_entities, get_entity_context (single doc), "
                    "create_bridge_qa, plan_neighbor_doc_coverage, mark_neighbor_explored (same edge), "
                    "get_bridge_statistics, get_doc_exploration_status."
                ),
                active_focus=active_focus,
            )

        focus_neighbor = self._normalize_name(active_focus.get("neighbor_name"))
        focus_entity = self._normalize_name(active_focus.get("entity_name"))
        focus_relationship = self._normalize_name(active_focus.get("relationship"))

        if tool_name == "search_entities":
            query = self._normalize_name(arguments.get("query"))
            if query != focus_neighbor:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message=(
                        "search_entities query must match the active focused neighbor exactly."
                    ),
                    hint=f"Use query='{active_focus.get('neighbor_name')}'.",
                    active_focus=active_focus,
                )

        elif tool_name == "plan_neighbor_doc_coverage":
            arg_doc = self._normalize_name(arguments.get("doc_id"))
            arg_entity = self._normalize_name(arguments.get("entity_name"))
            arg_neighbor = self._normalize_name(arguments.get("neighbor_name"))
            arg_relationship = self._normalize_name(arguments.get("relationship"))

            if arg_doc != self._normalize_name(current_doc):
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="plan_neighbor_doc_coverage doc_id must match the active document.",
                    hint=f"Use doc_id='{current_doc}'.",
                    active_focus=active_focus,
                )
            if arg_entity != focus_entity:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="plan_neighbor_doc_coverage entity_name must match active source entity.",
                    hint=f"Use entity_name='{active_focus.get('entity_name')}'.",
                    active_focus=active_focus,
                )
            if arg_neighbor != focus_neighbor:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="plan_neighbor_doc_coverage neighbor_name must match active focused neighbor.",
                    hint=f"Use neighbor_name='{active_focus.get('neighbor_name')}'.",
                    active_focus=active_focus,
                )
            if arg_relationship != focus_relationship:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="plan_neighbor_doc_coverage relationship must match active focused relationship.",
                    hint=f"Use relationship='{active_focus.get('relationship')}'.",
                    active_focus=active_focus,
                )

        elif tool_name == "get_entity_context":
            entity_name = self._normalize_name(arguments.get("entity_name"))
            if entity_name != focus_neighbor:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message=(
                        "get_entity_context entity_name must match the active focused neighbor."
                    ),
                    hint=f"Use entity_name='{active_focus.get('neighbor_name')}'.",
                    active_focus=active_focus,
                )
            doc_arg = arguments.get("doc_id")
            if not isinstance(doc_arg, str) or not doc_arg.strip():
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message=(
                        "get_entity_context requires a single doc_id string while focus lock is active."
                    ),
                    hint="Call get_entity_context(neighbor_name, 'doc_x') one document at a time.",
                    active_focus=active_focus,
                )

        elif tool_name == "mark_neighbor_explored":
            arg_doc = self._normalize_name(arguments.get("doc_id"))
            arg_entity = self._normalize_name(arguments.get("entity_name"))
            arg_neighbor = arguments.get("neighbor_name")
            arg_relationship = arguments.get("relationship")

            if isinstance(arg_neighbor, list):
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="Batch mark_neighbor_explored is blocked while focus lock is active.",
                    hint="Mark the active edge only with a single neighbor_name and relationship.",
                    active_focus=active_focus,
                )

            if arg_doc != self._normalize_name(current_doc):
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="mark_neighbor_explored doc_id must match the active document.",
                    hint=f"Use doc_id='{current_doc}'.",
                    active_focus=active_focus,
                )
            if arg_entity != focus_entity:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="mark_neighbor_explored entity_name must match the active source entity.",
                    hint=f"Use entity_name='{active_focus.get('entity_name')}'.",
                    active_focus=active_focus,
                )
            if self._normalize_name(arg_neighbor) != focus_neighbor:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="mark_neighbor_explored neighbor_name must match the active focused neighbor.",
                    hint=f"Use neighbor_name='{active_focus.get('neighbor_name')}'.",
                    active_focus=active_focus,
                )
            if self._normalize_name(arg_relationship) != focus_relationship:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message=(
                        "mark_neighbor_explored relationship must match the active focused relationship."
                    ),
                    hint=f"Use relationship='{active_focus.get('relationship')}'.",
                    active_focus=active_focus,
                )

            coverage = self.tools.get_neighbor_doc_coverage_status(
                doc_id=active_focus["doc_id"],
                entity_name=active_focus["entity_name"],
                neighbor_name=active_focus["neighbor_name"],
                relationship=active_focus["relationship"],
            )
            if coverage is None:
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="mark_neighbor_explored requires plan_neighbor_doc_coverage for the active edge.",
                    hint="Call plan_neighbor_doc_coverage first.",
                    active_focus=active_focus,
                    stage="coverage_guard",
                )
            if coverage.get("pending_mandatory_docs"):
                pending_docs = coverage["pending_mandatory_docs"]
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="Cannot mark edge explored before mandatory neighbor docs are covered.",
                    hint=f"Read pending mandatory docs with get_entity_context: {pending_docs}",
                    active_focus=active_focus,
                    stage="coverage_guard",
                )

        elif tool_name == "get_doc_exploration_status":
            arg_doc = arguments.get("doc_id")
            if arg_doc is not None and self._normalize_name(arg_doc) != self._normalize_name(current_doc):
                return self._focus_guard_error(
                    tool_name=tool_name,
                    message="get_doc_exploration_status doc_id must match the active document while locked.",
                    hint=f"Use doc_id='{current_doc}'.",
                    active_focus=active_focus,
                )

        return None

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with safety checks and validation enforcement.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool result or error dict
        """
        tool_method = getattr(self.tools, tool_name, None)
        if tool_method is None:
            logger.error(f"Unknown tool requested: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        guard_error = self._check_neighbor_focus_guard(tool_name, arguments)
        if guard_error is not None:
            logger.warning(
                "Focus guard blocked tool call [tool=%s doc=%s active_focus=%s]",
                tool_name,
                self._current_doc_exploration_doc_id,
                self.tools.active_neighbor_focus,
            )
            return guard_error

        # Execute the tool with error handling
        try:
            if tool_name == "create_bridge_qa":
                result = self._execute_create_bridge_tool(tool_method, arguments)
            else:
                result = tool_method(**arguments)
            logger.debug(f"Tool {tool_name} executed successfully")

            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tool": tool_name,
                "arguments": arguments
            }

    def _create_rlm_scheduler(self):
        """Build deterministic root scheduler + recursive edge worker for RLM mode."""
        from .rlm_pipeline import RootScheduler, RecursiveEdgeWorker

        worker = RecursiveEdgeWorker(
            reconciler=self,
            model_name=self.worker_model,
            max_depth=self.edge_max_depth,
            max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
        )
        return RootScheduler(
            reconciler=self,
            worker=worker,
            edge_max_depth=self.edge_max_depth,
            edge_max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
            max_optional_docs_per_edge=self.max_optional_docs_per_edge,
        )

    def _create_hybrid_scheduler(self):
        """Build planner-guided hybrid scheduler + recursive edge worker."""
        from .rlm_pipeline import HybridScheduler, RecursiveEdgeWorker

        worker = RecursiveEdgeWorker(
            reconciler=self,
            model_name=self.worker_model,
            max_depth=self.edge_max_depth,
            max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
        )
        return HybridScheduler(
            reconciler=self,
            worker=worker,
            edge_max_depth=self.edge_max_depth,
            edge_max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
            max_optional_docs_per_edge=self.max_optional_docs_per_edge,
            hybrid_scope=self.hybrid_scope,
        )

    def run_rlm_document(self, doc_id: str) -> Dict[str, Any]:
        """Run deterministic RLM exploration for one document."""
        scheduler = self._create_rlm_scheduler()
        return scheduler.run_document(doc_id)

    def run_rlm_corpus(self) -> Dict[str, Any]:
        """Run deterministic RLM exploration across corpus queue."""
        scheduler = self._create_rlm_scheduler()
        max_docs = self.budget.get("max_documents")
        if max_docs is None:
            # Backward compatibility for older callers.
            max_docs = self.budget.get("max_entities")
        return scheduler.run_corpus(max_documents=max_docs)

    def run_hybrid_document(self, doc_id: str) -> Dict[str, Any]:
        """Run hybrid planner-guided exploration for one document."""
        scheduler = self._create_hybrid_scheduler()
        return scheduler.run_document(doc_id)

    def run_hybrid_corpus(self) -> Dict[str, Any]:
        """Run hybrid planner-guided exploration across corpus queue."""
        scheduler = self._create_hybrid_scheduler()
        max_docs = self.budget.get("max_documents")
        if max_docs is None:
            max_docs = self.budget.get("max_entities")
        return scheduler.run_corpus(max_documents=max_docs)

    def explore_entity(self, entity_name: Optional[str] = None, doc_id: Optional[str] = None, max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Let agent explore and create bridges.

        When doc_id is provided, explores that specific document (graph-walk mode).
        When entity_name is provided, explores that specific entity.
        When neither is provided, explores the full corpus.

        Args:
            entity_name: Entity to explore, or None
            doc_id: Document to explore (graph-walk mode), or None
            max_iterations: Max tool calling rounds

        Returns:
            Summary of exploration (bridges created, tool calls, etc.)
        """
        if self.pipeline_mode == "rlm":
            # RLM mode is designed around document/corpus exploration with deterministic scheduling.
            if doc_id:
                return self.run_rlm_document(doc_id)
            if entity_name is None:
                return self.run_rlm_corpus()
            # For explicit entity exploration, keep legacy behavior to preserve compatibility.
            logger.info("RLM mode received entity-level request; falling back to legacy loop for entity=%s", entity_name)
        elif self.pipeline_mode == "hybrid":
            # Hybrid mode uses the same doc/corpus entry points with planner-guided control.
            if doc_id:
                return self.run_hybrid_document(doc_id)
            if entity_name is None:
                return self.run_hybrid_corpus()
            # Preserve explicit entity behavior for compatibility.
            logger.info("Hybrid mode received entity-level request; falling back to legacy loop for entity=%s", entity_name)

        label = doc_id or entity_name or "corpus"
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Exploring: {label}")
            print('='*80)

        if doc_id:
            user_msg = (
                f"Explore document '{doc_id}' and create bridge QA pairs. "
                f"Start by calling get_document_entities('{doc_id}') to see its entities, "
                "then walk relationship edges to find bridging entities in other documents "
                "and create chain bridges."
            )
        elif entity_name:
            user_msg = f"Explore the entity '{entity_name}' and create any useful bridge QA pairs you find by combining information across documents."
        else:
            user_msg = (
                "Explore this corpus of documents and create bridge QA pairs. "
                "Start by reading a document's entities, then walk relationship edges "
                "to find bridging entities in other documents and create chain bridges."
            )

        messages = [
            {"role": "system", "content": SLEEP_TIME_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]

        tool_calls_made = []
        bridges_created = 0

        for iteration in range(max_iterations):
            # Callback: Iteration start
            if self.output_callback:
                self.output_callback("iteration_start", {
                    "entity": label,
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations
                })
            elif self.verbose:
                print(f"\n[Iteration {iteration + 1}/{max_iterations}]")

            # Call LLM
            # Build API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "tools": self.tool_definitions,
                "tool_choice": "auto",
                "temperature": 0.6,   # Focused reasoning
                "top_p": 0.95,        # Nucleus sampling
                "max_tokens": 32768,
            }

            # vLLM-specific parameters (OpenAI-compatible server supports these extra sampling params)
            if self.provider == "vllm":
                api_params["presence_penalty"] = 0.4  # Reduce endless repetitions (0–2 range); SDK knows this param
                api_params["extra_body"] = {          # vLLM extensions — must go via extra_body (not in OpenAI SDK signature)
                    "top_k": 20,                      # Limit vocabulary per step for conciseness
                    "min_p": 0.0,                     # Minimum probability threshold
                }

            # Bedrock via LiteLLM
            elif self.provider == "bedrock":
                api_params["reasoning_effort"] = self.reasoning_effort
                # Force litellm to allow tools for newer Bedrock models it doesn't know yet
                api_params["allowed_openai_params"] = ["tools", "tool_choice", "reasoning_effort"]
                # Remove hardcoded max_tokens — Bedrock auto-calculates remaining budget
                # (prevents overflow on small-context models like Qwen3-32B with 32K context)
                api_params.pop("max_tokens", None)

            # Together AI-specific parameters (not supported by OpenAI)
            elif self.provider == "together":
                api_params["top_k"] = 20              # Limit vocabulary per step for conciseness
                api_params["presence_penalty"] = 0.8  # Reduce repetitive reasoning patterns

                # Add reasoning_effort only for GPT-OSS models
                if "gpt-oss" in self.model_name.lower():
                    api_params["reasoning_effort"] = self.reasoning_effort

            if self.provider == "bedrock":
                response = self.litellm.completion(**api_params)
            else:
                response = self.client.chat.completions.create(**api_params)

            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                self.tokens_used += response.usage.total_tokens
                self.input_tokens += getattr(response.usage, 'prompt_tokens', 0) or 0
                self.output_tokens += getattr(response.usage, 'completion_tokens', 0) or 0

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Extract reasoning/thinking content (per-provider)
            reasoning_content = self._extract_reasoning(assistant_message)

            # Debug: log available fields when verbose (helps diagnose missing reasoning)
            if self.verbose and not reasoning_content:
                fields = {k: type(v).__name__ for k, v in vars(assistant_message).items() if v is not None}
                content_preview = str(assistant_message.content)[:200] if assistant_message.content else "None"
                print(f"[DEBUG] No reasoning found. Fields: {fields}, content preview: {content_preview}")

            if reasoning_content and self.output_callback:
                self.output_callback("agent_thinking", {
                    "content": reasoning_content
                })

            # Check if agent wants to use tools
            if not assistant_message.tool_calls:
                # No tool call — agent is done, break out
                if self.output_callback:
                    self.output_callback("agent_thinking", {
                        "content": assistant_message.content
                    })
                elif self.verbose:
                    print(f"[Text-only turn] {assistant_message.content[:200]}")
                break

            # Execute tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # Callback: Tool call
                if self.output_callback:
                    self.output_callback("tool_call", {
                        "tool": tool_name,
                        "arguments": arguments
                    })
                elif self.verbose:
                    print(f"Tool: {tool_name}")
                    print(f"Args: {json.dumps(arguments, indent=2)}")

                # Execute tool
                previous_doc_mode = self._current_doc_exploration_doc_id
                if doc_id:
                    self._current_doc_exploration_doc_id = doc_id
                try:
                    result = self._execute_tool(tool_name, arguments)
                finally:
                    self._current_doc_exploration_doc_id = previous_doc_mode

                # Callback: Tool result
                if self.output_callback:
                    self.output_callback("tool_result", {
                        "tool": tool_name,
                        "result": result,
                        "is_error": isinstance(result, dict) and 'error' in result
                    })
                elif self.verbose:
                    print(f"Result: {str(result)}")

                # Track tool calls
                tool_calls_made.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })

                # Track bridge creation (count actual successes, handle batch)
                if tool_name == "create_bridge_qa":
                    if isinstance(result, list):
                        bridges_created += sum(1 for r in result if isinstance(r, dict) and r.get("success"))
                    elif isinstance(result, dict) and result.get("success"):
                        bridges_created += 1

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })

        # Mark entity/document as explored
        if entity_name:
            self.tools.mark_entity_explored(entity_name, bridges_created)
        elif doc_id:
            # Only auto-mark document when exploration plan is complete.
            doc_plan = self.tools.doc_exploration_plans.get(doc_id)
            pending_edges = int(doc_plan.get("pending_count", 0)) if isinstance(doc_plan, dict) else 0
            if (
                doc_id not in self.tools.explored_documents
                and self.tools.active_neighbor_focus is None
                and pending_edges == 0
            ):
                self.tools.mark_document_explored(doc_id, bridges_created)
            elif pending_edges > 0:
                logger.info(
                    "Skipping auto mark_document_explored for %s due to pending edges=%s",
                    doc_id,
                    pending_edges,
                )
        self.entities_explored += 1

        return {
            "entity": label,
            "iterations": iteration + 1,
            "tool_calls": len(tool_calls_made),
            "bridges_created": bridges_created,
            "tool_call_trace": tool_calls_made
        }

    def run_exploration(self, num_entities: Optional[int] = None) -> Dict[str, Any]:
        """
        Run autonomous exploration on multiple entities.

        Args:
            num_entities: Number of entities to explore (defaults to budget["max_entities"])

        Returns:
            Summary of entire exploration run
        """
        if self.pipeline_mode == "rlm":
            return self.run_rlm_corpus()
        if self.pipeline_mode == "hybrid":
            return self.run_hybrid_corpus()

        if num_entities is None:
            num_entities = self.budget.get("max_entities", 20)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting sleep-time exploration")
            print(f"Target: {num_entities} entities")
            print(f"Budget: {self.budget}")
            print('='*80)

        exploration_results = []

        for i in range(num_entities):
            # Check budget
            if "max_tokens" in self.budget and self.tokens_used >= self.budget["max_tokens"]:
                if self.verbose:
                    print(f"\n[Budget exhausted: {self.tokens_used} tokens used]")
                break

            # Get next entity to explore
            entity_name = self.tools.suggest_next_entity(strategy="high_degree")
            if not entity_name:
                if self.verbose:
                    print("\n[No more entities to explore]")
                break

            # Explore entity
            result = self.explore_entity(entity_name)
            exploration_results.append(result)

            if self.verbose:
                stats = self.tools.get_bridge_statistics()
                print(f"\nProgress: {i+1}/{num_entities} entities | {stats['total_bridges']} bridges | {self.tokens_used:,} tokens")

        # Final stats
        final_stats = self.tools.get_bridge_statistics()

        return {
            "entities_explored": len(exploration_results),
            "total_bridges": final_stats["total_bridges"],
            "tokens_used": self.tokens_used,
            "avg_confidence": final_stats["avg_confidence"],
            "exploration_results": exploration_results,
            "final_stats": final_stats
        }

    def get_all_bridges(self) -> List[Dict[str, Any]]:
        """Get all bridges created during exploration."""
        return self.tools.bridges_created
