"""
Agentic reconciler for sleep-time GSW exploration.

Uses Together AI's models to drive autonomous exploration and bridge creation.
"""

import json
import os
import logging
import traceback
from typing import List, Dict, Any, Optional

from .tools import GSWTools
from .prompts import SLEEP_TIME_SYSTEM_PROMPT

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
        reasoning_effort: str = "high"
    ):
        """
        Initialize agentic reconciler.

        Args:
            entity_searcher: Initialized EntitySearcher with loaded GSWs
            model_name: Model name - OpenAI (gpt-4o, gpt-4o-mini) or Together AI (openai/gpt-oss-120b, meta-llama/...)
            budget: Dict with "max_entities" and/or "max_tokens" limits
            verbose: Print progress messages
            output_callback: Optional callback function for interactive display
                            Format: callback(event_type: str, data: Dict[str, Any])
            reasoning_effort: Reasoning effort for Together AI models: "low", "medium", or "high" (default: "high")
        """
        self.entity_searcher = entity_searcher
        self.tools = GSWTools(entity_searcher)
        self.model_name = model_name
        self.verbose = verbose
        self.output_callback = output_callback  # Callback for interactive display
        self.reasoning_effort = reasoning_effort

        # Budget tracking
        self.budget = budget or {"max_entities": 20, "max_tokens": 1_000_000}
        self.tokens_used = 0
        self.entities_explored = 0

        # Safety: Cache valid document IDs for reference
        self._valid_doc_ids = set(entity_searcher.gsw_by_doc_id.keys())

        if self.verbose and not output_callback:  # Don't log if using callback (callback handles display)
            logger.info(f"Valid document range: doc_0 to doc_{max([int(d.split('_')[1]) for d in self._valid_doc_ids if d.startswith('doc_')])}")

        # Initialize appropriate client based on model name
        self._initialize_client(model_name)

        # Tool definitions in OpenAI format
        self.tool_definitions = self._create_tool_definitions()

        if self.verbose:
            print(f"✓ Initialized AgenticReconciler")
            print(f"  Model: {model_name}")
            print(f"  Provider: {self.provider}")
            print(f"  Budget: {self.budget}")
            print(f"  GSWs loaded: {len(self.entity_searcher.gsw_by_doc_id)}")

    def _initialize_client(self, model_name: str):
        """
        Initialize the appropriate LLM client based on model name.

        OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
        Together AI models: Contains '/' like openai/gpt-oss-120b, meta-llama/...
        """
        # Detect provider
        if model_name.startswith("gpt-"):
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
                "Expected OpenAI model (gpt-4o, gpt-4o-mini) or Together AI model (contains '/')"
            )

    def _create_tool_definitions(self) -> List[Dict[str, Any]]:
        """Create OpenAI-format tool definitions."""
        return [
            # Discovery tools
            {
                "type": "function",
                "function": {
                    "name": "get_entity_documents",
                    "description": "Get list of document IDs that mention this entity.",
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
                    "name": "get_document_entities",
                    "description": "Get list of entities mentioned in this document.",
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
            # Context tools
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
            {
                "type": "function",
                "function": {
                    "name": "reconcile_entity_across_docs",
                    "description": "Merge all information about an entity from all documents into unified view. Use this to see complete picture of an entity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity to reconcile"
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            # Bridges
            {
                "type": "function",
                "function": {
                    "name": "create_bridge_qa",
                    "description": "Create one or more bridge QA pairs. Can be used in two modes: (1) Single bridge: pass question, answer, source_docs, reasoning. (2) Multiple bridges (1-5): pass bridges array. Validation is performed automatically.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Bridge question (single bridge mode)"
                            },
                            "answer": {
                                "type": "string",
                                "description": "Bridge answer (single bridge mode)"
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
                                "description": "Array of bridge specifications for batch creation (multiple bridges mode). Each bridge must have: question, answer, source_docs, reasoning. Optional: confidence, entities_involved. Maximum 5 bridges per call.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "answer": {"type": "string"},
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
                                    "required": ["question", "answer", "source_docs", "reasoning"]
                                },
                                "minItems": 1,
                                "maxItems": 5
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
            # Strategy
            {
                "type": "function",
                "function": {
                    "name": "mark_entity_explored",
                    "description": "Mark an entity as explored (so suggest_next_entity won't suggest it again).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity that was explored"
                            },
                            "num_bridges_created": {
                                "type": "integer",
                                "description": "Number of bridges created for this entity",
                                "default": 0
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            # Exploration tracking tools
            {
                "type": "function",
                "function": {
                    "name": "plan_entity_exploration",
                    "description": "Create exploration plan for entity showing all relationships to check. Call ONCE after reconcile_entity_across_docs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity being explored"
                            },
                            "relationships": {
                                "type": "object",
                                "description": "merged_relationships dict from reconcile_entity_across_docs output",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "required": ["entity_name", "relationships"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_relationship_explored",
                    "description": "Mark one or more relationships as explored after checking all their documents. Supports batch mode for marking multiple at once. Returns updated checklist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Main entity being explored"
                            },
                            "relationship_name": {
                                "description": "Single entity name (string) or list of entity names (array) just explored",
                                "oneOf": [
                                    {
                                        "type": "string",
                                        "description": "Single relationship name"
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of relationship names for batch marking"
                                    }
                                ]
                            },
                            "bridges_created": {
                                "description": "Number of bridges created. Can be single integer (for one relationship or same count for all) or list of integers (individual counts for each relationship)",
                                "oneOf": [
                                    {
                                        "type": "integer",
                                        "description": "Single count (for one relationship or same count for all)",
                                        "default": 0
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "List of counts (one per relationship in same order)"
                                    }
                                ],
                                "default": 0
                            }
                        },
                        "required": ["entity_name", "relationship_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_exploration_status",
                    "description": "Check which relationships explored vs pending. Use before mark_entity_explored to verify completeness.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Entity to check status for"
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            }
        ]

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

        # Execute the tool with error handling
        try:
            result = tool_method(**arguments)
            logger.debug(f"Tool {tool_name} executed successfully")

            # Callback for bridge creation success
            if tool_name == "create_bridge_qa" and self.output_callback:
                if isinstance(result, dict) and result.get("success"):
                    self.output_callback("bridge_created", {
                        "bridge_id": result.get("bridge_id"),
                        "question": arguments.get("question", ""),
                        "answer": arguments.get("answer", ""),
                        "validation": result.get("validation", {})
                    })
                elif isinstance(result, dict) and not result.get("success"):
                    # Validation failed during bridge creation
                    self.output_callback("validation", result.get("validation", {}))

            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tool": tool_name,
                "arguments": arguments
            }

    def explore_entity(self, entity_name: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Let agent explore a single entity and create bridges.

        Args:
            entity_name: Entity to explore
            max_iterations: Max tool calling rounds

        Returns:
            Summary of exploration (bridges created, tool calls, etc.)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Exploring entity: {entity_name}")
            print('='*80)

        messages = [
            {"role": "system", "content": SLEEP_TIME_SYSTEM_PROMPT},
            {"role": "user", "content": f"Explore the entity '{entity_name}' and create any useful bridge QA pairs you find by combining information across documents."}
        ]

        tool_calls_made = []
        bridges_created = 0

        for iteration in range(max_iterations):
            # Callback: Iteration start
            if self.output_callback:
                self.output_callback("iteration_start", {
                    "entity": entity_name,
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
                "temperature": 0.6,          # Focused reasoning (was 0.7)
                "top_p": 0.95,               # Reduce long-tail tokens (was 1.0)
                "top_k": 20,                 # Limit vocabulary per step for conciseness
                "presence_penalty": 0.8,     # Reduce repetitive reasoning patterns
                "max_tokens": 32768          # API limit for most Together AI models
            }

            # Add reasoning_effort only for GPT-OSS models (not all Together AI models support it)
            if self.provider == "together" and "gpt-oss" in self.model_name.lower():
                api_params["reasoning_effort"] = self.reasoning_effort

            response = self.client.chat.completions.create(**api_params)

            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                self.tokens_used += response.usage.total_tokens

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Callback: Agent thinking (chain-of-thought reasoning)
            # For GPT-OSS models, reasoning is in a separate 'reasoning' field
            reasoning_content = None
            if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
                reasoning_content = assistant_message.reasoning
            elif assistant_message.content:
                reasoning_content = assistant_message.content

            if reasoning_content and self.output_callback:
                self.output_callback("agent_thinking", {
                    "content": reasoning_content
                })

            # Check if agent wants to use tools
            if not assistant_message.tool_calls:
                # Agent is done
                if self.output_callback:
                    self.output_callback("agent_finished", {
                        "content": assistant_message.content
                    })
                elif self.verbose:
                    print(f"Agent finished: {assistant_message.content}")
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
                result = self._execute_tool(tool_name, arguments)

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

                # Track bridge creation
                if tool_name == "create_bridge_qa":
                    bridges_created += 1

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })

        # Mark entity as explored
        self.tools.mark_entity_explored(entity_name, bridges_created)
        self.entities_explored += 1

        return {
            "entity": entity_name,
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
