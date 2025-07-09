"""
Narrative Q&A system with agentic tool calling capabilities.

This module provides the GSWQuestionAnswerer_Narrative class that uses
an agentic approach for multi-hop narrative question answering with
entity, verb phrase, and conversation summaries.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..memory.aggregators import EntitySummaryAggregator
from ..memory.models import EntityNode, GSWStructure
from .entity_extractor import QuestionEntityExtractor
from .matcher import EntityMatcher
from .reranker import SummaryReranker
from .narrative_tools import NarrativeQATools
from ..prompts.narrative_qa_prompts import NarrativeQAPrompts


class GSWQuestionAnswerer_Narrative:
    """
    Agentic Q&A system for GSW Memory with tool calling capabilities.

    This system uses an agentic approach where an LLM agent can iteratively
    call tools to search for entity, verb phrase, and conversation summaries,
    as well as retrieve original text chunks to answer complex narrative questions.
    """

    def __init__(
        self,
        gsw: Union[GSWStructure, List[GSWStructure]],
        entity_aggregator: Union[
            EntitySummaryAggregator, List[EntitySummaryAggregator]
        ],
        llm_config: Dict[str, Any],
        include_verb_phrases: bool = False,
        include_conversations: bool = False,
        verb_aggregator: Optional[Union[Any, List[Any]]] = None,
        conversation_aggregator: Optional[Union[Any, List[Any]]] = None,
        embedding_model: str = "voyage-3",
        openai_api_key: Optional[str] = None,
        chunks_folder_path: Optional[str] = None,
        initial_context_size: int = 5,
        reranker_size: int = 5,
        context_retrieval_size: int = 20,
        tool_result_limit: int = 5,
        max_turns: int = 10,
        verbose: bool = True,
        debug_log_file: Optional[str] = None,
    ):
        """
        Initialize the narrative Q&A system.

        Args:
            gsw: GSW structure(s) to query
            entity_aggregator: Entity summary aggregators
            llm_config: LLM configuration for entity extraction and agent
            include_verb_phrases: Whether to include verb phrase summaries
            include_conversations: Whether to include conversation summaries
            verb_aggregator: Verb phrase summary aggregators (required if include_verb_phrases=True)
            conversation_aggregator: Conversation summary aggregators (required if include_conversations=True)
            embedding_model: Embedding model name for reranking
            openai_api_key: OpenAI API key for agent (if None, uses environment variable)
            chunks_folder_path: Path to folder containing chunk text files (e.g., /path/to/chunks)
            initial_context_size: Number of summaries to include in initial context
            reranker_size: Number of summaries to return after reranking (for tools)
            context_retrieval_size: Number of summaries to retrieve before reranking
            tool_result_limit: Default k for tool calls
            max_turns: Maximum agent turns before stopping
            debug_log_file: Optional path to save detailed agent activity logs for debugging
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required for narrative Q&A. Install with: pip install openai")

        # Handle backward compatibility: convert single inputs to lists
        if isinstance(gsw, GSWStructure):
            self.gsws = [gsw]
        else:
            self.gsws = gsw

        if isinstance(entity_aggregator, EntitySummaryAggregator):
            self.entity_aggregators = [entity_aggregator]
        else:
            self.entity_aggregators = entity_aggregator

        # Handle verb phrase aggregators
        self.include_verb_phrases = include_verb_phrases
        if include_verb_phrases:
            if verb_aggregator is None:
                raise ValueError("verb_aggregator must be provided when include_verb_phrases=True")
            if not isinstance(verb_aggregator, list):
                self.verb_aggregators = [verb_aggregator]
            else:
                self.verb_aggregators = verb_aggregator
        else:
            self.verb_aggregators = []

        # Handle conversation aggregators
        self.include_conversations = include_conversations
        if include_conversations:
            if conversation_aggregator is None:
                raise ValueError("conversation_aggregator must be provided when include_conversations=True")
            if not isinstance(conversation_aggregator, list):
                self.conversation_aggregators = [conversation_aggregator]
            else:
                self.conversation_aggregators = conversation_aggregator
        else:
            self.conversation_aggregators = []

        # Validate that all aggregator lists match GSW count
        if len(self.gsws) != len(self.entity_aggregators):
            raise ValueError(f"Number of GSWs ({len(self.gsws)}) must match number of entity aggregators ({len(self.entity_aggregators)})")
        
        if self.include_verb_phrases and len(self.gsws) != len(self.verb_aggregators):
            raise ValueError(f"Number of GSWs ({len(self.gsws)}) must match number of verb aggregators ({len(self.verb_aggregators)})")
        
        if self.include_conversations and len(self.gsws) != len(self.conversation_aggregators):
            raise ValueError(f"Number of GSWs ({len(self.gsws)}) must match number of conversation aggregators ({len(self.conversation_aggregators)})")

        self.llm_config = llm_config
        self.embedding_model = embedding_model
        self.initial_context_size = initial_context_size
        self.reranker_size = reranker_size
        self.context_retrieval_size = context_retrieval_size
        self.tool_result_limit = tool_result_limit
        self.max_turns = max_turns
        self.verbose = verbose
        self.debug_log_file = debug_log_file

        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var

        # Initialize components
        self.entity_extractor = QuestionEntityExtractor(
            model_name=llm_config["model_name"],
            generation_params=llm_config.get("generation_params", {"temperature": 0.0}),
        )
        self.entity_matcher = EntityMatcher()
        self.summary_reranker = SummaryReranker(embedding_model)

        # Initialize tools with new parameters
        self.tools = NarrativeQATools(
            gsws=self.gsws,
            entity_aggregators=self.entity_aggregators,
            verb_aggregators=self.verb_aggregators,
            conversation_aggregators=self.conversation_aggregators,
            entity_extractor=self.entity_extractor,
            entity_matcher=self.entity_matcher,
            summary_reranker=self.summary_reranker,
            chunks_folder_path=chunks_folder_path,
            include_verb_phrases=self.include_verb_phrases,
            include_conversations=self.include_conversations,
            tool_result_limit=self.reranker_size,  # Use reranker_size for tool results
            context_retrieval_size=self.context_retrieval_size,
        )
        
        # Set up activity logging
        self.activity_log: List[Dict[str, Any]] = []
        self.tools.set_activity_logger(self._log_activity)

    def ask_narrative(self, question: str) -> Dict[str, Any]:
        """
        Ask a single question using the agentic approach.

        Args:
            question: The user's question

        Returns:
            Dict with final answer, activity log, and metadata
        """
        results = self.ask_narrative_batch([question])
        return results[0]

    def ask_narrative_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Ask multiple questions using the agentic approach.

        Note: Each question runs independently with its own agent session.

        Args:
            questions: List of questions to ask

        Returns:
            List of result dicts, one per question
        """
        results = []
        for question in questions:
            result = self._run_single_agent_session(question)
            results.append(result)
        return results

    def _run_single_agent_session(self, question: str) -> Dict[str, Any]:
        """
        Run a single agent session for one question.

        Returns:
            Dict with final answer, activity log, and metadata
        """
        # Reset activity log for this session
        self.activity_log = []
        self._log_activity("session_start", "Starting new agent session", {"question": question}, None)

        # Step 1: Build initial context
        initial_context = self._build_initial_context(question)
        
        # Log the complete initial setup
        initial_setup_log = {
            "question": question,
            "initial_context_length": len(initial_context),
            "context_sections": initial_context.count("==="),
            "system_config": {
                "max_turns": self.max_turns,
                "initial_context_size": self.initial_context_size,
                "context_retrieval_size": self.context_retrieval_size,
                "include_verb_phrases": self.include_verb_phrases,
                "include_conversations": self.include_conversations
            }
        }
        self._log_activity("session_setup", "Completed initial setup for agent session", initial_setup_log, {"initial_context": initial_context})
        
        # Step 2: Run agent loop
        final_answer = self._run_agent_loop(question, initial_context)

        result = {
            "question": question,
            "final_answer": final_answer,
            "activity_log": self.activity_log.copy(),
            "total_turns": len([log for log in self.activity_log if log["tool"] in ["search_entity_by_name", "search_entity_summaries", "search_vp_summaries", "search_conversation_summaries", "search_chunks"]]),
        }
        
        # Save debug log if requested
        if self.debug_log_file:
            self._save_debug_log(result)
        
        return result

    def _build_initial_context(self, question: str) -> str:
        """
        Build initial context from entity, verb phrase, and conversation summaries.

        Args:
            question: The user's question

        Returns:
            Formatted initial context string
        """
        self._log_activity("build_initial_context", "Building initial context for question", {"question": question}, None)

        context_sections = []

        # Extract entities and get entity summaries
        extracted_entities = self.entity_extractor.extract_entities(question)
        entity_extraction_log = {
            "question": question,
            "entities_found": len(extracted_entities),
            "entities_list": extracted_entities,
            "extraction_successful": True
        }
        self._log_activity("extract_entities", "Extracted entities from question", entity_extraction_log, extracted_entities)
        
        if self.verbose:
            if extracted_entities:
                print(f"🔍 Extracted entities: {', '.join(extracted_entities)}")
            else:
                print("🔍 No specific entities extracted from question")

        if extracted_entities:
            entity_matches_with_source = self.tools.find_matching_entities(extracted_entities)
            entity_summaries = self.tools.get_entity_summaries(entity_matches_with_source)
            
            # Log entity matching results
            entity_matching_log = {
                "extracted_entities": extracted_entities,
                "entities_matched": len(entity_matches_with_source),
                "matched_entities": [{"name": entity.name, "id": entity.id, "source_gsw": source} for entity, source in entity_matches_with_source],
                "summaries_retrieved": len(entity_summaries),
                "summary_lengths": [len(summary) for _, summary in entity_summaries]
            }
            self._log_activity("entity_matching", "Matched entities and retrieved summaries", entity_matching_log, {
                "matched_entities": [{"name": entity.name, "id": entity.id} for entity, _ in entity_matches_with_source],
                "summaries": [{"entity": name, "summary": summary[:200] + "..." if len(summary) > 200 else summary} for name, summary in entity_summaries]
            })
            
            # Use context_retrieval_size for initial retrieval, then rerank to initial_context_size
            if entity_summaries:
                initial_k = min(len(entity_summaries), self.context_retrieval_size)
                ranked_entity_summaries = self.summary_reranker.rerank_summaries(
                    entity_summaries, question, initial_k
                )
                final_entity_summaries = ranked_entity_summaries[:self.initial_context_size]
                
                # Log entity reranking results
                entity_reranking_log = {
                    "initial_summaries": len(entity_summaries),
                    "retrieval_k": initial_k,
                    "final_context_size": self.initial_context_size,
                    "final_summaries_count": len(final_entity_summaries),
                    "score_range": [min([score for _, _, score in ranked_entity_summaries]), max([score for _, _, score in ranked_entity_summaries])] if ranked_entity_summaries else [0, 0],
                    "selected_entities": [{"entity": name, "score": score} for name, _, score in final_entity_summaries]
                }
                self._log_activity("entity_reranking", f"Reranked {len(entity_summaries)} entity summaries, selected top {len(final_entity_summaries)}", entity_reranking_log, {
                    "ranked_summaries": [{"entity": name, "summary": summary[:150] + "...", "score": score} for name, summary, score in final_entity_summaries]
                })
                
                if final_entity_summaries:
                    context_sections.append("=== Entity Summaries ===")
                    for entity_name, summary, score in final_entity_summaries:
                        # Find the entity ID for this summary
                        entity_id = self._find_entity_id_for_summary(entity_name, summary)
                        context_sections.append(f"Entity: {entity_name} ({entity_id})")
                        context_sections.append(f"Summary: {summary}")
                        context_sections.append("")

        # Get verb phrase summaries if enabled (now with consistent reranking)
        if self.include_verb_phrases:
            # Get all VP summaries first
            all_vp_summaries = []
            vp_summaries_cache = self.tools._get_vp_summaries_cache()
            
            for cache_key, summary_data in vp_summaries_cache.items():
                gsw_index, vp_id = cache_key.split("_", 1)
                summary = summary_data["summary"]
                all_vp_summaries.append((vp_id, summary))
            
            # Rerank consistently like entities
            if all_vp_summaries:
                initial_k = min(len(all_vp_summaries), self.context_retrieval_size)
                ranked_vp_summaries = self.summary_reranker.rerank_summaries(
                    all_vp_summaries, question, initial_k
                )
                final_vp_summaries = ranked_vp_summaries[:self.initial_context_size]
                
                if final_vp_summaries:
                    context_sections.append("=== Verb Phrase Summaries ===")
                    for vp_id, summary, score in final_vp_summaries:
                        context_sections.append(f"VP: {vp_id}")
                        context_sections.append(f"Summary: {summary}")
                        context_sections.append("")

        # Get conversation summaries if enabled (now with consistent reranking)
        if self.include_conversations:
            # Get all conversation summaries first
            all_conv_summaries = []
            conv_summaries_cache = self.tools._get_conversation_summaries_cache()
            
            for cache_key, summary_data in conv_summaries_cache.items():
                gsw_index, conv_id = cache_key.split("_", 1)
                summary = summary_data["summary"]
                all_conv_summaries.append((conv_id, summary))
            
            # Rerank consistently like entities
            if all_conv_summaries:
                initial_k = min(len(all_conv_summaries), self.context_retrieval_size)
                ranked_conv_summaries = self.summary_reranker.rerank_summaries(
                    all_conv_summaries, question, initial_k
                )
                final_conv_summaries = ranked_conv_summaries[:self.initial_context_size]
                
                if final_conv_summaries:
                    context_sections.append("=== Conversation Summaries ===")
                    for conv_id, summary, score in final_conv_summaries:
                        context_sections.append(f"Conversation: {conv_id}")
                        context_sections.append(f"Summary: {summary}")
                        context_sections.append("")

        initial_context = "\n".join(context_sections)
        self._log_activity("initial_context_built", "Built initial context", {"context_length": len(initial_context)}, initial_context[:500] + "..." if len(initial_context) > 500 else initial_context)
        
        if self.verbose:
            context_stats = []
            if "=== Entity Summaries ===" in initial_context:
                entity_count = initial_context.count("Entity:")
                context_stats.append(f"{entity_count} entities")
            if "=== Verb Phrase Summaries ===" in initial_context:
                vp_count = initial_context.count("VP:")
                context_stats.append(f"{vp_count} verb phrases")
            if "=== Conversation Summaries ===" in initial_context:
                conv_count = initial_context.count("Conversation:")
                context_stats.append(f"{conv_count} conversations")
            
            if context_stats:
                print(f"📋 Built initial context with: {', '.join(context_stats)}")
            else:
                print("📋 Built initial context (no specific entities found)")
        
        return initial_context

    def _run_agent_loop(self, question: str, initial_context: str) -> Optional[Dict[str, Any]]:
        """
        Run the main agent loop with tool calling.

        Args:
            question: The user's question
            initial_context: Initial context to provide to agent

        Returns:
            Final answer dict or None if max turns reached
        """
        # Build agent tools
        tools = self.tools.build_agent_tools()

        # Get system prompt
        tools_available = [tool["function"]["name"] for tool in tools]
        system_prompt = NarrativeQAPrompts.get_system_prompt(tools_available)

        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nInitial Context:\n{initial_context}"
            }
        ]
        
        # Log conversation initialization
        conversation_init_log = {
            "tools_available": tools_available,
            "system_prompt_length": len(system_prompt),
            "user_message_length": len(messages[1]["content"]),
            "initial_message_count": len(messages)
        }
        self._log_activity("conversation_init", "Initialized conversation with system prompt and initial context", conversation_init_log, {
            "tools_available": tools_available,
            "system_prompt": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
            "initial_user_message": messages[1]["content"][:500] + "..." if len(messages[1]["content"]) > 500 else messages[1]["content"]
        })

        # Agent loop
        for turn in range(1, self.max_turns + 1):
            # Log turn start with current conversation state
            turn_start_log = {
                "turn": turn,
                "conversation_length": len(messages),
                "conversation_state": [{"role": msg["role"], "has_content": bool(msg.get("content")), "has_tool_calls": bool(msg.get("tool_calls"))} for msg in messages]
            }
            self._log_activity("agent_turn_start", f"Starting agent turn {turn}", turn_start_log, None)
            
            if self.verbose:
                print(f"\n🤖 Agent Turn {turn}/{self.max_turns}")
                print("💭 Agent is thinking...")

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_config["model_name"],
                    temperature=self.llm_config.get("generation_params", {}).get("temperature", 0.0),
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                response_msg = response.choices[0].message
                
                # Log the agent's raw response for detailed debugging
                agent_response_log = {
                    "turn": turn,
                    "has_content": bool(response_msg.content),
                    "content": response_msg.content,
                    "has_tool_calls": bool(response_msg.tool_calls),
                    "tool_calls_count": len(response_msg.tool_calls) if response_msg.tool_calls else 0,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None
                }
                self._log_activity("agent_response", f"Agent response in turn {turn}", agent_response_log, {
                    "content": response_msg.content,
                    "tool_calls": [{"function": call.function.name, "arguments": call.function.arguments} for call in response_msg.tool_calls] if response_msg.tool_calls else []
                })

                # Check if agent wants to call tools
                if response_msg.tool_calls:
                    if self.verbose:
                        print(f"🔧 Agent decided to use {len(response_msg.tool_calls)} tool(s)")
                    
                    # Log agent's decision to use tools with reasoning
                    tools_decision_log = {
                        "turn": turn,
                        "agent_thought": response_msg.content,  # Agent's reasoning if any
                        "num_tools": len(response_msg.tool_calls),
                        "tools_planned": [{"function": call.function.name, "arguments": call.function.arguments} for call in response_msg.tool_calls]
                    }
                    self._log_activity("agent_tools_decision", f"Agent decided to use {len(response_msg.tool_calls)} tools in turn {turn}", tools_decision_log, response_msg.tool_calls)
                    
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": response_msg.content,  # Include any reasoning content
                        "tool_calls": response_msg.tool_calls
                    })

                    # Execute each tool call with detailed logging
                    for i, call in enumerate(response_msg.tool_calls):
                        fn_name = call.function.name
                        fn_args = json.loads(call.function.arguments)
                        
                        # Log tool call details before execution
                        tool_call_log = {
                            "turn": turn,
                            "tool_index": i + 1,
                            "tool_name": fn_name,
                            "tool_call_id": call.id,
                            "raw_arguments": call.function.arguments,
                            "parsed_arguments": fn_args,
                            "reasoning": fn_args.get("reasoning", "No reasoning provided"),
                            "query": fn_args.get("query", fn_args.get("chunk_ids", "")),
                            "k_requested": fn_args.get("k", self.tool_result_limit)
                        }
                        self._log_activity("tool_call_start", f"Executing tool {fn_name} (call {i+1}/{len(response_msg.tool_calls)}) in turn {turn}", tool_call_log, None)
                        
                        if self.verbose:
                            reasoning = fn_args.get("reasoning", "No reasoning provided")
                            query = fn_args.get("query", fn_args.get("chunk_ids", ""))
                            k = fn_args.get("k", "")
                            
                            print(f"   🛠️  Tool {i+1}: {fn_name}")
                            print(f"       Query: {query}")
                            if k:
                                print(f"       Results requested: {k}")
                            print(f"       Reasoning: {reasoning}")

                        # Execute the tool using our tools class
                        tool_result = self.tools.execute_tool(fn_name, fn_args)
                        
                        # Log detailed tool results
                        tool_result_log = {
                            "turn": turn,
                            "tool_index": i + 1,
                            "tool_name": fn_name,
                            "tool_call_id": call.id,
                            "result_count": len(tool_result) if isinstance(tool_result, list) else 1,
                            "result_types": [type(item).__name__ for item in tool_result] if isinstance(tool_result, list) else [type(tool_result).__name__],
                            "execution_successful": True
                        }
                        self._log_activity("tool_call_result", f"Tool {fn_name} returned {len(tool_result) if isinstance(tool_result, list) else 1} results in turn {turn}", tool_result_log, tool_result)
                        
                        if self.verbose:
                            result_count = len(tool_result) if isinstance(tool_result, list) else 1
                            
                            # Extract IDs from results for display
                            ids_found = []
                            if isinstance(tool_result, list) and tool_result:
                                for result in tool_result[:5]:  # Show first 5 IDs
                                    if isinstance(result, dict):
                                        # Different tools return different ID fields
                                        if "entity_id" in result:
                                            ids_found.append(result["entity_id"])
                                        elif "vp_id" in result:
                                            ids_found.append(result["vp_id"])
                                        elif "conversation_id" in result:
                                            ids_found.append(result["conversation_id"])
                                        elif "chunk_id" in result:
                                            ids_found.append(result["chunk_id"])
                            
                            # Display results with IDs
                            if ids_found:
                                ids_display = ", ".join(ids_found)  # Show all IDs
                                print(f"       ✅ Found {result_count} result(s): {ids_display}")
                            else:
                                print(f"       ✅ Found {result_count} result(s)")

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result)
                        })

                    # Compress older tool results to manage conversation length
                    self._compress_older_tool_results(messages)
                    continue

                # No tool calls - expect final answer
                if self.verbose:
                    print("📝 Agent is providing final answer...")
                
                # Log final answer attempt with full details
                final_answer_attempt_log = {
                    "turn": turn,
                    "raw_content": response_msg.content,
                    "content_length": len(response_msg.content) if response_msg.content else 0,
                    "appears_to_be_json": response_msg.content.strip().startswith("{") if response_msg.content else False,
                    "contains_json_block": "```json" in response_msg.content if response_msg.content else False,
                    "finish_reason": response.choices[0].finish_reason
                }
                self._log_activity("final_answer_attempt", f"Agent attempting to provide final answer in turn {turn}", final_answer_attempt_log, response_msg.content)
                
                try:
                    response_content = response_msg.content
                    if response_content.strip().startswith("{"):
                        # JSON response
                        final_answer = json.loads(response_content)
                        parse_method = "direct_json"
                    else:
                        # Try to extract JSON from response
                        if "```json" in response_content:
                            json_content = response_content.split("```json")[1].split("```")[0].strip()
                            final_answer = json.loads(json_content)
                            parse_method = "json_block"
                        else:
                            # Plain text response - wrap it
                            final_answer = {"answer": response_content}
                            parse_method = "plain_text_wrapped"

                    # Log successful final answer parsing
                    final_answer_success_log = {
                        "turn": turn,
                        "parse_method": parse_method,
                        "answer_structure": list(final_answer.keys()) if isinstance(final_answer, dict) else type(final_answer).__name__,
                        "answer_length": len(str(final_answer.get("answer", final_answer))),
                        "has_sources": "sources" in final_answer if isinstance(final_answer, dict) else False,
                        "has_reasoning": "reasoning" in final_answer if isinstance(final_answer, dict) else False
                    }
                    self._log_activity("final_answer_success", f"Successfully parsed final answer in turn {turn}", final_answer_success_log, final_answer)

                    if self.verbose:
                        answer_preview = str(final_answer.get("answer", final_answer))[:100]
                        print(f"✅ Final answer ready: {answer_preview}{'...' if len(str(final_answer.get('answer', final_answer))) > 100 else ''}")
                    
                    return final_answer

                except (json.JSONDecodeError, Exception) as e:
                    if self.verbose:
                        print(f"⚠️  Could not parse final answer, agent will try again...")
                    self._log_activity("parse_error", f"Failed to parse final answer: {e}", {"turn": turn, "response": response_content}, None)
                    # Continue the loop to give agent another chance

            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in turn {turn}: {e}")
                self._log_activity("agent_error", f"Error in agent turn {turn}: {e}", {"turn": turn}, None)
                break

        # Max turns reached
        if self.verbose:
            print(f"⏰ Agent reached maximum turns ({self.max_turns}) without providing final answer")
        
        # Log final conversation state when max turns reached
        final_conversation_log = {
            "max_turns": self.max_turns,
            "final_conversation_length": len(messages),
            "message_roles": [msg["role"] for msg in messages],
            "last_message_role": messages[-1]["role"] if messages else None,
            "completion_status": "max_turns_reached"
        }
        self._log_activity("max_turns_reached", "Agent reached maximum turns without final answer", final_conversation_log, {"final_conversation": messages})
        return None

    def _find_entity_id_for_summary(self, entity_name: str, summary: str) -> str:
        """Find entity ID that matches the given name and summary."""
        entity_summaries_cache = self.tools._get_entity_summaries_cache()
        for cache_key, summary_data in entity_summaries_cache.items():
            if (summary_data["summary"] == summary and 
                summary_data.get("entity_name", "") == entity_name):
                gsw_index, entity_id = cache_key.split("_", 1)
                return entity_id
        return "unknown"

    def _compress_older_tool_results(self, messages: List[Dict[str, Any]], keep_recent: int = 2) -> None:
        """Compress older tool results to manage conversation length while preserving initial context."""
        # Find tool result messages (but never compress the initial context)
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg["role"] == "tool":
                tool_result_indices.append(i)
        
        # Compress older tool results (keep initial context safe)
        if len(tool_result_indices) > keep_recent:
            indices_to_compress = tool_result_indices[:-keep_recent]
            for idx in indices_to_compress:
                # Never compress anything from the first few messages (system + initial user message)
                if idx <= 2:  # Skip system message (0) and initial user message with context (1)
                    continue
                    
                tool_msg = messages[idx]
                try:
                    full_result = json.loads(tool_msg["content"])
                    summary = NarrativeQATools.summarize_tool_result(full_result)
                    messages[idx]["content"] = summary
                except:
                    # If can't parse, leave as is
                    pass

    def _save_debug_log(self, session_result: Dict[str, Any]) -> None:
        """Save detailed agent activity log to JSON file for debugging."""
        import os
        
        # Create enhanced debug log with metadata
        debug_log = {
            "session_metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "question": session_result["question"],
                "total_turns": session_result["total_turns"],
                "final_answer_provided": session_result["final_answer"] is not None,
                "llm_config": self.llm_config,
                "system_config": {
                    "max_turns": self.max_turns,
                    "initial_context_size": self.initial_context_size,
                    "context_retrieval_size": self.context_retrieval_size,
                    "tool_result_limit": self.tool_result_limit,
                    "include_verb_phrases": self.include_verb_phrases,
                    "include_conversations": self.include_conversations,
                }
            },
            "final_answer": session_result["final_answer"],
            "activity_log": session_result["activity_log"],
            "conversation_flow": self._extract_conversation_flow(session_result["activity_log"]),
            "turn_by_turn_analysis": self._analyze_turns(session_result["activity_log"]),
            "session_summary": {
                "tool_usage_count": {},
                "total_tool_calls": 0,
                "successful_completion": session_result["final_answer"] is not None,
                "context_building": self._extract_context_building_info(session_result["activity_log"]),
            }
        }
        
        # Analyze tool usage patterns
        for log_entry in session_result["activity_log"]:
            if log_entry["tool"] in ["search_entity_by_name", "search_entity_summaries", 
                                    "search_vp_summaries", "search_conversation_summaries", "search_chunks"]:
                tool_name = log_entry["tool"]
                debug_log["session_summary"]["tool_usage_count"][tool_name] = debug_log["session_summary"]["tool_usage_count"].get(tool_name, 0) + 1
                debug_log["session_summary"]["total_tool_calls"] += 1
        
        # Ensure directory exists
        log_dir = os.path.dirname(self.debug_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Save to file
        try:
            with open(self.debug_log_file, 'w', encoding='utf-8') as f:
                json.dump(debug_log, f, indent=2, ensure_ascii=False, default=str)
            
            if self.verbose:
                print(f"📄 Debug log saved to: {self.debug_log_file}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to save debug log: {e}")

    def _extract_conversation_flow(self, activity_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conversation flow for easier analysis."""
        flow = []
        for entry in activity_log:
            if entry["tool"] in ["conversation_init", "agent_turn_start", "agent_response", "agent_tools_decision", "tool_call_start", "tool_call_result", "final_answer_attempt", "final_answer_success"]:
                flow_entry = {
                    "timestamp": entry["timestamp"],
                    "step": entry["tool"],
                    "reasoning": entry["reasoning"],
                    "key_data": {}
                }
                
                # Extract key information based on step type
                if entry["tool"] == "agent_response":
                    flow_entry["key_data"] = {
                        "turn": entry["args"].get("turn"),
                        "has_content": entry["args"].get("has_content"),
                        "has_tool_calls": entry["args"].get("has_tool_calls"),
                        "content_preview": entry["result"]["content"][:100] + "..." if entry["result"] and entry["result"].get("content") and len(str(entry["result"]["content"])) > 100 else entry["result"].get("content") if entry["result"] else None
                    }
                elif entry["tool"] == "tool_call_start":
                    flow_entry["key_data"] = {
                        "turn": entry["args"].get("turn"),
                        "tool_name": entry["args"].get("tool_name"),
                        "query": entry["args"].get("query"),
                        "reasoning": entry["args"].get("reasoning")
                    }
                elif entry["tool"] == "tool_call_result":
                    flow_entry["key_data"] = {
                        "turn": entry["args"].get("turn"),
                        "tool_name": entry["args"].get("tool_name"),
                        "result_count": entry["args"].get("result_count"),
                        "execution_successful": entry["args"].get("execution_successful")
                    }
                elif entry["tool"] == "final_answer_success":
                    flow_entry["key_data"] = {
                        "turn": entry["args"].get("turn"),
                        "parse_method": entry["args"].get("parse_method"),
                        "answer_length": entry["args"].get("answer_length")
                    }
                
                flow.append(flow_entry)
        return flow

    def _analyze_turns(self, activity_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze each turn in detail."""
        turns = {}
        
        for entry in activity_log:
            if "turn" in entry.get("args", {}):
                turn_num = entry["args"]["turn"]
                if turn_num not in turns:
                    turns[turn_num] = {
                        "turn_number": turn_num,
                        "agent_thought": None,
                        "tools_used": [],
                        "tool_results": [],
                        "final_answer_attempt": False,
                        "successful": False
                    }
                
                if entry["tool"] == "agent_response":
                    turns[turn_num]["agent_thought"] = entry["result"].get("content") if entry["result"] else None
                elif entry["tool"] == "tool_call_start":
                    turns[turn_num]["tools_used"].append({
                        "tool_name": entry["args"].get("tool_name"),
                        "query": entry["args"].get("query"),
                        "reasoning": entry["args"].get("reasoning")
                    })
                elif entry["tool"] == "tool_call_result":
                    turns[turn_num]["tool_results"].append({
                        "tool_name": entry["args"].get("tool_name"),
                        "result_count": entry["args"].get("result_count"),
                        "successful": entry["args"].get("execution_successful")
                    })
                elif entry["tool"] == "final_answer_attempt":
                    turns[turn_num]["final_answer_attempt"] = True
                elif entry["tool"] == "final_answer_success":
                    turns[turn_num]["successful"] = True
        
        return list(turns.values())

    def _extract_context_building_info(self, activity_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract information about context building process."""
        context_info = {
            "entities_extracted": 0,
            "entities_matched": 0,
            "summaries_retrieved": 0,
            "reranking_performed": False,
            "final_context_items": 0
        }
        
        for entry in activity_log:
            if entry["tool"] == "extract_entities":
                context_info["entities_extracted"] = entry["args"].get("entities_found", 0)
            elif entry["tool"] == "entity_matching":
                context_info["entities_matched"] = entry["args"].get("entities_matched", 0)
                context_info["summaries_retrieved"] = entry["args"].get("summaries_retrieved", 0)
            elif entry["tool"] == "entity_reranking":
                context_info["reranking_performed"] = True
                context_info["final_context_items"] = entry["args"].get("final_summaries_count", 0)
        
        return context_info

    def _log_activity(self, tool: str, reasoning: str, args: Dict[str, Any], result: Any) -> None:
        """Log activity for debugging and analysis."""
        self.activity_log.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": tool,
            "reasoning": reasoning,
            "args": args,
            "result": result
        })
