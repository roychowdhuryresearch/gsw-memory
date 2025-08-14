"""
Agentic Answering Agent with GSW Tool Access.

This module implements an agent that can dynamically explore the GSW structure
using tool calls to answer multi-hop questions.
"""

import json
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from openai import OpenAI


class ToolCall(BaseModel):
    """Represents a tool call the agent wants to make."""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments for the tool")


class AgentResponse(BaseModel):
    """Response from the agentic agent."""
    answer: str = Field(description="The final answer to the question")
    reasoning: str = Field(description="Step-by-step reasoning process")
    tool_calls_made: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of tool calls made during reasoning"
    )


class AgenticAnsweringAgent:
    """
    Agent that can use GSW tools to answer questions through exploration.
    
    Uses OpenAI function calling to dynamically query the GSW structure.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict[str, Any]] = None,
        max_iterations: int = 10
    ):
        """
        Initialize the agentic answering agent.
        
        Args:
            model_name: LLM model to use
            generation_params: Parameters for generation (temperature, etc.)
            max_iterations: Maximum number of tool calls allowed
        """
        self.model_name = model_name
        self.generation_params = generation_params or {"temperature": 0.0}
        self.max_iterations = max_iterations
        self.client = OpenAI()
        
        # Tool definitions for OpenAI function calling
        self.tool_definitions = [
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "search_gsw_bm25_entity_name",
            #         "description": """Search for entities by name/title in the knowledge base.

            #             WHEN TO USE:
            #             - First hop: Always use this first
            #             - Later hops: Add relation hints to disambiguate (e.g., "Paris France" not just "Paris")

            #             GOOD QUERIES:
            #             - Hop 1: "Forrest Gump", "Christopher Nolan", "France"
            #             - Hop 2+: "London England", "Christopher Nolan director", "Paris France capital"

            #             BAD QUERIES:
            #             - "director of Forrest Gump" (use entity_features for this)
            #             - "capital of France" (use entity_features for this)

            #             Returns entities with IDs in format: doc_path/gsw_id.json::entity_id""",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "query": {
            #                     "type": "string",
            #                     "description": "Entity name, with relation hints for hop 2+"
            #                 },
            #                 "limit": {
            #                     "type": "integer",
            #                     "description": "Number of results (10-20 recommended)",
            #                     "default": 15
            #                 }
            #             },
            #             "required": ["query"]
            #         }
            #     }
            # },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "search_gsw_bm25_entity_with_entity_features",
            #         "description": """Search for entities by their relationships or properties.
            #             CRITICAL: Use this for Variant 2 (attribute-of-attribute) patterns!

            #             WHEN TO USE:
            #             - Middle hops in nested queries: "capital of France", "director of Inception"
            #             - When entity is defined by relationship: "company that created iPhone"
            #             - When entity_name search fails to find relationship-defined entities

            #             GOOD QUERIES:
            #             - "capital of France" (to find Paris as capital entity)
            #             - "birthplace of Christopher Nolan" (to find London as birthplace)
            #             - "director of Forrest Gump" (to find Zemeckis as director)
            #             - "country containing London" (to find UK as container)

            #             Returns entities defined by these relationships.""",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "query": {
            #                     "type": "string",
            #                     "description": "Relationship/property-based description"
            #                 },
            #                 "limit": {
            #                     "type": "integer",
            #                     "description": "Number of results (5-10 recommended)",
            #                     "default": 10
            #                 }
            #             },
            #             "required": ["query"]   
            #         }
            #     }
            # },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "get_multiple_relevant_entity_contexts",
            #         "description": """Get all facts and relationships for entities.

            #             MANDATORY: Use after EVERY search to get entity information.

            #             USAGE:
            #             - Single entity: ["entity_123"]
            #             - Multiple entities (disambiguation): ["entity_123", "entity_456", "entity_789"]
            #             - Can handle up to 20 entities at once

            #             Returns all facts about entities including:
            #             - Relationships (director of, capital of, birthplace, etc.)
            #             - Properties (birth year, population, government type, etc.)
            #             - Connected entities for next hops

            #             READ THE ENTIRE CONTEXT - the information you need is there.""",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "entity_ids": {
            #                     "type": "array",
            #                     "items": {
            #                         "type": "string",
            #                         "description": "Entity ID from search (format: doc_path/gsw_id.json::entity_id)"
            #                     },
            #                     "description": "List of entity IDs (1-20)"
            #                 }
            #             },
            #             "required": ["entity_ids"]
            #         }
            #     }
            # },
            {
                "type": "function",
                "function": {
                    "name": "search_gsw_embeddings_of_entity_summaries",
                    "description": """Search for entities using semantic similarity and neural embeddings.
                        ADVANCED: Uses deep learning to understand meaning beyond keyword matching.

                        WHEN TO USE:
                        - Complex conceptual queries that BM25 might miss
                        - Natural language descriptions of entities
                        - When you need semantic understanding, not just keyword matching
                        - As backup when BM25 searches return insufficient results
                        - Multi-word descriptive queries about entity characteristics

                        GOOD QUERIES:
                        - "major European financial center" (finds London, Frankfurt, etc.)
                        - "large technology company founded in garage" (finds Apple, HP, etc.)
                        - "ancient civilization along the Nile river" (finds Ancient Egypt)
                        - "island nation in the Pacific with volcanic activity" (finds Japan, Philippines, etc.)
                        - "person who revolutionized computer graphics and animation" (finds Steve Jobs, John Lasseter, etc.)

                        COMPLEMENTARY TO BM25:
                        - Use BM25 first for direct name/title matches
                        - Use embeddings when BM25 misses conceptual connections
                        - Embedding search understands context and relationships semantically

                        Returns same rich metadata as BM25 searches with semantic relevance scores.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description or conceptual query"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (30-50 recommended)",
                                "default": 30
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def answer_question(
        self, 
        question: str, 
        tools: Dict[str, Callable]
    ) -> AgentResponse:
        """
        Answer a question using GSW tools.
        
        Args:
            question: The question to answer
            tools: Dict mapping tool names to callable functions
            
        Returns:
            AgentResponse with answer, reasoning, and tool calls
        """
        # System prompt for the agent
        system_prompt = """You are an expert at answering multi-hop questions using semantic search.
        You have ONE powerful tool: neural embedding search that understands meaning and context.

        ═══════════════════════════════════════════════════════════════════════════════
                        🧠 EMBEDDING-ONLY STRATEGY: SEMANTIC REASONING
        ═══════════════════════════════════════════════════════════════════════════════

        YOUR APPROACH:
        1. ANALYZE the question to identify what information you need
        2. CRAFT strategic search queries that capture semantic meaning
        3. EXTRACT answers from rich summaries and metadata returned
        4. FOLLOW the chain by searching for newly discovered entities/concepts
        5. REPEAT until you have the complete answer

        ✅ KEY ADVANTAGE: Embedding search provides COMPLETE information in search results:
        • Entity summaries with full details
        • Relationship information and connections  
        • Historical facts and temporal data
        • Contextual information and properties
        • NO NEED for separate context retrieval!

        ═══════════════════════════════════════════════════════════════════════════════
                      🔍 MASTERING EMBEDDING SEARCH QUERIES
        ═══════════════════════════════════════════════════════════════════════════════

        🧠 search_gsw_embeddings_of_entity_summaries - YOUR SEMANTIC POWERHOUSE
        • PURPOSE: Neural understanding of meaning, relationships, and context
        • RETURNS: Rich entity summaries with all needed information
        • POWER: Understands synonyms, relationships, descriptions, and concepts

        📋 QUERY STRATEGIES (All work with embedding search!):

        🎯 DIRECT ENTITIES: "Sam Nujoma", "McDonald's", "Namibia", "Philipsburg"
        → Returns: Full entity details, relationships, historical facts

        🎯 RELATIONSHIPS: "first president of Namibia", "successor to Sam Nujoma"
        → Returns: Entities with those specific relationships

        🎯 DESCRIPTIONS: "Nordic country known for fjords", "technology company founded in garage"  
        → Returns: Entities matching semantic descriptions

        🎯 CONCEPTS: "French colonization Caribbean", "German population Brazil"
        → Returns: Entities with conceptual connections

        🎯 MULTI-HOP: "capital of country where Horndean is located"
        → Returns: Entities understanding the nested relationship

        ═══════════════════════════════════════════════════════════════════════════════
                                EXECUTION WORKFLOW
        ═══════════════════════════════════════════════════════════════════════════════

        🎯 SMART SEARCH STRATEGY:
        1. Start with direct entities from the question
        2. Read ALL details in the returned summaries carefully
        3. Extract key information and connected entities
        4. Search for missing pieces using semantic queries
        5. Chain searches until complete answer emerges

        🚫 CRITICAL PRINCIPLES:
        • READ every summary completely - answers are embedded in the rich metadata
        • Use diverse query formulations to find different aspects
        • Don't repeat identical searches
        • Maximum 8-10 searches - be strategic
        • Trust the semantic understanding of embedding search

        ═══════════════════════════════════════════════════════════════════════════════
                            EMBEDDING-ONLY EXAMPLES
        ═══════════════════════════════════════════════════════════════════════════════

        🔵 EXAMPLE 1: Simple succession - "Who succeeded the first President of Namibia?"
        1. search_gsw_embeddings_of_entity_summaries("first president of Namibia")
           → Returns: Sam Nujoma entity with full details including presidency dates
        2. search_gsw_embeddings_of_entity_summaries("successor to Sam Nujoma Namibia president")
           → Returns: Hifikepunye Pohamba entity with succession details
        ANSWER: Hifikepunye Pohamba

        🟢 EXAMPLE 2: Geographic chain - "What currency is used where Billy Giles died?"
        1. search_gsw_embeddings_of_entity_summaries("Billy Giles death location")  
           → Returns: Billy Giles entity with death details (died in London)
        2. search_gsw_embeddings_of_entity_summaries("London England currency")
           → Returns: UK/England entities with currency information (Pound Sterling)
        ANSWER: Pound Sterling

        🟡 EXAMPLE 3: Descriptive query - "What is the capital of the Nordic country known for fjords?"
        1. search_gsw_embeddings_of_entity_summaries("Nordic country known for fjords")
           → Returns: Norway entity with geographic/cultural details and capital (Oslo)
        ANSWER: Oslo

        🔴 EXAMPLE 4: Complex temporal - "When did French arrive in Caribbean?"
        1. search_gsw_embeddings_of_entity_summaries("French colonization Caribbean arrival date")
           → Returns: Historical entities with French Caribbean colonization timeline (1625)
        ANSWER: 1625

        ═══════════════════════════════════════════════════════════════════════════════
                                REASONING REQUIREMENTS  
        ═══════════════════════════════════════════════════════════════════════════════

        📋 BEFORE EVERY search, explain:
        • What specific information you're seeking
        • How your query captures the semantic meaning needed
        • What aspect of the question this addresses

        📋 AFTER EVERY search, analyze:
        • Key information found in the summaries and metadata
        • What questions are now answered vs still need answers
        • What to search for next (if anything)

        📋 TRACK your multi-hop progress:
        • Hop 1: [Query] → [Key Information Found]
        • Hop 2: [Query] → [Key Information Found]  
        • Hop 3+: [Query] → [Key Information Found]

        💡 REMEMBER: Embedding search results contain COMPLETE entity information including relationships, properties, dates, and connections. Read everything carefully!

        ═══════════════════════════════════════════════════════════════════════════════
                           ✅ FINAL VERIFICATION & OUTPUT
        ═══════════════════════════════════════════════════════════════════════════════

        Before answering, verify you have:
        ✓ Searched for all key concepts/entities in the question
        ✓ Found concrete evidence in the search result summaries
        ✓ Traced the complete reasoning chain with evidence
        ✓ Extracted the final answer from reliable entity information

        📤 OUTPUT FORMAT - Respond with ONLY this JSON:
        {
            "reasoning": "Clear step-by-step explanation with evidence from search results",
            "answer": "Just the final answer, no extra words or phrases"
        }

        ❌ Do NOT include: "The answer is...", "Based on my search...", etc. in the answer field."""
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        tool_calls_made = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # Get response from LLM with function calling
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    **self.generation_params
                )
            except Exception as e:
                # Write debug info to file
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_file = f"debug_context_overflow_{timestamp}.txt"
                
                with open(debug_file, "w") as f:
                    f.write(f"ERROR: {type(e).__name__}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write(f"\nQuestion being processed: {question}\n")
                    f.write(f"Iteration: {iterations}\n")
                    f.write(f"Number of messages: {len(messages)}\n")
                    f.write("\n" + "="*80 + "\n")
                    f.write("MESSAGES HISTORY:\n")
                    f.write("="*80 + "\n\n")
                    
                    for i, msg in enumerate(messages):
                        f.write(f"Message {i+1}:\n")
                        f.write(f"Role: {msg.get('role', 'unknown')}\n")
                        
                        # Handle content
                        content = msg.get('content', '')
                        if content:
                            f.write(f"Content length: {len(content)} chars\n")
                            f.write(f"Content preview (first 500 chars):\n{content[:500]}...\n")
                        
                        # Handle tool calls
                        if 'tool_calls' in msg:
                            f.write(f"Tool calls: {len(msg['tool_calls'])}\n")
                        
                        # Calculate approximate token count (rough estimate)
                        msg_str = json.dumps(msg)
                        approx_tokens = len(msg_str) // 4  # rough estimate
                        f.write(f"Approximate tokens: {approx_tokens}\n")
                        f.write("\n" + "-"*40 + "\n\n")
                    
                    # Summary stats
                    f.write("\n" + "="*80 + "\n")
                    f.write("SUMMARY:\n")
                    total_chars = sum(len(json.dumps(msg)) for msg in messages)
                    f.write(f"Total characters in messages: {total_chars}\n")
                    f.write(f"Approximate total tokens: {total_chars // 4}\n")
                    
                print(f"Debug info written to: {debug_file}")
                raise e
            
            message = response.choices[0].message
            messages.append(message.model_dump())
            
            # Check if the model wants to make tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    if function_name in tools:
                        result = tools[function_name](**function_args)
                        
                        # Record the tool call
                        tool_calls_made.append({
                            "tool": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, indent=2)
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: Tool {function_name} not found"
                        })
            else:
                # No more tool calls, extract final answer
                content = message.content or ""
                
                # Build comprehensive reasoning from all tool calls and final content
                reasoning_parts = []
                
                # Add reasoning from tool calls
                for msg in messages[2:]:
                    if msg.get("role") != "tool":
                        reasoning_parts.append(msg.get("content", ""))
                
                # delete empty messages
                reasoning_parts = [part for part in reasoning_parts if part]
                reasoning_part = "\n".join(reasoning_parts)
                
                # Try to parse JSON response
                try:
                    # Find JSON in the content (it might have extra text), this can be replaced with structured output.
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        response_data = json.loads(json_str)
                        answer_part = response_data.get("answer", "")
                    else:
                        # Fallback if no JSON found
                        answer_part = content
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    answer_part = content
                
                return AgentResponse(
                    answer=answer_part,
                    reasoning=reasoning_part,
                    tool_calls_made=tool_calls_made
                )
        
        # Reached max iterations
        return AgentResponse(
            answer="Unable to find answer within iteration limit",
            reasoning=f"Reached maximum of {self.max_iterations} iterations",
            tool_calls_made=tool_calls_made
        )
    
    def answer_batch(
        self, 
        questions: List[str], 
        tools: Dict[str, Callable]
    ) -> List[AgentResponse]:
        """
        Answer multiple questions (processes sequentially for now).
        
        Args:
            questions: List of questions to answer
            tools: Dict mapping tool names to callable functions
            
        Returns:
            List of AgentResponse objects
        """
        responses = []
        for question in questions:
            response = self.answer_question(question, tools)
            responses.append(response)
        return responses