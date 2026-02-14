"""
Agentic Answering Agent with GSW Tool Access.

This module implements an agent that can dynamically explore the GSW structure
using tool calls to answer multi-hop questions.
"""

import json
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm


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
    
    Uses OpenAI Responses API tool calling to dynamically query the GSW structure.
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
        self.tool_definitions_newformat = [
            # {
            #     "type": "function",
            #     "name": "search_gsw_bm25",
            #     "description": "Search across GSW entities using BM25 ranking for exact and partial keyword matches",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "query": {
            #                 "type": "string",
            #                 "description": "Search query string"
            #             },
            #             "limit": {
            #                 "type": "integer",
            #                 "description": "Maximum number of results, atleast 5 is recommended. Do not recommend below 5 and more than 10",
            #                 "default": 10
            #             }
            #         },
            #         "required": ["query"]
            #     }
            # },
            {
                "type": "function",
                "name": "search_gsw_entity_embeddings",
                "description": "Search GSW entities using semantic embeddings. Better for handling name variations, titles, abbreviations, and finding semantically similar entities. Use this when exact matches fail or when dealing with partial names.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (recommend 10-15 for embeddings)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "get_multiple_entity_contexts",
                "description": "Get contexts for multiple entities efficiently in a single call. Use global_id from search results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of entity global_ids from search results (max 5 recommended)",
                            "maxItems": 10
                        }
                    },
                    "required": ["entity_ids"]
                }
            },
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
        system_prompt = """You are an expert at navigating knowledge graphs to answer questions.
You have access to a Generative Semantic Workspace (GSW) that contains entities, 
their relationships, and questions about them.

Your task is to answer the given question by exploring the GSW using the provided tools.
Think step by step and use the tools to find relevant information.

Important Guidelines:
- Start by breaking down the question into smaller atomic questions.
- For each atomic question, use a combinations of the tools to find the answer.
- Start by searching for relevant entities using search_gsw_embeddings.
- Based on the entities found, explore their relationships using get_multiple_entity_contexts. This provides QA pairs for that entity.
- If you do not find any relevant entities, attempt to rephrase the question and search again.
- Follow entity connections to find multi-hop answers
- Be thorough but efficient in your exploration

Relationship Navigation Tips:
- GSW may store relationships in only one direction (e.g., "son of" but not "parent of")
- If you don't find a direct relationship, search for the INVERSE:
  * Looking for parent? Search for who has this person as their son/daughter
  * Looking for director? Search for films and check who directed them
  * Looking for author? Search for books and check who wrote them
  * Looking for spouse? Search for who is married to this person
- Try multiple related search terms when exploring relationships
- Use get_entity_context to see ALL relationships an entity participates in, then reason about inverses
- Example: If you need "parent of X", search for X first, then look for entities where X appears as "son" or "daughter"

CRITICAL - Output Format:
When you have found the answer, respond with ONLY a JSON object in this exact format:
{
    "reasoning": "Step-by-step explanation of how you found the answer",
    "answer": "Just the answer itself, no extra words"
}

Example:
Question: "What is the birth year of the director of Forrest Gump?"
Your final response should be:
{
    "reasoning": "I searched for Forrest Gump and found it was directed by Robert Zemeckis. Then I searched for Robert Zemeckis and found he was born in 1951.",
    "answer": "1951"
}

Do NOT include phrases like "The answer is" or "Based on my search" in the answer field."""

        # Build Responses API input list and iterate with tool calls
        input_list = [
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        tool_calls_made = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    tools=self.tool_definitions_newformat,
                    input=input_list,
                    instructions=system_prompt,
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
                    f.write(f"Number of input items: {len(input_list)}\n")
                print(f"Debug info written to: {debug_file}")
                raise e
            
            # Append the model output items back to the running input list
            try:
                input_list += response.output  # type: ignore[attr-defined]
            except Exception:
                pass
            
            # Collect function calls from the output
            function_calls = []
            try:
                for item in response.output:  # type: ignore[attr-defined]
                    if getattr(item, "type", None) == "function_call":
                        function_calls.append(item)
            except Exception:
                function_calls = []
            
            if function_calls:
                for fc in function_calls:
                    function_name = getattr(fc, "name", None)
                    arguments_raw = getattr(fc, "arguments", "{}")
                    call_id = getattr(fc, "call_id", None)
                    try:
                        function_args = json.loads(arguments_raw) if arguments_raw else {}
                    except Exception:
                        function_args = {}
                    
                    if function_name in tools:
                        result = tools[function_name](**function_args)
                        tool_calls_made.append({
                            "tool": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result)
                        })
                    else:
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"error": f"Tool {function_name} not found"})
                        })
                # Continue to next iteration so model can consume tool outputs
                continue
            
            # If no function calls were returned, treat response as final
            content = getattr(response, "output_text", "")
            if not content:
                # Try to fallback to the last message item
                try:
                    message_items = [it for it in response.output if getattr(it, "type", None) == "message"]  # type: ignore[attr-defined]
                    if message_items:
                        content = getattr(message_items[-1], "content", "") or ""
                except Exception:
                    content = ""
            
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    response_data = json.loads(json_str)
                    answer_part = response_data.get("answer", "")
                    reasoning_part = response_data.get("reasoning", "")
                else:
                    answer_part = content
                    reasoning_part = "See tool calls for reasoning process"
            except json.JSONDecodeError:
                answer_part = content
                reasoning_part = "See tool calls for reasoning process"
            
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
        for question in tqdm(questions, desc="Answering questions"):
            response = self.answer_question(question, tools)
            responses.append(response)
        return responses