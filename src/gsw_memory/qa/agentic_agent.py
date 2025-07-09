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
            {
                "type": "function",
                "function": {
                    "name": "search_gsw",
                    "description": "Search across GSW questions and entities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_entity_context",
                    "description": "Get all questions an entity participates in",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "ID of the entity"
                            }
                        },
                        "required": ["entity_id"]
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
        system_prompt = """You are an expert at navigating knowledge graphs to answer questions.
You have access to a Generative Semantic Workspace (GSW) that contains entities, 
their relationships, and questions about them.

Your task is to answer the given question by exploring the GSW using the provided tools.
Think step by step and use the tools to find relevant information.

Important Guidelines:
- Start by searching for relevant entities or questions
- Use get_entity_context to explore relationships
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        tool_calls_made = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # Get response from LLM with function calling
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto",
                **self.generation_params
            )
            
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
                
                # Try to parse JSON response
                try:
                    # Find JSON in the content (it might have extra text), this can be replaced with structured output.
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        response_data = json.loads(json_str)
                        answer_part = response_data.get("answer", "")
                        reasoning_part = response_data.get("reasoning", "")
                    else:
                        # Fallback if no JSON found
                        answer_part = content
                        reasoning_part = "See tool calls for reasoning process"
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
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
        for question in questions:
            response = self.answer_question(question, tools)
            responses.append(response)
        return responses