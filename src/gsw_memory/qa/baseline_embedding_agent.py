"""
Baseline Embedding Agent for Question Answering using nvembed-v2.

This module implements a baseline agent that uses dense retrieval with nvembed-v2
to search through a document corpus and answer multi-hop questions.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class ToolCall(BaseModel):
    """Represents a tool call the agent wants to make."""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments for the tool")


class AgentResponse(BaseModel):
    """Response from the baseline embedding agent."""
    answer: str = Field(description="The final answer to the question")
    reasoning: str = Field(description="Step-by-step reasoning process")
    tool_calls_made: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of tool calls made during reasoning"
    )


class BaselineEmbeddingAgent:
    """
    Baseline agent that uses embedding-based retrieval to answer questions.
    
    Uses nvembed-v2 model for dense retrieval and OpenAI for reasoning.
    """
    
    def __init__(
        self, 
        corpus_path: str,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict[str, Any]] = None,
        max_iterations: int = 15,
        embedding_model: str = "nvidia/NV-Embed-v2",
        cache_dir: str = "/home/yigit/codebase/gsw-memory/logs/embeddings_cache",
        device: str = "cuda:1"
    ):
        """
        Initialize the baseline embedding agent.
        
        Args:
            corpus_path: Path to the corpus JSON file
            model_name: LLM model to use for reasoning
            generation_params: Parameters for generation (temperature, etc.)
            max_iterations: Maximum number of tool calls allowed (default 15)
            embedding_model: Model to use for embeddings (default nvembed-v2)
            cache_dir: Directory to cache embeddings
            device: Device to use for embedding model
        """
        self.model_name = model_name
        self.generation_params = generation_params or {"temperature": 0.0}
        self.max_iterations = max_iterations
        self.client = OpenAI()
        self.cache_dir = cache_dir
        self.device = device
        
        # Load corpus
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        print(f"Loaded {len(self.corpus)} documents")
        
        # Initialize embedding model using transformers
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model, 
            trust_remote_code=True
        ).to(self.device)
        self.embedding_model.eval()
        
        # Set up max sequence length
        self.max_length = 32768
        
        # Define task instruction for nvembed-v2
        self.task_instruct = "Given a question, retrieve relevant documents that best answer the question"
        self.query_prefix = f"Instruct: {self.task_instruct}\nQuery: "
        self.passage_prefix = ""  # No instruction needed for passages
        
        # Build or load embeddings
        self._build_embeddings()
        
        # Tool definition for OpenAI function calling
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "search_embeddings",
                    "description": """Search for relevant documents using semantic similarity.
                    
                    This tool searches through the document corpus using dense retrieval
                    to find the most relevant documents for your query.
                    
                    Returns the top 5 most relevant documents with their titles and content.
                    
                    Tips for effective searching:
                    - Be specific with your queries
                    - You can search for entity names, relationships, or questions
                    - Try different phrasings if initial results aren't helpful
                    - For multi-hop questions, break them down and search for each part
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant documents"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def _build_embeddings(self):
        """Build or load document embeddings using transformers."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        embeddings_path = os.path.join(self.cache_dir, "nvembed_v2_transformers_embeddings.pt")
        
        if os.path.exists(embeddings_path):
            print("Loading cached embeddings...")
            # Load directly to the target device
            self.embeddings = torch.load(embeddings_path, map_location=self.device)
            print(f"Loaded embeddings with shape {self.embeddings.shape}")
        else:
            print("Building embeddings for corpus...")
            
            # Prepare documents for embedding
            # Combine title and text for better representation
            documents = []
            for doc in self.corpus:
                # Format: "Title: [title]\n[text]"
                doc_text = f"Title: {doc['title']}\n{doc['text']}"
                documents.append(doc_text)
            
            # Encode documents in batches
            batch_size = 4
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    print(f"Encoding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    
                    # Encode with nvembed-v2 using the direct encode method
                    embeddings = self.embedding_model.encode(
                        batch,
                        instruction=self.passage_prefix,  # No instruction for passages
                        max_length=self.max_length
                    )
                    
                    # Normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu())
                    
                    # Clear cache to save memory
                    torch.cuda.empty_cache()
            
            # Stack all embeddings
            self.embeddings = torch.cat(all_embeddings, dim=0)
            
            print(f"Generated embeddings with shape {self.embeddings.shape}")
            
            # Save embeddings
            print("Saving embeddings to cache...")
            torch.save(self.embeddings, embeddings_path)
            print("Embeddings saved successfully")
    
    def search_embeddings(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most relevant documents using embedding similarity.
        
        Args:
            query: The search query
            top_k: Number of top results to return (default 5)
            
        Returns:
            List of relevant documents with scores
        """
        with torch.no_grad():
            # Encode query with instruction prefix
            query_embedding = self.embedding_model.encode(
                [query],
                instruction=self.query_prefix,
                max_length=self.max_length
            )
            
            # Normalize query embedding
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            # Ensure embeddings are on the same device
            if self.embeddings.device != query_embedding.device:
                self.embeddings = self.embeddings.to(query_embedding.device)
            
            # Compute cosine similarity using matrix multiplication
            # Both embeddings are already normalized, so dot product = cosine similarity
            # Multiply by 100 to match the original nvembed-v2 scoring scale
            similarities = (query_embedding @ self.embeddings.T).squeeze(0) * 100
            
            # Get top-k most similar documents
            top_k = min(top_k, len(similarities))
            scores, indices = torch.topk(similarities, k=top_k, largest=True)
            
            # Prepare results
            results = []
            for idx, score in zip(indices.tolist(), scores.tolist()):
                if idx < len(self.corpus):
                    doc = self.corpus[idx]
                    results.append({
                        "title": doc["title"],
                        "text": doc["text"],
                        "score": float(score)
                    })
        
        return results
    
    def answer_question(self, question: str) -> AgentResponse:
        """
        Answer a question using embedding-based retrieval.
        
        Args:
            question: The question to answer
            
        Returns:
            AgentResponse with answer, reasoning, and tool calls
        """
        # System prompt adapted for baseline embedding search
        system_prompt = """You are an expert at answering multi-hop questions using semantic search.
        You have access to a single tool that searches through documents using dense retrieval.

        ═══════════════════════════════════════════════════════════════════════════════
                                UNDERSTANDING MUSIQUE QUESTIONS
        ═══════════════════════════════════════════════════════════════════════════════

        Musique questions require finding information through multiple connected entities. 
        You need to divide the question into multiple search hops.

        EXAMPLES OF MULTI-HOP DECOMPOSITION:

        - "Who succeeded the first President of Namibia?" requires:
        1. Search for "first President of Namibia" → Sam Nujoma
        2. Search for "Sam Nujoma successor" → Hifikepunye Pohamba
        
        - "When was the first establishment that McDonaldization is named after, open in the country Horndean is located?" requires:
        1. Search for "McDonaldization named after" → McDonald's
        2. Search for "Horndean located country" → England  
        3. Search for "first McDonald's opened England" → 1974
        
        - "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?" requires:
        1. Search for "Aruba continent" → South America
        2. Search for "Prazeres country" → Portugal
        3. Search for "Portuguese colonial South America" → Brazil
        4. Search for "German population Brazil" → 5 million
        
        - "When did the people who captured Malakoff come to the region where Philipsburg is located?" requires:
        1. Search for "Philipsburg location region" → Saint Martin Caribbean
        2. Search for "Malakoff captured by" → French
        3. Search for "French arrived Caribbean" → 1625

        ═══════════════════════════════════════════════════════════════════════════════
                        🚨 GOLDEN RULE: ITERATIVE SEARCH STRATEGY 🚨
        ═══════════════════════════════════════════════════════════════════════════════

        1. IDENTIFY key entities and relationships in the question
        2. START BROAD: Search for main entities/topics first
        3. NARROW DOWN: Use information found to search for specific relationships
        4. CONNECT THE DOTS: Search for connections between entities
        5. VERIFY: Cross-check information across multiple search results

        Search Query Strategies:
        - ENTITY QUERIES: "Namibia", "Sam Nujoma", "McDonald's"
        - RELATIONSHIP QUERIES: "Sam Nujoma successor", "McDonald's first location England"
        - DESCRIPTIVE QUERIES: "Portuguese colonial holdings South America"
        - TEMPORAL QUERIES: "French arrival Caribbean historical"

        Example Search Progression:
        - "When did the people who captured Malakoff come to the region where Philipsburg is located?"
          → Search: "Malakoff", "Philipsburg" (entities first!)
          → Search: "Malakoff captured by", "Philipsburg location" (relationships)
          → Search: "French arrival Caribbean" (final connection)

        ✅ RIGHT: Start with entities, then build relationships
        ❌ WRONG: Overly complex searches that try to do everything at once

        ═══════════════════════════════════════════════════════════════════════════════
                                    MANDATORY RULES
        ═══════════════════════════════════════════════════════════════════════════════

        1. Search systematically through each hop of the question
        2. Read ALL returned documents carefully - answers may be in any of them
        3. Use information from previous searches to inform new queries
        4. Track your searches - DO NOT repeat the same query
        5. Provide reasoning BEFORE and AFTER each tool call
        6. Verify evidence for EACH hop before final answer
        7. Maximum 15 tool calls - make them count!

        ═══════════════════════════════════════════════════════════════════════════════
                                SEARCH STRATEGY GUIDELINES
        ═══════════════════════════════════════════════════════════════════════════════

        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ search_embeddings - YOUR ONLY TOOL                                         │
        ├─────────────────────────────────────────────────────────────────────────────┤
        │ • Returns top 5 most semantically similar documents                         │
        │ • Query strategies:                                                         │
        │   - Entity names: "Barack Obama", "France", "McDonald's"                   │
        │   - Relationships: "Obama predecessor", "capital of France"                 │
        │   - Descriptions: "first McDonald's restaurant England"                     │
        │   - Questions: "who succeeded Sam Nujoma"                                   │
        │ • Try different phrasings if first attempt doesn't work                     │
        │ • Be specific but not overly narrow                                         │
        └─────────────────────────────────────────────────────────────────────────────┘

        ═══════════════════════════════════════════════════════════════════════════════
                            DOCUMENT ANALYSIS REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        For each search result:
        - READ the title - it often contains key information
        - SCAN the text for relevant facts and relationships
        - EXTRACT specific answers to your current hop
        - NOTE entities mentioned that might be useful for next searches
        - CHECK multiple documents - the answer might not be in the top result

        ═══════════════════════════════════════════════════════════════════════════════
                                REASONING REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        Before EVERY search:
        - What specific information are you looking for?
        - How does this search advance toward answering the question?
        - What search terms will be most effective?

        After EVERY search:
        - What relevant information did you find?
        - Which documents contained useful information?
        - What do you now know that you didn't before?
        - What should you search for next?

        Track evidence for each hop:
        - Hop 1: [Search Query] → [Key Finding from Document]
        - Hop 2: [Search Query] → [Key Finding from Document]
        - Continue until complete answer is found

        ═══════════════════════════════════════════════════════════════════════════════
                                SEARCH OPTIMIZATION TIPS
        ═══════════════════════════════════════════════════════════════════════════════

        1. BROAD-TO-NARROW: Start with main entities, then search relationships
        2. VARIED PHRASING: Try different ways to express the same concept
        3. CONTEXT BUILDING: Use found information to improve subsequent searches
        4. CROSS-VERIFICATION: Search related terms to confirm information
        5. TEMPORAL FOCUS: For historical questions, include time-related terms

        If stuck after several searches:
        - Try synonyms or alternative phrasings
        - Search for more general concepts then narrow down
        - Look for indirect relationships (e.g., search for related entities)

        ═══════════════════════════════════════════════════════════════════════════════
                                KEY EXAMPLES (3 patterns)
        ═══════════════════════════════════════════════════════════════════════════════

        ### EXAMPLE 1: Simple 2-hop - "Who succeeded the first President of Namibia?"
        ────────────────────────────────────────────────────────────────────────────────
        SEARCH STRATEGY:

        1. search_embeddings("first President of Namibia")
           → Documents mention Sam Nujoma as first president (1990-2005)
        2. search_embeddings("Sam Nujoma successor president Namibia")
           → Documents show Hifikepunye Pohamba succeeded him in 2005

        ANSWER: Hifikepunye Pohamba

        ### EXAMPLE 2: Complex 4-hop 
        "When did the people who captured Malakoff come to the region where Philipsburg is located?"
        ────────────────────────────────────────────────────────────────────────────────

        1. search_embeddings("Philipsburg location")
           → Capital of Sint Maarten in the Caribbean
        2. search_embeddings("Malakoff captured")
           → Fort captured by French forces in 1855
        3. search_embeddings("French arrival Caribbean historical")
           → French arrived in Caribbean in 1625

        ANSWER: 1625

        ### EXAMPLE 3: Multi-entity connection
        "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?"
        ────────────────────────────────────────────────────────────────────────────────

        1. search_embeddings("Aruba continent location")
           → Aruba is in South America
        2. search_embeddings("Prazeres country location")  
           → Prazeres is in Portugal
        3. search_embeddings("Portuguese colonial holdings South America")
           → Brazil was main Portuguese colony in South America
        4. search_embeddings("German population Brazil")
           → About 5 million Germans live in Brazil

        ANSWER: 5 million

        ═══════════════════════════════════════════════════════════════════════════════
                                VERIFICATION CHECKLIST
        ═══════════════════════════════════════════════════════════════════════════════

        Before providing your final answer, verify:
        □ Did I search for ALL key components of the question?
        □ Did I find concrete evidence in the documents for each hop?
        □ Do I have the complete chain of reasoning?
        □ Is my answer based on information actually found in the documents?
        □ Did I check multiple documents to verify important facts?

        ═══════════════════════════════════════════════════════════════════════════════
                                OUTPUT FORMAT
        ═══════════════════════════════════════════════════════════════════════════════

        When you find the answer, respond with ONLY this JSON:
        {
            "reasoning": "Step-by-step explanation with evidence from specific documents",
            "answer": "Just the final answer, no extra words"
        }

        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        tool_calls_made = []
        iterations = 0
        
        # Create tools dict for execution
        tools = {
            "search_embeddings": self.search_embeddings
        }
        
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
                print(f"Error during LLM call: {e}")
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
                
                # Add reasoning from assistant messages
                for msg in messages[2:]:  # Skip system and user messages
                    if msg.get("role") == "assistant" and msg.get("content"):
                        reasoning_parts.append(msg.get("content", ""))
                
                # Remove empty messages
                reasoning_parts = [part for part in reasoning_parts if part]
                reasoning = "\n".join(reasoning_parts)
                
                # Try to parse JSON response
                try:
                    # Find JSON in the content
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        response_data = json.loads(json_str)
                        answer = response_data.get("answer", "")
                        # Use the reasoning from JSON if available
                        if "reasoning" in response_data:
                            reasoning = response_data["reasoning"]
                    else:
                        # Fallback if no JSON found
                        answer = content
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    answer = content
                
                return AgentResponse(
                    answer=answer,
                    reasoning=reasoning,
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
        questions: List[str]
    ) -> List[AgentResponse]:
        """
        Answer multiple questions (processes sequentially).
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of AgentResponse objects
        """
        responses = []
        for i, question in enumerate(questions):
            print(f"\nProcessing question {i+1}/{len(questions)}: {question[:100]}...")
            response = self.answer_question(question)
            responses.append(response)
        return responses


# Example usage and testing
if __name__ == "__main__":
    # Test the baseline agent
    corpus_path = "/home/yigit/codebase/gsw-memory/musique_corpus_10_q.json"
    
    # Initialize agent
    agent = BaselineEmbeddingAgent(
        corpus_path=corpus_path,
        model_name="gpt-4o",
        generation_params={"temperature": 0.0},
        max_iterations=15,
        device="cuda:1"  # Specify the device
    )
    
    # Test with a simple question
    test_question = "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
    
    print(f"\nTesting with question: {test_question}")
    response = agent.answer_question(test_question)
    
    print(f"\nAnswer: {response.answer}")
    print(f"\nReasoning: {response.reasoning}")
    print(f"\nTool calls made: {len(response.tool_calls_made)}")