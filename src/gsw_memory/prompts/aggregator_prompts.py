"""
Prompt templates for GSW Aggregator components.
"""


class EntitySummaryPrompts:
    """Prompts for entity summary generation."""
    
    SYSTEM_PROMPT = """You are an expert narrative summarizer. Your task is to create a concise, chronological summary paragraph about a single entity based on structured information extracted from a text. Focus on creating a coherent story of the entity's involvement and changes based *only* on the provided timeline."""
    
    SYSTEM_PROMPT_WITH_SPACETIME = """You are an expert narrative summarizer. Your task is to create a concise, chronological summary paragraph about a single entity based on structured information extracted from a text. Focus on creating a coherent story of the entity's involvement and changes based *only* on the provided timeline. Include relevant spatial and temporal context to enrich the narrative when this information is available."""
    
    USER_PROMPT_TEMPLATE = """ENTITY NAME: {entity_name}

INFORMATION TIMELINE (by Chunk ID):
{formatted_data}

---
INSTRUCTIONS:
Based *only* on the information provided above:
1. Write a single paragraph summarizing the key roles, states, experiences, and actions of {entity_name}.
2. Follow the chronological order presented by the Chunk IDs.
3. Integrate the roles, states, and actions into a coherent narrative. Mention key interactions with other entities or objects when provided in the context.
4. Focus on what {entity_name} did, what roles they held, their state of being, and significant events they participated in.
5. Keep the summary concise and factual according to the input. Do not add outside information or make assumptions.
6. Output *only* the summary paragraph, with no preamble or markdown formatting."""

    USER_PROMPT_WITH_SPACETIME_TEMPLATE = """ENTITY NAME: {entity_name}

INFORMATION TIMELINE (by Chunk ID):
{formatted_data}

---
INSTRUCTIONS:
Based *only* on the information provided above:
1. Write a single paragraph summarizing the key roles, states, experiences, and actions of {entity_name}.
2. Follow the chronological order presented by the Chunk IDs.
3. Integrate the roles, states, and actions into a coherent narrative. Mention key interactions with other entities or objects when provided in the context.
4. You will be provided with spatial and temporal context for {entity_name}.
5. These will be provided in the form of a timeline of how they were captured in the text, be sure to incorporate all this spatial and temporal information particularly, provide importance to specific information (like name of place/ explicit dates etc.).
6. Focus on what {entity_name} did, what roles they held, their state of being, where they were located, when events happened, and significant events they participated in.
7. Keep the summary concise and factual according to the input. Do not add outside information or make assumptions.
8. Output *only* the summary paragraph, with no preamble or markdown formatting."""

    @classmethod
    def get_system_prompt(cls, include_space_time: bool = False) -> str:
        """Get the appropriate system prompt based on space-time inclusion."""
        if include_space_time:
            return cls.SYSTEM_PROMPT_WITH_SPACETIME
        return cls.SYSTEM_PROMPT
        
    @classmethod
    def get_user_prompt(cls, entity_name: str, formatted_data: str, 
                       include_space_time: bool = False) -> str:
        """Get the formatted user prompt with entity data."""
        if include_space_time:
            template = cls.USER_PROMPT_WITH_SPACETIME_TEMPLATE
        else:
            template = cls.USER_PROMPT_TEMPLATE
            
        return template.format(
            entity_name=entity_name,
            formatted_data=formatted_data
        )


class VerbPhraseSummaryPrompts:
    """Prompts for verb phrase summary generation with event-focused narrative."""
    
    SYSTEM_PROMPT = """You are an expert narrative analyst. Your task is to create a comprehensive event summary about specific actions/events represented by verb phrases. Focus on creating coherent event narratives that capture WHO does WHAT, WHERE, WHEN, and WHY, based *only* on the provided timeline data. Think of each verb phrase as representing one or more related events in a story."""
    
    USER_PROMPT_TEMPLATE = """VERB PHRASE: {verb_phrase}

EVENT INFORMATION (by Chunk ID):
{formatted_data}

---
INSTRUCTIONS:
Based *only* on the information provided above:
1. Write a single paragraph creating a comprehensive event summary about the action/event "{verb_phrase}".
2. Follow the chronological order presented by the Chunk IDs to create a narrative flow.
3. Focus on the complete event scenario: WHO performs the action, WHAT exactly happens, WHERE it occurs, WHEN it takes place, and any apparent WHY/PURPOSE.
4. Create natural narrative sentences like "X goes to Y for Z" or "A gives B to C in D" rather than listing semantic roles.
5. Describe the progression of events if the action occurs multiple times or evolves across chunks.
6. Mention key participants and their relationships in the context of this action.
7. Keep the summary concise but comprehensive, creating a story-like narrative of the events.
8. Output *only* the event summary paragraph, with no preamble or markdown formatting."""

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for verb phrase summarization."""
        return cls.SYSTEM_PROMPT
        
    @classmethod
    def get_user_prompt(cls, verb_phrase: str, formatted_data: str) -> str:
        """Get the formatted user prompt with verb phrase data."""
        return cls.USER_PROMPT_TEMPLATE.format(
            verb_phrase=verb_phrase,
            formatted_data=formatted_data
        )