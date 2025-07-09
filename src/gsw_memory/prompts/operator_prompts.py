"""
Prompt templates for GSW Operator components.
"""


class CorefPrompts:
    """Prompts for coreference resolution."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that performs coreference resolution. Your task is to replace all pronouns and references with their full names or explicit references. Maintain the original flow and readability of the text."""
    
    USER_PROMPT_TEMPLATE = """Please perform coreference resolution on the following text. Replace all pronouns and references with their full names or explicit references:

{text}"""


class ContextPrompts:
    """Prompts for context generation."""
    
    SYSTEM_PROMPT = """You are a helpful assistant. Your task is to provide a brief context that explains how a given text chunk fits within the larger document it comes from. Be concise and focus on the information needed to understand the chunk's place in the narrative or argument."""
    
    USER_PROMPT_TEMPLATE = """<document>
{doc_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class OperatorPrompts:
    """Prompts for GSW generation - Episodic version with sophisticated semantic role extraction."""
    
    SYSTEM_PROMPT = """You are an expert linguist focused on semantic role extraction and relationship mapping given documents in a certain situation. Your primary task is to analyze text to create structured semantic networks anchored by verb phrases, ensuring ALL semantically relevant questions are captured. We will call this task the operator extraction."""
    
    USER_PROMPT_TEMPLATE = """
Given the following:
1. Input chunk from a document

<input_text>
{{input_text}}
</input_text>

2. Background context situating this chunk within the overall document

<background_context>
{{background_context}}
</background_context>

You are required to perform the operator extraction, you should follow the following steps:

Task 1: Actor Identification

Your first task is to identify all actors from the given context. An actor can be:
1. A person (e.g., directors, authors, family members)
2. An organization (e.g., schools, festivals)
3. A place (e.g., cities, countries)
4. A creative work (e.g., films, books)
5. A temporal entity (dates, years)
6. A physical object or item (e.g., artifacts, products) 
7. An abstract entity (e.g., awards, concepts that function as actors)

Guidelines for Actor Extraction:
- Ground actor extraction in the background context (<background_context>).
- Include all mentioned dates as temporal entities
- Do not include phrases or complete sentences
- Extract each actor only once, even if mentioned multiple times
- For temporal entities, note that they will later be assigned:
  * Roles describing their type (e.g., "birth date", "graduation date")
  * States describing their relationship to other entities (e.g., "birth year of [Person]")

Example:
Given: "John Smith received the Golden Globe award for his film at Harvard University in 1995." 
Actors: 
- John Smith (person) 
- Golden Globe (abstract entity/award) 
- film (creative work) 
- Harvard University (organization) 
- 1995 (temporal entity)

Generate a list of all actors from the given context. It is important you do not miss any actors. 


Task 2: Role Assignment

A role is a situation-relevant descriptor (noun phrase) that describes how an actor functions or exists within the context. Roles define the potential relationships an actor can have with other actors.

Guidelines for Role Assignment:
1. For People:
   - Professional roles (e.g., "director", "author")
   - Family roles (e.g., "father", "daughter")
   - Note: One person can have multiple roles

2. For Organizations and Institutions:
   - Functional roles (e.g., "educational institution", "film festival")
   - Purpose roles (e.g., "competition organizer")

3. For Places:
   - Geographic roles (e.g., "city", "location")
   - Functional roles if applicable (e.g., "filming location")

4. For Creative Works:
   - Medium roles (e.g., "film", "novel")
   - Adaptation roles if applicable (e.g., "film adaptation")

5. For Temporal Entities:
   - Type of date in relation to connected entity (e.g., "birth date", "graduation date")
   - Note: These roles should be simple and descriptive

Example:
Given: "John Smith, a professor at Harvard University, published a book in 1995."
Role Assignment:
- John Smith: "professor", "author"
- Harvard University: "educational institution"
- 1995: "publication date"

For each actor identified in Task 1, list all applicable roles from the context.


Task 3: State Identification

A state is a condition or description (using adjectives or verb phrases) that characterizes how an actor exists in their role at a specific point. States provide additional context about the actor's condition, status, or situation.

Guidelines for State Assignment:

1. For People in their Roles:
   - Current status (e.g., "graduated", "married")
   - Actions (e.g., "won award", "directed film")
   - Note: States should capture the descriptors from the original text

2. For Organizations/Institutions in their Roles:
   - Status (e.g., "active", "established")
   - Function (e.g., "awards prizes")

3. For Places in their Roles:
   - Relevant conditions (e.g., "location of graduation")
   - Function in context (e.g., "setting of film")

4. For Creative Works in their Roles:
   - Status (e.g., "adapted from novel", "award-winning")
   - Reception (e.g., "controversial", "bestselling")

5. For Temporal Entities in their Roles:
   - Connection to other entities (e.g., "birth year of [Person]", "release year of [Film]")
   - Note: States for temporal entities should explicitly reference the related actor

Key Rules:
- States must be extracted from the context, not inferred
- States describe the role, not the actor directly
- Multiple states for the same role are possible
- States can include temporal information
- Do NOT include actor names in states except for temporal entities

Example:
Given: "John Smith, a respected professor at Harvard, published a bestselling book in 1995."
State Assignment:
- John Smith
  * Role: professor → State: "respected"
  * Role: author → State: "published bestseller"
- 1995
  * Role: publication date → State: "publication year of bestselling book"

For each role identified in Task 2, list all applicable states from the context.
  
Task 4: Explicit Verb Phrase Identification

Identify key verb phrases (base form) that are **explicitly stated** in the `<input_text>` and directly connect actors identified in Task 1.

Guidelines:

1.  **Find Explicit Verbs:** Scan the `<input_text>` for main verbs or verb phrases where actors from Task 1 serve as subjects, objects, or are linked via prepositions within the same clause or sentence.
2.  **Extract Base Phrase:** Record the base form of these explicitly stated verb phrases (e.g., "survey", "cast", "draw").
3.  **Focus:** Only include verbs explicitly present in the text connecting actors. Do **not** infer verbs or actions in this task.

Example (Conceptual for "The scientist (Dr. Aris) presented findings. The conference needed space."):
- Explicit Verbs Found: "present", "need"
- Extracted Verb Phrase Instances: "present", "need"

Generate a list of explicit verb phrase instances found in the text connecting actors.

Task 4.5: Implicit Action Phrase Inference

Based on the actors (Task 1) and their assigned roles (Task 2), infer additional action phrases that represent relationships implied by the roles and context, but not captured by explicit verbs in Task 4.

Guidelines:

1.  **Analyze Roles and Context:** Examine the actors and their roles from Tasks 1 & 2. Look for combinations where an actor's role strongly implies a specific action connecting it to another related actor present in the context.
2.  **Infer Action Phrase:** If such an implication exists, infer and extract the corresponding implied verb phrase (base form).
    * **Example Inference Rule:** If Actor A (Role: participant) is contextually associated with Actor B (Role: event/workshop), infer the verb phrase "participate in" connecting Actor A (Agent) and Actor B (Location/Event).
    * **Other Potential Inferences:** "author" + "book" -> "write"; "director" + "film" -> "direct"; "resident" + "location" -> "reside in".
3.  **Focus:** Only infer phrases strongly suggested by role combinations and context. Do not guess wildly. List only the inferred verb phrase itself.

Example (Conceptual Context: "Jane Doe [Role: author/writer] ... Book X [Role: novel] ..."):
- Role Combination Found: author/writer (Jane Doe) + novel (Book X)
- -> Inferred Action Phrase Instance: "write"

Generate a list of action phrases inferred from actor roles and context.


Task 5: Prototypical Semantic Role Question Generation (Checklist)

For each verb phrase instance identified in Task 4 (Explicit Verbs) and Task 4.5 (Implicit Actions) (e.g., "<verb_phrase>"), generate a comprehensive set of **generic, prototypical questions** by explicitly attempting to formulate a question for each potential semantic role listed below. These questions act as placeholders (valences).

Instructions:

1.  **Generate Questions via Checklist:** For the given `<verb_phrase>`, attempt to generate a generic question for each of the following core roles. If a role is clearly nonsensical or grammatically impossible for the verb phrase, you may omit the question for that specific role.
    * **Agent Role:** Generate a "Who/What [verb phrase base form]...?" question. (e.g., Who presented?)
    * **Patient/Theme Role:** Generate a "What/Who was [verb phrase past participle]...?" or similar question capturing the entity directly affected by the verb. (e.g., What was presented?)
    * **Location Role:** Generate a "Where did ... [verb phrase]...?" question. (e.g., Where was [something] presented?)
    * **Time Role:** Generate a "When did ... [verb phrase]...?" question. (e.g., When was [something] presented?)
2.  **Consider Other Relevant Roles:** Also consider if other common roles are relevant to the verb phrase's typical meaning. If so, generate appropriate generic questions:
    * **Manner Role:** (If applicable) Generate a "How did ... [verb phrase]...?" question.
    * **Purpose Role:** (If applicable) Generate a "Why did ... [verb phrase]...?" question.
    * *(Consider others like Instrument, Source, Goal only if they seem highly relevant to the verb's core meaning)*
3.  **Generality and Format:**
    * All generated questions MUST remain generic. Do **NOT** contextualize them with specific actors from the text in this Task. Use placeholders like "[something]" or "[someone]" if needed for grammatical sense, but avoid specific names/entities from the context.
    * It is expected that many questions will be answered `None` or `TEXT:` in Task 6. The goal here is comprehensive question generation based on the verb's potential arguments.
    * Use natural phrasing for questions. List the generated questions for the verb phrase.

Example (Conceptual for verb phrase "present"):
- Agent Question: "Who presented?"
- Patient/Theme Question: "What was presented?"
- Location Question: "Where was [something] presented?"
- Time Question: "When was [something] presented?"
- Manner Question: (Potentially applicable) "How was [something] presented?"
- Purpose Question: (Potentially applicable) "Why was [something] presented?"

Generate the complete set of applicable generic prototypical questions for each verb phrase instance from Task 4 and Task 4.5 based on this checklist approach.

Task 6: Answer Mapping and Actor Connection

For each verb phrase instance and its associated generic questions from Task 5, provide answers by consulting the `<input_text>` context associated with the verb phrase instance. Connect answers to actors identified in Task 1 where possible.

Guidelines for Answer Mapping:

1.  **Answer Requirements:**
    * Examine the `<input_text>` to find the answer to each generated question from Task 5.
    * **If the answer corresponds directly to an actor identified in Task 1:** Provide the actor's ID(s) (e.g., `["e1"]`, `["e2", "e3"]`).
    * **If the answer exists in the `<input_text>` but does NOT correspond to any actor identified in Task 1:** Provide the answer as a direct quote or minimal paraphrase from the text, prefixed with `TEXT:` (e.g., `["TEXT:under the harsh glare"]`). Use this for concepts, details, or entities missed in Task 1 but present as answers in the text.
    * **If the answer is NOT available in the `<input_text>`:** Provide `["None"]`.
    * Multiple actors or multiple `TEXT:` answers can be provided for a single question if applicable.
2.  **Answer Types (Informational):**
    * Person/Organization answers typically map to actor IDs.
    * Location answers might map to place actor IDs or `TEXT:` answers if the specific place wasn't an actor.
    * Temporal answers might map to temporal entity actor IDs or `TEXT:` answers.
    * Event/Work/Concept answers might map to creative work/abstract entity actor IDs or `TEXT:` answers.
3.  **Connection Rules:**
    * Each non-`None`, non-`TEXT:` answer must link to an actor ID from Task 1.
    * Maintain all connections derived from the text.
    * Do not infer answers not supported by the context.

Example (Continuing "premiere at" example, assuming context mentions Film Title (e.g., e5) premiered at City Name (e.g., e8), but Agent and Time are unknown):
Verb Phrase: "premiere at"
Questions (from Task 5) and Answers (Task 6):
- "What premiered?" -> Answer: `["e5"]`
- "Who premiered [Something]?" -> Answer: `["None"]`
- "Where did [Something] premiere?" -> Answer: `["e8"]`
- "When did [Something] premiere?" -> Answer: `["None"]`

Example (Conceptual for "cast" - "Where were shadows cast?", assuming context says "under the rugged terrain" and "rugged terrain" is actor e5):
- "Where were shadows cast?" -> Answer: `["e5"]`
Example (Conceptual for "cast" - "Where were shadows cast?", assuming context says "across the field" and "field" was NOT identified as an actor):
- "Where were shadows cast?" -> Answer: `["TEXT:across the field"]`


For each verb phrase instance and its questions from Task 5, provide all possible answers according to these guidelines.


Show all your reasoning and task by task breakdown within <semantic_construction></semantic_construction> and provide the final answer in the following format (Note the updated `answers` possibilities):

```json
{
  "entity_nodes": [
    {
      "id": "e1",
      "name": "<entity>",
      "roles": [
        {
          "role": "<role>",
          "states": ["<state1>", "<state2>"]
        }
      ]
    }
  ],
  "verb_phrase_nodes": [
    {
      "id": "v1",
      "phrase": "<verb_phrase>",
      "questions": [
        {
          "id": "q1",
          "text": "<generic_question>",
          "answers": ["<entity_id>"] // Option 1: Link to existing entity
        },
        {
          "id": "q2",
          "text": "<generic_question_2>",
          "answers": ["None"] // Option 2: No answer in context
        },
        {
          "id": "q3",
          "text": "<generic_question_3>",
          "answers": ["TEXT:answer from text"] // Option 3: Textual answer
        }
      ]
    }
  ]
}
```"""


class SpaceTimePrompts:
    """Prompts for space-time linking."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that is an expert at understanding spatio-temporal relationships between entities. 

You will be given a list of entities along with the context of the narrative in which they appear. 

Your task is to link entities that share a spatio-temporal relationship."""
    
    USER_PROMPT_TEMPLATE = """
**Analyze the Text and Semantic Map for Shared Time/Place**

**Input:**

1.  **Text Chunk:**
    ```
    {text_chunk_content}
    ```
2.  **Operator Output (JSON):**
    ```json
    {operator_output_json}
    ```

**Task:**

Read the `Text Chunk` and examine the entities in the `Operator Output`. Identify groups of entity IDs that share the same location (spatial context) or the same time/date (temporal context) based on the events described.

The entities have the following attributes:

* `id`: (String) The entity ID.
* `name`: (String) The entity name.
* `roles`: A role is a situation-relevant descriptor (noun phrase) that describes how an actor functions or exists within the context. Roles define the potential relationships an actor can have with other actors.
* `states`: A state is a condition or description (using adjectives or verb phrases) that characterizes how an actor exists in their role at a specific point. States provide additional context about the actor's condition, status, or situation. 


**Output Format:**

Return a JSON object with a single key "spatio_temporal_links". The value should be a list of link objects. Each link object must have:

* `linked_entities`: (List of Strings) Entity IDs sharing the context (e.g., `["e1", "e2", "e3"]`).
* `tag_type`: (String) Either "spatial" or "temporal".
* `tag_value`: (String or Null)
    * If the specific location/time/date is mentioned in the `Text Chunk` for this group, extract it.
    * Otherwise, use `null`.

**Example Output Structure:**

```json
{{
"spatio_temporal_links": [
    {{
    "linked_entities": ["e1", "e2", "e3"],
    "tag_type": "spatial",
    "tag_value": "Yosemite National Park"
    }},
    {{
    "linked_entities": ["e5", "e6"],
    "tag_type": "temporal",
    "tag_value": "December 25th 2025"
    }}
]
}}
```
"""

class EventBoundaryPrompts:
    """Prompts for event boundary detection."""
    
    SYSTEM_PROMPT = """You are an expert at analyzing narrative text and identifying where new events begin. 

DEFINITION OF AN EVENT:
An event is a distinct occurrence that has:
- A unique PLACE (location/setting)
- A unique TIME (temporal moment or period)  
- Specific PARTICIPANTS (characters involved)

Your task: Identify sentence indices where new events start in the provided text.

RULES:
1. ONLY return sentence indices that appear in the provided text (look at the [X] numbers)
2. Don't include the first sentence index (text always starts with an event)
3. Mark boundaries when there are changes in:
   - Location/setting (place changes)
   - Time shifts (temporal transitions)
   - Participant groups (different characters become active)
4. SPECIAL ATTENTION TO CONVERSATIONS:
   - Keep complete conversations within the same event/chunk
   - Only mark a boundary AFTER a conversation ends, not during it
   - Look for dialogue end markers like speaker changes to different scenes
5. Be conservative - only mark clear event boundaries
6. Return "NONE" if no clear boundaries exist
7. IMPORTANT: Only use indices that you can see in the brackets [X] in the text below

OUTPUT FORMAT:
Return a JSON list where each item contains:
- "index": sentence number where new event starts
- "event_summary": brief description of the new event starting at that sentence

The text below has explicit sentence indices. For example:
[0] First sentence here.
[1] Second sentence here.
[2] Third sentence here.

Example response: 
[
  {"index": 5, "event_summary": "Characters move to kitchen for breakfast"},
  {"index": 12, "event_summary": "Phone conversation with mother begins"},
  {"index": 18, "event_summary": "Scene shifts to office workplace"}
]

Or return: "NONE" if no clear boundaries exist
"""
    USER_PROMPT_TEMPLATE = """
Analyze this text and identify sentence indices where new events begin:

{text}
"""

class ConversationAnalysisPrompts:
    """Prompts for conversation analysis."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that analyzes conversations and dialogue in text.

You will be given a text chunk that contains dialogue/conversation and the current GSW (semantic map) structure.

Your task is to extract detailed information about the conversation including participants, topics, location, timing, and motivation.
"""

    USER_PROMPT_TEMPLATE = """
**Text Chunk:**
```
{input_data['text_chunk_content']}
```

**Current GSW Structure:**
```json
{json.dumps(input_data['gsw_structure'], indent=2)}
```

**Task:**
Analyze the conversation in the text chunk and extract the following information:

1. **Participants**: Identify who is participating in the conversation (speaking, listening, or present)
2. **Topics**: What/who is being discussed
3. **Context**: When and where the conversation takes place
4. **Purpose**: Why is this conversation happening
5. **Summary**: Brief overview of the conversation
6. **Participant Roles**: How each participant contributes

**Instructions:**
- Use existing entity IDs from the GSW structure when possible
- For new speakers not in the existing entities, you can reference them by name (we'll create entities for them)
- Link to existing space/time nodes when the conversation occurs in an established location/time
- Distinguish between conversation participants (who speak/listen) and conversation topics (what/who is discussed)

**Notes:**
- Set `location_id` and `time_id` to null if no existing space/time nodes apply
- Include entities in `topics_entity` only if they are subjects of discussion, not participants
- In `participants`, use existing entity IDs when available, or speaker names for new entities
- For `new_entities`, infer appropriate roles and states from the conversation context

**Example Response:**
{{
    "conversation_node": {{
        "id": "cv_0",
        "chunk_id": "example_chunk_id",
        "participants": ["e2", "e3", "e4"],
        "topics_entity": ["e1", "e6"],
        "topics_general": ["Chris's medical condition", "war news"],
        "location_id": "sp_0",
        "time_id": "tm_2",
        "motivation": "Mrs Grey tries to inform Kitty of Chris's condition and whereabouts",
        "summary": "Mrs Grey arrives to tell Kitty that Chris is alive in a hospital in Boulogne. Kitty is skeptical and questions her motives.",
        "participant_summaries": {{
            "e2": "Bearer of news; anxious but determined to deliver information",
            "e3": "Wife receiving news; defensive and interrogative", 
            "e4": "Silent observer; embarrassed by the confrontation"
        }}
    }},
    "new_entities": [
        {{
            "name": "Doctor Smith",
            "roles": [
                {{
                    "role": "medical professional",
                    "states": ["treating Chris", "mentioned in conversation"]
                }}
            ]
        }}
    ]
}}
"""

class ConversationDetectionPrompts:
    SYSTEM_PROMPT = """You are a helpful assistant that identifies whether a text chunk contains dialogue or conversation.

You will be given a text chunk and need to determine if it contains any dialogue between characters.

Dialogue includes:
- Direct speech with quotation marks
- Reported dialogue (e.g., "he said that...")
- Back-and-forth exchanges between characters
- Internal monologue presented as dialogue

Your task is to classify whether the chunk is "conversation-heavy" (contains dialogue).
"""

    USER_PROMPT_TEMPLATE = """
**Text Chunk:**
```
{input_data['text_chunk_content']}
```

**Task:**
Analyze the text chunk and determine if it contains dialogue or conversation between characters.

Provide:
- `has_conversation`: True if the chunk contains dialogue/conversation, False otherwise
- `confidence`: Your confidence in this classification (0.0-1.0)
- `reasoning`: Brief explanation for your decision

**Example Response:**
{{
    "has_conversation": true,
    "confidence": 0.9,
    "reasoning": "Contains direct dialogue with quotation marks between multiple characters discussing Chris's condition"
}}
"""
    