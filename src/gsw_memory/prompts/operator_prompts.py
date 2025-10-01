"""
Prompt templates for GSW Operator components.
"""

from enum import Enum


class PromptType(Enum):
    """Enum for different types of operator prompts."""
    EPISODIC = "episodic"
    FACTUAL = "factual"


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
{{  
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "<entity>",
      "roles": [
        {{
          "role": "<role>",
          "states": ["<state1>", "<state2>"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "<verb_phrase>",
      "questions": [
        {{
          "id": "q1",
          "text": "<generic_question>",
          "answers": ["<entity_id>"] // Option 1: Link to existing entity
        }},
        {{
          "id": "q2",
          "text": "<generic_question_2>",
          "answers": ["None"] // Option 2: No answer in context
        }},
        {{
          "id": "q3",
          "text": "<generic_question_3>",
          "answers": ["TEXT:answer from text"] // Option 3: Textual answer
        }}
      ]
    }}
  ]
}}  
```"""


class FactualExtractionPrompts:
    """Prompts for factual GSW generation - optimized for Wikipedia-style content and 2wiki QA."""
    
    SYSTEM_PROMPT = """You are an expert linguist focused on extracting factual relationships and attributes from Wikipedia-style content. Your primary task is to analyze text to create structured semantic networks that capture key factual information such as dates, places, nationalities, and other attributes needed for multi-hop question answering."""
    
    USER_PROMPT_TEMPLATE = """
Given the following text, extract factual relationships and attributes following this structure:

<input_text>
{input_text}
</input_text>

<background_context>
{background_context}
</background_context>

Follow these examples for the desired extraction pattern:

Per-Rule Micro-Examples
1) Atomic entities
- Input: “Ubisoft’s Assassin’s Creed was announced on 29 October 1923 via New York Times with the request of US government.”
- Do: Entities “Ubisoft” (organization), “Assassin’s Creed” (title/work), “29 October 1923” (date), “New York Times” (media), “US government” (government). Phrase: “announced”.
- Don’t: Single entity “Ubisoft’s Assassin’s Creed”. DO NOT pass any entity can be bundled into a single entity.

2) Temporal rules (no fabrication)
- Input: "On 29 October 1923, Parliament passed the Green Act."
- Do: Entity “29 October 1923”. Connect via an event/action: <Green Act> — “occurred during” → “29 October 1923”.

3) Abbreviation & alias
- Input: “German Aerospace Center (DLR) led the study.”
- Do: Entities “German Aerospace Center (DLR)” (org) and “DLR” (alias); phrase “also known as”.
- Don’t: Expand “UN” to “United Nations” if only “UN” appears.

4) Two questions + complete content and recipient
- Input: “Finance Minister Harald Jensen announced to Parliament that the budget would increase on May 19, 1919.”
- Do: Create proposition “the budget would increase”; add phrases with exactly one unknown per question:

  - “announced to ... that” (recipient is the unknown):
    - A→B: To whom did Finance Minister Harald Jensen announce **that the budget would increase on May 19, 1919**?
    - B→A: Who announced to Parliament **that the budget would increase on May 19, 1919**?
  - “announced that ... to” (content is the unknown):
    - A→B: What did Finance Minister Harald Jensen announce to Parliament on May 19, 1919?
    - B→A: Who announced **that the budget would increase on May 19, 1919**?
  - “announced on ... to ... that” (date/time is the unknown):
    - A→B: When did Finance Minister Harald Jensen announce to Parliament **that the budget would increase**?
    - B→A: What did Finance Minister Harald Jensen announce to Parliament **on May 19, 1919**?
    
  - **DO NOT** omit the **that** content in any question.


5) Question format + IDs only
- Input: “On 29 October 1923, Parliament passed the Green Act.”
- Do: “passed”: A→B “Who passed the Green Act on 29 October 1923?” / B→A “What did Parliament pass on 29 October 1923?”; answers are IDs.
- Don’t: “Who passed it?” or answers with names.

6) Universal object/content capture
- Input: “The architect built the museum.”
- Do: “built” (agent→object) and “built by” (object→agent).
- Input: “The minister reported to Cabinet that taxes would rise.”
- Do: “reported to” (recipient) and “reported that” (proposition).

7) Authority/leadership context
- Input: “Parliament passed the Green Act over President Henry Wallace’s veto.”
- Do: “over veto of”; and “in office during”.

8) Special relationships (conditionality, purpose, temporal, comparative)
- Conditional: “… would join NATO if forces were reorganized.” → “conditional on”.
- Purpose: “… launched reforms to reduce inflation.” → proposition “reduce inflation”; “for purpose”.
- Temporal: “… after elections.” → temporal qualifier “after”.

9) Complete content capture
- Input: “The CEO announced that the company would expand operations after securing funding.”
- Do: Proposition includes “after securing funding”.

10) Entity completeness
- Input: “Congress voted on the controversial tax reform bill.”
- Do: Entity “controversial tax reform bill” so answers can reference it by ID.

11) Mandatory connectivity
- Input: "On March 15, 2020, the policy was announced."
- Do: Ensure the date appears in answers via a temporal phrase like "announced on".

12) Document Title Capturing
When a document title represents an alternative name for an entity:
- Put the title in parentheses after the primary entity name
- Create a separate entity node for the title itself
- Connect them with a "known as" relationship
- Input: "Ivan the Terrible \n Ivan IV was a historical figure who ruled Russia."
- Do: Create entity "Ivan IV (Ivan the Terrible)" and separate entity "Ivan the Terrible", then connect them with "known as" relationship.
- Output: Ivan IV (Ivan the Terrible) -> ruled -> Russia (location) , Ivan IV (Ivan the Terrible) -> known as -> Ivan the Terrible


**Follow these examples for the desired extraction pattern:**

#### Example 1: Biographical
**Input:**  
"Clara of Verden (d. 5 April 910) was the daughter of Otto of Saxony, a member of the Brunonen family. In June 889, she married King Rudolf II of Burgundy (880–937) in Lausanne."

**Output:**
```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "Clara of Verden",
      "roles": [
        {{
          "role": "person",
          "states": ["deceased", "historical figure"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "5 April 910",
      "roles": [
        {{
          "role": "date",
          "states": ["death date"]
        }}
      ]
    }},
    {{
      "id": "e3",
      "name": "Otto of Saxony",
      "roles": [
        {{
          "role": "person",
          "states": ["historical figure"]
        }}
      ]
    }},
    {{
      "id": "e4",
      "name": "Brunonen family",
      "roles": [
        {{
          "role": "family",
          "states": ["noble family"]
        }}
      ]
    }},
    {{
      "id": "e5",
      "name": "June 889",
      "roles": [
        {{
          "role": "date",
          "states": ["marriage date"]
        }}
      ]
    }},
    {{
      "id": "e6",
      "name": "Lausanne",
      "roles": [
        {{
          "role": "location",
          "states": ["marriage location"]
        }}
      ]
    }},
    {{
      "id": "e7",
      "name": "Rudolf II of Burgundy",
      "roles": [
        {{
          "role": "person",
          "states": ["ruler", "historical figure"]
        }}
      ]
    }},
    {{
      "id": "e8",
      "name": "King of Burgundy",
      "roles": [
        {{
          "role": "title",
          "states": ["royal title"]
        }}
      ]
    }},
    {{
      "id": "e9",
      "name": "880–937",
      "roles": [
        {{
          "role": "date range",
          "states": ["life span"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "died on",
      "questions": [
        {{
          "id": "q1",
          "text": "Who died on 5 April 910?",
          "answers": ["e1"]
        }},
        {{
          "id": "q2",
          "text": "When did Clara of Verden die?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "daughter of",
      "questions": [
        {{
          "id": "q3",
          "text": "Who is the daughter of Otto of Saxony?",
          "answers": ["e1"]
        }},
        {{
          "id": "q4",
          "text": "Who is Clara of Verden the daughter of?",
          "answers": ["e3"]
        }}
      ]
    }},
    {{
      "id": "v3",
      "phrase": "married",
      "questions": [
        {{
          "id": "q5",
          "text": "Who did Clara of Verden marry in June 889 in Lausanne?",
          "answers": ["e7"]
        }},
        {{
          "id": "q6",
          "text": "Who married Rudolf II of Burgundy in June 889 in Lausanne?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v4",
      "phrase": "married in",
      "questions": [
        {{
          "id": "q7",
          "text": "Where did Clara of Verden marry Rudolf II of Burgundy in June 889?",
          "answers": ["e6"]
        }},
        {{
          "id": "q8",
          "text": "Who married in Lausanne in June 889?",
          "answers": ["e1", "e7"]
        }}
      ]
    }},
    {{
      "id": "v5",
      "phrase": "married on",
      "questions": [
        {{
          "id": "q9",
          "text": "When did Clara of Verden marry Rudolf II of Burgundy in Lausanne?",
          "answers": ["e5"]
        }},
        {{
          "id": "q10",
          "text": "Who married on June 889 in Lausanne?",
          "answers": ["e1", "e7"]
        }}
      ]
    }},
    {{
      "id": "v6",
      "phrase": "member of",
      "questions": [
        {{
          "id": "q11",
          "text": "Which family was Otto of Saxony a member of?",
          "answers": ["e4"]
        }},
        {{
          "id": "q12",
          "text": "Who was a member of the Brunonen family?",
          "answers": ["e3"]
        }}
      ]
    }},
    {{
      "id": "v7",
      "phrase": "holds title",
      "questions": [
        {{
          "id": "q13",
          "text": "Which title did Rudolf II of Burgundy hold?",
          "answers": ["e8"]
        }},
        {{
          "id": "q14",
          "text": "Who held the title King of Burgundy?",
          "answers": ["e7"]
        }}
      ]
    }}
  ]
}}
```

---

#### Example 2: Organization with Abbreviation
**Input:**  
"Orion-7 was a European experimental satellite, part of the Stellar Communications Project between ESA and the German Aerospace Center (DLR)."

**Output:**
```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "Orion-7",
      "roles": [
        {{
          "role": "satellite",
          "states": ["experimental", "European"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "ESA",
      "roles": [
        {{
          "role": "organization",
          "states": ["space agency", "Europe"]
        }}
      ]
    }},
    {{
      "id": "e3",
      "name": "Stellar Communications Project",
      "roles": [
        {{
          "role": "program",
          "states": ["telecommunications experiment"]
        }}
      ]
    }},
    {{
      "id": "e4",
      "name": "German Aerospace Center (DLR)",
      "roles": [
        {{
          "role": "organization",
          "states": ["aerospace", "Germany"]
        }}
      ]
    }},
    {{
      "id": "e5",
      "name": "DLR",
      "roles": [
        {{
          "role": "alias",
          "states": ["abbreviation"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "part of",
      "questions": [
        {{
          "id": "q1",
          "text": "Which program was Orion-7 part of?",
          "answers": ["e3"]
        }},
        {{
          "id": "q2",
          "text": "What was part of the Stellar Communications Project?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "also known as",
      "questions": [
        {{
          "id": "q3",
          "text": "German Aerospace Center (DLR) is also known as what?",
          "answers": ["e5"]
        }},
        {{
          "id": "q4",
          "text": "Which organization is also known as DLR?",
          "answers": ["e4"]
        }}
      ]
    }},
    {{
      "id": "v3",
      "phrase": "collaborated in",
      "questions": [
        {{
          "id": "q5",
          "text": "Which organizations collaborated in the Stellar Communications Project?",
          "answers": ["e2", "e4"]
        }},
        {{
          "id": "q6",
          "text": "In which project did ESA and the German Aerospace Center (DLR) collaborate?",
          "answers": ["e3"]
        }}
      ]
    }}
  ]
}}
```

---

#### Example 3: Statistical + Implicit Info
**Input:**  
"During the 1940s, the U.S. Department of Labor, specifically the Bureau of Labor Statistics (BLS), began collecting employment information via monthly household surveys. 
The unemployment rate has varied from as low as 1% during World War I to as high as 25% during the Great Depression. It later returned to double digits during the 1980s recession."

**Output:**
```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "U.S. Department of Labor",
      "roles": [
        {{
          "role": "organization",
          "states": ["government agency"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "Bureau of Labor Statistics (BLS)",
      "roles": [
        {{
          "role": "organization",
          "states": ["division of Department of Labor"]
        }}
      ]
    }},
    {{
      "id": "e3",
      "name": "BLS",
      "roles": [
        {{
          "role": "alias",
          "states": ["abbreviation"]
        }}
      ]
    }},
    {{
      "id": "e4",
      "name": "1940s",
      "roles": [
        {{
          "role": "time period",
          "states": ["start of survey collection"]
        }}
      ]
    }},
    {{
      "id": "e5",
      "name": "monthly household surveys",
      "roles": [
        {{
          "role": "method",
          "states": ["employment data collection"]
        }}
      ]
    }},
    {{
      "id": "e6",
      "name": "1%",
      "roles": [
        {{
          "role": "rate",
          "states": ["lowest unemployment"]
        }}
      ]
    }},
    {{
      "id": "e7",
      "name": "World War I",
      "roles": [
        {{
          "role": "event",
          "states": ["historical conflict"]
        }}
      ]
    }},
    {{
      "id": "e8",
      "name": "25%",
      "roles": [
        {{
          "role": "rate",
          "states": ["highest unemployment"]
        }}
      ]
    }},
    {{
      "id": "e9",
      "name": "Great Depression",
      "roles": [
        {{
          "role": "event",
          "states": ["economic crisis"]
        }}
      ]
    }},
    {{
      "id": "e10",
      "name": "1980s recession",
      "roles": [
        {{
          "role": "event",
          "states": ["economic downturn"]
        }}
      ]
    }},
    {{
      "id": "e11",
      "name": "double digit unemployment",
      "roles": [
        {{
          "role": "rate",
          "states": ["return of high unemployment"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "began collecting",
      "questions": [
        {{
          "id": "q1",
          "text": "Who began collecting employment information during the 1940s?",
          "answers": ["e1", "e2"]
        }},
        {{
          "id": "q2",
          "text": "What did the Bureau of Labor Statistics (BLS) begin collecting during the 1940s?",
          "answers": ["e5"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "occurred during",
      "questions": [
        {{
          "id": "q3",
          "text": "When did the Bureau of Labor Statistics begin collecting employment information?",
          "answers": ["e4"]
        }},
        {{
          "id": "q4",
          "text": "What began during the 1940s?",
          "answers": ["e1", "e2"]
        }}
      ]
    }},
    {{
      "id": "v3",
      "phrase": "also known as",
      "questions": [
        {{
          "id": "q5",
          "text": "Bureau of Labor Statistics (BLS) is also known as what?",
          "answers": ["e3"]
        }},
        {{
          "id": "q6",
          "text": "Which organization is also known as BLS?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v4",
      "phrase": "lowest unemployment during",
      "questions": [
        {{
          "id": "q7",
          "text": "What was the unemployment rate during World War I?",
          "answers": ["e6"]
        }},
        {{
          "id": "q8",
          "text": "Which event is associated with the 1% unemployment rate?",
          "answers": ["e7"]
        }}
      ]
    }},
    {{
      "id": "v5",
      "phrase": "highest unemployment during",
      "questions": [
        {{
          "id": "q9",
          "text": "What was the unemployment rate during the Great Depression?",
          "answers": ["e8"]
        }},
        {{
          "id": "q10",
          "text": "Which event is associated with the highest unemployment rate of 25%?",
          "answers": ["e9"]
        }}
      ]
    }},
    {{
      "id": "v6",
      "phrase": "returned during",
      "questions": [
        {{
          "id": "q11",
          "text": "What returned during the 1980s recession?",
          "answers": ["e11"]
        }},
        {{
          "id": "q12",
          "text": "During which event did double digit unemployment return?",
          "answers": ["e10"]
        }}
      ]
    }}
  ]
}}
```

---

#### Example 4: Implicit Information (Leadership Context)
**Input:**  
"On 12 May 1955, Prime Minister Harald Jensen reported to Parliament that he would support joining NATO if Denmark’s defense forces were reorganized under a new command.”

**Output:**
```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "12 May 1955",
      "roles": [
        {{
          "role": "date",
          "states": ["historical event"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "Harald Jensen",
      "roles": [
        {{
          "role": "person",
          "states": ["Prime Minister", "Denmark"]
        }}
      ]
    }},
    {{
      "id": "e3",
      "name": "Parliament",
      "roles": [
        {{
          "role": "organization",
          "states": ["political body", "Denmark"]
        }}
      ]
    }},
    {{
      "id": "e4",
      "name": "NATO",
      "roles": [
        {{
          "role": "organization",
          "states": ["defense alliance"]
        }}
      ]
    }},
    {{
      "id": "e5",
      "name": "Denmark's defense forces",
      "roles": [
        {{
          "role": "organization",
          "states": ["military", "Denmark"]
        }}
      ]
    }},
    {{
      "id": "e6",
      "name": "new command structure",
      "roles": [
        {{
          "role": "concept",
          "states": ["military reorganization"]
        }}
      ]
    }},
    {{
      "id": "e7",
      "name": "support joining NATO",
      "roles": [
        {{
          "role": "proposition",
          "states": ["political intention"]
        }}
      ]
    }},
    {{
      "id": "e8",
      "name": "Denmark's defense forces reorganized under a new command",
      "roles": [
        {{
          "role": "proposition",
          "states": ["condition"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "reported to ... that",
      "questions": [
        {{
          "id": "q1",
          "text": "To whom did Prime Minister Harald Jensen report that he would support joining NATO on 12 May 1955?",
          "answers": ["e3"]
        }},
        {{
          "id": "q2",
          "text": "Who reported to Parliament that he would support joining NATO on 12 May 1955?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "reported that ... to",
      "questions": [
        {{
          "id": "q3",
          "text": "What did Prime Minister Harald Jensen report to Parliament on 12 May 1955?",
          "answers": ["e7"]
        }},
        {{
          "id": "q4",
          "text": "Who reported the intention to support joining NATO on 12 May 1955?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v3",
      "phrase": "reported on ... to ... that",
      "questions": [
        {{
          "id": "q7",
          "text": "When did Prime Minister Harald Jensen report to Parliament that he would support joining NATO?",
          "answers": ["e1"]
        }},
        {{
          "id": "q8",
          "text": "What did Prime Minister Harald Jensen report to Parliament on 12 May 1955?",
          "answers": ["e7"]
        }}
      ]
    }},
    {{
      "id": "v4",
      "phrase": "conditional on",
      "questions": [
        {{
          "id": "q5",
          "text": "Prime Minister Harald Jensen's intention to support joining NATO was conditional on what?",
          "answers": ["e8"]
        }},
        {{
          "id": "q6",
          "text": "Which intention was conditional on Denmark's defense forces being reorganized under a new command?",
          "answers": ["e7"]
        }}
      ]
    }}
  ]
}}
```

---

#### Example 5: Contextual Authority / In Office
**Input:**  
"On 2 March 1923, Parliament passed the Green Act, over President Henry Wallace’s veto."

**Output:**
```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "2 March 1923",
      "roles": [
        {{
          "role": "date",
          "states": ["event date"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "Green Act",
      "roles": [
        {{
          "role": "law",
          "states": ["legislation"]
        }}
      ]
    }},
    {{
      "id": "e3",
      "name": "Parliament",
      "roles": [
        {{
          "role": "organization",
          "states": ["legislative body"]
        }}
      ]
    }},
    {{
      "id": "e4",
      "name": "President Henry Wallace",
      "roles": [
        {{
          "role": "person",
          "states": ["head of state"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "passed",
      "questions": [
        {{
          "id": "q1",
          "text": "Who passed the Green Act on 2 March 1923?",
          "answers": ["e3"]
        }},
        {{
          "id": "q2",
          "text": "What did Parliament pass on 2 March 1923?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "passed on",
      "questions": [
        {{
          "id": "q3",
          "text": "When was the Green Act passed by Parliament?",
          "answers": ["e1"]
        }},
        {{
          "id": "q4",
          "text": "What did Parliament pass on 2 March 1923?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v3",
      "phrase": "over veto of",
      "questions": [
        {{
          "id": "q5",
          "text": "Whose veto did Parliament override when passing the Green Act?",
          "answers": ["e4"]
        }},
        {{
          "id": "q6",
          "text": "What was passed over President Henry Wallace's veto?",
          "answers": ["e2"]
        }}
      ]
    }},
    {{
      "id": "v4",
      "phrase": "in office during",
      "questions": [
        {{
          "id": "q7",
          "text": "During whose presidency was the Green Act passed on 2 March 1923?",
          "answers": ["e4"]
        }},
        {{
          "id": "q8",
          "text": "Which law was passed during President Henry Wallace's presidency?",
          "answers": ["e2"]
        }}
      ]
    }}
  ]
}}
```
---

#### Example 6: Fictional Dual Identities / Alter-Egos
**Input:**  
"Shadow Knight: City of Glass is a 2024 American animated superhero direct-to-streaming film produced by Northlight Animation and released by StreamWave. 
It is the third feature in the Vigil Universe Animated Films series. It was released on March 15, 2024. 
The cast includes Alex Rivera as Daniel Cross / Shadow Knight, Priya Shah as Wraith / Lena Kade, Marcus Lee as the Trickster, and Sofia Park as Oracle. 
The screenplay was written by Jordan Quinn, who also wrote the ‘City of Glass’ arc in the monthly Vigil Comics series."

**Output:**
```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "Shadow Knight: City of Glass",
      "roles": [
        {{
          "role": "title/work",
          "states": ["film", "animated", "superhero", "direct-to-streaming"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "2024",
      "roles": [
        {{
          "role": "date",
          "states": ["release year"]
        }}
      ]
    }},
    {{
      "id": "e3",
      "name": "March 15, 2024",
      "roles": [
        {{
          "role": "date",
          "states": ["release date"]
        }}
      ]
    }},
    {{
      "id": "e4",
      "name": "Northlight Animation",
      "roles": [
        {{
          "role": "organization",
          "states": ["producer", "animation studio"]
        }}
      ]
    }},
    {{
      "id": "e5",
      "name": "StreamWave",
      "roles": [
        {{
          "role": "organization",
          "states": ["distributor", "streaming"]
        }}
      ]
    }},
    {{
      "id": "e6",
      "name": "Vigil Universe Animated Films",
      "roles": [
        {{
          "role": "series",
          "states": ["film series"]
        }}
      ]
    }},
    {{
      "id": "e7",
      "name": "third feature",
      "roles": [
        {{
          "role": "number",
          "states": ["ordinal position in series"]
        }}
      ]
    }},
    {{
      "id": "e8",
      "name": "Alex Rivera",
      "roles": [
        {{
          "role": "person",
          "states": ["actor"]
        }}
      ]
    }},
    {{
      "id": "e9",
      "name": "Daniel Cross (Shadow Knight)",
      "roles": [
        {{
          "role": "character",
          "states": ["fictional character", "dual identity"]
        }}
      ]
    }},
    {{
      "id": "e10",
      "name": "Shadow Knight",
      "roles": [
        {{
          "role": "alias",
          "states": ["alter-ego"]
        }}
      ]
    }},
    {{
      "id": "e11",
      "name": "Priya Shah",
      "roles": [
        {{
          "role": "person",
          "states": ["actor"]
        }}
      ]
    }},
    {{
      "id": "e12",
      "name": "Wraith (Lena Kade)",
      "roles": [
        {{
          "role": "character",
          "states": ["fictional character", "dual identity"]
        }}
      ]
    }},
    {{
      "id": "e13",
      "name": "Wraith",
      "roles": [
        {{
          "role": "alias",
          "states": ["alter-ego"]
        }}
      ]
    }},
    {{
      "id": "e14",
      "name": "Marcus Lee",
      "roles": [
        {{
          "role": "person",
          "states": ["actor"]
        }}
      ]
    }},
    {{
      "id": "e15",
      "name": "Trickster",
      "roles": [
        {{
          "role": "character",
          "states": ["fictional character"]
        }}
      ]
    }},
    {{
      "id": "e16",
      "name": "Sofia Park",
      "roles": [
        {{
          "role": "person",
          "states": ["actor"]
        }}
      ]
    }},
    {{
      "id": "e17",
      "name": "Oracle",
      "roles": [
        {{
          "role": "character",
          "states": ["fictional character"]
        }}
      ]
    }},
    {{
      "id": "e18",
      "name": "screenplay",
      "roles": [
        {{
          "role": "title/work",
          "states": ["screenplay"]
        }}
      ]
    }},
    {{
      "id": "e19",
      "name": "Jordan Quinn",
      "roles": [
        {{
          "role": "person",
          "states": ["writer"]
        }}
      ]
    }},
    {{
      "id": "e20",
      "name": "'City of Glass' arc",
      "roles": [
        {{
          "role": "title/work",
          "states": ["comic arc"]
        }}
      ]
    }},
    {{
      "id": "e21",
      "name": "Vigil Comics",
      "roles": [
        {{
          "role": "title/work",
          "states": ["monthly comic series"]
        }}
      ]
    }},
    {{
      "id": "e22",
      "name": "American",
      "roles": [
        {{
          "role": "concept",
          "states": ["nationality"]
        }}
      ]
    }},
    {{
      "id": "e23",
      "name": "animated",
      "roles": [
        {{
          "role": "concept",
          "states": ["medium"]
        }}
      ]
    }},
    {{
      "id": "e24",
      "name": "superhero",
      "roles": [
        {{
          "role": "concept",
          "states": ["genre"]
        }}
      ]
    }},
    {{
      "id": "e25",
      "name": "direct-to-streaming",
      "roles": [
        {{
          "role": "concept",
          "states": ["distribution format"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "has release year",
      "questions": [
        {{
          "id": "q1",
          "text": "What year was Shadow Knight: City of Glass released?",
          "answers": ["e2"]
        }},
        {{
          "id": "q2",
          "text": "Which film was released in 2024?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "released on",
      "questions": [
        {{
          "id": "q3",
          "text": "When was Shadow Knight: City of Glass released?",
          "answers": ["e3"]
        }},
        {{
          "id": "q4",
          "text": "Which film was released on March 15, 2024?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v3",
      "phrase": "produced by",
      "questions": [
        {{
          "id": "q5",
          "text": "Who produced Shadow Knight: City of Glass?",
          "answers": ["e4"]
        }},
        {{
          "id": "q6",
          "text": "What did Northlight Animation produce?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v4",
      "phrase": "released by",
      "questions": [
        {{
          "id": "q7",
          "text": "Who released Shadow Knight: City of Glass?",
          "answers": ["e5"]
        }},
        {{
          "id": "q8",
          "text": "What did StreamWave release?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v5",
      "phrase": "part of series",
      "questions": [
        {{
          "id": "q9",
          "text": "Which series is Shadow Knight: City of Glass part of?",
          "answers": ["e6"]
        }},
        {{
          "id": "q10",
          "text": "Which film is part of the Vigil Universe Animated Films series?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v6",
      "phrase": "position in series",
      "questions": [
        {{
          "id": "q11",
          "text": "What is Shadow Knight: City of Glass's position within the Vigil Universe Animated Films series?",
          "answers": ["e7"]
        }},
        {{
          "id": "q12",
          "text": "Which film is the third feature in the Vigil Universe Animated Films series?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v7",
      "phrase": "has nationality",
      "questions": [
        {{
          "id": "q13",
          "text": "What nationality is Shadow Knight: City of Glass described as?",
          "answers": ["e22"]
        }},
        {{
          "id": "q14",
          "text": "Which film is described as American?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v8",
      "phrase": "is animated",
      "questions": [
        {{
          "id": "q15",
          "text": "What medium describes Shadow Knight: City of Glass?",
          "answers": ["e23"]
        }},
        {{
          "id": "q16",
          "text": "Which film is described as animated?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v9",
      "phrase": "has genre",
      "questions": [
        {{
          "id": "q17",
          "text": "What genre is Shadow Knight: City of Glass described as?",
          "answers": ["e24"]
        }},
        {{
          "id": "q18",
          "text": "Which film is described as a superhero film?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v10",
      "phrase": "has distribution format",
      "questions": [
        {{
          "id": "q19",
          "text": "What distribution format does Shadow Knight: City of Glass use?",
          "answers": ["e25"]
        }},
        {{
          "id": "q20",
          "text": "Which film is described as direct-to-streaming?",
          "answers": ["e1"]
        }}
      ]
    }},
    {{
      "id": "v11",
      "phrase": "stars as",
      "questions": [
        {{
          "id": "q21",
          "text": "Who stars as Daniel Cross (Shadow Knight) in Shadow Knight: City of Glass?",
          "answers": ["e8"]
        }},
        {{
          "id": "q22",
          "text": "Which character does Alex Rivera portray in Shadow Knight: City of Glass?",
          "answers": ["e9"]
        }}
      ]
    }},
    {{
      "id": "v12",
      "phrase": "also known as",
      "questions": [
        {{
          "id": "q23",
          "text": "Daniel Cross (Shadow Knight) is also known as what in Shadow Knight: City of Glass?",
          "answers": ["e10"]
        }},
        {{
          "id": "q24",
          "text": "Which character is also known as Shadow Knight in Shadow Knight: City of Glass?",
          "answers": ["e9"]
        }}
      ]
    }},
    {{
      "id": "v13",
      "phrase": "stars as",
      "questions": [
        {{
          "id": "q25",
          "text": "Who stars as Wraith (Lena Kade) in Shadow Knight: City of Glass?",
          "answers": ["e11"]
        }},
        {{
          "id": "q26",
          "text": "Which character does Priya Shah portray in Shadow Knight: City of Glass?",
          "answers": ["e12"]
        }}
      ]
    }},
    {{
      "id": "v14",
      "phrase": "also known as",
      "questions": [
        {{
          "id": "q27",
          "text": "Wraith (Lena Kade) is also known as what in Shadow Knight: City of Glass?",
          "answers": ["e13"]
        }},
        {{
          "id": "q28",
          "text": "Which character is also known as Wraith in Shadow Knight: City of Glass?",
          "answers": ["e12"]
        }}
      ]
    }},
    {{
      "id": "v15",
      "phrase": "stars as",
      "questions": [
        {{
          "id": "q29",
          "text": "Who stars as the Trickster in Shadow Knight: City of Glass?",
          "answers": ["e14"]
        }},
        {{
          "id": "q30",
          "text": "Which character does Marcus Lee portray in Shadow Knight: City of Glass?",
          "answers": ["e15"]
        }}
      ]
    }},
    {{
      "id": "v16",
      "phrase": "stars as",
      "questions": [
        {{
          "id": "q31",
          "text": "Who stars as Oracle in Shadow Knight: City of Glass?",
          "answers": ["e16"]
        }},
        {{
          "id": "q32",
          "text": "Which character does Sofia Park portray in Shadow Knight: City of Glass?",
          "answers": ["e17"]
        }}
      ]
    }},
    {{
      "id": "v17",
      "phrase": "written by",
      "questions": [
        {{
          "id": "q33",
          "text": "Who wrote the screenplay for Shadow Knight: City of Glass?",
          "answers": ["e19"]
        }},
        {{
          "id": "q34",
          "text": "Which work did Jordan Quinn write for Shadow Knight: City of Glass?",
          "answers": ["e18"]
        }}
      ]
    }},
    {{
      "id": "v18",
      "phrase": "wrote",
      "questions": [
        {{
          "id": "q35",
          "text": "Who wrote the 'City of Glass' arc in Vigil Comics?",
          "answers": ["e19"]
        }},
        {{
          "id": "q36",
          "text": "What did Jordan Quinn write in Vigil Comics?",
          "answers": ["e20"]
        }}
      ]
    }},
    {{
      "id": "v19",
      "phrase": "arc in",
      "questions": [
        {{
          "id": "q37",
          "text": "The 'City of Glass' arc appears in which publication?",
          "answers": ["e21"]
        }},
        {{
          "id": "q38",
          "text": "Which work within Vigil Comics is referenced as an arc?",
          "answers": ["e20"]
        }}
      ]
    }}
  ]
}}
```

**Key Instructions:**

1. **Extract ALL entities**: Include people, places, dates, titles, nationalities, professions, etc.

2. **Create relationship phrases**: These can be:
   - Factual attributes: "born on", "died on", "nationality", "profession"
   - Relationships: "directed by", "married to", "daughter of", "member of"  
   - Properties: "English title", "released in", "located in"

3. **Generate bidirectional questions**: Always create questions from both directions:
   - "Who was born in X?" AND "Where was Y born?"
   - "Who directed X?" AND "What did Y direct?"
   
3. **No pronouns in questions or uncertain objects in questions.**
  - Input: “Michael Jackson stated that his verses are about the convict life in Brasil in his song Care About US”
  - Do: Who talked about the convict life in Brasil in his song Care About US?
  - Don’t: Who talked about the convict life in Brasil in his song?

4. **Capture temporal information**: Ensure dates and temporal relationships are connected to relevant entities.

5. **Include biographical details**: Birth/death dates, family relationships, professions, nationalities.

6. **Include work attributes**: For films, books, etc. capture directors, release dates, genres, etc.

7. **Two questions per relationship phrase but phrases can be repeated for different entities**:
  - Input: "John Smith and Jane Doe were born in New York on 1990 and died in Los Angeles on 2020."
  - Do: "born in" and "died in" for John Smith and Jane Doe separately and 2 questions for each phrase
  - Don't: "born in" and "died in" for John Smith and Jane Doe together and more than 2 questions for each phrase
  

Now extract the factual relationships from the given input text STRICTLY following this pattern:

```json
{{
  "entity_nodes": [
    {{
      "id": "e1",
      "name": "<entity>",
      "roles": [
        {{
          "role": "<entity_type>", 
          "states": ["<general_status>", "<context>"]
        }}
      ]
    }},
    {{
      "id": "e2",
      "name": "<entity>",
      "roles": [
        {{
          "role": "<entity_type>", 
          "states": ["<general_status>", "<context>"]
        }}
      ]
    }}
  ],
  "verb_phrase_nodes": [
    {{
      "id": "v1",
      "phrase": "<relationship_phrase>",
      "questions": [
        {{
          "id": "q1",
          "text": "<bidirectional_question_a_to_b>",
          "answers": ["<entity_id>"]
        }},
        {{
          "id": "q2",
          "text": "<bidirectional_question_b_to_a>",
          "answers": ["<entity_id>"]
        }}
      ]
    }},
    {{
      "id": "v2",
      "phrase": "<relationship_phrase>",
      "questions": [
        {{
          "id": "q3",
          "text": "<bidirectional_question_set_a_to_b>",
          "answers": ["<entity_id>"]
        }},
        {{
          "id": "q4",
          "text": "<bidirectional_question_set_b_to_a>",
          "answers": ["<entity_id>"]
        }},
      ]
    }}
  ]
}}
```

**Guidelines for Roles and States:**
- **Roles**: General entity types (person, location, date, film, title, etc.)
- **States**: Simple status indicators (deceased, historical figure, release year, etc.)
- Keep roles/states general since detailed facts are captured in verb phrase questions
"""

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
