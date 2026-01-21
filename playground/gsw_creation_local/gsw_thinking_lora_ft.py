#!/usr/bin/env python3
"""
LoRA/DoRA Fine-Tuning Script for GSW Creation with Thinking Traces

This script fine-tunes a language model using LoRA or DoRA to generate
Generative Semantic Workspace (GSW) structures from text, with optional
thinking trace reasoning.

Usage:
    # Train with thinking traces (recommended)
    python gsw_thinking_lora_ft.py \\
        --thinking_traces_path pred_gsws_train_thinking_traces.json \\
        --model_id Qwen/Qwen3-8B \\
        --use_thinking \\
        --num_train_epochs 3

    # Train with DoRA (better performance)
    python gsw_thinking_lora_ft.py \\
        --thinking_traces_path pred_gsws_train_thinking_traces.json \\
        --model_id Qwen/Qwen3-8B \\
        --use_thinking \\
        --use_dora \\
        --num_train_epochs 3

    # Train without thinking traces (faster inference)
    python gsw_thinking_lora_ft.py \\
        --thinking_traces_path pred_gsws_train_thinking_traces.json \\
        --model_id Qwen/Qwen3-8B \\
        --num_train_epochs 3

    # Use specific GPUs (0, 1, 2)
    CUDA_VISIBLE_DEVICES=0,1,2 python gsw_thinking_lora_ft.py \\
        --thinking_traces_path gsw_platinum_dataset.json \\
        --model_id Qwen/Qwen3-8B \\
        --use_thinking \\
        --num_train_epochs 3

    # Test template only
    python gsw_thinking_lora_ft.py \\
        --model_id Qwen/Qwen3-8B \\
        --use_thinking \\
        --test_template_only

    # Push to HuggingFace Hub
    python gsw_thinking_lora_ft.py \\
        --model_id Qwen/Qwen3-8B \\
        --use_thinking \\
        --use_dora \\
        --push_to_hub \\
        --hub_model_id yigitturali/GSW-Operator-Qwen3-8B-Lora-thinking
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# ==============================================================================
# GSW Extraction Prompt (from operator_prompts.py)
# ==============================================================================

GSW_EXTRACTION_PROMPT = """
    Given the following text, extract factual relationships and attributes following this structure:
  <input_text>
  {input_text}
  </input_text>

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
    - Do not encode answer-bearing values only as states; ensure a corresponding entity node exists (e.g., dates/times, locations, numbers/ordinals, titles/works, organizations, concepts) so When/Where/Who/What questions can reference them by ID.

  2. **Create relationship phrases**: These can be:
    - Factual attributes: "born on", "died on", "nationality", "profession"
    - Relationships: "directed by", "married to", "daughter of", "member of"  
    - Properties: "English title", "released in", "located in"

  3. **Generate bidirectional questions**: Always create questions from both directions:
    - "Who was born in X?" AND "Where was Y born?"
    - "Who directed X?" AND "What did Y direct?"
    - QA inversion: If the A→B question contains entity A and its answers are the IDs of entity B, then the B→A question must contain entity B and its answers must be the IDs of entity A. The two questions must swap sides; do not repeat the same side in both questions’ texts or answers.
    
  3. **No pronouns in questions or uncertain objects in questions.**
    - Input: “Michael Jackson stated that his verses are about the convict life in Brasil in his song Care About US”
    - Do: Who talked about the convict life in Brasil in his song Care About US?
    - Don’t: Who talked about the convict life in Brasil in his song?

  4. **Capture temporal information**: Ensure dates and temporal relationships are connected to relevant entities.
    - Attribute lifting via containment: If entity A is specified as part of/from/issued on/by entity B and B carries an explicit attribute in the same passage (e.g., date/time, location, number/ordinal), also attach that attribute to A via an appropriate relation, using the same granularity and only when unambiguous. Do not fabricate precision or lift conflicting attributes.

  5. **Include biographical details**: Birth/death dates, family relationships, professions, nationalities as entities.

  6. **Include work attributes**: For films, books, etc. capture directors, release dates, genres, etc.

  7. **Two questions per relationship phrase but phrases can be repeated for different entities**:
    - Input: "John Smith and Jane Doe were born in New York on 1990 and died in Los Angeles on 2020."
    - Do: "born in" and "died in" for John Smith and Jane Doe separately and 2 questions for each phrase
    - Don't: "born in" and "died in" for John Smith and Jane Doe together and more than 2 questions for each phrase
    

  8. **Ensure entity coverage in QAs**: Every entity node must participate in at least one verb phrase question set — either appearing in the question text or as an answer ID. If an entity would otherwise be orphaned (only present via roles/states), add a minimal factual relation to connect it (e.g., released in [date], located in [place], part of/member of [container], has role/type [concept], known as [alias]). Use only attributes stated in the passage; do not fabricate.


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


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_thinking_traces_data(thinking_traces_path: str) -> tuple:
    """
    Load thinking traces from JSON file.

    Supports three formats:
    - Raw format: data['entries'] array
    - Platinum format: data['samples'] array with quality metadata
    - Matched pairs format: list with 'global_id', 'text', 'gsw' fields

    Args:
        thinking_traces_path: Path to thinking traces JSON file

    Returns:
        Tuple of (entries, is_matched_pairs_format)
        - entries: List of entries with raw_text/text, thinking_trace (optional), and gsw
        - is_matched_pairs_format: Boolean indicating if this is matched pairs format
    """
    print(f"Loading training data from: {thinking_traces_path}")

    with open(thinking_traces_path, 'r') as f:
        data = json.load(f)

    # Detect format: list (matched pairs) or dict (thinking traces)
    if isinstance(data, list):
        print(f"📋 Detected MATCHED PAIRS format")
        print(f"  Total pairs: {len(data)}")

        # Validate that entries have required fields
        valid_entries = []
        for i, entry in enumerate(data):
            if 'text' in entry and 'gsw' in entry:
                valid_entries.append(entry)
            else:
                print(f"  Skipping entry {i}: missing 'text' or 'gsw'")

        print(f"  Valid pairs: {len(valid_entries)}")
        return valid_entries, True

    # Support both formats: 'entries' (raw) and 'samples' (platinum)
    entries = data.get('entries') or data.get('samples', [])

    # Detect format and show statistics
    is_platinum = 'samples' in data
    if is_platinum:
        print(f"✨ Detected PLATINUM dataset format")
        metadata = data.get('metadata', {})
        print(f"  📊 Dataset Statistics:")
        print(f"    - Source samples: {metadata.get('total_samples', 'unknown')}")
        print(f"    - Platinum count: {metadata.get('platinum_count', len(entries))}")
        print(f"    - AI assessed: {metadata.get('ai_assessed_count', 'unknown')}")
        print(f"    - Created at: {metadata.get('created_at', 'unknown')}")

        # Show quality statistics from AI assessments
        ai_assessed = [e for e in entries if 'ai_assessment' in e]
        if ai_assessed:
            avg_score = sum(e['ai_assessment']['overall_score'] for e in ai_assessed) / len(ai_assessed)
            avg_entity = sum(e['ai_assessment']['entity_completeness'] for e in ai_assessed) / len(ai_assessed)
            avg_relationship = sum(e['ai_assessment']['relationship_accuracy'] for e in ai_assessed) / len(ai_assessed)
            avg_format = sum(e['ai_assessment']['format_compliance'] for e in ai_assessed) / len(ai_assessed)
            avg_hallucination = sum(e['ai_assessment']['hallucination_score'] for e in ai_assessed) / len(ai_assessed)

            print(f"  🎯 AI Quality Metrics (average):")
            print(f"    - Overall Score: {avg_score:.1f}/100")
            print(f"    - Entity Completeness: {avg_entity:.2f}")
            print(f"    - Relationship Accuracy: {avg_relationship:.2f}")
            print(f"    - Format Compliance: {avg_format:.2f}")
            print(f"    - Hallucination Score: {avg_hallucination:.2f}")
    else:
        print(f"📝 Detected raw format")

    print(f"Loaded {len(entries)} samples")

    # Filter out samples without raw_text or gsw
    valid_entries = []
    for entry in entries:
        if entry.get('raw_text') and entry.get('gsw'):
            valid_entries.append(entry)
        else:
            print(f"  Skipping entry {entry.get('index', 'unknown')}: missing raw_text or gsw")

    print(f"Valid samples: {len(valid_entries)}")

    return valid_entries, False


# ==============================================================================
# Data Formatting Functions
# ==============================================================================

def create_chat_messages_with_thinking(example: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create chat format with thinking traces.

    Args:
        example: Entry with raw_text, thinking_trace, and gsw

    Returns:
        Dict with 'messages' key containing user and assistant messages
    """
    raw_text = example['raw_text']
    thinking_trace = example.get('thinking_trace', '')
    gsw = example['gsw']

    # Create user prompt
    user_prompt = GSW_EXTRACTION_PROMPT.format(input_text=raw_text)

    # Create assistant response WITH thinking
    gsw_json = json.dumps(gsw, indent=2)

    if thinking_trace:
        assistant_response = f"<think>\n{thinking_trace}\n</think>\n{gsw_json}"
    else:
        # If no thinking trace, just output GSW
        assistant_response = gsw_json

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def create_chat_messages_direct(example: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create chat format without thinking traces (direct GSW output).

    Args:
        example: Entry with raw_text and gsw

    Returns:
        Dict with 'messages' key containing user and assistant messages
    """
    raw_text = example['raw_text']
    gsw = example['gsw']

    # Create user prompt
    user_prompt = GSW_EXTRACTION_PROMPT.format(input_text=raw_text)

    # Create assistant response WITHOUT thinking
    gsw_json = json.dumps(gsw, indent=2)
    assistant_response = gsw_json

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def create_chat_messages_from_matched_pairs(example: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create chat format from matched pairs (text + gsw).

    Args:
        example: Entry with 'text', 'gsw', and optionally 'title' fields

    Returns:
        Dict with 'messages' key containing user and assistant messages
    """
    from gsw_memory.prompts.operator_prompts import FactualExtractionPrompts
    from gsw_memory.memory.models import GSWStructure

    # Get text (may have title prefix)
    text = example['text']
    gsw = example['gsw']

    # Create user prompt using FactualExtractionPrompts
    system_prompt = FactualExtractionPrompts.SYSTEM_PROMPT
    user_prompt = FactualExtractionPrompts.USER_PROMPT_TEMPLATE.format(
        input_text=text,
        background_context=""
    )

    # Serialize GSW to JSON format
    if isinstance(gsw, dict):
        # Convert dict to GSWStructure for proper serialization
        gsw_struct = GSWStructure(**gsw)
        assistant_response = gsw_struct.model_dump_json(indent=4)
    else:
        assistant_response = gsw.model_dump_json(indent=4)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def create_training_dataset(entries: List[Dict[str, Any]], use_thinking: bool = False, validation_split: float = 0.1):
    """
    Create HuggingFace Dataset from thinking traces entries.

    Args:
        entries: List of entries from thinking traces JSON
        use_thinking: Whether to include thinking traces in training
        validation_split: Fraction of data to use for validation (default: 0.1)

    Returns:
        Tuple of (train_dataset, eval_dataset) with 'messages' column
    """
    print("\nCreating training dataset...")
    print(f"  Use thinking traces: {use_thinking}")
    print(f"  Validation split: {validation_split:.1%}")

    # Apply formatting function
    if use_thinking:
        formatted_data = [create_chat_messages_with_thinking(entry) for entry in entries]
    else:
        formatted_data = [create_chat_messages_direct(entry) for entry in entries]

    # Split into train and validation
    split_idx = int(len(formatted_data) * (1 - validation_split))
    train_data = formatted_data[:split_idx]
    eval_data = formatted_data[split_idx:]

    # Create HuggingFace Datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    print(f"Training dataset created:")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")
    print(f"  Column names: {train_dataset.column_names}")
    # breakpoint()

    return train_dataset, eval_dataset


def create_training_dataset_from_matched_pairs(matched_pairs: List[Dict[str, Any]], validation_split: float = 0.1):
    """
    Create HuggingFace Dataset from matched pairs (text + gsw format).

    Args:
        matched_pairs: List of entries with 'text' and 'gsw' fields
        validation_split: Fraction of data to use for validation (default: 0.1)

    Returns:
        Tuple of (train_dataset, eval_dataset) with 'messages' column
    """
    print("\nCreating training dataset from matched pairs...")
    print(f"  Total examples: {len(matched_pairs)}")
    print(f"  Validation split: {validation_split:.1%}")

    # Apply formatting function
    formatted_data = [create_chat_messages_from_matched_pairs(entry) for entry in matched_pairs]

    # Split into train and validation
    split_idx = int(len(formatted_data) * (1 - validation_split))
    train_data = formatted_data[:split_idx]
    eval_data = formatted_data[split_idx:]

    # Create HuggingFace Datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    print(f"Training dataset created:")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")
    print(f"  Column names: {train_dataset.column_names}")

    return train_dataset, eval_dataset


# ==============================================================================
# Template Testing Functions
# ==============================================================================

def load_non_thinking_template(template_path: str):
    """
    Load and fix the non-thinking chat template.

    Args:
        template_path: Path to the Jinja2 template file

    Returns:
        Modified template string with <think> tags removed
    """
    print(f"\nLoading non-thinking template from: {template_path}")

    with open(template_path, 'r') as f:
        template_content = f.read()

    print("✓ No Thinking Template Loaded")

    return template_content


def test_chat_template(tokenizer: Any, training_dataset: Dataset, use_thinking: bool = False, chat_template: str = None):
    """
    Test that the chat template works correctly before training.

    Args:
        tokenizer: Tokenizer instance
        training_dataset: Dataset with 'messages' column
        use_thinking: Whether thinking traces are included
        chat_template: Optional custom chat template string
    """
    print("\nTesting chat template compatibility...")

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("✓ Chat template found!")
        print(f"  Template preview (first 200 chars): {str(tokenizer.chat_template)[:200]}...")
    else:
        print("✗ WARNING: No chat template found! This may cause errors.")
        print("  Consider using an instruct-tuned model variant.")

    # Test with sample
    print("\nTesting formatting with sample data...")
    try:
        sample = training_dataset[0]

        # Apply chat template (use custom template if provided)
        if chat_template:
            formatted = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template
            )
        else:
            formatted = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False
            )

        print("✓ Formatting successful!")
        print(f"  Original message length: {len(str(sample['messages']))}")
        print(f"  Formatted text length: {len(formatted)}")
        print(f"\nFormatted output preview (first 500 chars):")
        print(formatted[:500])
        print("\n... [truncated] ...")
        print(f"\nLast 300 chars:")
        print(formatted[-300:])

        # Check for thinking tags
        if use_thinking:
            if '<think>' in formatted:
                print("\n✓ <think> tags found in formatted output (thinking mode active)")
            else:
                print("\n⚠ WARNING: <think> tags NOT found in formatted output!")
                print("  The template may not support thinking mode.")
        else:
            if '<think>' in formatted:
                print("\n⚠ WARNING: <think> tags found but use_thinking=False")
            else:
                print("\n✓ No <think> tags in formatted output (direct mode)")

    except Exception as e:
        print(f"\n✗ ERROR during formatting: {e}")
        print("  You may need to adjust the formatting function or use a different model.")

    # breakpoint()
    print("\n" + "="*60)
    print("Test complete! Review the output above before training.")
    print("="*60)


# ==============================================================================
# LoRA/DoRA Training Functions
# ==============================================================================

def train(
    model_id: str,
    tokenizer: Any,
    dataset: Dataset,
    training_args: TrainingArguments,
    use_dora: bool = False,
    eval_dataset: Dataset = None,
    chat_template: str = None
):
    """
    Train a model with LoRA or DoRA.

    Args:
        model_id: HuggingFace model identifier
        tokenizer: Tokenizer instance
        dataset: Training dataset with 'messages' column
        training_args: TrainingArguments instance
        use_dora: Whether to use DoRA instead of LoRA
        eval_dataset: Optional evaluation dataset with 'messages' column
        chat_template: Optional custom chat template string

    Returns:
        Trained SFTTrainer instance
    """
    print(f"\nLoading model: {model_id}")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Configuring {'DoRA' if use_dora else 'LoRA'}...")

    # LoRA/DoRA configuration
    lora_config = LoraConfig(
        r=256,                  # Rank
        lora_alpha=512,         # Scaling factor
        lora_dropout=0.05,
        target_modules=[        # Qwen/Llama attention + MLP modules
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora      # Enable DoRA if requested
    )

    def formatting_function(example):
        """Format a single example using the tokenizer's chat template."""
        if chat_template:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template
            )
        else:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )

    print("Initializing trainer...")

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_function,
    )

    print("Starting training...")
    print("="*60)

    # Train
    trainer.train()

    print("\n" + "="*60)
    print("Training complete!")

    return trainer


# ==============================================================================
# LoRA Merge and Push Functions
# ==============================================================================

def merge_and_push(
    adapter_path: str,
    base_model_id: str,
    hub_model_id: str = None,
    output_dir: str = None,
    push_to_hub: bool = False,
    chat_template: str = None
):
    """
    Merge LoRA adapter with base model and optionally push to HuggingFace Hub.

    Args:
        adapter_path: Path to the saved LoRA adapter
        base_model_id: HuggingFace base model ID
        hub_model_id: Target HuggingFace Hub model ID (required if push_to_hub=True)
        output_dir: Local directory to save merged model (optional)
        push_to_hub: Whether to push the merged model to HuggingFace Hub
        chat_template: Optional custom chat template to save with the tokenizer
    """
    from peft import PeftModel

    print("\n" + "="*60)
    print("MERGE LORA ADAPTER WITH BASE MODEL")
    print("="*60)
    print(f"Base model: {base_model_id}")
    print(f"Adapter path: {adapter_path}")
    if push_to_hub:
        print(f"Hub model ID: {hub_model_id}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # Verify adapter path exists
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    # Check for required adapter files
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    print("Step 1: Loading base model...")
    print(f"  Model: {base_model_id}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("  ✓ Base model loaded")

    print("\nStep 2: Loading LoRA adapter...")
    print(f"  Adapter: {adapter_path}")

    # Load adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch.bfloat16
    )
    print("  ✓ Adapter loaded")

    print("\nStep 3: Merging adapter with base model...")
    # Merge and unload adapter weights into the base model
    merged_model = model.merge_and_unload()
    print("  ✓ Adapter merged successfully")

    print("\nStep 4: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    print("  ✓ Tokenizer loaded")

    # Set custom chat template if provided
    if chat_template:
        print("  Setting custom chat template on tokenizer...")
        tokenizer.chat_template = chat_template
        print("  ✓ Custom chat template set")

    # Save locally if output_dir specified
    if output_dir:
        print(f"\nStep 5: Saving merged model locally...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"  Saving to: {output_path}")
        merged_model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        print(f"  ✓ Model saved to {output_path}")

    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("--merged-hub-model-id is required when --push-merged-to-hub is set")

        print(f"\nStep 6: Pushing to HuggingFace Hub...")
        print(f"  Target: {hub_model_id}")
        print("  This may take several minutes depending on model size...")

        # Push model
        merged_model.push_to_hub(
            hub_model_id,
            use_temp_dir=True,
            commit_message="Upload merged LoRA model"
        )
        print("  ✓ Model pushed to Hub")

        # Push tokenizer
        tokenizer.push_to_hub(
            hub_model_id,
            use_temp_dir=True,
            commit_message="Upload tokenizer"
        )
        print("  ✓ Tokenizer pushed to Hub")

        print(f"\n✓ Successfully pushed to: https://huggingface.co/{hub_model_id}")

    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)

    if push_to_hub:
        print(f"\nYour merged model is available at:")
        print(f"  https://huggingface.co/{hub_model_id}")
        print("\nTo use the merged model:")
        print("```python")
        print("from transformers import AutoModelForCausalLM, AutoTokenizer")
        print()
        print(f'model = AutoModelForCausalLM.from_pretrained("{hub_model_id}")')
        print(f'tokenizer = AutoTokenizer.from_pretrained("{hub_model_id}")')
        print("```")
    elif output_dir:
        print(f"\nMerged model saved to: {output_dir}")
        print("\nTo use the merged model:")
        print("```python")
        print("from transformers import AutoModelForCausalLM, AutoTokenizer")
        print()
        print(f'model = AutoModelForCausalLM.from_pretrained("{output_dir}")')
        print(f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')
        print("```")


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA/DoRA Fine-Tuning for GSW Creation with Thinking Traces"
    )

    # Data arguments
    parser.add_argument(
        "--thinking_traces_path",
        type=str,
        default="pred_gsws_train_thinking_traces.json",
        help="Path to thinking traces JSON file"
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model identifier"
    )

    # Training mode arguments
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        help="Train with thinking traces (recommended for better reasoning)"
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        help="Use DoRA instead of LoRA (better performance, slightly slower)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (0.0-1.0)"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Metrics reporting (wandb, tensorboard, none)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory efficiency"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for the wandb/tensorboard run (optional)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gsw_thinking_lora",
        help="Output directory for checkpoints"
    )

    # Testing arguments
    parser.add_argument(
        "--test_template_only",
        action="store_true",
        help="Test chat template without training"
    )

    # HuggingFace Hub arguments
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (e.g., username/model-name)"
    )

    # LoRA merge arguments
    parser.add_argument(
        "--merge_after_training",
        action="store_true",
        help="Merge LoRA adapter with base model after training"
    )
    parser.add_argument(
        "--merge_output_dir",
        type=str,
        default=None,
        help="Directory to save merged model locally"
    )
    parser.add_argument(
        "--push_merged_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--merged_hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID for merged model (e.g., username/merged-model-name)"
    )

    # Non-thinking template arguments
    parser.add_argument(
        "--use_non_thinking_template",
        action="store_true",
        help="Use the non-thinking chat template (removes <think> tags during training)"
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default="playground/gsw_creation_local/qwen3_nonthinking.jinja",
        help="Path to custom chat template file"
    )

    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("GSW THINKING LORA/DORA TRAINING")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Training mode: {'Thinking' if args.use_thinking else 'Direct'}")
    print(f"Adapter type: {'DoRA' if args.use_dora else 'LoRA'}")
    print(f"Thinking traces: {args.thinking_traces_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print("="*60 + "\n")

    # Print GPU info
    print("="*60)
    print("GPU Configuration")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Calculate effective batch size
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * num_gpus
    print(f"\nTraining Configuration:")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Effective batch size: {effective_batch_size}")
    print()

    # Load data
    entries, is_matched_pairs = load_thinking_traces_data(args.thinking_traces_path)

    # Create training and validation datasets based on format
    if is_matched_pairs:
        print("\nUsing matched pairs format (no thinking traces available)")
        training_dataset, eval_dataset = create_training_dataset_from_matched_pairs(
            entries,
            validation_split=args.validation_split
        )
    else:
        print(f"\nUsing thinking traces format (use_thinking={args.use_thinking})")
        training_dataset, eval_dataset = create_training_dataset(
            entries,
            use_thinking=args.use_thinking,
            validation_split=args.validation_split
        )

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token")

    # Load non-thinking chat template if requested
    chat_template = None
    if args.use_non_thinking_template:
        print("\n" + "="*60)
        print("LOADING NON-THINKING CHAT TEMPLATE")
        print("="*60)
        chat_template = load_non_thinking_template(args.chat_template_path)

    # Test template
    test_chat_template(tokenizer, training_dataset, use_thinking=args.use_thinking, chat_template=chat_template)

    if args.test_template_only:
        print("\n✓ Template test complete. Exiting (--test_template_only flag set)")
        return

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,                          # Use bfloat16 on modern GPUs
        fp16=False,                         # Explicitly disable fp16
        save_strategy="steps",              # Save checkpoints at each epoch
        eval_strategy="steps",              # Evaluate at end of each epoch
        eval_steps=10,                      # Evaluate every 100 steps       
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),  # Tensorboard/wandb logs
        warmup_steps=100,
        optim="adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,  # Memory optimization
        save_total_limit=3,                 # Keep only last 3 checkpoints
        report_to=args.report_to,           # Metrics reporting (wandb/tensorboard/none)
        run_name=args.run_name,             # Name for wandb/tensorboard run
        load_best_model_at_end=True,        # Load best model based on eval loss
        metric_for_best_model="eval_loss",  # Use validation loss as metric
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    # Train
    trainer = train(
        model_id=args.model_id,
        tokenizer=tokenizer,
        dataset=training_dataset,
        training_args=training_args,
        use_dora=args.use_dora,
        eval_dataset=eval_dataset,
        chat_template=chat_template
    )

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    print(f"\n✓ Model saved to: {os.path.join(args.output_dir, 'final')}")

    # Push to hub if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            print("\n⚠ WARNING: --push_to_hub requires --hub_model_id")
        else:
            print(f"\nPushing to HuggingFace Hub: {args.hub_model_id}")
            trainer.push_to_hub()
            print("✓ Model pushed to Hub!")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAdapter saved to: {args.output_dir}")

    # Merge adapter with base model if requested
    if args.merge_after_training:
        print("\n" + "="*60)
        print("MERGING ADAPTER WITH BASE MODEL")
        print("="*60)

        adapter_path = os.path.join(args.output_dir, "final")

        merge_and_push(
            adapter_path=adapter_path,
            base_model_id=args.model_id,
            hub_model_id=args.merged_hub_model_id,
            output_dir=args.merge_output_dir,
            push_to_hub=args.push_merged_to_hub,
            chat_template=chat_template
        )

    print("\nTo use the trained model:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("from peft import PeftModel")
    print()
    print(f'base_model = AutoModelForCausalLM.from_pretrained("{args.model_id}")')
    print(f'model = PeftModel.from_pretrained(base_model, "{os.path.join(args.output_dir, "final")}")')
    print(f'tokenizer = AutoTokenizer.from_pretrained("{args.model_id}")')
    print("```")


if __name__ == "__main__":
    main()
