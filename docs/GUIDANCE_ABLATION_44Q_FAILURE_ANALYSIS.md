---
title: Guidance Ablation 44Q Failure Analysis
tags:
  - experiment
  - guidance-ablation
  - curriculum
  - sleep-time
  - obsidian
  - failure-analysis
status: completed
experiment_id: 2wiki_lookup_diverse_44q_b8_aggressive
dataset: 2wikimultihopqa
date: 2026-04-06
models:
  - bedrock/openai.gpt-oss-120b-1:0
---

# Guidance Ablation 44Q Failure Analysis

> [!summary]
> The latest completed 44-question aggressive guidance ablation finished successfully in both arms. Strict bridge tagging succeeded end to end, but `guidance_off` still beat `guidance_on` on the 16-question test split: **F1 0.6471 vs 0.6138**. The dominant remaining issues are bridge fact quality, missing relation-specific kept evidence, and place-granularity mismatches rather than tagging infrastructure.

> [!important] Key Takeaways
> - **75% of test questions (12/16) are answered semantically correctly.** The reported F1 of 0.65 understates quality because 7 questions lose points on paraphrasing (place granularity, name variants, demonym vs country).
> - **Only 4 questions are truly wrong:** 1 name-confusion error (Palo Alto: daughter's name mistaken for father's) and 3 missing-relation bridge gaps.
> - **Guidance had zero effect on 87.5% of test questions** — 14 of 16 produced identical F1 in both arms. On the 2 where it differed, guidance **hurt both times** (Romanoff -0.44, Palo Alto -0.10).
> - The bottleneck is **answer surface form** (paraphrasing), not answer correctness — tagging succeeded 100% in both arms.
> - Guidance reduced bridge volume by 16% (2026 vs 2421 accepted) without improving any test question.

## Summary

This note audits the latest completed 44-question aggressive run at [comparison.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/comparison.json).

Scope:
- `44` total questions in the manifest
- `16` audited test questions
- side-by-side `guidance_on` versus `guidance_off`
- fixed 6-way taxonomy applied once per audited question

Core read:
- both arms completed successfully
- fast threaded sharded generation was active
- strict bridge tagging succeeded for every uncached surface in both arms
- `guidance_off` outperformed `guidance_on` slightly on test F1
- the guided arm produced fewer total accepted bridges
- the bottleneck is still bridge usefulness at retrieval and answer time

## Run Configuration

| Parameter | Value |
| --- | --- |
| Dataset | `2wikimultihopqa` |
| Experiment ID | `2wiki_lookup_diverse_44q_b8_aggressive` |
| Family label | `lookup_diverse_44q` |
| Total questions | `44` |
| Audited test questions | `16` |
| Orchestration mode | `curriculum` |
| Curriculum batch size | `8` |
| Seed batch size | `8` |
| Generation executor | `thread` |
| Parallel enabled | `true` |
| Parallel workers | `6` |
| Model | `bedrock/openai.gpt-oss-120b-1:0` |

Primary artifacts:
- [comparison.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/comparison.json)
- [guidance_on bridge_test_results.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/bridge_test_results.json)
- [guidance_off bridge_test_results.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/bridge_test_results.json)

## Overall Outcome

### Test metrics

| Arm | Test EM | Test F1 |
| --- | ---: | ---: |
| `guidance_on` | 0.25 | 0.6138 |
| `guidance_off` | 0.25 | 0.6471 |

Both arms had bridge usage rate `1.0`, avg retrieved hits `5.0`, avg kept hits `4.5625`.

### Bridge inventory and tagging totals

| Arm | Accepted | Uncached | LLM-tagged | Failed |
| --- | ---: | ---: | ---: | ---: |
| `guidance_on` | 2026 | 2846 | 2846 | 0 |
| `guidance_off` | 2421 | 3990 | 3990 | 0 |

### Per-family test metrics

| Family | ON F1 | OFF F1 | Delta |
| --- | ---: | ---: | ---: |
| `birth_place` | 0.80 | 0.80 | +0.00 |
| `burial_place` | 0.15 | 0.14 | +0.01 |
| `child` | 0.75 | 0.75 | +0.00 |
| `death_place` | 0.67 | 0.67 | +0.00 |
| `maternal_parent` | 1.00 | 1.00 | +0.00 |
| `other` | 0.58 | 0.58 | +0.00 |
| `paternal_parent` | 0.70 | 0.75 | -0.05 |
| `place` | 0.67 | 0.67 | +0.00 |
| `spouse` | 0.50 | 0.50 | +0.00 |
| `title` | 0.00 | 0.44 | **-0.44** |

> [!warning]
> This run used the aggressive width setting: batch size `8` with `6` generation workers on a threaded shard executor. The infrastructure held up, but the quality result still moved slightly against guidance.

## Failure Taxonomy

Primary bucket definitions used in this note:

1. **`bad_bridge_fact`** — top kept bridge evidence directly supports a wrong fact
2. **`missing_bridge_family`** — retrieved or kept evidence lacks the relation family the question needs
3. **`wrong_reranking`** — relevant bridge family exists, but weaker adjacent evidence outranks it in the kept set
4. **`answer_synthesis_error`** — kept evidence supports the gold answer better than the final answer
5. **`eval_granularity_mismatch`** — prediction is semantically right but scored down due to form or granularity mismatch
6. **`mixed_or_unclear`** — no single failure mode dominates, or the case is effectively a clean success

### Taxonomy counts across the 16 audited test questions

| Label | Count | Share | Semantically correct? |
| --- | ---: | ---: | --- |
| `eval_granularity_mismatch` | 5 | 31.2% | Yes — place over-specification |
| `mixed_or_unclear` | 4 | 25.0% | Yes — clean successes |
| `missing_bridge_family` | 3 | 18.8% | **No** — LLM hallucinates from latent knowledge |
| `name_variant_mismatch` | 2 | 12.5% | Yes — birth name vs stage name, demonym vs country |
| `wrong_reranking` | 1 | 6.2% | Yes — same final answer despite bad ranking |
| `bad_bridge_fact` | 1 | 6.2% | **No** — bridge confused daughter with father |
| `answer_synthesis_error` | 0 | 0.0% | — |

Interpretation:
- **12 of 16 questions (75%) are semantically correct.** The system identifies the right entity or place in all but 4 cases.
- Only **4 truly wrong answers**: 1 name-confusion (`bad_bridge_fact`) + 3 missing-relation bridges (`missing_bridge_family`).
- **7 questions** lose F1 due to answer surface form: place granularity (5), name variants (2). These are paraphrasing problems, not knowledge problems.
- There were no `answer_synthesis_error` cases; when the LLM had correct evidence, it produced correct answers.

## Adjusted Accuracy Summary

> [!important] Most F1 loss comes from answer formatting, not answer correctness.

| Category | Count | Semantically correct? | F1 loss cause |
| --- | ---: | --- | --- |
| Perfect (F1 = 1.0) | 4 | Yes | — |
| Correct answer, over-specified place | 5 | Yes | "Hoboken, NJ" vs "Hoboken", "New York, New York" vs "New York", etc. |
| Correct answer, name variant | 1 | Yes | Birth name "Maria del Pilar Cordero" vs stage name "Mapy Cortes" (same person) |
| Correct answer, demonym vs country | 1 | Yes | "English" (demonym) vs "United Kingdom" (country) — same origin, different form |
| Correct answer, buried at wrong rank | 1 | Yes | Right bridge at rank 5 instead of rank 1 (same final answer) |
| Wrong answer from confused name | 1 | **No** | Bridge confused Gian-**Carla** (daughter) with Gian-**Carlo** (father) — different people |
| Wrong answer from missing bridge | 3 | **Partially** | No target-relation bridge; LLM answers from latent knowledge (sometimes close) |

| Metric | Raw | Adjusted (paraphrase-corrected) |
| --- | ---: | ---: |
| Semantically correct answers | 12 / 16 | 12 / 16 |
| Truly wrong answers | 4 / 16 | 4 / 16 |
| Test F1 (guidance_off) | 0.6471 | ~0.85 (if eval accepted name variants + place granularity) |
| Test EM (guidance_off) | 0.25 | ~0.63 (if eval accepted name variants + place granularity) |

**Bottom line:** 12 of 16 test questions (75%) are answered semantically correctly. The reported F1 of 0.65 significantly understates system quality because 7 questions lose points on paraphrasing — place granularity (5), name variants (1), and demonym-vs-country (1). Only 4 questions are truly wrong: 1 name-confusion error (Palo Alto) and 3 missing-relation bridge gaps. The dominant problem is **answer surface form**, not answer correctness.

## Failure Analysis by Category

This section groups the 12 incorrect questions (F1 < 1.0) by failure type for easier pattern recognition.

### Evaluation Artifacts (5 questions) — not real failures

> [!note] These questions are answered **correctly** but penalized by evaluation granularity.
> The system consistently over-specifies place answers by including state/country. A simple answer normalization step would fix all 5.

| Question | Predicted | Gold | F1 |
| --- | --- | --- | ---: |
| Where was the performer of song Come Dance With Me born? | Hoboken, **New Jersey** | Hoboken | 0.50 |
| Where was the director of film The Outlaw Express born? | Santa Rosa, **California** | Santa Rosa | 0.80 |
| What is the place of birth of the director of film The Return Of Swamp Thing? | Glen Cove, **Long Island, New York** | New York | 0.50 |
| Where was the place of death of the director of film The Notorious Miss Lisle? | New York, **New York** | New York | 0.67 |
| Where did the composer of film Camille (1926 Feature Film) die? | Ukiah, **California** | Ukiah | 0.67 |

**Pattern:** All 5 are place-family questions. The retrieval is correct — the top bridge in each case directly answers the target relation with the right city. The only issue is that the bridge and answer include geographic qualifiers (state, country) that the gold answer omits.

**Proposed fix:** Answer normalization — strip state/country suffixes when the question family expects city-level granularity. **Expected impact:** recovers ~0.17 avg F1 across these 5 questions. **Effort:** Low.

### Name / Paraphrasing Mismatches (2 questions) — correct entity, wrong surface form

> [!note] The bridge identifies the **correct entity** but uses a different name form (birth name vs stage name, demonym vs country). These are paraphrasing failures, not factual errors.

#### Romanoff And Juliet — demonym vs country (delta -0.44)

**Question:** Which country the director of film Romanoff And Juliet (1961 Film) is from?

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | England | Peter Ustinov was from the United Kingdom |
| **F1** | 0.00 | 0.44 |
| **Rank-1 bridge** | "What is the nationality of the person who directed Romanoff and Juliet?" → **English** (0.89) | "When was the director of Romanoff and Juliet born?" → 16 April 1921 (0.81) |

**Gold:** `United Kingdom`

**Retrieved evidence (guidance_on):**

| # | Score | Type | Bridge question → Answer |
| ---: | ---: | --- | --- |
| 1 | 0.89 | bridge | What is the nationality of the person who directed Romanoff and Juliet? → **English** |
| 2 | 0.77 | bridge | What is the profession of the director of Romanoff and Juliet? → **radio broadcaster** |
| 3 | 0.71 | bridge | Which award was won by the director of Romanoff and Juliet? → **Emmy Awards** |
| 4 | 0.70 | bridge | Which award did the director of Romanoff and Juliet win? → **BAFTA Awards** |
| 5 | 0.69 | doc_entity | Who directed Romanoff and Juliet? → **Peter Ustinov** |

**LLM response (ON):** `"Answer: England."`
**LLM response (OFF):** `"The director, Peter Ustinov, was from the United Kingdom."`

**Root cause:** The bridge correctly identifies Peter Ustinov's origin but uses the demonym "English" instead of the country "United Kingdom." The corpus describes him as an "English actor" — the bridge faithfully extracted this, but the gold answer expects the country name. This is the same type of issue as the place-granularity mismatches (England ⊂ United Kingdom, just as Hoboken ⊂ New Jersey).

**Why guidance hurt:** Guidance surfaced this demonym bridge at high confidence (0.89), locking the LLM into "England." Without it, the LLM inferred "United Kingdom" from broader context.

#### My Three Merry Widows — birth name vs stage name (delta 0.00)

**Question:** Who is the father of the director of film Palo Alto (2013 Film)?

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | GianCarla 'Gia' Coppola | Christopher Coppola |
| **F1** | 0.40 | 0.50 |
| **Rank-1 bridge** | "Who is the father of the director of Palo Alto?" → Gian-Carla 'Gia' Coppola (0.95) | "Who is the parent of the director of Palo Alto?" → Gian-Carla 'Gia' Coppola (0.90) |

**Gold:** `Gian-Carlo Coppola`

**Retrieved evidence (guidance_on):**

| # | Score | Bridge question → Answer |
| ---: | ---: | --- |
| 1 | 0.95 | Who is the father of the director of Palo Alto? → **Gian-Carla 'Gia' Coppola** |
| 2 | 0.89 | Who is the grandfather of the director of Palo Alto? → **Gian-Carla 'Gia' Coppola** |
| 3 | 0.82 | Who is the film director of the film Palo Alto? → **Gian-Carla 'Gia' Coppola** |
| 4 | 0.82 | What is the nationality of the director of Palo Alto? → **American** |
| 5 | 0.82 | What is the nationality of the director of Palo Alto? → **American** |

**LLM response (ON):** `"Thought: The provided QA pair states that the father of the director of Palo Alto is Gian-Carla 'Gia' Coppola. Therefore, the answer is that person. Answer: Gian-Carla 'Gia' Coppola."`
**LLM response (OFF):** `"Christopher Coppola."`

**Root cause:** This is the only genuinely wrong bridge fact in the audit. Gian-**Carla** 'Gia' Coppola (the director, born 1987) and Gian-**Carlo** Coppola (her father, 1963-1986) are different people with near-identical names — she was named after her father who died before she was born. The bridge confused daughter with father. In guidance_on, the LLM explicitly reasons "the QA pair states the father is Gia Coppola" and trusts it. In guidance_off, the LLM ignores the bridge and guesses "Christopher Coppola" (Gia's uncle), landing slightly closer to gold.

> [!info] Source document verification
> Corpus says Gia Coppola is "the daughter of **Gian-Carlo Coppola**." The correct answer is in the document — the bridge confused the two similar names during generation.

#### My Three Merry Widows — shared wrong spouse (delta 0.00)

**Question:** Who is the spouse of the director of film My Three Merry Widows?

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Maria del Pilar Cordero | Maria del Pilar Cordero |
| **F1** | 0.00 | 0.00 |
| **Rank-1 bridge** | "Who was married to the director of My Three Merry Widows?" → Maria del Pilar Cordero (0.96) | "Who was married to the director of My Three Merry Widows?" → Maria del Pilar Cordero (0.96) |

**Gold:** `Mapy Cortes`

**Retrieved evidence (guidance_on):**

| # | Score | Bridge question → Answer |
| ---: | ---: | --- |
| 1 | 0.96 | Who was married to the director of My Three Merry Widows? → **Maria del Pilar Cordero** |
| 2 | 0.87 | Who worked with the director of the film My Three Merry Widows? → **Resortes** |
| 3 | 0.87 | What was the profession of the director of My Three Merry Widows? → **writer** |
| 4 | 0.83 | What is the nationality of the director of My Three Merry Widows? → **Puerto Rican** |
| 5 | 0.83 | What is the nationality of the director of My Three Merry Widows? → **Puerto Rican** |

**LLM response (ON):** `"Thought: The provided QA pair states that Maria del Pilar Cordero was married to the director of My Three Merry Widows, which directly answers the question about the director's spouse. Answer: Maria del Pilar Cordero."`

**Root cause:** The bridge correctly identifies the director's spouse — María del Pilar Cordero IS Mapy Cortés. The corpus says: "Fernando Cortés married Puerto Rican childhood friend María del Pilar Cordero, who adopted the stage name of **Mapy Cortés**." The bridge stored the birth name; the gold answer uses the stage name. The entity is correct, only the name form differs.

> [!info] Source document verification
> Both answers refer to the correct entities. The bridges faithfully extracted facts from the corpus — the issue is name-form selection.
> - **Romanoff:** Corpus calls Ustinov an "**English** actor" and notes honours from "the **United Kingdom**." The bridge used the demonym; the gold expects the country.
> - **Merry Widows:** Corpus says the spouse is "María del Pilar Cordero, who adopted the stage name of **Mapy Cortés**." The bridge used the birth name; the gold uses the stage name. Same person.

**Proposed fix:** Answer normalization — resolve demonyms to countries (English → United Kingdom) and resolve name aliases using the corpus (birth name → stage name when available). This is the same class of fix as the place-granularity normalization. **Expected impact:** recovers F1 for both. **Effort:** Low-Medium.

### Factual Bridge Error (1 question) — genuinely wrong entity

> [!danger] The bridge confuses two people with near-identical names.

#### Palo Alto — daughter confused with father (delta -0.10)

### Missing Relation Bridges (3 questions) — right entity, wrong relation

> [!warning] The system finds the right entity but no bridge covering the **target relation** survives into kept evidence.

#### Inherent Vice — composer birthplace missing

| Detail | Value |
| --- | --- |
| **Question** | What is the place of birth of the composer of film Inherent Vice (Film)? |
| **Predicted** | Oxford, England |
| **Gold** | Oxford |
| **F1** | 0.67 (both arms) |

**Retrieved evidence (all 5, guidance_on):**

| # | Score | Type | Bridge question → Answer |
| ---: | ---: | --- | --- |
| 1 | 0.66 | bridge | Which film did the writer-director of Inherent Vice also direct? → **There Will Be Blood** |
| 2 | 0.65 | bridge | Who is known as Vice, the name in the title of the novel author of Inherent Vice? → **Eric Aguirre** |
| 3 | 0.65 | bridge | Which film was directed by the person who wrote and directed Inherent Vice? → **There Will Be Blood** |
| 4 | 0.59 | doc_qa | What film did Paul Thomas Anderson write and direct? → **Inherent Vice** |
| 5 | 0.59 | doc_qa | Who wrote and directed Inherent Vice? → **Paul Thomas Anderson** |

**LLM response:** `"Answer: Oxford, England."`

**Root cause:** The system finds Inherent Vice entities but all kept bridges are about the writer-director (Paul Thomas Anderson), not the composer (Jonny Greenwood). None of the 5 evidence items mention Oxford, a composer, or Jonny Greenwood. The answer "Oxford, England" is entirely from the model's latent knowledge — no composer-birthplace bridge was generated or survived retrieval.

#### Tabitha, Stand Up — death-place bridge absent

| Detail | Value |
| --- | --- |
| **Question** | Where was the place of death of the director of film Tabitha, Stand Up? |
| **Predicted** | Berlin, Germany |
| **Gold** | Berlin |
| **F1** | 0.67 (both arms) |

**Retrieved evidence (all 5, guidance_on):**

| # | Score | Bridge question → Answer |
| ---: | ---: | --- |
| 1 | 0.85 | On what date did the director of Tabitha, Stand Up or Tabitha, Arise! die? → **8 March 1972** |
| 2 | 0.84 | When did the director of Tabitha, Stand Up or Tabitha, Arise! die? → **8 March 1972** |
| 3 | 0.82 | Where is the director of Tabitha, Stand Up or Tabitha, Arise! buried? → **Friedhof Heerstraße cemetery** |
| 4 | 0.73 | When was the director of Tabitha, Stand Up or Tabitha, Arise! born? → **23 October 1874** |
| 5 | 0.73 | When was the director of Tabitha, Stand Up or Tabitha, Arise! born? → **23 October 1874** |

**LLM response:** `"Answer: Berlin, Germany."`

**Root cause:** The system keeps death-*date* and burial-*place* bridges but no death-*place* bridge. None of the 5 retrieved evidence items mention Berlin at all. The final answer "Berlin, Germany" is hallucinated from the model's latent knowledge (likely inferring from the Friedhof Heerstraße cemetery being in Berlin), not directly supported by any kept evidence. Also an eval_granularity overlap ("Berlin, Germany" vs "Berlin").

#### Cleopatra of Jerusalem — burial place missing

| Detail | Value |
| --- | --- |
| **Question** | Where was the place of burial of Cleopatra Of Jerusalem's husband? |
| **Gold** | Herodium |
| **F1** | 0.15 / 0.14 |

**Retrieved evidence (all 5, guidance_on):**

| # | Score | Type | Bridge question → Answer |
| ---: | ---: | --- | --- |
| 1 | 0.70 | bridge | Which region was ruled by the husband of Cleopatra of Jerusalem? → **Judea** |
| 2 | 0.69 | bridge | What did the husband of Cleopatra of Jerusalem order? → **Massacre of the Innocents** |
| 3 | 0.64 | bridge | Who was the Patriarch of the city where Cleopatra of Jerusalem was born? → **Basil of Jerusalem** |
| 4 | 0.63 | doc_entity | Who was Cleopatra of Jerusalem married to? → **Herod the Great** |
| 5 | 0.58 | doc_entity | Who was born in Jerusalem? → **Cleopatra of Jerusalem** |

**LLM response (ON):** `"Herod the Great was buried in the Herodium, a tomb and palace complex near Jerusalem."`
**LLM response (OFF):** `"Answer: He was buried at the Herodium, the tomb he built in the Judean desert near Jerusalem."`

**Root cause:** Neither arm keeps a burial-place bridge. None of the 5 evidence items mention burial, Herodium, or a tomb. The evidence only tells the model that the husband ruled Judea and was Herod the Great. The verbose answers containing "Herodium" are entirely from the model's latent knowledge about Herod — making this the most fragile result in the audit and explaining the very low F1 (the model outputs a full sentence instead of just "Herodium").

> [!info] Source document verification
> All 3 gold answers exist in the corpus documents. The correct facts are available — the system failed to generate or retrieve the right bridge.
> - **Inherent Vice:** Corpus says Jonny Greenwood attended school "near **Oxford**, England." No composer-birthplace bridge was generated; all bridges focused on the writer-director Paul Thomas Anderson.
> - **Tabitha, Stand Up:** Corpus says Robert Dinesen is "buried... at the Friedhof Heerstraße cemetery in **Berlin**-Westend." A burial-place bridge was generated, but no death-place bridge. The death location (Berlin) was inferable but never explicitly bridged.
> - **Cleopatra of Jerusalem:** Corpus mentions Herod's construction of "**Herodium**." The burial-place relation was never bridged — only political role bridges (ruled Judea, ordered Massacre) were generated.

**Proposed fix for all 3:** Relation-aware reranking — explicitly boost bridges whose relation label matches the question's target relation. Currently, high-confidence but relation-irrelevant bridges crowd out weaker but relation-correct ones. **Expected impact:** surfaces same-family bridges higher. **Effort:** Medium.

### Evidence Reranking Issue (1 question)

#### Track & Field News — right bridge buried at rank 5

| Detail | Value |
| --- | --- |
| **Question** | Where was the founder of magazine Track & Field News born? |
| **Predicted** | San Diego, California |
| **Gold** | San Diego |
| **F1** | 0.80 (both arms) |

**Retrieved evidence (all 5, guidance_on):**

| # | Score | Bridge question → Answer |
| ---: | ---: | --- |
| 1 | 0.88 | When was the founder of Track & Field News born? → **November 17, 1921** |
| 2 | 0.88 | When was the founder of Track & Field News born? → **November 17, 1921** |
| 3 | 0.79 | Which publication was founded by the founder of Track & Field News? → **Tohoku Mathematical Journal** |
| 4 | 0.76 | What was established by the founder of Track & Field News? → **TAFNEWS Press** |
| 5 | 0.76 | Where was Bert Nelson, founder of Track & Field News, born? → **San Diego, California** |

**Retrieved evidence (all 5, guidance_off):**

| # | Score | Bridge question → Answer |
| ---: | ---: | --- |
| 1 | **0.98** | **Where was the founder of Track & Field News born? → San Diego, California** |
| 2 | 0.88 | When was the founder of Track & Field News born? → November 17, 1921 |
| 3 | 0.88 | When was the founder of Track & Field News born? → November 17, 1921 |
| 4 | 0.87 | When was the person who founded Track & Field News born? → November 17, 1921 |
| 5 | 0.73 | When did the founder of Track & Field News die? → January 9, 1994 |

**LLM response (ON):** `"Thought: The founder of Track & Field News is Bert Nelson, and the provided information states that he was born in San Diego, California. Answer: San Diego, California."`

**Root cause:** Guidance_on retrieves the correct birthplace bridge but buries it at rank 5 (0.76) behind birth-date and publication noise. Guidance_off puts the same type of bridge at rank 1 (0.98). The LLM still recovers the correct city from the rank-5 bridge, but this shows guidance is degrading evidence ordering.

### Correct Despite Weak Evidence (4 questions)

These 4 questions achieved F1 = 1.0 in both arms and are labeled `mixed_or_unclear`. They succeeded but the evidence paths are worth noting:

| Question (short) | Gold | Note |
| --- | --- | --- |
| Death place of Abdul-Aziz's father | Diriyah | Evidence is mostly kinship/death-year context, not a direct death-place bridge |
| Birthplace of husband of Joanna Elisabeth | Dornburg | Clean success — direct spouse-birthplace bridges at rank 1 |
| Detention place of performer of B Boy | SCI Chester | Clean success — direct detention-location bridge at rank 1 |
| Death place of Sancha of Castile's mother | Palencia | Clean success — direct death-place bridge at rank 1 |

## Train Split Analysis (28 questions)

The test split (16 questions) showed guidance had no positive effect. Does the train split tell the same story?

### Train overall metrics

| Arm | Train EM | Train F1 |
| --- | ---: | ---: |
| `guidance_on` | 0.25 | 0.3936 |
| `guidance_off` | 0.25 | 0.4199 |
| **Delta** | **+0.00** | **-0.026** |

Same pattern as test: guidance_off slightly outperforms. Train F1 is lower than test F1 (0.39 vs 0.61) — the train split contains harder questions.

### Questions where guidance changed the outcome

| Direction | Question | Family | ON F1 | OFF F1 | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| **ON wins** | Death place of Maurice, Prince of Orange's father? | `paternal_parent` | 1.00 | 0.67 | +0.33 |
| **ON wins** | Death date of director of Madame La Presidente? | `other` | 0.33 | 0.00 | +0.33 |
| **ON wins** | Burial of Euphemia of Pomerania's father? | `burial_place` | 0.14 | 0.00 | +0.14 |
| **OFF wins** | Child of performer of Me And Bobby McGee? | `child` | 0.00 | **1.00** | **-1.00** |
| **OFF wins** | Workplace of director of A Nest Of Noblemen? | `place` | 0.00 | 0.18 | -0.18 |
| **OFF wins** | Death place of Sylvia Burka's husband? | `spouse` | 0.00 | 0.18 | -0.18 |
| **OFF wins** | Burial of Osthryth's father? | `burial_place` | 0.31 | 0.44 | -0.14 |
| **OFF wins** | Where did Woolsey's father study? | `paternal_parent` | 0.40 | 0.44 | -0.04 |
| No change | 20 other questions | — | — | — | 0.00 |

**Guidance helped 3 train questions** (total +0.80 F1) **but hurt 5** (total -1.54 F1). Net: **-0.74 F1 across train**.

The single biggest swing is Bobby McGee: guidance_on answered "She did not have children" (F1=0.0) while guidance_off correctly answered "Dean Miller" (F1=1.0).

### Train per-family breakdown

| Family | # Q | ON F1 | OFF F1 | Delta | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `birth_place` | 3 | 1.00 | 1.00 | +0.00 | All perfect both arms |
| `burial_place` | 2 | 0.23 | 0.22 | +0.01 | Mixed — one each direction |
| `child` | 4 | 0.29 | 0.54 | **-0.25** | Bobby McGee drives the gap |
| `death_place` | 2 | 0.50 | 0.50 | +0.00 | |
| `maternal_parent` | 2 | 0.50 | 0.50 | +0.00 | |
| `other` | 3 | 0.67 | 0.56 | +0.11 | Only family where ON wins |
| `paternal_parent` | 4 | 0.39 | 0.32 | +0.07 | ON wins (Maurice), but mixed |
| `place` | 2 | 0.00 | 0.09 | -0.09 | Both arms struggle |
| `spouse` | 4 | 0.22 | 0.26 | -0.05 | |
| `title` | 2 | 0.00 | 0.00 | +0.00 | Both arms fail completely |

### Combined train + test verdict

| Split | # Q | ON F1 | OFF F1 | Delta | Guidance helped | Guidance hurt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 16 | 0.6138 | 0.6471 | -0.033 | 0 questions | 2 questions |
| Train | 28 | 0.3936 | 0.4199 | -0.026 | 3 questions | 5 questions |
| **All 44** | **44** | **0.47** | **0.50** | **-0.03** | **3 questions** | **7 questions** |

**Conclusion:** Across all 44 questions, guidance helped 3 and hurt 7. The net F1 effect is -0.03 — guidance is a net negative on this experiment. The pattern is consistent across both splits.

## Impact of Guidance (combined)

| | Test (16 Q) | Train (28 Q) | All (44 Q) |
| --- | ---: | ---: | ---: |
| Guidance helped | 0 | 3 | 3 |
| Guidance hurt | 2 | 5 | 7 |
| No effect | 14 | 20 | 34 |
| Net F1 delta | -0.033 | -0.026 | **-0.03** |

**Conclusion:** Across all 44 questions, guidance helped 3 and hurt 7 — a 1:2.3 ratio against. It reduced bridge volume by 16% (2026 vs 2421 accepted) while delivering a net negative F1 impact on both splits. Guidance is consistently a net negative in this experiment.

## Guidance-Specific Regressions (Detail)

Only 2 of 16 test questions showed any F1 delta between arms. Both are guidance-specific regressions.

### Romanoff And Juliet (delta -0.44)

**Question:** Which country the director of film Romanoff And Juliet (1961 Film) is from?

| Arm | Answer | F1 |
| --- | --- | ---: |
| `guidance_on` | England | 0.00 |
| `guidance_off` | The director, Peter Ustinov, was from the United Kingdom | 0.44 |

**Root cause:** Guidance_on ranked a nationality bridge answering "English" (the adjective) at confidence 0.89. This steered the answer to "England" rather than the gold "United Kingdom".

**Root cause → fix:** Bridge factuality verification should flag "English" vs "United Kingdom" as a nationality/country mismatch.

### Palo Alto (delta -0.10)

**Question:** Who is the father of the director of film Palo Alto (2013 Film)?

| Arm | Answer | F1 |
| --- | --- | ---: |
| `guidance_on` | GianCarla 'Gia' Coppola | 0.40 |
| `guidance_off` | Christopher Coppola | 0.50 |

**Root cause:** Guidance_on ranked a direct "father of director" bridge at confidence 0.95, but that bridge's answer (Gia Coppola) is the director herself, not the father. Higher confidence made the wrong answer more dominant.

**Root cause → fix:** Cross-reference bridge answers against entity roles — if the bridge answer is the same person as the question subject, flag it.

## Systematic Patterns

1. **Direct bridge tagging is no longer the bottleneck.**
   Both arms completed with zero failed uncached surfaces. Aggregate tagging totals are fully clean: `2846/2846` for `guidance_on` and `3990/3990` for `guidance_off`.

2. **Guidance reduced bridge volume without improving the audited test slice.**
   `guidance_on` accepted `2026` bridges vs `guidance_off` accepted `2421` bridges (net difference: `395` more in the unguided arm).

3. **Several partial misses are evaluation-form mismatches, not retrieval failures.**
   `Come Dance With Me`, `The Outlaw Express`, `The Return of Swamp Thing`, `The Notorious Miss Lisle`, and `Camille` all land on a more specific place string than the gold answer. Those cases keep direct relation-correct bridges.

4. **The harder failures are usually evidence-quality failures, not empty retrieval.**
   `My Three Merry Widows` and `Romanoff And Juliet` are driven by bad bridge facts. `Cleopatra Of Jerusalem`, `Tabitha, Stand Up`, and `Inherent Vice` are driven by missing relation-specific kept evidence.

5. **Guidance changed question-pattern labels in a few places, but pattern drift was not the main quality lever.**
   Example: `Romanoff And Juliet` shifts from `director_country.lookup.place` to `director_origin.lookup.place` between arms. That drift matters less than the actual bridge content that gets ranked to the top.

## Why Does Guidance Hurt?

Guidance is "just advisory" — it injects historical context from prior batches into the bridge generation prompt. But it has two mechanical effects that can backfire:

### 1. Budget reallocation starves low-demand edges

With guidance, the system dynamically adjusts how many bridge generation attempts each GSW edge gets:

| | Without guidance | With guidance |
| --- | --- | --- |
| Budget per edge | ~2 calls (uniform) | 2-10 calls (demand-weighted) |
| High-demand edges | 2 calls | up to 10 calls |
| Low-demand edges | 2 calls | **still 2 calls** |
| Total bridges generated | 2421 accepted | 2026 accepted (**-16%**) |

The problem: guidance concentrates generation budget on edges that matched prior batch patterns, but **starves edges that didn't match any prior pattern** — even if those edges contain the answer for the current batch. This explains the 16% fewer total bridges.

### 2. Pattern lock-in from prior batches

Guidance is built from feedback on earlier batches. It tells the generator:
- Which patterns were `[BRIDGE HELPFUL]` in prior batches
- Which patterns `[NEEDS IMPROVEMENT]`
- Exemplar bridges from prior batches to mimic

This creates a **feedback loop**: if early batches happen to succeed on `birth_place` patterns, guidance steers all future batches toward generating more `birth_place` bridges — even if later batches need `burial_place` or `spouse` bridges instead.

### 3. Higher confidence on steered-but-wrong bridges

When the LLM is told "this pattern is in demand, here are exemplar bridges," it generates bridges with higher confidence for those patterns. If the content is wrong (like the "English" nationality bridge at 0.89), the guidance-boosted confidence makes the wrong bridge rank higher and dominate the answer more than it would without guidance.

### Net effect

Guidance trades bridge **diversity** for bridge **focus**. But the focus is based on patterns from prior batches that may not generalize to the current batch's questions. The result: fewer total bridges, and the focused ones aren't better quality — just more confident.

> [!note] This doesn't mean guidance is fundamentally broken.
> The mechanism is sound — the problem is that with only 44 questions across 10 families, prior-batch feedback is too noisy to generalize. Guidance may work better with (a) more questions per family, (b) better edge-demand calibration, or (c) a softer budget reallocation that floors at higher than 2 calls.

## Hypotheses

1. **Guidance trades diversity for focus, but the focus is miscalibrated.** The guided arm is 16% smaller because budget reallocation starves low-demand edges, and the concentrated bridges aren't better quality.

2. **Bridge factuality is still the main precision problem on the hardest misses.** A high-scoring but wrong bridge can dominate the final answer even when retrieval itself is confident.

3. **Relation-aware reranking is still too weak in edge cases.** `Track & Field News` shows that the right bridge can exist but get buried under date or publication noise.

4. **The place-answer evaluation is underselling some semantically correct behavior.** Several place questions miss only because the system returns the city plus state or country.

## Proposed Fixes and Expected Impact

| Fix | Targets | Expected Impact | Effort |
| --- | --- | --- | --- |
| Answer normalization (place granularity) | 5 `eval_granularity` questions | ~+0.17 avg F1; EM recovers for 3-4 | Low |
| Answer normalization (name variants + demonyms) | 2 `name_variant_mismatch` questions | Recovers F1 for Romanoff + Merry Widows | Low-Medium |
| Relation-aware reranking | 1 `wrong_reranking` + 3 `missing_family` | Surface same-family bridges higher in kept set | Medium |
| Bridge entity disambiguation | 1 `bad_bridge_fact` (Palo Alto) | Prevent near-identical name confusion | Medium |
| Run balanced 44Q preset | All 16 test questions | Validate findings before trusting higher-width conclusions | Low |

> [!todo]
> **Priority order:** Answer normalization first (quick win, low effort), then relation-aware reranking (highest potential gain across 4 questions), then bridge factuality verification.

## Test Question Audit (Full Detail)

> [!info]- Expand for the complete per-question audit with evidence
>
> The sections below contain the full side-by-side comparison for all 16 test questions, including top-3 kept evidence with confidence scores.

### Compact summary

| QID | Question | Family | ON F1 | OFF F1 | Delta | Label |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `cda45916` | Father of director of Palo Alto? | `paternal_parent` | 0.40 | 0.50 | -0.10 | `bad_bridge_fact` |
| `b642318c` | Death place of Abdul-Aziz's father? | `paternal_parent` | 1.00 | 1.00 | +0.00 | `mixed_or_unclear` |
| `8e07f1f0` | Spouse of director of My Three Merry Widows? | `spouse` | 0.00 | 0.00 | +0.00 | `name_variant_mismatch` |
| `8f783c7a` | Birthplace of husband of Joanna Elisabeth? | `spouse` | 1.00 | 1.00 | +0.00 | `mixed_or_unclear` |
| `0dfe41f6` | Birthplace of performer of Come Dance With Me? | `child` | 0.50 | 0.50 | +0.00 | `eval_granularity_mismatch` |
| `50e4a65c` | Detention place of performer of B Boy? | `child` | 1.00 | 1.00 | +0.00 | `mixed_or_unclear` |
| `76444678` | Birthplace of director of The Outlaw Express? | `birth_place` | 0.80 | 0.80 | +0.00 | `eval_granularity_mismatch` |
| `8cdc97de` | Birthplace of founder of Track & Field News? | `birth_place` | 0.80 | 0.80 | +0.00 | `wrong_reranking` |
| `7cb81afc` | Birthplace of director of Return Of Swamp Thing? | `other` | 0.50 | 0.50 | +0.00 | `eval_granularity_mismatch` |
| `b672c1ee` | Birthplace of composer of Inherent Vice? | `other` | 0.67 | 0.67 | +0.00 | `missing_bridge_family` |
| `e2de5850` | Death place of director of Notorious Miss Lisle? | `place` | 0.67 | 0.67 | +0.00 | `eval_granularity_mismatch` |
| `38c77480` | Death place of director of Tabitha, Stand Up? | `place` | 0.67 | 0.67 | +0.00 | `missing_bridge_family` |
| `3ab84686` | Death place of composer of Camille (1926)? | `death_place` | 0.67 | 0.67 | +0.00 | `eval_granularity_mismatch` |
| `c91ee648` | Burial place of Cleopatra Of Jerusalem's husband? | `burial_place` | 0.15 | 0.14 | +0.01 | `missing_bridge_family` |
| `96b7c1a6` | Death place of Sancha Of Castile's mother? | `maternal_parent` | 1.00 | 1.00 | +0.00 | `mixed_or_unclear` |
| `8055aca4` | Country of director of Romanoff And Juliet? | `title` | 0.00 | 0.44 | **-0.44** | `name_variant_mismatch` |

### Detailed question-by-question audit

---

#### `cda45916` — Father of the director of Palo Alto

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | GianCarla 'Gia' Coppola | Christopher Coppola |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.40 | 0.50 |
| **Pattern** | `father_of_director.lookup.person` | `father_of_director.lookup.person` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Gian-Carlo Coppola` | **Family:** `paternal_parent` | **Batch:** 0 | **Label:** `bad_bridge_fact`

**Diagnosis:** The top guidance_on bridge explicitly answers the father question with Gia Coppola, and the off-arm evidence is also parent-confused. Guidance_on is worse because its rank-1 bridge directly states the wrong father fact.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_2a11858a` — Who is the father of the director of Palo Alto? → Gian-Carla 'Gia' Coppola *(0.95)*
> 2. `bridge_aaf45404` — Who is the grandfather of the director of Palo Alto? → Gian-Carla 'Gia' Coppola *(0.89)*
> 3. `bridge_fce36cb2` — Who is the film director of the film Palo Alto? → Gian-Carla 'Gia' Coppola *(0.82)*
>
> **Guidance OFF:**
> 1. `bridge_a7b74c52` — Who is the parent of the director of Palo Alto? → Gian-Carla 'Gia' Coppola *(0.90)*
> 2. `bridge_80d32716` — What is the nationality of the director of Palo Alto? → American *(0.82)*
> 3. `bridge_bb5f97f9` — What is the nationality of the director of Palo Alto? → American *(0.82)*

---

#### `b642318c` — Death place of Abdul-Aziz Bin Muhammad's father

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Diriyah | Diriyah |
| **EM** | 1.0 | 1.0 |
| **F1** | 1.00 | 1.00 |
| **Pattern** | `father_place_of_death.lookup.place` | `father_death_place.lookup.place` |
| **Hits** | 5/3 | 5/2 |

**Gold:** `Diriyah` | **Family:** `paternal_parent` | **Batch:** 0 | **Label:** `mixed_or_unclear`

**Diagnosis:** Both arms answer correctly, but the kept evidence is mostly death-year and kinship context rather than a direct father death-place bridge. The outcome is correct, but the path is fragile rather than clean.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_995f04dc` — Who died in the year Imam Abdul Aziz bin Muhammad bin Saud began his rule? → Muhammad ibn Saud ibn Muhammad ibn Muqrin Al-Maridi
> 2. `bridge_86f2740b` — Who died in the year that Imam Abdul Aziz bin Muhammad bin Saud started ruling? → Muhammad ibn Saud ibn Muhammad ibn Muqrin Al-Maridi
> 3. `doc_entity` — Who is Imam Abdul Aziz bin Muhammad bin Saud the son of? → Muhammad bin Saud
>
> **Guidance OFF:**
> 1. `bridge_50f44678` — Who died in the year Imam Abdul Aziz bin Muhammad bin Saud started ruling? → Muhammad ibn Saud ibn Muhammad ibn Muqrin Al-Maridi
> 2. `bridge_d39d8f03` — Who died in the year Imam Abdul Aziz bin Muhammad bin Saud started ruling? → Muhammad ibn Saud ibn Muhammad ibn Muqrin Al-Maridi
> 3. `doc_entity` — Who is Imam Abdul Aziz bin Muhammad bin Saud the son of? → Muhammad bin Saud

---

#### `8e07f1f0` — Spouse of the director of My Three Merry Widows

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Maria del Pilar Cordero | Maria del Pilar Cordero |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.00 | 0.00 |
| **Pattern** | `spouse_of_director.lookup.person` | `spouse_of_director.lookup.person` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Mapy Cortes` | **Family:** `spouse` | **Batch:** 1 | **Label:** `bad_bridge_fact`

**Diagnosis:** Both arms rank a spouse bridge that directly answers with Maria del Pilar Cordero. The top bridge itself conflicts with the gold spouse Mapy Cortes, so the failure is in bridge content, not downstream synthesis.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_b089a10b` — Who was married to the director of My Three Merry Widows? → Maria del Pilar Cordero *(0.96)*
> 2. `bridge_90c04477` — Who worked with the director of the film My Three Merry Widows? → Resortes *(0.87)*
> 3. `bridge_3d21b78a` — What was the profession of the director of My Three Merry Widows? → writer *(0.87)*
>
> **Guidance OFF:**
> 1. `bridge_7d820e6a` — Who was married to the director of My Three Merry Widows? → Maria del Pilar Cordero *(0.96)*
> 2. `bridge_31b26941` — When did the director of My Three Merry Widows marry? → 1932 *(0.90)*
> 3. `bridge_98584c03` — Who worked with the director of My Three Merry Widows? → Resortes *(0.87)*

---

#### `8f783c7a` — Birthplace of husband of Joanna Elisabeth Of Holstein-Gottorp

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Dornburg | Dornburg |
| **EM** | 1.0 | 1.0 |
| **F1** | 1.00 | 1.00 |
| **Pattern** | `husband_birth_place.lookup.place` | `husband_birthplace.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Dornburg` | **Family:** `spouse` | **Batch:** 1 | **Label:** `mixed_or_unclear`

**Diagnosis:** Both arms keep direct spouse-birthplace bridges at the top and answer exactly. Clean success.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_54d603d2` — In which place was the spouse of Johanna Elisabeth of Holstein-Gottorp born? → Dornburg
> 2. `bridge_312bdd92` — Where was the spouse of Johanna Elisabeth of Holstein-Gottorp born? → Dornburg
> 3. `bridge_d483d3e9` — To which house did the husband of Johanna Elisabeth of Holstein-Gottorp belong? → House of Ascania
>
> **Guidance OFF:**
> 1. `bridge_0333b3d7` — Where was the husband of Johanna Elisabeth of Holstein-Gottorp born? → Dornburg
> 2. `bridge_f08138c6` — Where was the spouse of Johanna Elisabeth of Holstein-Gottorp born? → Dornburg
> 3. `bridge_d95afb79` — When was the spouse of Johanna Elisabeth of Holstein-Gottorp born? → 29 November 1690

---

#### `0dfe41f6` — Birthplace of performer of Come Dance With Me

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Hoboken, New Jersey | Hoboken, New Jersey |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.50 | 0.50 |
| **Pattern** | `performer_birth_place.lookup.place` | `performer_birth_place.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Hoboken` | **Family:** `child` | **Batch:** 2 | **Label:** `eval_granularity_mismatch`

**Diagnosis:** The top bridge gives Hoboken, New Jersey while the gold answer is Hoboken. The city is correct; the score drop comes from over-specific formatting.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_d40b2ce2` — Where was the performer of the song "Come Dance With Me" born? → Hoboken, New Jersey *(0.98)*
> 2. `bridge_81639198` — When was the person who performed "Come Dance With Me" born? → December 12, 1915 *(0.87)*
> 3. `bridge_af770c7d` — What is the origin of the parents of the performer of "Come Dance With Me"? → Italian immigrants *(0.82)*
>
> **Guidance OFF:**
> 1. `bridge_ae511e59` — Where was the performer of "Come Dance With Me" born? → Hoboken, New Jersey *(0.97)*
> 2. `bridge_5f027a78` — When was the performer of "Come Dance With Me" born? → December 12, 1915 *(0.88)*
> 3. `bridge_dee1876e` — What is the origin of the parents of the performer of "Come Dance With Me"? → Italian immigrants *(0.82)*

---

#### `50e4a65c` — Detention place of performer of B Boy

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | State Correctional Institution - Chester | State Correctional Institution - Chester |
| **EM** | 1.0 | 1.0 |
| **F1** | 1.00 | 1.00 |
| **Pattern** | `performer_detention_location.lookup.place` | `performer_detention_location.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `State Correctional Institution - Chester` | **Family:** `child` | **Batch:** 2 | **Label:** `mixed_or_unclear`

**Diagnosis:** Both arms keep a direct detention-location bridge at rank 1 and answer exactly. Clean success.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_66e08280` — Where did the performer of B Boy serve five months? → State Correctional Institution - Chester
> 2. `bridge_98bd963c` — Where was the performer of the song B Boy born? → Philadelphia
> 3. `bridge_8c932cf8` — Who is known professionally as the performer of the song B Boy? → Robert Rihmeek Williams
>
> **Guidance OFF:**
> 1. `bridge_acb35004` — Where did the performer of B Boy serve time? → State Correctional Institution - Chester
> 2. `bridge_d2d8a4d0` — Where was the performer of the song B Boy born? → Philadelphia
> 3. `bridge_b3a46da6` — What is the nationality of the performer of the song 'B Boy'? → American

---

#### `76444678` — Birthplace of director of The Outlaw Express

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Santa Rosa, California | Santa Rosa, California |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.80 | 0.80 |
| **Pattern** | `director_birth_place.lookup.place` | `director_birth_place.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Santa Rosa` | **Family:** `birth_place` | **Batch:** 2 | **Label:** `eval_granularity_mismatch`

**Diagnosis:** The direct birthplace bridge is correct but over-specific: Santa Rosa, California versus the gold Santa Rosa.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_a9ace902` — Where was the director of The Outlaw Express born? → Santa Rosa, California *(0.99)*
> 2. `bridge_cc30005f` — When was the director of The Outlaw Express born? → January 4, 1888 *(0.92)*
> 3. `bridge_92ab533e` — When was the director of The Outlaw Express born? → January 4, 1888 *(0.92)*
>
> **Guidance OFF:**
> 1. `bridge_a9ace902` — Where was the director of The Outlaw Express born? → Santa Rosa, California *(0.99)*
> 2. `bridge_92ab533e` — When was the director of The Outlaw Express born? → January 4, 1888 *(0.92)*
> 3. `bridge_6532c1d1` — When was the director of The Outlaw Express born? → January 4, 1888 *(0.92)*

---

#### `8cdc97de` — Birthplace of founder of Track & Field News

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | San Diego, California | San Diego, California |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.80 | 0.80 |
| **Pattern** | `founder_birth_place.lookup.place` | `founder_birth_place.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `San Diego` | **Family:** `birth_place` | **Batch:** 2 | **Label:** `wrong_reranking`

**Diagnosis:** Guidance_on does retrieve the correct birthplace bridge, but only at kept rank 5 behind birth-date and publication noise. Guidance_off puts the birthplace bridge at rank 1.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_4422a9ac` — When was the founder of Track & Field News born? → November 17, 1921 *(0.88)*
> 2. `bridge_1147692e` — When was the founder of Track & Field News born? → November 17, 1921 *(0.88)*
> 3. `bridge_4daeff28` — Which publication was founded by the founder of Track & Field News? → Tohoku Mathematical Journal *(0.79)*
>
> **Guidance OFF:**
> 1. `bridge_3964da86` — Where was the founder of Track & Field News born? → San Diego, California *(0.98)*
> 2. `bridge_be7be7f5` — When was the founder of Track & Field News born? → November 17, 1921 *(0.88)*
> 3. `bridge_1147692e` — When was the founder of Track & Field News born? → November 17, 1921 *(0.88)*

---

#### `7cb81afc` — Birthplace of director of The Return Of Swamp Thing

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Glen Cove, Long Island, New York | Glen Cove, Long Island, New York |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.50 | 0.50 |
| **Pattern** | `director_birth_place.lookup.place` | `director_birth_place.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `New York` | **Family:** `other` | **Batch:** 3 | **Label:** `eval_granularity_mismatch`

**Diagnosis:** Both arms answer Glen Cove, Long Island, New York while the gold answer is New York. The evidence is relation-correct but more specific than the evaluation target.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_de30966f` — Where was the director of The Return of Swamp Thing born? → Glen Cove, Long Island, New York *(0.97)*
> 2. `bridge_a06475a5` — Where was the director of The Return of Swamp Thing born? → Glen Cove, Long Island, New York *(0.97)*
> 3. `bridge_f276b569` — When was the director of The Return of Swamp Thing born? → August 14, 1950 *(0.89)*
>
> **Guidance OFF:**
> 1. `bridge_0bfefcce` — Where was the director of The Return of Swamp Thing born? → Glen Cove, Long Island, New York *(0.97)*
> 2. `bridge_de30966f` — Where was the director of The Return of Swamp Thing born? → Glen Cove, Long Island, New York *(0.97)*
> 3. `bridge_54a436ab` — When was the director of The Return of Swamp Thing born? → August 14, 1950 *(0.89)*

---

#### `b672c1ee` — Birthplace of composer of Inherent Vice

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Oxford, England | Oxford, England |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.67 | 0.67 |
| **Pattern** | `composer_birth_place.lookup.place` | `composer_birth_place.lookup.place` |
| **Hits** | 5/3 | 5/3 |

**Gold:** `Oxford` | **Family:** `other` | **Batch:** 3 | **Label:** `missing_bridge_family`

**Diagnosis:** Neither arm keeps a direct composer-birthplace bridge. The kept evidence is about the writer-director, nominations, and adjacent film metadata, so the required relation family is missing from the final kept set.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_20eb7d04` — Which film did the writerdirector of Inherent Vice also direct? → There Will Be Blood *(0.66)*
> 2. `bridge_58b3eacd` — Who is known as Vice, the name appearing in the title of the novel author of Inherent Vice? → Eric Aguirre *(0.65)*
> 3. `bridge_eb726e34` — Which film was directed by the person who wrote and directed Inherent Vice? → There Will Be Blood *(0.65)*
>
> **Guidance OFF:**
> 1. `bridge_641e730e` — Which film was directed by the director who also wrote and directed Inherent Vice? → There Will Be Blood *(0.64)*
> 2. `bridge_fd325e65` — What other film did the writerdirector of Inherent Vice direct? → There Will Be Blood *(0.63)*
> 3. `bridge_5326764d` — Who was nominated for the Academy Award that Inherent Vice was nominated for? → Jonathan Richard Guy Greenwood *(0.61)*

---

#### `e2de5850` — Death place of director of The Notorious Miss Lisle

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | New York, New York | New York, New York |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.67 | 0.67 |
| **Pattern** | `place_of_death.lookup.place` | `death_place.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `New York` | **Family:** `place` | **Batch:** 3 | **Label:** `eval_granularity_mismatch`

**Diagnosis:** The top death-place bridge answers New York, New York while the gold answer is New York. Evidence quality is good; the miss is a form/granularity issue.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_3a830b61` — In which place did the director of The Notorious Miss Lisle die? → New York, New York *(0.97)*
> 2. `bridge_9caf6325` — When did the director of The Notorious Miss Lisle die? → June 9, 1948 *(0.89)*
> 3. `bridge_a845b12b` — When did the director of The Notorious Miss Lisle die? → June 9, 1948 *(0.89)*
>
> **Guidance OFF:**
> 1. `bridge_e262a4ac` — In which city did the director of The Notorious Miss Lisle die? → New York, New York *(0.95)*
> 2. `bridge_ef91c991` — On what date did the director of The Notorious Miss Lisle die? → June 9, 1948 *(0.89)*
> 3. `bridge_9caf6325` — When did the director of The Notorious Miss Lisle die? → June 9, 1948 *(0.89)*

---

#### `38c77480` — Death place of director of Tabitha, Stand Up

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Berlin, Germany | Berlin, Germany |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.67 | 0.67 |
| **Pattern** | `place_of_death.lookup.place` | `place_of_death.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Berlin` | **Family:** `place` | **Batch:** 3 | **Label:** `missing_bridge_family`

**Diagnosis:** Both arms keep death-date and burial-place bridges but no direct death-place bridge for the director. The final Berlin answer is plausible but inferred, not directly supported.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_ed396e8c` — On what date did the director of Tabitha, Stand Up or Tabitha, Arise! die? → 8 March 1972 *(0.85)*
> 2. `bridge_44421aac` — When did the director of Tabitha, Stand Up or Tabitha, Arise! die? → 8 March 1972 *(0.84)*
> 3. `bridge_f8e47e9e` — Where is the director of Tabitha, Stand Up or Tabitha, Arise! buried? → Friedhof Heerstrae cemetery *(0.82)*
>
> **Guidance OFF:**
> 1. `bridge_44421aac` — When did the director of Tabitha, Stand Up or Tabitha, Arise! die? → 8 March 1972 *(0.84)*
> 2. `bridge_b68c3358` — When did the director of Tabitha, Stand Up or Tabitha, Arise! die? → 8 March 1972 *(0.84)*
> 3. `bridge_c515cbab` — Where is the director of Tabitha, Stand Up or Tabitha, Arise! buried? → Friedhof Heerstrae cemetery *(0.82)*

---

#### `3ab84686` — Death place of composer of Camille (1926)

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Ukiah, California | Ukiah, California |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.67 | 0.67 |
| **Pattern** | `composer_death_place.lookup.place` | `death_place.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Ukiah` | **Family:** `death_place` | **Batch:** 4 | **Label:** `eval_granularity_mismatch`

**Diagnosis:** The top bridge correctly gives Ukiah, California, while the gold answer is Ukiah. Place-granularity mismatch, not a missing evidence problem.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_2ba4f7ad` — Where did the composer of Camille die? → Ukiah, California *(0.90)*
> 2. `bridge_7836f5ea` — When did the composer of Camille die? → February 13, 1959 *(0.83)*
> 3. `bridge_62d041ec` — Where did the composer of Camille retire to? → Laytonville, California *(0.78)*
>
> **Guidance OFF:**
> 1. `bridge_915cf2e5` — In which location did the composer of the film score Camille die? → Ukiah, California *(0.91)*
> 2. `bridge_bc22ee3b` — When did the composer of the film score Camille die? → February 13, 1959 *(0.86)*
> 3. `bridge_b659180c` — Which place did the composer of the film score Camille retire to? → Laytonville, California *(0.80)*

---

#### `c91ee648` — Burial place of Cleopatra Of Jerusalem's husband

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Herod the Great was buried in the Herodium... | He was buried at the Herodium... |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.15 | 0.14 |
| **Pattern** | `burial_place.lookup.place` | `burial_place.lookup.place` |
| **Hits** | 5/3 | 5/4 |

**Gold:** `Herodium` | **Family:** `burial_place` | **Batch:** 4 | **Label:** `missing_bridge_family`

**Diagnosis:** The kept evidence identifies Cleopatra of Jerusalem's husband and his region, but no burial-place bridge survives into the kept set. Both answers are generated from latent knowledge.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_16840b9f` — Which region was ruled by the husband of Cleopatra of Jerusalem? → Judea *(0.70)*
> 2. `bridge_4260f7a6` — What did the husband of Cleopatra of Jerusalem order? → Massacre of the Innocents *(0.69)*
> 3. `bridge_519be99a` — Who was the Patriarch of the city where Cleopatra of Jerusalem was born? → Basil of Jerusalem *(0.64)*
>
> **Guidance OFF:**
> 1. `bridge_3bf93dc9` — What did the husband of Cleopatra of Jerusalem rule? → Judea *(0.74)*
> 2. `bridge_543c24cf` — What did the husband of Cleopatra of Jerusalem order? → Massacre of the Innocents *(0.69)*
> 3. `bridge_2f318442` — Who ruled Judea whose wife was Cleopatra of Jerusalem? → Herod *(0.65)*

---

#### `96b7c1a6` — Death place of mother of Sancha Of Castile

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | Palencia | Palencia |
| **EM** | 1.0 | 1.0 |
| **F1** | 1.00 | 1.00 |
| **Pattern** | `place_of_death.lookup.place` | `mother_place_of_death.lookup.place` |
| **Hits** | 5/5 | 5/5 |

**Gold:** `Palencia` | **Family:** `maternal_parent` | **Batch:** 5 | **Label:** `mixed_or_unclear`

**Diagnosis:** Both arms keep a direct death-place bridge at rank 1 and answer exactly. Clean success.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_0c4a69ec` — In which place did the mother of Sancha of Castile die? → Palencia
> 2. `bridge_32521eda` — Where is the mother of Sancha of Castile buried? → Cathedral of Santiago de Compostela
> 3. `bridge_de9e94c3` — When did the mother of Sancha of Castile die? → January 15, 1149
>
> **Guidance OFF:**
> 1. `bridge_07a1e2ed` — Where did the mother of Sancha of Castile die? → Palencia
> 2. `bridge_32521eda` — Where is the mother of Sancha of Castile buried? → Cathedral of Santiago de Compostela
> 3. `bridge_19b6e571` — When did Sancha of Castile's mother die? → January 15, 1149

---

#### `8055aca4` — Country of origin of director of Romanoff And Juliet

| | Guidance ON | Guidance OFF |
| --- | --- | --- |
| **Answer** | England | Peter Ustinov was from the United Kingdom |
| **EM** | 0.0 | 0.0 |
| **F1** | 0.00 | 0.44 |
| **Pattern** | `director_country.lookup.place` | `director_origin.lookup.place` |
| **Hits** | 5/4 | 5/4 |

**Gold:** `United Kingdom` | **Family:** `title` | **Batch:** 5 | **Label:** `bad_bridge_fact`

**Diagnosis:** Guidance_on ranks a nationality bridge whose answer is English, which steers the final answer toward England. Guidance_off avoids that bridge and produces a closer United Kingdom answer despite weaker direct relation evidence.

> [!example]- Top 3 kept evidence
>
> **Guidance ON:**
> 1. `bridge_2474289e` — What is the nationality of the person who directed Romanoff and Juliet? → English *(0.89)*
> 2. `bridge_4dcae33c` — What is the profession of the director of Romanoff and Juliet? → radio broadcaster *(0.77)*
> 3. `bridge_e5382486` — Which award was won by the director of Romanoff and Juliet? → Emmy Awards *(0.71)*
>
> **Guidance OFF:**
> 1. `bridge_c535c6ee` — When was the director of Romanoff and Juliet born? → 16 April 1921 *(0.81)*
> 2. `bridge_ef3d32ea` — Who directed the film Romanoff and Juliet? → Peter Alexander Ustinov *(0.78)*
> 3. `bridge_7bb32b6a` — What profession does the director of Romanoff and Juliet have? → television presenter *(0.77)*

## Source Artifacts

Run-level artifacts:
- [comparison.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/comparison.json)
- [guidance_on bridge_test_results.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/bridge_test_results.json)
- [guidance_off bridge_test_results.json](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/bridge_test_results.json)

> [!info]- Per-batch query outputs
>
> - [guidance_on batch 0](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_0/query_answer_results.json)
> - [guidance_on batch 1](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_1/query_answer_results.json)
> - [guidance_on batch 2](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_2/query_answer_results.json)
> - [guidance_on batch 3](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_3/query_answer_results.json)
> - [guidance_on batch 4](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_4/query_answer_results.json)
> - [guidance_on batch 5](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_5/query_answer_results.json)
> - [guidance_off batch 0](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_0/query_answer_results.json)
> - [guidance_off batch 1](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_1/query_answer_results.json)
> - [guidance_off batch 2](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_2/query_answer_results.json)
> - [guidance_off batch 3](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_3/query_answer_results.json)
> - [guidance_off batch 4](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_4/query_answer_results.json)
> - [guidance_off batch 5](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_5/query_answer_results.json)

> [!info]- Per-batch bridge-tagging reports
>
> - [guidance_on batch 0](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_0/bridge_pattern_classification_report.json)
> - [guidance_on batch 1](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_1/bridge_pattern_classification_report.json)
> - [guidance_on batch 2](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_2/bridge_pattern_classification_report.json)
> - [guidance_on batch 3](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_3/bridge_pattern_classification_report.json)
> - [guidance_on batch 4](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_4/bridge_pattern_classification_report.json)
> - [guidance_on batch 5](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_5/bridge_pattern_classification_report.json)
> - [guidance_off batch 0](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_0/bridge_pattern_classification_report.json)
> - [guidance_off batch 1](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_1/bridge_pattern_classification_report.json)
> - [guidance_off batch 2](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_2/bridge_pattern_classification_report.json)
> - [guidance_off batch 3](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_3/bridge_pattern_classification_report.json)
> - [guidance_off batch 4](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_4/bridge_pattern_classification_report.json)
> - [guidance_off batch 5](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_5/bridge_pattern_classification_report.json)

> [!info]- Per-batch registry snapshots
>
> - [guidance_on batch 0](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_0/bridge_registry_snapshot.json)
> - [guidance_on batch 1](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_1/bridge_registry_snapshot.json)
> - [guidance_on batch 2](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_2/bridge_registry_snapshot.json)
> - [guidance_on batch 3](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_3/bridge_registry_snapshot.json)
> - [guidance_on batch 4](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_4/bridge_registry_snapshot.json)
> - [guidance_on batch 5](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_on/batch_5/bridge_registry_snapshot.json)
> - [guidance_off batch 0](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_0/bridge_registry_snapshot.json)
> - [guidance_off batch 1](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_1/bridge_registry_snapshot.json)
> - [guidance_off batch 2](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_2/bridge_registry_snapshot.json)
> - [guidance_off batch 3](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_3/bridge_registry_snapshot.json)
> - [guidance_off batch 4](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_4/bridge_registry_snapshot.json)
> - [guidance_off batch 5](../logs/experiments/2wiki_lookup_diverse_44q_b8_aggressive/guidance_off/batch_5/bridge_registry_snapshot.json)
