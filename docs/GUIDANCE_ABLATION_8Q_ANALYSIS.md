---
title: Guidance Ablation 8Q Analysis
tags:
  - experiment
  - guidance-ablation
  - curriculum
  - sleep-time
  - obsidian
status: completed
experiment_id: 2wiki_lookup_mixed_guidance_ablation
dataset: 2wikimultihopqa
date: 2026-04-03
models:
  - bedrock/openai.gpt-oss-120b-1:0
---

# Guidance Ablation 8Q Analysis

> [!summary]
> Latest completed 8-question guidance ablation finished successfully in both arms. Fast threaded sharded generation and strict LLM bridge tagging both worked. Guidance still produced no measurable test-quality gain on this slice.

## Summary

This note analyzes the latest completed run of the 8-question mixed lookup guidance ablation.

Core result:
- `guidance_on` and `guidance_off` reached identical test performance.
- The guided arm generated fewer accepted bridges.
- Bridge tagging is no longer the bottleneck or failure mode in this run.
- The remaining problem is bridge usefulness at retrieval / answer time.

## Run Configuration

- Dataset: `2wikimultihopqa`
- Family label: `mixed_lookup_director_birth_place_and_nationality`
- Questions: `8`
- Orchestration mode: `curriculum`
- Curriculum batch size: `2`
- Seed batch size: `2`
- Generation mode: `question_group_sharded`
- Generation executor: `thread`
- Generation parallel workers: `4`
- Model: `bedrock/openai.gpt-oss-120b-1:0`

Primary artifacts:
- [comparison.json](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/comparison.json)
- [guidance_on results](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/bridge_test_results.json)
- [guidance_off results](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/bridge_test_results.json)

## Main Findings

### 1. No measurable guidance lift on test performance

| Arm | Test EM | Test F1 | Bridge usage rate |
| --- | --- | --- | --- |
| `guidance_on` | `0.25` | `0.5417` | `1.0` |
| `guidance_off` | `0.25` | `0.5417` | `1.0` |

Per-family test metrics were also identical:

| Family | EM | F1 |
| --- | --- | --- |
| `director_birth_place` | `0.0` | `0.5833` |
| `director_nationality` | `0.5` | `0.5` |

### 2. Guidance changed the run trajectory, but not the outcome

| Arm | Total accepted bridges |
| --- | --- |
| `guidance_on` | `336` |
| `guidance_off` | `380` |

Per-batch accepted bridge additions:

| Batch | Guidance ON | Guidance OFF |
| --- | --- | --- |
| 0 | `47` | `51` |
| 1 | `81` | `89` |
| 2 | `71` | `89` |
| 3 | `137` | `151` |

Interpretation:
- guidance is affecting what gets generated
- but the altered bridge inventory is not improving downstream answers on this slice

### 3. Strict bridge tagging succeeded end-to-end

Across all four batches, both arms show:
- `status = ok`
- `failed_surface_count = 0`
- `llm_tagged_surface_count == uncached_surface_count`

Aggregate uncached surface tagging totals:

| Arm | Uncached surfaces | LLM-tagged | Failed |
| --- | --- | --- | --- |
| `guidance_on` | `407` | `407` | `0` |
| `guidance_off` | `494` | `494` | `0` |

Representative reports:
- [guidance_on batch 0](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/batch_0/bridge_pattern_classification_report.json)
- [guidance_on batch 3](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/batch_3/bridge_pattern_classification_report.json)
- [guidance_off batch 0](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/batch_0/bridge_pattern_classification_report.json)
- [guidance_off batch 3](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/batch_3/bridge_pattern_classification_report.json)

> [!warning]
> This completed run predates the newer fallback-retry counters. The evidence for success here is `uncached_surface_count == llm_tagged_surface_count` and `failed_surface_count == 0`, not the presence of fallback fields.

## Evidence

### Test-set outcomes

| Question | Guidance ON | Guidance OFF | Notes |
| --- | --- | --- | --- |
| `Peter's Friends` birth place | `Belfast, Northern Ireland.` | `Belfast, Northern Ireland.` | Same over-specific answer, `F1 = 0.5` |
| `Overland Adventure` nationality | `Australian.` | `Australian.` | Clean success, `F1 = 1.0` |
| `The Trail of the Lonesome Pine` birth place | `Richmond, Virginia.` | `Richmond, Virginia.` | Near miss, `F1 = 0.6667` |
| `3096 Days` nationality | `German-American.` | `German-American.` | Same wrong answer, `F1 = 0.0` |

### Retrieval signal was present but not discriminative enough

Both arms show on the test split:
- average retrieved bridge hit count: `5.0`
- average kept bridge hit count: `5.0`
- bridge usage rate: `1.0`

Interpretation:
- the system is retrieving bridge evidence consistently
- failure is not "no bridge found"
- failure is "wrong bridge family / wrong bridge content / wrong answer synthesis despite bridge presence"

## Failure Analysis

### Case 1: `Peter's Friends`

Question:
- `What is the place of birth of the director of film Peter'S Friends?`

Outcome:
- prediction: `Belfast, Northern Ireland.`
- gold: `Belfast`
- `EM = 0.0`, `F1 = 0.5`

Top evidence in `guidance_on`:
1. `When was the director of Peter's Friends born?` (`0.8559`)
2. `When was the director of Peter's Friends born?` (`0.8559`)
3. `What is the nationality of the director of Peter's Friends?` (`0.8349`)

Top evidence in `guidance_off` is materially the same family mix.

Assessment:
- retrieval is relation-loose here
- the top bridge evidence is birth-date / nationality adjacent, not a clean birth-place bridge
- the final answer still contains the right city, but with extra specificity
- likely issue: reranking and answer normalization, not bridge tagging

### Case 2: `3096 Days`

Question:
- `What nationality is the director of film 3096 Days?`

Outcome:
- prediction: `German-American.`
- gold: `American`
- `EM = 0.0`, `F1 = 0.0`

Top evidence in `guidance_on`:
1. `What is the nationality of the director of 3096 Days?` (`0.9797`)
2. `In which country was the director of 3096 Days born?` (`0.9082`)
3. `When was the director of 3096 Days born?` (`0.8739`)

Top evidence in `guidance_off`:
1. `Which nationality does the director of 3096 Days have?` (`0.9816`)
2. `What is the nationality of the director of 3096 Days?` (`0.9801`)
3. `What is the nationality of the director of 3096 Days?` (`0.9798`)

Assessment:
- retrieval found highly similar nationality bridges in both arms
- the final answer is still wrong
- likely issue is either:
  - bad bridge content in the registry, or
  - answer synthesis trusting a wrong nationality bridge
- this is not a guidance failure alone; both arms converge on the same wrong evidence pattern

## Hypotheses

1. **Guidance is changing bridge volume more than bridge utility**
   - fewer bridges are generated in the guided arm
   - but the added selectivity is not improving test retrieval quality

2. **Retriever/reranker is not relation-strict enough**
   - birth-place questions are still keeping birth-date and nationality bridges near the top
   - semantic similarity is overpowering relation alignment

3. **Some high-scoring bridges are factually wrong or too coarse**
   - `3096 Days` suggests the system can retrieve a very confident but wrong nationality bridge
   - answer generation then amplifies that error

4. **Bridge presence is no longer the main blocker**
   - bridge coverage and tagging both look healthy on this run
   - remaining failures are downstream of generation/tagging

## Next Experiments

> [!todo]
> Focus the next iteration on evidence quality, not on more bridge-tagging work.

1. Add relation-aware reranking features so birth-place questions prefer birth-place bridges over adjacent relations like birth-date or nationality.
2. Audit the retrieved bridges for `3096 Days` to determine whether the wrong answer comes from bad bridge content or bad answer synthesis.
3. Add a small answer-normalization check for place granularity mismatches like `Belfast` vs `Belfast, Northern Ireland`.
4. Run a targeted analysis on the top kept bridges for the failed test questions and classify each failure as:
   - bad bridge fact
   - wrong bridge ranking
   - answer synthesis error

## Source Artifacts

Run-level artifacts:
- [comparison.json](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/comparison.json)
- [guidance_on bridge_test_results.json](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/bridge_test_results.json)
- [guidance_off bridge_test_results.json](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/bridge_test_results.json)

Per-batch tagging artifacts:
- [guidance_on batch 0 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/batch_0/bridge_pattern_classification_report.json)
- [guidance_on batch 1 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/batch_1/bridge_pattern_classification_report.json)
- [guidance_on batch 2 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/batch_2/bridge_pattern_classification_report.json)
- [guidance_on batch 3 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/batch_3/bridge_pattern_classification_report.json)
- [guidance_off batch 0 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/batch_0/bridge_pattern_classification_report.json)
- [guidance_off batch 1 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/batch_1/bridge_pattern_classification_report.json)
- [guidance_off batch 2 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/batch_2/bridge_pattern_classification_report.json)
- [guidance_off batch 3 report](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/batch_3/bridge_pattern_classification_report.json)

Operational noise references:
- [guidance_on stderr](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_on/runner_stderr.log)
- [guidance_off stderr](../logs/experiments/2wiki_lookup_mixed_guidance_ablation/guidance_off/runner_stderr.log)
