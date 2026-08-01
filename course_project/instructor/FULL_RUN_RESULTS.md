# PANINI full answer-key run

The run uses the frozen configuration `B=5`, `k=15`, `M=60`, 4-bit Qwen
models, supplied corpus embeddings, and exact-query caches. Neural model speed
was measured on an RTX A6000 and must not be presented as Colab T4 speed.

## Decomposition

| dataset | all_plans | questions | valid_plan_rate | subquestion_count_exact | dependency_edge_precision | dependency_edge_recall | dependency_edge_f1 | retrieval_reasoning_accuracy | mean_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 100 | 20 | 1.0000 | 0.6000 | 0.6923 | 0.5455 | 0.6102 | 0.8913 | 2.5618 |
| musique | 100 | 80 | 1.0000 | 0.7875 | 0.9070 | 0.8603 | 0.8830 | 1.0000 | 2.9472 |

## Development results

| dataset | group | questions | decomposition_valid | supporting_qa_recall | supporting_document_recall | complete_chain_recovery | surviving_chains_mean | unique_current_answers_mean | EM | F1 | latency_mean_ms | latency_p95_ms | evidence_mean | reranked_candidates_mean | answer_tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | bridge_comparison | 20 | 1.0000 | 0.6000 | 0.8375 | 0.2000 | 5.0000 | 3.3000 | 0.6000 | 0.6000 | 18198.8683 | 22047.3771 | 8.0500 | 417.0000 | 249.7500 |
| 2wiki | comparison | 20 | 1.0000 | 0.6875 | 0.8750 | 0.4500 | 5.0000 | 4.1000 | 0.8000 | 0.8714 | 15625.2456 | 18079.8367 | 6.4500 | 402.0000 | 196.4500 |
| 2wiki | compositional | 20 | 1.0000 | 0.6250 | 0.8750 | 0.3500 | 5.0000 | 4.3500 | 0.3000 | 0.3400 | 4362.3693 | 5429.6444 | 6.5500 | 357.0000 | 202.8000 |
| 2wiki | inference | 20 | 1.0000 | 0.3750 | 0.7250 | 0.1000 | 5.0000 | 4.7000 | 0.1500 | 0.2333 | 6809.8737 | 10039.8404 | 11.8000 | 159.0000 | 393.5000 |
| musique | 2 | 40 | 1.0000 | 0.5125 | 0.8000 | 0.4000 | 5.0000 | 4.5000 | 0.2750 | 0.3323 | 5543.5635 | 8954.9652 | 5.8500 | 352.5000 | 180.1750 |
| musique | 3 | 24 | 1.0000 | 0.4444 | 0.6250 | 0.0833 | 5.0000 | 3.8333 | 0.2917 | 0.3125 | 7261.0328 | 9505.9416 | 6.8333 | 560.0000 | 208.4583 |
| musique | 4 | 16 | 1.0000 | 0.2031 | 0.4688 | 0.0000 | 5.0000 | 2.6875 | 0.0625 | 0.1518 | 10973.4697 | 19908.3984 | 7.3750 | 817.5000 | 217.4375 |

## Ablations

| dataset | configuration | questions | chain_recovery | EM | F1 | retrieval_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 2wiki | beam_1 | 20 | 0.1500 | 0.3500 | 0.4083 | 0.0049 |
| 2wiki | beam_3 | 20 | 0.2500 | 0.3500 | 0.4196 | 0.0052 |
| 2wiki | bm25 | 20 | 0.2000 | 0.3500 | 0.4262 | 3.8554 |
| 2wiki | default | 20 | 0.3000 | 0.4500 | 0.5262 | 0.0056 |
| 2wiki | dense | 20 | 0.5500 | 0.6000 | 0.6429 | 0.1268 |
| 2wiki | k_5 | 20 | 0.3000 | 0.4500 | 0.4991 | 0.0049 |
| 2wiki | last_hop | 20 | 0.3500 | 0.3500 | 0.4548 | 0.0050 |
| 2wiki | parent_threshold_off | 20 | 0.3000 | 0.4500 | 0.5262 | 0.0041 |
| 2wiki | rrf | 20 | 0.3500 | 0.5000 | 0.5369 | 0.0638 |
| 2wiki | unique_off | 20 | 0.3000 | 0.4500 | 0.5262 | 0.0053 |
| musique | beam_1 | 20 | 0.1000 | 0.1500 | 0.1750 | 0.0059 |
| musique | beam_3 | 20 | 0.1500 | 0.2000 | 0.2000 | 0.0061 |
| musique | bm25 | 20 | 0.0500 | 0.0500 | 0.1077 | 0.0127 |
| musique | default | 20 | 0.1500 | 0.2000 | 0.2505 | 0.0065 |
| musique | dense | 20 | 0.3500 | 0.2500 | 0.3036 | 0.0065 |
| musique | k_5 | 20 | 0.1500 | 0.2000 | 0.2505 | 0.0063 |
| musique | last_hop | 20 | 0.1000 | 0.2000 | 0.2750 | 0.0068 |
| musique | parent_threshold_off | 20 | 0.1500 | 0.2000 | 0.2505 | 0.0068 |
| musique | rrf | 20 | 0.2500 | 0.3000 | 0.3536 | 0.0068 |
| musique | unique_off | 20 | 0.1500 | 0.2000 | 0.2505 | 0.0065 |

The ablation `retrieval_seconds` column is resume-cache accounting from this
reference run, not a cold end-to-end latency comparison. Use a fresh timed
Colab run for latency conclusions; EM, F1, and chain recovery are the intended
comparative outputs of this table.

## Required files

| dataset | split | records |
| --- | --- | --- |
| 2wiki | development | 80 |
| 2wiki | held_out | 20 |
| musique | development | 80 |
| musique | held_out | 20 |

The runnable notebook is `Panini_Full_Answer_Key_Colab.ipynb`. The populated
notebook is `Panini_Full_Answer_Key_Executed.ipynb`.
