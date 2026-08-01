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
| 2wiki | bridge_comparison | 20 | 1.0000 | 0.4625 | 0.4875 | 0.0000 | 10.0000 | 10.0000 | 0.0000 | 0.0134 | 9926.4907 | 12040.5708 | 2.0000 | 120.0000 | 32.8500 |
| 2wiki | comparison | 20 | 1.0000 | 0.8500 | 0.8250 | 0.7000 | 9.7500 | 9.6500 | 0.1500 | 0.3657 | 9386.4657 | 10520.5473 | 1.9500 | 117.0000 | 38.3000 |
| 2wiki | compositional | 20 | 1.0000 | 0.6250 | 0.7000 | 0.3500 | 5.0000 | 5.0000 | 0.1000 | 0.1000 | 20948.9693 | 21901.5987 | 1.8000 | 360.0000 | 33.3500 |
| 2wiki | inference | 20 | 1.0000 | 0.4000 | 0.3500 | 0.1500 | 5.0000 | 5.0000 | 0.0000 | 0.0357 | 10467.5505 | 21329.7207 | 1.2000 | 165.0000 | 27.1500 |
| musique | 2 | 40 | 1.0000 | 0.5000 | 0.5750 | 0.3250 | 5.0000 | 5.0000 | 0.1250 | 0.1458 | 20949.7817 | 22543.5404 | 1.5500 | 360.0000 | 29.1000 |
| musique | 3 | 24 | 1.0000 | 0.4583 | 0.5417 | 0.0833 | 5.6250 | 5.5833 | 0.0833 | 0.0833 | 34501.1377 | 51943.6311 | 2.5417 | 615.0000 | 50.0000 |
| musique | 4 | 16 | 1.0000 | 0.1875 | 0.3750 | 0.0000 | 6.2500 | 5.6875 | 0.0000 | 0.0000 | 47125.3904 | 68723.3768 | 2.7500 | 843.7500 | 50.1875 |

## Ablations

| dataset | configuration | questions | chain_recovery | EM | F1 | retrieval_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 2wiki | beam_1 | 20 | 0.1500 | 0.1000 | 0.1861 | 0.0021 |
| 2wiki | beam_3 | 20 | 0.3000 | 0.1000 | 0.1861 | 0.0022 |
| 2wiki | bm25 | 20 | 0.3500 | 0.1000 | 0.1908 | 1.5492 |
| 2wiki | default | 20 | 0.3500 | 0.1000 | 0.1861 | 0.0023 |
| 2wiki | dense | 20 | 0.4500 | 0.2000 | 0.2908 | 0.3322 |
| 2wiki | k_5 | 20 | 0.3500 | 0.1000 | 0.1861 | 0.0020 |
| 2wiki | last_hop | 20 | 0.3000 | 0.1000 | 0.1861 | 0.0021 |
| 2wiki | rrf | 20 | 0.3000 | 0.0500 | 0.1408 | 0.1669 |
| 2wiki | unique_off | 20 | 0.3500 | 0.1000 | 0.1861 | 0.0021 |
| musique | beam_1 | 20 | 0.1000 | 0.1000 | 0.1000 | 0.0036 |
| musique | beam_3 | 20 | 0.1500 | 0.1000 | 0.1000 | 0.1603 |
| musique | bm25 | 20 | 0.0500 | 0.0000 | 0.0250 | 13.9633 |
| musique | default | 20 | 0.1500 | 0.1000 | 0.1000 | 0.0041 |
| musique | dense | 20 | 0.2500 | 0.1500 | 0.1500 | 6.3843 |
| musique | k_5 | 20 | 0.1500 | 0.1000 | 0.1000 | 0.4441 |
| musique | last_hop | 20 | 0.1500 | 0.0000 | 0.0250 | 3.8164 |
| musique | rrf | 20 | 0.2000 | 0.0500 | 0.0500 | 3.4779 |
| musique | unique_off | 20 | 0.1500 | 0.1000 | 0.1000 | 0.0038 |

## Required files

| dataset | split | records |
| --- | --- | --- |
| 2wiki | development | 80 |
| 2wiki | held_out | 20 |
| musique | development | 80 |
| musique | held_out | 20 |

The runnable notebook is `Panini_Full_Answer_Key_Colab.ipynb`. The populated
notebook is `Panini_Full_Answer_Key_Executed.ipynb`.
