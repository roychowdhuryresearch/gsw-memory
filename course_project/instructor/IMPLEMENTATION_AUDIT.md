# PANINI implementation and score audit

## Finding

The first course-scale run was not a faithful execution of the PANINI code in
`playground/multi_hop_qa_chains.py` and did not use the answer protocol in
`playground/evaluate_panini.py`. Its low score was therefore not an honest
measurement of the paper's algorithm.

The corrected student-scale implementation now:

- forms connected retrieval components and executes each component in
  deterministic topological order;
- handles a retrieval node with multiple parents by scoring the Cartesian
  product of parent beams with a harmonic mean and threshold 0.3;
- groups QA expansions by answer entity only at intermediate hops;
- retains QA-level alternatives at the final hop;
- constructs context from every surviving final beam, including answer
  role/state text;
- namespaces document-local entity IDs before diversity pruning; and
- uses the research evaluator's four-message, one-shot `Thought:`/`Answer:`
  prompt without an `N/A` instruction on answerable splits.

The local course configuration keeps the 4B decomposer, supplied Qwen3
embeddings, 4-bit Qwen3-Reranker-8B, and 4-bit Qwen3-4B answer model. The
0.5 retrieval-prior/0.5 reranker hybrid is frozen from the retrieval study
because the local reranker's probability alone is not calibrated like the
research run's Voyage reranker.

## Measured effect on the same 80-question development subsets

| Run | 2Wiki EM | 2Wiki F1 | MuSiQue EM | MuSiQue F1 |
|---|---:|---:|---:|---:|
| Previous simplified course run | 0.0625 | 0.1287 | 0.0875 | 0.0979 |
| Corrected student-scale run | 0.4625 | 0.5112 | 0.2375 | 0.2903 |
| Research PANINI logs on the same questions | 0.8000 | 0.8314 | 0.5125 | 0.5784 |

Final-answer accuracy and complete-chain recovery are not interchangeable.
On 2Wiki, corrected complete-chain recovery changes from 0.300 to 0.275 (24 to
22 questions), while EM rises by 0.400. Nineteen corrected 2Wiki predictions
are exact despite missing at least one annotated supporting QA. On MuSiQue,
both measures rise: chain recovery changes from 0.1875 to 0.225 and EM from
0.0875 to 0.2375. The old recovery count is also not directly comparable: the
linear executor could count disconnected branches as a complete path, whereas
the corrected metric is computed after a valid joint DAG execution.

The research comparison is computed from
`logs/gsw_rr_5_beam_corrected_qa_full_scale_2wikimultihopqa_qwen_q_answering_4B.json`
and
`logs/gsw_rr_5_beam_corrected_qa_full_scale_musique_qwen_q_answering_4B.json`.
It is included to separate implementation fidelity from the deliberate
course-scale model substitution; it is not a student target or a result of the
public package.

## Network reconciliation audit

The network analysis was run separately on both complete 100-question
packages. The giant-component fractions show how strongly the apparent global
network depends on the reconciliation rule:

| Dataset | Unreconciled | Exact surface | Conservative |
|---|---:|---:|---:|
| 2Wiki | 0.0056 | 0.5679 | 0.1035 |
| MuSiQue | 0.0041 | 0.4902 | 0.0760 |

On the fixed 30-pair manual audit, excluding one uncertain pair, the
conservative rule has estimated precision 1.00 and recall 0.55. Exact-surface
merging creates false bridges through repeated dates, nationalities,
professions, and other attributes. The conservative rule avoids the audited
false merges but misses aliases such as `USA`/`United States`. Thus
reconciliation is a useful network-analysis assignment, but neither graph is
safe as an operational Panini index. The retrieval pipeline keeps
document-local IDs and creates cross-document chains dynamically at read time.

## Remaining gap

The corrected executor removes the large artificial failure, but the local
reranker still loses supporting QA records that the research reranker keeps.
On the corrected development traces, supporting-QA recall is 0.572 for 2Wiki
and 0.430 for MuSiQue. MuSiQue complete-chain recovery is 0.400 for two-hop,
0.083 for three-hop, and 0.000 for four-hop questions. This compounding loss,
especially on compositional/inference and four-hop cases, explains most of the
remaining difference from the research logs.
