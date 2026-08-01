# Free-tier Colab feasibility

Validated July 31, 2026.

## Conclusion

The project is memory-feasible on a single 15–16 GiB GPU when the query encoder
and reranker are loaded in separate phases and a T4 uses the 4B reranker
fallback. The 8B reranker is retained for GPUs with at least 18 GiB. The
complete 200-question evaluation is restartable after every query and
question; it should not be advertised as guaranteed to finish in one free
Colab session.

Google states that free Colab resources, GPU types, usage limits, and maximum
runtime availability vary. Free notebooks can run for at most 12 hours
depending on availability and usage patterns:
<https://research.google.com/colaboratory/faq.html>.

## Measured single-GPU memory

The measurements below used one visible NVIDIA RTX A6000. Speed is not
presented as a T4 estimate.

| Stage | Configuration | Observation | Example inference |
|---|---|---:|---:|
| Decomposition | GSW-QA-Decomposer-Qwen3-4B, NF4, FP16 compute | 2.97 GiB peak PyTorch allocation | 4.83 s/question |
| Rejected simultaneous path | Qwen3-Embedding-8B + Qwen3-Reranker-8B, both NF4 | 10.99 GiB PyTorch allocation but 20.15 GiB driver-reported process memory | 0.48 s/query embedding; 0.43 s/8 reranked pairs |
| Reranker-only reference | Qwen3-Reranker-8B, NF4, batch 1, 256 tokens | 15,278 MiB driver-reported process memory | unsafe margin on a 15,360 MiB T4 |
| T4 reranker fallback | Qwen3-Reranker-4B, NF4, batch 1, 256 tokens | 7,628 MiB driver-reported process memory | completed the same restartable scheduler |

The driver-level measurement is the relevant warning for a 15--16 GiB
runtime. Even alone, the 8B reranker leaves only 82 MiB on a nominal 15,360
MiB T4, which is not a responsible default. The final notebook therefore
alternates encoder-only and reranker-only rounds by RICR depth and selects the
official Qwen3-Reranker-4B checkpoint below 18 GiB. Query vectors are written
to disk before the encoder is unloaded; reranked pools are then written before
the next RICR round begins. The selected checkpoint is stored in every trace.

The existing real two-hop validation run in `E2E_REPORT.md` used the required
4B decomposer, 8B query encoder, and 8B reranker and recovered the correct
answer. That run took 11.99 seconds for decomposition and 9.91 seconds for
retrieval plus RICR on faster local hardware.

## Recommended Colab execution

1. Use the supplied corpus and fixed-query embeddings; never regenerate them.
2. Mount Drive and save raw plans, query vectors, reranked pools, traces, and
   answers immediately as they are produced.
3. Use `QUESTION_SLICE` for optional 10-question shards; cached stable IDs make
   overlapping restarts safe.
4. Run decomposition, neural retrieval/RICR, and answer generation as separate
   stages. Within retrieval, alternate encoder-only and reranker-only rounds.
5. Use NF4, reranker batch size 1, and 256-token reranker inputs. Use the 4B
   reranker on a 15 GiB T4 and the 8B reference reranker at 18 GiB or above.
6. If Colab assigns a smaller GPU or denies GPU access, run the CPU smoke path
   and resume the full shard when a suitable runtime is available.

The complete instructor notebook implementing this workflow is
`Panini_Full_Answer_Key_Colab.ipynb`. It is intentionally excluded from the
student release.
