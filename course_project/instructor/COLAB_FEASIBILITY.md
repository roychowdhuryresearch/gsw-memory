# Free-tier Colab feasibility

Validated July 31, 2026.

## Conclusion

The project is memory-feasible on a single 15–16 GiB GPU when the neural
models are loaded in 4-bit mode. The complete 200-question evaluation should
be executed as restartable 10-question shards; it should not be advertised as
guaranteed to finish in one free Colab session.

Google states that free Colab resources, GPU types, usage limits, and maximum
runtime availability vary. Free notebooks can run for at most 12 hours
depending on availability and usage patterns:
<https://research.google.com/colaboratory/faq.html>.

## Measured single-GPU memory

The measurements below used one visible NVIDIA RTX A6000. They measure CUDA
allocation for the same 4-bit code path used by the Colab notebook; speed is
not presented as a T4 estimate.

| Stage | Configuration | Peak allocated GPU memory | Example inference |
|---|---|---:|---:|
| Decomposition | GSW-QA-Decomposer-Qwen3-4B, NF4, FP16 compute | 2.97 GiB | 4.83 s/question |
| Runtime query encoding + reranking | Qwen3-Embedding-8B + Qwen3-Reranker-8B, both NF4, batch 2, max length 512 | 10.99 GiB | 0.48 s/query embedding; 0.43 s/8 reranked pairs |

The existing real two-hop validation run in `E2E_REPORT.md` used the required
4B decomposer, 8B query encoder, and 8B reranker and recovered the correct
answer. That run took 11.99 seconds for decomposition and 9.91 seconds for
retrieval plus RICR on faster local hardware.

## Recommended Colab execution

1. Use the supplied corpus and fixed-query embeddings; never regenerate them.
2. Mount Drive and save one JSONL record immediately after each question.
3. Process 10-question shards using `SHARD_START` and `SHARD_SIZE`.
4. Run decomposition, neural retrieval/RICR, and answer generation as separate
   stages. Delete the current models and clear CUDA memory between stages.
5. Use the 4B decomposer, 8B encoder, and 8B reranker in NF4; use reranker batch
   size 2 and 512-token inputs on a 15–16 GiB GPU.
6. If Colab assigns a smaller GPU or denies GPU access, run the CPU smoke path
   and resume the full shard when a suitable runtime is available.

The instructor notebook implementing this workflow is
`Panini_Student_Like_Solution.ipynb`. It is intentionally excluded from the
student release.
