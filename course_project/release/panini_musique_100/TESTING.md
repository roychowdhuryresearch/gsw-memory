# Using the starter tests

Install the course requirements and run:

```bash
pytest -q tests
```

Before you implement RICR, the package, graph, embedding, and retrieval tests
should pass. Tests that depend on the two RICR TODOs are automatically skipped
and report:

```text
Complete the two RICR TODOs to enable this test.
```

After implementing `prune_unique_answers` and `run_linear_ricr`, run the same
command again. The skipped tests will enable automatically. They check:

- answer normalization and duplicate-answer pruning;
- deterministic tie handling;
- empty retrieval results;
- answer substitution across hops;
- geometric-mean chain scores; and
- the `B=1` case.

The supplied tests are examples, not the complete grading suite. Add your own
tests for reconciliation (intended merges, homonym non-merges, and edge-weight
aggregation), parallel decomposition branches, repeated QA IDs, malformed
placeholders, and the specific failure cases you find during evaluation.
