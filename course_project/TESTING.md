# Using the starter tests

Install the course requirements and run:

```bash
pytest -q tests
```

Before you implement RICR, the package, graph, embedding, and retrieval tests
should pass. Tests that depend on the RICR TODOs are automatically skipped
and report:

```text
Complete run_panini_ricr to enable this test.
```

After implementing `identify_retrieval_components` and `run_panini_ricr`, run the same
command again. The skipped tests will enable automatically. They check:

- topological ordering of a converging retrieval DAG;
- substitution of all parent answers at a multi-parent node;
- harmonic-mean parent-beam combination;
- intermediate entity grouping versus final QA-level selection;
- evidence collected from every surviving final beam; and
- the original-question fallback for singleton plans.

The supplied tests are examples, not the complete grading suite. Add your own
tests for reconciliation (intended merges, homonym non-merges, and edge-weight
aggregation), document-namespaced local IDs, parent combinations below the
quality threshold, repeated QA IDs, malformed placeholders, and the specific
failure cases you find during evaluation.
