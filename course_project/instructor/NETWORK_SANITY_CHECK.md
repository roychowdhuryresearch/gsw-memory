# Instructor network sanity check

Computed on July 27, 2026 from the finalized 100-question packages. The native
statistics use the simple undirected view of the directed multigraph. The
entity projections connect document-local answer nodes that occur under the
same verb-phrase node.

| Dataset and graph | Nodes | Edges | Components | Giant | Giant fraction | Isolates | Mean degree | Transitivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki native | 11,051 | 9,225 | 1,986 | 54 | 0.0049 | 1,052 | 1.670 | 0.0000 |
| 2Wiki unreconciled entity projection | 6,805 | 6,436 | 1,984 | 38 | 0.0056 | 1,102 | 1.892 | 0.3795 |
| 2Wiki exact-surface projection | 5,438 | 6,333 | 1,012 | 3,088 | 0.5679 | 758 | 2.329 | 0.2761 |
| MuSiQue native | 13,056 | 10,605 | 2,643 | 46 | 0.0035 | 1,124 | 1.625 | 0.0000 |
| MuSiQue unreconciled entity projection | 8,260 | 8,931 | 2,638 | 34 | 0.0041 | 1,308 | 2.162 | 0.7213 |
| MuSiQue exact-surface projection | 6,736 | 8,780 | 1,425 | 3,302 | 0.4902 | 835 | 2.607 | 0.5587 |

The native node totals are 6,805 entity plus 4,246 verb-phrase nodes for
2Wiki, and 8,260 entity plus 4,796 verb-phrase nodes for MuSiQue. The directed
multigraphs contain 9,733 and 11,123 QA-answer edges, respectively.

## Reconciliation sensitivity

- Exact normalized surfaces collapse 1,367 2Wiki occurrences and 1,524
  MuSiQue occurrences.
- The largest component jumps from 38 to 3,088 entities on 2Wiki and from 34
  to 3,302 on MuSiQue. Global connectivity is therefore dominated by the
  reconciliation rule, not simply by local GSW structure.
- High-degree exact-surface hubs include `american`, `actor`,
  `film director`, `united states`, countries, and years. These are often
  attribute values or types rather than unique real-world entities.
- `One More Time` is an observed 2Wiki collision spanning album, song, film,
  title, music-group, and creative-work roles. It is a useful false-merge
  example.
- Exact normalization also misses aliases. MuSiQue contains both `U.S.` and
  `United States`, and both `New York (New York City)` and `New York City`.

## Interpretation

The reconciled graph is appropriate for a course exercise about identity
resolution and the sensitivity of network statistics. It must not be used as
the operational Panini graph. Panini reconciles within a document, retains the
originating document-local entity node in its entity index, and forms
cross-document chains dynamically through RICR. Precomputing exact-surface
cross-document links introduces false hubs and irreversible topology errors;
it can also miss aliases that retrieval can match semantically.
