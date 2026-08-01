#!/usr/bin/env python3
"""Generate the self-contained instructor Colab answer-key notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Panini_Full_Answer_Key_Colab.ipynb"
STUDENT_LIKE_OUTPUT = HERE / "Panini_Student_Like_Solution.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def main() -> int:
    library = (HERE / "colab_full_solution.py").read_text(encoding="utf-8")
    audit_csv_text = (HERE / "reconciliation_audit.csv").read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": "Panini Full Instructor Answer Key"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        md(
            """
            # PANINI course project — full instructor answer key and Colab run

            This notebook answers Questions 1–12 as one continuous system
            build. It audits both 100-question packages, constructs and
            analyzes alternative memory graphs, evaluates retrieval, implements
            RICR, runs the neural system on all 200 questions, performs the
            required ablations, and writes the four submission JSONL files.

            **Instructor-only:** this notebook contains reference algorithms and
            explanatory answers. Do not copy it into the student repository.
            Expensive work is appended to Drive after every question, so `Run
            all` can be repeated after a Colab interruption without losing
            completed records.
            """
        ),
        code(
            """
            # Run controls. The defaults perform the complete assignment.
            import os

            # Keep the submitted notebook at full-run defaults.  The environment
            # overrides are useful for a fast CPU-only structural validation.
            RUN_FULL = os.environ.get('PANINI_RUN_FULL', '1') == '1'
            RUN_ABLATIONS = os.environ.get('PANINI_RUN_ABLATIONS', '1') == '1'
            MOUNT_DRIVE = True
            DATASETS = ('2wiki', 'musique')

            # For a quick validation before the full run, set this to slice(0, 2).
            # Return it to slice(None) for the required 200-question run.
            QUESTION_SLICE = slice(None)
            """
        ),
        code(
            """
            from pathlib import Path
            import json, os, subprocess, sys

            IN_COLAB = 'google.colab' in sys.modules
            if IN_COLAB:
                REPO_ROOT = Path('/content/panini-course-project')
                if not (REPO_ROOT / 'manifest.json').exists():
                    subprocess.run([
                        'git', 'clone', '--depth', '1',
                        'https://github.com/YigitTurali/panini-course-project.git',
                        str(REPO_ROOT),
                    ], check=True)
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-q', '-r',
                    str(REPO_ROOT / 'requirements-colab.txt')
                ], check=True)
                sys.path.insert(0, str(REPO_ROOT))
                if MOUNT_DRIVE:
                    from google.colab import drive
                    drive.mount('/content/drive')
                    WORK_ROOT = Path('/content/drive/MyDrive/panini-full-answer-key')
                else:
                    WORK_ROOT = Path('/content/panini-full-answer-key')
                PACKAGE_ROOTS = {
                    '2wiki': REPO_ROOT,
                    'musique': REPO_ROOT / 'packages/panini_musique_100',
                }
                AUDIT_CSV = None
            else:
                source_override = os.environ.get('PANINI_SOURCE_ROOT')
                SOURCE_ROOT = Path(source_override).resolve() if source_override else Path.cwd().resolve()
                while SOURCE_ROOT != SOURCE_ROOT.parent and not (SOURCE_ROOT / 'course_project').exists():
                    SOURCE_ROOT = SOURCE_ROOT.parent
                if not (SOURCE_ROOT / 'course_project').exists():
                    raise FileNotFoundError('Set PANINI_SOURCE_ROOT to the gsw-memory checkout.')
                sys.path.insert(0, str(SOURCE_ROOT / 'course_project/src'))
                WORK_ROOT = Path(os.environ.get(
                    'PANINI_WORK_ROOT',
                    SOURCE_ROOT / 'course_project/instructor/full_run_v4'))
                PACKAGE_ROOTS = {
                    '2wiki': SOURCE_ROOT / 'course_project/release/panini_2wiki_100',
                    'musique': SOURCE_ROOT / 'course_project/release/panini_musique_100',
                }
                AUDIT_CSV = SOURCE_ROOT / 'course_project/instructor/reconciliation_audit.csv'

            CACHE_BASE = WORK_ROOT / 'cache'
            OUTPUT_ROOT = WORK_ROOT / 'submission'
            CACHE_BASE.mkdir(parents=True, exist_ok=True)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            print({'work_root': str(WORK_ROOT), 'colab': IN_COLAB})
            """
        ),
        md(
            """
            ## Reference implementation library

            The next cell is intentionally long and collapsed in Colab. It is
            the complete reference implementation used by later answer cells:
            conservative reconciliation, plan validation, global beam pruning,
            exact-query caches, sequential model loading, metrics, and output
            materialization. Keeping it inside the notebook makes this answer
            key self-contained.
            """
        ),
        code(library),
        code(
            """
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from panini_course import CoursePackage

            CONFIG = RunConfig()
            packages = {name: CoursePackage(PACKAGE_ROOTS[name]) for name in DATASETS}
            jobs = []
            for name, package in packages.items():
                questions = (package.questions('public') + package.questions('held_out'))[QUESTION_SLICE]
                cache_root = CACHE_BASE / name
                cache_root.mkdir(parents=True, exist_ok=True)
                jobs.append((name, package, questions, cache_root))
            print(gpu_snapshot(), {'questions_in_this_run': sum(len(job[2]) for job in jobs)})
            """
        ),
        md(
            """
            ## Question 1 — understand and verify the two packages (6 points)

            The trusted boundary is an identifier audit, not a row-count audit.
            `entity_uid` and `qa_uid` carry document, GSW, and local-node
            provenance. Two arrays can have equal lengths while one is
            permuted, duplicated, or missing an ID; row counts would pass while
            retrieval silently attaches a vector to the wrong QA. The checks
            below therefore require uniqueness *and* set equality between
            metadata IDs and embedding IDs. Held-out rows are also inspected by
            field name, because a nonempty input file is correct but any answer
            or evidence field would violate the evaluation boundary. The final
            table also distinguishes named-node coverage from grounded-QA
            coverage. Three development questions have atomic answers present
            as entity nodes but not as answers of a grounded QA edge; they are
            retained and counted as operational retrieval failures rather than
            silently removed after observing labels.
            """
        ),
        code(
            """
            audit_rows, schema_examples = [], {}
            for dataset, package in packages.items():
                rows, examples = audit_package(package, dataset)
                audit_rows.extend(rows)
                schema_examples[dataset] = examples
            audit_table = pd.DataFrame(audit_rows)
            display(audit_table)
            for dataset, examples in schema_examples.items():
                print(f'\\n{dataset} question example:', json.dumps(examples['question'], indent=2)[:1600])
                print('entity example:', json.dumps(examples['entity'], indent=2)[:1200])
                print('QA example:', json.dumps(examples['qa'], indent=2)[:1200])

            qa_coverage_exceptions = []
            for dataset, package in packages.items():
                qa_rows = package.qa_pairs()
                for question in package.questions('public'):
                    if not gold_qa_ids(question, qa_rows):
                        qa_coverage_exceptions.append({
                            'dataset': dataset,
                            'question_id': question['question_id'],
                            'question': question['question'],
                            'named_node_but_no_grounded_QA_answer': True,
                        })
            display(pd.DataFrame(qa_coverage_exceptions))
            """
        ),
        md(
            """
            **Reference explanation.** A question ID identifies the evaluation
            unit; a document ID locates provenance; an entity UID identifies one
            document-local occurrence; and a QA UID identifies one grounded
            verb-question-answer edge. None is interchangeable with a Python
            list position. A plausible undetected failure is sorting
            `qa_pairs.jsonl` without applying the same permutation to
            `qa_embeddings.npy`: shape checks still pass, FAISS still returns
            valid row numbers, and every displayed answer is nevertheless
            attached to the wrong vector. The set-equality and ID-to-row maps
            above prevent that failure.
            """
        ),
        md(
            """
            ## Questions 2–3 — construct, reconcile, and analyze the network (24 points)

            The native multigraph records what each GSW actually asserted. The
            unreconciled projection keeps document-local occurrence IDs. The
            exact-surface projection is an intentionally aggressive sensitivity
            condition. The conservative mapping accepts the same normalized
            surface and node type only when it also has identity-bearing role or
            neighborhood evidence; nationality, profession, date, number,
            genre, and other attribute values are explicitly blocked. These
            projections are analysis objects and are never passed to retrieval.
            """
        ),
        code(
            """
            from panini_course.graph import (
                build_entity_projection, build_native_gsw_graph,
                build_unreconciled_entity_projection,
            )

            graph_sets, network_rows, decision_sets = {}, [], {}
            for dataset, package in packages.items():
                native = build_native_gsw_graph(package.gsw_paths())
                unreconciled = build_unreconciled_entity_projection(native)
                exact = build_entity_projection(native)
                mapping, decisions = conservative_entity_mapping(native)
                conservative = aggregate_projection(unreconciled, mapping)
                graph_sets[dataset] = {
                    'native': native, 'unreconciled': unreconciled,
                    'exact_surface': exact, 'conservative': conservative,
                }
                decision_sets[dataset] = decisions
                network_rows.extend(
                    {'dataset': dataset, **network_statistics(name, graph)}
                    for name, graph in graph_sets[dataset].items()
                )
                # Edge direction, provenance, and cross-document non-merge checks.
                assert all(data['edge_type'] == 'qa' for *_, data in native.edges(data=True))
                assert all('document_id' in data for *_, data in native.edges(data=True))
                assert all('::' in node for node in unreconciled.nodes)
            network_table = pd.DataFrame(network_rows)
            display(network_table)
            """
        ),
        code(
            """
            # Fixed, independently labeled 30-pair audit used for the answer key.
            from io import StringIO

            manual_audit = pd.read_csv(StringIO(__AUDIT_CSV_TEXT__))
            decision_lookup = {
                (dataset, row['left_uid'], row['right_uid']): row['accepted']
                for dataset, rows in decision_sets.items() for row in rows
            }
            manual_audit['conservative_decision'] = [
                decision_lookup.get((row.dataset, row.left_uid, row.right_uid), False)
                for row in manual_audit.itertuples()
            ]
            certain = manual_audit[manual_audit.reference_label != 'uncertain']
            precision = np.mean([
                label == 'correct' for label in certain[certain.conservative_decision].reference_label
            ]) if certain.conservative_decision.any() else np.nan
            recall = np.mean(
                certain[certain.reference_label == 'correct'].conservative_decision
            )
            display(manual_audit)
            print({'audited_pairs': len(manual_audit),
                   'uncertain': int((manual_audit.reference_label == 'uncertain').sum()),
                   'estimated_precision_excluding_uncertain': precision,
                   'reference_recall_on_audited_correct_pairs': recall})

            missed_aliases = pd.DataFrame([
                {'dataset': 'musique', 'left': 'USA', 'right': 'United States',
                 'reason': 'surface normalization does not expand abbreviations'},
                {'dataset': 'musique', 'left': 'UK', 'right': 'United Kingdom',
                 'reason': 'surface normalization does not expand abbreviations'},
            ])
            display(missed_aliases)
            """.replace("__AUDIT_CSV_TEXT__", repr(audit_csv_text))
        ),
        code(
            """
            # Degree PMF and CCDF: two panels per dataset, four plots total.
            fig, axes = plt.subplots(len(graph_sets), 2, figsize=(13, 5 * len(graph_sets)))
            for row_index, (dataset, variants) in enumerate(graph_sets.items()):
                for name, graph in variants.items():
                    simple = nx.Graph(graph)
                    degrees = np.array([degree for _, degree in simple.degree()])
                    positive = degrees[degrees > 0]
                    values, counts = np.unique(positive, return_counts=True)
                    axes[row_index, 0].loglog(values, counts / counts.sum(), marker='.', label=name)
                    ordered = np.sort(positive)
                    ccdf = 1 - np.arange(len(ordered)) / len(ordered)
                    axes[row_index, 1].loglog(ordered, ccdf, marker='.', label=name)
                axes[row_index, 0].set(title=f'{dataset}: degree PMF', xlabel='degree', ylabel='P(k)')
                axes[row_index, 1].set(title=f'{dataset}: degree CCDF', xlabel='degree', ylabel='P(K ≥ k)')
                axes[row_index, 0].legend(); axes[row_index, 1].legend()
            plt.tight_layout(); plt.show()

            centrality_rows = []
            for dataset, variants in graph_sets.items():
                for variant in ('exact_surface', 'conservative'):
                    scores = top_centralities(variants[variant])
                    for metric, ranked in scores.items():
                        for rank, (node, score) in enumerate(ranked, start=1):
                            centrality_rows.append({'dataset': dataset, 'graph': variant,
                                'metric': metric, 'rank': rank, 'node': node, 'score': score})
            display(pd.DataFrame(centrality_rows))
            """
        ),
        code(
            """
            # Visualize one complete gold path per dataset using the supplied evidence.
            fig, axes = plt.subplots(1, len(packages), figsize=(15, 5))
            for axis, (dataset, package) in zip(np.atleast_1d(axes), packages.items()):
                question = package.questions('public')[0]
                path = nx.DiGraph()
                previous = 'USER QUESTION'
                path.add_node(previous, kind='question')
                for index, task in enumerate(atomic_tasks(question), start=1):
                    query_node = f'Q{index}: {task["question"]}'
                    answer_node = f'A{index}: {task["answer"]}'
                    path.add_edge(previous, query_node, label='requires')
                    path.add_edge(query_node, answer_node, label='GSW QA')
                    previous = answer_node
                positions = nx.spring_layout(path, seed=CONFIG.seed)
                nx.draw_networkx(path, positions, ax=axis, node_size=800, font_size=7, arrows=True)
                nx.draw_networkx_edge_labels(path, positions,
                    edge_labels=nx.get_edge_attributes(path, 'label'), ax=axis, font_size=7)
                axis.set_title(f'{dataset}: {question["question_id"]}')
                axis.axis('off')
            plt.tight_layout(); plt.show()
            """
        ),
        md(
            """
            **Reference interpretation.** The unreconciled giant component is
            tiny because local GSWs are deliberately document-scoped. Exact
            surface merging produces a much larger giant component, but that is
            not evidence that write-time memory was globally connected: it is
            evidence that the identity rule inserted cross-document bridges.
            The audit shows why `American`, `actor`, and repeated dates are
            dangerous hubs, while people such as Charles Mingus are defensible
            merges. The conservative graph still changes when its rule changes,
            and aliases such as USA/United States remain split. Therefore no
            reconciled projection is used operationally. Entity retrieval
            returns a document-local occurrence and expands only within its
            originating GSW; RICR creates cross-document chains at read time.
            Degree or PageRank alone cannot distinguish a real semantic hub from
            a merge artifact—the role type, documents, neighborhood, and manual
            audit provide that evidence. The log–log plots may look heavy-tailed,
            but no power-law claim is made without a fitted comparison test.
            """
        ),
        md(
            """
            ## Question 4 — decomposition and dependency graphs (14 points)

            Raw model text is appended before parsing. Validation rejects empty
            plans, future or missing references, and malformed nodes. A
            placeholder creates a dependency edge. Retrieval nodes form weakly
            connected components that are processed in topological order. A
            component may converge: a retrieval node referencing Q1 and Q2 must
            receive both parent bindings before it issues its concrete query.
            Deterministic comparison and intersection nodes remain distinct
            because they combine retrieved values rather than search memory.
            """
        ),
        code(
            """
            if RUN_FULL:
                run_decomposition_stage(jobs, CONFIG)
            else:
                print('Full decomposition skipped by RUN_FULL=False.')

            decomposition_tables = []
            decomposition_error_rows = []
            for dataset, package, _, cache_root in jobs:
                records = read_jsonl(cache_root / 'decompositions.jsonl')
                predicted = {row['question_id']: row['predicted_decomposition']
                             for row in records if row.get('predicted_decomposition')}
                if predicted:
                    decomposition_tables.append({'dataset': dataset,
                        **decomposition_metrics(predicted, package.decompositions())})
                    for qid, reviewed_plan in package.decompositions().items():
                        predicted_plan = predicted.get(qid, [])
                        reviewed_edges = set(map(tuple, validate_plan(reviewed_plan)['edges']))
                        predicted_edges = set(map(tuple, validate_plan(predicted_plan)['edges']))
                        if len(predicted_plan) != len(reviewed_plan):
                            category = ('missing hop' if len(predicted_plan) < len(reviewed_plan)
                                        else 'extra hop')
                        elif predicted_edges != reviewed_edges:
                            category = 'wrong dependency'
                        else:
                            continue
                        decomposition_error_rows.append({
                            'dataset': dataset, 'question_id': qid, 'category': category,
                            'predicted': json.dumps(predicted_plan, ensure_ascii=False),
                            'reviewed': json.dumps(reviewed_plan, ensure_ascii=False),
                        })
            display(pd.DataFrame(decomposition_tables))
            error_frame = pd.DataFrame(decomposition_error_rows)
            if not error_frame.empty:
                display(error_frame.groupby('dataset', group_keys=False).head(2))
            """
        ),
        md(
            """
            **Reference error analysis.** Invalid JSON or an absent list is a
            malformed-output error. A placeholder pointing to the wrong earlier
            answer is a dependency error. Too few retrieval nodes is a missing
            hop; an unnecessary lookup is an extra hop. A valid graph can still
            ask the wrong atomic question, which is a semantic error rather than
            a parser error. For a two-branch comparison, film→director and
            director→death-date execute independently; “which is later?” then
            compares the two dates, and the final operation maps the winning
            director back to the film. It is reasoning because the corpus need
            not contain that comparison as a stored QA pair.
            """
        ),
        md(
            """
            ## Question 5 — sparse retrieval baselines (12 points)

            TF–IDF, BM25-QA, and BM25-entity expansion use the same normalized
            evidence tasks and stable QA IDs. Entity expansion follows only the
            local verb neighborhood attached to the retrieved occurrence.
            """
        ),
        code(
            """
            from panini_course import BM25Index, DualRetriever, TfidfIndex

            def relevant_for_task(task, question, qa_rows):
                documents = ({task['document_id']} if task['document_id'] else
                             set(question['context_document_ids']))
                answer = normalize_text(task['answer'])
                candidates = [row for row in qa_rows
                              if row['document_id'] in documents and
                              answer in {normalize_text(value) for value in row['answer_names']}]
                if not candidates:
                    return set()
                query_tokens = set(WORD.findall(task['question'].casefold())) - STOPWORDS
                def overlap(row):
                    text = f"{row.get('question','')} {row.get('verb_phrase','')}"
                    tokens = set(WORD.findall(text.casefold())) - STOPWORDS
                    return len(query_tokens & tokens) / max(len(query_tokens | tokens), 1)
                best = sorted(candidates, key=lambda row: (-overlap(row), row['qa_uid']))[0]
                return {best['qa_uid']}

            sparse_rows, sparse_disagreements = [], []
            for dataset, package in packages.items():
                root, qa_rows = package.root, package.qa_pairs()
                qa_by_id = {row['qa_uid']: row for row in qa_rows}
                entity = BM25Index.load(root/'indices/entity_bm25.joblib', root/'indices/entity_ids.json')
                bm25 = BM25Index.load(root/'indices/qa_bm25.joblib', root/'indices/qa_ids.json')
                tfidf = TfidfIndex.load(root/'indices/qa_tfidf.npz',
                    root/'indices/qa_tfidf_vectorizer.joblib', root/'indices/qa_ids.json')
                expansion = DualRetriever(entity_index=entity, qa_index=bm25,
                    entity_rows=package.entities(), qa_rows=qa_rows)
                for question in package.questions('public'):
                    group = question['type'] if dataset == '2wiki' else question['hop_count']
                    for task in atomic_tasks(question):
                        relevant = relevant_for_task(task, question, qa_rows)
                        tfidf_hits = tfidf.search(task['question'], 15)
                        bm25_hits = bm25.search(task['question'], 15)
                        entity_hits = entity.search(task['question'], 20)
                        expanded = []
                        for hit in entity_hits:
                            expanded.extend(expansion.qa_ids_by_entity.get(hit.item_id, ()))
                        rankings = {
                            'tfidf': [hit.item_id for hit in tfidf_hits],
                            'bm25_qa': [hit.item_id for hit in bm25_hits],
                            'bm25_entity_expansion': list(dict.fromkeys(expanded))[:15],
                        }
                        if (rankings['tfidf'][:1] != rankings['bm25_qa'][:1]
                                and sum(row['dataset'] == dataset for row in sparse_disagreements) < 2):
                            query_tokens = set(WORD.findall(task['question'].casefold()))
                            for method, hits in [('tfidf', tfidf_hits[:5]), ('bm25_qa', bm25_hits[:5])]:
                                for hit in hits:
                                    qa = qa_by_id[hit.item_id]
                                    matched = sorted(query_tokens & set(WORD.findall(qa['search_text'].casefold())))
                                    sparse_disagreements.append({
                                        'dataset': dataset, 'query': task['question'],
                                        'method': method, 'rank': hit.rank, 'score': hit.score,
                                        'stored_question': qa['question'],
                                        'answer': '; '.join(qa['answer_names']),
                                        'matched_terms': ', '.join(matched),
                                    })
                        for method, ranked in rankings.items():
                            for k in (1, 5, 10, 15):
                                sparse_rows.append({'dataset': dataset, 'group': group,
                                    'method': method, 'k': k,
                                    **evaluate_ranked_ids(ranked, relevant, k)})
            sparse_results = pd.DataFrame(sparse_rows)
            display(sparse_results.groupby(['dataset','group','method','k'], as_index=False)
                    [['recall','reciprocal_rank']].mean())
            display(pd.DataFrame(sparse_disagreements))
            """
        ),
        md(
            """
            **Reference interpretation.** TF–IDF strongly rewards rare exact
            terms but does not saturate repeated term frequency. BM25 adds term
            saturation and document-length normalization, which can move a short
            focused QA above a long record containing the same words. Entity
            expansion solves a different problem: a named surface can retrieve
            a local node even when the attached QA paraphrases the query. It can
            also add irrelevant sibling questions, so it is a recall route, not
            a final ranker. Disagreements should be explained using the displayed
            terms, lengths, document frequencies, and source entity—not by
            saying one method is simply “more semantic.”
            """
        ),
        md(
            """
            ## Questions 6–7 — dense, RRF, dual retrieval, and reranking (24 points)

            All query vectors reached by the required deterministic runs are
            supplied, so Stage B loads no embedding model. It loads the 4-bit
            Qwen3-Reranker-8B alone with batch size 1 and 256-token inputs; if a
            particular T4 raises an OOM, it records and uses the official 4B
            fallback. It warms the exact gold atomic-query rankings and then
            runs every predicted plan.
            Each query cache stores BM25, dense, RRF, reranker-only,
            retrieval-only, and 0.5/0.5 hybrid rankings over the same dual pool.
            The fixed candidate pool makes the Question 7 comparison controlled.
            """
        ),
        code(
            """
            if RUN_FULL:
                run_neural_retrieval_stage_low_memory(
                    jobs, CONFIG, run_ablations=RUN_ABLATIONS)
            else:
                print('Full neural retrieval skipped by RUN_FULL=False.')

            reranking_rows, disagreement_rows = [], []
            for dataset, package, _, cache_root in jobs:
                cache = {row['query_key']: row for row in read_jsonl(cache_root/'retrieval_cache.jsonl')}
                qa_rows = package.qa_pairs()
                for question in package.questions('public'):
                    for task in atomic_tasks(question):
                        record = cache.get(normalize_text(task['question']))
                        if not record:
                            continue
                        relevant = relevant_for_task(task, question, qa_rows)
                        ranks = {}
                        for method in ('bm25','dense','rrf','retrieval_only','reranker_only','dual_hybrid'):
                            ids = [row['qa_uid'] for row in record['rankings'][method]]
                            metric = evaluate_ranked_ids(ids, relevant, 15)
                            ranks[method] = metric['reciprocal_rank']
                            reranking_rows.append({'dataset': dataset, 'method': method, **metric})
                        disagreement_rows.append({'dataset': dataset, 'query': task['question'], **ranks})
            reranking_table = pd.DataFrame(reranking_rows)
            if not reranking_table.empty:
                display(reranking_table.groupby(['dataset','method'], as_index=False).mean(numeric_only=True))
                disagreements = pd.DataFrame(disagreement_rows)
                display(disagreements.sort_values('reranker_only', ascending=False).head(5))
                display(disagreements.sort_values('reranker_only', ascending=True).head(5))
                disagreements['rerank_delta'] = (
                    disagreements.reranker_only - disagreements.retrieval_only
                )
                trace_queries = [
                    disagreements.sort_values('rerank_delta', ascending=False).iloc[0],
                    disagreements.sort_values('rerank_delta', ascending=True).iloc[0],
                ]
                annotated = []
                for selected in trace_queries:
                    cache_root = CACHE_BASE / selected.dataset
                    record = {row['query_key']: row for row in
                              read_jsonl(cache_root/'retrieval_cache.jsonl')}[
                                  normalize_text(selected['query'])]
                    for method in ('retrieval_only', 'reranker_only'):
                        for rank, candidate in enumerate(record['rankings'][method][:5], start=1):
                            annotated.append({'dataset': selected.dataset,
                                'query': selected['query'], 'delta': selected.rerank_delta,
                                'method': method, 'rank': rank,
                                'qa_uid': candidate['qa_uid'], 'answer': candidate['answer'],
                                'score': candidate['score'],
                                'reranker_score': candidate.get('reranker_score'),
                                'retrieval_rank': candidate.get('retrieval_rank'),
                                'routes': candidate.get('routes')})
                display(pd.DataFrame(annotated))

                helped = disagreements[
                    disagreements.dual_hybrid > disagreements[['dense','retrieval_only']].max(axis=1)
                ]
                if not helped.empty:
                    selected = helped.iloc[0]
                    record = {row['query_key']: row for row in
                              read_jsonl((CACHE_BASE/selected.dataset)/'retrieval_cache.jsonl')}[
                                  normalize_text(selected['query'])]
                    display(pd.DataFrame(record['rankings']['dual_hybrid'][:15])[
                        ['qa_uid','answer','score','routes']])
            """
        ),
        code(
            """
            # Manual FAISS inner-product consistency for five supplied fixed queries.
            from panini_course import DenseIndex, QueryEmbeddingStore
            consistency = []
            for dataset, package in packages.items():
                root = package.root
                dense = DenseIndex.load(root/'indices/qa_qwen3_8b_ip.faiss', root/'indices/qa_ids.json')
                store = QueryEmbeddingStore.load(root/'embeddings/query_embeddings.npy',
                    root/'embeddings/query_ids.json', root/'embeddings/queries.jsonl')
                matrix = np.load(root/'embeddings/qa_embeddings.npy', mmap_mode='r')
                qa_ids = json.loads((root/'embeddings/qa_ids.json').read_text())
                query_rows = read_jsonl(root/'embeddings/queries.jsonl')[:5]
                for query_row in query_rows:
                    query_text = query_row['text']
                    vector = store.get(query_text)
                    manual = qa_ids[int(np.argmax(matrix @ vector))]
                    faiss_top = dense.search(vector, 1)[0].item_id
                    consistency.append({'dataset': dataset, 'query': query_text,
                                        'manual_top': manual, 'faiss_top': faiss_top,
                                        'match': manual == faiss_top})
            display(pd.DataFrame(consistency))
            assert all(row['match'] for row in consistency)
            """
        ),
        md(
            """
            **Reference interpretation.** Dense QA retrieval helps when the
            atomic query paraphrases stored text, while entity expansion helps
            when a named node is a stronger doorway than the QA wording. Their
            union therefore changes the candidate set, not merely the score.
            RRF makes lexical and dense ranks comparable without pretending
            their raw scores share a scale. Reranking can improve a relevant
            candidate by reading query and QA jointly, but it can also demote a
            terse correct relation in favor of a fluent topical distractor. The
            controlled table answers “does it help?” empirically: compare first
            relevant ranks within the identical candidate pool and inspect the
            reranker probability alongside the retrieval prior.
            """
        ),
        md(
            """
            ## Question 8 — complete RICR implementation (22 points)

            The reference solution executes connected retrieval DAGs, not
            independent linear branches. It topologically processes a component;
            a multi-parent node ranks Cartesian products by harmonic mean before
            substituting all parents. Intermediate hops keep the best state per
            namespaced answer entity, whereas the final hop keeps QA records
            directly. Evidence is the deduplicated union from every surviving
            final beam. These are the exact research-code semantics at a smaller
            corpus/model scale.
            """
        ),
        code(
            """
            toy_plan = [
                {'question': 'Who founded lab A?', 'requires_retrieval': True},
                {'question': 'Who founded lab B?', 'requires_retrieval': True},
                {'question': 'Who published first, <ENTITY_Q1> or <ENTITY_Q2>?',
                 'requires_retrieval': True},
            ]
            toy = {
                'Who founded lab A?': [
                    {'qa_uid':'q1','answer_names':['Ada'],'answer_ids':['d1::e1'],
                     'question':'founder A','document_id':'d1','score':0.90},
                    {'qa_uid':'q2','answer_names':['Alan'],'answer_ids':['d2::e1'],
                     'question':'founder A','document_id':'d2','score':0.60}],
                'Who founded lab B?': [
                    {'qa_uid':'q3','answer_names':['Grace'],'answer_ids':['d3::e1'],
                     'question':'founder B','document_id':'d3','score':0.80},
                    {'qa_uid':'q4','answer_names':['Katherine'],'answer_ids':['d4::e1'],
                     'question':'founder B','document_id':'d4','score':0.70}],
                'Who published first, Ada or Grace?': [
                    {'qa_uid':'q5','answer_names':['Ada'],'answer_ids':['d1::e1'],
                     'question':'comparison','document_id':'d5','score':0.95}],
                'Who published first, Ada or Katherine?': [
                    {'qa_uid':'q6','answer_names':['Ada'],'answer_ids':['d1::e1'],
                     'question':'comparison','document_id':'d6','score':0.75}],
            }
            toy_result = execute_panini_plan(
                toy_plan, lambda query, k: toy[query][:k],
                replace(CONFIG, beam_width=2, candidates_per_hop=2),
                original_question='Who published first?')
            display(pd.DataFrame(toy_result['component_traces'][0]['steps'][-1]['kept']))
            print(json.dumps(toy_result['component_traces'], indent=2))
            print('all-final-beam evidence:', [row['qa_uid'] for row in toy_result['evidence']])
            """
        ),
        md(
            """
            **Hand calculation.** The normalized root scores are Ada `.95`,
            Alan `.80`, Grace `.90`, and Katherine `.85`. The four parent
            products have harmonic scores `.924`, `.897`, `.847`, and `.824` in
            that order, so `B=2` sends only `Ada + Grace` and `Ada + Katherine`
            to the converging retrieval node. Both final QA records answer Ada,
            but final-hop selection does not entity-deduplicate them. Their
            parent evidence also remains in the context because evidence is
            collected from both final beams, not just the best one.
            """
        ),
        md(
            """
            ## Question 9 — controlled RICR ablations (14 points)

            The fixed-seed subset contains 20 development questions per
            dataset. Before seeing results, the directional predictions are:
            narrower beams and `k=5` reduce latency but lower chain recovery;
            disabling intermediate entity grouping lowers substitution diversity;
            removing the multi-parent threshold increases joint queries; last-hop
            scoring is less stable because it forgets early weak links; BM25
            loses paraphrases; dense-only loses entity-doorway candidates; RRF
            is competitive but lacks the cross-encoder's joint judgment.
            """
        ),
        code(
            """
            prediction_table = pd.DataFrame([
                {'configuration':'beam_1', 'prediction':'lower chain recovery and latency'},
                {'configuration':'beam_3', 'prediction':'between beam 1 and beam 5'},
                {'configuration':'k_5', 'prediction':'lower recall and reranking cost'},
                {'configuration':'unique_off', 'prediction':'fewer distinct substitutions'},
                {'configuration':'parent_threshold_off', 'prediction':'more joint queries and latency'},
                {'configuration':'last_hop', 'prediction':'unstable; ignores weak early evidence'},
                {'configuration':'bm25', 'prediction':'loses paraphrased evidence'},
                {'configuration':'dense', 'prediction':'loses entity-doorway candidates'},
                {'configuration':'rrf', 'prediction':'better coverage than one route but no cross-encoder'},
            ])
            display(prediction_table)

            ablation_frames = []
            for dataset, _, _, cache_root in jobs:
                rows = read_jsonl(cache_root/'ablation_answers.jsonl')
                if rows:
                    frame = pd.DataFrame(rows)
                    summary = frame.groupby('configuration', as_index=False).agg(
                        questions=('question_id','count'),
                        supporting_qa_recall=('supporting_qa_recall','mean'),
                        complete_chain_recovery=('complete_chain_recovery','mean'),
                        EM=('exact_match','mean'), F1=('token_f1','mean'),
                        latency=('retrieval_seconds','mean'),
                        evidence_count=('evidence_count','mean'))
                    summary.insert(0, 'dataset', dataset)
                    ablation_frames.append(summary)
            if ablation_frames:
                ablation_table = pd.concat(ablation_frames, ignore_index=True)
                display(ablation_table)
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for dataset, rows in ablation_table.groupby('dataset'):
                    axes[0].scatter(rows.latency, rows.F1, label=dataset)
                    axes[1].scatter(rows.evidence_count, rows.F1, label=dataset)
                axes[0].set(xlabel='mean retrieval seconds', ylabel='F1', title='Accuracy vs latency')
                axes[1].set(xlabel='mean evidence count', ylabel='F1', title='Accuracy vs evidence')
                axes[0].legend(); axes[1].legend(); plt.tight_layout(); plt.show()
            else:
                print('Ablation answers appear after stages B and C finish.')
            """
        ),
        md(
            """
            **Reference recommendation.** Deploy the default only if its
            measured F1/latency point dominates the cheaper alternatives. Beam
            width buys protection against an early retrieval error, but returns
            diminish once the correct answer is already retained. Candidate
            count has a similar cost because every added item reaches the
            reranker. Unique-answer pruning is retained because duplicated
            surface forms consume capacity without creating a new substitution.
            Geometric-mean scoring is retained because a chain should not be
            rescued by one strong final hop after weak earlier evidence. The
            retrieval choice is made from the measured table rather than model
            size: dual retrieval is justified only when its additional
            supporting-QA recall survives reranking and improves complete-chain
            recovery enough to offset latency.
            """
        ),
        md(
            """
            ## Questions 10–11 — frozen 2Wiki run and MuSiQue transfer (22 points)

            Stage C unloads retrieval models, loads only the 4-bit Qwen3-4B
            answerer, and supplies deduplicated evidence from all surviving
            RICR chains, including answer role/state strings. It uses the same
            four-message one-shot `Thought:`/`Answer:` prompt as the research
            evaluator and does not add an N/A instruction for these answerable
            splits. It never receives source documents, graph neighbors, or
            labels. The configuration is identical for both datasets. Outputs
            are split into 80-row development and 20-row held-out files.
            """
        ),
        code(
            """
            if RUN_FULL:
                run_answer_stage(jobs, OUTPUT_ROOT, CONFIG, run_ablations=RUN_ABLATIONS)
            else:
                print('Full answer generation skipped by RUN_FULL=False.')

            final_tables = []
            for dataset, package in packages.items():
                records = read_jsonl(OUTPUT_ROOT/'results'/f'{dataset}_dev.jsonl')
                if not records:
                    continue
                question_by_id = {row['question_id']: row for row in package.questions('public')}
                group_field = 'type' if dataset == '2wiki' else 'hop_count'
                enriched = [{**row, group_field: question_by_id[row['question_id']][group_field]}
                            for row in records]
                summary = pd.DataFrame(result_summary(enriched, group_field))
                summary.insert(0, 'dataset', dataset)
                final_tables.append(summary)
                successful = next((row for row in enriched
                                   if row.get('complete_chain_recovery') == 1
                                   and row.get('exact_match') == 1), None)
                failed = next((row for row in enriched if row.get('exact_match') == 0), None)
                trace_by_id = {row['question_id']: row for row in
                               read_jsonl((CACHE_BASE/dataset)/'traces.jsonl')}
                for label, selected in [('successful', successful), ('failed', failed)]:
                    if selected is None:
                        continue
                    if not selected.get('decomposition_valid'):
                        first_error = 'decomposition'
                    elif selected.get('supporting_qa_recall', 0) == 0:
                        first_error = 'first-hop/candidate retrieval'
                    elif selected.get('complete_chain_recovery', 0) == 0:
                        first_error = 'later-hop substitution or global pruning'
                    elif selected.get('exact_match', 0) == 0:
                        first_error = 'answer generation after complete evidence'
                    else:
                        first_error = 'none'
                    print(dataset, label, 'trace:', selected['question_id'],
                          'first irreversible error:', first_error)
                    print(json.dumps(trace_by_id.get(selected['question_id'], {}),
                                     ensure_ascii=False, indent=2)[:8000])
            if final_tables:
                final_table = pd.concat(final_tables, ignore_index=True)
                display(final_table)
                if 'hop_count' in final_table:
                    musique = final_table[final_table.dataset == 'musique']
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].plot(musique.hop_count, musique.complete_chain_recovery, marker='o')
                    axes[1].plot(musique.hop_count, musique.F1, marker='o')
                    axes[0].set(xlabel='hop count', ylabel='complete-chain recovery')
                    axes[1].set(xlabel='hop count', ylabel='F1')
                    plt.tight_layout(); plt.show()
            """
        ),
        md(
            """
            **Reference transfer interpretation.** Complete-chain recovery is
            conjunctive: every required hop must survive, so one additional hop
            adds another opportunity for irreversible failure. Answer F1 is not
            strictly conjunctive. The answer model may recover a short answer
            from partial or redundant evidence, or receive the final fact even
            when an intermediate gold QA was missed. Therefore chain recovery
            can fall faster than F1. Hop count is not the only dataset change;
            wording, entity distribution, and GSW coverage also differ. The
            saved traces attribute transfer loss by counting invalid plans,
            missing first-hop gold answers, successful first hops followed by
            missing later answers, and correct evidence followed by wrong
            generation. That separation is stronger evidence than attributing
            every MuSiQue loss to length.
            """
        ),
        md(
            """
            ## Question 12 — reproducibility and handoff (12 points)

            The last cell verifies the required file counts, records the runtime
            and model configuration, and writes exact resume instructions. A
            cache is complete only when its stable question IDs match the input
            IDs; file existence alone is not sufficient.
            """
        ),
        code(
            """
            required = {
                OUTPUT_ROOT/'results/2wiki_dev.jsonl': 80,
                OUTPUT_ROOT/'predictions/2wiki_heldout.jsonl': 20,
                OUTPUT_ROOT/'results/musique_dev.jsonl': 80,
                OUTPUT_ROOT/'predictions/musique_heldout.jsonl': 20,
            }
            if QUESTION_SLICE == slice(None) and RUN_FULL:
                for path, expected in required.items():
                    actual = len(read_jsonl(path))
                    assert actual == expected, f'{path}: expected {expected}, found {actual}'
            write_environment(OUTPUT_ROOT/'environment.txt', CONFIG)
            runme = chr(10).join([
                '# PANINI full run', '',
                '1. Open this notebook in a fresh Colab GPU runtime.',
                '2. Leave `QUESTION_SLICE = slice(None)` and run all cells.',
                '3. If the runtime stops, reconnect and run all again; JSONL caches skip completed IDs.',
                f'4. Final files are under `{OUTPUT_ROOT}`.',
                f'5. Configuration: `{asdict(CONFIG)}`.',
                '',
            ])
            (OUTPUT_ROOT/'RUNME.md').write_text(runme, encoding='utf-8')
            print({str(path): len(read_jsonl(path)) for path in required})
            print('environment:', OUTPUT_ROOT/'environment.txt')
            print('run guide:', OUTPUT_ROOT/'RUNME.md')
            """
        ),
        md(
            """
            **System story.** A user question first becomes a validated
            dependency graph. Each retrieval component is topologically
            executed; converging nodes combine parent beams before substituting
            all answers. BM25 entity expansion and dense QA search create a
            union candidate pool, and the reranker orders it. RICR groups
            intermediate answer entities but retains final QA alternatives.
            Evidence from all surviving final beams is deduplicated by stable QA
            ID before it becomes the answer model's only context. This design
            preserves provenance and makes
            failures inspectable. Its main protection is beam diversity: a
            plausible but wrong first hop does not immediately destroy the
            correct path. Its main failure mode is earlier than generation: if
            decomposition omits a hop, or the correct answer never enters the
            candidate pool, later reranking and answer generation cannot invent
            grounded evidence. The raw plan, per-query rankings, beam trace, and
            final answer are cached separately so that the first irreversible
            error can be located rather than guessed.
            """
        ),
    ]

    # Hide the embedded implementation by default without hiding its source.
    notebook["cells"][4]["metadata"] = {"jupyter": {"source_hidden": True}}
    nbf.write(notebook, OUTPUT)
    student_like = nbf.from_dict(notebook)
    student_like["metadata"]["colab"]["name"] = "Panini Student-Like Reference Solution"
    student_like["cells"][0]["source"] = dedent(
        """
        # PANINI course project — student-like reference solution and Colab run

        This notebook presents one complete, reproducible solution to Questions
        1–12. It audits both 100-question packages, constructs and analyzes the
        alternative memory graphs, evaluates retrieval, implements connected-DAG
        RICR, runs the neural system on all 200 questions, performs the required
        ablations, and writes the four submission JSONL files. Expensive work is
        appended to Drive after every question, so `Run all` can be repeated
        after a Colab interruption without losing completed records.
        """
    ).strip()
    student_labels = {
        "## Reference implementation library": "## Solution implementation",
        "**Reference explanation.**": "**Explanation.**",
        "**Reference interpretation.**": "**Interpretation.**",
        "**Reference error analysis.**": "**Error analysis.**",
        "**Reference recommendation.**": "**Recommendation.**",
        "**Reference transfer interpretation.**": "**Transfer interpretation.**",
    }
    for cell in student_like["cells"]:
        if cell["cell_type"] == "markdown":
            for old, new in student_labels.items():
                cell["source"] = cell["source"].replace(old, new)
        elif cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace(
                "Instructor reference pipeline used by the full Colab answer key.",
                "Complete pipeline used by this Colab solution.",
            ).replace(
                "Fixed, independently labeled 30-pair audit used for the answer key.",
                "Fixed, independently labeled 30-pair reconciliation audit.",
            )
    nbf.write(student_like, STUDENT_LIKE_OUTPUT)
    print(OUTPUT)
    print(STUDENT_LIKE_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
