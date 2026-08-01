#!/usr/bin/env python3
"""Generate the public, memory-efficient PANINI student starter notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "Panini_Course_Project.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def explanation(question: str, guidance: str, words: str = "150–200"):
    return md(
        f"""
        ### Your written response for {question}

        Replace this cell with a **{words}-word explanation in your own words**.
        {guidance}

        > **Write your response here.** Do not leave a list of numbers without
        > interpreting what they mean and what could have caused them.
        """
    )


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": "PANINI Course Project — Student Starter"},
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
            # ECE 232E course project — structured memory networks and RICR

            This is the student starter for the complete 150-point project. It
            follows the same execution order as the instructor run, but it does
            **not** contain the reconciliation rule, network conclusions,
            retrieval evaluator, reranking policy, RICR implementation,
            ablation conclusions, answers, or report prose.

            You will work with both supplied 100-question packages. Read
            `PROJECT_HANDOUT.pdf` before editing this notebook. Every section
            below corresponds to a numbered handout question, and every
            explanation cell is part of the submission.
            """
        ),
        md(
            """
            ## How to run this notebook on a free Colab GPU

            The neural stages are deliberately separated. Never keep two Qwen
            models resident at once.

            ```text
            CPU audit and indices
                    ↓
            Stage A: decomposer → save JSONL → unload model and clear CUDA
                    ↓
            Stage B: reranker   → save traces/rankings → unload and clear CUDA
                    ↓
            Stage C: answerer   → save predictions → unload and clear CUDA
            ```

            During development, keep `QUESTION_LIMIT = 2`. Once your tests and
            output schemas are correct, set it to `None`, enable exactly one
            neural stage, and run all cells. Completed question IDs are read
            from JSONL checkpoints, so reconnecting and running again resumes
            rather than starts over. The supplied query vectors mean that you
            must not load or regenerate the Qwen embedding model.
            """
        ),
        code(
            """
            # Run controls. Change one neural stage at a time.
            RUN_DECOMPOSITION_STAGE = False
            RUN_RERANK_AND_RICR_STAGE = False
            RUN_ANSWER_STAGE = False
            RUN_ABLATIONS = False

            # Use 2 while developing. Set to None for every required final run.
            QUESTION_LIMIT = 2
            MOUNT_DRIVE_IN_COLAB = True
            DATASETS = ('2wiki', 'musique')

            # Frozen default configuration from the project specification.
            BEAM_WIDTH = 5
            CANDIDATES_PER_HOP = 15
            RETRIEVAL_POOL = 60
            RRF_CONSTANT = 60.0
            MULTI_PARENT_THRESHOLD = 0.3
            """
        ),
        code(
            """
            from pathlib import Path
            import gc, json, os, platform, shutil, subprocess, sys, time

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
                    str(REPO_ROOT / 'requirements-colab.txt'),
                ], check=True)
                if MOUNT_DRIVE_IN_COLAB:
                    from google.colab import drive
                    drive.mount('/content/drive')
                    WORK_ROOT = Path('/content/drive/MyDrive/panini-course-project-work')
                else:
                    WORK_ROOT = Path('/content/panini-course-project-work')
            else:
                here = Path.cwd().resolve()
                if (here / 'manifest.json').exists():
                    REPO_ROOT = here
                else:
                    source = here
                    while source != source.parent and not (source / 'course_project').exists():
                        source = source.parent
                    if not (source / 'course_project').exists():
                        raise FileNotFoundError('Run from the public repository or the gsw-memory checkout.')
                    REPO_ROOT = source / 'course_project/release/panini_2wiki_100'
                WORK_ROOT = Path(os.environ.get('PANINI_STUDENT_WORK', here / 'panini-student-work'))

            PACKAGE_ROOTS = {
                '2wiki': REPO_ROOT,
                'musique': REPO_ROOT / 'packages/panini_musique_100'
                    if (REPO_ROOT / 'packages').exists()
                    else REPO_ROOT.parent / 'panini_musique_100',
            }
            sys.path.insert(0, str(REPO_ROOT))
            WORK_ROOT.mkdir(parents=True, exist_ok=True)

            # Keep graded source edits in Drive/work storage, not only in the
            # disposable /content clone. Edit this persistent file, then rerun
            # the notebook so it is copied into the importable package.
            STUDENT_CODE_ROOT = WORK_ROOT / 'student_code'
            STUDENT_CODE_ROOT.mkdir(parents=True, exist_ok=True)
            PERSISTENT_RICR = STUDENT_CODE_ROOT / 'ricr.py'
            RUNTIME_RICR = REPO_ROOT / 'panini_course' / 'ricr.py'
            if not PERSISTENT_RICR.exists():
                shutil.copy2(RUNTIME_RICR, PERSISTENT_RICR)
            shutil.copy2(PERSISTENT_RICR, RUNTIME_RICR)

            PERSISTENT_TESTS = STUDENT_CODE_ROOT / 'test_student_ricr.py'
            if not PERSISTENT_TESTS.exists():
                PERSISTENT_TESTS.write_text(
                    "import pytest\\n\\n"
                    "@pytest.mark.skip(reason='Replace with your own reconciliation/RICR test')\\n"
                    "def test_student_failure_case():\\n"
                    "    pass\\n",
                    encoding='utf-8',
                )
            RUNTIME_STUDENT_TESTS = REPO_ROOT / 'tests' / 'test_student_work.py'
            shutil.copy2(PERSISTENT_TESTS, RUNTIME_STUDENT_TESTS)
            print({
                'repo': str(REPO_ROOT), 'work': str(WORK_ROOT), 'colab': IN_COLAB,
                'edit_ricr_here': str(PERSISTENT_RICR),
                'edit_tests_here': str(PERSISTENT_TESTS),
            })
            """
        ),
        md(
            """
            ## Shared checkpoint and memory helpers

            These helpers are infrastructure, not answers to a graded
            algorithm. Each expensive stage appends one complete record at a
            time. Do not keep results only in Python variables.
            """
        ),
        code(
            """
            def read_jsonl(path):
                path = Path(path)
                if not path.exists():
                    return []
                with path.open(encoding='utf-8') as handle:
                    return [json.loads(line) for line in handle if line.strip()]

            def append_jsonl(path, row):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\\n')

            def completed_ids(path, *, configuration=None):
                rows = read_jsonl(path)
                if configuration is not None:
                    rows = [row for row in rows if row.get('configuration') == configuration]
                return {str(row['question_id']) for row in rows}

            def release_gpu(*objects):
                # Delete caller-owned model variables before calling this helper.
                for value in objects:
                    del value
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

            def selected_questions(package):
                rows = package.questions('public') + package.questions('held_out')
                return rows if QUESTION_LIMIT is None else rows[:QUESTION_LIMIT]

            def require_full_run():
                assert QUESTION_LIMIT is None, 'Set QUESTION_LIMIT = None before producing final files.'
            """
        ),
        code(
            """
            import numpy as np
            import pandas as pd
            import networkx as nx
            import matplotlib.pyplot as plt
            from IPython.display import display
            from panini_course import CoursePackage

            packages = {name: CoursePackage(path) for name, path in PACKAGE_ROOTS.items()}
            for name, package in packages.items():
                print(name, package.manifest['counts'])
            """
        ),
        md(
            """
            ## Question 1 — understand and verify the two packages (6 points)

            Audit stable identifiers and artifact alignment before doing any
            retrieval. Equal row counts alone are not sufficient. Inspect one
            public question, held-out question, entity, and QA record from each
            dataset. Confirm that held-out rows expose no answer or gold
            evidence fields.
            """
        ),
        code(
            """
            def audit_package(package, dataset):
                # TODO Q1:
                # 1. Check uniqueness of entity_uid and qa_uid.
                # 2. Check exact ID-set equality between metadata and every
                #    embedding/index ID file used later.
                # 3. Check matrix row counts and embedding dimensions.
                # 4. Check held-out field names for label leakage.
                # 5. Return one flat dictionary for a two-row audit table.
                raise NotImplementedError('Complete the Question 1 package audit')

            # Uncomment after implementing audit_package.
            # audit_table = pd.DataFrame([
            #     audit_package(package, dataset)
            #     for dataset, package in packages.items()
            # ])
            # display(audit_table)
            """
        ),
        explanation(
            "Question 1",
            "Explain why stable-ID set equality catches failures that shape checks miss, and describe one plausible silent alignment bug.",
        ),
        md(
            """
            ## Question 2 — build the GSW network and reconcile entities (12 points)

            Keep three graph objects separate: the document-local occurrence
            projection, the supplied exact-surface sensitivity baseline, and
            your conservative reconciliation. Reconciliation is an analysis
            mapping only; it must never rewrite the packaged IDs.
            """
        ),
        code(
            """
            from panini_course.graph import (
                build_entity_projection,
                build_native_gsw_graph,
                build_unreconciled_entity_projection,
            )

            graph_sets = {}
            for dataset, package in packages.items():
                native = build_native_gsw_graph(package.gsw_paths())
                unreconciled = build_unreconciled_entity_projection(native)
                exact_surface = build_entity_projection(native)
                graph_sets[dataset] = {
                    'native': native,
                    'unreconciled': unreconciled,
                    'exact_surface': exact_surface,
                }
                print(dataset, {name: (g.number_of_nodes(), g.number_of_edges())
                                for name, g in graph_sets[dataset].items()})

            def conservative_entity_mapping(native_graph):
                # TODO Q2: return (occurrence_uid_to_global_id, decision_rows).
                # Your rule must use surface, node type/role compatibility, and
                # local-neighborhood evidence. Block generic attribute values.
                raise NotImplementedError('Implement and document your reconciliation rule')

            def aggregate_projection(unreconciled_graph, mapping):
                # TODO Q2: build a weighted graph after applying the mapping.
                # Preserve occurrence counts and contributing document IDs.
                raise NotImplementedError('Aggregate the conservative projection')

            # TODO Q2: run the fixed-seed manual audit of at least 15 proposed
            # cross-document merges and add the conservative graphs to graph_sets.
            """
        ),
        explanation(
            "Question 2",
            "State your decision rule precisely. Use one false merge and one missed alias to explain why reconciliation changes the mathematical graph rather than merely cleaning labels.",
            "200–300",
        ),
        md(
            """
            ## Question 3 — analyze the structured-memory network (12 points)

            Perform the full sensitivity analysis on 2Wiki. For MuSiQue,
            produce the compact transfer table required by the handout. A
            centrality ranking is not self-interpreting: inspect the semantic
            role, contributing documents, and reconciliation decision for each
            apparent hub.
            """
        ),
        code(
            """
            def network_statistics(graph):
                # TODO Q3: components, giant component, isolates, component-size
                # summary, degree summary, clustering, and assortativity.
                raise NotImplementedError('Compute the required network properties')

            def centrality_audit(graph, top_n=10, seed=232):
                # TODO Q3: weighted degree, PageRank, and exact or fixed-seed
                # approximate betweenness, followed by semantic hub labels.
                raise NotImplementedError('Compute and audit centrality rankings')

            # TODO Q3:
            # - create the required 2Wiki PMF/CCDF plots on log-log axes;
            # - build the sensitivity and MuSiQue transfer tables;
            # - visualize one complete multi-hop evidence path;
            # - save figures under WORK_ROOT / 'figures'.
            """
        ),
        explanation(
            "Question 3",
            "Separate structure present in local GSWs from bridges introduced by reconciliation. Explain why degree alone cannot establish that a hub is semantically real, and do not claim a power law without a fitted comparison.",
            "300–450 total",
        ),
        md(
            """
            ## Stage A — Question 4: decomposition and dependency graphs (14 points)

            This is the first neural stage. It loads only the 4B decomposer,
            writes the raw response before parsing, and unloads the model when
            finished. Implement validation before starting the 200-question
            run. Invalid JSON, forward references, missing nodes, and cycles
            must remain visible in the cache rather than being silently fixed.
            """
        ),
        code(
            """
            import re
            PLACEHOLDER = re.compile(r'<ENTITY_Q(\\d+)>')

            def validate_decomposition(plan):
                # TODO Q4: return {'valid': bool, 'errors': [...], 'edges': [...]}.
                # Check schema, nonempty question text, reference bounds,
                # topological direction, and cycles.
                raise NotImplementedError('Implement decomposition validation')

            def decomposition_metrics(predicted, reviewed):
                # TODO Q4: validity, subquestion-count exact match, dependency
                # edge precision/recall/F1, and retrieval/reasoning flag accuracy.
                raise NotImplementedError('Implement decomposition evaluation')
            """
        ),
        code(
            """
            # Stage A: restartable decomposition. Run only after the validator works.
            if RUN_DECOMPOSITION_STAGE:
                from panini_course.qwen_models import QwenDecomposer

                model_cfg = json.loads((PACKAGE_ROOTS['2wiki'] / 'models/model_config.json').read_text())
                decomposer = QwenDecomposer(
                    model_cfg['decomposer']['model'],
                    PACKAGE_ROOTS['2wiki'] / model_cfg['decomposer']['prompt'],
                    quantized=True,
                )
                try:
                    for dataset, package in packages.items():
                        output = WORK_ROOT / 'cache' / dataset / 'decompositions.jsonl'
                        done = completed_ids(output)
                        for question in selected_questions(package):
                            qid = str(question['question_id'])
                            if qid in done:
                                continue
                            started = time.perf_counter()
                            raw = decomposer.generate_raw(question['question'])
                            record = {
                                'dataset': dataset,
                                'question_id': qid,
                                'question': question['question'],
                                'raw_response': raw,
                                'predicted_decomposition': None,
                                'decomposition_valid': False,
                                'validation_errors': [],
                                'seconds': time.perf_counter() - started,
                            }
                            try:
                                plan = decomposer.parse_response(raw)
                                validation = validate_decomposition(plan)
                                record.update({
                                    'predicted_decomposition': plan,
                                    'decomposition_valid': validation['valid'],
                                    'validation_errors': validation['errors'],
                                    'dependency_edges': validation['edges'],
                                })
                            except Exception as error:
                                record['validation_errors'] = [repr(error)]
                            append_jsonl(output, record)
                            done.add(qid)
                finally:
                    del decomposer
                    release_gpu()
            else:
                print('Stage A disabled. Set RUN_DECOMPOSITION_STAGE=True when ready.')
            """
        ),
        explanation(
            "Question 4",
            "Report the validation and dependency metrics, then classify the first irreversible error in representative failed plans. Explain the difference between a retrieval DAG and a memory graph.",
            "200–300",
        ),
        md(
            """
            ## Question 5 — sparse retrieval baselines (12 points)

            Evaluate TF–IDF and BM25 for both direct QA retrieval and entity
            retrieval followed by local GSW expansion. Use stable QA IDs for
            relevance; do not evaluate by comparing array positions.
            """
        ),
        code(
            """
            from panini_course import BM25Index, TfidfIndex
            from panini_course.metrics import recall_at_k, reciprocal_rank

            def load_sparse_artifacts(root):
                return {
                    'qa_tfidf': TfidfIndex.load(
                        root/'indices/qa_tfidf.npz',
                        root/'indices/qa_tfidf_vectorizer.joblib',
                        root/'indices/qa_ids.json', source='qa_tfidf'),
                    'qa_bm25': BM25Index.load(
                        root/'indices/qa_bm25.joblib',
                        root/'indices/qa_ids.json', source='qa_bm25'),
                    'entity_tfidf': TfidfIndex.load(
                        root/'indices/entity_tfidf.npz',
                        root/'indices/entity_tfidf_vectorizer.joblib',
                        root/'indices/entity_ids.json', source='entity_tfidf'),
                    'entity_bm25': BM25Index.load(
                        root/'indices/entity_bm25.joblib',
                        root/'indices/entity_ids.json', source='entity_bm25'),
                }

            sparse = {name: load_sparse_artifacts(root) for name, root in PACKAGE_ROOTS.items()}

            def evaluate_sparse_retrieval(dataset, package, artifacts, k_values=(1, 5, 10, 15)):
                # TODO Q5: evaluate atomic gold retrieval tasks overall and by
                # question type/hop count. Include latency and failure examples.
                raise NotImplementedError('Implement the sparse baseline evaluator')
            """
        ),
        explanation(
            "Question 5",
            "Compare TF–IDF and BM25 using both aggregate metrics and concrete successes/failures. Explain which wording or token-frequency properties caused the difference.",
        ),
        md(
            """
            ## Question 6 — dense, hybrid, and paper-style dual retrieval (16 points)

            All corpus and required query embeddings are supplied. Load them by
            stable ID and exact query text. A missing required query means your
            deterministic plan or RICR path diverged; it is not permission to
            generate a new embedding.
            """
        ),
        code(
            """
            from panini_course import (
                DenseIndex, DualRetriever, QueryEmbeddingStore,
                reciprocal_rank_fusion,
            )

            def load_dense_artifacts(root):
                query_store = QueryEmbeddingStore.load(
                    root/'embeddings/query_embeddings.npy',
                    root/'embeddings/query_ids.json',
                    root/'embeddings/queries.jsonl')
                return {
                    'queries': query_store,
                    'qa_dense': DenseIndex.load(
                        root/'indices/qa_qwen3_8b_ip.faiss',
                        root/'indices/qa_ids.json', source='qa_dense'),
                    'entity_dense': DenseIndex.load(
                        root/'indices/entity_qwen3_8b_ip.faiss',
                        root/'indices/entity_ids.json', source='entity_dense'),
                }

            dense = {name: load_dense_artifacts(root) for name, root in PACKAGE_ROOTS.items()}

            def retrieve_candidate_pool(dataset, query, pool_size=RETRIEVAL_POOL, backend='dual'):
                # TODO Q6:
                # - implement direct dense QA, RRF, and paper-style dual retrieval;
                # - dual retrieval is BM25 entity search + local GSW QA expansion
                #   unioned with direct dense QA retrieval;
                # - deduplicate by qa_uid and retain provenance/ranks;
                # - return QA records ready for Question 7 reranking.
                raise NotImplementedError('Implement dense, RRF, and dual candidate retrieval')

            # TODO Q6: manually verify FAISS inner products for five supplied
            # query vectors, then evaluate all retrieval variants consistently.
            """
        ),
        explanation(
            "Question 6",
            "Use the measured tables to explain where dense retrieval, RRF, and entity expansion help or hurt. Include candidate-set overlap and at least one error attributable to the retrieval pool rather than reranking.",
            "200–300",
        ),
        md(
            """
            ## Stage B — Question 7: reranking without exceeding Colab memory (8 points)

            The reranker is the only neural model in this stage. Use the 8B
            checkpoint in 4-bit mode when it fits; after an actual CUDA OOM,
            clear memory and use the configured 4B fallback with batch size 1
            and 256-token inputs. Cache rankings by exact instantiated query so
            RICR replay and ablations do not rerun the same model call.
            """
        ),
        code(
            """
            from panini_course import Candidate

            def format_qa_for_reranker(qa_row):
                # TODO Q7: format only the grounded QA record and its answer
                # role/state information. Do not pass source documents.
                raise NotImplementedError('Format one QA candidate')

            def rerank_and_convert(dataset, query, reranker, *, top_k=CANDIDATES_PER_HOP):
                # TODO Q7:
                # 1. retrieve a union pool with retrieve_candidate_pool;
                # 2. score its formatted QA records with QwenReranker;
                # 3. combine/calibrate retrieval and reranker scores as specified;
                # 4. return Candidate objects, one per QA, keeping all answers;
                # 5. namespace each local answer ID with document/GSW provenance.
                raise NotImplementedError('Implement reranking and Candidate conversion')

            def choose_reranker_config(model_cfg):
                try:
                    import torch
                    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
                except Exception:
                    total_gib = 0.0
                use_8b = total_gib >= model_cfg['reranker']['use_8b_at_or_above_gib']
                return {
                    'model': (model_cfg['reranker']['model'] if use_8b
                              else model_cfg['reranker']['free_colab_t4_fallback']),
                    'batch_size': 1 if total_gib < 20 else 8,
                    'max_length': 256 if total_gib < 20 else 2048,
                    'total_gib': total_gib,
                }
            """
        ),
        explanation(
            "Question 7",
            "State whether reranking improved the candidate order and whether it changed complete-pool recall. Show a case where reranking demoted a strong exact match or promoted a semantically better candidate.",
        ),
        md(
            """
            ## Question 8 — implement connected-DAG RICR (22 points)

            Complete `panini_course/ricr.py`, not an alternative linear search
            in this notebook. Your implementation must combine all parents at a
            converging node, use harmonic-mean parent combinations and the 0.3
            threshold/fallback, group intermediate answer entities, retain
            final-hop QA alternatives, and gather evidence from all final
            beams. Run the supplied tests before starting Stage B.
            """
        ),
        code(
            """
            # In Colab, edit PERSISTENT_RICR and PERSISTENT_TESTS through the
            # file browser, then rerun from setup to sync ricr.py into the clone.
            # These tests are skipped until the two scaffold functions no longer
            # raise NotImplementedError, then turn on automatically.
            test_root = REPO_ROOT / 'tests'
            subprocess.run(
                [sys.executable, '-m', 'pytest', '-q', str(test_root)],
                check=False,
            )

            from panini_course import run_panini_ricr

            toy_plan = [
                {'question': 'Who directed Film A?', 'requires_retrieval': True},
                {'question': 'Who directed Film B?', 'requires_retrieval': True},
                {'question': 'Who was born later, <ENTITY_Q1> or <ENTITY_Q2>?',
                 'requires_retrieval': True},
            ]
            # TODO Q8: add hand-calculable candidates, execute this converging
            # DAG, and assert its issued query, beam scores, and evidence set.
            """
        ),
        code(
            """
            # Stage B: the completed RICR implementation drives dynamic queries.
            if RUN_RERANK_AND_RICR_STAGE:
                from dataclasses import asdict
                from panini_course.qwen_models import QwenReranker

                model_cfg = json.loads((PACKAGE_ROOTS['2wiki'] / 'models/model_config.json').read_text())
                selected = choose_reranker_config(model_cfg)
                print('reranker selection:', selected)
                try:
                    reranker = QwenReranker(
                        selected['model'], quantized=True,
                        max_length=selected['max_length'])
                except RuntimeError as error:
                    if 'out of memory' not in str(error).casefold():
                        raise
                    release_gpu()
                    selected.update({
                        'model': model_cfg['reranker']['free_colab_t4_fallback'],
                        'batch_size': 1,
                        'max_length': 256,
                    })
                    reranker = QwenReranker(
                        selected['model'], quantized=True, max_length=256)
                try:
                    for dataset, package in packages.items():
                        output = WORK_ROOT / 'cache' / dataset / 'ricr_traces.jsonl'
                        done = completed_ids(output, configuration='default')
                        plans = {row['question_id']: row for row in read_jsonl(
                            WORK_ROOT / 'cache' / dataset / 'decompositions.jsonl')}
                        for question in selected_questions(package):
                            qid = str(question['question_id'])
                            if qid in done:
                                continue
                            plan_row = plans.get(qid)
                            if not plan_row or not plan_row.get('decomposition_valid'):
                                append_jsonl(output, {
                                    'dataset': dataset, 'configuration': 'default',
                                    'question_id': qid, 'error': 'invalid decomposition'})
                                continue
                            started = time.perf_counter()
                            result = run_panini_ricr(
                                plan_row['predicted_decomposition'],
                                lambda query, k: rerank_and_convert(
                                    dataset, query, reranker, top_k=k),
                                original_question=question['question'],
                                beam_width=BEAM_WIDTH,
                                candidates_per_hop=CANDIDATES_PER_HOP,
                                multi_dependency_threshold=MULTI_PARENT_THRESHOLD,
                            )
                            append_jsonl(output, {
                                'dataset': dataset,
                                'configuration': 'default',
                                'question_id': qid,
                                'question': question['question'],
                                'reranker_model': selected['model'],
                                'trace': asdict(result),
                                'retrieval_seconds': time.perf_counter() - started,
                                'error': None,
                            })
                            done.add(qid)
                finally:
                    del reranker
                    release_gpu()
            else:
                print('Stage B disabled. Complete Questions 6–8 first.')
            """
        ),
        explanation(
            "Question 8",
            "Show your hand calculation and explain why independent linear branches are not equivalent to executing a converging retrieval DAG. Identify where beam diversity is gained or lost.",
            "200–300",
        ),
        md(
            """
            ## Question 9 — controlled RICR ablations (14 points)

            Change one factor at a time on the fixed 20-question ablation slice.
            Reuse cached candidate rankings whenever the instantiated query and
            backend are unchanged. Do not compare a warm cache time with an
            uncached model time as if they were equivalent.
            """
        ),
        code(
            """
            ablation_configs = [
                {'name': 'default', 'beam': 5, 'k': 15, 'backend': 'dual',
                 'unique': True, 'score': 'geometric', 'parent_threshold': 0.3},
                {'name': 'beam_1', 'beam': 1},
                {'name': 'beam_3', 'beam': 3},
                {'name': 'k_5', 'k': 5},
                {'name': 'unique_off', 'unique': False},
                {'name': 'last_hop', 'score': 'last_hop'},
                {'name': 'parent_threshold_off', 'parent_threshold': 0.0},
                {'name': 'bm25', 'backend': 'bm25'},
                {'name': 'dense', 'backend': 'dense'},
                {'name': 'rrf', 'backend': 'rrf'},
            ]

            def run_ablation(configuration, dataset, questions):
                # TODO Q9: merge each partial dictionary with the default,
                # execute exactly the same fixed questions, append traces, and
                # report chain recovery, answer metrics, evidence size, and a
                # valid cold or consistently cached latency measurement.
                raise NotImplementedError('Implement the controlled ablation runner')

            if RUN_ABLATIONS:
                require_full_run()
                # TODO Q9: use the first 20 public questions from each dataset.
                raise NotImplementedError('Run and checkpoint every ablation')
            """
        ),
        explanation(
            "Question 9",
            "Use the measured table to recommend a configuration. Discuss accuracy, chain recovery, latency, and at least one interaction that a one-factor ablation cannot establish.",
            "250–350",
        ),
        md(
            """
            ## Stage C — Question 10: end-to-end 2Wiki evaluation (10 points)

            Stage C reads saved RICR traces and loads only Qwen3-4B. The answer
            model receives deduplicated QA evidence from every surviving final
            beam, including role/state strings. It must not receive source
            documents, graph neighbors, or gold labels. The supplied wrapper
            already implements the required four-message one-shot PANINI prompt;
            do not add an `N/A` instruction for these answerable splits.
            """
        ),
        code(
            """
            from panini_course.metrics import exact_match, token_f1

            def format_evidence(candidate):
                # TODO Q10: produce one grounded line containing stored question,
                # all answer names, and answer role/state strings. Do not include
                # source passages or gold final answers.
                raise NotImplementedError('Format answer-model evidence')

            if RUN_ANSWER_STAGE:
                from panini_course.qwen_models import QwenAnswerer

                model_cfg = json.loads((PACKAGE_ROOTS['2wiki'] / 'models/model_config.json').read_text())
                answerer = QwenAnswerer(model_cfg['answer_model']['model'], quantized=True)
                try:
                    for dataset, package in packages.items():
                        traces = {row['question_id']: row for row in read_jsonl(
                            WORK_ROOT / 'cache' / dataset / 'ricr_traces.jsonl')
                            if row.get('configuration') == 'default'}
                        output = WORK_ROOT / 'cache' / dataset / 'answers.jsonl'
                        done = completed_ids(output)
                        for question in selected_questions(package):
                            qid = str(question['question_id'])
                            if qid in done or qid not in traces:
                                continue
                            trace = traces[qid]
                            if trace.get('error'):
                                continue
                            evidence_rows = trace['trace']['evidence']
                            evidence = [format_evidence(row) for row in evidence_rows]
                            started = time.perf_counter()
                            generated = answerer.answer_with_trace(question['question'], evidence)
                            record = {
                                'dataset': dataset,
                                'question_id': qid,
                                'question': question['question'],
                                'predicted_answer': generated['answer'],
                                'raw_answer_response': generated['response'],
                                'answer_evidence': evidence,
                                'answer_seconds': time.perf_counter() - started,
                            }
                            if 'answer' in question:
                                gold = [question['answer'], *question.get('answer_aliases', [])]
                                record['exact_match'] = exact_match(generated['answer'], gold)
                                record['token_f1'] = token_f1(generated['answer'], gold)
                            append_jsonl(output, record)
                            done.add(qid)
                finally:
                    del answerer
                    release_gpu()
            else:
                print('Stage C disabled. Complete and checkpoint Stage B first.')
            """
        ),
        code(
            """
            def score_default_run(dataset, package):
                # TODO Q10/Q11: join plans, traces, answers, and public labels by
                # question_id. Compute supporting-QA/document recall,
                # complete-chain recovery, EM/F1, evidence size, and latency;
                # summarize 2Wiki by type and MuSiQue by hop_count.
                raise NotImplementedError('Implement end-to-end scoring')

            # TODO Q10: display the overall and per-type 2Wiki tables, plus one
            # successful and one failed trace with the first irreversible error.
            """
        ),
        explanation(
            "Question 10",
            "Interpret the 2Wiki result by failure stage: decomposition, candidate recall, later-hop substitution/pruning, or answer generation. A correct final answer does not by itself prove complete-chain recovery.",
            "250–350",
        ),
        md(
            """
            ## Question 11 — MuSiQue transfer and scaling (12 points)

            Freeze all 2Wiki choices before inspecting MuSiQue results. Use the
            same decomposition prompt, retrieval settings, beam rules,
            reranker, answer prompt, and metrics. Compare 2-, 3-, and 4-hop
            questions without tuning on MuSiQue labels.
            """
        ),
        code(
            """
            # TODO Q11: run score_default_run for MuSiQue, display results by
            # hop_count, and plot both complete-chain recovery and answer F1 as
            # chain length increases. Attribute errors using saved traces.

            # Example shape only; replace with your measured dataframe.
            musique_by_hop = pd.DataFrame(columns=[
                'hop_count', 'questions', 'supporting_qa_recall',
                'complete_chain_recovery', 'EM', 'F1'])
            display(musique_by_hop)
            """
        ),
        explanation(
            "Question 11",
            "Explain why complete-chain recovery is conjunctive while answer F1 is not. Separate the effect of hop count from dataset wording, entity distribution, and GSW coverage.",
            "250–350",
        ),
        md(
            """
            ## Question 12 — reproducibility and final submission (12 points)

            Materialize exactly four submission files: 80 labeled development
            results and 20 label-free held-out predictions for each dataset.
            Verify counts by stable question ID, not just line count. Record the
            environment, frozen configuration, model IDs, resume instructions,
            and which timings were cold versus cache replay.
            """
        ),
        code(
            """
            SUBMISSION_ROOT = WORK_ROOT / 'submission'
            required_files = {
                SUBMISSION_ROOT/'results/2wiki_dev.jsonl': 80,
                SUBMISSION_ROOT/'predictions/2wiki_heldout.jsonl': 20,
                SUBMISSION_ROOT/'results/musique_dev.jsonl': 80,
                SUBMISSION_ROOT/'predictions/musique_heldout.jsonl': 20,
            }

            def materialize_submission(dataset, package):
                # TODO Q12: split cached answers by the package's public and
                # held-out ID sets. Development rows include metrics; held-out
                # rows contain only IDs, questions, predictions, and allowed
                # trace/runtime metadata. Never copy labels into held-out files.
                raise NotImplementedError('Write the required submission files')

            def validate_submission_file(path, expected, allowed_heldout_fields=None):
                # TODO Q12: uniqueness, exact ID-set match, count, valid JSON,
                # finite metrics, and held-out field allowlist checks.
                raise NotImplementedError('Implement final submission validation')

            if QUESTION_LIMIT is None and all(path.exists() for path in required_files):
                for path, expected in required_files.items():
                    validate_submission_file(path, expected)
                print('All required submission files validated.')
            else:
                print('Final validation waits until QUESTION_LIMIT=None and all files exist.')
            """
        ),
        code(
            """
            # TODO Q12: write environment.txt and RUNME.md. Include at least:
            environment = {
                'python': sys.version,
                'platform': platform.platform(),
                'question_limit': QUESTION_LIMIT,
                'beam_width': BEAM_WIDTH,
                'candidates_per_hop': CANDIDATES_PER_HOP,
                'retrieval_pool': RETRIEVAL_POOL,
                'rrf_constant': RRF_CONSTANT,
                'multi_parent_threshold': MULTI_PARENT_THRESHOLD,
                'package_manifests': {
                    name: package.manifest for name, package in packages.items()
                },
            }
            (WORK_ROOT/'environment.json').write_text(
                json.dumps(environment, indent=2, ensure_ascii=False), encoding='utf-8')
            print('environment:', WORK_ROOT/'environment.json')
            """
        ),
        explanation(
            "Question 12",
            "Tell the complete system story from input question to final answer, identify the design’s main protection and main failure mode, and explain how another student can resume and reproduce your run.",
            "300–450",
        ),
        md(
            """
            ## Final pre-submission checklist

            - [ ] `QUESTION_LIMIT` was set to `None` for final runs.
            - [ ] All supplied tests pass, plus your reconciliation and failure-case tests.
            - [ ] Every expensive stage resumes from JSONL by stable question ID.
            - [ ] Only one Qwen model was resident at a time.
            - [ ] No embeddings were generated.
            - [ ] The 2Wiki configuration was frozen before MuSiQue evaluation.
            - [ ] All twelve written-response cells were replaced with your own analysis.
            - [ ] Held-out outputs contain no answer, alias, or supporting-evidence labels.
            - [ ] Four required JSONL files, the executed notebook, figures, tests,
              `environment.json`, and `RUNME.md` are included.
            """
        ),
    ]

    nbf.write(notebook, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
