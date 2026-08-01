# Instructor-only material

Do not include this directory in the student distribution.

`panini_2wiki_100_gold/questions_with_gold.jsonl` contains all 100 answer and
evidence records. `reviewed_decompositions.jsonl` contains the complete
reviewed decomposition set. Question IDs match the standalone
`release/panini_2wiki_100` package.

`Panini_Student_Like_Solution.ipynb` is the instructor reference notebook. It
keeps the narrative and implementation style expected from a strong student
submission, includes a CPU-only smoke mode, and splits the full neural run into
cached Colab-sized stages. Do not copy it into a student release because it
contains a reference RICR implementation.

That earlier notebook is retained as a compact smoke demonstration. Use
`Panini_Full_Answer_Key_Colab.ipynb` for the graded Questions 1--12 workflow and
the low-memory encoder/reranker schedule.

`COLAB_FEASIBILITY.md` records the measured 4-bit GPU memory and timing checks,
the recommended shard size, and the limitations of free Colab runtimes.

`Panini_Full_Answer_Key_Colab.ipynb` is the complete Questions 1--12 answer
key. It runs both 100-question packages, saves every expensive stage to Drive,
performs the RICR ablations, and writes the four required development/held-out
JSONL files. `colab_full_solution.py` is its testable source implementation;
`build_full_answer_key_notebook.py` embeds that implementation into the
self-contained notebook.

`reconciliation_audit.csv` is the fixed 30-pair human audit used in Questions
2--3. It includes correct, incorrect, and uncertain decisions plus two missed
alias examples discussed in the notebook.

After the complete run, `summarize_full_run.py` writes the measured tables to
`FULL_RUN_RESULTS.md` and copies the four required JSONL deliverables plus the
runtime record into `full_run_outputs/`. These are instructor-only reference
outputs and must not be published in the student repository.
