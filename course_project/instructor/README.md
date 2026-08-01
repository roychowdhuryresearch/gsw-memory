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

`COLAB_FEASIBILITY.md` records the measured 4-bit GPU memory and timing checks,
the recommended shard size, and the limitations of free Colab runtimes.
