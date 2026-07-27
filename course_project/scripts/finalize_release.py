#!/usr/bin/env python3
"""Copy standalone code and assets into a student release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


MODEL_CONFIG = {
    "decomposer": {
        "model": "yigitturali/GSW-QA-Decomposer-Qwen3-4B",
        "models": {
            "colab_default_4b": "yigitturali/GSW-QA-Decomposer-Qwen3-4B",
            "higher_capacity_8b": "yigitturali/GSW-QA-Decomposer-Qwen3-8B",
        },
        "model_links": {
            "colab_default_4b": (
                "https://huggingface.co/yigitturali/"
                "GSW-QA-Decomposer-Qwen3-4B"
            ),
            "higher_capacity_8b": (
                "https://huggingface.co/yigitturali/"
                "GSW-QA-Decomposer-Qwen3-8B"
            ),
        },
        "prompt": "models/decomposition_prompt.txt",
        "quantization": "4bit-nf4-recommended-for-colab",
        "temperature": 0.0,
    },
    "embedding": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "dimension": 4096,
        "normalized": True,
        "corpus_embeddings_supplied": True,
    },
    "reranker": {
        "model": "Qwen/Qwen3-Reranker-8B",
        "quantization": "4bit-nf4-recommended-for-colab",
    },
    "answer_model": {
        "model": "Qwen/Qwen3-4B",
        "quantization": "4bit-nf4-recommended-for-colab",
        "evidence_only": True,
        "insufficient_evidence_token": "N/A",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def standalone_prompt(source: Path) -> str:
    text = source.read_text(encoding="utf-8").strip()
    if text.startswith('f"""') and text.endswith('"""'):
        text = text[4:-3]
    text = text.replace("{input['question']}", "{question}")
    return text.strip() + "\n"


def remove_checksum_fields(value: Any) -> Any:
    """Remove legacy checksum metadata from student-facing JSON."""
    if isinstance(value, dict):
        return {
            key: remove_checksum_fields(item)
            for key, item in value.items()
            if not key.endswith("_sha256")
        }
    if isinstance(value, list):
        return [remove_checksum_fields(item) for item in value]
    return value


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    package = args.package.resolve()
    project = args.project_root.resolve()
    instructor_output = args.instructor_output.resolve()

    source_instructor = package / "instructor"
    if source_instructor.exists():
        if instructor_output.exists():
            shutil.rmtree(instructor_output)
        shutil.copytree(source_instructor, instructor_output)
        shutil.rmtree(source_instructor)

    source_package = project / "src" / "panini_course"
    target_package = package / "panini_course"
    if target_package.exists():
        shutil.rmtree(target_package)
    shutil.copytree(
        source_package,
        target_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    student_overrides = project / "student_overrides" / "panini_course"
    for source in student_overrides.glob("*.py"):
        shutil.copy2(source, target_package / source.name)

    shutil.copy2(
        project / "requirements-colab.txt",
        package / "requirements-colab.txt",
    )
    shutil.copy2(project / "PROJECT_SPEC.md", package / "PROJECT_SPEC.md")
    shutil.copy2(args.data_card.resolve(), package / "DATA_CARD.md")
    shutil.copy2(project / "quickstart.py", package / "quickstart.py")
    shutil.copy2(project / "TESTING.md", package / "TESTING.md")
    shutil.copy2(
        project / "Panini_Course_Project.ipynb",
        package / "Panini_Course_Project.ipynb",
    )
    shutil.copy2(args.readme.resolve(), package / "README.md")
    target_tests = package / "tests"
    if target_tests.exists():
        shutil.rmtree(target_tests)
    shutil.copytree(
        project / "tests",
        target_tests,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    handout = project / "handout" / "project3.pdf"
    if handout.exists():
        shutil.copy2(handout, package / "PROJECT_HANDOUT.pdf")

    models_dir = package / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prompt = standalone_prompt(args.decomposition_prompt.resolve())
    (models_dir / "decomposition_prompt.txt").write_text(
        prompt, encoding="utf-8"
    )
    write_json(models_dir / "model_config.json", MODEL_CONFIG)

    # Remove legacy checksum artifacts from previously finalized packages.
    for obsolete in ("verify_release.py", "release_manifest.json"):
        path = package / obsolete
        if path.exists():
            path.unlink()

    for relative in ("manifest.json", "embeddings/embedding_manifest.json"):
        path = package / relative
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, remove_checksum_fields(metadata))

    dataset_manifest = json.loads((package / "manifest.json").read_text(
        encoding="utf-8"
    ))
    return {
        "dataset_counts": dataset_manifest["counts"],
        "models": MODEL_CONFIG,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--instructor-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "instructor"
        / "panini_2wiki_100_gold",
        help="Separate destination for held-out labels and full decompositions.",
    )
    parser.add_argument("--decomposition-prompt", type=Path, required=True)
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "STANDALONE_README.md",
    )
    parser.add_argument(
        "--data-card",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "DATA_CARD.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    release = finalize(parse_args(argv))
    print(json.dumps(
        {
            "dataset_counts": release["dataset_counts"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
