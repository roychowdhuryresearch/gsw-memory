#!/usr/bin/env python3
"""Verify SHA-256 hashes in a standalone Panini course release."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative_path, expected in manifest["artifacts"].items():
        path = root / relative_path
        if not path.exists():
            failures.append(f"missing: {relative_path}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"hash mismatch: {relative_path}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(manifest['artifacts'])} release artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
