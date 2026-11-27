#!/usr/bin/env python3
"""Compares expected vs. actual interop results produced by run-queries.sh.

Comparison rules (see the implementation plan for rationale):
  - Hit document ID order: exact match (both platforms must agree on ranking).
  - Stored field values: exact match (raw bytes are platform-independent).
  - Lexical query scores: relative error <= 1e-5.
  - Vector query scores: relative error <= 1e-3 (query embedding is
    recomputed on the actual side, so bit-identical scores are not expected).

Usage: compare-results.py <expected-dir> <actual-dir> <queries-tsv>
"""

import json
import sys
from pathlib import Path

EPSILON_BY_MODE = {"lexical": 1e-5, "vector": 1e-3}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_search_results(path):
    """The results of a `laurus --format json search` output file.

    The CLI prints `{"results": [...]}`, with `"total_hits"` when asked for
    (Issue #1472); it printed a bare array of results before. Both sides of
    the check run the same binary, so both files have this shape.
    """
    return load_json(path)["results"]


def relative_error(a, b):
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def compare_search_results(name, mode, expected, actual, epsilon, errors):
    exp_ids = [r["id"] for r in expected]
    act_ids = [r["id"] for r in actual]
    if exp_ids != act_ids:
        errors.append(
            f"[{name}] hit doc ID order mismatch:\n"
            f"  expected: {exp_ids}\n"
            f"  actual:   {act_ids}"
        )
        return

    for exp_r, act_r in zip(expected, actual):
        exp_doc = exp_r.get("fields")
        act_doc = act_r.get("fields")
        if exp_doc != act_doc:
            errors.append(
                f"[{name}] stored fields mismatch for doc '{exp_r['id']}':\n"
                f"  expected: {exp_doc}\n"
                f"  actual:   {act_doc}"
            )

        err = relative_error(exp_r["score"], act_r["score"])
        if err > epsilon:
            errors.append(
                f"[{name}] score mismatch for doc '{exp_r['id']}': "
                f"expected={exp_r['score']}, actual={act_r['score']}, "
                f"relative_error={err:.2e} > epsilon={epsilon:.0e}"
            )


def compare_stats(expected_dir, actual_dir, errors):
    expected = load_json(expected_dir / "stats.json")
    actual = load_json(actual_dir / "stats.json")
    if expected != actual:
        errors.append(f"[stats] mismatch:\n  expected: {expected}\n  actual:   {actual}")


def compare_docs(expected_dir, actual_dir, errors):
    doc_files = sorted(expected_dir.glob("doc-*.json"))
    if not doc_files:
        errors.append(f"[docs] no doc-*.json files found under {expected_dir}")
    for path in doc_files:
        name = path.name
        expected = load_json(path)
        actual_path = actual_dir / name
        if not actual_path.exists():
            errors.append(f"[{name}] missing in actual results")
            continue
        actual = load_json(actual_path)
        if expected != actual:
            errors.append(f"[{name}] mismatch:\n  expected: {expected}\n  actual:   {actual}")


def load_queries(queries_tsv):
    queries = []
    with open(queries_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            name, mode, _limit, _query = line.split("\t", 3)
            queries.append((name, mode))
    return queries


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <expected-dir> <actual-dir> <queries-tsv>", file=sys.stderr)
        return 2

    expected_dir = Path(sys.argv[1])
    actual_dir = Path(sys.argv[2])
    queries_tsv = Path(sys.argv[3])

    errors = []
    compare_stats(expected_dir, actual_dir, errors)
    compare_docs(expected_dir, actual_dir, errors)

    queries = load_queries(queries_tsv)
    for name, mode in queries:
        epsilon = EPSILON_BY_MODE.get(mode)
        if epsilon is None:
            errors.append(f"[{name}] unknown mode '{mode}' in {queries_tsv}")
            continue
        filename = f"query-{name}.json"
        expected_path = expected_dir / filename
        actual_path = actual_dir / filename
        if not expected_path.exists() or not actual_path.exists():
            errors.append(f"[{name}] missing result file(s): {filename}")
            continue
        expected = load_search_results(expected_path)
        actual = load_search_results(actual_path)
        compare_search_results(name, mode, expected, actual, epsilon, errors)

    if errors:
        print(f"FAIL: {len(errors)} mismatch(es) found\n", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
            print(file=sys.stderr)
        return 1

    print(f"OK: all {len(queries)} queries + stats + docs matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
