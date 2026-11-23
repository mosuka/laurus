#!/usr/bin/env python3
"""Check that the change filters of the Regression workflow follow the
crates' dependencies.

Detect Changes in `.github/workflows/regression.yml` runs
`dorny/paths-filter`, and each test job runs on a pull request only when its
filter matches a changed path. A filter that misses a directory its job
builds skips the job on a pull request that changes only that directory:
before Issue #1714, a change to laurus-server alone did not run the tests of
laurus-mcp and laurus-cli, which depend on it.

This script reads the filters, and the workspace crates and their
dependencies on each other from `cargo metadata`, and fails when the filter
of a crate's test job does not match

* a path in the crate and in every workspace crate it depends on, directly
  or not (normal, build and dev dependencies alike);
* the inputs every job reads (`SHARED_INPUTS` and each composite action in
  `.github/actions/`);
* the extra inputs of its job (`EXTRA_INPUTS`).

The `code` filter (Format, Clippy and Rustdoc) must match the shared inputs,
every crate and its extra inputs (`EXTRA_INPUTS`, the script of the Rustdoc
job). Which filter tests which crate is the table `CRATE_FILTERS`; every
workspace crate must be in it, and every filter of the workflow must be in it
or in `OTHER_FILTERS`, so a new crate or filter fails until the table says
what it is.

The filters are read without a YAML library, which the CI runner does not
have: only the forms the workflow uses are read (`name:`, `name: &anchor`,
`- 'pattern'`, `- "pattern"`, `- pattern`, `- *alias`, comments and blank
lines), and any other line fails the check rather than being misread. An
alias stands for the patterns of the filter that defined the anchor, as
`dorny/paths-filter` flattens nested lists. Patterns are matched as
picomatch does with `{dot: true}`, which the action uses, for the glob
syntax they may contain: `**` as a whole path segment spans any number of
segments (none included), and `*` within a segment matches anything but
`/`, a leading `.` included. Any other glob character fails the check.

Usage:
  python3 scripts/check-ci-filters.py
  python3 scripts/check-ci-filters.py --workflow <file> --metadata <file>

`--workflow` reads the filters from another file (by default the workflow
above), and `--metadata` reads the output of
`cargo metadata --format-version 1 --no-deps` from a file instead of running
cargo. Exit code 0 when every filter matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

WORKFLOW = Path(".github/workflows/regression.yml")

# The filter of each crate's test job, and the crate it tests.
CRATE_FILTERS = {
    "laurus": "laurus",
    "server": "laurus-server",
    "mcp": "laurus-mcp",
    "cli": "laurus-cli",
    "python": "laurus-python",
    "nodejs": "laurus-nodejs",
    "wasm": "laurus-wasm",
    "ruby": "laurus-ruby",
    "php": "laurus-php",
    # Index Interop builds the CLI (.github/workflows/index-interop.yml).
    "interop": "laurus-cli",
}

# Filters that gate no single crate's test job.
OTHER_FILTERS = {
    # Anchor only: the inputs every filter below includes.
    "shared",
    # Format and Clippy, checked against every crate instead.
    "code",
    # Binding Snippets, which builds no crate.
    "snippets",
}

# Inputs every job reads, besides the composite actions.
SHARED_INPUTS = [
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/config.toml",
    str(WORKFLOW),
]

# Inputs of a job besides its crates and the shared inputs.
EXTRA_INPUTS = {
    "interop": [".github/workflows/index-interop.yml", ".github/interop/probe"],
    # The Rustdoc job runs this script (Issue #1675).
    "code": ["scripts/check-rustdoc.py"],
}

KEY = re.compile(r"(?P<name>[A-Za-z0-9_-]+):(?:\s+&(?P<anchor>[A-Za-z0-9_-]+))?\s*(?:#.*)?")
# A plain (unquoted) pattern is limited to the characters a pattern may use,
# and cannot start with `*`, which YAML reads as an alias; anything else (a
# `status: pattern` mapping, a flow list `[a, b]`, an escape in quotes) is
# not read.
ITEM = re.compile(
    r"-\s+(?:'(?P<single>[^']*)'"
    r"|\"(?P<double>[^\"\\]*)\""
    r"|\*(?P<alias>[A-Za-z0-9_-]+)"
    r"|(?P<plain>[A-Za-z0-9._/-][A-Za-z0-9._/*-]*))\s*(?:#.*)?"
)
GLOB_SEGMENT = re.compile(r"[A-Za-z0-9._*-]+")


class FilterError(Exception):
    """A filter block this script cannot read, or a pattern it cannot match."""


def filters_block(text: str) -> list[tuple[int, str]]:
    """The lines of the one `filters: |` block in `text`, numbered from 1.

    The block ends at the first non-blank line indented no deeper than the
    `filters:` key.
    """
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*filters: \|\s*", line)]
    if len(starts) != 1:
        raise FilterError(f"expected one `filters: |` block, found {len(starts)}")
    start = starts[0]
    key_indent = len(lines[start]) - len(lines[start].lstrip())
    block = []
    for number, line in enumerate(lines[start + 1 :], start + 2):
        if line.strip() and len(line) - len(line.lstrip()) <= key_indent:
            break
        block.append((number, line))
    return block


def parse_filters(block: list[tuple[int, str]]) -> dict[str, list[str]]:
    """Each filter's patterns, with aliases replaced by the anchored filter's."""
    filters: dict[str, list[str]] = {}
    anchors: dict[str, str] = {}
    key_indent = None
    current = None
    for number, line in block:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if key_indent is None:
            key_indent = indent
        key = KEY.fullmatch(stripped)
        item = ITEM.fullmatch(stripped)
        if indent == key_indent and key:
            current = key["name"]
            if current in filters:
                raise FilterError(f"line {number}: filter `{current}` defined twice")
            filters[current] = []
            if key["anchor"]:
                anchors[key["anchor"]] = current
        elif indent > key_indent and item and current is not None:
            if item["alias"]:
                target = anchors.get(item["alias"])
                if target is None or target == current:
                    raise FilterError(
                        f"line {number}: alias `*{item['alias']}` names no earlier anchor"
                    )
                filters[current].extend(filters[target])
            else:
                pattern = next(item[g] for g in ("single", "double", "plain") if item[g] is not None)
                filters[current].append(pattern)
        else:
            raise FilterError(f"line {number}: unsupported filter syntax: {stripped}")
    for name, patterns in filters.items():
        if not patterns:
            raise FilterError(f"filter `{name}` has no patterns")
    return filters


def glob_regex(pattern: str) -> re.Pattern[str]:
    """The regular expression picomatch, with `{dot: true}`, matches `pattern` as."""
    segments = pattern.split("/")
    parts = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            if index and segments[index - 1] == "**":
                raise FilterError(f"unsupported pattern: {pattern}")
            if len(segments) == 1:
                parts.append(".*")
            elif index == 0:
                parts.append("(?:.*/)?")
            elif last:
                # `a/**` matches `a` itself too: drop the slash before it.
                parts[-1] = parts[-1][: -len("/")]
                parts.append("(?:/.*)?")
            else:
                parts.append("(?:.*/)?")
            continue
        if not GLOB_SEGMENT.fullmatch(segment) or "**" in segment:
            raise FilterError(f"unsupported pattern: {pattern}")
        regex = "".join("[^/]*" if c == "*" else re.escape(c) for c in segment)
        parts.append(regex if last else regex + "/")
    return re.compile("".join(parts))


def matches(patterns: list[str], path: str) -> bool:
    return any(glob_regex(p).fullmatch(path) for p in patterns)


def workspace(metadata: dict) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Each workspace crate's directory (relative to the workspace root) and
    the workspace crates it depends on, transitively."""
    root = Path(metadata["workspace_root"])
    ids = set(metadata["workspace_members"])
    crates = {p["name"]: p for p in metadata["packages"] if p["id"] in ids}
    dirs = {
        name: Path(package["manifest_path"]).parent.relative_to(root).as_posix()
        for name, package in crates.items()
    }
    direct = {
        name: {d["name"] for d in package["dependencies"] if d["name"] in crates and "path" in d}
        for name, package in crates.items()
    }
    closed = {}
    for name in crates:
        seen: set[str] = set()
        stack = list(direct[name])
        while stack:
            dep = stack.pop()
            if dep not in seen:
                seen.add(dep)
                stack.extend(direct[dep])
        closed[name] = seen
    return dirs, closed


def crate_paths(directory: str) -> list[str]:
    """Paths that stand for a crate's directory."""
    prefix = "" if directory in ("", ".") else directory + "/"
    return [f"{prefix}Cargo.toml", f"{prefix}src/probe.rs"]


def check(
    filters: dict[str, list[str]],
    dirs: dict[str, str],
    deps: dict[str, set[str]],
    actions: list[str],
) -> list[str]:
    """Every way the filters fall short; empty when they do not."""
    problems = []
    for crate in sorted(set(dirs) - set(CRATE_FILTERS.values())):
        problems.append(f"crate {crate} is not in CRATE_FILTERS: which filter tests it?")
    for name in sorted(set(filters) - set(CRATE_FILTERS) - OTHER_FILTERS):
        problems.append(f"filter `{name}` is in neither CRATE_FILTERS nor OTHER_FILTERS")
    for name in sorted((set(CRATE_FILTERS) | OTHER_FILTERS) - set(filters) - {"shared"}):
        problems.append(f"filter `{name}` is not in the workflow")
    for name, crate in sorted(CRATE_FILTERS.items()):
        if crate not in dirs:
            problems.append(f"CRATE_FILTERS maps `{name}` to {crate}, not a workspace crate")

    shared = [(path, "an input every job reads") for path in SHARED_INPUTS + actions]
    for name, crate in sorted(CRATE_FILTERS.items()):
        if name not in filters or crate not in dirs:
            continue
        required = list(shared)
        required += [(p, "the crate itself") for p in crate_paths(dirs[crate])]
        for dep in sorted(deps[crate]):
            required += [(p, f"{crate} depends on {dep}") for p in crate_paths(dirs[dep])]
        required += [(p, "an input of the job") for p in EXTRA_INPUTS.get(name, [])]
        for path, why in required:
            if not matches(filters[name], path):
                problems.append(
                    f"filter `{name}` (tests {crate}) does not match {path}: {why}"
                )
    if "code" in filters:
        required = list(shared)
        for crate in sorted(dirs):
            required += [(p, f"a path of {crate}") for p in crate_paths(dirs[crate])]
        required += [(p, "an input of the job") for p in EXTRA_INPUTS.get("code", [])]
        for path, why in required:
            if not matches(filters["code"], path):
                problems.append(
                    f"filter `code` (Format, Clippy and Rustdoc) does not match {path}: {why}"
                )
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workflow", type=Path, default=root / WORKFLOW)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    try:
        filters = parse_filters(filters_block(args.workflow.read_text(encoding="utf-8")))
        for patterns in filters.values():
            for pattern in patterns:
                glob_regex(pattern)
    except FilterError as error:
        print(f"{args.workflow}: {error}", file=sys.stderr)
        return 1
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    else:
        output = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        metadata = json.loads(output)
    dirs, deps = workspace(metadata)
    actions = sorted(
        path.relative_to(root).as_posix()
        for path in (root / ".github" / "actions").glob("*/action.y*ml")
    )

    problems = check(filters, dirs, deps, actions)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problems in the filters of {args.workflow}", file=sys.stderr)
        return 1
    print(f"OK: {len(filters)} filters checked against {len(dirs)} crates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
