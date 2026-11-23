#!/usr/bin/env python3
"""Tests of scripts/check-ci-filters.py (Issue #1714).

Usage: python3 scripts/test_check_ci_filters.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check-ci-filters.py"
_spec = importlib.util.spec_from_file_location("check_ci_filters", SCRIPT)
ccf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccf)

SHARED = [
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/**",
    ".github/actions/**",
    ".github/workflows/regression.yml",
]
ACTIONS = [".github/actions/setup/action.yml"]

# Each crate's directory is its name; the dependencies mirror the workspace.
DIRS = {crate: crate for crate in set(ccf.CRATE_FILTERS.values())}
DEPS = {crate: {"laurus"} for crate in DIRS}
DEPS["laurus"] = set()
DEPS["laurus-mcp"] = {"laurus", "laurus-server"}
DEPS["laurus-cli"] = {"laurus", "laurus-server", "laurus-mcp"}


def complete_filters() -> dict[str, list[str]]:
    """Filters that match everything the check requires."""
    filters = {"shared": list(SHARED), "snippets": ["**/*.md"]}
    filters["code"] = SHARED + ["**/*.rs", "**/Cargo.toml"] + ccf.EXTRA_INPUTS.get("code", [])
    for name, crate in ccf.CRATE_FILTERS.items():
        dirs = sorted({crate} | DEPS[crate])
        filters[name] = SHARED + [f"{d}/**" for d in dirs] + ccf.EXTRA_INPUTS.get(name, [])
    return filters


def block(text: str) -> list[tuple[int, str]]:
    return ccf.filters_block(text)


WORKFLOW = """\
jobs:
  changes:
    steps:
      - uses: dorny/paths-filter@v4
        with:
          # a comment outside the block
          filters: |
            # shared inputs
            base: &base
              - 'Cargo.toml'
              - "Cargo.lock"

            core: &core
              - *base
              - core/**   # a plain pattern with a comment
            top:
              - *core
              - 'top/**'
      - name: next step
        run: echo done
"""


class FiltersBlockTest(unittest.TestCase):
    def test_reads_the_block_up_to_the_first_line_indented_no_deeper(self):
        lines = [line for _, line in block(WORKFLOW)]
        self.assertEqual(lines[0].strip(), "# shared inputs")
        self.assertEqual(lines[-1].strip(), "- 'top/**'")
        self.assertNotIn("next step", "".join(lines))

    def test_numbers_lines_from_one(self):
        number, line = block(WORKFLOW)[0]
        self.assertEqual(WORKFLOW.split("\n")[number - 1], line)

    def test_requires_exactly_one_block(self):
        with self.assertRaisesRegex(ccf.FilterError, "found 0"):
            ccf.filters_block("jobs: {}\n")
        with self.assertRaisesRegex(ccf.FilterError, "found 2"):
            ccf.filters_block("filters: |\n  a:\n    - x\nfilters: |\n  b:\n    - y\n")


class ParseFiltersTest(unittest.TestCase):
    def test_aliases_stand_for_the_anchored_patterns_nested(self):
        filters = ccf.parse_filters(block(WORKFLOW))
        self.assertEqual(filters["base"], ["Cargo.toml", "Cargo.lock"])
        self.assertEqual(filters["core"], ["Cargo.toml", "Cargo.lock", "core/**"])
        self.assertEqual(filters["top"], ["Cargo.toml", "Cargo.lock", "core/**", "top/**"])

    def parse(self, body: str) -> dict[str, list[str]]:
        return ccf.parse_filters(block("filters: |\n" + body))

    def test_fails_on_an_unsupported_line(self):
        for body in [
            "  a:\n    nested:\n      - x\n",
            "  a:\n    - added|modified: '**'\n    - [x, y]\n",
            "  a: [x, y]\n",
            "  - x\n",
            "  a:\n    - &anchor x\n",
        ]:
            with self.subTest(body=body):
                with self.assertRaisesRegex(ccf.FilterError, "unsupported filter syntax"):
                    self.parse(body)

    def test_fails_on_an_alias_with_no_earlier_anchor(self):
        with self.assertRaisesRegex(ccf.FilterError, "names no earlier anchor"):
            self.parse("  a:\n    - *later\n  b: &later\n    - x\n")
        with self.assertRaisesRegex(ccf.FilterError, "names no earlier anchor"):
            self.parse("  a: &self\n    - *self\n")

    def test_fails_on_a_filter_defined_twice_or_with_no_patterns(self):
        with self.assertRaisesRegex(ccf.FilterError, "defined twice"):
            self.parse("  a:\n    - x\n  a:\n    - y\n")
        with self.assertRaisesRegex(ccf.FilterError, "has no patterns"):
            self.parse("  a:\n  b:\n    - x\n")


class GlobTest(unittest.TestCase):
    # Verdicts of picomatch 2.3.1 with {dot: true}, as dorny/paths-filter@v4
    # calls it.
    CASES = [
        ("laurus/**", "laurus", True),
        ("laurus/**", "laurus/Cargo.toml", True),
        ("laurus/**", "laurus/src/a/b.rs", True),
        ("laurus/**", "laurus-server/Cargo.toml", False),
        ("laurus/**", "laurusx/a", False),
        ("**/*.rs", "a.rs", True),
        ("**/*.rs", "src/a.rs", True),
        ("**/*.rs", "a/b/c.rs", True),
        ("**/*.rs", "a.rsx", False),
        ("**/Cargo.toml", "Cargo.toml", True),
        ("**/Cargo.toml", "laurus/Cargo.toml", True),
        (".cargo/**", ".cargo/config.toml", True),
        (".github/actions/**", ".github/actions/install-protoc/action.yml", True),
        ("**", ".github/x.yml", True),
        ("**", "a", True),
        ("*.md", "a/b.md", False),
        ("*.md", "b.md", True),
        ("*", ".hidden", True),
        ("a/*/c", "a/b/c", True),
        ("a/*/c", "a/b/x/c", False),
        ("a/**/c", "a/c", True),
        ("a/**/c", "a/b/x/c", True),
        ("Cargo.lock", "Cargo.lock", True),
        ("Cargo.lock", "x/Cargo.lock", False),
    ]

    def test_matches_as_picomatch_does(self):
        wrong = [
            (pattern, path, expected)
            for pattern, path, expected in self.CASES
            if bool(ccf.glob_regex(pattern).fullmatch(path)) != expected
        ]
        self.assertEqual(wrong, [])

    def test_fails_on_an_unsupported_pattern(self):
        for pattern in ["a/?.rs", "{a,b}/**", "[ab]/x", "!a/**", "a/**/**", "a**", "/a", "a//b", "a b"]:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ccf.FilterError, "unsupported pattern"):
                    ccf.glob_regex(pattern)


class WorkspaceTest(unittest.TestCase):
    def metadata(self) -> dict:
        def package(name, deps):
            return {
                "name": name,
                "id": f"{name}-id",
                "manifest_path": f"/repo/{name}/Cargo.toml",
                "dependencies": deps,
            }

        def path_dep(name, kind=None):
            return {"name": name, "kind": kind, "path": f"/repo/{name}"}

        return {
            "workspace_root": "/repo",
            "workspace_members": ["a-id", "b-id", "c-id", "d-id"],
            "packages": [
                package("a", [path_dep("b"), path_dep("d", "dev"), {"name": "serde", "kind": None}]),
                package("b", [path_dep("c", "build")]),
                package("c", []),
                package("d", []),
                package("outside", []),
            ],
        }

    def test_dependencies_are_closed_transitively_with_every_kind(self):
        dirs, deps = ccf.workspace(self.metadata())
        self.assertEqual(dirs, {"a": "a", "b": "b", "c": "c", "d": "d"})
        self.assertEqual(deps["a"], {"b", "c", "d"})
        self.assertEqual(deps["b"], {"c"})
        self.assertEqual(deps["c"], set())


class CheckTest(unittest.TestCase):
    def check(self, filters):
        return ccf.check(filters, DIRS, DEPS, ACTIONS)

    def test_complete_filters_pass(self):
        self.assertEqual(self.check(complete_filters()), [])

    def test_a_missing_dependency_is_named(self):
        filters = complete_filters()
        filters["mcp"].remove("laurus-server/**")
        self.assertEqual(
            self.check(filters),
            [
                "filter `mcp` (tests laurus-mcp) does not match laurus-server/Cargo.toml:"
                " laurus-mcp depends on laurus-server",
                "filter `mcp` (tests laurus-mcp) does not match laurus-server/src/probe.rs:"
                " laurus-mcp depends on laurus-server",
            ],
        )

    def test_a_missing_shared_input_is_named_for_every_filter(self):
        filters = complete_filters()
        for name in filters:
            if ".cargo/**" in filters[name]:
                filters[name].remove(".cargo/**")
        problems = self.check(filters)
        self.assertEqual(len(problems), len(ccf.CRATE_FILTERS) + 1)
        self.assertTrue(all(".cargo/config.toml: an input every job reads" in p for p in problems))

    def test_a_missing_composite_action_is_named(self):
        filters = complete_filters()
        filters["wasm"].remove(".github/actions/**")
        self.assertEqual(
            self.check(filters),
            [
                "filter `wasm` (tests laurus-wasm) does not match"
                " .github/actions/setup/action.yml: an input every job reads"
            ],
        )

    def test_an_extra_input_and_the_crate_itself_are_required(self):
        filters = complete_filters()
        filters["interop"].remove(".github/workflows/index-interop.yml")
        filters["php"].remove("laurus-php/**")
        problems = self.check(filters)
        self.assertIn(
            "filter `interop` (tests laurus-cli) does not match"
            " .github/workflows/index-interop.yml: an input of the job",
            problems,
        )
        self.assertIn(
            "filter `php` (tests laurus-php) does not match laurus-php/Cargo.toml:"
            " the crate itself",
            problems,
        )

    def test_code_must_match_every_crate(self):
        filters = complete_filters()
        filters["code"] = SHARED + ["laurus/**"]
        problems = self.check(filters)
        self.assertIn(
            "filter `code` (Format, Clippy and Rustdoc) does not match"
            " laurus-ruby/Cargo.toml: a path of laurus-ruby",
            problems,
        )
        self.assertFalse(any("laurus/Cargo.toml" in p for p in problems))

    def test_code_must_match_the_rustdoc_script(self):
        filters = complete_filters()
        filters["code"].remove("scripts/check-rustdoc.py")
        self.assertEqual(
            self.check(filters),
            [
                "filter `code` (Format, Clippy and Rustdoc) does not match"
                " scripts/check-rustdoc.py: an input of the job"
            ],
        )

    def test_the_table_has_to_cover_every_crate_and_filter(self):
        filters = complete_filters()
        filters["docs"] = ["docs/**"]
        del filters["snippets"]
        dirs = dict(DIRS, **{"laurus-new": "laurus-new"})
        deps = dict(DEPS, **{"laurus-new": {"laurus"}})
        problems = ccf.check(filters, dirs, deps, ACTIONS)
        self.assertEqual(
            problems,
            [
                "crate laurus-new is not in CRATE_FILTERS: which filter tests it?",
                "filter `docs` is in neither CRATE_FILTERS nor OTHER_FILTERS",
                "filter `snippets` is not in the workflow",
            ],
        )

    def test_a_table_entry_for_no_crate_is_named(self):
        dirs = {k: v for k, v in DIRS.items() if k != "laurus-wasm"}
        problems = ccf.check(complete_filters(), dirs, DEPS, ACTIONS)
        self.assertEqual(
            problems, ["CRATE_FILTERS maps `wasm` to laurus-wasm, not a workspace crate"]
        )


class MainTest(unittest.TestCase):
    """The script as CI runs it, on a workflow and metadata in files."""

    def run_script(self, filters: dict[str, list[str]]) -> subprocess.CompletedProcess:
        lines = ["jobs:", "  changes:", "    steps:", "      - with:", "          filters: |"]
        for name, patterns in filters.items():
            lines.append(f"            {name}:")
            lines += [f"              - '{p}'" for p in patterns]
        members = sorted(DIRS)
        metadata = {
            "workspace_root": "/repo",
            "workspace_members": [f"{c}-id" for c in members],
            "packages": [
                {
                    "name": c,
                    "id": f"{c}-id",
                    "manifest_path": f"/repo/{c}/Cargo.toml",
                    "dependencies": [{"name": d, "path": f"/repo/{d}"} for d in sorted(DEPS[c])],
                }
                for c in members
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            workflow = Path(tmp) / "regression.yml"
            workflow.write_text("\n".join(lines) + "\n", encoding="utf-8")
            meta = Path(tmp) / "metadata.json"
            meta.write_text(json.dumps(metadata), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--workflow", str(workflow), "--metadata", str(meta)],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_passes_with_exit_code_0(self):
        result = self.run_script(complete_filters())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK: 13 filters checked against 9 crates", result.stdout)

    def test_fails_with_exit_code_1_naming_the_gap(self):
        filters = complete_filters()
        filters["cli"].remove("laurus-server/**")
        result = self.run_script(filters)
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "filter `cli` (tests laurus-cli) does not match laurus-server/Cargo.toml:"
            " laurus-cli depends on laurus-server",
            result.stderr,
        )

    def test_fails_on_an_unsupported_pattern(self):
        filters = complete_filters()
        filters["snippets"] = ["{a,b}/**"]
        result = self.run_script(filters)
        self.assertEqual(result.returncode, 1)
        self.assertIn("unsupported pattern: {a,b}/**", result.stderr)


if __name__ == "__main__":
    unittest.main()
