#!/usr/bin/env python3
"""Document every workspace crate with rustdoc, and fail on any warning.

Each crate is documented on its own with `cargo doc --no-deps -p <crate>`,
as docs.rs documents it, with `-D warnings` added to `RUSTDOCFLAGS`:

* once with its default features, which is what docs.rs builds (no crate
  sets `[package.metadata.docs.rs]`), and once with `--all-features`, which
  also documents what only a feature builds. Each set can report a warning
  the other does not: a link to an item that a feature adds is unresolved
  without the feature, and a module that only a feature builds is not
  documented without it. A crate with no feature besides `default` is
  documented once.
* for the target in `TARGETS` when it has one (`laurus-wasm` builds for
  wasm32 only), and for the host otherwise. The public docs of `laurus`
  also depend on the target architecture, through its SIMD kernel modules,
  so CI runs this script for `laurus` on aarch64 as well as on x86_64.

One `cargo doc --workspace` would not do: the lib of `laurus-python` is
also named `laurus`, and cargo stops with "document output filename
collision"; the binary of `laurus-cli`, also named `laurus`, writes to the
same directory as the `laurus` lib; and one run unifies the features of all
the crates, so a crate would not be documented as it is on its own.

A failed run does not stop the next one, so one run of the script names
every crate with a warning (Issue #1675).

Usage:
  python3 scripts/check-rustdoc.py
  python3 scripts/check-rustdoc.py --package laurus [--package <name> ...]

`--package` documents only the crates named. Exit code 0 when every run
passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# The target of each crate that is not documented for the host.
TARGETS = {
    "laurus-wasm": "wasm32-unknown-unknown",
}


def members(metadata: dict) -> dict[str, set[str]]:
    """The workspace crates, each with the features it declares."""
    ids = set(metadata["workspace_members"])
    return {
        package["name"]: set(package["features"])
        for package in metadata["packages"]
        if package["id"] in ids
    }


def runs(crates: dict[str, set[str]]) -> list[tuple[str, str, list[str]]]:
    """Each `cargo doc` run: the crate, a label, and its arguments."""
    result = []
    for crate in sorted(crates):
        target = ["--target", TARGETS[crate]] if crate in TARGETS else []
        result.append((crate, "default features", target))
        if crates[crate] - {"default"}:
            result.append((crate, "all features", ["--all-features", *target]))
    return result


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="NAME",
        help="document only this crate (repeatable)",
    )
    args = parser.parse_args()

    output = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    crates = members(json.loads(output))
    unknown = sorted(set(args.package) - set(crates))
    if unknown:
        print(f"not a workspace crate: {', '.join(unknown)}", file=sys.stderr)
        return 1
    if args.package:
        crates = {name: crates[name] for name in args.package}

    env = dict(os.environ)
    env["RUSTDOCFLAGS"] = (env.get("RUSTDOCFLAGS", "") + " -D warnings").strip()

    failed = []
    planned = runs(crates)
    for crate, label, extra in planned:
        command = ["cargo", "doc", "--no-deps", "-p", crate, *extra]
        print(f"--- {crate} ({label}): {' '.join(command)}", flush=True)
        if subprocess.run(command, cwd=root, env=env).returncode == 0:
            print(f"ok {crate} ({label})", flush=True)
        else:
            print(f"FAILED {crate} ({label})", flush=True)
            failed.append(f"{crate} ({label})")

    if failed:
        print(f"{len(failed)} of {len(planned)} runs failed:", file=sys.stderr)
        for run in failed:
            print(f"  {run}", file=sys.stderr)
        return 1
    print(f"OK: {len(planned)} runs, no rustdoc warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
