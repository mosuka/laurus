#!/usr/bin/env python3
"""バインディングのスキーマ定義スニペットにおける位置引数のズレを検出する。

Node.js / WASM / PHP の `addTextField` は `docValues` を `analyzer` の**前**に
挿入した（Issue #1047）ため、README・docs・rustdoc・examples の呼び出しが
analyzer 名を 5 番目（`docValues`）に渡したまま残っていた（Issue #1188）。
`addHnswField` も `defaultEfSearch` が `embedder` の前に挿入された同じ形をしている。

このスクリプトは `git ls-files` が返すスニペットを含みうるファイルを走査し、
1 行に収まる `addTextField(...)` / `addHnswField(...)` の呼び出しについて

* `addTextField` の 5 番目の引数が文字列リテラル、または
  `true` / `false` / `null` / `undefined`（および引数名そのものの
  `docValues` / `doc_values`）以外の識別子である
* `addHnswField` の 6 番目の引数が文字列リテラルである
* `setVectorQuery` の 1 番目の引数が文字列リテラルである（旧 2 引数形
  `setVectorQuery(field, vector)`。現行は `setVectorQuery(new VectorQuery(field, vector))`）
* 旧 `SearchRequest` API の呼び出しが残っている: 削除済みメソッド
  `setFilterQuery` / `setLexicalTermQuery`、数値を直接渡す `setRrfFusion(k)` /
  `setWeightedSumFusion(a, b)`（現行は `RRF` / `WeightedSum` オブジェクト）、
  `new SearchRequest(limit)`（現行は `new SearchRequest({ limit })`）

場合を違反として報告し、終了コード 1 を返す。

制限: 複数行にまたがる呼び出しは検査しない（現状リポジトリに存在しない）。
Python / Ruby はキーワード引数なので対象外。docs/book/（生成物）は除外する。

使い方: python3 scripts/check-binding-snippets.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXTENSIONS = {".md", ".mjs", ".js", ".cjs", ".ts", ".php", ".rs", ".html"}
EXCLUDED_PREFIXES = ("docs/book/",)

# 5 番目（docValues）に置いてよい識別子。真偽値リテラルと、シグネチャ表記で
# 使われる引数名そのもの。
ALLOWED_DOC_VALUES_IDENTS = {
    "true",
    "false",
    "null",
    "undefined",
    "docValues",
    "doc_values",
    "$docValues",
}

CALL_RE = re.compile(
    r"\b(addTextField|addHnswField|setVectorQuery|setVectorTextQuery"
    r"|setFilterQuery|setLexicalTermQuery|setLexicalPhraseQuery"
    r"|setRrfFusion|setWeightedSumFusion|SearchRequest)\s*\("
)
IDENT_RE = re.compile(r"^\$?[A-Za-z_][A-Za-z0-9_.]*$")
NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# 現行 API から削除されたメソッド → 置き換え先。
REMOVED_METHODS = {
    "setFilterQuery": "setFilterTerm(new TermQuery(field, term))",
    "setLexicalTermQuery": "setLexicalTerm(new TermQuery(field, term))",
    "setLexicalPhraseQuery": "setLexicalPhrase(new PhraseQuery(field, terms))",
}


def split_arguments(text: str, open_paren: int) -> list[str] | None:
    """`text[open_paren]` が `(` である呼び出しの引数を、クォートと括弧の
    入れ子を尊重してトップレベルのカンマで分割する。同じ行で閉じなければ None。"""
    depth = 0
    quote: str | None = None
    args: list[str] = []
    current: list[str] = []
    i = open_paren
    while i < len(text):
        c = text[i]
        if quote is not None:
            current.append(c)
            if c == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
            current.append(c)
        elif c in "([{":
            depth += 1
            if depth > 1:
                current.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                return args
            current.append(c)
        elif c == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    return None


def is_string_literal(arg: str) -> bool:
    return len(arg) >= 2 and arg[0] in "\"'`" and arg[-1] == arg[0]


def violation_for(method: str, args: list[str]) -> str | None:
    if method == "addTextField":
        if len(args) < 5:
            return None
        fifth = args[4]
        if is_string_literal(fifth):
            return "analyzer string in the docValues slot (5th argument of addTextField)"
        if IDENT_RE.match(fifth) and fifth not in ALLOWED_DOC_VALUES_IDENTS:
            return (
                f"identifier `{fifth}` in the docValues slot (5th argument of "
                "addTextField); pass a boolean literal or move the analyzer to the 6th slot"
            )
        return None
    if method == "addHnswField":
        if len(args) < 6:
            return None
        sixth = args[5]
        if is_string_literal(sixth):
            return "string in the defaultEfSearch slot (6th argument of addHnswField); the embedder is the 7th"
        return None
    if method in ("setVectorQuery", "setVectorTextQuery"):
        if len(args) >= 1 and is_string_literal(args[0]):
            wrapper = "VectorQuery" if method == "setVectorQuery" else "VectorTextQuery"
            return (
                f"stale two-argument {method}(field, ...); "
                f"use {method}(new {wrapper}(field, ...))"
            )
        return None
    if method in REMOVED_METHODS:
        return f"removed method {method}(); use {REMOVED_METHODS[method]}"
    if method in ("setRrfFusion", "setWeightedSumFusion"):
        wrapper = "RRF" if method == "setRrfFusion" else "WeightedSum"
        if not args or args[0] == "":
            return f"stale zero-argument {method}(); pass a `new {wrapper}(...)` object"
        if NUMBER_RE.match(args[0]):
            return f"stale numeric {method}(...); pass a `new {wrapper}(...)` object"
        return None
    if method == "SearchRequest":
        if args and NUMBER_RE.match(args[0]):
            return "stale new SearchRequest(limit[, offset]); use new SearchRequest({ limit, offset })"
        return None
    return None


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout
    files = []
    for name in out.split("\0"):
        if not name or name.startswith(EXCLUDED_PREFIXES):
            continue
        path = Path(name)
        if path.suffix in EXTENSIONS:
            files.append(path)
    return files


def main() -> int:
    violations: list[tuple[Path, int, str, str]] = []
    call_sites = 0
    files = tracked_files()
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as err:
            print(f"warning: cannot read {path}: {err}", file=sys.stderr)
            continue
        for lineno, line in enumerate(lines, start=1):
            for match in CALL_RE.finditer(line):
                args = split_arguments(line, match.end() - 1)
                if args is None:
                    continue
                call_sites += 1
                reason = violation_for(match.group(1), args)
                if reason:
                    violations.append((path, lineno, reason, line.strip()))

    for path, lineno, reason, line in violations:
        print(f"{path}:{lineno}: {reason}")
        print(f"    {line}")
    if violations:
        print(
            f"\n{len(violations)} violation(s) in "
            f"{len({v[0] for v in violations})} file(s) "
            f"({call_sites} call sites checked in {len(files)} files)"
        )
        return 1
    print(f"OK: {call_sites} call sites checked in {len(files)} files, no violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
