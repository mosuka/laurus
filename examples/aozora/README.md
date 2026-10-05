# Aozora Bunko Example

Index and search public-domain Japanese literature from [Aozora Bunko](https://www.aozora.gr.jp/)
using morphological analysis (Lindera + IPADIC) and semantic vector search
(Candle BERT embedding). Demonstrates per-field analyzers (a field can be
tokenized morphologically for partial matches while another uses exact
matching), quoted-phrase vs. unquoted-OR query semantics, Unicode-safe
result display, and hybrid (lexical + vector) search.

## Prerequisites

- [python3](https://www.python.org/) for parsing the work list CSV, decoding
  body text (CP932), and stripping ruby/annotation markup
- [curl](https://curl.se/) and [unzip](https://linux.die.net/man/1/unzip) for
  downloading and extracting the Lindera IPADIC dictionary
- Network access on first run (downloads the ~15 MB IPADIC dictionary, the
  ~470 MB text embedding model, and the Aozora Bunko work list; all are
  cached afterward)
- The `embeddings-candle` build feature must be enabled (all scripts in this
  example already pass it)
- No dataset needs to be cloned separately (unlike the `movies` example)

## Schema

The [schema.toml](schema.toml) defines the following fields:

| Field | Type | Analyzer / Embedder | Indexed | Stored | Description |
| ----- | ---- | -------------------- | ------- | ------ | ----------- |
| `title` | Text | `ja_ipadic` | Yes | Yes | 作品名 (work title) |
| `author` | Text | `ja_ipadic` | Yes | Yes | 著者名, morphologically analyzed (e.g. `author:太宰` matches) |
| `author_exact` | Text | `keyword` | Yes | Yes | 著者名, exact match only (e.g. `author_exact:太宰治`) |
| `body` | Text | `ja_ipadic` | Yes | No | 本文 (body text); not stored — see `excerpt` |
| `excerpt` | Text | — | No | Yes | First ~200 characters of the body, for display |
| `ndc` | Text | Standard (default) | Yes | Yes | NDC classification code (e.g. `"913"`) |
| `chars` | Integer | — | Yes | Yes | Character count of the body text |
| `card_url` | Text | — | No | Yes | Aozora Bunko card page URL |
| `title_vec` | Hnsw (384-dim) | `ja_text_embedder` | Yes | No | Semantic vector of the title |
| `body_vec` | Hnsw (384-dim) | `ja_text_embedder` | Yes | No | Semantic vector of the body (see [Embedder](#embedder) for the truncation caveat) |

Default search fields: `title`, `author`, `body`

### Japanese analyzer

`[analyzers.ja_ipadic]` defines a custom pipeline: NFKC normalization →
Japanese iteration-mark expansion (々/ゝ/ゞ) → Lindera morphological
tokenization (IPADIC, `mode = "normal"`) → lowercasing.

This intentionally does **not** use the built-in `{language = "japanese"}`
preset. That preset applies a Japanese stop-word filter, and because the
filter removes tokens without renumbering token positions, a `PhraseQuery`
(which assumes consecutive positions) silently returns zero hits for any
phrase containing a particle — e.g. `title:"銀河鉄道の夜"` would never
match. The custom `ja_ipadic` definition keeps every morpheme, including
particles, so phrase queries behave as expected.

### Per-field analyzers: `author` vs. `author_exact`

The same input value is indexed into two fields with different analyzers,
which is deliberately the clearest demonstration of what a per-field
analyzer buys you:

- `author:太宰` — matches via Lindera (partial match on the surname)
- `author_exact:太宰治` — matches via the keyword analyzer (exact match)
- `author_exact:太宰` — does **not** match (the keyword analyzer never
  splits the value)

### Embedder

`[embedders.ja_text_embedder]` uses
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
(384 dimensions, downloaded once from HuggingFace Hub — ~470 MB — and
cached under `$HF_HOME`, or `~/.cache/huggingface` if unset). This is not a
Japanese-specific model: at the time this schema was written, the
Japanese-specific alternatives (`cl-nagoya/sup-simcse-ja-base`,
`sonoisa/sentence-bert-base-ja-mean-tokens-v2`, `pkshatech/GLuCoSE-base-ja`)
each lacked either `model.safetensors` or `tokenizer.json` on the Hub, which
laurus's `CandleBertEmbedder` requires — this multilingual model was the one
confirmed to actually load and work.

`CandleBertEmbedder` truncates input to the model's `max_seq_length` from
`sentence_bert_config.json` (128 tokens for this model), like
sentence-transformers. A long Aozora Bunko body is therefore safely truncated
rather than causing an error — `body_vec`'s embedding reflects the opening of
the work, not the full text.

`author` is intentionally not vectorized: a person's name is a proper noun,
and semantic similarity search over names is not meaningful. Use the
existing `author` (partial match) / `author_exact` (exact match) lexical
fields instead.

## Sample document

`build_dataset.py` writes one `put docs` record per line (`{"id": ..., "fields": {...}}`)
to `examples/aozora/data/aozora.jsonl`. Two real records from that file, with `body`/`excerpt`/`body_vec`
truncated (`…`) for readability — the actual file has the full, untruncated text on one line each:

```json
{"id": "000919", "fields": {
  "title": "いなか、の、じけん",
  "author": "夢野 久作",
  "author_exact": "夢野久作",
  "body": "いなか、の、じけん　備考\n\n　みんな、私の郷里、北九州の某地方の出来事で、私が見聞致しましたことばかりです。…",
  "excerpt": "いなか、の、じけん　備考　みんな、私の郷里、北九州の某地方の出来事で…",
  "ndc": "913",
  "chars": 169,
  "card_url": "https://www.aozora.gr.jp/cards/000919/card919.html",
  "title_vec": "いなか、の、じけん",
  "body_vec": "いなか、の、じけん　備考\n\n　みんな、私の郷里、北九州の某地方の出来事で、私が見聞致しましたことばかりです。…"
}}
{"id": "001140", "fields": {
  "title": "長崎",
  "author": "芥川 竜之介",
  "author_exact": "芥川竜之介",
  "body": "菱形の凧。サント・モンタニの空に揚つた凧。うらうらと幾つも漂つた凧。\n　路ばたに商ふ夏蜜柑やバナナ。…",
  "excerpt": "菱形の凧。サント・モンタニの空に揚つた凧。うらうらと幾つも漂つた凧。　路ばたに商ふ夏蜜柑やバナナ。…",
  "ndc": "914",
  "chars": 318,
  "card_url": "https://www.aozora.gr.jp/cards/001140/card1140.html",
  "title_vec": "長崎",
  "body_vec": "菱形の凧。サント・モンタニの空に揚つた凧。うらうらと幾つも漂つた凧。\n　路ばたに商ふ夏蜜柑やバナナ。…"
}}
```

`title_vec`/`body_vec` copy the same text as `title`/`body` — the engine embeds it automatically
at index time via the field's configured embedder (see [Embedder](#embedder)).

## Usage

### 1. Create the index

```bash
bash examples/aozora/scripts/create_index.sh
```

This builds the release binary, downloads the Lindera IPADIC dictionary
(cached after the first run — see [scripts/fetch_dict.sh](scripts/fetch_dict.sh)),
renders `schema.toml`'s `@IPADIC_DIR@` placeholder into an absolute path, and
creates an empty index at `examples/aozora/index/`.

### 2. Index works

```bash
bash examples/aozora/scripts/index_aozora.sh
```

Options (all passed through to `build_dataset.py`):

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--limit N` | `1000` | Index only the first N works, ordered by work ID ascending (`0` = all ~17,000 public-domain works) |
| `--ndc CODE` | — | Only works whose NDC code contains `CODE` (e.g. `913` for Japanese novels) |
| `--author NAME` | — | Only works whose author name contains `NAME` |
| `--parallel N` | `4` | Concurrent body-text downloads |
| `--sleep SECONDS` | `0.2` | Delay between downloads, per worker |
| `--refresh-list` | — | Re-download the work list CSV even if cached |
| `--yes` | — | Skip the confirmation delay for `--limit 0` |

```bash
# Quick smoke test
bash examples/aozora/scripts/index_aozora.sh --limit 20

# Only Natsume Sōseki's works
bash examples/aozora/scripts/index_aozora.sh --author 夏目漱石

# All public-domain works (downloads from aozora.gr.jp for every one — see the note below)
bash examples/aozora/scripts/index_aozora.sh --limit 0 --yes
```

This script:

1. Builds the release binary
2. Runs `build_dataset.py`, which downloads the Aozora Bunko work list,
   selects public-domain works with a body-text URL, folds
   author/translator/editor rows into one record per work, downloads and
   decodes (CP932) each work's body text, strips ruby/annotation markup,
   and writes a `put docs` JSONL file
3. Bulk-loads the JSONL file via `laurus put docs`

> Aozora Bunko is run by volunteers. `--limit 0` downloads the body text of
> every public-domain work (tens of thousands of requests) — please use the
> default limit unless you have a specific reason not to.

Aozora Bunko's work IDs are assigned roughly in registration order, and many
well-known works have low IDs (e.g. Akutagawa's 羅生門, Miyazawa's
銀河鉄道の夜, Sōseki's こころ), so the default "first 1,000 works" already
tends to include familiar titles.

### 3. Run example searches

```bash
bash examples/aozora/scripts/search_aozora.sh
```

Runs several example queries:

- 全文検索 (full-text search) — `羅生門`
- Field-specific search — `title:こころ`, `body:蜘蛛の糸`
- Author search contrast — `author:芥川` (partial) vs. `author_exact:芥川竜之介` (exact) vs. `author_exact:芥川` (deliberately zero hits)
- Phrase vs. OR-relaxed search — `title:"銀河鉄道の夜"` (strict) vs. `title:銀河鉄道の夜` (unquoted, OR of morphemes)
- A phrase containing particles — `title:"吾輩は猫である"`
- Natural-sentence search (quoted, since punctuation requires quoting) — `"ある日の暮方の事である"`
- Boolean operators (`AND`, `OR`, `-`), NDC/character-count filters
- Vector search (`title_vec:"人間の孤独と疎外感"`) and hybrid search
  (`title:こころ body_vec:"人間の孤独感"`, and with `+` to require the
  vector clause) — see [Vector and hybrid search](#vector-and-hybrid-search)
- JSON output format

## Vector and hybrid search

`title_vec` and `body_vec` are semantic (Hnsw) fields — see
[Embedder](#embedder). Unlike the lexical fields above, they match by
meaning, not by morpheme: a query can surface works that never contain the
literal query text.

```bash
# Vector search alone
./target/release/laurus --index-dir examples/aozora/index \
  search 'title_vec:"人間の孤独と疎外感"' --limit 5

# Hybrid: lexical OR vector (fused with RRF by default)
./target/release/laurus --index-dir examples/aozora/index \
  search 'title:こころ body_vec:"人間の孤独感"' --limit 5

# Hybrid: require the vector clause to match too
./target/release/laurus --index-dir examples/aozora/index \
  search 'title:こころ +body_vec:"人間の孤独感"' --limit 5
```

See [Query DSL](../../docs/src/concepts/query_dsl.md) and
[Hybrid search](../../docs/src/concepts/search/hybrid_search.md) for the
full syntax, including fusion algorithm details.

## Query tips for Japanese

- Bare (unquoted) terms accept Unicode letters and numbers, but **not**
  punctuation (`。、「」` etc.). Quote any string containing punctuation:
  `"ある日の暮方の事である"` parses; `ある日の暮方の事である。` does not.
- A quoted phrase (`"..."`) requires the exact morpheme sequence
  (`PhraseQuery`). An unquoted term that the analyzer splits into several
  morphemes is OR'd across those morphemes (`BooleanQuery`), so it matches
  more loosely — similar to Lucene's `match` query.
- Field names must be ASCII (`title`, `author`, ...); values may be
  Japanese.

## Manual search

```bash
./target/release/laurus --index-dir examples/aozora/index search 'title:"銀河鉄道の夜"' --limit 10
```

Or start an interactive session:

```bash
./target/release/laurus --index-dir examples/aozora/index repl
```

## gRPC server and MCP server

Instead of the CLI, you can serve this index over gRPC (with an optional HTTP gateway) and
expose it to an MCP client (e.g. Claude Code).

```bash
# gRPC server (+ HTTP gateway on --http-port) over the already-built aozora index.
# The index directory's own schema.toml (including the rendered IPADIC dict path) is
# used automatically — no extra --schema flag exists or is needed.
./target/release/laurus --index-dir examples/aozora/index serve --port 50051 --http-port 8080
```

```bash
# HTTP gateway: plain REST/JSON, no gRPC client needed.
curl http://localhost:8080/v1/index
curl -X POST http://localhost:8080/v1/search -H "Content-Type: application/json" -d '{"query":"title:こころ","limit":3}'
```

In another terminal, the MCP server proxies to that same gRPC endpoint over stdio:

```bash
./target/release/laurus mcp --endpoint http://localhost:50051
```

To register it with Claude Code:

```bash
claude mcp add laurus-aozora -- ./target/release/laurus mcp --endpoint http://localhost:50051
```

`title_vec`/`body_vec` need the binary built with `embeddings-candle` (already the case if you
followed [Usage](#usage) above) — without it, vector/hybrid queries against this index fail at
request time even though the server starts fine.

## Troubleshooting

- **"Failed to resolve analyzer for field 'title'"** — the IPADIC dictionary
  is missing or corrupted. Re-run `bash examples/aozora/scripts/fetch_dict.sh --force`.
- **Want to rebuild the index from scratch?** —
  `rm -rf examples/aozora/index/store examples/aozora/index/schema.toml`,
  then re-run `create_index.sh`.
- **All queries return zero hits** — confirm the `laurus` binary you're
  running was built from this branch; the CLI's `search`/`repl search`
  commands must use the schema's own per-field analyzer (via
  `Engine::unified_query_parser()`), not a hardcoded English analyzer.
- **Downloads are slow or failing** — retry with `--parallel 2 --sleep 0.5`;
  already-downloaded works are cached and are skipped on retry.
- **First run is slow / fails to download the embedding model** — the first
  `create_index.sh`/`index_aozora.sh`/`search_aozora.sh` invocation downloads
  ~470 MB from HuggingFace Hub for `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  It is cached under `$HF_HOME` (or `~/.cache/huggingface`) afterward. If the
  download fails partway, delete the model's cache directory under
  `$HF_HOME/hub` and retry.

## File structure

```text
examples/aozora/
├── README.md
├── README_ja.md
├── schema.toml               # Index schema template (@IPADIC_DIR@ placeholder)
├── scripts/
│   ├── fetch_dict.sh         # Download and extract the Lindera IPADIC dictionary
│   ├── create_index.sh       # Build, fetch the dictionary, render the schema, create the index
│   ├── build_dataset.py      # Fetch work list → filter → download body text → clean → JSONL
│   ├── index_aozora.sh       # Build the dataset and bulk-load it
│   └── search_aozora.sh      # Example search queries
├── dict/                     # Extracted Lindera IPADIC dictionary (git-ignored)
├── data/                     # Cached CSV/zips and the generated JSONL dataset (git-ignored)
└── index/                    # Generated index data (git-ignored)
```
