# コマンドリファレンス

## グローバルオプション

すべてのコマンドで以下のオプションが使用できます:

| オプション | 環境変数 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `--index-dir <PATH>` | `LAURUS_INDEX_DIR` | `./laurus_index` | インデックスデータディレクトリのパス |
| `--format <FORMAT>` | — | `table` | 出力形式: `table` または `json` |

```bash
# 例: カスタムデータディレクトリで JSON 出力を使用
laurus --index-dir /var/data/my_index --format json search "title:rust"
```

---

## `create` — リソースの作成

### `create index`

新しいインデックスを作成します。`--schema` が指定された場合はその TOML ファイルを使用し、省略された場合は対話型スキーマウィザードが起動します。

```bash
laurus create index [--schema <FILE>] [--train-pq-codebook <JSONL>]
```

**引数:**

| フラグ | 必須 | 説明 |
| :--- | :--- | :--- |
| `--schema <FILE>` | いいえ | インデックススキーマを定義する TOML ファイルのパス。省略時はインデックスディレクトリに既存の `schema.toml` があればそれを使用し、なければ対話型ウィザードが起動します。 |
| `--train-pq-codebook <JSONL>` | いいえ | インデックス作成の一部として共有 PQ codebook を学習します（Issue #920）。`ProductQuantization`（または `pq-fastscan` feature 有効時は `ProductQuantizationFastScan`）+ `pq_codebook_path` を設定したすべての HNSW フィールドを、作成直後にこの JSONL ファイル（`put docs` / `add docs` と同じ形式、フィールド値は素の数値配列）から学習します。最初の commit がすぐに codebook でエンコードできるため、create → `train pq-codebook` → ingest の順序を手動で守る必要がなくなります。ファイル不在または対象フィールドなしの場合は、何も作成する前にエラーになります。 |

**スキーマファイルの形式:**

スキーマファイルは Laurus ライブラリの `Schema` 型と同じ構造に従います。詳細は[スキーマフォーマットリファレンス](schema_format.md)を参照してください。例:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
stored = true
indexed = true

[fields.body.Text]
stored = true
indexed = true

[fields.category.Text]
stored = true
indexed = true
```

**例:**

```bash
# スキーマファイルから作成
laurus --index-dir ./my_index create index --schema schema.toml
# Index created at ./my_index.

# 対話型ウィザード（--schema フラグなし）
laurus --index-dir ./my_index create index
# === Laurus Schema Generator ===
# Field name: title
# ...
# Index created at ./my_index.

# 作成と共有 PQ codebook の学習を1ステップで（Issue #920）
laurus --index-dir ./my_index create index --schema schema.toml \
    --train-pq-codebook train.jsonl
# Index created at ./my_index.
# Training PQ codebook for field 'embedding' on 300 vectors...
# Trained codebook 'embedding.pqcb' (m = 4, k = 256, sub_dim = 8, dimension = 32) from 300 vectors.
```

> **注意:** `schema.toml` と `store/` の両方が存在する場合はエラーが返されます。再作成するにはインデックスディレクトリを削除してください。`schema.toml` のみ存在する場合（作成が中断された場合など）は、`--schema` なしで `create index` を実行すると既存スキーマからストレージが復旧されます。

### `create schema`

対話式ウィザードを通じてスキーマ TOML ファイルを生成します。

```bash
laurus create schema [--output <FILE>]
```

**引数:**

| フラグ | 必須 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `--output <FILE>` | いいえ | `schema.toml` | 生成されるスキーマの出力ファイルパス |

ウィザードは以下の手順で進みます:

1. **フィールド定義** — フィールド名を入力し、型を選択し、型固有のオプションを設定
2. **繰り返し** — 必要な数だけフィールドを追加
3. **デフォルトフィールド** — デフォルトの検索対象とする Lexical フィールドを選択
4. **プレビュー** — 保存前に生成された TOML を確認
5. **保存** — スキーマファイルを書き出し

**サポートされるフィールド型:**

| 型 | カテゴリ | オプション |
| :--- | :--- | :--- |
| `Text` | Lexical | `indexed`, `stored`, `term_vectors` |
| `Integer` | Lexical | `indexed`, `stored` |
| `Float` | Lexical | `indexed`, `stored` |
| `Boolean` | Lexical | `indexed`, `stored` |
| `DateTime` | Lexical | `indexed`, `stored` |
| `Geo` | Lexical | `indexed`, `stored` |
| `Geo3d` | Lexical | `indexed`, `stored` |
| `Bytes` | Lexical | `stored` |
| `Hnsw` | Vector | `dimension`, `distance`, `m`, `ef_construction` |
| `Flat` | Vector | `dimension`, `distance` |
| `Ivf` | Vector | `dimension`, `distance`, `n_clusters`, `n_probe` |

**例:**

```bash
# schema.toml を対話的に生成
laurus create schema

# 出力パスを指定
laurus create schema --output my_schema.toml

# 生成されたスキーマからインデックスを作成
laurus create index --schema schema.toml
```

---

## `get` — リソースの取得

### `get stats`

インデックスの統計情報を表示します。

```bash
laurus get stats
```

**テーブル出力の例:**

```text
Document count: 42

Vector fields:
╭──────────┬─────────┬───────────╮
│ Field    │ Vectors │ Dimension │
├──────────┼─────────┼───────────┤
│ text_vec │ 42      │ 384       │
╰──────────┴─────────┴───────────╯
```

**JSON 出力の例:**

```bash
laurus --format json get stats
```

```json
{
  "document_count": 42,
  "fields": {
    "text_vec": {
      "vector_count": 42,
      "dimension": 384
    }
  }
}
```

### `get schema`

現在のインデックスのスキーマを JSON 形式で表示します。

```bash
laurus get schema
```

**例:**

```bash
laurus get schema
# {
#   "fields": { ... },
#   "default_fields": ["title", "body"],
#   ...
# }
```

### `get docs`

外部 ID で全ドキュメント（チャンクを含む）を取得します。

```bash
laurus get docs --id <ID>
```

**テーブル出力の例:**

```text
╭──────┬─────────────────────────────────────────╮
│ ID   │ Fields                                  │
├──────┼─────────────────────────────────────────┤
│ doc1 │ body: This is a test, title: Hello World │
╰──────┴─────────────────────────────────────────╯
```

**JSON 出力の例:**

```bash
laurus --format json get docs --id doc1
```

```json
[
  {
    "id": "doc1",
    "document": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }
]
```

---

## `add` — リソースの追加

### `add doc`

インデックスにドキュメントを追加します。ドキュメントは `commit` を実行するまで検索対象になりません。

```bash
laurus add doc --id <ID> --data <JSON>
```

**引数:**

| フラグ | 必須 | 説明 |
| :--- | :--- | :--- |
| `--id <ID>` | はい | 外部ドキュメント ID（文字列） |
| `--data <JSON>` | はい | JSON 文字列としてのドキュメントフィールド |

JSON は `{"fields": {...}}` 形式です: 各フィールド名にその素の値を対応付けます。型タグはありません
— 値の宣言済みスキーマ型（未宣言フィールドの場合は推論された型）が曖昧さを解決します。

```json
{
  "fields": {
    "title": "Introduction to Rust",
    "body": "Rust is a systems programming language.",
    "year": 2024
  }
}
```

**例:**

```bash
laurus add doc --id doc1 --data '{"fields":{"title":"Hello World","body":"This is a test document."}}'
# Document 'doc1' added. Run 'commit' to persist changes.
```

> **ヒント:** 複数のドキュメントが同じ外部 ID を共有できます（チャンキングパターン）。各チャンクに対して `add doc` を使用してください。

### `add docs`

JSONL ファイルからドキュメントチャンクをバルク追加します — 1 行に 1 エントリの `{"id": "...", "fields": {...}}` 形式で、外部 ID は `add doc --data` と同じ `fields` 形式に並ぶトップレベルキーです。エントリはエンジンのバッチ API（バッチごとに WAL fsync 1 回）で適用され、`add doc` と異なり**自動的にコミット**します（`--commit-every` 件ごと + 最後に 1 回）。

```bash
laurus add docs --file <JSONL> [--batch-size 1000] [--commit-every 0]
```

**引数:**

| フラグ | 必須 | 説明 |
| :--- | :--- | :--- |
| `--file <JSONL>` | はい | 取り込む JSONL ファイルのパス |
| `--batch-size <N>` | いいえ | エンジンのバッチ呼び出しあたりのドキュメント数（既定 `1000`） |
| `--commit-every <N>` | いいえ | N 件適用ごとにコミット。`0` = 最後の 1 回のみ（既定） |

繰り返した ID はチャンクとして蓄積されます。途中で失敗した場合、エラーは該当行を示し、適用済みの prefix はコミットされるため、残りの行から再実行してインジェストを継続できます。

---

## `put` — リソースの上書き（Upsert）

### `put doc`

インデックスにドキュメントを上書き（upsert）します。同じ ID のドキュメントが既に存在する場合、全チャンクが削除されてから新しいドキュメントがインデックスされます。ドキュメントは `commit` を実行するまで検索対象になりません。

```bash
laurus put doc --id <ID> --data <JSON>
```

**引数:**

| フラグ | 必須 | 説明 |
| :--- | :--- | :--- |
| `--id <ID>` | はい | 外部ドキュメント ID（文字列） |
| `--data <JSON>` | はい | JSON 文字列としてのドキュメントフィールド |

**例:**

```bash
laurus put doc --id doc1 --data '{"fields":{"title":"Updated Title","body":"This replaces the existing document."}}'
# Document 'doc1' put (upserted). Run 'commit' to persist changes.
```

> **注意:** `add doc` とは異なり、`put doc` は指定 ID の既存チャンクをすべて置き換えます。チャンクを追記したい場合は `add doc` を、ドキュメント全体を置き換えたい場合は `put doc` を使用してください。

### `put docs`

JSONL ファイルからドキュメントをバルク Upsert します — 1 行に 1 エントリの `{"id": "...", "fields": {...}}` 形式で、エンジンのバッチ API（バッチごとに WAL fsync 1 回）で適用されます。重複した ID は順にデデュープされます（最後の出現が勝ち）。`add docs` と同じく**自動的にコミット**します。

```bash
laurus put docs --file <JSONL> [--batch-size 1000] [--commit-every 0]
```

引数は `add docs` と同じです。途中で失敗した場合、エラーは該当行を示し、適用済みの prefix はコミットされます。put は冪等なので、ファイル全体（または残りの suffix）の再実行は安全です。

**例:**

```bash
cat > docs.jsonl <<'JSONL'
{"id": "doc1", "fields": {"title": "Hello"}}
{"id": "doc2", "fields": {"title": "World"}}
JSONL
laurus put docs --file docs.jsonl
# 2 documents put (upserted) and committed.
```

---

### `add field`

既存のインデックスにフィールドを動的に追加します。

```bash
laurus add field --index-dir ./data \
    --name category \
    --field-option '{"Text": {"indexed": true, "stored": true}}'
```

`--field-option` 引数はスキーマファイルと同じ外部タグ付き JSON 形式を受け付けます。
フィールド追加後、スキーマは自動的に永続化されます。

---

## `delete` — リソースの削除

### `delete field`

スキーマからフィールドを動的に削除します。既にインデックスされたデータは残りますが、削除されたフィールドにはアクセスできなくなります。

```bash
laurus delete field --name <FIELD_NAME>
```

**例:**

```bash
laurus delete field --name category
# Field 'category' deleted.
```

### `delete docs`

外部 ID で全ドキュメント（チャンクを含む）を削除します。

```bash
laurus delete docs --id <ID>
```

**例:**

```bash
laurus delete docs --id doc1
# Documents 'doc1' deleted. Run 'commit' to persist changes.
```

---

## `commit`

保留中の変更（追加と削除）をインデックスにコミットします。コミットするまで、変更は検索に反映されません。

```bash
laurus commit
```

**例:**

```bash
laurus --index-dir ./my_index commit
# Changes committed successfully.
```

---

## `train`

### `train pq-codebook`

HNSW ベクトルフィールド用の**共有 PQ codebook** を学習します
（Issue #631）。codebook を代表サンプルで一度だけ学習し、以後の
commit と merge のすべてで再利用します — segment ごとの k-means
再学習が無くなるため PQ フィールドの commit は大幅に高速化し、
小さな per-commit segment も PQ を維持します。

```bash
laurus train pq-codebook --field <FIELD> (--input <JSONL> | --from-index) \
    [--sample-size <N>] [--output <NAME>] [--update-schema]
```

| 引数 | 説明 |
| :--- | :--- |
| `--field` | 学習対象の HNSW ベクトルフィールド。`ProductQuantization` quantizer（または `pq-fastscan` feature 有効時は `ProductQuantizationFastScan` — その場合 codebook は k=16 で学習されます、Issue #920）が設定されている必要があります。 |
| `--input` | JSONL 学習ファイル — `put docs` / `add docs` と同じ `{"id": "...", "fields": {...}}` 形式。フィールド値は素の数値配列（例: `"embedding": [0.1, 0.2, ...]`）である必要があります（embedder 生成の入力は未対応）。`--input` と `--from-index` はどちらか一方のみ指定できます。 |
| `--from-index` | ファイルの代わりに、このインデックスにコミット済みのベクトルを直接サンプリングします（Issue #920）— JSONL エクスポート不要。`--input` と `--from-index` はどちらか一方のみ指定できます。注意: 既に PQ エンコード済みのフィールドではサンプルは有損の再構成ベクトルになります。想定フローはフィールドの PQ 有効化**前**にコミットしたベクトルからの学習です。 |
| `--sample-size` | 先頭 N 件のみを使用（決定的: `--input` はファイル順、`--from-index` は doc_id 昇順）。省略時は全件を使用。代表的なベクトル数千件で十分です。 |
| `--output` | ストレージ相対の codebook ファイル名。デフォルトはフィールドの `pq_codebook_path`、未設定なら `{field}.pqcb`。稼働中の codebook の横に v2 を学習する場合に使用。 |
| `--update-schema` | フィールドの `pq_codebook_path` が学習済みファイルを指すよう `schema.toml` を書き換えます。 |

commit が codebook を使うのは、スキーマの `pq_codebook_path` が
そのファイルを指している場合のみです（[スキーマ形式](schema_format.md#product-quantization-hnsw-のみ)
参照）— 学習と同時に設定するには `--update-schema` を渡してください。
`pq_codebook_path` が設定済みで codebook が未学習のまま commit
すると、本コマンドを示すエラーで失敗します（per-segment 学習への
無言のフォールバックはありません）。codebook は index open 時に
読み込まれるため、ingest する `add` / `put` / `commit` の**前に**
学習してください（CLI は呼び出しごとに index を開き直すため、
以後のコマンドはすべて反映済みです）。

**例:**

```bash
cat > train.jsonl <<'JSONL'
{"id": "t1", "fields": {"embedding": [0.1, 0.2, 0.3, 0.4]}}
{"id": "t2", "fields": {"embedding": [0.5, 0.6, 0.7, 0.8]}}
JSONL
laurus train pq-codebook --field embedding --input train.jsonl --update-schema
# Training PQ codebook for field 'embedding' on 2 vectors...
# Trained codebook 'embedding.pqcb' (m = 2, k = 256, sub_dim = 2, dimension = 4) from 2 vectors.
# Updated schema.toml: embedding.pq_codebook_path = "embedding.pqcb".
```

または、インデックスにコミット済みのベクトルから直接サンプリングします
— JSONL エクスポートは不要です:

```bash
laurus train pq-codebook --field embedding --from-index --sample-size 5000 --update-schema
```

---

## `search`

[Query DSL](../concepts/query_dsl.md) を使用して検索クエリを実行します。

```bash
laurus search <QUERY> [--limit <N>] [--offset <N>] [--highlight <FIELD>]...
```

クエリ文字列は、各フィールドに設定されたアナライザー自身で解析されます。例えば `schema.toml` で日本語（Lindera）アナライザーを設定したフィールドは、インデックス時と同じ方法でクエリ時にも解析されます。スキーマに宣言されていないフィールドを参照すると、そのフィールド名を含むエラーで拒否されます（typo の検出に役立ちます）。予約済みの `_id` フィールドはスキーマに現れませんが、常に検索可能です。

**引数:**

| 引数 / フラグ | 必須 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `<QUERY>` | はい | — | Laurus Query DSL によるクエリ文字列 |
| `--limit <N>` | いいえ | `10` | 最大結果件数 |
| `--offset <N>` | いいえ | `0` | スキップする結果件数 |
| `--highlight <FIELD>` | いいえ | （なし） | ハイライトする保存済みテキストフィールド（Issue #1134）。複数フィールドを指定する場合は繰り返す。既定の `HighlightConfig` を使用 — one-shot 専用で [REPL](repl.md) では使用不可 |

**クエリ構文の例:**

```bash
# Term クエリ
laurus search "body:rust"

# Phrase クエリ
laurus search 'body:"machine learning"'

# Boolean クエリ
laurus search "+body:programming -body:python"

# Fuzzy クエリ（タイポ許容）
laurus search "body:programing~2"

# Wildcard クエリ
laurus search "title:intro*"

# Range クエリ
laurus search "price:[10 TO 50]"

# 3D 地理クエリ（球 / バウンディングボックス / k-NN）
laurus search "position:geo3d_distance(-3955182, 3350553, 3700276, 5000)"
laurus search "position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)"
laurus search "position:geo3d_nearest(-3955182, 3350553, 3700276, 10)"

# "body" フィールドをハイライト
laurus search "body:rust" --highlight body

# 複数フィールドをハイライト
laurus search "body:rust" --highlight title --highlight body
```

**テーブル出力の例:**

```text
╭──────┬────────┬─────────────────────────────────────────╮
│ ID   │ Score  │ Fields                                  │
├──────┼────────┼─────────────────────────────────────────┤
│ doc1 │ 0.8532 │ body: Rust is a systems..., title: Intr │
│ doc3 │ 0.4210 │ body: JavaScript powers..., title: Web  │
╰──────┴────────┴─────────────────────────────────────────╯
```

`--highlight` を指定した場合のみ `Highlights` 列が追加されます。

```text
╭──────┬────────┬─────────────────────────┬──────────────────────────────╮
│ ID   │ Score  │ Fields                  │ Highlights                   │
├──────┼────────┼─────────────────────────┼──────────────────────────────┤
│ doc1 │ 0.8532 │ body: Rust is a syst... │ body: <mark>Rust</mark> is a │
╰──────┴────────┴─────────────────────────┴──────────────────────────────╯
```

**JSON 出力の例:**

```bash
laurus --format json search "body:rust" --limit 5
```

```json
[
  {
    "id": "doc1",
    "score": 0.8532,
    "fields": {
      "title": "Introduction to Rust",
      "body": "Rust is a systems programming language."
    }
  }
]
```

`--highlight body` を指定すると、マッチした結果には `"highlights"` キーが追加されます（少なくとも1つのフィールドが実際にハイライトされた場合のみ）。

```json
[
  {
    "id": "doc1",
    "score": 0.8532,
    "fields": {
      "title": "Introduction to Rust",
      "body": "Rust is a systems programming language."
    },
    "highlights": {
      "body": ["<mark>Rust</mark> is a systems programming language."]
    }
  }
]
```

完全な意味論（フィールド選択、フィールドごとのアナライザー、`filter_query` がハイライトされない理由など）は[ハイライト](../laurus/highlighting.md)を参照してください。

---

## `repl`

対話型 REPL セッションを開始します。詳細は [REPL](repl.md) を参照してください。

```bash
laurus repl
```

---

## `serve`

gRPC サーバー（およびオプションで HTTP Gateway）を起動します。

```bash
laurus serve [OPTIONS]
```

起動オプション、設定、使用例については [laurus-server のドキュメント](../laurus-server.md)を参照してください:

- [はじめに](../laurus-server/getting_started.md) — 起動オプションと gRPC 接続例
- [設定](../laurus-server/configuration.md) — TOML 設定ファイル、環境変数、優先順位
- [ハンズオンチュートリアル](../laurus-server/tutorial.md) — ステップバイステップの操作ガイド

---

## `mcp`

[Model Context Protocol](https://modelcontextprotocol.io/)（MCP）サーバーを stdio 上で起動します。MCP サーバーを介して、Claude Code や Claude Desktop のような AI アシスタントが標準化されたツール群（`create_index`、`add_document`、`search` など）で稼働中の laurus-server を操作できます。

```bash
laurus mcp [--endpoint <URL>]
```

**引数:**

| フラグ | 環境変数 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `--endpoint <URL>` | `LAURUS_ENDPOINT` | いいえ | 稼働中の laurus-server の gRPC エンドポイント（例: `http://localhost:50051`）。省略すると未接続で起動し、クライアントから後で `connect` MCP ツールを呼び出して接続できます。 |

**使用例:**

```bash
# ローカルの laurus-server に事前接続して MCP サーバーを起動
laurus mcp --endpoint http://localhost:50051

# 未接続で起動し、クライアントが最初に `connect` を呼ぶ運用
laurus mcp
```

MCP サーバーが公開する全ツールの一覧、および Claude Code や Claude Desktop と連携する設定方法については [laurus-mcp のドキュメント](../laurus-mcp.md)を参照してください。
