# REPL（対話モード）

REPL は、毎回 `laurus` コマンドをフルで入力することなく、インデックスを操作できる対話型セッションを提供します。

## REPL の起動

```bash
laurus --index-dir ./my_index repl
```

指定されたディレクトリにインデックスが存在する場合、自動的に開かれます:

```text
Laurus REPL (type 'help' for commands, 'quit' to exit)
laurus>
```

インデックスがまだ存在しない場合、インデックスなしで REPL が起動し、作成を案内します:

```text
Laurus REPL — no index found at ./my_index.
Use 'create index <schema_path>' to create one, or 'help' for commands.
laurus>
```

## 利用可能なコマンド

コマンドは CLI と同じ `<操作> <リソース>` の順序に従います。

| コマンド | 説明 |
| :--- | :--- |
| `create index [schema_path]` | インデックスを作成（パス省略時は対話型ウィザード） |
| `create schema <output_path>` | 対話型スキーマ生成ウィザード |
| `search <query>` | インデックスを検索（Lexical / Vector / ハイブリッド DSL） |
| `add field <name> <json>` | スキーマにフィールドを追加 |
| `add doc <id> <json>` | ドキュメントを追加（追記、同一 ID で複数チャンク可） |
| `put doc <id> <json>` | ドキュメントを上書き（同一 ID の既存チャンクを置換） |
| `get stats` | インデックスの統計情報を表示 |
| `get schema` | 現在のスキーマを表示 |
| `get docs <id>` | ID で全ドキュメント（チャンクを含む）を取得 |
| `delete field <name>` | スキーマからフィールドを削除 |
| `delete docs <id>` | ID で全ドキュメント（チャンクを含む）を削除 |
| `commit` | 保留中の変更をコミット |
| `help` | 利用可能なコマンドを表示 |
| `quit` / `exit` | REPL を終了 |

> **注意:** `create`、`help`、`quit` 以外のコマンドはインデックスがロードされている必要があります。インデックスがロードされていない場合、まず `create index` を実行するようメッセージが表示されます。

## 使用例

### インデックスの作成

```text
laurus> create index ./schema.toml
Index created at ./my_index.
laurus> add doc doc1 {"fields":{"title":"Hello","body":"World"}}
Document 'doc1' added.
```

### 検索

```text
laurus> search body:rust
╭──────┬────────┬────────────────────────────────────╮
│ ID   │ Score  │ Fields                             │
├──────┼────────┼────────────────────────────────────┤
│ doc1 │ 0.8532 │ body: Rust is a systems..., title… │
╰──────┴────────┴────────────────────────────────────╯
```

`search` は [Query DSL](../concepts/query_dsl.md) の全体（Lexical・Vector・ハイブリッド句）を受け付け、単発の `laurus search` コマンドと同様に、クエリを各フィールドに設定されたアナライザー自身で解析します。結果件数は常に 10 件に制限されます。異なる件数が必要な場合は、REPL の外で `laurus search --limit N` を使用してください。同様に REPL には `--highlight`（Issue #1134）に相当する機能もありません — [ハイライト付きの結果](commands.md#search)が必要な場合は単発の `laurus search --highlight <FIELD>` コマンドを使用してください。

### フィールドの管理

```text
laurus> add field category {"Text": {"indexed": true, "stored": true}}
Field 'category' added.
laurus> delete field category
Field 'category' deleted.
```

### ドキュメントの追加とコミット

```text
laurus> add doc doc4 {"fields":{"title":"New Document","body":"Some content here."}}
Document 'doc4' added.
laurus> commit
Changes committed.
```

### 情報の取得

```text
laurus> get stats
Document count: 3

laurus> get schema
{
  "fields": { ... },
  "default_fields": ["title", "body"]
}

laurus> get docs doc4
╭──────┬───────────────────────────────────────────────╮
│ ID   │ Fields                                        │
├──────┼───────────────────────────────────────────────┤
│ doc4 │ body: Some content here., title: New Document │
╰──────┴───────────────────────────────────────────────╯
```

### ドキュメントの削除

```text
laurus> delete docs doc4
Documents 'doc4' deleted.
laurus> commit
Changes committed.
```

## 機能

- **行編集** — 矢印キー、Home/End キー、および標準的な readline ショートカット
- **履歴** — 上下矢印キーで以前のコマンドを呼び出し
- **Ctrl+C / Ctrl+D** — REPL を正常に終了
