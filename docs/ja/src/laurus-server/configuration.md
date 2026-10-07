# 設定

laurus-server は CLI 引数、環境変数、TOML 設定ファイルで設定できます。

## 設定の優先順位

サーバーとインデックスの設定は以下の順序で解決されます（優先度が高い順）。

```text
CLI 引数 > 環境変数 > 設定ファイル > デフォルト値
```

ログの詳細度は `RUST_LOG` 環境変数でのみ制御します（デフォルト: `info`）。

例:

```bash
# CLI 引数が環境変数と設定ファイルより優先される
LAURUS_PORT=4567 laurus serve --config config.toml --port 1234
# -> ポート 1234 でリッスン

# 環境変数が設定ファイルより優先される
LAURUS_PORT=4567 laurus serve --config config.toml
# -> ポート 4567 でリッスン

# CLI 引数も環境変数も未設定の場合、設定ファイルの値が使用される
laurus serve --config config.toml
# -> config.toml のポートを使用（未設定の場合はデフォルト 50051）
```

## TOML 設定ファイル

### フォーマット

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080  # オプション: HTTP ゲートウェイを有効化
max_result_window = 10000  # オプション: 1 回の検索の offset + limit の上限

[index]
data_dir = "./laurus_data"

[index.wal]
sync_policy = "group"          # "per_record"（デフォルト） | "group"
group_max_records = 1024       # オプション; デフォルト 1024
group_max_bytes = 1048576      # オプション; デフォルト 1 MiB
group_max_interval_ms = 1000   # オプション; 未設定時は background timer なし（native のみ）

[index.commit]
policy = "every_docs"          # "manual"（デフォルト） | "every_docs" | "interval"
every_docs = 1000              # オプション; N 件ごとに commit（0/未設定で無効）
interval_ms = 1000             # "interval" のみ; 最低 N ミリ秒ごとに commit（native のみ）
```

ログの詳細度は設定ファイルではなく、`RUST_LOG` 環境変数で制御します（デフォルト: `info`）。

### フィールドリファレンス

#### `[server]` セクション

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `host` | String | `"0.0.0.0"` | gRPC サーバーのリッスンアドレス |
| `port` | Integer | `50051` | gRPC サーバーのリッスンポート |
| `http_port` | Integer | -- | HTTP ゲートウェイポート。設定すると gRPC と並行して HTTP/JSON ゲートウェイが起動 |
| `max_result_window` | Integer | `10000` | 1 回の検索で要求できる `offset + limit` の上限（`limit` 未指定時は既定の 10 を数える）。超える検索（gRPC・HTTP・MCP）は `INVALID_ARGUMENT` / HTTP 400 で拒否される。1 以上。Elasticsearch の `index.max_result_window` に相当（Issue #1367） |

#### `[index]` セクション

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `data_dir` | String | `"./laurus_data"` | インデックスデータディレクトリのパス |

#### `[index.wal]` セクション

Write-Ahead Log（WAL）の耐久性ポリシーを制御します。セクション全体を省略した場合、
WAL は **per-record** fsync を使用します（各書き込みは返る前に durable 化されます）。
このポリシーは、起動時に開かれるインデックスと、後から `CreateIndex` で作成される
インデックスの両方に適用されます。耐久性のトレードオフについては
[永続化と WAL → WAL 耐久性ポリシー](../laurus/persistence.md#wal-耐久性ポリシー)
を参照してください。

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `sync_policy` | String | `"per_record"` | 耐久性ポリシー: `"per_record"`（書き込みごとに fsync）または `"group"`（fsync をバッチ化） |
| `group_max_records` | Integer | `1024` | グループコミットのみ。前回 sync 以降にこの件数のレコードが蓄積したら flush |
| `group_max_bytes` | Integer | `1048576` | グループコミットのみ。前回 sync 以降にこのバイト数が蓄積したら flush（デフォルト 1 MiB） |
| `group_max_interval_ms` | Integer | -- | グループコミットのみ。定期 background flush の間隔（ミリ秒）。未設定時は timer なし。**native ターゲットのみ** — `wasm32` では無視される |

`sync_policy = "group"` の場合、WAL は前回 sync 以降に **`group_max_records` 件**または
**`group_max_bytes` バイト**のいずれか（先に到達した方）が蓄積した時点、および commit 時に
無条件で flush します。クラッシュ時には未 sync の最終バッチまで失う可能性があります
（SQLite `synchronous = NORMAL` に相当）。途中で切れた末尾レコードはリカバリ時に
破棄されるため、復旧後のログにはギャップが生じません。

#### `[index.commit]` セクション

自動コミットポリシーを制御します。セクション全体を省略した場合、エンジンは
**manual** — サーバーがインデックスを materialize するときのみ commit します
（インジェスト中の自動コミットはありません）。このポリシーは、起動時に開かれる
インデックスと、後から `CreateIndex` で作成されるインデックスの両方に適用されます。
セマンティクスは
[永続化と WAL → 自動コミットポリシー](../laurus/persistence.md#自動コミットポリシーauto-commit-policy)
を参照してください。`"interval"` は `"every_docs"` の時間ベース版であり、**native のみ** —
WebAssembly では background thread がないため no-op になります。

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `policy` | String | `"manual"` | 自動コミットポリシー: `"manual"`（呼び出し側が commit を駆動）、`"every_docs"`（`every_docs` 件ごとに commit）、または `"interval"`（最低 `interval_ms` ミリ秒ごとに commit） |
| `every_docs` | Integer | -- | `every_docs` ポリシーのみ。適用ドキュメント数がこの値に達するごとに commit。未設定（または `0`）で自動コミット無効（`"manual"` と同等） |
| `interval_ms` | Integer | -- | `interval` ポリシーのみ。background timer により最低この間隔（ミリ秒）ごとに自動コミットするため、インジェストがアイドルでも末尾の部分バッチが commit されます — `every_docs` の時間ベース版。**native ターゲットのみ** — `wasm32` では no-op として扱われます（background thread なし） |

`CommitPolicy` は `[index.wal]` と直交します。WAL セクションは *append がいつ fsync
されるか* を、このセクションは *ストアがいつ materialize するか* を制御します。commit は
必ず WAL flush から始まるため、自動コミットは任意の WAL ポリシー下で機能します。

## 環境変数

| 変数 | 対応する設定 | 説明 |
| :--- | :--- | :--- |
| `LAURUS_HOST` | `server.host` | リッスンアドレス |
| `LAURUS_PORT` | `server.port` | gRPC リッスンポート |
| `LAURUS_HTTP_PORT` | `server.http_port` | HTTP ゲートウェイポート |
| `LAURUS_MAX_RESULT_WINDOW` | `server.max_result_window` | 1 回の検索の `offset + limit` の上限 |
| `LAURUS_INDEX_DIR` | `index.data_dir` | インデックスデータディレクトリ |
| `RUST_LOG` | -- | ログフィルタディレクティブ（例: `info`, `debug`, `laurus=debug,tonic=warn`） |
| `LAURUS_CONFIG` | -- | TOML 設定ファイルのパス |

## CLI 引数

| オプション | 短縮形 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `--config <PATH>` | `-c` | -- | TOML 設定ファイルのパス |
| `--host <HOST>` | `-H` | `0.0.0.0` | リッスンアドレス |
| `--port <PORT>` | `-p` | `50051` | gRPC リッスンポート |
| `--http-port <PORT>` | -- | -- | HTTP ゲートウェイポート |
| `--max-result-window <N>` | -- | `10000` | 1 回の検索の `offset + limit` の上限 |
| `--index-dir <PATH>` | -- | `./laurus_index` | インデックスデータディレクトリ（グローバルオプション） |

## よくある設定例

### 開発環境（gRPC のみ）

```toml
[server]
host = "127.0.0.1"
port = 50051

[index]
data_dir = "./dev_data"
```

```bash
RUST_LOG=debug laurus serve --config config.toml
```

### 本番環境（gRPC + HTTP ゲートウェイ）

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080

[index]
data_dir = "/var/lib/laurus/data"
```

### 最小構成（環境変数のみ）

```bash
export LAURUS_INDEX_DIR=/var/lib/laurus/data
export LAURUS_PORT=50051
export LAURUS_HTTP_PORT=8080
export RUST_LOG=info
laurus serve
```
