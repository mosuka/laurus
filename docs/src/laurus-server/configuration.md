# Configuration

The laurus-server can be configured through CLI arguments, environment variables, and a TOML configuration file.

## Configuration Priority

Server and index settings are resolved in the following order (highest priority first):

```text
CLI arguments > Environment variables > Config file > Defaults
```

Log verbosity is controlled exclusively by the `RUST_LOG` environment variable (default: `info`).

For example:

```bash
# CLI argument wins over environment variable and config file
LAURUS_PORT=4567 laurus serve --config config.toml --port 1234
# -> Listens on port 1234

# Environment variable wins over config file
LAURUS_PORT=4567 laurus serve --config config.toml
# -> Listens on port 4567

# Config file value is used when no CLI argument or env var is set
laurus serve --config config.toml
# -> Uses port from config.toml (or default 50051 if not set)
```

## TOML Configuration File

### Format

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080  # Optional: enables HTTP Gateway
max_result_window = 10000  # Optional; largest offset + limit per search

[index]
data_dir = "./laurus_data"

[index.wal]
sync_policy = "group"          # "per_record" (default) | "group"
group_max_records = 1024       # optional; default 1024
group_max_bytes = 1048576      # optional; default 1 MiB
group_max_interval_ms = 1000   # optional; no background timer when unset (native only)

[index.commit]
policy = "every_docs"          # "manual" (default) | "every_docs" | "interval"
every_docs = 1000              # optional; commit every N docs (0/unset disables)
interval_ms = 1000             # "interval" only; commit at least every N ms (native only)
```

Log verbosity is controlled by the `RUST_LOG` environment variable (default: `info`), not through the config file.

### Field Reference

#### `[server]` Section

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `host` | String | `"0.0.0.0"` | Listen address for the gRPC server |
| `port` | Integer | `50051` | Listen port for the gRPC server |
| `http_port` | Integer | -- | HTTP Gateway port. When set, the HTTP/JSON gateway starts alongside gRPC. |
| `max_result_window` | Integer | `10000` | Largest `offset + limit` a search may request (the default `limit` of 10 counts when `limit` is unset). Larger searches (gRPC, HTTP and MCP) are rejected with `INVALID_ARGUMENT` / HTTP 400. Must be at least 1. Like Elasticsearch's `index.max_result_window` (Issue #1367). |

#### `[index]` Section

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | String | `"./laurus_data"` | Path to the index data directory |

#### `[index.wal]` Section

Controls the Write-Ahead Log (WAL) durability policy. When the whole section is
omitted, the WAL uses **per-record** fsync (every write is durable before it
returns). The policy applies to both an index opened at boot and any index
created later through `CreateIndex`. See
[Persistence & WAL → WAL Durability Policy](../laurus/persistence.md#wal-durability-policy)
for the durability trade-off.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sync_policy` | String | `"per_record"` | Durability policy: `"per_record"` (fsync every write) or `"group"` (batch fsyncs) |
| `group_max_records` | Integer | `1024` | Group commit only. Flush once this many records accumulate since the last sync |
| `group_max_bytes` | Integer | `1048576` | Group commit only. Flush once this many bytes accumulate since the last sync (default 1 MiB) |
| `group_max_interval_ms` | Integer | -- | Group commit only. Periodic background flush interval in milliseconds. No timer runs when unset. **Native targets only** — ignored on `wasm32` |

Under `sync_policy = "group"`, the WAL flushes when **either** `group_max_records`
records **or** `group_max_bytes` bytes have accumulated since the last sync
(whichever comes first), and unconditionally on commit. A crash can lose up to
the last unsynced batch (comparable to SQLite `synchronous = NORMAL`); a torn
trailing record is dropped on recovery, so the recovered log is gap-free.

#### `[index.commit]` Section

Controls the auto-commit policy. When the whole section is omitted, the engine
is **manual** — it commits only when the server materializes an index (there is
no automatic commit during ingestion). The policy applies to both an index
opened at boot and any index created later through `CreateIndex`. See
[Persistence & WAL → Auto-commit Policy](../laurus/persistence.md#auto-commit-policy)
for the semantics. `"interval"` is the time-based counterpart of `"every_docs"`
and is **native only** — under WebAssembly it is a no-op because there are no
background threads.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `policy` | String | `"manual"` | Auto-commit policy: `"manual"` (caller-driven commits), `"every_docs"` (commit every `every_docs` applied documents), or `"interval"` (commit at least every `interval_ms` milliseconds) |
| `every_docs` | Integer | -- | `every_docs` policy only. Commit after this many applied documents. Unset (or `0`) disables auto-commit, equivalent to `"manual"` |
| `interval_ms` | Integer | -- | `interval` policy only. Auto-commit at least this often (milliseconds) via a background timer, so a trailing partial batch commits even while ingestion is idle — the time-based counterpart of `every_docs`. **Native targets only** — treated as a no-op on `wasm32` (no background threads) |

`CommitPolicy` is orthogonal to `[index.wal]`: the WAL section governs *when
appends are fsync'd*, while this section governs *when the stores materialize*.
An auto-commit works under any WAL policy because a commit always begins with a
WAL flush.

## Environment Variables

| Variable | Maps To | Description |
| :--- | :--- | :--- |
| `LAURUS_HOST` | `server.host` | Listen address |
| `LAURUS_PORT` | `server.port` | gRPC listen port |
| `LAURUS_HTTP_PORT` | `server.http_port` | HTTP Gateway port |
| `LAURUS_MAX_RESULT_WINDOW` | `server.max_result_window` | Largest `offset + limit` per search |
| `LAURUS_INDEX_DIR` | `index.data_dir` | Index data directory |
| `RUST_LOG` | -- | Log filter directive (e.g. `info`, `debug`, `laurus=debug,tonic=warn`) |
| `LAURUS_CONFIG` | -- | Path to TOML config file |

## CLI Arguments

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config <PATH>` | `-c` | -- | Path to TOML configuration file |
| `--host <HOST>` | `-H` | `0.0.0.0` | Listen address |
| `--port <PORT>` | `-p` | `50051` | gRPC listen port |
| `--http-port <PORT>` | -- | -- | HTTP Gateway port |
| `--max-result-window <N>` | -- | `10000` | Largest `offset + limit` per search |
| `--index-dir <PATH>` | -- | `./laurus_index` | Index data directory (global option) |

## Common Configurations

### Development (gRPC only)

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

### Production (gRPC + HTTP Gateway)

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080

[index]
data_dir = "/var/lib/laurus/data"
```

### Minimal (environment variables only)

```bash
export LAURUS_INDEX_DIR=/var/lib/laurus/data
export LAURUS_PORT=50051
export LAURUS_HTTP_PORT=8080
export RUST_LOG=info
laurus serve
```
