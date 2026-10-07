# Getting Started with the gRPC Server

## Starting the Server

The gRPC server is started via the `serve` subcommand of the `laurus` CLI:

```bash
laurus serve [OPTIONS]
```

### Options

| Option | Short | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config <PATH>` | `-c` | `LAURUS_CONFIG` | -- | Path to a TOML configuration file |
| `--host <HOST>` | `-H` | `LAURUS_HOST` | `0.0.0.0` | Listen address |
| `--port <PORT>` | `-p` | `LAURUS_PORT` | `50051` | Listen port |
| `--http-port <PORT>` | -- | `LAURUS_HTTP_PORT` | -- | HTTP Gateway port (enables HTTP gateway when set) |
| `--max-result-window <N>` | -- | `LAURUS_MAX_RESULT_WINDOW` | `10000` | Largest `offset + limit` per search; larger searches are rejected |

Log verbosity is controlled by the standard `RUST_LOG` environment variable (default: `info`).
See [env_logger syntax](https://docs.rs/env_logger/latest/env_logger/#enabling-logging) for filter directives such as `RUST_LOG=laurus=debug,tonic=warn`.

The global `--index-dir` option (env: `LAURUS_INDEX_DIR`) specifies the index data directory:

```bash
# Using CLI arguments
laurus --index-dir ./my_index serve --port 8080

# Using environment variables
export LAURUS_INDEX_DIR=./my_index
export LAURUS_PORT=8080
export RUST_LOG=debug
laurus serve
```

### Startup Behavior

On startup, the server attempts to open an existing index at the configured data directory. If no index exists, the server starts without one -- you can create an index later via the `CreateIndex` RPC.

## Configuration

You can use a TOML configuration file instead of (or in addition to) command-line options. See [Configuration](configuration.md) for the full reference.

```bash
laurus serve --config config.toml
```

## HTTP Gateway

When `--http-port` is set, an HTTP/JSON gateway starts alongside the gRPC server. See [HTTP Gateway](http_gateway.md) for the full endpoint reference and examples.

```bash
laurus serve --http-port 8080
```

## Graceful Shutdown

When the server receives a shutdown signal (Ctrl+C / SIGINT), it automatically:

1. Stops accepting new connections
2. Commits any pending changes to the index
3. Exits cleanly

## Connecting via gRPC

Any gRPC client can connect to the server. For quick testing, [grpcurl](https://github.com/fullstorydev/grpcurl) is useful:

```bash
# Health check
grpcurl -plaintext localhost:50051 laurus.v1.HealthService/Check

# Create an index
grpcurl -plaintext -d '{
  "schema": {
    "fields": {
      "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
      "body": {"text": {"indexed": true, "stored": true, "term_vectors": true}}
    },
    "default_fields": ["title", "body"]
  }
}' localhost:50051 laurus.v1.IndexService/CreateIndex

# Add a document
grpcurl -plaintext -d '{
  "id": "doc1",
  "document": {
    "fields": {
      "title": {"text_value": "Hello World"},
      "body": {"text_value": "This is a test document."}
    }
  }
}' localhost:50051 laurus.v1.DocumentService/AddDocument

# Commit
grpcurl -plaintext localhost:50051 laurus.v1.DocumentService/Commit

# Search
grpcurl -plaintext -d '{"query": "body:test", "limit": 10}' \
  localhost:50051 laurus.v1.SearchService/Search
```

See [gRPC API Reference](grpc_api.md) for the full API documentation, or try the [Hands-on Tutorial](tutorial.md) for a step-by-step walkthrough using the HTTP Gateway.
