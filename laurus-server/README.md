# laurus-server

[![Crates.io](https://img.shields.io/crates/v/laurus-server.svg)](https://crates.io/crates/laurus-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

gRPC server with HTTP/JSON gateway for the [Laurus](https://github.com/mosuka/laurus) search engine.

## Features

- **Persistent engine** -- The index stays open across requests; no WAL replay on every call
- **Full gRPC API** -- Index management, document CRUD, commit, and search (unary + streaming)
- **HTTP Gateway** -- Optional HTTP/JSON gateway alongside gRPC for REST-style access
- **Health checking** -- Standard health check endpoint for load balancers and orchestrators
- **Graceful shutdown** -- Pending changes are committed automatically on Ctrl+C / SIGINT, or SIGTERM on Unix
- **TOML configuration** -- Optional config file with CLI and environment variable overrides

## Quick Start

```bash
# Start with default settings (gRPC on port 50051)
laurus serve

# Start with HTTP Gateway
laurus serve --http-port 8080

# Start with a configuration file
laurus serve --config config.toml
```

## gRPC Services

| Service | RPCs |
| :--- | :--- |
| `HealthService` | `Check` |
| `IndexService` | `CreateIndex`, `GetIndex`, `GetSchema`, `AddField`, `DeleteField` |
| `DocumentService` | `PutDocument`, `AddDocument`, `GetDocuments`, `DeleteDocuments`, `Commit` |
| `SearchService` | `Search`, `SearchStream` |

## HTTP Gateway Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/index` | Create index |
| `GET` | `/v1/index` | Get index stats |
| `GET` | `/v1/schema` | Get schema |
| `POST` | `/v1/schema/fields` | Add field |
| `DELETE` | `/v1/schema/fields/{name}` | Delete field |
| `PUT` | `/v1/documents/{id}` | Put (upsert) document |
| `POST` | `/v1/documents/{id}` | Add document (chunk) |
| `GET` | `/v1/documents/{id}` | Get documents |
| `DELETE` | `/v1/documents/{id}` | Delete documents |
| `POST` | `/v1/commit` | Commit changes |
| `POST` | `/v1/search` | Search |
| `POST` | `/v1/search/stream` | Streaming search (SSE) |

## Documentation

- [Server Guide](https://mosuka.github.io/laurus/laurus-server.html)
- [Getting Started](https://mosuka.github.io/laurus/laurus-server/getting_started.html)
- [Configuration](https://mosuka.github.io/laurus/laurus-server/configuration.html)
- [gRPC API Reference](https://mosuka.github.io/laurus/laurus-server/grpc_api.html)
- [HTTP Gateway](https://mosuka.github.io/laurus/laurus-server/http_gateway.html)
- [Tutorial](https://mosuka.github.io/laurus/laurus-server/tutorial.html)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
