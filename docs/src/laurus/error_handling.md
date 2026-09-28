# Error Handling

Laurus uses a unified error type for all operations. Understanding the error system helps you write robust applications that handle failures gracefully.

## LaurusError

All Laurus operations return `Result<T>`, which is an alias for `std::result::Result<T, LaurusError>`.

`LaurusError` is an enum with variants for each category of failure:

| Variant | Description | Common Causes |
| :--- | :--- | :--- |
| `Io` | I/O errors | File not found, permission denied, disk full |
| `Index` | Index operation errors | Corrupt index, segment read failure |
| `Schema` | Schema-related errors | Unknown field name, type mismatch |
| `Analysis` | Text analysis errors | Tokenizer failure, invalid filter config |
| `Query` | Query parsing/execution errors | Malformed Query DSL, unknown field in query |
| `Storage` | Storage backend errors | Failed to open storage, write failure |
| `Field` | Field definition errors | Invalid field options, duplicate field name |
| `InvalidArgument` | The caller passed an invalid argument | Unknown field in a DSL query, wrong value type in a document, out-of-range value |
| `BenchmarkFailed` | Benchmark errors | Benchmark execution failure |
| `ThreadJoinError` | Thread join errors | Panic in a worker thread |
| `Json` | JSON serialization errors | Malformed document JSON |
| `Anyhow` | Wrapped anyhow errors | Errors from third-party crates via `anyhow` |
| `InvalidOperation` | Invalid operation | Searching before commit, double close |
| `ResourceExhausted` | Resource limits exceeded | Out of memory, too many open files |
| `SerializationError` | Binary serialization errors | Corrupt data on disk |
| `OperationCancelled` | Operation was cancelled | Timeout, user cancellation |
| `NotImplemented` | Feature not available | Unimplemented operation |
| `Other` | Generic errors | Timeout, invalid config, not found |

## Basic Error Handling

### Using the `?` Operator

The simplest approach — propagate errors to the caller:

```rust
use laurus::{Engine, Result};

async fn index_documents(engine: &Engine) -> Result<()> {
    let doc = laurus::Document::builder()
        .add_text("title", "Rust Programming")
        .build();

    engine.put_document("doc1", doc).await?;
    engine.commit().await?;
    Ok(())
}
```

### Matching on Error Variants

When you need different behavior for different error types:

```rust
use laurus::{Engine, LaurusError};

async fn safe_search(engine: &Engine, query: &str) {
    match engine.search(/* request */).await {
        Ok(results) => {
            for result in results {
                println!("{}: {}", result.id, result.score);
            }
        }
        Err(LaurusError::Query(msg)) => {
            eprintln!("Invalid query syntax: {}", msg);
        }
        Err(LaurusError::Io(e)) => {
            eprintln!("Storage I/O error: {}", e);
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
        }
    }
}
```

### Checking Error Types with `downcast`

Since `LaurusError` implements `std::error::Error`, you can use standard error handling patterns:

```rust
use laurus::LaurusError;

fn is_retriable(error: &LaurusError) -> bool {
    matches!(error, LaurusError::Io(_) | LaurusError::ResourceExhausted(_))
}
```

## Common Error Scenarios

### Schema Mismatch

Adding a document with fields that don't match the schema:

```rust
// Schema has "title" (Text) and "year" (Integer)
let doc = Document::builder()
    .add_text("title", "Hello")
    .add_text("unknown_field", "this field is not in schema")
    .build();

// Fields not in the schema are silently ignored during indexing.
// No error is raised — only schema-defined fields are processed.
```

### Query Parsing Errors

Invalid Query DSL syntax returns a `Query` error:

```rust
use laurus::engine::query::UnifiedQueryParser;

let parser = UnifiedQueryParser::new();
match parser.parse("title:\"unclosed phrase") {
    Ok(request) => { /* ... */ }
    Err(LaurusError::Query(msg)) => {
        // msg contains details about the parse failure
        eprintln!("Bad query: {}", msg);
    }
    Err(e) => { /* other errors */ }
}
```

### Storage I/O Errors

File-based storage may encounter I/O errors:

```rust
use laurus::storage::{StorageConfig, StorageFactory};

match StorageFactory::open(StorageConfig::File {
    path: "/nonexistent/path".into(),
    loading_mode: Default::default(),
}) {
    Ok(storage) => { /* ... */ }
    Err(LaurusError::Io(e)) => {
        eprintln!("Cannot open storage: {}", e);
    }
    Err(e) => { /* other errors */ }
}
```

## Convenience Constructors

`LaurusError` provides factory methods for creating errors in custom implementations:

| Method | Creates |
| :--- | :--- |
| `LaurusError::index(msg)` | `Index` variant |
| `LaurusError::schema(msg)` | `Schema` variant |
| `LaurusError::analysis(msg)` | `Analysis` variant |
| `LaurusError::query(msg)` | `Query` variant |
| `LaurusError::storage(msg)` | `Storage` variant |
| `LaurusError::field(msg)` | `Field` variant |
| `LaurusError::other(msg)` | `Other` variant |
| `LaurusError::cancelled(msg)` | `OperationCancelled` variant |
| `LaurusError::invalid_argument(msg)` | `InvalidArgument` variant |
| `LaurusError::invalid_config(msg)` | `Other` with "Invalid configuration" prefix |
| `LaurusError::not_found(msg)` | `Other` with "Not found" prefix |
| `LaurusError::timeout(msg)` | `Other` with "Timeout" prefix |

These are useful when implementing custom [Analyzer, Embedder, or Storage](extensibility.md) traits:

```rust
use laurus::{LaurusError, Result};

fn validate_dimension(dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(LaurusError::invalid_argument("dimension must be > 0"));
    }
    Ok(())
}
```

## Automatic Conversions

`LaurusError` implements `From` for common error types, so they convert automatically with `?`:

| Source Type | Target Variant |
| :--- | :--- |
| `std::io::Error` | `LaurusError::Io` |
| `serde_json::Error` | `LaurusError::Json` |
| `anyhow::Error` | `LaurusError::Anyhow` |

## Next Steps

- [Extensibility](extensibility.md) — implement custom traits with proper error handling
- [API Reference](api_reference.md) — full method signatures and return types
