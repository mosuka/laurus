# Contributing to Laurus

Thank you for your interest in contributing to Laurus! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- **Rust** stable toolchain (1.85+, edition 2024)
- **Cargo** (comes with Rust)
- **Git**

### Setup

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
cargo build
```

### Feature Flags

Laurus has optional feature flags for embedding support:

| Flag | Description |
| :--- | :--- |
| `embeddings-candle` | Local BERT embeddings via candle |
| `embeddings-multimodal` | CLIP multimodal embeddings (text + image) |
| `embeddings-openai` | OpenAI API embeddings |
| `embeddings-all` | All embedding features |

Build with all features:

```bash
cargo build --all-features
```

## Development Workflow

### Building

```bash
make build          # Release build
cargo build         # Debug build
```

### Running Tests

```bash
make test                     # Run all workspace tests (default features)
cargo test --workspace        # Same as above
cargo test --all-features     # Run tests with all features enabled
```

For language bindings, use the dedicated targets, which set up the language
toolchain (venv, npm, bundler, composer) before running tests. `laurus-wasm`
additionally requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/)
to be installed:

```bash
make test-laurus-python   # Rust unit tests + Python pytest (via Maturin)
make test-laurus-nodejs   # Build + npm test
make test-laurus-wasm     # wasm32-unknown-unknown build check + wasm-pack tests
make test-laurus-ruby     # Rust unit tests + Ruby minitest
make test-laurus-php      # Build + PHPUnit
```

### Formatting

All code must pass `cargo fmt`:

```bash
make format         # Format all code
cargo fmt --all     # Same as above
```

CI enforces formatting with `cargo fmt --all -- --check`.

### Linting

All code must pass `cargo clippy` with warnings as errors:

```bash
make lint                                              # Lint all code
cargo clippy --workspace --all-targets -- -D warnings  # Same as above
```

CI runs clippy with `--all-features` to check all code paths.

### Benchmarks

```bash
make bench                                # Run all benchmarks
cargo bench -p laurus                     # Same as above
cargo bench -p laurus --bench distance_bench  # Run a single benchmark
```

For more on benchmark methodology, baseline/compare workflow, and the
`scripts/bench-stable.sh` wrapper (CPU pinning on Linux), see
[Benchmarking](docs/src/development/benchmarking.md).

### Documentation

Build the mdBook documentation:

```bash
cd docs
mdbook serve    # Serve locally with live reload at http://localhost:3000
mdbook build    # Build static HTML to docs/book/
```

Generate Rustdoc API documentation:

```bash
cargo doc --open                # Default features
cargo doc --all-features --open # All features
```

## Code Style

### General

- Standard `cargo fmt` formatting (rustfmt defaults)
- All clippy warnings treated as errors

### Naming Conventions

| Element | Convention | Example |
| :--- | :--- | :--- |
| Modules | `snake_case` | `token_filter`, `char_filter` |
| Structs / Enums | `PascalCase` | `LaurusError`, `DataValue` |
| Functions / Methods | `snake_case` | `open_input`, `create_output` |
| Constants | `UPPER_SNAKE_CASE` | `VERSION` |
| Feature flags | `kebab-case` | `embeddings-candle` |

### Documentation Comments

- Module-level docs use `//!` (inner doc comments)
- Public items use `///` with:
  - Brief description
  - `# Arguments` section for parameters
  - `# Returns` section for return values
  - `# Example` section with compilable code

### Language

- Code comments, commit messages, and log messages in **English**
- Doc comments in English

### Error Handling

- Use `LaurusError` variants (via `thiserror`)
- Use the `Result<T>` type alias (`Result<T, LaurusError>`)
- Use convenience constructors: `LaurusError::index(msg)`, `LaurusError::schema(msg)`, etc.

### Async

- Uses `tokio` runtime
- `async-trait` for async trait methods
- Engine operations are async

### Concurrency

- `parking_lot` for mutexes/rwlocks (instead of `std`)
- `rayon` for data parallelism
- `crossbeam` for concurrent data structures
- `Arc` for shared ownership across async boundaries

## Module Organization

Each major component uses a two-level module pattern:

```
component.rs       # Module root: declarations + public re-exports
component/         # Directory: actual implementations
  sub_module.rs
  another.rs
```

Public API is re-exported in `lib.rs`.

## Testing

- Unit tests: `#[cfg(test)]` module at the bottom of each file
- Integration tests: `laurus/tests/` directory
- Sync tests: `#[test]`
- Async tests: `#[tokio::test]`
- Test naming: `test_<what_is_being_tested>` (e.g., `test_error_construction`)
- Benchmarks: `laurus/benches/` using criterion

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Ensure all tests pass: `cargo test --all-features`
3. Ensure code is formatted: `cargo fmt --all -- --check`
4. Ensure clippy passes: `cargo clippy --all-targets --all-features -- -D warnings`
5. Write tests for new functionality
6. Update documentation if applicable (mdBook docs and/or Rustdoc)
7. Submit a pull request with a clear description of your changes

### CI Pipeline

Pull requests automatically run:

1. **Format** — `cargo fmt --all -- --check`
2. **Clippy** — `cargo clippy --all-targets --all-features -- -D warnings`
3. **Test** — `cargo test` on multiple platforms:
   - Linux (x86_64, aarch64)
   - Windows (x86_64)

   Language binding test jobs (`test-laurus-python`, `test-laurus-nodejs`,
   `test-laurus-wasm`, `test-laurus-ruby`, `test-laurus-php`) run when their
   respective crate or shared sources change.

   Release tags additionally build binaries for macOS (x86_64, aarch64) and
   static Linux musl (`x86_64-unknown-linux-musl`,
   `aarch64-unknown-linux-musl`) in `build-binary` -- these are build-only
   and not part of the PR test matrix above; see
   [Prebuilt binaries](https://mosuka.github.io/laurus/laurus-cli/installation.html#prebuilt-binaries).

All checks must pass before a PR can be merged.

## License

By contributing to Laurus, you agree that your contributions will be licensed under the MIT License.
