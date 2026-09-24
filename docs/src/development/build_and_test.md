# Build & Test

## Prerequisites

- **Rust** 1.85 or later (edition 2024)
- **Cargo** (included with Rust)
- **protobuf compiler** (`protoc`) -- required for building `laurus-server`
- **[cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild)** --
  optional, only needed to cross-compile static musl binaries locally (see
  [Cross-compiling static musl binaries](#cross-compiling-static-musl-binaries))

## Building

```bash
# Build all crates
cargo build

# Build with specific features
cargo build --features embeddings-candle

# Build in release mode
cargo build --release
```

## Cross-compiling static musl binaries

`laurus-cli`'s release workflow builds fully static
`x86_64-unknown-linux-musl` / `aarch64-unknown-linux-musl` binaries in
addition to the dynamically-linked glibc ones (see
[Prebuilt binaries](../laurus-cli/installation.md#prebuilt-binaries)).

Prerequisites:

```bash
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
pip install cargo-zigbuild
```

Then build with `cargo zigbuild` in place of `cargo build`:

```bash
cargo zigbuild --release --target x86_64-unknown-linux-musl \
  -p laurus-cli --features embeddings-all
```

(equivalent to `make build-laurus-cli-musl` for the `x86_64` target).

**Why `cargo-zigbuild` and not `apt install musl-tools`?** Ubuntu's
`musl-tools` package provides `musl-gcc` (C) but no `musl-g++` (C++). Most of
laurus's `--features embeddings-all` dependency graph is C-only or pure Rust
(`aws-lc-sys`, `onig_sys`), but a plain `musl-tools` setup is one dependency
feature flip away from needing C++ again (`tokenizers`' `esaxx_fast`, which
laurus deliberately disables -- see [Feature Flags](feature_flags.md)).
`cargo-zigbuild` uses [Zig](https://ziglang.org/) as the cross C/C++
toolchain, which bundles musl headers and libraries for every target and
needs no Docker.

Verify a build is genuinely static:

```bash
file target/x86_64-unknown-linux-musl/release/laurus
# -> ELF 64-bit LSB executable, ..., statically linked
readelf -d target/x86_64-unknown-linux-musl/release/laurus | grep NEEDED
# -> (no output)
```

See `build-binary` in
[`.github/workflows/release.yml`](../../../.github/workflows/release.yml)
for the CI configuration this mirrors.

## Testing

```bash
# Run all workspace tests (default features)
cargo test

# Run a specific test by name
cargo test <test_name>

# Run tests for a specific crate
cargo test -p laurus
cargo test -p laurus-cli
cargo test -p laurus-server
cargo test -p laurus-mcp
```

### Language binding tests

Each language binding has its own toolchain (Python virtualenv, Node.js
npm, Ruby Bundler, PHP Composer, `wasm32-unknown-unknown` target). The
Makefile wraps these so each target sets up the toolchain before
running the suite:

```bash
make test-laurus-python   # cargo test -p laurus-python + pytest via Maturin
make test-laurus-nodejs   # npm run build:debug + npm test
make test-laurus-wasm     # cargo build -p laurus-wasm --target wasm32-unknown-unknown
make test-laurus-ruby     # cargo test -p laurus-ruby + Ruby minitest
make test-laurus-php      # cargo build -p laurus-php --release + PHPUnit
```

`laurus-php` is excluded from the Cargo workspace because of `links =
"clang"` conflicts with `laurus-ruby`; it builds and tests as a
standalone crate via the Makefile target above. See
[`Makefile`](../../../Makefile) for the full target list including the
matching `format-laurus-*` / `lint-laurus-*` / `build-laurus-*`
variants.

## Linting

```bash
# Run clippy with warnings as errors
cargo clippy -- -D warnings
```

## Formatting

```bash
# Check formatting
cargo fmt --check

# Apply formatting
cargo fmt
```

## Documentation

### API Documentation

```bash
# Generate and open Rust API docs
cargo doc --no-deps --open
```

### mdBook Documentation

```bash
# Build the documentation site
mdbook build docs

# Start a local preview server (http://localhost:3000)
mdbook serve docs

# Lint markdown files
markdownlint-cli2 "docs/src/**/*.md"

# Check binding README / docs / example snippets against the current
# Node.js / WASM / PHP signatures (also run by `make lint` and CI)
python3 scripts/check-binding-snippets.py
```
