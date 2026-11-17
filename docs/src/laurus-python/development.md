# Development Setup

This page covers how to set up a local development environment for the
`laurus-python` binding, build it, and run the test suite.

## Prerequisites

- **Rust** 1.89 or later with Cargo
- **Python** 3.8 or later
- Repository cloned locally

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## Python virtual environment

All Python tooling (Maturin, pytest, …) is managed inside a dedicated virtual
environment located at `laurus-python/.venv`.

```bash
# Create the venv and install maturin + pytest
make venv
```

This is equivalent to:

```bash
python3 -m venv laurus-python/.venv
laurus-python/.venv/bin/pip install maturin pytest
```

> **Note:** You do not need to activate the venv manually.
> All `make` targets invoke the venv binaries directly.

## Build

### Development build (editable install)

Compiles the Rust extension and installs it into the venv in one step.
Re-run after any Rust source change.

```bash
cd laurus-python
VIRTUAL_ENV=$(pwd)/.venv .venv/bin/maturin develop
```

Or use the Makefile shortcut that also builds a distributable wheel:

```bash
make build-laurus-python
```

This produces a release wheel under `target/wheels/`:

```text
target/wheels/laurus-0.x.y-cp312-cp312-manylinux_2_34_x86_64.whl
```

### Verify the build

```python
# With the venv activated, or using its Python directly:
laurus-python/.venv/bin/python -c "import laurus; print(laurus.Index())"
# Index()
```

## Testing

`make test-laurus-python` runs two test suites in order:

1. **Rust unit tests** via `cargo test -p laurus-python`
2. **Python integration tests** via `pytest` (after a fresh `maturin develop`)

```bash
make test-laurus-python
```

To run only the Python tests (skipping the Rust step):

```bash
cd laurus-python
VIRTUAL_ENV=$(pwd)/.venv .venv/bin/maturin develop --quiet
.venv/bin/pytest tests/ -v
```

To run a single test by name:

```bash
.venv/bin/pytest tests/ -v -k test_vector_query
```

## Linting and formatting

```bash
# Rust lint (Clippy)
make lint-laurus-python

# Rust formatting
make format-laurus-python
```

## Cleaning up

```bash
# Remove the venv only
make venv-clean

# Remove everything (venv + all Cargo build artifacts)
make clean
```

## Makefile reference

| Target | Description |
| :--- | :--- |
| `make venv` | Create `.venv` and install `maturin` + `pytest` |
| `make venv-clean` | Remove `.venv` |
| `make build-laurus-python` | Build a release wheel via `maturin build` |
| `make test-laurus-python` | Rust unit tests + Python pytest |
| `make lint-laurus-python` | Clippy with `-D warnings` |
| `make format-laurus-python` | `cargo fmt -p laurus-python` |
| `make clean` | Remove venv and all Cargo build artifacts |

## Project layout

```text
laurus-python/
├── Cargo.toml          # Rust crate manifest
├── pyproject.toml      # Python package metadata (Maturin / PEP 517)
├── README.md           # English README
├── README_ja.md        # Japanese README
├── src/                # Rust source (PyO3 binding)
│   ├── lib.rs          # Module registration
│   ├── index.rs        # Index class
│   ├── schema.rs       # Schema class
│   ├── query.rs        # Query classes
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # Python ↔ DataValue conversion
│   └── errors.rs       # Error mapping
├── tests/              # Python pytest integration tests
│   └── test_index.py
└── examples/           # Runnable Python examples
    ├── quickstart.py
    ├── lexical_search.py
    ├── vector_search.py
    ├── hybrid_search.py
    ├── synonym_graph_filter.py
    ├── search_with_openai.py
    └── multimodal_search.py
```
