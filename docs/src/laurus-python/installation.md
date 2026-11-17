# Installation

## From PyPI

```bash
pip install laurus
```

## From source

Building from source requires a Rust toolchain (1.89 or later, matching `workspace.package.rust-version` in the root `Cargo.toml`) and [Maturin](https://github.com/PyO3/maturin).

```bash
# Install Maturin
pip install maturin

# Clone the repository
git clone https://github.com/mosuka/laurus.git
cd laurus/laurus-python

# Build and install in development mode
maturin develop

# Or build a release wheel
maturin build --release
pip install target/wheels/laurus-*.whl
```

## Verify

```python
import laurus
index = laurus.Index()
print(index)  # Index()
```

## Requirements

- Python 3.10 or later
- No runtime dependencies beyond the compiled native extension
