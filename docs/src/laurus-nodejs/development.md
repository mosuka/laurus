# Development Setup

This page covers how to set up a local development environment
for the `laurus-nodejs` binding, build it, and run the test
suite.

## Prerequisites

- **Rust** 1.89 or later with Cargo
- **Node.js** 24.15 or later with npm (matches the `engines.node` requirement in `package.json`)
- Repository cloned locally

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## Build

### Development build

Compiles the Rust native addon in debug mode. Re-run after
any Rust source change.

```bash
cd laurus-nodejs
npm install
npm run build:debug
```

### Release build

```bash
npm run build
```

### Verify the build

```javascript
node -e "
const { Index } = require('./index.js');
Index.create().then(idx => console.log(idx.stats()));
"
// { documentCount: 0, vectorFields: {} }
```

## Testing

Tests use [Vitest](https://vitest.dev/) and are located in
`__tests__/`.

```bash
# Run all tests
npm test
```

To run a specific test by name:

```bash
npx vitest run -t "searches with DSL string"
```

## Linting and formatting

```bash
# Rust lint (Clippy)
cargo clippy -p laurus-nodejs -- -D warnings

# Rust formatting
cargo fmt -p laurus-nodejs --check

# Apply formatting
cargo fmt -p laurus-nodejs
```

## Cleaning up

```bash
# Remove build artifacts
rm -f *.node index.js index.d.ts

# Remove node_modules
rm -rf node_modules
```

## Project layout

```text
laurus-nodejs/
├── Cargo.toml          # Rust crate manifest
├── build.rs            # napi-build setup
├── package.json        # npm package metadata
├── README.md           # English README
├── README_ja.md        # Japanese README
├── src/                # Rust source (napi-rs binding)
│   ├── lib.rs          # Module registration
│   ├── index.rs        # Index class
│   ├── schema.rs       # Schema class
│   ├── query.rs        # Query classes
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # JS ↔ DataValue conversion
│   └── errors.rs       # Error mapping
├── __tests__/          # Vitest integration tests
│   └── index.spec.mjs
└── examples/           # Runnable Node.js examples
    ├── quickstart.mjs
    ├── lexical-search.mjs
    ├── vector-search.mjs
    └── hybrid-search.mjs
```
