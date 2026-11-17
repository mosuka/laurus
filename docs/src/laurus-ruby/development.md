# Development Setup

This page covers how to set up a local development environment
for the `laurus-ruby` binding, build it, and run the test
suite.

## Prerequisites

- **Rust** 1.89 or later with Cargo
- **Ruby** 3.1 or later with Bundler
- Repository cloned locally

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## Build

### Development build

Compiles the Rust native extension in debug mode. Re-run after
any Rust source change.

```bash
cd laurus-ruby
bundle install
bundle exec rake compile
```

### Release build

```bash
gem build laurus.gemspec
```

### Verify the build

```ruby
ruby -e "
require 'laurus'
index = Laurus::Index.new
puts index.stats
"
# {"document_count"=>0, "vector_fields"=>{}}
```

## Testing

Tests use [Minitest](https://github.com/minitest/minitest) and are located in
`test/`.

```bash
# Run all tests
bundle exec rake test
```

To run a specific test file:

```bash
bundle exec ruby -Ilib -Itest test/test_index.rb
```

## Linting and formatting

```bash
# Rust lint (Clippy)
cargo clippy -p laurus-ruby -- -D warnings

# Rust formatting
cargo fmt -p laurus-ruby --check

# Apply formatting
cargo fmt -p laurus-ruby
```

## Cleaning up

```bash
# Remove build artifacts
bundle exec rake clean

# Remove installed gems
rm -rf vendor/bundle
```

## Makefile reference

| Target | Description |
| :--- | :--- |
| `make build-laurus-ruby` | Bundler install + `bundle exec rake compile` (release gem) |
| `make test-laurus-ruby` | Rust unit tests + Ruby minitest |
| `make lint-laurus-ruby` | Clippy with `-D warnings` |
| `make format-laurus-ruby` | `cargo fmt -p laurus-ruby` |

## Project layout

```text
laurus-ruby/
├── Cargo.toml          # Rust crate manifest
├── laurus.gemspec      # Gem specification
├── Gemfile             # Bundler dependency file
├── Rakefile            # Rake tasks (compile, test, clean)
├── lib/
│   └── laurus.rb       # Ruby entrypoint (loads native extension)
├── ext/
│   └── laurus_ruby/    # Native extension build configuration
│       └── extconf.rb  # rb_sys extension configuration
├── src/                # Rust source (Magnus binding)
│   ├── lib.rs          # Module registration
│   ├── index.rs        # Index class
│   ├── schema.rs       # Schema class
│   ├── query.rs        # Query classes
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # Ruby ↔ DataValue conversion
│   └── errors.rs       # Error mapping
├── test/               # Minitest tests
│   ├── test_helper.rb
│   └── test_index.rb
└── examples/           # Runnable Ruby examples
```
