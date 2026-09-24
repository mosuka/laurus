LAURUS_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="laurus") | .version')

USER_AGENT ?= $(shell curl --version | head -n1 | awk '{print $1"/"$2}')
USER ?= $(shell whoami)
HOSTNAME ?= $(shell hostname)

# ── Python venv ─────────────────────────────────────────────────────────────
PYTHON_VENV_DIR := laurus-python/.venv
PYTHON          := $(PYTHON_VENV_DIR)/bin/python
PIP             := $(PYTHON_VENV_DIR)/bin/pip
MATURIN         := $(PYTHON_VENV_DIR)/bin/maturin
PYTEST          := $(PYTHON_VENV_DIR)/bin/pytest

.DEFAULT_GOAL := help

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-30s %s\n", $$1, $$2}'

# ── Python venv setup ───────────────────────────────────────────────────────

$(PYTHON_VENV_DIR):
	python3 -m venv $(PYTHON_VENV_DIR)
	$(PIP) install --quiet --upgrade pip

venv: $(PYTHON_VENV_DIR) ## Create laurus-python venv and install dev dependencies
	$(PIP) install --quiet maturin pytest

venv-clean: ## Remove the laurus-python venv
	rm -rf $(PYTHON_VENV_DIR)

# ── Clean ──────────────────────────────────────────────────────────────────

clean: venv-clean ## Clean all build artifacts (including venv)
	cargo clean

# ── Format ─────────────────────────────────────────────────────────────────

format: ## Format all crates
	cargo fmt

format-laurus: ## Format laurus
	cargo fmt -p laurus

format-laurus-cli: ## Format laurus-cli
	cargo fmt -p laurus-cli

format-laurus-server: ## Format laurus-server
	cargo fmt -p laurus-server

format-laurus-mcp: ## Format laurus-mcp
	cargo fmt -p laurus-mcp

format-laurus-python: ## Format laurus-python
	cargo fmt -p laurus-python

format-laurus-nodejs: ## Format laurus-nodejs
	cargo fmt -p laurus-nodejs

format-laurus-wasm: ## Format laurus-wasm
	cargo fmt -p laurus-wasm

format-laurus-ruby: ## Format laurus-ruby
	cargo fmt -p laurus-ruby

format-laurus-php: ## Format laurus-php
	cargo fmt -p laurus-php

# ── Lint ───────────────────────────────────────────────────────────────────

lint: ## Lint all crates and the binding snippets
	cargo clippy --workspace --all-targets -- -D warnings
	python3 scripts/check-binding-snippets.py

lint-laurus: ## Lint laurus
	cargo clippy -p laurus --all-targets -- -D warnings

lint-laurus-cli: ## Lint laurus-cli
	cargo clippy -p laurus-cli --all-targets -- -D warnings

lint-laurus-server: ## Lint laurus-server
	cargo clippy -p laurus-server --all-targets -- -D warnings

lint-laurus-mcp: ## Lint laurus-mcp
	cargo clippy -p laurus-mcp --all-targets -- -D warnings

lint-laurus-python: ## Lint laurus-python
	cargo clippy -p laurus-python -- -D warnings

lint-laurus-nodejs: ## Lint laurus-nodejs
	cargo clippy -p laurus-nodejs -- -D warnings

lint-laurus-wasm: ## Lint laurus-wasm (wasm32 target)
	cargo clippy -p laurus-wasm --target wasm32-unknown-unknown -- -D warnings

lint-laurus-ruby: ## Lint laurus-ruby
	cargo clippy -p laurus-ruby -- -D warnings

lint-laurus-php: ## Lint laurus-php
	cargo clippy -p laurus-php -- -D warnings

lint-snippets: ## Check binding README / docs / example snippets against the current signatures
	python3 scripts/check-binding-snippets.py

# ── Test ───────────────────────────────────────────────────────────────────

test: ## Test all crates
	cargo test --workspace

test-laurus: ## Test laurus
	cargo test -p laurus

test-laurus-cli: ## Test laurus-cli
	cargo test -p laurus-cli

test-laurus-server: ## Test laurus-server
	cargo test -p laurus-server

test-laurus-mcp: ## Test laurus-mcp
	cargo test -p laurus-mcp

test-laurus-python: venv ## Test laurus-python (Rust unit tests + Python pytest)
	cargo test -p laurus-python
	cd laurus-python && VIRTUAL_ENV=$(abspath $(PYTHON_VENV_DIR)) $(abspath $(MATURIN)) develop --quiet && $(abspath $(PYTEST)) tests/ -v

test-laurus-nodejs: ## Test laurus-nodejs
	cd laurus-nodejs && npm run build:debug && npm test

test-laurus-ruby: ## Test laurus-ruby (Rust unit tests + Ruby minitest)
	cargo test -p laurus-ruby
	cd laurus-ruby && bundle install --quiet && bundle exec rake compile && bundle exec ruby -Ilib -Itest -e 'Dir["test/test_*.rb"].each { |f| require_relative f }'

test-laurus-php: ## Test laurus-php (build + PHPUnit)
ifeq ($(shell uname -s),Darwin)
	RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup" cargo build -p laurus-php --release
	$(eval LAURUS_PHP_EXT := ../target/release/liblaurus_php.dylib)
else
	cargo build -p laurus-php --release
	$(eval LAURUS_PHP_EXT := ../target/release/liblaurus_php.so)
endif
	cd laurus-php && composer install --quiet && php -d extension=$(LAURUS_PHP_EXT) vendor/bin/phpunit tests/LaurusTest.php

test-laurus-wasm: ## Build-test laurus-wasm (wasm32 target)
	cargo build -p laurus-wasm --target wasm32-unknown-unknown

# ── Build ──────────────────────────────────────────────────────────────────

build: ## Build all crates (release, all features)
	cargo build --release --all-features

build-laurus: ## Build laurus (release)
	cargo build -p laurus --release

build-laurus-cli: ## Build laurus-cli (release)
	cargo build -p laurus-cli --release

build-laurus-cli-musl: ## Build laurus-cli for x86_64-unknown-linux-musl (static, requires cargo-zigbuild)
	cargo zigbuild --release -p laurus-cli --features embeddings-all \
	  --target x86_64-unknown-linux-musl

build-laurus-server: ## Build laurus-server (release)
	cargo build -p laurus-server --release

build-laurus-mcp: ## Build laurus-mcp (release)
	cargo build -p laurus-mcp --release

build-laurus-python: venv ## Build laurus-python wheel (release)
	cd laurus-python && VIRTUAL_ENV=$(abspath $(PYTHON_VENV_DIR)) $(abspath $(MATURIN)) build --release

build-laurus-nodejs: ## Build laurus-nodejs (release)
	cd laurus-nodejs && npm run build

build-laurus-ruby: ## Build laurus-ruby gem (release)
	cd laurus-ruby && bundle install --quiet && bundle exec rake compile

build-laurus-php: ## Build laurus-php (release)
ifeq ($(shell uname -s),Darwin)
	RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup" cargo build -p laurus-php --release
else
	cargo build -p laurus-php --release
endif

build-laurus-wasm: ## Build laurus-wasm (wasm-pack, --target web)
	cd laurus-wasm && wasm-pack build --target web --release

# ── Benchmark ──────────────────────────────────────────────────────────────
#
# BENCH (optional) selects a single criterion bench by name, e.g.
#   make bench BENCH=distance_bench
# Without BENCH, every bench registered in laurus/Cargo.toml runs.
#
# bench-baseline saves the current run as a baseline named `main`. Run it on
# the reference state (typically main).
# bench-compare diffs the next run against that saved baseline. Run it on the
# feature branch.
#
# Full workflow: docs/src/development/benchmarking.md.
BENCH ?=
BENCH_FLAG := $(if $(BENCH),--bench $(BENCH),)

bench: ## Run laurus benchmarks (BENCH=name selects one)
	cargo bench -p laurus $(BENCH_FLAG)

bench-baseline: ## Save the current results as the `main` baseline
	cargo bench -p laurus $(BENCH_FLAG) -- --save-baseline main

bench-compare: ## Compare against the `main` baseline saved earlier
	cargo bench -p laurus $(BENCH_FLAG) -- --baseline main

# ── Tag & Publish ──────────────────────────────────────────────────────────

tag: ## Make a new tag for the current version
	git tag v$(LAURUS_VERSION)
	git push origin v$(LAURUS_VERSION)

publish: ## Publish the crate to crates.io
ifeq ($(shell curl -s -XGET -H "User-Agent: $(USER_AGENT) ($(USER)@$(HOSTNAME))" https://crates.io/api/v1/crates/laurus | jq -r 'select(.versions != null) | .versions[].num' 2>/dev/null | grep -Fx "$(LAURUS_VERSION)"),)
	(cd laurus && cargo package && cargo publish)
endif
