# Installation

## From RubyGems

```bash
gem install laurus
```

Or add it to your `Gemfile`:

```ruby
gem "laurus"
```

Then run:

```bash
bundle install
```

## From source

Building from source requires a Rust toolchain (1.89 or later) and [rb_sys](https://github.com/oxidize-rb/rb-sys).

```bash
# Clone the repository
git clone https://github.com/mosuka/laurus.git
cd laurus/laurus-ruby

# Install dependencies
bundle install

# Compile the native extension
bundle exec rake compile

# Or install the gem locally
gem build laurus.gemspec
gem install laurus-*.gem
```

## Verify

```ruby
require "laurus"
index = Laurus::Index.new
puts index  # Index()
```

## Requirements

- Ruby 3.1 or later
- Rust toolchain (automatically invoked during gem install via `rb_sys`)
- No runtime dependencies beyond the compiled native extension
