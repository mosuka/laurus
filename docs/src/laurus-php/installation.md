# Installation

`laurus-php` is a PHP extension written in Rust. It is not published to PECL and Composer is not used as the installation channel — the `composer.json` in the repository only declares dev-time test dependencies (PHPUnit). Build the shared library from source with Cargo, drop it next to your PHP installation's other extensions, and enable it via `php.ini`.

## From source

Building from source requires a Rust toolchain (1.89 or later) and PHP 8.1 or later with development headers.

```bash
# Clone the repository
git clone https://github.com/mosuka/laurus.git
cd laurus/laurus-php

# Build the native extension
cargo build --release

# Copy the shared library to the PHP extensions directory
# (the exact path depends on your OS and PHP version)
cp ../target/release/liblaurus_php.so $(php -r "echo ini_get('extension_dir');")
```

Then add the extension to your `php.ini`:

```ini
extension=laurus_php.so
```

Alternatively, you can load the extension on the command line:

```bash
php -d extension=liblaurus_php.so your_script.php
```

## Verify

```php
<?php

use Laurus\Index;

$index = new Index();
echo $index;  // Index()
```

## Requirements

- PHP 8.1 or later with development headers (`php-dev` / `php-devel`)
- Rust toolchain 1.89 or later with Cargo
- No runtime dependencies beyond the compiled native extension
