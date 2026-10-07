//! gRPC server launcher for the laurus search engine.
//!
//! Builds a server configuration from an optional TOML config file merged with
//! CLI flags and environment variables, then starts the `laurus-server` gRPC
//! server (with an optional HTTP gateway).

use std::path::Path;

use anyhow::Result;

use crate::cli::ServeCommand;

/// Start the gRPC server (and optional HTTP gateway).
///
/// Loads a base configuration from the TOML file specified in `cmd.config`
/// (or uses defaults), applies any overrides from CLI flags / environment
/// variables, sets the index index directory, and delegates to the
/// `laurus-server` runtime.
///
/// # Arguments
///
/// * `cmd` - Parsed [`ServeCommand`] containing optional config path, host,
///   port, and HTTP port.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Returns
///
/// This function runs indefinitely (until the server is shut down) and
/// returns `Ok(())` on graceful termination.
///
/// # Errors
///
/// Returns an error if:
/// - The configuration file cannot be read or parsed.
/// - The server fails to start or encounters a fatal runtime error.
pub async fn run(cmd: ServeCommand, index_dir: &Path) -> Result<()> {
    let mut config = match &cmd.config {
        Some(path) => laurus_server::config::Config::from_file(path)?,
        None => laurus_server::config::Config::default(),
    };

    // Override with CLI arguments / environment variables when provided.
    if let Some(host) = cmd.host {
        config.server.host = host;
    }
    if let Some(port) = cmd.port {
        config.server.port = port;
    }
    if cmd.http_port.is_some() {
        config.server.http_port = cmd.http_port;
    }
    if let Some(max_result_window) = cmd.max_result_window {
        config.server.max_result_window = max_result_window;
    }
    config.index.data_dir = index_dir.to_path_buf();

    laurus_server::server::run(&config).await
}
