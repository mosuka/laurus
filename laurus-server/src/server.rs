//! Server bootstrap logic.
//!
//! The [`run`] function initialises logging, opens or creates the search index,
//! starts the gRPC server (and optionally the HTTP gateway), and waits for a
//! shutdown signal (`Ctrl+C` / SIGINT, or SIGTERM on Unix).

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use laurus::{CommitPolicy, WalSyncPolicy};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tonic::transport::{Endpoint, Server};
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use crate::context;
use crate::gateway;
use crate::proto::laurus::v1::{
    document_service_server::DocumentServiceServer, health_service_server::HealthServiceServer,
    index_service_server::IndexServiceServer, search_service_server::SearchServiceServer,
};
use crate::service::{
    document::DocumentService, health::HealthService, index::IndexService, search::SearchService,
};

/// Starts the server based on the given configuration.
///
/// If `http_port` is configured, both the gRPC server and HTTP Gateway are
/// started concurrently. Otherwise, only the gRPC server is started.
///
/// The function blocks until a shutdown signal (`Ctrl+C` / SIGINT, or SIGTERM
/// on Unix) is received. On shutdown it commits any pending index changes
/// before exiting.
///
/// # Arguments
///
/// * `config` - Server, index, and logging configuration.
///
/// # Returns
///
/// `Ok(())` when the server shuts down cleanly.
///
/// # Errors
///
/// Returns an error if the address cannot be parsed, the gRPC server fails to
/// start, the HTTP gateway listener cannot bind, or an existing index at the
/// configured data directory exists but fails to open.
pub async fn run(config: &Config) -> anyhow::Result<()> {
    // Initialize tracing using the RUST_LOG environment variable.
    // Falls back to "info" when RUST_LOG is not set.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Laurus server starting");
    tracing::info!("Data directory: {}", config.index.data_dir.display());

    // Install shutdown signal handling before opening the index, so a
    // signal that arrives during a long WAL replay is recorded instead of
    // using the OS default disposition -- which, for SIGTERM, would
    // terminate the process before Rust can unwind the stack and release
    // the engine lock (Issue #1480).
    let signals = ShutdownSignals::install()?;

    // Resolve the WAL durability and auto-commit policies once; they apply to
    // both the boot-time open below and any index created later via the
    // CreateIndex RPC.
    let wal_policy = config.index.wal.to_policy();
    let commit_policy = config.index.commit.to_policy();

    // Open an existing index. If none exists, start without an index. If
    // one exists but fails to open, abort startup rather than silently
    // running as if the directory were empty (Issue #1480).
    let engine = open_index_at_startup(&config.index.data_dir, wal_policy, commit_policy).await?;

    let engine = Arc::new(RwLock::new(engine));
    let data_dir = config.index.data_dir.clone();

    let health_service = HealthService;
    let document_service = DocumentService {
        engine: engine.clone(),
    };
    let index_service = IndexService {
        engine: engine.clone(),
        data_dir,
        wal_policy,
        commit_policy,
    };
    let search_service = SearchService {
        engine: engine.clone(),
        max_result_window: config.server.max_result_window,
    };

    let grpc_addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    tracing::info!("gRPC server listening on {grpc_addr}");

    let grpc_server = Server::builder()
        .add_service(HealthServiceServer::new(health_service))
        .add_service(DocumentServiceServer::new(document_service))
        .add_service(IndexServiceServer::new(index_service))
        .add_service(SearchServiceServer::new(search_service));

    if let Some(http_port) = config.server.http_port {
        // Also start the gRPC Gateway (HTTP server) concurrently.
        let channel = Endpoint::from_shared(format!("http://127.0.0.1:{}", config.server.port))?
            .connect_lazy();
        let gateway_state = gateway::GatewayState::new(channel);
        let router = gateway::create_router(gateway_state);

        let http_addr: SocketAddr = format!("{}:{}", config.server.host, http_port).parse()?;
        let listener = TcpListener::bind(http_addr).await?;
        tracing::info!("HTTP gateway listening on {http_addr}");

        // Run the gRPC server and HTTP Gateway concurrently, shutting both down on Ctrl+C.
        tokio::select! {
            result = grpc_server.serve(grpc_addr) => {
                if let Err(e) = result {
                    tracing::error!("gRPC server error: {e}");
                }
            }
            result = axum::serve(listener, router) => {
                if let Err(e) = result {
                    tracing::error!("HTTP gateway error: {e}");
                }
            }
            _ = shutdown_signal(engine.clone(), signals) => {}
        }
    } else {
        // Start only the gRPC server (legacy behavior).
        grpc_server
            .serve_with_shutdown(grpc_addr, shutdown_signal(engine.clone(), signals))
            .await?;
    }

    Ok(())
}

/// Opens the index at `data_dir` if the directory holds one.
///
/// Returns `Ok(None)` when `data_dir` has no index yet -- the server then
/// starts without one, and a client creates one later via the `CreateIndex`
/// RPC. Returns `Err` when an index exists (a `schema.toml` is present) but
/// fails to open -- a stale lock, an unreadable schema, a failed WAL replay,
/// etc. -- so the caller aborts startup instead of presenting a non-empty
/// directory as if it had no index (Issue #1480).
async fn open_index_at_startup(
    data_dir: &Path,
    wal_policy: WalSyncPolicy,
    commit_policy: CommitPolicy,
) -> anyhow::Result<Option<laurus::Engine>> {
    if !context::index_exists(data_dir) {
        tracing::info!("No existing index found. Use CreateIndex RPC to create one.");
        return Ok(None);
    }

    match context::open_index(data_dir, wal_policy, commit_policy).await {
        Ok(engine) => {
            tracing::info!("Opened existing index");
            Ok(Some(engine))
        }
        Err(e) => {
            tracing::error!(
                "Failed to open the existing index at {}: {e:#}",
                data_dir.display()
            );
            Err(e.context(format!(
                "Failed to open the existing index at {}",
                data_dir.display()
            )))
        }
    }
}

/// Shutdown signals this process listens for (Issue #1480).
///
/// `docker stop`, Kubernetes and systemd send SIGTERM, not SIGINT, to ask a
/// process to shut down. Without a handler for it, the OS default
/// disposition terminates the process immediately -- before Rust can
/// unwind the stack and release the engine lock -- so SIGTERM is handled
/// the same way as Ctrl+C (SIGINT) here.
struct ShutdownSignals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
}

impl ShutdownSignals {
    /// Installs the handlers for every signal this process treats as a
    /// shutdown request.
    fn install() -> anyhow::Result<Self> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            Ok(Self {
                interrupt: signal(SignalKind::interrupt())?,
                terminate: signal(SignalKind::terminate())?,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self {})
        }
    }

    /// Waits for the next shutdown signal and returns its name for logging.
    ///
    /// Non-Unix platforms only receive Ctrl+C; there is no SIGTERM
    /// equivalent wired up here.
    async fn wait(&mut self) -> &'static str {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.interrupt.recv() => "SIGINT",
                _ = self.terminate.recv() => "SIGTERM",
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
            "Ctrl+C"
        }
    }
}

/// Waits for a shutdown signal and commits pending changes before exiting.
async fn shutdown_signal(
    engine: Arc<RwLock<Option<laurus::Engine>>>,
    mut signals: ShutdownSignals,
) {
    let signal_name = signals.wait().await;

    tracing::info!("{signal_name} received, committing pending changes...");

    let guard = engine.read().await;
    if let Some(engine) = guard.as_ref() {
        if let Some(reason) = engine.write_stop_reason() {
            // The engine refuses commits once writes stopped (Issue #1492);
            // its write-ahead log keeps every pending change for the replay.
            tracing::warn!(
                "Not committing on shutdown, writes are stopped: {reason}. The write-ahead log is replayed on the next start."
            );
        } else if let Err(e) = engine.commit().await {
            tracing::error!("Failed to commit on shutdown: {e}");
        } else {
            tracing::info!("Committed successfully");
        }
    }

    tracing::info!("Server shutting down");
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use laurus::{CommitPolicy, Document, Schema, TextOption, WalSyncPolicy};
    use tokio::sync::RwLock;

    use super::*;

    /// Issue #1480: a start against an empty directory must behave exactly
    /// as before -- no error, no index.
    #[tokio::test]
    async fn open_index_at_startup_returns_none_when_no_index_exists() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");

        let engine =
            open_index_at_startup(&data_dir, WalSyncPolicy::default(), CommitPolicy::Manual)
                .await
                .unwrap();

        assert!(engine.is_none());
    }

    /// Issue #1480: a start against a directory with a valid index must
    /// still open it.
    #[tokio::test]
    async fn open_index_at_startup_opens_an_existing_index() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        context::create_index(
            &data_dir,
            &Schema::new(),
            WalSyncPolicy::default(),
            CommitPolicy::Manual,
        )
        .await
        .unwrap();

        let engine =
            open_index_at_startup(&data_dir, WalSyncPolicy::default(), CommitPolicy::Manual)
                .await
                .unwrap();

        assert!(engine.is_some());
    }

    /// Issue #1480: `schema.toml` existing but the engine failing to open
    /// (here, a stale `engine.lock` left by a crash) must surface as an
    /// `Err` naming both the real cause and the data directory, not the
    /// `No existing index found` message reserved for a directory without
    /// an index.
    #[tokio::test]
    async fn open_index_at_startup_fails_on_a_stale_lock_instead_of_reporting_no_index() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        let engine = context::create_index(
            &data_dir,
            &Schema::new(),
            WalSyncPolicy::default(),
            CommitPolicy::Manual,
        )
        .await
        .unwrap();
        drop(engine); // Releases the real lock...

        // ...then place a stale one by hand, simulating a crash that left
        // `store/engine.lock` behind.
        std::fs::write(data_dir.join("store").join("engine.lock"), b"").unwrap();

        let err = open_index_at_startup(&data_dir, WalSyncPolicy::default(), CommitPolicy::Manual)
            .await
            .map(|_engine| ())
            .unwrap_err();

        let message = format!("{err:#}");
        assert!(message.contains("already locked"), "got: {message}");
        assert!(
            message.contains(&data_dir.display().to_string()),
            "got: {message}"
        );
    }

    /// Issue #1480: SIGTERM must resolve `shutdown_signal` the same way
    /// Ctrl+C does, reaching the commit and (once the last `Engine` handle
    /// is dropped, as happens when `run()` returns) the lock release --
    /// instead of the OS default disposition killing the process first.
    #[tokio::test]
    async fn shutdown_signal_commits_and_releases_the_lock_on_sigterm() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        let schema = Schema::builder()
            .add_text_field("title", TextOption::default())
            .build();

        let engine = context::create_index(
            &data_dir,
            &schema,
            WalSyncPolicy::default(),
            CommitPolicy::Manual,
        )
        .await
        .unwrap();
        engine
            .add_document(
                "doc1",
                Document::builder().add_text("title", "hello").build(),
            )
            .await
            .unwrap();

        let engine = Arc::new(RwLock::new(Some(engine)));
        let signals = ShutdownSignals::install().unwrap();

        let shutdown = tokio::spawn(shutdown_signal(engine.clone(), signals));

        // Safe: raises SIGTERM against this process. `ShutdownSignals::install`
        // above already replaced the OS default disposition for it, so this
        // reaches `shutdown_signal` instead of killing the test binary.
        unsafe {
            libc::raise(libc::SIGTERM);
        }

        tokio::time::timeout(Duration::from_secs(5), shutdown)
            .await
            .expect("shutdown_signal must return once SIGTERM is delivered")
            .unwrap();

        // `run()` drops its only `Arc<RwLock<Option<Engine>>>` when it
        // returns; emulate that here so the engine's `Drop` -- which
        // releases the storage lock -- actually runs.
        *engine.write().await = None;

        let lock_path = data_dir.join("store").join("engine.lock");
        assert!(
            !lock_path.exists(),
            "engine.lock must be gone once the engine holding it is dropped"
        );

        let reopened =
            context::open_index(&data_dir, WalSyncPolicy::default(), CommitPolicy::Manual)
                .await
                .unwrap();
        assert_eq!(
            reopened.stats().unwrap().document_count,
            1,
            "the document added before SIGTERM must have been committed on shutdown"
        );
    }
}
