//! Server bootstrap logic.
//!
//! The [`run`] function initialises logging, opens or creates the search index,
//! starts the gRPC server (and optionally the HTTP gateway), and waits for a
//! shutdown signal (`Ctrl+C`).

use std::net::SocketAddr;
use std::sync::Arc;

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
/// The function blocks until a shutdown signal (`Ctrl+C`) is received.
/// On shutdown it commits any pending index changes before exiting.
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
/// start, or the HTTP gateway listener cannot bind.
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

    // Resolve the WAL durability and auto-commit policies once; they apply to
    // both the boot-time open below and any index created later via the
    // CreateIndex RPC.
    let wal_policy = config.index.wal.to_policy();
    let commit_policy = config.index.commit.to_policy();

    // Open an existing index. If none exists, start without an index.
    let engine = match context::open_index(&config.index.data_dir, wal_policy, commit_policy).await
    {
        Ok(engine) => {
            tracing::info!("Opened existing index");
            Some(engine)
        }
        Err(_) => {
            tracing::info!("No existing index found. Use CreateIndex RPC to create one.");
            None
        }
    };

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
            _ = shutdown_signal(engine.clone()) => {}
        }
    } else {
        // Start only the gRPC server (legacy behavior).
        grpc_server
            .serve_with_shutdown(grpc_addr, shutdown_signal(engine.clone()))
            .await?;
    }

    Ok(())
}

/// Waits for a shutdown signal (Ctrl+C) and commits pending changes before exiting.
async fn shutdown_signal(engine: Arc<RwLock<Option<laurus::Engine>>>) {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");

    tracing::info!("Shutdown signal received, committing pending changes...");

    let guard = engine.read().await;
    if let Some(engine) = guard.as_ref() {
        if let Err(e) = engine.commit().await {
            tracing::error!("Failed to commit on shutdown: {e}");
        } else {
            tracing::info!("Committed successfully");
        }
    }

    tracing::info!("Server shutting down");
}
