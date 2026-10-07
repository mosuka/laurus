//! Client-server gRPC benchmark for the `SearchBatch` RPC
//! (issue [#727](https://github.com/mosuka/laurus/issues/727), follow-up to
//! Phase 3 of [#648](https://github.com/mosuka/laurus/issues/648)).
//!
//! The in-process bench added in PR #722
//! (`laurus/benches/engine_search_batch_bench.rs`) could not observe the
//! IPC / serialisation savings of batching, because it called the engine
//! directly. This bench boots the gRPC `SearchService` on an ephemeral
//! loopback port and compares, from a real `tonic` client:
//!
//! - **`search_loop/B`**: `B` sequential `Search` RPCs (B round trips).
//! - **`search_batch/B`**: one `SearchBatch` RPC carrying `B` queries (1 round trip).
//!
//! The difference isolates the per-round-trip fixed cost (HTTP/2 framing +
//! protobuf encode/decode + request dispatch) that batching amortises.
//! Over loopback the per-RTT cost is small but measurable; over a real
//! network it grows with link latency, so the gap widens.
//!
//! Run:
//!
//! ```sh
//! cargo bench -p laurus-server --bench search_batch_grpc_bench
//! ```

use std::hint::black_box;
use std::net::SocketAddr;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Server};

use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{DataValue, Document};
use laurus::{Engine, FieldOption, Schema};

use laurus_server::proto::laurus::v1::search_service_client::SearchServiceClient;
use laurus_server::proto::laurus::v1::search_service_server::SearchServiceServer;
use laurus_server::proto::laurus::v1::{SearchBatchRequest, SearchRequest};
use laurus_server::service::search::SearchService;

const CORPUS_SIZE: usize = 1_000;
const TERMS: &[&str] = &[
    "rust", "vector", "search", "engine", "index", "query", "field", "data", "system", "lexical",
];

async fn build_engine() -> Engine {
    let storage =
        StorageFactory::create(StorageConfig::Memory(Default::default())).expect("storage");
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(Default::default()))
        .build();
    let engine = Engine::new(storage, schema).await.expect("engine");

    for i in 0..CORPUS_SIZE {
        let term = TERMS[i % TERMS.len()];
        let companion = TERMS[(i + 3) % TERMS.len()];
        let doc = Document::builder()
            .add_field(
                "title",
                DataValue::Text(format!("{term} {companion} doc{i}")),
            )
            .build();
        engine
            .put_document(&format!("doc{i}"), doc)
            .await
            .expect("put_document");
    }
    engine.commit().await.expect("commit");
    engine
}

/// Boot the gRPC `SearchService` on an ephemeral loopback port and return
/// the bound address.
///
/// Each accepted connection has `TCP_NODELAY` enabled. tonic sets this
/// automatically on the sockets it accepts via `serve(addr)`, but
/// `serve_with_incoming` hands tonic a caller-supplied stream, so the
/// option must be applied here. Without it, small HTTP/2 frames hit
/// Nagle's algorithm + delayed-ACK and every RPC pays a ~40ms penalty,
/// which would otherwise dwarf the actual per-query work and distort the
/// batching comparison (issue #732).
async fn start_server(engine: Arc<RwLock<Option<Engine>>>) -> SocketAddr {
    use tokio_stream::StreamExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let svc = SearchService {
        engine,
        max_result_window: laurus_server::config::DEFAULT_MAX_RESULT_WINDOW,
    };
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener).map(|conn| {
        if let Ok(ref stream) = conn {
            let _ = stream.set_nodelay(true);
        }
        conn
    });
    tokio::spawn(async move {
        Server::builder()
            .add_service(SearchServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
            .expect("serve");
    });
    addr
}

fn make_query(i: usize) -> SearchRequest {
    SearchRequest {
        query: format!("title:{}", TERMS[i % TERMS.len()]),
        limit: 10,
        ..Default::default()
    }
}

fn bench_search_batch_grpc(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    let client = rt.block_on(async {
        let engine = Arc::new(RwLock::new(Some(build_engine().await)));
        let addr = start_server(engine).await;
        // `connect().await` waits until the server is accepting connections,
        // avoiding a race between spawn and the first RPC.
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        SearchServiceClient::new(channel)
    });

    let mut group = c.benchmark_group("search_batch_grpc");
    for b in [1_usize, 4, 16, 64] {
        group.bench_with_input(BenchmarkId::new("search_loop", b), &b, |bench, &b| {
            bench.to_async(&rt).iter(|| {
                let mut client = client.clone();
                async move {
                    let mut all = Vec::with_capacity(b);
                    for i in 0..b {
                        let resp = client.search(make_query(i)).await.expect("search");
                        all.push(resp);
                    }
                    black_box(all);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("search_batch", b), &b, |bench, &b| {
            bench.to_async(&rt).iter(|| {
                let mut client = client.clone();
                async move {
                    let queries = (0..b).map(make_query).collect();
                    let resp = client
                        .search_batch(SearchBatchRequest { queries })
                        .await
                        .expect("search_batch");
                    black_box(resp);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search_batch_grpc);
criterion_main!(benches);
