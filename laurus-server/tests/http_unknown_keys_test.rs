//! Request bodies with a key the HTTP gateway does not read (Issue #1706).
//!
//! The gateway reads each body by hand and used to drop any key it did not
//! read, so `"indexd": false` left a field indexed, a misspelled
//! `"dry_rn": true` applied a field change instead of trying it, a bulk
//! entry's `"fileds"` vanished, and a search's `"limt"` left the default
//! limit. Such a key is now a 400 that names it, and nothing is written.
//!
//! The gateway forwards each call to the gRPC services, so these requests
//! go through both, over a gRPC connection on an ephemeral port.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Endpoint, Server};
use tower::ServiceExt;

use laurus::{CommitPolicy, WalSyncPolicy};
use laurus_server::config::DEFAULT_MAX_RESULT_WINDOW;
use laurus_server::gateway::{self, GatewayState};
use laurus_server::proto::laurus::v1::document_service_server::DocumentServiceServer;
use laurus_server::proto::laurus::v1::index_service_server::IndexServiceServer;
use laurus_server::proto::laurus::v1::search_service_server::SearchServiceServer;
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// Serve the index, document and search services over `data_dir`, with no
/// index yet, on an ephemeral port, and return the gateway router in front
/// of them.
async fn gateway(data_dir: PathBuf) -> Router {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine = Arc::new(RwLock::new(None));
    let index = IndexService {
        engine: engine.clone(),
        data_dir,
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    };
    let documents = DocumentService {
        engine: engine.clone(),
    };
    let search = SearchService {
        engine,
        max_result_window: DEFAULT_MAX_RESULT_WINDOW,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(index))
            .add_service(DocumentServiceServer::new(documents))
            .add_service(SearchServiceServer::new(search))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    gateway::create_router(GatewayState::new(channel))
}

/// Send `body` as JSON to `uri` and return the status and the body: its
/// JSON, or its text when it is not JSON.
async fn send(router: &Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

/// Send `body` and assert a 200.
async fn ok(router: &Router, method: &str, uri: &str, body: Value) -> Value {
    let (status, body) = send(router, method, uri, body).await;
    assert_eq!(status, StatusCode::OK, "{method} {uri}: {body}");
    body
}

/// Send `body`, assert a 400 whose message contains each of `parts`, and
/// return the message.
async fn rejected(router: &Router, method: &str, uri: &str, body: Value, parts: &[&str]) -> String {
    let (status, response) = send(router, method, uri, body).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{method} {uri}: {response}"
    );
    let message = response["error"]["message"]
        .as_str()
        .unwrap_or("")
        .to_string();
    for part in parts {
        assert!(
            message.contains(part),
            "{method} {uri}: missing {part:?}: {message}"
        );
    }
    message
}

/// Create an index whose only field is `body`, a stored text field.
async fn index_with_body(router: &Router) {
    ok(
        router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"body": {"text": {"stored": true}}}}}),
    )
    .await;
}

#[tokio::test]
async fn creating_an_index_with_an_unknown_key_creates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("idx");
    let router = gateway(data_dir.clone()).await;

    rejected(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"body": {"text": {"indexd": false}}}}}),
        &[
            "field \"body\"",
            "unknown key \"indexd\" in the text option",
        ],
    )
    .await;
    rejected(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"body": {"text": {}}}}, "dry_run": true}),
        &["unknown key \"dry_run\" in the request body"],
    )
    .await;
    assert!(
        !data_dir.join("schema.toml").exists(),
        "a rejected create wrote a schema"
    );
}

#[tokio::test]
async fn a_misspelled_dry_run_does_not_apply_the_change() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("idx")).await;
    index_with_body(&router).await;

    rejected(
        &router,
        "PATCH",
        "/v1/schema/fields/body",
        json!({"field_option": {"text": {"stored": false}}, "dry_rn": true}),
        &["unknown key \"dry_rn\" in the request body", "\"dry_run\""],
    )
    .await;
    let schema = ok(&router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(
        schema["schema"]["fields"]["body"]["text"]["stored"],
        json!(true),
        "the change was applied: {schema}"
    );
}

#[tokio::test]
async fn adding_a_field_with_an_unknown_key_adds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("idx")).await;
    index_with_body(&router).await;

    rejected(
        &router,
        "POST",
        "/v1/schema/fields",
        json!({"name": "n", "field_option": {"integer": {}}, "reindx": true}),
        &["unknown key \"reindx\" in the request body"],
    )
    .await;
    let schema = ok(&router, "GET", "/v1/schema", json!({})).await;
    assert!(
        schema["schema"]["fields"].get("n").is_none(),
        "the field was added: {schema}"
    );
}

#[tokio::test]
async fn documents_with_an_unknown_key_are_not_stored() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("idx")).await;
    index_with_body(&router).await;

    for method in ["PUT", "POST"] {
        rejected(
            &router,
            method,
            "/v1/documents/1",
            json!({"fields": {"body": "hello"}, "id": "other"}),
            &["unknown key \"id\" in the request body (expected \"fields\")"],
        )
        .await;
    }
    rejected(
        &router,
        "POST",
        "/v1/documents:bulk",
        json!({"documents": [
            {"id": "2", "fields": {"body": "hello"}},
            {"id": "3", "fields": {"body": "again"}, "fileds": {"body": "lost"}}
        ]}),
        &["documents[1]: unknown key \"fileds\" in the document entry"],
    )
    .await;
    rejected(
        &router,
        "POST",
        "/v1/documents:bulk",
        json!({"documents": [{"id": "4", "fields": {"body": "hello"}}], "dryrun": true}),
        &["unknown key \"dryrun\" in the request body (expected \"documents\")"],
    )
    .await;

    ok(&router, "POST", "/v1/commit", json!({})).await;
    for id in ["1", "2", "3", "4"] {
        let found = ok(&router, "GET", &format!("/v1/documents/{id}"), json!({})).await;
        assert_eq!(
            found["documents"],
            json!([]),
            "document {id} was stored: {found}"
        );
    }
}

#[tokio::test]
async fn searches_with_an_unknown_key_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("idx")).await;
    index_with_body(&router).await;

    for uri in ["/v1/search", "/v1/search/stream"] {
        rejected(
            &router,
            "POST",
            uri,
            json!({"query": "body:hello", "limt": 1}),
            &["unknown key \"limt\" in the search request"],
        )
        .await;
    }
    rejected(
        &router,
        "POST",
        "/v1/search",
        json!({"query": "body:hello", "lexical_params": {"min_scor": 1000}}),
        &["unknown key \"min_scor\" in lexical_params"],
    )
    .await;
    rejected(
        &router,
        "POST",
        "/v1/search",
        json!({"query": "body:hello", "fusion": {"rrf": {"k": 60, "kk": 1}}}),
        &["unknown key \"kk\" in fusion.rrf"],
    )
    .await;
    rejected(
        &router,
        "POST",
        "/v1/search/batch",
        json!({"queries": [{"query": "body:hello"}], "limit": 1}),
        &["unknown key \"limit\" in the request body (expected \"queries\")"],
    )
    .await;
    rejected(
        &router,
        "POST",
        "/v1/search/batch",
        json!({"queries": [{"query": "body:hello", "limt": 1}]}),
        &["queries[0]: unknown key \"limt\" in the search request"],
    )
    .await;
}

/// `null` is the same as omitting a key, so an unknown key set to `null`
/// is not reported.
#[tokio::test]
async fn an_unknown_key_set_to_null_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("idx")).await;
    ok(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"body": {"text": {"stored": true, "indexd": null}}}}}),
    )
    .await;
    ok(
        &router,
        "POST",
        "/v1/search",
        json!({"query": "body:hello", "limt": null}),
    )
    .await;
}

/// The schema `GET /v1/schema` returns, with fields of many types, an
/// analyzer and an embedder, can be sent back to `POST /v1/index`: the
/// gateway takes every key it writes.
#[tokio::test]
async fn the_schema_the_gateway_returns_can_be_sent_back() {
    let dir = tempfile::tempdir().unwrap();
    let first = gateway(dir.path().join("first")).await;
    let flags = json!({"indexed": true, "stored": true});
    ok(
        &first,
        "POST",
        "/v1/index",
        json!({"schema": {
            "fields": {
                "body": {"text": {"stored": true, "analyzer": "a", "position_increment_gap": 10}},
                "count": {"integer": flags},
                "price": {"float": flags},
                "flag": {"boolean": flags},
                "at": {"date_time": flags},
                "loc": {"geo": flags},
                "pos": {"geo3d": flags},
                "blob": {"bytes": {"stored": true}},
                "h": {"hnsw": {"dimension": 4, "distance": "cosine", "m": 8,
                               "quantizer": {"method": "scalar_8bit"}, "rerank_storage": "F32",
                               "embedder": "e", "default_ef_search": 40}},
                "fl": {"flat": {"dimension": 4, "distance": "euclidean"}},
                "iv": {"ivf": {"dimension": 4, "distance": "cosine", "n_clusters": 2, "n_probe": 1}},
                "mv": {"multi_vector": {"dimension": 4, "distance": "cosine", "storage": "f16"}}
            },
            "default_fields": ["body"],
            "analyzers": {"a": {
                "char_filters": [{"type": "unicode_normalization", "form": "nfkc"}],
                "tokenizer": {"type": "whitespace"},
                "token_filters": [{"type": "lowercase"}]
            }},
            "embedders": {"e": {"type": "precomputed"}},
            "dynamic_field_policy": "strict"
        }}),
    )
    .await;
    let schema = ok(&first, "GET", "/v1/schema", json!({})).await["schema"].clone();

    let second = gateway(dir.path().join("second")).await;
    ok(&second, "POST", "/v1/index", json!({"schema": schema})).await;
    let again = ok(&second, "GET", "/v1/schema", json!({})).await["schema"].clone();
    assert_eq!(again, schema, "the schema changed on the way back");
}
