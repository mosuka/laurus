//! The HTTP gateway passes analyzer components through as sent, and
//! `GET /v1/schema` returns them in the shape and with the types a request
//! uses (Issue #1478).
//!
//! The gateway used to turn every parameter into a string for the gRPC
//! services: a stop filter's `words` was joined with commas and split again,
//! so `"new,york"` became two words and `" padded "` lost its spaces; an
//! array item or a mapping value that was not a string was dropped; and the
//! response returned `"min_gram": "2"`, `"words": "the,a"`, and a mapping's
//! entries at the top level of the component, where a key `type` replaced
//! the component's type. A parameter of the wrong type is now a 400 from the
//! server's serde, which reads it by the rules `schema.toml` is read by.
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
use laurus_server::gateway::{self, GatewayState};
use laurus_server::proto::laurus::v1::index_service_server::IndexServiceServer;
use laurus_server::service::index::IndexService;

/// Serve an `IndexService` over `data_dir`, with no index yet, on an
/// ephemeral port, and return the gateway router in front of it.
async fn gateway(data_dir: PathBuf) -> Router {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let index = IndexService {
        engine: Arc::new(RwLock::new(None)),
        data_dir,
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(index))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    gateway::create_router(GatewayState::new(channel))
}

/// Send `body` as JSON to `uri` and return the status and the JSON body.
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

/// A schema with one text field and `analyzers`.
fn schema(analyzers: Value) -> Value {
    json!({
        "schema": {
            "fields": {"body": {"text": {"indexed": true, "stored": true}}},
            "analyzers": analyzers
        }
    })
}

/// Analyzers that use every component type except `lindera`, which needs a
/// dictionary on disk, each with every key the response writes, so that
/// the response is the request: stop words with commas and spaces, an empty
/// stop list, a mapping with a key `type`, whole numbers and a boost of 0.1.
fn analyzers_sent() -> Value {
    json!({
        "full": {
            "char_filters": [
                {"type": "unicode_normalization", "form": "nfd"},
                {"type": "pattern_replace", "pattern": "a+", "replacement": "b"},
                {"type": "mapping", "mapping": {"type": "t", "ph": "f"}},
                {"type": "japanese_iteration_mark", "kanji": false, "kana": true}
            ],
            "tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3},
            "token_filters": [
                {"type": "lowercase"},
                {"type": "stop", "words": ["new,york", " padded ", "x"]},
                {"type": "stop", "words": []},
                {"type": "stem", "stem_type": "simple"},
                {"type": "boost", "boost": 0.1},
                {"type": "limit", "limit": 5},
                {"type": "strip"},
                {"type": "remove_empty"},
                {"type": "flatten_graph"}
            ]
        },
        "regex": {"tokenizer": {"type": "regex", "pattern": "[a-z]+", "gaps": true}},
        "whitespace": {"tokenizer": {"type": "whitespace"}},
        "unicode_word": {"tokenizer": {"type": "unicode_word"}},
        "whole": {"tokenizer": {"type": "whole"}}
    })
}

#[tokio::test]
async fn components_come_back_as_sent_and_post_back_to_the_same_schema() {
    let first = tempfile::tempdir().unwrap();
    let first_dir = first.path().join("data");
    let router = gateway(first_dir.clone()).await;
    let (status, body) = send(&router, "POST", "/v1/index", schema(analyzers_sent())).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, response) = send(&router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["schema"]["analyzers"], analyzers_sent());
    let min_gram = &response["schema"]["analyzers"]["full"]["tokenizer"]["min_gram"];
    assert!(min_gram.is_u64(), "{min_gram}");

    // The response posted to another server gives the same schema.toml.
    let second = tempfile::tempdir().unwrap();
    let second_dir = second.path().join("data");
    let router = gateway(second_dir.clone()).await;
    let (status, body) = send(&router, "POST", "/v1/index", response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first_toml = std::fs::read_to_string(first_dir.join("schema.toml")).unwrap();
    let second_toml = std::fs::read_to_string(second_dir.join("schema.toml")).unwrap();
    assert_eq!(first_toml, second_toml);
}

/// A key left out, or `null`, gets the default `schema.toml` gives it, and
/// the response writes the default.
#[tokio::test]
async fn omitted_and_null_parameters_get_the_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    let analyzers = json!({
        "a": {
            "char_filters": [{"type": "japanese_iteration_mark", "kanji": null}],
            "tokenizer": {"type": "regex"},
            "token_filters": [{"type": "stop", "words": null}, {"type": "stem"}]
        }
    });
    let (status, body) = send(&router, "POST", "/v1/index", schema(analyzers)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, response) = send(&router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(
        response["schema"]["analyzers"],
        json!({
            "a": {
                "char_filters": [{"type": "japanese_iteration_mark", "kanji": true, "kana": true}],
                "tokenizer": {"type": "regex", "pattern": "\\w+", "gaps": false},
                "token_filters": [{"type": "stop"}, {"type": "stem"}]
            }
        })
    );
}

/// A component the core types cannot read is a 400 naming the analyzer and
/// the component, and creates nothing.
#[tokio::test]
async fn a_component_of_the_wrong_shape_is_a_400_and_creates_nothing() {
    let whitespace = json!({"type": "whitespace"});
    let analyzer = |char_filters: Value, tokenizer: Value, token_filters: Value| json!({"a": {"char_filters": char_filters, "tokenizer": tokenizer, "token_filters": token_filters}});
    // Nested deeper than prost's recursion limit allows, which the
    // gateway's depth limit turns into a 400 rather than a 500.
    let mut deep = json!("x");
    for _ in 0..40 {
        deep = json!({"k": deep});
    }
    let cases = [
        (
            analyzer(
                json!([]),
                json!({"type": "ngram", "min_gram": "2", "max_gram": 3}),
                json!([]),
            ),
            "Analyzer 'a': tokenizer: invalid type: string \"2\", expected usize",
        ),
        (
            analyzer(
                json!([]),
                whitespace.clone(),
                json!([{"type": "stop", "words": "a,b"}]),
            ),
            "Analyzer 'a': token_filters[0]: invalid type: string \"a,b\", expected a sequence",
        ),
        (
            analyzer(
                json!([]),
                whitespace.clone(),
                json!([{"type": "lowercase"}, {"type": "stop", "words": [1, "a"]}]),
            ),
            "Analyzer 'a': token_filters[1]: invalid type: integer `1`, expected a string",
        ),
        (
            analyzer(
                json!([{"type": "mapping", "mapping": {"a": 1}}]),
                whitespace.clone(),
                json!([]),
            ),
            "Analyzer 'a': char_filters[0]: invalid type: integer `1`, expected a string",
        ),
        (
            analyzer(
                json!([{"type": "japanese_iteration_mark", "kanji": "False"}]),
                whitespace.clone(),
                json!([]),
            ),
            "Analyzer 'a': char_filters[0]: invalid type: string \"False\", expected a boolean",
        ),
        (
            analyzer(
                json!([{"type": "unicode_normalization"}]),
                whitespace.clone(),
                json!([]),
            ),
            "Analyzer 'a': char_filters[0]: missing field `form`",
        ),
        (
            analyzer(json!([]), json!({"pattern": "x"}), json!([])),
            "Analyzer 'a': tokenizer: missing field `type`",
        ),
        (
            analyzer(
                json!([{"type": "mapping", "mapping": deep}]),
                whitespace.clone(),
                json!([]),
            ),
            "analyzer \"a\": char_filters[0]: component config must not nest more than 8 levels deep",
        ),
    ];
    // Every request that was not rejected as expected, so a failure lists
    // all of them rather than the first.
    let mut wrong = Vec::new();
    for (analyzers, expected) in cases {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let router = gateway(data_dir.clone()).await;
        let (status, body) = send(&router, "POST", "/v1/index", schema(analyzers.clone())).await;
        if status != StatusCode::BAD_REQUEST || body["error"]["message"] != json!(expected) {
            wrong.push(format!("{analyzers}: {status} {body}"));
        }
        if data_dir.exists() {
            wrong.push(format!("{analyzers}: an index was created"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
