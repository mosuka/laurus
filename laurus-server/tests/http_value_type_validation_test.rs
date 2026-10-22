//! The HTTP gateway rejects a value of the wrong type for every key it
//! reads with a 400 naming the key, and nothing is written or searched
//! (Issue #1524). Such a value used to be read as omitted or dropped:
//! `"order": "DESC"` sorted ascending, `{"text": 1}` created a field that is
//! neither indexed nor stored, and `"query": 5` turned a hybrid search into
//! a vector-only one. A missing key and `null` keep their defaults.
//!
//! The gateway forwards each call to the gRPC services and maps the status
//! they return, so these requests go through both, over a gRPC connection
//! to the services on an ephemeral port.

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

/// Serve an `IndexService`, a `DocumentService` and a `SearchService` over
/// `data_dir`, with no index yet, on an ephemeral port, and return the
/// gateway router in front of them.
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
    let document = DocumentService {
        engine: engine.clone(),
    };
    let search = SearchService {
        engine,
        max_result_window: DEFAULT_MAX_RESULT_WINDOW,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(index))
            .add_service(DocumentServiceServer::new(document))
            .add_service(SearchServiceServer::new(search))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    gateway::create_router(GatewayState::new(channel))
}

/// Send `body` as JSON to `uri` and return the status and the body: its
/// JSON, or its text when it is not JSON (a search stream).
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

/// Create an index with `fields` through the gateway.
async fn create_index(router: &Router, fields: Value) {
    let schema = json!({"schema": {"fields": fields}});
    let (status, body) = send(router, "POST", "/v1/index", schema).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// The schema `GET /v1/schema` returns.
async fn schema(router: &Router) -> Value {
    let (status, body) = send(router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

/// The fields of the index the search tests use: two stored text fields,
/// a sortable integer and a 2-dimensional vector.
fn search_fields() -> Value {
    json!({
        "title": {"text": {"indexed": true, "stored": true}},
        "body": {"text": {"indexed": true, "stored": true}},
        "year": {"integer": {"indexed": true, "stored": true, "doc_values": true}},
        "vec": {"flat": {"dimension": 2}}
    })
}

/// `POST /v1/index` with a value of the wrong type is a 400 that names the
/// key, and creates nothing.
#[tokio::test]
async fn create_index_answers_400_for_a_value_of_the_wrong_type() {
    let title = json!({"title": {"text": {"indexed": true}}});
    let japanese = json!({"language": "japanese", "dict": "d", "mode": 1});
    let ws =
        |key: &str, value: Value| json!({"ws": {"tokenizer": {"type": "whitespace"}, key: value}});
    let cases = [
        (json!(5), "schema must be an object"),
        (
            json!({"fields": {"title": {"text": 1}}}),
            "field \"title\": text option must be an object",
        ),
        (
            json!({"fields": {"title": {"text": null}}}),
            "field \"title\": text option must be an object",
        ),
        (
            json!({"fields": {"vec": {"flat": 1}}}),
            "field \"vec\": flat option must be an object",
        ),
        (
            json!({"fields": {"title": {"text": {}, "integer": {}}}}),
            "field \"title\": a field option must have one type, got \"text\" and \"integer\"",
        ),
        (
            json!({"fields": {"vec": {"flat": {"dimension": 2, "base_weight": "2"}}}}),
            "field \"vec\": base_weight must be a number",
        ),
        (
            json!({"fields": {"vec": {"flat": {"dimension": 2, "embedder": 1}}}}),
            "field \"vec\": embedder must be a string",
        ),
        (
            json!({"fields": {"vec": {"hnsw": {"dimension": 2, "pq_codebook_path": 1}}}}),
            "field \"vec\": pq_codebook_path must be a string",
        ),
        (
            json!({"fields": {"title": {"text": {"analyzer": japanese}}}}),
            "field \"title\": analyzer: mode must be a string",
        ),
        (
            json!({"fields": title, "default_fields": [1, "title"]}),
            "default_fields must be an array of strings",
        ),
        (
            json!({"fields": title, "default_fields": "title"}),
            "default_fields must be an array of strings",
        ),
        (
            json!({"fields": title, "analyzers": "x"}),
            "analyzers must be an object",
        ),
        (
            json!({"fields": title, "embedders": "x"}),
            "embedders must be an object",
        ),
        (
            json!({"fields": title, "analyzers": ws("token_filters", json!("lowercase"))}),
            "analyzer \"ws\": token_filters must be an array",
        ),
    ];
    // Every request that was not rejected as expected, so a failure lists
    // all of them rather than the first.
    let mut wrong = Vec::new();
    for (schema, expected) in cases {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let router = gateway(data_dir.clone()).await;

        let request = json!({ "schema": schema });
        let (status, body) = send(&router, "POST", "/v1/index", request).await;
        if status != StatusCode::BAD_REQUEST || body["error"]["message"] != json!(expected) {
            wrong.push(format!("{schema}: {status} {body}"));
        }
        if data_dir.exists() {
            wrong.push(format!("{schema}: an index was created"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `POST /v1/schema/fields` and `PATCH /v1/schema/fields/:name` with a
/// field option of the wrong type are a 400 that names it, and the schema
/// is unchanged. `{"text": 1}` used to add a field that is neither indexed
/// nor stored.
#[tokio::test]
async fn add_and_update_field_answer_400_for_a_field_option_of_the_wrong_type() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    create_index(
        &router,
        json!({
            "title": {"text": {"indexed": true}},
            "vec": {"flat": {"dimension": 2}}
        }),
    )
    .await;
    let before = schema(&router).await;

    let added = [
        (json!({"text": 1}), "text option must be an object"),
        (json!({"text": null}), "text option must be an object"),
        (
            json!({"text": {}, "integer": {}}),
            "a field option must have one type, got \"text\" and \"integer\"",
        ),
        (
            json!({"flat": {"dimension": 2, "base_weight": "2"}}),
            "base_weight must be a number",
        ),
    ];
    let updated = [
        (json!({"flat": 1}), "flat option must be an object"),
        (json!({"flat": null}), "flat option must be an object"),
        (
            json!({"flat": {"dimension": 2, "base_weight": "2"}}),
            "base_weight must be a number",
        ),
        (
            json!({"flat": {"dimension": 2, "embedder": 1}}),
            "embedder must be a string",
        ),
    ];
    let requests = added
        .into_iter()
        .map(|(option, expected)| {
            let body = json!({"name": "extra", "field_option": option});
            ("POST", "/v1/schema/fields", body, expected)
        })
        .chain(updated.into_iter().map(|(option, expected)| {
            let body = json!({"field_option": option, "reindex": true});
            ("PATCH", "/v1/schema/fields/vec", body, expected)
        }));
    let mut wrong = Vec::new();
    for (method, uri, request, expected) in requests {
        let (status, body) = send(&router, method, uri, request.clone()).await;
        if status != StatusCode::BAD_REQUEST || body["error"]["message"] != json!(expected) {
            wrong.push(format!("{method} {uri} {request}: {status} {body}"));
        }
        let after = schema(&router).await;
        if after != before {
            wrong.push(format!(
                "{method} {uri} {request}: the schema changed to {after}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A search with a value of the wrong type is a 400 naming the key on
/// `POST /v1/search`, `/v1/search/stream` and `/v1/search/batch` (where the
/// message starts with the entry's position), instead of a search that
/// reads the value as omitted or drops it.
#[tokio::test]
async fn search_answers_400_for_a_value_of_the_wrong_type() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    create_index(&router, search_fields()).await;

    let vectors = json!([{"vector": [1.0, 0.0]}]);
    let with_query = |key: &str, value: Value| json!({"query": "body:rust", key: value});
    let shape = "highlight must be an array of strings or an object";
    let exactly_one = "fusion must have exactly one of \"rrf\" and \"weighted_sum\"";
    let cases = [
        (json!(5), "search request must be an object"),
        (json!({"query": 5}), "query must be a string"),
        (
            json!({"query": 5, "query_vectors": vectors}),
            "query must be a string",
        ),
        (
            with_query("lexical_params", json!("x")),
            "lexical_params must be an object",
        ),
        (
            with_query("lexical_params", json!({"min_score": "100"})),
            "lexical_params: min_score must be a number",
        ),
        (
            with_query("lexical_params", json!({"sort_by": "year"})),
            "lexical_params: sort_by must be an object",
        ),
        (
            with_query("lexical_params", json!({"sort_by": {"order": "desc"}})),
            "lexical_params: sort_by: field must be a string",
        ),
        (
            with_query(
                "lexical_params",
                json!({"sort_by": {"field": "year", "order": "descending"}}),
            ),
            "lexical_params: sort_by: order must be \"asc\" or \"desc\", got \"descending\"",
        ),
        (
            with_query("field_boosts", json!({"title": "2"})),
            "field_boosts: \"title\" must be a number",
        ),
        (
            with_query("field_boosts", json!([["title", 2]])),
            "field_boosts must be an object",
        ),
        (
            with_query("fusion", json!({"rrf": {"k": "1"}})),
            "fusion: rrf: k must be a number",
        ),
        (with_query("fusion", json!({})), exactly_one),
        (
            with_query("fusion", json!({"rrf": {"k": 1}, "weighted_sum": {}})),
            exactly_one,
        ),
        (
            with_query("fusion", json!({"weighted_sum": {"lexical_weight": "1"}})),
            "fusion: weighted_sum: lexical_weight must be a number",
        ),
        (with_query("highlight", json!(["body", 1])), shape),
        (with_query("highlight", json!("body")), shape),
        (
            with_query("highlight", json!({"fields": ["body"], "tag": 1})),
            "highlight: tag must be a string",
        ),
    ];
    let mut wrong = Vec::new();
    for (search, expected) in cases {
        for (uri, request, expected) in [
            ("/v1/search", search.clone(), expected.to_string()),
            ("/v1/search/stream", search.clone(), expected.to_string()),
            (
                "/v1/search/batch",
                json!({"queries": [search.clone()]}),
                format!("queries[0]: {expected}"),
            ),
            (
                "/v1/search/batch",
                json!({"queries": [{"query": "body:rust"}, search.clone()]}),
                format!("queries[1]: {expected}"),
            ),
        ] {
            let (status, body) = send(&router, "POST", uri, request.clone()).await;
            if status != StatusCode::BAD_REQUEST || body["error"]["message"] != json!(expected) {
                wrong.push(format!("{uri} {request}: {status} {body}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The case in #1476 and in the Issue: `sort_by.order` is matched
/// case-insensitively, so `"DESC"` sorts descending instead of ascending,
/// and an omitted or `null` order sorts ascending. Keys of the right type
/// or `null` still search.
#[tokio::test]
async fn search_sorts_by_an_order_matched_case_insensitively() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    create_index(&router, search_fields()).await;
    for (id, title, body, year, vec) in [
        (
            "doc1",
            "rust",
            "rust programming language",
            2020,
            [1.0, 0.0],
        ),
        ("doc2", "go", "rust and go", 2024, [0.0, 1.0]),
    ] {
        let document = json!({"fields": {"title": title, "body": body, "year": year, "vec": vec}});
        let uri = format!("/v1/documents/{id}");
        let (status, body) = send(&router, "PUT", &uri, document).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, body) = send(&router, "POST", "/v1/commit", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The ids of the hits, in order.
    let ids = |body: &Value| -> Vec<String> {
        body["results"]
            .as_array()
            .unwrap_or_else(|| panic!("no results in {body}"))
            .iter()
            .map(|hit| hit["id"].as_str().unwrap().to_string())
            .collect()
    };
    let orders = [
        (json!("desc"), ["doc2", "doc1"]),
        (json!("DESC"), ["doc2", "doc1"]),
        (json!("Desc"), ["doc2", "doc1"]),
        (json!("asc"), ["doc1", "doc2"]),
        (json!("ASC"), ["doc1", "doc2"]),
        (json!(null), ["doc1", "doc2"]),
    ];
    let mut wrong = Vec::new();
    for (order, expected) in orders {
        let search = json!({
            "query": "body:rust",
            "lexical_params": {"sort_by": {"field": "year", "order": order}}
        });
        let (status, body) = send(&router, "POST", "/v1/search", search).await;
        if status != StatusCode::OK || ids(&body) != expected {
            wrong.push(format!("order {order}: {status} {body}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");

    let search = json!({
        "query": "body:rust",
        "query_vectors": [{"vector": [1.0, 0.0]}],
        "lexical_params": {"min_score": null, "sort_by": null},
        "field_boosts": {"title": 2, "body": null},
        "fusion": {"rrf": {"k": 1}, "weighted_sum": null},
        "highlight": {"fields": ["body"], "tag": "em", "css_class": null}
    });
    let (status, body) = send(&router, "POST", "/v1/search", search).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids(&body).len(), 2, "{body}");
}
