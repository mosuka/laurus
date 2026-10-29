//! `SearchRequest.vector_params` end to end through the gRPC services
//! (Issue #1342): values that used to be dropped now change the results,
//! and values that cannot apply are rejected.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::{Code, Request};

use laurus::{CommitPolicy, Document, FlatOption, Schema, TextOption, WalSyncPolicy};
use laurus_server::config::DEFAULT_MAX_RESULT_WINDOW;
use laurus_server::convert::{document as doc_convert, schema as schema_convert};
use laurus_server::proto::laurus::v1::document_service_server::DocumentService as _;
use laurus_server::proto::laurus::v1::index_service_server::IndexService as _;
use laurus_server::proto::laurus::v1::search_service_server::SearchService as _;
use laurus_server::proto::laurus::v1::{
    self, CommitRequest, CreateIndexRequest, DocumentEntry, PutDocumentsRequest,
};
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// Two 2-d vector fields `a` and `b`. Against the query `[1, 0]`, `in_a`
/// and `in_b` match exactly and `far_a` is orthogonal.
const CORPUS: [(&str, &str, [f32; 2]); 3] = [
    ("in_a", "a", [1.0, 0.0]),
    ("in_b", "b", [1.0, 0.0]),
    ("far_a", "a", [0.0, 1.0]),
];

async fn search_service(dir: &tempfile::TempDir) -> SearchService {
    let engine = Arc::new(RwLock::new(None));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_flat_field("a", FlatOption::new(2))
        .add_flat_field("b", FlatOption::new(2))
        .build();
    IndexService {
        engine: engine.clone(),
        data_dir: dir.path().join("data"),
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    }
    .create_index(Request::new(CreateIndexRequest {
        schema: Some(schema_convert::to_proto(&schema)),
    }))
    .await
    .unwrap();

    let documents = CORPUS
        .iter()
        .map(|(id, field, vector)| DocumentEntry {
            id: id.to_string(),
            document: Some(doc_convert::to_proto(
                &Document::builder()
                    .add_text("title", "rust")
                    .add_vector(*field, vector.to_vec())
                    .build(),
            )),
        })
        .collect();
    let documents_service = DocumentService {
        engine: engine.clone(),
    };
    documents_service
        .put_documents(Request::new(PutDocumentsRequest { documents }))
        .await
        .unwrap();
    documents_service
        .commit(Request::new(CommitRequest {}))
        .await
        .unwrap();
    SearchService {
        engine,
        max_result_window: DEFAULT_MAX_RESULT_WINDOW,
    }
}

/// A search with one field-less query vector `[1, 0]`.
fn vector_request(vector_params: v1::VectorParams) -> v1::SearchRequest {
    v1::SearchRequest {
        query_vectors: vec![v1::QueryVector {
            vector: vec![1.0, 0.0],
            weight: 1.0,
            fields: Vec::new(),
        }],
        vector_params: Some(vector_params),
        limit: 10,
        ..Default::default()
    }
}

async fn ids(service: &SearchService, request: v1::SearchRequest) -> Vec<String> {
    let mut ids: Vec<String> = service
        .search(Request::new(request))
        .await
        .unwrap()
        .into_inner()
        .results
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    ids
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fields_routes_a_field_less_query_vector() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;

    let all = ids(&service, vector_request(v1::VectorParams::default())).await;
    assert_eq!(all, ["far_a", "in_a", "in_b"]);

    let only_a = ids(
        &service,
        vector_request(v1::VectorParams {
            fields: vec!["a".to_string()],
            ..Default::default()
        }),
    )
    .await;
    assert_eq!(only_a, ["far_a", "in_a"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_min_score_filters_results() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;

    let close = ids(
        &service,
        vector_request(v1::VectorParams {
            min_score: 0.9,
            ..Default::default()
        }),
    )
    .await;
    assert_eq!(close, ["in_a", "in_b"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unusable_vector_params_are_invalid_argument() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;

    let zero_ef = vector_request(v1::VectorParams {
        ef_search: Some(0),
        ..Default::default()
    });
    // No query_vectors and no query: nothing the params could apply to.
    let no_vector_part = v1::SearchRequest {
        vector_params: Some(v1::VectorParams::default()),
        limit: 10,
        ..Default::default()
    };
    for request in [zero_ef, no_vector_part] {
        let status = service
            .search(Request::new(request))
            .await
            .expect_err("expected INVALID_ARGUMENT");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    }
}

/// Issue #1552: a query vector weight that is negative or not finite is
/// `INVALID_ARGUMENT` naming the weight, instead of inverting the ranking
/// or giving `NaN` scores. A weight of `0` or `-0` means unset (proto3),
/// so it searches as `1.0`.
#[tokio::test(flavor = "multi_thread")]
async fn test_out_of_range_query_vector_weights_are_invalid_argument() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;
    let with_weight = |weight: f32| {
        let mut request = vector_request(v1::VectorParams::default());
        request.query_vectors[0].weight = weight;
        request
    };

    let mut accepted = Vec::new();
    for weight in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        match service.search(Request::new(with_weight(weight))).await {
            Err(status)
                if status.code() == Code::InvalidArgument
                    && status
                        .message()
                        .contains("query vector weight must be finite and at least 0") => {}
            other => accepted.push(format!("{weight}: {other:?}")),
        }
    }
    assert!(accepted.is_empty(), "{accepted:#?}");

    let scores = |weight: f32| {
        let service = &service;
        async move {
            let mut scores: Vec<(String, f32)> = service
                .search(Request::new(with_weight(weight)))
                .await
                .unwrap()
                .into_inner()
                .results
                .into_iter()
                .map(|r| (r.id, r.score))
                .collect();
            scores.sort_by(|a, b| a.0.cmp(&b.0));
            scores
        }
    };
    let one = scores(1.0).await;
    assert_eq!(one.len(), 3, "{one:?}");
    assert_eq!(scores(0.0).await, one, "a weight of 0 searches as 1.0");
    assert_eq!(scores(-0.0).await, one, "so does -0");
}
