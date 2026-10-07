//! `SearchRequest.vector_params` end to end through the gRPC services
//! (Issue #1342): values that used to be dropped now change the results,
//! and values that cannot apply are rejected.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::{Code, Request};

use laurus::{CommitPolicy, Document, FlatOption, Schema, TextOption, WalSyncPolicy};
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
    SearchService { engine }
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
    let lexical_only = v1::SearchRequest {
        query: "title:rust".to_string(),
        lexical_params: Some(v1::LexicalParams::default()),
        vector_params: Some(v1::VectorParams::default()),
        ..Default::default()
    };
    for request in [zero_ef, lexical_only] {
        let status = service
            .search(Request::new(request))
            .await
            .expect_err("expected INVALID_ARGUMENT");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    }
}
