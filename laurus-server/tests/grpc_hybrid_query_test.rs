//! Issue #1366: `query` (unified DSL) and `query_vectors` sent together
//! through the gRPC services form one search: the vectors used to be
//! silently dropped. With `lexical_params` / `field_boosts`, `query` is still
//! the unified DSL, so an unknown field is rejected instead of matching
//! nothing.

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

/// `lexical` matches `title:rust` only; `vector` is close to `[1, 0]` only.
const CORPUS: [(&str, &str, [f32; 2]); 2] = [
    ("lexical", "rust", [0.0, 1.0]),
    ("vector", "go", [1.0, 0.0]),
];

async fn search_service(dir: &tempfile::TempDir) -> SearchService {
    let engine = Arc::new(RwLock::new(None));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_flat_field("vec", FlatOption::new(2))
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
        .map(|(id, title, vector)| DocumentEntry {
            id: id.to_string(),
            document: Some(doc_convert::to_proto(
                &Document::builder()
                    .add_text("title", *title)
                    .add_vector("vec", vector.to_vec())
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

fn hybrid_request(query: &str) -> v1::SearchRequest {
    v1::SearchRequest {
        query: query.to_string(),
        query_vectors: vec![v1::QueryVector {
            vector: vec![1.0, 0.0],
            weight: 1.0,
            fields: Vec::new(),
        }],
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
async fn test_query_with_query_vectors_is_a_hybrid_search() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;

    let found = ids(&service, hybrid_request("title:rust")).await;
    assert_eq!(found, ["lexical", "vector"]);

    let mut with_boosts = hybrid_request("title:rust");
    with_boosts.field_boosts = [("title".to_string(), 2.0)].into_iter().collect();
    assert_eq!(ids(&service, with_boosts).await, ["lexical", "vector"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unknown_field_is_rejected_with_lexical_overrides() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;

    let request = v1::SearchRequest {
        query: "titl:rust".to_string(),
        field_boosts: [("title".to_string(), 2.0)].into_iter().collect(),
        limit: 10,
        ..Default::default()
    };
    let status = service
        .search(Request::new(request))
        .await
        .expect_err("an unknown field must be rejected");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
}
