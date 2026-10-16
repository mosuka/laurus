//! Result grouping end to end through the gRPC services (Issue #1297).
//!
//! Three documents are added as chunks sharing their `_id`. Vectors are
//! Flat, so the vector ranking, and with it the groups, are exact.

use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tonic::{Code, Request};

use laurus::{CommitPolicy, DataValue, Document, FlatOption, Schema, TextOption, WalSyncPolicy};
use laurus_server::config::DEFAULT_MAX_RESULT_WINDOW;
use laurus_server::convert::search::group_by_params_from_json;
use laurus_server::convert::{document as doc_convert, schema as schema_convert};
use laurus_server::proto::laurus::v1::document_service_server::DocumentService as _;
use laurus_server::proto::laurus::v1::index_service_server::IndexService as _;
use laurus_server::proto::laurus::v1::search_service_server::SearchService as _;
use laurus_server::proto::laurus::v1::{
    self, AddDocumentsRequest, CommitRequest, CreateIndexRequest, DocumentEntry, GroupByParams,
    SearchBatchRequest,
};
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// `(id, title, vec)` of one chunk. Against `[1, 0]` the chunks rank
/// a "guide" 1.0, a "intro" 0.994, b "book" 0.970, c 0.832, a "notes"
/// 0.394, b "history" 0.243.
const CHUNKS: [(&str, &str, [f32; 2]); 6] = [
    ("a", "rust rust rust guide", [1.0, 0.0]),
    ("a", "rust rust intro", [0.9, 0.1]),
    ("a", "rust appendix notes", [0.3, 0.7]),
    ("b", "rust rust book chapter", [0.8, 0.2]),
    ("b", "rust history of programming", [0.2, 0.8]),
    ("c", "rust", [0.6, 0.4]),
];

/// Create the chunk index through `IndexService` and `DocumentService`,
/// and return a `SearchService` over it.
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

    let documents = CHUNKS
        .iter()
        .map(|(id, title, vec)| {
            let doc = Document::builder()
                .add_text("title", *title)
                .add_vector("vec", vec.to_vec())
                .build();
            DocumentEntry {
                id: id.to_string(),
                document: Some(doc_convert::to_proto(&doc)),
            }
        })
        .collect();
    let documents_service = DocumentService {
        engine: engine.clone(),
    };
    documents_service
        .add_documents(Request::new(AddDocumentsRequest { documents }))
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

fn vector_request(group_by: Option<GroupByParams>) -> v1::SearchRequest {
    v1::SearchRequest {
        query_vectors: vec![v1::QueryVector {
            vector: vec![1.0, 0.0],
            weight: 1.0,
            fields: vec!["vec".to_string()],
        }],
        limit: 10,
        group_by,
        ..Default::default()
    }
}

fn by_id(group_size: Option<u32>) -> Option<GroupByParams> {
    Some(GroupByParams {
        field: "_id".to_string(),
        group_size,
    })
}

fn title(result: &v1::SearchResult) -> String {
    let doc = doc_convert::from_proto(result.document.as_ref().expect("a stored document"));
    match doc.fields.get("title") {
        Some(DataValue::Text(title)) => title.clone(),
        other => panic!("{} has no title: {other:?}", result.id),
    }
}

/// Each group as `(id, group key, titles of its hits, leader first)`.
fn groups(results: &[v1::SearchResult]) -> Vec<(String, Option<DataValue>, Vec<String>)> {
    results
        .iter()
        .map(|leader| {
            for hit in &leader.inner_hits {
                assert_eq!(hit.group_key, leader.group_key, "inner hits share the key");
                assert!(hit.inner_hits.is_empty(), "inner hits nest no further");
            }
            (
                leader.id.clone(),
                leader
                    .group_key
                    .as_ref()
                    .map(doc_convert::data_value_from_proto),
                std::iter::once(leader)
                    .chain(&leader.inner_hits)
                    .map(title)
                    .collect(),
            )
        })
        .collect()
}

fn expected_groups() -> Vec<(String, Option<DataValue>, Vec<String>)> {
    let group = |id: &str, titles: &[&str]| {
        (
            id.to_string(),
            Some(DataValue::Text(id.to_string())),
            titles.iter().map(|t| t.to_string()).collect(),
        )
    };
    vec![
        group("a", &["rust rust rust guide", "rust rust intro"]),
        group(
            "b",
            &["rust rust book chapter", "rust history of programming"],
        ),
        group("c", &["rust"]),
    ]
}

#[tokio::test]
async fn test_search_groups_chunks_by_id() {
    let dir = tempfile::tempdir().unwrap();
    let service = search_service(&dir).await;

    let ungrouped = service
        .search(Request::new(vector_request(None)))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(ungrouped.results.len(), CHUNKS.len());
    assert!(
        ungrouped
            .results
            .iter()
            .all(|r| r.group_key.is_none() && r.inner_hits.is_empty())
    );

    let grouped = service
        .search(Request::new(vector_request(by_id(Some(2)))))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(groups(&grouped.results), expected_groups());

    // An unset group_size keeps one hit per group.
    let one_each = service
        .search(Request::new(vector_request(by_id(None))))
        .await
        .unwrap()
        .into_inner();
    let ids: Vec<&str> = one_each.results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["a", "b", "c"]);
    assert!(one_each.results.iter().all(|r| r.inner_hits.is_empty()));

    // The JSON form shared by the HTTP gateway and MCP yields the same.
    let from_json =
        group_by_params_from_json(&serde_json::json!({"field": "_id", "group_size": 2})).unwrap();
    let grouped_from_json = service
        .search(Request::new(vector_request(Some(from_json))))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(groups(&grouped_from_json.results), expected_groups());
}

#[tokio::test]
async fn test_search_stream_sends_one_group_per_message() {
    let dir = tempfile::tempdir().unwrap();
    let service = search_service(&dir).await;

    let mut stream = service
        .search_stream(Request::new(vector_request(by_id(Some(2)))))
        .await
        .unwrap()
        .into_inner();
    let mut results = Vec::new();
    while let Some(result) = stream.next().await {
        results.push(result.unwrap());
    }
    assert_eq!(groups(&results), expected_groups());
}

#[tokio::test]
async fn test_search_batch_groups_each_query() {
    let dir = tempfile::tempdir().unwrap();
    let service = search_service(&dir).await;

    let response = service
        .search_batch(Request::new(SearchBatchRequest {
            queries: vec![vector_request(by_id(Some(2))), vector_request(None)],
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.results.len(), 2);
    assert_eq!(groups(&response.results[0].results), expected_groups());
    assert_eq!(response.results[1].results.len(), CHUNKS.len());
}

#[tokio::test]
async fn test_invalid_group_by_is_invalid_argument() {
    let dir = tempfile::tempdir().unwrap();
    let service = search_service(&dir).await;

    let cases = [
        ("unknown field", "missing", Some(1)),
        ("vector field", "vec", Some(1)),
        ("zero size", "_id", Some(0)),
        ("size above 100", "_id", Some(101)),
    ];
    for (case, field, group_size) in cases {
        let request = vector_request(Some(GroupByParams {
            field: field.to_string(),
            group_size,
        }));
        let status = service.search(Request::new(request)).await.unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument, "{case}: {status:?}");
        assert!(status.message().contains("group_by"), "{case}: {status:?}");
    }
}
