//! Late-interaction rescore end to end through the gRPC services
//! (Issue #1351).
//!
//! Uses the corpus of `laurus/tests/late_interaction_rescore_test.rs`, so a
//! rescore requested over gRPC must return the ranking the Rust API returns.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::{Code, Request};

use laurus::{
    CommitPolicy, DistanceMetric, Document, FieldOption, FlatOption, MultiVectorOption, Schema,
    TextOption, WalSyncPolicy,
};
use laurus_server::config::DEFAULT_MAX_RESULT_WINDOW;
use laurus_server::convert::search::rescore_params_from_json;
use laurus_server::convert::{document as doc_convert, schema as schema_convert};
use laurus_server::proto::laurus::v1::document_service_server::DocumentService as _;
use laurus_server::proto::laurus::v1::index_service_server::IndexService as _;
use laurus_server::proto::laurus::v1::search_service_server::SearchService as _;
use laurus_server::proto::laurus::v1::{
    self, CommitRequest, CreateIndexRequest, DocumentEntry, PutDocumentsRequest, RescoreParams,
};
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// `(id, title, vec, tokens)` of one corpus document.
type Entry = (&'static str, &'static str, [f32; 2], &'static [[f32; 2]]);

/// Against the query `[[1, 0], [0, 1]]` the late-interaction scores are
/// c 1.1, b 1.0, d 0.9, a 0.1, e 0.05.
const CORPUS: [Entry; 5] = [
    ("a", "rust", [1.0, 0.0], &[[0.1, 0.0]]),
    ("b", "rust rust", [0.9, 0.1], &[[0.5, 0.5], [0.0, 0.2]]),
    ("c", "rust language", [0.5, 0.5], &[[0.9, 0.2]]),
    ("d", "rust rust rust", [0.2, 0.8], &[[0.3, 0.3], [0.6, 0.0]]),
    ("e", "learning rust today", [0.0, 1.0], &[[0.02, 0.03]]),
];

/// Create the corpus index through `IndexService` and `DocumentService`,
/// and return a `SearchService` over it.
async fn search_service(dir: &tempfile::TempDir) -> SearchService {
    let engine = Arc::new(RwLock::new(None));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_flat_field("vec", FlatOption::new(2))
        .add_field(
            "tokens",
            FieldOption::MultiVector(
                MultiVectorOption::new(2).distance(DistanceMetric::DotProduct),
            ),
        )
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
        .map(|(id, title, vec, tokens)| {
            let doc = Document::builder()
                .add_text("title", *title)
                .add_vector("vec", vec.to_vec())
                .add_vector_array("tokens", tokens.iter().map(|t| t.to_vec()).collect())
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

fn lexical_request(rescore: Option<RescoreParams>) -> v1::SearchRequest {
    v1::SearchRequest {
        query: "title:rust".to_string(),
        limit: 10,
        rescore,
        ..Default::default()
    }
}

fn vectors_rescore(window_size: Option<u32>) -> RescoreParams {
    RescoreParams {
        window_size,
        rescorer: Some(v1::rescore_params::Rescorer::LateInteraction(
            v1::LateInteractionRescore {
                field: "tokens".to_string(),
                query: Some(v1::late_interaction_rescore::Query::Vectors(
                    v1::VectorArrayValue {
                        dimension: 2,
                        values: vec![1.0, 0.0, 0.0, 1.0],
                    },
                )),
            },
        )),
    }
}

async fn search(
    service: &SearchService,
    request: v1::SearchRequest,
) -> Result<Vec<(String, f32)>, tonic::Status> {
    Ok(service
        .search(Request::new(request))
        .await?
        .into_inner()
        .results
        .into_iter()
        .map(|r| (r.id, r.score))
        .collect())
}

fn ids(results: &[(String, f32)]) -> Vec<&str> {
    results.iter().map(|(id, _)| id.as_str()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rescore_matches_the_rust_api_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let service = search_service(&dir).await;

    let baseline = search(&service, lexical_request(None)).await.unwrap();
    let rescored = search(&service, lexical_request(Some(vectors_rescore(None))))
        .await
        .unwrap();
    assert_eq!(ids(&rescored), ["c", "b", "d", "a", "e"]);
    assert_ne!(ids(&baseline), ids(&rescored));
    for ((_, score), expected) in rescored.iter().zip([1.1, 1.0, 0.9, 0.1, 0.05]) {
        assert!((score - expected).abs() < 1e-5, "{rescored:?}");
    }

    // The JSON form shared by the HTTP gateway and MCP yields the same
    // request; a one-candidate window only reorders the first hit.
    let json = serde_json::json!({
        "window_size": 1,
        "late_interaction": {"field": "tokens", "vectors": [[1, 0], [0, 1]]}
    });
    let windowed = search(
        &service,
        lexical_request(Some(rescore_params_from_json(&json).unwrap())),
    )
    .await
    .unwrap();
    assert_eq!(ids(&windowed)[1..], ids(&baseline)[1..]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_rescore_is_invalid_argument() {
    let dir = tempfile::tempdir().unwrap();
    let service = search_service(&dir).await;

    let text_without_embedder = RescoreParams {
        window_size: None,
        rescorer: Some(v1::rescore_params::Rescorer::LateInteraction(
            v1::LateInteractionRescore {
                field: "tokens".to_string(),
                query: Some(v1::late_interaction_rescore::Query::Text(
                    "rust".to_string(),
                )),
            },
        )),
    };
    let cases = [
        (vectors_rescore(Some(0)), "window_size must be between"),
        (text_without_embedder, "no token-level embedder"),
        (
            RescoreParams {
                window_size: None,
                rescorer: None,
            },
            "rescore.late_interaction must be set",
        ),
    ];
    for (rescore, expected) in cases {
        let status = search(&service, lexical_request(Some(rescore)))
            .await
            .unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains(expected), "{status:?}");
    }
}
