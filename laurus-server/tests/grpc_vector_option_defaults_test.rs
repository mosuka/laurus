//! HNSW and IVF fields created over gRPC with only `dimension` set
//! (Issue #1479).
//!
//! An unset `m`, `ef_construction`, `n_clusters` or `n_probe` used to reach
//! the engine as 0: with HNSW `m` 0 every document write failed, even one
//! without a vector, and with IVF `n_probe` 0 every search returned no
//! hits. They now get the engine's defaults.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::Request;

use laurus::{CommitPolicy, Document, WalSyncPolicy};
use laurus_server::config::DEFAULT_MAX_RESULT_WINDOW;
use laurus_server::convert::document as doc_convert;
use laurus_server::proto::laurus::v1::document_service_server::DocumentService as _;
use laurus_server::proto::laurus::v1::field_option::Option as Opt;
use laurus_server::proto::laurus::v1::index_service_server::IndexService as _;
use laurus_server::proto::laurus::v1::search_service_server::SearchService as _;
use laurus_server::proto::laurus::v1::{
    self, CommitRequest, CreateIndexRequest, DocumentEntry, PutDocumentsRequest,
};
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// Against the query `[1, 0]`, `near` matches exactly and `far` is
/// orthogonal. `text_only` has no vector.
const CORPUS: [(&str, Option<[f32; 2]>); 4] = [
    ("near", Some([1.0, 0.0])),
    ("mid", Some([0.7, 0.7])),
    ("far", Some([0.0, 1.0])),
    ("text_only", None),
];

/// Create an index whose field `vec` is `vector_field`, put and commit
/// [`CORPUS`], and return a search service over it.
///
/// The schema is a proto message built directly: going through a core
/// `Schema` and `schema_convert::to_proto` would fill in the engine's
/// defaults and hide what an unset proto field does.
async fn search_service(dir: &tempfile::TempDir, vector_field: Opt) -> SearchService {
    let engine = Arc::new(RwLock::new(None));
    let fields = HashMap::from([
        (
            "title".to_string(),
            v1::FieldOption {
                option: Some(Opt::Text(v1::TextOption {
                    indexed: true,
                    stored: true,
                    ..Default::default()
                })),
            },
        ),
        (
            "vec".to_string(),
            v1::FieldOption {
                option: Some(vector_field),
            },
        ),
    ]);
    IndexService {
        engine: engine.clone(),
        data_dir: dir.path().join("data"),
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    }
    .create_index(Request::new(CreateIndexRequest {
        schema: Some(v1::Schema {
            fields,
            ..Default::default()
        }),
    }))
    .await
    .unwrap();

    let documents = CORPUS
        .iter()
        .map(|(id, vector)| {
            let builder = Document::builder().add_text("title", "rust");
            let document = match vector {
                Some(vector) => builder.add_vector("vec", vector.to_vec()),
                None => builder,
            }
            .build();
            DocumentEntry {
                id: id.to_string(),
                document: Some(doc_convert::to_proto(&document)),
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

/// The ids a vector search for `[1, 0]` on `vec` returns.
async fn vector_hits(service: &SearchService) -> Vec<String> {
    let request = v1::SearchRequest {
        query_vectors: vec![v1::QueryVector {
            vector: vec![1.0, 0.0],
            weight: 1.0,
            fields: vec!["vec".to_string()],
        }],
        limit: 10,
        ..Default::default()
    };
    service
        .search(Request::new(request))
        .await
        .unwrap()
        .into_inner()
        .results
        .into_iter()
        .map(|r| r.id)
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn hnsw_field_without_m_or_ef_construction_takes_documents() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(
        &dir,
        Opt::Hnsw(v1::HnswOption {
            dimension: 2,
            ..Default::default()
        }),
    )
    .await;

    let hits = vector_hits(&service).await;
    assert!(hits.contains(&"near".to_string()), "{hits:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn ivf_field_without_n_clusters_or_n_probe_finds_documents() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(
        &dir,
        Opt::Ivf(v1::IvfOption {
            dimension: 2,
            ..Default::default()
        }),
    )
    .await;

    let hits = vector_hits(&service).await;
    assert!(hits.contains(&"near".to_string()), "{hits:?}");
}
