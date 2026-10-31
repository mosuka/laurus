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

/// Issue #1374: a `field_boosts` key that names no lexical field (a typo, or
/// the vector field) is rejected instead of being silently ignored.
#[tokio::test(flavor = "multi_thread")]
async fn test_field_boost_on_a_non_lexical_field_is_rejected() {
    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;

    for key in ["titel", "vec"] {
        let request = v1::SearchRequest {
            query: "title:rust".to_string(),
            field_boosts: [(key.to_string(), 2.0)].into_iter().collect(),
            limit: 10,
            ..Default::default()
        };
        let status = service.search(Request::new(request)).await.expect_err(key);
        assert_eq!(status.code(), Code::InvalidArgument, "{key}: {status:?}");
        assert!(status.message().contains(key), "{key}: {status:?}");
    }
}

/// Issue #1540: a `NaN` boost, lexical `min_score`, RRF `k` or weight (which
/// only gRPC can send) and an empty `WeightedSum`, whose weights proto3
/// sends as 0 and 0, are `INVALID_ARGUMENT` naming the value. A `NaN` boost
/// is rejected without a `query` too, and so is a boost on the vector field
/// (Issue #1374), now that the boosts reach the engine without a `query`.
/// An empty `Rrf` (`k = 0`) is accepted.
#[tokio::test(flavor = "multi_thread")]
async fn test_out_of_range_search_parameters_are_rejected() {
    use v1::fusion_algorithm::Algorithm;

    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;
    let fusion = |algorithm| {
        Some(v1::FusionAlgorithm {
            algorithm: Some(algorithm),
        })
    };
    let boost = |field: &str, boost: f32| -> std::collections::HashMap<String, f32> {
        [(field.to_string(), boost)].into_iter().collect()
    };
    let cases = [
        (
            "NaN boost",
            v1::SearchRequest {
                field_boosts: boost("title", f32::NAN),
                ..hybrid_request("title:rust")
            },
            "field_boosts: boost for field \"title\" must be finite and at least 0, got NaN",
        ),
        (
            "NaN boost without a query",
            v1::SearchRequest {
                field_boosts: boost("title", f32::NAN),
                ..hybrid_request("")
            },
            "field_boosts: boost for field \"title\" must be finite and at least 0, got NaN",
        ),
        (
            "vector field boost without a query",
            v1::SearchRequest {
                field_boosts: boost("vec", 2.0),
                ..hybrid_request("")
            },
            "field_boosts names field(s) that are not a lexical field: vec",
        ),
        (
            "NaN min_score",
            v1::SearchRequest {
                lexical_params: Some(v1::LexicalParams {
                    min_score: f32::NAN,
                    ..Default::default()
                }),
                ..hybrid_request("title:rust")
            },
            "lexical min_score must be finite, got NaN",
        ),
        (
            "NaN k",
            v1::SearchRequest {
                fusion: fusion(Algorithm::Rrf(v1::Rrf { k: Some(f64::NAN) })),
                ..hybrid_request("title:rust")
            },
            "fusion: rrf: k must be finite and at least 0, got NaN",
        ),
        (
            "NaN weight",
            v1::SearchRequest {
                fusion: fusion(Algorithm::WeightedSum(v1::WeightedSum {
                    lexical_weight: Some(0.5),
                    vector_weight: Some(f32::NAN),
                })),
                ..hybrid_request("title:rust")
            },
            "fusion: weighted_sum: vector_weight must be between 0 and 1, got NaN",
        ),
        (
            "zero weights",
            v1::SearchRequest {
                fusion: fusion(Algorithm::WeightedSum(v1::WeightedSum {
                    lexical_weight: Some(0.0),
                    vector_weight: Some(0.0),
                })),
                ..hybrid_request("title:rust")
            },
            "fusion: weighted_sum: lexical_weight or vector_weight must be greater than 0",
        ),
    ];
    // Every request that is not rejected as expected, so a failure lists all
    // of them rather than the first.
    let mut wrong = Vec::new();
    for (case, request, expected) in cases {
        match service.search(Request::new(request)).await {
            Err(status)
                if status.code() == Code::InvalidArgument
                    && status.message().contains(expected) => {}
            other => wrong.push(format!(
                "{case}: {:?}",
                other.map(|response| response.into_inner().results.len())
            )),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The `(id, score)` pairs a hybrid search for `title:rust` returns with
/// `fusion` set to `algorithm` (unset when `None`), or the error message.
async fn fused_scores(
    service: &SearchService,
    algorithm: Option<v1::fusion_algorithm::Algorithm>,
) -> Result<Vec<(String, f32)>, String> {
    let request = v1::SearchRequest {
        fusion: algorithm.map(|algorithm| v1::FusionAlgorithm {
            algorithm: Some(algorithm),
        }),
        ..hybrid_request("title:rust")
    };
    match service.search(Request::new(request)).await {
        Ok(response) => Ok(response
            .into_inner()
            .results
            .into_iter()
            .map(|r| (r.id, r.score))
            .collect()),
        Err(status) => Err(status.message().to_string()),
    }
}

/// Issue #1551: an unset RRF `k` means 60 and an unset WeightedSum weight
/// 0.5, as over the HTTP gateway and MCP, and an explicit 0 is kept. proto3
/// used to send an unset `k` as 0, which fused with `1 / rank`, and unset
/// weights as 0, which the engine rejects.
#[tokio::test(flavor = "multi_thread")]
async fn test_unset_fusion_parameters_take_the_defaults_of_the_http_gateway() {
    use v1::fusion_algorithm::Algorithm;

    let dir = tempfile::TempDir::new().unwrap();
    let service = search_service(&dir).await;
    let rrf = |k| Some(Algorithm::Rrf(v1::Rrf { k }));
    let weighted_sum = |lexical_weight, vector_weight| {
        Some(Algorithm::WeightedSum(v1::WeightedSum {
            lexical_weight,
            vector_weight,
        }))
    };

    // Every comparison that does not hold, so a failure lists all of them.
    let mut wrong = Vec::new();
    let mut expect_same = |what: &str, left, right| {
        if left != right {
            wrong.push(format!("{what}: {left:?} != {right:?}"));
        }
    };

    let k_60 = fused_scores(&service, rrf(Some(60.0))).await;
    expect_same(
        "an unset fusion is RRF with k 60",
        fused_scores(&service, None).await,
        k_60.clone(),
    );
    expect_same(
        "an unset k is 60",
        fused_scores(&service, rrf(None)).await,
        k_60,
    );
    // `lexical` is first in the lexical list and second in the vector list,
    // `vector` first in the vector list: 1/1 + 1/2 and 1/1 with k 0.
    expect_same(
        "an explicit k of 0 is kept",
        fused_scores(&service, rrf(Some(0.0))).await,
        Ok(vec![
            ("lexical".to_string(), 1.5),
            ("vector".to_string(), 1.0),
        ]),
    );

    expect_same(
        "unset weights are 0.5",
        fused_scores(&service, weighted_sum(None, None)).await,
        fused_scores(&service, weighted_sum(Some(0.5), Some(0.5))).await,
    );
    expect_same(
        "an unset vector weight is 0.5",
        fused_scores(&service, weighted_sum(Some(1.0), None)).await,
        fused_scores(&service, weighted_sum(Some(1.0), Some(0.5))).await,
    );
    // With a vector weight of 0, `vector` (absent from the lexical list)
    // scores 0; with 0.5 it would score 0.5.
    let zero_vector_weight = fused_scores(&service, weighted_sum(Some(1.0), Some(0.0))).await;
    let vector_score = zero_vector_weight.as_ref().ok().and_then(|hits| {
        hits.iter()
            .find(|(id, _)| id == "vector")
            .map(|(_, score)| *score)
    });
    if vector_score != Some(0.0) {
        wrong.push(format!(
            "an explicit vector weight of 0 is kept: {zero_vector_weight:?}"
        ));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
