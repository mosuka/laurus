//! Embedders created over gRPC are stored and returned as sent (Issue
//! #1702).
//!
//! An embedder used to be `type` and `params`, a map of strings that the
//! server read key by key: `"query_maxlen": "32"` was a number, and a key
//! the type does not take, such as a misspelled `"doc_max_len"` or
//! `"model"` on a `precomputed` embedder, was dropped. An embedder is now
//! `config`, a `google.protobuf.Struct` in the shape `schema.toml` uses,
//! read with the core type's serde, and the old form is rejected.
//!
//! Only embedders that no field uses are built from a model, so every
//! embedder type can be created here without downloading one.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use prost_types::value::Kind;
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tonic::{Code, Request};

use laurus::{CommitPolicy, Schema, WalSyncPolicy};
use laurus_server::proto::laurus::v1::field_option::Option as Opt;
use laurus_server::proto::laurus::v1::index_service_server::IndexService as _;
use laurus_server::proto::laurus::v1::{self, CreateIndexRequest, GetSchemaRequest};
use laurus_server::service::index::IndexService;

/// A `google.protobuf.Struct` holding the JSON object `json`, as a gRPC
/// client builds one.
fn to_struct(json: &Value) -> prost_types::Struct {
    fn to_value(json: &Value) -> prost_types::Value {
        let kind = match json {
            Value::Null => Kind::NullValue(0),
            Value::Bool(b) => Kind::BoolValue(*b),
            Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap()),
            Value::String(s) => Kind::StringValue(s.clone()),
            Value::Array(items) => Kind::ListValue(prost_types::ListValue {
                values: items.iter().map(to_value).collect(),
            }),
            Value::Object(_) => Kind::StructValue(to_struct(json)),
        };
        prost_types::Value { kind: Some(kind) }
    }
    prost_types::Struct {
        fields: json
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), to_value(v)))
            .collect(),
    }
}

fn embedder(json: &Value) -> v1::EmbedderConfig {
    v1::EmbedderConfig {
        config: Some(to_struct(json)),
        ..Default::default()
    }
}

fn index_service(data_dir: &Path) -> IndexService {
    IndexService {
        engine: Arc::new(RwLock::new(None)),
        data_dir: data_dir.to_path_buf(),
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    }
}

/// A schema with a text field, an HNSW field that names the `precomputed`
/// embedder `pre` when `embedders` has one, and `embedders`.
fn schema(embedders: HashMap<String, v1::EmbedderConfig>) -> v1::Schema {
    let mut fields = HashMap::from([(
        "body".to_string(),
        v1::FieldOption {
            option: Some(Opt::Text(v1::TextOption {
                indexed: Some(true),
                stored: Some(true),
                ..Default::default()
            })),
        },
    )]);
    if embedders.contains_key("pre") {
        fields.insert(
            "vec".to_string(),
            v1::FieldOption {
                option: Some(Opt::Hnsw(v1::HnswOption {
                    dimension: 4,
                    embedder: "pre".to_string(),
                    ..Default::default()
                })),
            },
        );
    }
    v1::Schema {
        fields,
        embedders,
        ..Default::default()
    }
}

/// A schema whose only embedder is `e`, which no field uses.
fn schema_with(e: v1::EmbedderConfig) -> v1::Schema {
    schema(HashMap::from([("e".to_string(), e)]))
}

/// One embedder of every type, `candle_colbert` with all of its parameters
/// and with none; only `pre` is used by a field.
fn embedders_sent() -> Vec<(&'static str, Value)> {
    vec![
        ("pre", json!({"type": "precomputed"})),
        (
            "bert",
            json!({"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}),
        ),
        (
            "clip",
            json!({"type": "candle_clip", "model": "openai/clip-vit-base-patch32"}),
        ),
        (
            "openai",
            json!({"type": "openai", "model": "text-embedding-3-small"}),
        ),
        (
            "colbert",
            json!({
                "type": "candle_colbert",
                "model": "colbert-ir/colbertv2.0",
                "revision": "c1e84128",
                "query_maxlen": 32,
                "doc_maxlen": 180
            }),
        ),
        (
            "colbert_small",
            json!({"type": "candle_colbert", "model": "answerdotai/answerai-colbert-small-v1"}),
        ),
    ]
}

#[tokio::test]
async fn embedders_are_stored_and_returned_as_sent() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let service = index_service(&data_dir);
    let sent: HashMap<String, v1::EmbedderConfig> = embedders_sent()
        .iter()
        .map(|(name, json)| (name.to_string(), embedder(json)))
        .collect();
    service
        .create_index(Request::new(CreateIndexRequest {
            schema: Some(schema(sent.clone())),
        }))
        .await
        .unwrap();

    // schema.toml holds the embedders as sent.
    let toml = std::fs::read_to_string(data_dir.join("schema.toml")).unwrap();
    let stored = Schema::from_toml(&toml).unwrap();
    for (name, json) in embedders_sent() {
        assert_eq!(
            serde_json::to_value(&stored.embedders[name]).unwrap(),
            json,
            "{name}"
        );
    }

    // GetSchema returns what was sent.
    let returned = service
        .get_schema(Request::new(GetSchemaRequest {}))
        .await
        .unwrap()
        .into_inner()
        .schema
        .unwrap();
    assert_eq!(returned.embedders, sent);
}

#[tokio::test]
async fn an_embedder_in_the_form_before_issue_1702_is_rejected() {
    #[allow(deprecated)]
    let cases = [
        // What a client from before Issue #1702 sends.
        v1::EmbedderConfig {
            r#type: "precomputed".into(),
            ..Default::default()
        },
        v1::EmbedderConfig {
            r#type: "candle_bert".into(),
            params: HashMap::from([("model".to_string(), "m".to_string())]),
            ..Default::default()
        },
        // The old fields set next to `config`.
        v1::EmbedderConfig {
            params: HashMap::from([("model".to_string(), "m".to_string())]),
            ..embedder(&json!({"type": "candle_bert", "model": "m"}))
        },
    ];
    for e in cases {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let status = index_service(&data_dir)
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema_with(e.clone())),
            }))
            .await
            .unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument, "{e:?}");
        assert!(
            status
                .message()
                .starts_with("Embedder 'e': the embedder sets `type` or `params`"),
            "{}",
            status.message()
        );
        assert!(!data_dir.exists(), "{e:?}: an index was created");
    }
}

/// An embedder the core type cannot read is `INVALID_ARGUMENT` naming the
/// embedder, and no index is created, instead of an embedder stored with
/// another type or without the key.
#[tokio::test]
async fn an_embedder_of_the_wrong_shape_is_rejected() {
    let cases = [
        (
            json!({"type": "candle_colbert", "model": "m", "query_maxlen": "32"}),
            "Embedder 'e': invalid type: string \"32\", expected usize",
        ),
        (
            json!({"type": "candle_colbert", "model": "m", "revision": 123}),
            "Embedder 'e': invalid type: integer `123`, expected a string",
        ),
        (
            json!({"type": "candle_bert", "model": true}),
            "Embedder 'e': invalid type: boolean `true`, expected a string",
        ),
        (
            json!({"type": "candle_colbert", "model": "m", "doc_max_len": 300}),
            "Embedder 'e': unknown field `doc_max_len`, expected one of `model`, `revision`, \
             `query_maxlen`, `doc_maxlen`",
        ),
        (
            json!({"type": "precomputed", "model": "x"}),
            "Embedder 'e': unknown field `model`, there are no fields",
        ),
        (
            json!({"type": "candle_bert", "pooling": "mean", "model": "m"}),
            "Embedder 'e': unknown field `pooling`, expected `model`",
        ),
        (
            json!({"type": "candle_bert"}),
            "Embedder 'e': missing field `model`",
        ),
    ];
    // Every request that was not rejected as expected, so a failure lists
    // all of them rather than the first.
    let mut wrong = Vec::new();
    for (json, expected) in cases {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let result = index_service(&data_dir)
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema_with(embedder(&json))),
            }))
            .await;
        match result {
            Err(status)
                if status.code() == Code::InvalidArgument && status.message() == expected => {}
            other => wrong.push(format!("{json}: {other:?}")),
        }
        if data_dir.exists() {
            wrong.push(format!("{json}: an index was created"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
