//! Analyzer components created over gRPC are stored and returned as sent
//! (Issue #1478).
//!
//! A component used to be `type` and `params`, a map of strings: a stop
//! filter's `words` was one comma-separated string, so `"new,york"` became
//! the two words `new` and `york` and each word was trimmed, and an empty
//! list could not be sent. A component is now `config`, a
//! `google.protobuf.Struct` in the shape `schema.toml` uses, and the old
//! form is rejected.

use std::collections::{BTreeMap, HashMap};
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

fn component(json: Value) -> v1::ComponentConfig {
    v1::ComponentConfig {
        config: Some(to_struct(&json)),
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

/// A schema with a text field and the analyzers in `analyzers`.
fn schema(analyzers: HashMap<String, v1::AnalyzerDefinition>) -> v1::Schema {
    v1::Schema {
        fields: HashMap::from([(
            "body".to_string(),
            v1::FieldOption {
                option: Some(Opt::Text(v1::TextOption {
                    indexed: true,
                    stored: true,
                    ..Default::default()
                })),
            },
        )]),
        analyzers,
        ..Default::default()
    }
}

/// The components the test creates, each with every key `schema.toml`
/// writes, so that what comes back is what was sent: stop words with
/// commas and spaces, an empty stop list, a mapping with a key `type`, an
/// n-gram and a boost of 0.1.
fn analyzers_sent() -> (Vec<Value>, Value, Vec<Value>) {
    (
        vec![
            json!({"type": "mapping", "mapping": {"type": "t", "ph": "f"}}),
            json!({"type": "japanese_iteration_mark", "kanji": false, "kana": true}),
        ],
        json!({"type": "ngram", "min_gram": 2, "max_gram": 3}),
        vec![
            json!({"type": "stop", "words": ["new,york", " padded ", "x"]}),
            json!({"type": "stop", "words": []}),
            json!({"type": "boost", "boost": 0.1}),
        ],
    )
}

#[tokio::test]
async fn components_are_stored_and_returned_as_sent() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let service = index_service(&data_dir);
    let (char_filters, tokenizer, token_filters) = analyzers_sent();
    let definition = v1::AnalyzerDefinition {
        char_filters: char_filters.iter().cloned().map(component).collect(),
        tokenizer: Some(component(tokenizer.clone())),
        token_filters: token_filters.iter().cloned().map(component).collect(),
    };
    service
        .create_index(Request::new(CreateIndexRequest {
            schema: Some(schema(HashMap::from([(
                "a".to_string(),
                definition.clone(),
            )]))),
        }))
        .await
        .unwrap();

    // schema.toml holds the components as sent.
    let toml = std::fs::read_to_string(data_dir.join("schema.toml")).unwrap();
    let stored = Schema::from_toml(&toml).unwrap();
    let stored = serde_json::to_value(&stored.analyzers["a"]).unwrap();
    // A boost of 0.1 is an f32 in the core types, so it is compared as one.
    let boost_as_f32 = f64::from(0.1_f32);
    let mut expected_token_filters = token_filters.clone();
    expected_token_filters[2]["boost"] = json!(boost_as_f32);
    assert_eq!(
        stored,
        json!({
            "char_filters": char_filters,
            "tokenizer": tokenizer,
            "token_filters": expected_token_filters,
        })
    );

    // GetSchema returns what was sent.
    let returned = service
        .get_schema(Request::new(GetSchemaRequest {}))
        .await
        .unwrap()
        .into_inner()
        .schema
        .unwrap();
    assert_eq!(returned.analyzers["a"], definition);
}

#[tokio::test]
async fn a_component_in_the_form_before_issue_1478_is_rejected() {
    let whitespace = component(json!({"type": "whitespace"}));
    #[allow(deprecated)]
    let cases = [
        // What a client from before Issue #1478 sends.
        v1::ComponentConfig {
            r#type: "whitespace".into(),
            ..Default::default()
        },
        v1::ComponentConfig {
            r#type: "regex".into(),
            params: HashMap::from([("pattern".to_string(), "[a-z]+".to_string())]),
            ..Default::default()
        },
        // The old fields set next to `config`.
        v1::ComponentConfig {
            params: HashMap::from([("pattern".to_string(), "[a-z]+".to_string())]),
            ..whitespace.clone()
        },
    ];
    for tokenizer in cases {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let definition = v1::AnalyzerDefinition {
            tokenizer: Some(tokenizer.clone()),
            ..Default::default()
        };
        let status = index_service(&data_dir)
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema(HashMap::from([("a".to_string(), definition)]))),
            }))
            .await
            .unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument, "{tokenizer:?}");
        assert!(
            status
                .message()
                .starts_with("Analyzer 'a': tokenizer: the component sets `type` or `params`"),
            "{}",
            status.message()
        );
        assert!(!data_dir.exists(), "{tokenizer:?}: an index was created");
    }
}

/// An unset optional parameter is left out of `config`, rather than sent
/// as `null`, and an empty table is kept.
#[tokio::test]
async fn get_schema_leaves_out_unset_parameters_and_keeps_empty_tables() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let service = index_service(&data_dir);
    let definition = v1::AnalyzerDefinition {
        char_filters: vec![component(json!({"type": "mapping", "mapping": {}}))],
        tokenizer: Some(component(json!({"type": "regex"}))),
        token_filters: vec![
            component(json!({"type": "stop"})),
            component(json!({"type": "stem"})),
        ],
    };
    service
        .create_index(Request::new(CreateIndexRequest {
            schema: Some(schema(HashMap::from([("a".to_string(), definition)]))),
        }))
        .await
        .unwrap();
    let returned = service
        .get_schema(Request::new(GetSchemaRequest {}))
        .await
        .unwrap()
        .into_inner()
        .schema
        .unwrap();
    let analyzer = &returned.analyzers["a"];
    let keys = |c: &v1::ComponentConfig| -> BTreeMap<String, Kind> {
        c.config
            .as_ref()
            .unwrap()
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), v.kind.clone().unwrap()))
            .collect()
    };
    assert_eq!(
        keys(&analyzer.char_filters[0]),
        BTreeMap::from([
            ("type".to_string(), Kind::StringValue("mapping".into())),
            (
                "mapping".to_string(),
                Kind::StructValue(prost_types::Struct::default())
            ),
        ])
    );
    // The regex defaults are filled in.
    assert_eq!(
        keys(analyzer.tokenizer.as_ref().unwrap()),
        BTreeMap::from([
            ("type".to_string(), Kind::StringValue("regex".into())),
            ("pattern".to_string(), Kind::StringValue(r"\w+".into())),
            ("gaps".to_string(), Kind::BoolValue(false)),
        ])
    );
    for (filter, name) in analyzer.token_filters.iter().zip(["stop", "stem"]) {
        assert_eq!(
            keys(filter),
            BTreeMap::from([("type".to_string(), Kind::StringValue(name.into()))])
        );
    }
}
