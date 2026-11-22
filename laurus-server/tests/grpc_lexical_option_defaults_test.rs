//! Lexical fields created or changed over gRPC with `indexed` and `stored`
//! left out (Issue #1709).
//!
//! Both used to be plain proto `bool`s, so a client that left them out
//! created a field that was neither indexed nor stored: a search found no
//! document and a document came back without the field's value. They are
//! now `optional bool`, and an unset one gets the core option's default
//! (`true`), as an omitted key in `schema.toml` does.

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
    self, AddFieldRequest, CommitRequest, CreateIndexRequest, DocumentEntry, GetDocumentsRequest,
    GetSchemaRequest, PutDocumentsRequest, UpdateFieldRequest,
};
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// The services over one index in `dir`.
struct Services {
    index: IndexService,
    documents: DocumentService,
    search: SearchService,
}

/// Create an index in `dir` whose fields are `fields`, as proto messages
/// built directly: going through a core `Schema` and `schema_convert::to_proto`
/// would fill in the engine's defaults and hide what an unset proto field
/// does.
async fn create_index(dir: &tempfile::TempDir, fields: HashMap<String, Opt>) -> Services {
    let engine = Arc::new(RwLock::new(None));
    let index = IndexService {
        engine: engine.clone(),
        data_dir: dir.path().join("data"),
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    };
    let fields = fields
        .into_iter()
        .map(|(name, option)| {
            (
                name,
                v1::FieldOption {
                    option: Some(option),
                },
            )
        })
        .collect();
    index
        .create_index(Request::new(CreateIndexRequest {
            schema: Some(v1::Schema {
                fields,
                ..Default::default()
            }),
        }))
        .await
        .unwrap();
    Services {
        index,
        documents: DocumentService {
            engine: engine.clone(),
        },
        search: SearchService {
            engine,
            max_result_window: DEFAULT_MAX_RESULT_WINDOW,
        },
    }
}

/// `indexed` and `stored` of the field `name` in the schema `GetSchema`
/// returns.
async fn flags(index: &IndexService, name: &str) -> (Option<bool>, Option<bool>) {
    let schema = index
        .get_schema(Request::new(GetSchemaRequest {}))
        .await
        .unwrap()
        .into_inner()
        .schema
        .unwrap();
    match &schema.fields[name].option {
        Some(Opt::Text(o)) => (o.indexed, o.stored),
        Some(Opt::Integer(o)) => (o.indexed, o.stored),
        other => panic!("{name}: unexpected option {other:?}"),
    }
}

/// A text field and an integer field created with `indexed` and `stored`
/// left out are indexed and stored: a search finds the document, and the
/// document comes back with both values.
#[tokio::test(flavor = "multi_thread")]
async fn fields_created_without_indexed_or_stored_are_indexed_and_stored() {
    let dir = tempfile::TempDir::new().unwrap();
    let services = create_index(
        &dir,
        HashMap::from([
            ("body".to_string(), Opt::Text(v1::TextOption::default())),
            (
                "year".to_string(),
                Opt::Integer(v1::IntegerOption::default()),
            ),
        ]),
    )
    .await;
    assert_eq!(
        flags(&services.index, "body").await,
        (Some(true), Some(true))
    );
    assert_eq!(
        flags(&services.index, "year").await,
        (Some(true), Some(true))
    );

    let document = Document::builder()
        .add_text("body", "rust")
        .add_integer("year", 2024)
        .build();
    services
        .documents
        .put_documents(Request::new(PutDocumentsRequest {
            documents: vec![DocumentEntry {
                id: "doc1".to_string(),
                document: Some(doc_convert::to_proto(&document)),
            }],
        }))
        .await
        .unwrap();
    services
        .documents
        .commit(Request::new(CommitRequest {}))
        .await
        .unwrap();

    let hits: Vec<String> = services
        .search
        .search(Request::new(v1::SearchRequest {
            query: "body:rust".to_string(),
            limit: 10,
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner()
        .results
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    assert_eq!(hits, ["doc1"]);

    let stored = services
        .documents
        .get_documents(Request::new(GetDocumentsRequest {
            id: "doc1".to_string(),
        }))
        .await
        .unwrap()
        .into_inner()
        .documents;
    assert_eq!(stored.len(), 1, "{stored:?}");
    let mut names: Vec<&str> = stored[0].fields.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(names, ["body", "year"], "{stored:?}");
}

/// A field option with `indexed` and `stored` left out gets `true` for
/// both in `AddField` and in `UpdateField`, which replaces the whole
/// option.
#[tokio::test(flavor = "multi_thread")]
async fn added_and_updated_fields_without_indexed_or_stored_are_indexed_and_stored() {
    let dir = tempfile::TempDir::new().unwrap();
    let explicit = v1::TextOption {
        indexed: Some(true),
        stored: Some(true),
        ..Default::default()
    };
    let services = create_index(
        &dir,
        HashMap::from([("title".to_string(), Opt::Text(explicit))]),
    )
    .await;

    services
        .index
        .add_field(Request::new(AddFieldRequest {
            name: "body".to_string(),
            field_option: Some(v1::FieldOption {
                option: Some(Opt::Text(v1::TextOption::default())),
            }),
        }))
        .await
        .unwrap();
    assert_eq!(
        flags(&services.index, "body").await,
        (Some(true), Some(true))
    );

    services
        .index
        .update_field(Request::new(UpdateFieldRequest {
            name: "title".to_string(),
            field_option: Some(v1::FieldOption {
                option: Some(Opt::Text(v1::TextOption::default())),
            }),
            reindex: true,
            dry_run: false,
        }))
        .await
        .unwrap();
    assert_eq!(
        flags(&services.index, "title").await,
        (Some(true), Some(true))
    );
}

/// An explicit `false` is kept: the default applies only to an unset value.
#[tokio::test(flavor = "multi_thread")]
async fn explicit_false_indexed_and_stored_are_kept() {
    let dir = tempfile::TempDir::new().unwrap();
    let services = create_index(
        &dir,
        HashMap::from([
            (
                "body".to_string(),
                Opt::Text(v1::TextOption {
                    indexed: Some(false),
                    stored: Some(true),
                    ..Default::default()
                }),
            ),
            (
                "year".to_string(),
                Opt::Integer(v1::IntegerOption {
                    indexed: Some(true),
                    stored: Some(false),
                    ..Default::default()
                }),
            ),
        ]),
    )
    .await;
    assert_eq!(
        flags(&services.index, "body").await,
        (Some(false), Some(true))
    );
    assert_eq!(
        flags(&services.index, "year").await,
        (Some(true), Some(false))
    );
}
