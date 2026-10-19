//! Index management gRPC service.
//!
//! Handles index creation, metadata retrieval, and schema inspection through
//! the `IndexService` gRPC trait.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use laurus::{CommitPolicy, Engine, UpdateFieldOptions, WalSyncPolicy};

use crate::context;
use crate::convert::{error, schema as schema_convert};
use crate::proto::laurus::v1::{
    AddFieldRequest, AddFieldResponse, CreateIndexRequest, CreateIndexResponse, DeleteFieldRequest,
    DeleteFieldResponse, GetIndexRequest, GetIndexResponse, GetSchemaRequest, GetSchemaResponse,
    UpdateFieldRequest, UpdateFieldResponse, VectorFieldStats,
    index_service_server::IndexService as IndexServiceTrait,
};

/// gRPC IndexService implementation.
#[derive(Clone)]
pub struct IndexService {
    /// Shared, mutable reference to the current search engine instance.
    /// `None` when no index has been created yet.
    pub engine: Arc<RwLock<Option<Engine>>>,
    /// Filesystem path where the index data is persisted.
    pub data_dir: PathBuf,
    /// WAL durability policy applied to indices created via this service.
    pub wal_policy: WalSyncPolicy,
    /// Auto-commit policy applied to indices created via this service.
    pub commit_policy: CommitPolicy,
}

#[tonic::async_trait]
impl IndexServiceTrait for IndexService {
    /// Creates a new index with the given schema. Fails if an index already exists.
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        let req = request.into_inner();
        let proto_schema = req
            .schema
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("schema is required"))?;
        let schema = schema_convert::from_proto(proto_schema).map_err(Status::invalid_argument)?;
        // Here rather than in `context::create_index`, whose anyhow error
        // would be reported as Internal instead of InvalidArgument.
        schema.validate_for_create().map_err(error::to_status)?;

        let mut guard = self.engine.write().await;
        if guard.is_some() {
            return Err(Status::already_exists("Index already exists"));
        }

        let engine =
            context::create_index(&self.data_dir, &schema, self.wal_policy, self.commit_policy)
                .await
                .map_err(error::anyhow_to_status)?;
        *guard = Some(engine);

        tracing::info!("Index created at {}", self.data_dir.display());
        Ok(Response::new(CreateIndexResponse {}))
    }

    /// Returns index-level statistics such as document count and per-field vector stats.
    async fn get_index(
        &self,
        _request: Request<GetIndexRequest>,
    ) -> Result<Response<GetIndexResponse>, Status> {
        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let stats = engine.stats().map_err(error::to_status)?;

        let vector_fields = stats
            .vector_fields
            .iter()
            .map(|(name, fs)| {
                (
                    name.clone(),
                    VectorFieldStats {
                        vector_count: fs.vector_count as u64,
                        dimension: fs.dimension as u64,
                    },
                )
            })
            .collect();

        Ok(Response::new(GetIndexResponse {
            document_count: stats.document_count,
            vector_fields,
        }))
    }

    /// Returns the schema definition of the current index.
    async fn get_schema(
        &self,
        _request: Request<GetSchemaRequest>,
    ) -> Result<Response<GetSchemaResponse>, Status> {
        let schema = context::read_schema(&self.data_dir).map_err(error::anyhow_to_status)?;
        let proto_schema = schema_convert::to_proto(&schema);
        Ok(Response::new(GetSchemaResponse {
            schema: Some(proto_schema),
        }))
    }

    /// Dynamically adds a new field to the current index and persists the updated schema.
    async fn add_field(
        &self,
        request: Request<AddFieldRequest>,
    ) -> Result<Response<AddFieldResponse>, Status> {
        let req = request.into_inner();
        let name = req.name;
        if name.is_empty() {
            return Err(Status::invalid_argument("field name is required"));
        }
        let proto_field_option = req
            .field_option
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("field_option is required"))?;
        let field_option = schema_convert::field_option_from_proto(proto_field_option)
            .map_err(Status::invalid_argument)?;

        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let updated_schema = engine
            .add_field(&name, field_option)
            .await
            .map_err(error::to_status)?;

        tracing::info!("Field '{}' added to index", name);
        let proto_schema = schema_convert::to_proto(&updated_schema);
        Ok(Response::new(AddFieldResponse {
            schema: Some(proto_schema),
        }))
    }

    /// Removes a field from the current index schema and persists the updated schema.
    async fn delete_field(
        &self,
        request: Request<DeleteFieldRequest>,
    ) -> Result<Response<DeleteFieldResponse>, Status> {
        let req = request.into_inner();
        let name = req.name;
        if name.is_empty() {
            return Err(Status::invalid_argument("field name is required"));
        }

        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let updated_schema = engine.delete_field(&name).await.map_err(error::to_status)?;

        tracing::info!("Field '{}' deleted from index", name);
        let proto_schema = schema_convert::to_proto(&updated_schema);
        Ok(Response::new(DeleteFieldResponse {
            schema: Some(proto_schema),
        }))
    }

    /// Changes an existing field's type/options, optionally rebuilding or
    /// discarding its on-disk data, and persists the updated schema.
    async fn update_field(
        &self,
        request: Request<UpdateFieldRequest>,
    ) -> Result<Response<UpdateFieldResponse>, Status> {
        let req = request.into_inner();
        let name = req.name;
        if name.is_empty() {
            return Err(Status::invalid_argument("field name is required"));
        }
        let proto_field_option = req
            .field_option
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("field_option is required"))?;
        let field_option = schema_convert::field_option_from_proto(proto_field_option)
            .map_err(Status::invalid_argument)?;

        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let outcome = engine
            .update_field(
                &name,
                field_option,
                UpdateFieldOptions {
                    reindex: req.reindex,
                    dry_run: req.dry_run,
                },
            )
            .await
            .map_err(error::to_status)?;

        tracing::info!(
            "Field '{}' updated (classification: {:?})",
            name,
            outcome.classification
        );
        let proto_schema = schema_convert::to_proto(&outcome.schema);
        Ok(Response::new(UpdateFieldResponse {
            classification: schema_convert::field_change_kind_to_proto(outcome.classification)
                as i32,
            schema: Some(proto_schema),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::proto::laurus::v1;
    use laurus::Schema;

    /// An `IndexService` over a fresh index created from `schema_toml`. The
    /// returned directory must outlive the service.
    async fn service_with_index(schema_toml: &str) -> (tempfile::TempDir, IndexService) {
        let dir = tempfile::tempdir().unwrap();
        let service = IndexService {
            engine: Arc::new(RwLock::new(None)),
            data_dir: dir.path().join("data"),
            wal_policy: WalSyncPolicy::PerRecord,
            commit_policy: CommitPolicy::Manual,
        };
        let schema = Schema::from_toml(schema_toml).unwrap();
        service
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema_convert::to_proto(&schema)),
            }))
            .await
            .unwrap();
        (dir, service)
    }

    /// The schema `GetSchema` returns, which is read back from disk.
    async fn schema_of(service: &IndexService) -> v1::Schema {
        service
            .get_schema(Request::new(GetSchemaRequest {}))
            .await
            .unwrap()
            .into_inner()
            .schema
            .unwrap()
    }

    /// The proto option of an HNSW field whose quantizer carries the raw
    /// `method` value, which may be one this build does not know.
    fn hnsw_with_quantizer_method(method: i32) -> v1::FieldOption {
        v1::FieldOption {
            option: Some(v1::field_option::Option::Hnsw(v1::HnswOption {
                dimension: 8,
                quantizer: Some(v1::QuantizationConfig {
                    method,
                    subvector_count: 4,
                }),
                ..Default::default()
            })),
        }
    }

    /// Issue #1310: CreateIndex rejects an `[analyzers.*]` entry named after
    /// a built-in analyzer as InvalidArgument, and writes nothing.
    #[tokio::test]
    async fn create_index_rejects_reserved_analyzer_name() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let service = IndexService {
            engine: Arc::new(RwLock::new(None)),
            data_dir: data_dir.clone(),
            wal_policy: WalSyncPolicy::PerRecord,
            commit_policy: CommitPolicy::Manual,
        };
        let schema = Schema::from_toml(
            r#"
            [analyzers.standard]
            tokenizer = { type = "whitespace" }

            [fields.body.Text]
            analyzer = "standard"
            "#,
        )
        .unwrap();

        let status = service
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema_convert::to_proto(&schema)),
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status
                .message()
                .contains("Analyzer name 'standard' is reserved for a built-in analyzer"),
            "got: {}",
            status.message()
        );
        assert!(!data_dir.exists(), "a rejected create must write nothing");
        assert!(service.engine.read().await.is_none());
    }

    /// Issue #1329: CreateIndex rejects a `_`-prefixed field other than
    /// `_id` as InvalidArgument, and writes nothing.
    #[tokio::test]
    async fn create_index_rejects_reserved_field_name() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let service = IndexService {
            engine: Arc::new(RwLock::new(None)),
            data_dir: data_dir.clone(),
            wal_policy: WalSyncPolicy::PerRecord,
            commit_policy: CommitPolicy::Manual,
        };
        let schema = Schema::from_toml(
            r#"
            [fields._secret.Text]
            "#,
        )
        .unwrap();

        let status = service
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema_convert::to_proto(&schema)),
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status
                .message()
                .contains("Field name '_secret' is reserved"),
            "got: {}",
            status.message()
        );
        assert!(!data_dir.exists(), "a rejected create must write nothing");
        assert!(service.engine.read().await.is_none());
    }

    /// Issue #1457: CreateIndex rejects a Flat field with a product
    /// quantizer as InvalidArgument, and writes nothing.
    #[tokio::test]
    async fn create_index_rejects_product_quantization_on_flat() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let service = IndexService {
            engine: Arc::new(RwLock::new(None)),
            data_dir: data_dir.clone(),
            wal_policy: WalSyncPolicy::PerRecord,
            commit_policy: CommitPolicy::Manual,
        };
        let schema = Schema::from_toml(
            r#"
            [fields.vec.Flat]
            dimension = 8

            [fields.vec.Flat.quantizer.ProductQuantization]
            subvector_count = 4
            "#,
        )
        .unwrap();

        let status = service
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(schema_convert::to_proto(&schema)),
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status
                .message()
                .contains("field 'vec': Flat does not support quantizer ProductQuantization"),
            "got: {}",
            status.message()
        );
        assert!(!data_dir.exists(), "a rejected create must write nothing");
        assert!(service.engine.read().await.is_none());
    }

    /// Issue #1459: CreateIndex rejects a quantizer method this server does
    /// not know as InvalidArgument, instead of storing the field as
    /// Scalar8Bit, and writes nothing.
    #[tokio::test]
    async fn create_index_rejects_unknown_quantization_method() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let service = IndexService {
            engine: Arc::new(RwLock::new(None)),
            data_dir: data_dir.clone(),
            wal_policy: WalSyncPolicy::PerRecord,
            commit_policy: CommitPolicy::Manual,
        };

        let status = service
            .create_index(Request::new(CreateIndexRequest {
                schema: Some(v1::Schema {
                    fields: HashMap::from([("vec".to_string(), hnsw_with_quantizer_method(99))]),
                    ..Default::default()
                }),
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(
            status.message(),
            "Field 'vec': unknown quantization method 99"
        );
        assert!(!data_dir.exists(), "a rejected create must write nothing");
        assert!(service.engine.read().await.is_none());
    }

    /// Issue #1459: AddField rejects a quantizer method this server does not
    /// know as InvalidArgument and leaves the schema as it was.
    #[tokio::test]
    async fn add_field_rejects_unknown_quantization_method() {
        let (_dir, service) = service_with_index(
            r#"
            [fields.body.Text]
            "#,
        )
        .await;
        let before = schema_of(&service).await;

        let status = service
            .add_field(Request::new(AddFieldRequest {
                name: "vec".to_string(),
                field_option: Some(hnsw_with_quantizer_method(99)),
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "unknown quantization method 99");
        assert_eq!(schema_of(&service).await, before);
    }

    /// Issue #1459: UpdateField rejects a quantizer method this server does
    /// not know as InvalidArgument and leaves the field as it was.
    #[tokio::test]
    async fn update_field_rejects_unknown_quantization_method() {
        let (_dir, service) = service_with_index(
            r#"
            [fields.vec.Hnsw]
            dimension = 8
            "#,
        )
        .await;
        let before = schema_of(&service).await;

        let status = service
            .update_field(Request::new(UpdateFieldRequest {
                name: "vec".to_string(),
                field_option: Some(hnsw_with_quantizer_method(99)),
                reindex: true,
                dry_run: false,
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "unknown quantization method 99");
        assert_eq!(schema_of(&service).await, before);
    }

    /// Issue #1459: a server built without `pq-fastscan` rejects a FastScan
    /// quantizer as InvalidArgument, as the proto comment says, instead of
    /// adding the field as Scalar8Bit.
    #[cfg(not(feature = "pq-fastscan"))]
    #[tokio::test]
    async fn add_field_rejects_fastscan_without_the_feature() {
        let (_dir, service) = service_with_index(
            r#"
            [fields.body.Text]
            "#,
        )
        .await;
        let before = schema_of(&service).await;

        let status = service
            .add_field(Request::new(AddFieldRequest {
                name: "vec".to_string(),
                field_option: Some(hnsw_with_quantizer_method(
                    v1::QuantizationMethod::ProductQuantizationFastscan as i32,
                )),
            }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("`pq-fastscan` feature"),
            "got: {}",
            status.message()
        );
        assert_eq!(schema_of(&service).await, before);
    }
}
