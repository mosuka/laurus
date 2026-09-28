//! Search gRPC service.
//!
//! Provides unary and server-streaming RPCs for executing lexical, vector,
//! and hybrid search queries against the index. The unified query DSL
//! (including vector clauses like `field:"text"`) is handled by the engine
//! internally — no query-syntax branching is needed in the service layer.

use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use laurus::Engine;

use crate::convert::{error, search as search_convert};
use crate::proto::laurus::v1::{
    SearchBatchRequest, SearchBatchResponse, SearchRequest, SearchResponse, SearchResult,
    search_service_server::SearchService as SearchServiceTrait,
};

/// gRPC SearchService implementation.
#[derive(Clone)]
pub struct SearchService {
    /// Shared, mutable reference to the current search engine instance.
    /// `None` when no index has been created yet.
    pub engine: Arc<RwLock<Option<Engine>>>,
}

#[tonic::async_trait]
impl SearchServiceTrait for SearchService {
    /// Executes a search query and returns all results in a single response.
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let search_request = search_convert::from_proto(&req)?;

        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let results = engine
            .search(search_request)
            .await
            .map_err(error::to_status)?;
        let total_hits = results.len() as u64;
        let results: Vec<SearchResult> = results
            .iter()
            .map(search_convert::result_to_proto)
            .collect();

        Ok(Response::new(SearchResponse {
            results,
            total_hits,
        }))
    }

    type SearchStreamStream = ReceiverStream<Result<SearchResult, Status>>;

    /// Executes a search query and streams results back one at a time.
    async fn search_stream(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<Self::SearchStreamStream>, Status> {
        let req = request.into_inner();
        let search_request = search_convert::from_proto(&req)?;

        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let results = engine
            .search(search_request)
            .await
            .map_err(error::to_status)?;

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            for result in &results {
                let proto = search_convert::result_to_proto(result);
                if tx.send(Ok(proto)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    /// Executes multiple independent search queries in a single round trip.
    ///
    /// Each `SearchRequest` in `queries` is dispatched in parallel on the
    /// server via [`laurus::Engine::search_batch`]. Returns one
    /// `SearchResponse` per input query, in the same order. Empty input
    /// short-circuits to an empty `results` list without invoking the
    /// engine.
    ///
    /// Issue [#716](https://github.com/mosuka/laurus/issues/716)
    /// Phase 3a of [#648](https://github.com/mosuka/laurus/issues/648).
    async fn search_batch(
        &self,
        request: Request<SearchBatchRequest>,
    ) -> Result<Response<SearchBatchResponse>, Status> {
        let req = request.into_inner();

        if req.queries.is_empty() {
            return Ok(Response::new(SearchBatchResponse {
                results: Vec::new(),
            }));
        }

        let search_requests: Vec<laurus::SearchRequest> = req
            .queries
            .iter()
            .map(search_convert::from_proto)
            .collect::<Result<Vec<_>, _>>()?;

        let guard = self.engine.read().await;
        let engine = guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open"))?;

        let batch_results = engine
            .search_batch(search_requests)
            .await
            .map_err(error::to_status)?;

        let results: Vec<SearchResponse> = batch_results
            .into_iter()
            .map(|per_query_results| {
                let total_hits = per_query_results.len() as u64;
                let proto_results: Vec<SearchResult> = per_query_results
                    .iter()
                    .map(search_convert::result_to_proto)
                    .collect();
                SearchResponse {
                    results: proto_results,
                    total_hits,
                }
            })
            .collect();

        Ok(Response::new(SearchBatchResponse { results }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::storage::memory::MemoryStorage;
    use laurus::{Schema, Storage, TextOption};

    async fn service_with_title_field() -> SearchService {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_text_field("title", TextOption::default())
            .build();
        let engine = Engine::builder(storage, schema).build().await.unwrap();
        SearchService {
            engine: Arc::new(RwLock::new(Some(engine))),
        }
    }

    /// Issue #1253: a DSL query that names an undeclared field is the
    /// caller's mistake, not a server failure.
    #[tokio::test]
    async fn unknown_field_query_is_invalid_argument() {
        let service = service_with_title_field().await;

        let status = service
            .search(Request::new(SearchRequest {
                query: "nope:rust".to_string(),
                limit: 10,
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("unknown field"),
            "unexpected message: {}",
            status.message()
        );
    }
}
