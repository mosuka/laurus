//! Read side of the multi-vector index: a point-in-time view of its
//! segments, the [`VectorIndexReader`] facade, and the searcher that rejects
//! vector search.

use std::borrow::Cow;
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::multivector::format::SegmentReader;
use crate::vector::reader::{
    ValidationReport, VectorIndexMetadata, VectorIndexReader, VectorIterator, VectorStats,
};
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher,
};

/// The sealed segments of a multi-vector index at one point in time.
///
/// A document's vectors come from the newest segment that holds it, and a
/// logically deleted document has none, the same newest-wins rule the
/// single-vector segment fan-out applies.
#[derive(Debug, Clone)]
pub struct MultiVectorSnapshot {
    dimension: usize,
    distance: DistanceMetric,
    /// Newest generation first.
    segments: Vec<Arc<SegmentReader>>,
    deletion: Option<Arc<DeletionBitmap>>,
}

impl MultiVectorSnapshot {
    pub(crate) fn new(
        dimension: usize,
        distance: DistanceMetric,
        segments: Vec<Arc<SegmentReader>>,
        deletion: Option<Arc<DeletionBitmap>>,
    ) -> Self {
        Self {
            dimension,
            distance,
            segments,
            deletion,
        }
    }

    /// Dimension of every token vector.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Token similarity of the field. Stored vectors are already
    /// L2-normalized when this is [`DistanceMetric::Cosine`].
    pub fn distance(&self) -> DistanceMetric {
        self.distance
    }

    /// The token vectors of `doc_id`, row-major (`vector count × dimension`
    /// values), or `None` when the document has none or is deleted.
    ///
    /// # Errors
    ///
    /// Returns an error when reading a segment fails.
    pub fn vectors(&self, doc_id: u64) -> Result<Option<Cow<'_, [f32]>>> {
        if self.is_deleted(doc_id) {
            return Ok(None);
        }
        for segment in &self.segments {
            if let Some(vectors) = segment.vectors(doc_id)? {
                return Ok(Some(vectors));
            }
        }
        Ok(None)
    }

    fn is_deleted(&self, doc_id: u64) -> bool {
        self.deletion
            .as_ref()
            .is_some_and(|bitmap| bitmap.is_deleted(doc_id))
    }

    /// Ids of the documents that hold token vectors, ascending.
    fn live_doc_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .segments
            .iter()
            .flat_map(|segment| segment.doc_ids().iter().copied())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.retain(|&id| !self.is_deleted(id));
        ids
    }
}

/// [`VectorIndexReader`] over a multi-vector index.
///
/// Token vectors are not single vectors, so the per-vector accessors report
/// nothing; the document-level ones (`vector_count`, `doc_ids_for_field`,
/// `contains_vector`) count documents. Late-interaction rescoring reads the
/// vectors through [`MultiVectorSnapshot`] instead.
#[derive(Debug)]
pub(crate) struct MultiVectorReaderFacade {
    snapshot: MultiVectorSnapshot,
    live_doc_ids: Arc<[u64]>,
}

impl MultiVectorReaderFacade {
    pub(crate) fn new(snapshot: MultiVectorSnapshot) -> Self {
        let live_doc_ids = snapshot.live_doc_ids().into();
        Self {
            snapshot,
            live_doc_ids,
        }
    }
}

impl VectorIndexReader for MultiVectorReaderFacade {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, _doc_id: u64, _field_name: &str) -> Result<Option<Vector>> {
        Ok(None)
    }

    fn get_vectors_for_doc(&self, _doc_id: u64) -> Result<Vec<(String, Vector)>> {
        Ok(Vec::new())
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        Ok(vec![None; doc_ids.len()])
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        Ok(Vec::new())
    }

    fn doc_ids_for_field(&self, _field_name: &str) -> Arc<[u64]> {
        self.live_doc_ids.clone()
    }

    fn vector_count(&self) -> usize {
        self.live_doc_ids.len()
    }

    fn dimension(&self) -> usize {
        self.snapshot.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.snapshot.distance
    }

    fn stats(&self) -> VectorStats {
        VectorStats {
            vector_count: self.live_doc_ids.len(),
            dimension: self.snapshot.dimension,
            memory_usage: 0,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, _field_name: &str) -> bool {
        self.live_doc_ids.binary_search(&doc_id).is_ok()
    }

    fn get_vector_range(
        &self,
        _start_doc_id: u64,
        _end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        Ok(Vec::new())
    }

    fn get_vectors_by_field(&self, _field_name: &str) -> Result<Vec<(u64, Vector)>> {
        Ok(Vec::new())
    }

    fn field_names(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(EmptyVectorIterator))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        let now = chrono::Utc::now();
        Ok(VectorIndexMetadata {
            index_type: "MultiVector".to_string(),
            created_at: now,
            modified_at: now,
            version: "1".to_string(),
            build_config: serde_json::json!({
                "dimension": self.snapshot.dimension,
                "distance": self.snapshot.distance.name(),
            }),
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let errors: Vec<String> = self
            .snapshot
            .segments
            .iter()
            .filter_map(|segment| segment.verify_payload().err())
            .map(|e| e.to_string())
            .collect();
        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            repair_suggestions: Vec::new(),
        })
    }
}

/// Iterator over the (no) single vectors of a multi-vector index.
struct EmptyVectorIterator;

impl VectorIterator for EmptyVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        Ok(None)
    }

    fn skip_to(&mut self, _doc_id: u64, _field_name: &str) -> Result<bool> {
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        (0, String::new())
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Searcher of a multi-vector index: token vectors are read by
/// late-interaction rescoring, never searched directly.
#[derive(Debug)]
pub(crate) struct MultiVectorSearcher;

impl MultiVectorSearcher {
    fn rejected() -> LaurusError {
        LaurusError::invalid_argument(
            "a MultiVector field is not a vector-search target; its token vectors are \
             read by late-interaction rescoring",
        )
    }
}

impl VectorIndexSearcher for MultiVectorSearcher {
    fn search(&self, _request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        Err(Self::rejected())
    }

    fn count(&self, _request: VectorIndexQuery) -> Result<u64> {
        Err(Self::rejected())
    }
}
