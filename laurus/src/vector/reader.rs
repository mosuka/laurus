//! Vector index reader traits and implementations.
//!
//! This module defines the [`VectorIndexReader`] trait, which provides a
//! uniform interface for reading vectors from any index type (Flat, HNSW, IVF),
//! as well as the [`VectorIterator`] trait for sequential traversal.
//!
//! A concrete [`SimpleVectorReader`] is provided for in-memory use cases, and
//! [`VectorIndexReaderFactory`] serves as the entry point for constructing
//! readers from serialized index data.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;

/// Statistics about a vector index.
#[derive(Debug, Clone)]
pub struct VectorStats {
    /// Total number of vectors in the index.
    pub vector_count: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Index memory usage in bytes.
    pub memory_usage: usize,
    /// Build time in milliseconds.
    pub build_time_ms: u64,
}

/// Metadata about a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexMetadata {
    /// Index type (HNSW, Flat, IVF, etc.).
    pub index_type: String,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp.
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Index version.
    pub version: String,
    /// Build configuration.
    pub build_config: serde_json::Value,
    /// Custom metadata.
    pub custom_metadata: std::collections::HashMap<String, String>,
}

/// Index validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the index is valid.
    pub is_valid: bool,
    /// Validation errors found.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
    /// Repair suggestions.
    pub repair_suggestions: Vec<String>,
}

/// Trait for reading vector indexes (similar to IndexReader for inverted indexes).
///
/// Implementations provide read-only access to the vectors stored in an index.
/// Concrete readers exist for each index type (Flat, HNSW, IVF) as well as
/// the in-memory [`SimpleVectorReader`].
///
/// Note that this trait does **not** include a `search()` method; similarity
/// search is handled by
/// [`VectorIndexSearcher`](crate::vector::search::searcher::VectorIndexSearcher).
pub trait VectorIndexReader: Send + Sync + std::fmt::Debug {
    /// Cast to Any for downcasting to concrete types.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get a vector by document ID and field name.
    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>>;

    /// Get all vectors for a document ID (all fields).
    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>>;

    /// Get multiple vectors by document IDs and field names.
    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>>;

    /// Get all vector IDs with their field names in the index.
    fn vector_ids(&self) -> Result<Vec<(u64, String)>>;

    /// Doc IDs that belong to a specific field.
    ///
    /// Concrete readers should override with an O(1) lookup into a
    /// pre-built per-field index. The default implementation falls back
    /// to filtering [`vector_ids`](Self::vector_ids) every call, which
    /// allocates a fresh `Vec<(u64, String)>` of size `total_vectors` —
    /// the cost #405 (per-field `vector_ids` cache) targets.
    ///
    /// Returns an `Arc<[u64]>` so a cached implementation can hand back
    /// a refcount bump rather than a fresh allocation. An empty
    /// `Arc<[u64]>` is returned when the field has no vectors.
    fn doc_ids_for_field(&self, field_name: &str) -> Arc<[u64]> {
        self.vector_ids()
            .map(|all| {
                all.into_iter()
                    .filter_map(|(id, f)| (f == field_name).then_some(id))
                    .collect::<Vec<u64>>()
                    .into()
            })
            .unwrap_or_else(|_| Vec::new().into())
    }

    /// Get the total number of vectors.
    fn vector_count(&self) -> usize;

    /// Get the vector dimension.
    fn dimension(&self) -> usize;

    /// Get the distance metric used.
    fn distance_metric(&self) -> DistanceMetric;

    /// Get index statistics.
    ///
    /// Returns a [`VectorStats`] containing the total vector count, vector
    /// dimension, estimated memory usage in bytes, and build time.
    fn stats(&self) -> VectorStats;

    /// Check if a vector exists for a specific field.
    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool;

    /// Get vectors in a specific range with field names.
    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>>;

    /// Get all vectors with a specific field name.
    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>>;

    /// Get all unique field names in the index.
    fn field_names(&self) -> Result<Vec<String>>;

    /// Get an iterator over all vectors.
    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>>;

    /// Get index metadata.
    fn metadata(&self) -> Result<VectorIndexMetadata>;

    /// Validate index integrity.
    ///
    /// Checks that all stored vectors are valid (e.g., no NaN or infinite
    /// values) and returns a [`ValidationReport`] with any errors, warnings,
    /// and repair suggestions.
    fn validate(&self) -> Result<ValidationReport>;
}

/// Iterator over vectors in an index.
pub trait VectorIterator: Send {
    /// Get the next vector with field name.
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>>;

    /// Skip to a specific document ID and field.
    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool>;

    /// Get the current position.
    fn position(&self) -> (u64, String);

    /// Reset to the beginning.
    fn reset(&mut self) -> Result<()>;
}

/// Simple in-memory vector reader for basic use cases.
/// This is a lightweight reader that holds vectors in memory.
#[derive(Debug)]
pub struct SimpleVectorReader {
    vectors: HashMap<(u64, String), Vector>,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl SimpleVectorReader {
    /// Create a new simple vector reader.
    pub fn new(
        vectors: Vec<(u64, String, Vector)>,
        dimension: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        let vector_ids: Vec<(u64, String)> = vectors
            .iter()
            .map(|(id, field, _)| (*id, field.clone()))
            .collect();
        let vectors: HashMap<(u64, String), Vector> = vectors
            .into_iter()
            .map(|(id, field, vec)| ((id, field), vec))
            .collect();

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
            deletion_bitmap: None,
        })
    }

    /// Attach a deletion bitmap so that deleted documents are transparently
    /// excluded from query results.
    ///
    /// # Arguments
    ///
    /// * `bitmap` - Shared deletion bitmap indicating which document IDs have
    ///   been logically deleted.
    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    fn is_deleted(&self, doc_id: u64) -> bool {
        if let Some(bitmap) = &self.deletion_bitmap {
            bitmap.is_deleted(doc_id)
        } else {
            false
        }
    }
}

impl VectorIndexReader for SimpleVectorReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        if self.is_deleted(doc_id) {
            return Ok(None);
        }
        Ok(self.vectors.get(&(doc_id, field_name.to_string())).cloned())
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((id, _), _)| *id == doc_id)
            .map(|((_, field), vec)| (field.clone(), vec.clone()))
            .collect())
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        Ok(doc_ids
            .iter()
            .map(|(id, field)| self.vectors.get(&(*id, field.clone())).cloned())
            .collect())
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        Ok(self.vector_ids.clone())
    }

    fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        let memory_usage = self.vectors.len() * (8 + self.dimension * 4);
        VectorStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            memory_usage,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.vectors.contains_key(&(doc_id, field_name.to_string()))
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((id, _), _)| *id >= start_doc_id && *id <= end_doc_id)
            .map(|((id, field), v)| (*id, field.clone(), v.clone()))
            .collect())
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((_, field), _)| field == field_name)
            .map(|((id, _), vec)| (*id, vec.clone()))
            .collect())
    }

    fn field_names(&self) -> Result<Vec<String>> {
        use std::collections::HashSet;
        let fields: HashSet<String> = self
            .vectors
            .keys()
            .map(|(_, field)| field.clone())
            .collect();
        Ok(fields.into_iter().collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        // Issue #1155 (mirrors #1152's IVF fix): `self.vectors` is a
        // `HashMap`, so iterating it directly yields an arbitrary,
        // unspecified order -- but `SimpleVectorIterator::skip_to`
        // assumes global ascending `(doc_id, field_name)` order. Sort
        // the collected `Vec` before constructing the iterator.
        let mut entries: Vec<(u64, String, Vector)> = self
            .vectors
            .iter()
            .map(|((id, field), v)| (*id, field.clone(), v.clone()))
            .collect();
        entries.sort_by(|(id_a, field_a, _), (id_b, field_b, _)| {
            id_a.cmp(id_b).then_with(|| field_a.cmp(field_b))
        });
        Ok(Box::new(SimpleVectorIterator::new(entries)))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "Simple".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1.0".to_string(),
            build_config: serde_json::json!({}),
            custom_metadata: HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let all_valid = self.vectors.values().all(|v| v.is_valid());
        Ok(ValidationReport {
            is_valid: all_valid,
            errors: vec![],
            warnings: vec![],
            repair_suggestions: vec![],
        })
    }
}

/// Simple vector iterator.
pub struct SimpleVectorIterator {
    vectors: Vec<(u64, String, Vector)>,
    position: usize,
}

impl SimpleVectorIterator {
    /// Create a new simple vector iterator.
    pub fn new(vectors: Vec<(u64, String, Vector)>) -> Self {
        Self {
            vectors,
            position: 0,
        }
    }
}

impl VectorIterator for SimpleVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        if self.position < self.vectors.len() {
            let result = self.vectors[self.position].clone();
            self.position += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.position < self.vectors.len() {
            let (id, field, _) = &self.vectors[self.position];
            if *id > doc_id || (*id == doc_id && field.as_str() >= field_name) {
                return Ok(true);
            }
            self.position += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.position < self.vectors.len() {
            let (id, field, _) = &self.vectors[self.position];
            (*id, field.clone())
        } else {
            (u64::MAX, String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
}

/// Factory for creating vector index readers.
///
/// This provides a single entry point for constructing concrete reader
/// implementations (Flat, HNSW, IVF) from serialized index data.
pub struct VectorIndexReaderFactory;

impl VectorIndexReaderFactory {
    /// Create a reader for a specific index type from serialized bytes.
    pub fn create_reader(
        index_type: &str,
        index_data: &[u8],
        deletion_bitmap: Option<Arc<DeletionBitmap>>,
    ) -> Result<Arc<dyn VectorIndexReader>> {
        use crate::vector::index::flat::reader::FlatVectorIndexReader;
        use crate::vector::index::hnsw::reader::HnswIndexReader;
        use crate::vector::index::ivf::reader::IvfIndexReader;

        match index_type.to_lowercase().as_str() {
            "flat" => {
                let mut reader = FlatVectorIndexReader::from_bytes(index_data)?;
                if let Some(bitmap) = deletion_bitmap {
                    reader.set_deletion_bitmap(bitmap);
                }
                Ok(Arc::new(reader))
            }
            "hnsw" => {
                let mut reader = HnswIndexReader::from_bytes(index_data)?;
                if let Some(bitmap) = deletion_bitmap {
                    reader.set_deletion_bitmap(bitmap);
                }
                Ok(Arc::new(reader))
            }
            "ivf" => {
                let mut reader = IvfIndexReader::from_bytes(index_data)?;
                if let Some(bitmap) = deletion_bitmap {
                    reader.set_deletion_bitmap(bitmap);
                }
                Ok(Arc::new(reader))
            }
            _ => Err(crate::error::LaurusError::InvalidOperation(format!(
                "Unknown index type: {index_type}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corpus spread across enough distinct doc_id/field pairs that
    /// `HashMap` iteration order is vanishingly unlikely to coincidentally
    /// already be `(doc_id, field)`-sorted, so these tests reliably
    /// exercise Issue #1155's bug rather than passing by luck.
    fn build_multi_field_reader() -> SimpleVectorReader {
        let mut vectors = Vec::new();
        for doc_id in 0..10u64 {
            for field in ["alpha", "beta", "gamma"] {
                vectors.push((doc_id, field.to_string(), Vector::new(vec![doc_id as f32])));
            }
        }
        SimpleVectorReader::new(vectors, 1, DistanceMetric::Euclidean).unwrap()
    }

    /// Issue #1155: `SimpleVectorIterator`'s backing `Vec` was built by
    /// iterating a `HashMap` directly, giving an arbitrary order --
    /// `.next()` must yield entries in globally ascending `(doc_id,
    /// field)` order regardless.
    #[test]
    fn vector_iterator_yields_globally_ascending_doc_id_and_field_despite_hashmap_order() {
        let reader = build_multi_field_reader();

        let mut iter = reader.vector_iterator().unwrap();
        let mut seen = Vec::new();
        while let Some((doc_id, field, _)) = iter.next().unwrap() {
            seen.push((doc_id, field));
        }
        assert_eq!(seen.len(), 30);
        let mut sorted = seen.clone();
        sorted.sort();
        assert_eq!(
            seen, sorted,
            "vector_iterator must yield (doc_id, field) pairs in globally ascending order"
        );
    }

    /// Issue #1155: `skip_to(doc_id, field)` must return the correct
    /// lower-bound key regardless of the backing `HashMap`'s iteration
    /// order.
    #[test]
    fn skip_to_returns_correct_lower_bound_despite_hashmap_order() {
        let reader = build_multi_field_reader();

        // Brute-force expectation: every (doc_id, field) pair, globally
        // sorted (matches build_multi_field_reader's construction).
        let mut expected: Vec<(u64, &str)> = Vec::new();
        for doc_id in 0..10u64 {
            for field in ["alpha", "beta", "gamma"] {
                expected.push((doc_id, field));
            }
        }
        expected.sort();

        let targets: Vec<(u64, &str)> = vec![
            (0, "alpha"),
            (0, "aardvark"), // before "alpha" for doc_id 0
            (0, "zzz"),      // after "gamma" for doc_id 0 -> next is (1, "alpha")
            (4, "beta"),
            (9, "gamma"),
            (9, "zzz"), // past the last entry entirely
            (10, "alpha"),
        ];

        for (target_id, target_field) in targets {
            let mut iter = reader.vector_iterator().unwrap();
            let found = iter.skip_to(target_id, target_field).unwrap();
            let expected_next = expected
                .iter()
                .find(|&&(id, field)| (id, field) >= (target_id, target_field));
            match expected_next {
                Some(&(expected_id, expected_field)) => {
                    assert!(
                        found,
                        "expected to find a key >= ({target_id}, {target_field})"
                    );
                    assert_eq!(
                        iter.position(),
                        (expected_id, expected_field.to_string()),
                        "skip_to({target_id}, {target_field}) landed on the wrong key"
                    );
                }
                None => {
                    assert!(
                        !found,
                        "no key >= ({target_id}, {target_field}) should exist"
                    );
                }
            }
        }
    }
}
