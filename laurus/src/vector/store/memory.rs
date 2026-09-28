//! VectorStore インメモリフィールド実装
//!
//! このモジュールはインメモリでベクトルを管理するフィールド実装を提供する。

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use crate::error::{LaurusError, Result};
use crate::vector::core::vector::StoredVector;
use crate::vector::core::vector::Vector;
use crate::vector::index::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorField, VectorFieldReader,
    VectorFieldStats, VectorFieldWriter,
};
use crate::vector::store::config::VectorFieldConfig;

#[derive(Clone, Debug)]
pub struct FieldHandle {
    pub field: Arc<dyn VectorField>,
    pub runtime: Arc<FieldRuntime>,
}

#[derive(Debug)]
pub struct FieldRuntime {
    default_reader: Arc<dyn VectorFieldReader>,
    current_reader: RwLock<Arc<dyn VectorFieldReader>>,
    writer: Arc<dyn VectorFieldWriter>,
}

impl FieldRuntime {
    pub fn new(reader: Arc<dyn VectorFieldReader>, writer: Arc<dyn VectorFieldWriter>) -> Self {
        Self {
            current_reader: RwLock::new(reader.clone()),
            default_reader: reader,
            writer,
        }
    }

    pub fn from_field(field: &Arc<dyn VectorField>) -> Arc<Self> {
        Arc::new(Self::new(field.reader_handle(), field.writer_handle()))
    }

    pub fn reader(&self) -> Arc<dyn VectorFieldReader> {
        self.current_reader.read().clone()
    }

    pub fn writer(&self) -> Arc<dyn VectorFieldWriter> {
        self.writer.clone()
    }

    pub fn replace_reader(&self, reader: Arc<dyn VectorFieldReader>) -> Arc<dyn VectorFieldReader> {
        let mut guard = self.current_reader.write();
        std::mem::replace(&mut *guard, reader)
    }

    pub fn reset_reader(&self) -> Arc<dyn VectorFieldReader> {
        self.replace_reader(self.default_reader.clone())
    }
}

#[derive(Debug)]
pub struct InMemoryVectorField {
    name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    writer: Arc<InMemoryFieldWriter>,
    reader: Arc<InMemoryFieldReader>,
}

impl InMemoryVectorField {
    pub fn new(
        name: String,
        config: VectorFieldConfig,
        delegate_writer: Option<Arc<dyn VectorFieldWriter>>,
        delegate_reader: Option<Arc<dyn VectorFieldReader>>,
    ) -> Result<Self> {
        let store = Arc::new(FieldStore::default());
        let writer = Arc::new(InMemoryFieldWriter::new(
            name.clone(),
            config.clone(),
            store.clone(),
            delegate_writer,
        ));
        let reader = Arc::new(InMemoryFieldReader::new(
            name.clone(),
            config.clone(),
            store.clone(),
            delegate_reader,
        ));
        Ok(Self {
            name,
            config,
            store,
            writer,
            reader,
        })
    }

    pub fn vector_tuples(&self) -> Vec<(u64, String, Vector)> {
        self.store.vector_tuples(&self.name)
    }
}

#[async_trait]
impl VectorField for InMemoryVectorField {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &VectorFieldConfig {
        &self.config
    }

    fn writer(&self) -> &dyn VectorFieldWriter {
        self.writer.as_ref()
    }

    fn reader(&self) -> &dyn VectorFieldReader {
        self.reader.as_ref()
    }

    fn writer_handle(&self) -> Arc<dyn VectorFieldWriter> {
        self.writer.clone()
    }

    fn reader_handle(&self) -> Arc<dyn VectorFieldReader> {
        self.reader.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub(crate) struct InMemoryFieldWriter {
    field_name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    delegate: Option<Arc<dyn VectorFieldWriter>>,
}

impl InMemoryFieldWriter {
    fn new(
        field_name: String,
        config: VectorFieldConfig,
        store: Arc<FieldStore>,
        delegate: Option<Arc<dyn VectorFieldWriter>>,
    ) -> Self {
        Self {
            field_name,
            config,
            store,
            delegate,
        }
    }

    fn convert_vector(&self, stored: &StoredVector) -> Result<Vector> {
        let vector = stored.to_vector();
        // Safe default 0 if no vector config (shouldn't happen if we are adding vectors)
        let dimension = self
            .config
            .vector
            .as_ref()
            .map(|v| v.dimension())
            .unwrap_or(0);

        if vector.dimension() != dimension {
            return Err(LaurusError::invalid_argument(format!(
                "vector dimension mismatch for field '{}': expected {}, got {}",
                self.field_name,
                dimension,
                vector.dimension()
            )));
        }
        if !vector.is_valid() {
            return Err(LaurusError::invalid_argument(format!(
                "vector for field '{}' contains invalid values",
                self.field_name
            )));
        }
        Ok(vector)
    }
}

#[async_trait]
impl VectorFieldWriter for InMemoryFieldWriter {
    async fn add_value(
        &self,
        doc_id: u64,
        value: &crate::data::DataValue,
        version: u64,
    ) -> Result<()> {
        // If we have a delegate, let it handle embedding (EmbeddingVectorIndexWriter)
        if let Some(delegate) = &self.delegate {
            // Get count before to find new vector
            let before_count = delegate.vectors().await.len();

            // Delegate handles embedding
            delegate.add_value(doc_id, value, version).await?;

            // Retrieve the newly added vector and store it locally
            let vectors = delegate.vectors().await;
            if vectors.len() > before_count {
                // Get the last added vector
                let (_, _, ref vec) = vectors[vectors.len() - 1];
                let stored = StoredVector::new(vec.data.to_vec());
                let converted = self.convert_vector(&stored)?;
                self.store.replace(
                    doc_id,
                    FieldStoreEntry {
                        vectors: vec![converted],
                    },
                );
            }
            return Ok(());
        }

        // No delegate - only accept pre-computed vectors
        if let crate::data::DataValue::Vector(v) = value {
            let stored = StoredVector::new(v.clone());
            self.add_stored_vector(doc_id, &stored, version).await
        } else {
            Err(LaurusError::invalid_argument(
                "add_value not supported for this field writer (needs embedding helper)",
            ))
        }
    }

    async fn add_stored_vector(
        &self,
        doc_id: u64,
        vector: &StoredVector,
        version: u64,
    ) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            delegate.add_stored_vector(doc_id, vector, version).await?;
        }

        let converted = self.convert_vector(vector)?;
        self.store.replace(
            doc_id,
            FieldStoreEntry {
                vectors: vec![converted],
            },
        );
        Ok(())
    }

    async fn has_storage(&self) -> bool {
        if let Some(delegate) = &self.delegate {
            delegate.has_storage().await
        } else {
            false
        }
    }

    async fn rebuild(&self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let vectors_clone = vectors.clone();

        if let Some(delegate) = &self.delegate {
            delegate.rebuild(vectors).await?;
        }

        let mut guard = self.store.entries.write();
        guard.clear();
        for (doc_id, _, vector) in vectors_clone {
            guard.insert(
                doc_id,
                FieldStoreEntry {
                    vectors: vec![vector],
                },
            );
        }
        Ok(())
    }

    async fn vectors(&self) -> Vec<(u64, String, Vector)> {
        if let Some(delegate) = &self.delegate {
            delegate.vectors().await
        } else {
            Vec::new()
        }
    }

    async fn delete_document(&self, doc_id: u64, version: u64) -> Result<()> {
        self.store.remove(doc_id);
        if let Some(delegate) = &self.delegate {
            delegate.delete_document(doc_id, version).await?;
        }
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            delegate.flush().await?;
        }
        Ok(())
    }

    async fn optimize(&self) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            // Rebuild delegate using vectors from RAM store
            let vectors = self.store.vector_tuples(&self.field_name);
            delegate.rebuild(vectors).await?;
            delegate.flush().await?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct InMemoryFieldReader {
    field_name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    delegate: Option<Arc<dyn VectorFieldReader>>,
}

impl InMemoryFieldReader {
    fn new(
        field_name: String,
        config: VectorFieldConfig,
        store: Arc<FieldStore>,
        delegate: Option<Arc<dyn VectorFieldReader>>,
    ) -> Self {
        Self {
            field_name,
            config,
            store,
            delegate,
        }
    }
}

impl VectorFieldReader for InMemoryFieldReader {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        if request.field != self.field_name {
            return Err(LaurusError::internal(format!(
                "field mismatch: expected '{}', got '{}'",
                self.field_name, request.field
            )));
        }

        if request.query_vectors.is_empty() {
            return Ok(FieldSearchResults::default());
        }

        let limit = request.limit;

        let snapshot = self.store.snapshot();
        let mut merged: HashMap<u64, FieldHit> = HashMap::new();

        let (dimension, distance_metric) = match &self.config.vector {
            Some(opt) => (opt.dimension(), opt.distance()),
            None => return Ok(FieldSearchResults::default()), // No vector support
        };

        for query in &request.query_vectors {
            let query_vector_data = &query.vector.data;
            if query_vector_data.len() != dimension {
                return Err(LaurusError::invalid_argument(format!(
                    "query vector dimension mismatch for field '{}': expected {}, got {}",
                    self.field_name,
                    dimension,
                    query_vector_data.len()
                )));
            }
            let effective_weight = query.weight;
            if effective_weight == 0.0 {
                continue;
            }

            for (doc_id, entry) in &snapshot {
                for vector in &entry.vectors {
                    let similarity =
                        distance_metric.similarity(&query.vector.data, &vector.data)?;
                    let weighted_score = similarity * effective_weight;
                    let distance = distance_metric.distance(&query.vector.data, &vector.data)?;

                    match merged.entry(*doc_id) {
                        Entry::Vacant(slot) => {
                            slot.insert(FieldHit {
                                doc_id: *doc_id,
                                field: self.field_name.clone(),
                                score: weighted_score,
                                distance,
                            });
                        }
                        Entry::Occupied(mut slot) => {
                            let hit = slot.get_mut();
                            hit.score += weighted_score;
                            hit.distance = hit.distance.min(distance);
                        }
                    }
                }
            }
        }

        // Merge with delegate results if available
        if let Some(delegate) = &self.delegate {
            let delegate_results = delegate.search(request)?;
            for hit in delegate_results.hits {
                match merged.entry(hit.doc_id) {
                    Entry::Vacant(slot) => {
                        slot.insert(hit);
                    }
                    Entry::Occupied(_slot) => {
                        // If same doc exists in both (unlikely for typical update pattern unless update is insert+delete),
                        // we prioritize memory version as it is newer, OR we sum scores?
                        // For now, let's assume memory overrides disk (newer).
                        // Or maybe we treat them as separate parts?
                        // In Laurus, updates are usually full document replacements with new IDs?
                        // If same ID, it means update. Memory is newer.
                        // So we do NOT overwrite if already in memory.
                        // Actually, wait. The loop above puts memory hits into `merged`.
                        // So if it's already in `merged`, it came from memory.
                        // We should KEEP the memory version.
                        // So do nothing.
                    }
                }
            }
        }

        let mut hits: Vec<FieldHit> = merged.into_values().collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        if hits.len() > limit {
            hits.truncate(limit);
        }

        Ok(FieldSearchResults { hits })
    }

    fn stats(&self) -> Result<VectorFieldStats> {
        let dimension = self
            .config
            .vector
            .as_ref()
            .map(|v| v.dimension())
            .unwrap_or(0);
        Ok(VectorFieldStats {
            vector_count: self.store.total_vectors(),
            dimension,
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct FieldStore {
    entries: RwLock<HashMap<u64, FieldStoreEntry>>,
}

impl FieldStore {
    pub(crate) fn replace(&self, doc_id: u64, entry: FieldStoreEntry) {
        self.entries.write().insert(doc_id, entry);
    }

    pub(crate) fn remove(&self, doc_id: u64) {
        self.entries.write().remove(&doc_id);
    }

    pub(crate) fn snapshot(&self) -> HashMap<u64, FieldStoreEntry> {
        self.entries.read().clone()
    }

    pub(crate) fn total_vectors(&self) -> usize {
        self.entries
            .read()
            .values()
            .map(|entry| entry.vectors.len())
            .sum()
    }

    pub(crate) fn vector_tuples(&self, field_name: &str) -> Vec<(u64, String, Vector)> {
        let guard = self.entries.read();
        let mut tuples = Vec::new();
        let name = field_name.to_string();
        for (doc_id, entry) in guard.iter() {
            for vector in &entry.vectors {
                tuples.push((*doc_id, name.clone(), vector.clone()));
            }
        }
        tuples
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FieldStoreEntry {
    pub(crate) vectors: Vec<Vector>,
}
