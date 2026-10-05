//! The multi-vector index: segment-per-commit storage of per-document token
//! vectors (Issue #1177).
//!
//! Each commit seals the buffered documents into one LMV1 segment and
//! registers it in the shared segment manifest. Segment counts in the
//! manifest are **document** counts, so the merge policy and the compaction
//! ratio (deleted documents / documents) work in documents.
//!
//! Deletions are logical: a document-level bitmap in [`SegmentedCore`],
//! consulted at lookup time. A newer segment's copy of a document shadows
//! older ones. Merges drop deleted documents and stale copies, and copy the
//! surviving payloads without re-encoding them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::field::multi_vector_params_error;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::MultiVectorIndexConfig;
use crate::vector::index::multivector::LAYOUT;
use crate::vector::index::multivector::format::{DocEntry, SegmentReader, write_segment};
use crate::vector::index::multivector::reader::{
    MultiVectorReaderFacade, MultiVectorSearcher, MultiVectorSnapshot,
};
use crate::vector::index::segment::manager::{
    ManagedSegmentInfo, MergeCandidate, SegmentManager, SegmentManagerConfig,
};
use crate::vector::index::segment::merge_policy::TieredMergePolicy;
use crate::vector::index::segment::reader_cache::SegmentedReaderCache;
use crate::vector::index::segment::segmented_core::SegmentedCore;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::writer::VectorIndexWriter;

/// State shared by the index handle and its writers.
#[derive(Debug)]
struct Shared {
    core: SegmentedCore,
    readers: SegmentedReaderCache<SegmentReader>,
    config: MultiVectorIndexConfig,
}

impl Shared {
    fn file_name(segment_id: &str) -> String {
        format!("{segment_id}{}", LAYOUT.primary)
    }

    /// Sealed segment readers, newest generation first.
    fn readers_newest_first(&self) -> Result<Vec<Arc<SegmentReader>>> {
        let mut segments = self.core.manager().list_segments();
        segments.sort_by_key(|s| std::cmp::Reverse(s.generation));
        self.load_readers(&segments)
    }

    fn load_readers(&self, segments: &[ManagedSegmentInfo]) -> Result<Vec<Arc<SegmentReader>>> {
        segments
            .iter()
            .map(|info| {
                self.readers.get_or_load(&info.segment_id, || {
                    let file_name = Self::file_name(&info.segment_id);
                    let reader = SegmentReader::open(self.core.storage().as_ref(), &file_name)?;
                    // Reading a segment under another dimension would split
                    // its payload into wrong-sized vectors.
                    if reader.dimension() != self.config.dimension {
                        return Err(LaurusError::index(format!(
                            "multi-vector segment '{file_name}' has dimension {}, but the \
                             index is configured for {}",
                            reader.dimension(),
                            self.config.dimension
                        )));
                    }
                    Ok(reader)
                })
            })
            .collect()
    }

    fn snapshot(&self) -> Result<MultiVectorSnapshot> {
        Ok(MultiVectorSnapshot::new(
            self.config.dimension,
            self.config.distance_metric,
            self.readers_newest_first()?,
            self.core.bitmap()?,
        ))
    }

    /// Merge `segments` into one new segment and publish it.
    ///
    /// Pass 1 reads only the document tables, newest generation first,
    /// dropping deleted documents and stale copies. Pass 2 copies each
    /// surviving document's payload, so memory stays proportional to the
    /// document count.
    fn merge(&self, segments: Vec<ManagedSegmentInfo>) -> Result<()> {
        let mut sources = segments.clone();
        sources.sort_by_key(|s| std::cmp::Reverse(s.generation));
        let readers = self.load_readers(&sources)?;
        for reader in &readers {
            reader.verify_payload()?;
        }

        let mut seen = HashSet::new();
        let mut survivors: Vec<(DocEntry, usize)> = Vec::new();
        for (source, reader) in readers.iter().enumerate() {
            for entry in reader.entries() {
                if self.core.is_deleted(entry.doc_id)? || !seen.insert(entry.doc_id) {
                    continue;
                }
                survivors.push((entry, source));
            }
        }
        survivors.sort_unstable_by_key(|(entry, _)| entry.doc_id);
        let entries: Vec<DocEntry> = survivors.iter().map(|(entry, _)| *entry).collect();

        let merged_id = self.core.manager().generate_segment_id();
        write_segment(
            self.core.storage().as_ref(),
            &Self::file_name(&merged_id),
            self.config.dimension,
            &entries,
            |i, sink| {
                let (entry, source) = survivors[i];
                let vectors = readers[source].vectors(entry.doc_id)?.ok_or_else(|| {
                    LaurusError::internal("merge source lost a document listed in its table")
                })?;
                sink.write(&vectors)
            },
        )?;

        // The merged segment holds data no newer than its newest source, so
        // it inherits the highest source generation (not max + 1): a newer
        // segment outside the merge must keep shadowing it.
        let generation = segments.iter().map(|s| s.generation).max().unwrap_or(0);
        let merged = ManagedSegmentInfo::new(merged_id, entries.len() as u64, 0, generation);
        let candidate = MergeCandidate {
            total_vectors: segments.iter().map(|s| s.vector_count).sum(),
            total_size: segments.iter().map(|s| s.size_bytes).sum(),
            segments,
        };
        let source_ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        self.core.manager().apply_merge(candidate, merged)?;
        for id in &source_ids {
            self.readers.invalidate(id);
        }
        Ok(())
    }
}

/// Segment-per-commit store of per-document token vectors (see the module
/// docs). It is not a vector-search target: its searcher rejects queries,
/// and late-interaction rescoring reads documents through
/// [`MultiVectorIndex::snapshot`].
#[derive(Debug)]
pub struct MultiVectorIndex {
    shared: Arc<Shared>,
    closed: AtomicBool,
}

impl MultiVectorIndex {
    /// Open an existing multi-vector index or create a new one.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend (already scoped to the field).
    /// * `name` - Index name, the prefix of the deletion bitmap file.
    /// * `config` - Index configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_config`] when `name` collides with the
    /// generated segment-id namespace or the dimension / distance cannot
    /// describe a multi-vector field, and an error when the segment manifest
    /// fails to load.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: MultiVectorIndexConfig,
    ) -> Result<Self> {
        if let Some(ordinal) = name.strip_prefix("segment_")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(LaurusError::invalid_config(format!(
                "index name '{name}' collides with the reserved segment-id namespace \
                 (segment_<digits>)"
            )));
        }
        if let Some(reason) = multi_vector_params_error(config.dimension, config.distance_metric) {
            return Err(LaurusError::invalid_config(reason));
        }

        let manager = Arc::new(SegmentManager::new(
            SegmentManagerConfig {
                max_vectors_per_segment: config.max_documents_per_segment,
                merge_factor: config.merge_factor,
                max_segments: config.max_segments,
                ..SegmentManagerConfig::default()
            },
            storage.clone(),
            LAYOUT,
        )?);
        Ok(Self {
            shared: Arc::new(Shared {
                core: SegmentedCore::new(name, storage, manager),
                readers: SegmentedReaderCache::new(),
                config,
            }),
            closed: AtomicBool::new(false),
        })
    }

    /// A point-in-time view of the committed documents.
    ///
    /// # Errors
    ///
    /// Returns an error when the index is closed or a segment fails to load.
    pub fn snapshot(&self) -> Result<MultiVectorSnapshot> {
        self.check_closed()?;
        self.shared.snapshot()
    }

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(LaurusError::InvalidOperation("Index is closed".to_string()));
        }
        Ok(())
    }

    /// Merge one policy-selected window of segments; returns whether a
    /// merge ran.
    fn merge_once(&self) -> Result<bool> {
        let Some(candidate) = self
            .shared
            .core
            .manager()
            .check_merge(&TieredMergePolicy::new())
        else {
            return Ok(false);
        };
        self.shared.merge(candidate.segments)?;
        Ok(true)
    }
}

impl VectorIndex for MultiVectorIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        Ok(Arc::new(MultiVectorReaderFacade::new(self.snapshot()?)))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;
        Ok(Box::new(MultiVectorWriter {
            segment_id: self.shared.core.manager().generate_segment_id(),
            shared: self.shared.clone(),
            buffer: Vec::new(),
            sealed_len: None,
            closed: false,
        }))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        self.shared.core.storage()
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;
        // Documents counted per segment: a stale same-id copy awaiting a
        // merge is counted once per segment until the merge collapses it.
        let manager = self.shared.core.manager();
        let deleted = self.shared.core.deleted_count()?;
        Ok(VectorIndexStats {
            vector_count: manager.total_vectors().saturating_sub(deleted),
            dimension: self.shared.config.dimension,
            total_size: manager.stats().total_size,
            deleted_count: deleted,
            last_modified: 0,
        })
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;
        let segments = self.shared.core.manager().list_segments();
        if segments.is_empty() {
            return Ok(());
        }
        self.shared.merge(segments)?;
        // Every logically deleted document was dropped by the merge.
        self.shared.core.clear_deletions()
    }

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        Ok(Box::new(MultiVectorSearcher))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.shared.config.embedder)
    }

    fn last_wal_seq(&self) -> u64 {
        self.shared.core.last_wal_seq()
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        // Published by `persist_deletions`, once everything up to `seq` is
        // durable.
        self.shared.core.set_pending_wal_seq(seq);
        Ok(())
    }

    fn supports_soft_delete(&self) -> bool {
        true
    }

    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        self.check_closed()?;
        self.shared.core.mark_deleted(doc_id)?;
        Ok(())
    }

    fn persist_deletions(&self) -> Result<()> {
        self.shared.core.persist_deletions()
    }

    fn maybe_auto_compact(&self) -> Result<bool> {
        if self.merge_once()? {
            return Ok(true);
        }
        let config = &self.shared.config;
        if config.auto_compaction
            && self.shared.core.deleted_count()? > 0
            && self.shared.core.deletion_ratio()? >= config.compaction_threshold
        {
            self.optimize()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn multi_vector_snapshot(&self, _field: &str) -> Result<Option<MultiVectorSnapshot>> {
        self.snapshot().map(Some)
    }
}

/// Active-segment writer of a [`MultiVectorIndex`].
///
/// Buffers token vectors as `(doc_id, field, vector)` triples, one per
/// token, and seals them as one new segment on commit. A sealed writer is
/// done: committing further changes is rejected, since re-writing a
/// registered segment would keep its generation while newer segments may
/// have been sealed meanwhile.
#[derive(Debug)]
struct MultiVectorWriter {
    shared: Arc<Shared>,
    segment_id: String,
    buffer: Vec<(u64, String, Vector)>,
    /// Buffer length when the segment was sealed; `None` until then.
    sealed_len: Option<usize>,
    closed: bool,
}

impl MultiVectorWriter {
    /// Buffered documents, ascending by doc id, with the buffer positions
    /// of their vectors in insertion order.
    fn documents(&self) -> Vec<(u64, Vec<usize>)> {
        let mut by_doc: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, (doc_id, _, _)) in self.buffer.iter().enumerate() {
            by_doc.entry(*doc_id).or_default().push(i);
        }
        let mut docs: Vec<(u64, Vec<usize>)> = by_doc.into_iter().collect();
        docs.sort_unstable_by_key(|(doc_id, _)| *doc_id);
        docs
    }
}

impl VectorIndexWriter for MultiVectorWriter {
    fn next_vector_id(&self) -> u64 {
        self.buffer
            .iter()
            .map(|(doc_id, _, _)| doc_id + 1)
            .max()
            .unwrap_or(0)
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.add_vectors(vectors)
    }

    /// Buffer token vectors. The vectors a call carries for a document
    /// replace any it already buffered, and clear the document's deletion
    /// mark (the second half of an upsert).
    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let dimension = self.shared.config.dimension;
        if let Some((doc_id, _, vector)) =
            vectors.iter().find(|(_, _, v)| v.dimension() != dimension)
        {
            return Err(LaurusError::invalid_argument(format!(
                "token vector of doc {doc_id} has dimension {}, expected {dimension}",
                vector.dimension()
            )));
        }
        let incoming: HashSet<u64> = vectors.iter().map(|(doc_id, _, _)| *doc_id).collect();
        self.buffer
            .retain(|(doc_id, _, _)| !incoming.contains(doc_id));
        self.shared.core.unmark_deleted(incoming.iter().copied())?;

        let normalize = self.shared.config.distance_metric == DistanceMetric::Cosine;
        self.buffer
            .extend(vectors.into_iter().map(|(doc_id, field, mut vector)| {
                if normalize {
                    vector.normalize();
                }
                (doc_id, field, vector)
            }));
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn progress(&self) -> f32 {
        if self.sealed_len.is_some() { 1.0 } else { 0.0 }
    }

    fn estimated_memory_usage(&self) -> usize {
        self.buffer
            .iter()
            .map(|(_, field, vector)| {
                field.len() + vector.dimension() * std::mem::size_of::<f32>() + 64
            })
            .sum()
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.buffer
    }

    /// Write the buffered documents as this writer's segment file (without
    /// registering it; see [`VectorIndexWriter::commit`]).
    fn write(&self) -> Result<()> {
        let docs = self.documents();
        let entries: Vec<DocEntry> = docs
            .iter()
            .map(|(doc_id, positions)| DocEntry {
                doc_id: *doc_id,
                vector_count: positions.len() as u32,
            })
            .collect();
        write_segment(
            self.shared.core.storage().as_ref(),
            &Shared::file_name(&self.segment_id),
            self.shared.config.dimension,
            &entries,
            |i, sink| {
                for &position in &docs[i].1 {
                    sink.write(self.buffer[position].2.data.as_slice())?;
                }
                Ok(())
            },
        )
    }

    fn has_storage(&self) -> bool {
        true
    }

    /// Upsert delete-first: drop the buffered copy and hide the sealed ones.
    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        self.shared.core.mark_deleted(doc_id)?;
        self.buffer.retain(|(id, _, _)| *id != doc_id);
        Ok(())
    }

    fn has_pending_changes(&self) -> bool {
        match self.sealed_len {
            Some(sealed) => self.buffer.len() != sealed,
            None => !self.buffer.is_empty(),
        }
    }

    fn commit(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if let Some(sealed) = self.sealed_len {
            if self.buffer.len() == sealed {
                return Ok(());
            }
            return Err(LaurusError::InvalidOperation(format!(
                "segment '{}' is already sealed; obtain a fresh writer for further changes",
                self.segment_id
            )));
        }
        self.write()?;
        let doc_count = self.documents().len() as u64;
        // Generation 0: the manager stamps max + 1 and measures the size.
        let info = ManagedSegmentInfo::new(self.segment_id.clone(), doc_count, 0, 0);
        self.shared.core.manager().add_segment(info)?;
        self.sealed_len = Some(self.buffer.len());
        self.shared.readers.invalidate(&self.segment_id);
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // The discarded mutations' WAL records must stay replayable.
        self.shared.core.rollback_pending_wal_seq();
        self.buffer.clear();
        self.sealed_len = None;
        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        if self.sealed_len.is_some() {
            return 0;
        }
        self.buffer
            .iter()
            .map(|(doc_id, _, _)| *doc_id)
            .collect::<HashSet<_>>()
            .len() as u64
    }

    fn close(&mut self) -> Result<()> {
        if self.has_pending_changes() {
            self.commit()?;
        }
        self.closed = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn build_reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        Ok(Arc::new(MultiVectorReaderFacade::new(
            self.shared.snapshot()?,
        )))
    }
}

impl Drop for MultiVectorWriter {
    fn drop(&mut self) {
        // Dropping unsealed mutations loses them while the pending WAL
        // checkpoint may already cover them; roll it back so recovery
        // replays them.
        if self.has_pending_changes() {
            self.shared.core.rollback_pending_wal_seq();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    const FIELD: &str = "tokens";

    fn storage() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    fn config(distance: DistanceMetric) -> MultiVectorIndexConfig {
        MultiVectorIndexConfig {
            dimension: 2,
            distance_metric: distance,
            ..MultiVectorIndexConfig::default()
        }
    }

    fn open(storage: &Arc<dyn Storage>) -> MultiVectorIndex {
        MultiVectorIndex::open_or_create(
            storage.clone(),
            "index",
            config(DistanceMetric::DotProduct),
        )
        .unwrap()
    }

    fn tokens(doc_id: u64, vectors: &[[f32; 2]]) -> Vec<(u64, String, Vector)> {
        vectors
            .iter()
            .map(|v| (doc_id, FIELD.to_string(), Vector::new(v.to_vec())))
            .collect()
    }

    /// Commit `docs` as one segment through a fresh writer, then run the
    /// rest of the store's commit sequence.
    fn commit(index: &MultiVectorIndex, docs: &[(u64, &[[f32; 2]])]) {
        let mut writer = index.writer().unwrap();
        for (doc_id, vectors) in docs {
            writer.delete_document(*doc_id).unwrap();
            writer.add_vectors(tokens(*doc_id, vectors)).unwrap();
        }
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
    }

    fn vectors_of(index: &MultiVectorIndex, doc_id: u64) -> Option<Vec<f32>> {
        index
            .snapshot()
            .unwrap()
            .vectors(doc_id)
            .unwrap()
            .map(|v| v.into_owned())
    }

    #[test]
    fn test_commit_and_reopen() {
        let storage = storage();
        let index = open(&storage);
        commit(
            &index,
            &[(3, &[[1.0, 2.0], [3.0, 4.0]]), (1, &[[5.0, 6.0]])],
        );
        assert_eq!(vectors_of(&index, 3), Some(vec![1.0, 2.0, 3.0, 4.0]));

        let reopened = open(&storage);
        assert_eq!(vectors_of(&reopened, 3), Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(vectors_of(&reopened, 1), Some(vec![5.0, 6.0]));
        assert_eq!(vectors_of(&reopened, 2), None);
        assert_eq!(reopened.stats().unwrap().vector_count, 2);
        let reader = reopened.reader().unwrap();
        assert_eq!(reader.vector_count(), 2);
        assert_eq!(reader.doc_ids_for_field(FIELD).as_ref(), &[1, 3]);
        assert!(reader.contains_vector(3, FIELD));
        assert!(reader.validate().unwrap().is_valid);
    }

    #[test]
    fn test_cosine_normalizes_and_keeps_zero_vectors() {
        let storage = storage();
        let index =
            MultiVectorIndex::open_or_create(storage, "index", config(DistanceMetric::Cosine))
                .unwrap();
        commit(&index, &[(1, &[[3.0, 4.0], [0.0, 0.0]])]);
        assert_eq!(vectors_of(&index, 1), Some(vec![0.6, 0.8, 0.0, 0.0]));
    }

    #[test]
    fn test_newest_segment_wins_and_upsert_replaces() {
        let storage = storage();
        let index = open(&storage);
        commit(&index, &[(1, &[[1.0, 1.0]]), (2, &[[2.0, 2.0]])]);
        commit(&index, &[(1, &[[9.0, 9.0], [8.0, 8.0]])]);
        assert_eq!(vectors_of(&index, 1), Some(vec![9.0, 9.0, 8.0, 8.0]));
        assert_eq!(vectors_of(&index, 2), Some(vec![2.0, 2.0]));

        // Re-adding within one writer replaces the buffered copy.
        let mut writer = index.writer().unwrap();
        writer.add_vectors(tokens(5, &[[1.0, 0.0]])).unwrap();
        writer.add_vectors(tokens(5, &[[0.0, 1.0]])).unwrap();
        writer.commit().unwrap();
        assert_eq!(vectors_of(&index, 5), Some(vec![0.0, 1.0]));
    }

    #[test]
    fn test_delete_persists_and_readd_clears_it() {
        let storage = storage();
        let index = open(&storage);
        commit(&index, &[(1, &[[1.0, 1.0]]), (2, &[[2.0, 2.0]])]);
        index.soft_delete_document(1).unwrap();
        index.persist_deletions().unwrap();
        assert_eq!(vectors_of(&index, 1), None);

        let reopened = open(&storage);
        assert_eq!(vectors_of(&reopened, 1), None);
        assert_eq!(reopened.stats().unwrap().deleted_count, 1);
        assert_eq!(reopened.reader().unwrap().vector_count(), 1);

        commit(&reopened, &[(1, &[[7.0, 7.0]])]);
        assert_eq!(vectors_of(&reopened, 1), Some(vec![7.0, 7.0]));
    }

    #[test]
    fn test_optimize_drops_deleted_and_stale_copies() {
        let storage = storage();
        let index = open(&storage);
        commit(&index, &[(1, &[[1.0, 1.0]]), (2, &[[2.0, 2.0]])]);
        commit(&index, &[(2, &[[3.0, 3.0]]), (3, &[[4.0, 4.0]])]);
        index.soft_delete_document(1).unwrap();
        index.persist_deletions().unwrap();

        index.optimize().unwrap();
        let segments = index.shared.core.manager().list_segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].vector_count, 2);
        assert_eq!(vectors_of(&index, 1), None);
        assert_eq!(vectors_of(&index, 2), Some(vec![3.0, 3.0]));
        assert_eq!(vectors_of(&index, 3), Some(vec![4.0, 4.0]));
        assert_eq!(index.stats().unwrap().deleted_count, 0);
        assert!(!storage.file_exists("index.delmap"));

        // Only the merged segment's file remains.
        let mv_files: Vec<String> = storage
            .list_files()
            .unwrap()
            .into_iter()
            .filter(|f| f.ends_with(".mv"))
            .collect();
        assert_eq!(mv_files, vec![format!("{}.mv", segments[0].segment_id)]);
        assert_eq!(vectors_of(&open(&storage), 2), Some(vec![3.0, 3.0]));
    }

    #[test]
    fn test_tiered_merge_keeps_newest_copy() {
        let storage = storage();
        let index = MultiVectorIndex::open_or_create(
            storage.clone(),
            "index",
            MultiVectorIndexConfig {
                merge_factor: 2,
                ..config(DistanceMetric::DotProduct)
            },
        )
        .unwrap();
        commit(&index, &[(1, &[[1.0, 1.0]])]);
        commit(&index, &[(1, &[[2.0, 2.0]])]);
        assert!(index.maybe_auto_compact().unwrap());
        assert_eq!(index.shared.core.manager().list_segments().len(), 1);
        assert_eq!(vectors_of(&index, 1), Some(vec![2.0, 2.0]));
    }

    #[test]
    fn test_auto_compaction_counts_documents() {
        let storage = storage();
        let index = MultiVectorIndex::open_or_create(
            storage.clone(),
            "index",
            MultiVectorIndexConfig {
                auto_compaction: true,
                compaction_threshold: 0.5,
                ..config(DistanceMetric::DotProduct)
            },
        )
        .unwrap();
        // Doc 1 has many vectors, doc 2 one: by documents half are deleted.
        commit(&index, &[(1, &[[1.0, 1.0]; 8]), (2, &[[2.0, 2.0]])]);
        index.soft_delete_document(2).unwrap();
        assert!(index.maybe_auto_compact().unwrap());
        assert_eq!(index.stats().unwrap().vector_count, 1);
        assert_eq!(vectors_of(&index, 1).map(|v| v.len()), Some(16));
    }

    #[test]
    fn test_wal_seq_is_published_by_persist_and_survives_reopen() {
        let storage = storage();
        let index = open(&storage);
        index.set_last_wal_seq(7).unwrap();
        assert_eq!(index.last_wal_seq(), 0);
        commit(&index, &[(1, &[[1.0, 1.0]])]);
        assert_eq!(index.last_wal_seq(), 7);
        assert_eq!(open(&storage).last_wal_seq(), 7);
    }

    #[test]
    fn test_dropping_a_dirty_writer_rolls_back_the_pending_wal_seq() {
        let storage = storage();
        let index = open(&storage);
        index.set_last_wal_seq(3).unwrap();
        {
            let mut writer = index.writer().unwrap();
            writer.add_vectors(tokens(1, &[[1.0, 1.0]])).unwrap();
        }
        index.persist_deletions().unwrap();
        assert_eq!(index.last_wal_seq(), 0);
    }

    #[test]
    fn test_sealed_writer_rejects_more_changes() {
        let storage = storage();
        let index = open(&storage);
        let mut writer = index.writer().unwrap();
        writer.add_vectors(tokens(1, &[[1.0, 1.0]])).unwrap();
        writer.commit().unwrap();
        writer.commit().unwrap();
        writer.add_vectors(tokens(2, &[[1.0, 1.0]])).unwrap();
        assert!(writer.commit().is_err());
        writer.rollback().unwrap();
    }

    #[test]
    fn test_rejects_wrong_dimension_and_bad_config() {
        let storage = storage();
        let index = open(&storage);
        let mut writer = index.writer().unwrap();
        let err = writer
            .add_vectors(vec![(1, FIELD.to_string(), Vector::new(vec![1.0]))])
            .unwrap_err();
        assert!(err.to_string().contains("dimension 1, expected 2"), "{err}");

        for bad in [
            MultiVectorIndexConfig {
                dimension: 0,
                ..config(DistanceMetric::DotProduct)
            },
            config(DistanceMetric::Euclidean),
        ] {
            assert!(MultiVectorIndex::open_or_create(storage.clone(), "x", bad).is_err());
        }
        assert!(
            MultiVectorIndex::open_or_create(
                storage.clone(),
                "segment_000001",
                config(DistanceMetric::DotProduct)
            )
            .is_err()
        );
    }

    #[test]
    fn test_segment_of_another_dimension_is_rejected() {
        let storage = storage();
        commit(&open(&storage), &[(1, &[[1.0, 1.0]])]);
        let wider = MultiVectorIndex::open_or_create(
            storage,
            "index",
            MultiVectorIndexConfig {
                dimension: 4,
                ..config(DistanceMetric::DotProduct)
            },
        )
        .unwrap();
        let err = wider.snapshot().unwrap_err();
        assert!(err.to_string().contains("has dimension 2"), "{err}");
    }

    #[test]
    fn test_searcher_rejects_queries() {
        let index = open(&storage());
        let searcher = index.searcher().unwrap();
        let err = searcher
            .search(&crate::vector::search::searcher::VectorIndexQuery::new(
                Vector::new(vec![1.0, 0.0]),
            ))
            .unwrap_err();
        assert!(
            err.to_string().contains("not a vector-search target"),
            "{err}"
        );
    }
}
