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
//! surviving payloads without re-encoding them when they are already
//! encoded the way the merged segment is.
//!
//! A field with a `pool_factor` above 1 pools every document's token
//! vectors as they are buffered (see [`TokenPooling`]) and marks them in
//! the segment's document table with that factor. A merge pools the
//! documents still marked as ingested, which is how turning pooling on for
//! an existing field (a Reindex) pools its stored vectors, and leaves
//! pooled documents alone.
//!
//! A field configured for [`MultiVectorStorage::Binary`] learns its
//! [`BinaryParams`] once, from the first flush or merge that holds at least
//! [`BinaryParams::MIN_TRAINING_ROWS`] token vectors, and encodes every
//! later segment with them. Segments written before that are stored as
//! [`INTERIM_KIND`] and converted by a later merge.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::field::multi_vector_params_error;
use crate::vector::core::multi_vector::{BinaryParams, MultiVectorStorage, OwnedCodec};
use crate::vector::core::token_pooling::{
    POOLING_BUDGET_BYTES, PooledTokens, TokenPooling, pool_windows, pooled_count, pooling_bytes,
};
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
use crate::vector::writer::{FieldVectors, VectorIndexWriter};

/// Element kind of the segments a `Binary` field writes before it has
/// learned its parameters.
const INTERIM_KIND: MultiVectorStorage = MultiVectorStorage::F16;

/// What a field knows about its [`BinaryParams`].
#[derive(Debug, Default)]
enum BinaryState {
    /// Not looked up in the sealed segments yet.
    #[default]
    NotLoaded,
    /// No sealed segment holds parameters.
    Absent,
    /// The parameters every new `Binary` segment is encoded with: those of
    /// the sealed `Binary` segments.
    Known(Arc<BinaryParams>),
}

/// State shared by the index handle and its writers.
#[derive(Debug)]
struct Shared {
    core: SegmentedCore,
    readers: SegmentedReaderCache<SegmentReader>,
    config: MultiVectorIndexConfig,
    /// The field's `Binary` parameters; unused for other kinds. Held from
    /// choosing a segment's encoding until the segment is registered, so a
    /// flush and a merge cannot learn two different parameter sets. Lock
    /// order: taken before the reader-cache, manifest and deletion locks,
    /// never inside a reader loader.
    binary: Mutex<BinaryState>,
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

    /// How a new segment of `rows` token vectors is encoded, and whether
    /// that took learning new [`BinaryParams`] (from `sample`, called only
    /// then). Learned parameters must be passed to [`Self::publish`] once
    /// the segment is registered.
    ///
    /// For a kind other than `Binary` this is simply that kind. A `Binary`
    /// field uses its known parameters, learns them when `rows` reaches
    /// [`BinaryParams::MIN_TRAINING_ROWS`], and otherwise writes
    /// [`INTERIM_KIND`].
    fn resolve_target(
        &self,
        state: &mut BinaryState,
        rows: usize,
        sample: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<(OwnedCodec, bool)> {
        if let Some(codec) = OwnedCodec::plain(self.config.storage) {
            return Ok((codec, false));
        }
        if matches!(state, BinaryState::NotLoaded) {
            *state = self.load_binary_params()?;
        }
        if let BinaryState::Known(params) = state {
            return Ok((OwnedCodec::Binary(params.clone()), false));
        }
        if rows >= BinaryParams::MIN_TRAINING_ROWS {
            let params = BinaryParams::train(self.config.dimension, &sample()?)?;
            return Ok((OwnedCodec::Binary(Arc::new(params)), true));
        }
        let interim =
            OwnedCodec::plain(INTERIM_KIND).expect("the interim kind needs no parameters");
        Ok((interim, false))
    }

    /// The parameters of the newest sealed `Binary` segment, if any.
    fn load_binary_params(&self) -> Result<BinaryState> {
        let mut segments = self.core.manager().list_segments();
        segments.sort_by_key(|s| std::cmp::Reverse(s.generation));
        for segment in segments {
            let reader = self.load_readers(std::slice::from_ref(&segment))?;
            if let Some(params) = reader[0].binary_params() {
                return Ok(BinaryState::Known(params.clone()));
            }
        }
        Ok(BinaryState::Absent)
    }

    /// Record parameters [`Self::resolve_target`] learned, once the segment
    /// encoded with them is registered.
    fn publish(state: &mut BinaryState, codec: &OwnedCodec, trained: bool) {
        if trained && let OwnedCodec::Binary(params) = codec {
            *state = BinaryState::Known(params.clone());
        }
    }

    /// The pool factor recorded for documents pooled under this field's
    /// configuration: the factor, or `0` when the field does not pool.
    fn pool_marker(&self) -> u32 {
        if self.config.pool_factor > 1 {
            self.config.pool_factor
        } else {
            0
        }
    }

    /// How this field pools its documents: `None` when it does not pool
    /// (a factor of 1).
    fn token_pooling(&self) -> Option<TokenPooling> {
        (self.config.pool_factor > 1).then(|| {
            TokenPooling::new(
                self.config.dimension,
                self.config.pool_factor,
                self.config.distance_metric == DistanceMetric::Cosine,
            )
        })
    }

    /// Pool one document's stored token vectors (row-major) by the field's
    /// factor, re-normalizing the means in a `Cosine` field; `None` when
    /// pooling leaves them unchanged or the field does not pool.
    fn pool_rows(&self, rows: &[f32]) -> Option<Vec<f32>> {
        self.token_pooling()?.pool_rows(rows)
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
    /// dropping the documents `deletions` (a
    /// [`SegmentedCore::deletion_snapshot`]) marks and stale copies. Pass 2
    /// copies each surviving document's payload, so memory stays
    /// proportional to the document count.
    fn merge(
        &self,
        segments: Vec<ManagedSegmentInfo>,
        deletions: Option<&DeletionBitmap>,
    ) -> Result<()> {
        self.merge_within(segments, deletions, POOLING_BUDGET_BYTES)
    }

    /// [`Self::merge`], pooling the documents it pools in windows of at most
    /// `pooling_budget` bytes (see [`pool_windows`]), each window's documents
    /// at the same time (Issue #1429). Rows are still read and written in
    /// doc-id order, and pooling is deterministic, so the merged segment is
    /// the same whatever the windows and the threads.
    fn merge_within(
        &self,
        segments: Vec<ManagedSegmentInfo>,
        deletions: Option<&DeletionBitmap>,
        pooling_budget: usize,
    ) -> Result<()> {
        let mut binary = self.binary.lock();
        let mut sources = segments.clone();
        sources.sort_by_key(|s| std::cmp::Reverse(s.generation));
        let readers = self.load_readers(&sources)?;
        for reader in &readers {
            reader.verify_payload()?;
        }

        // Documents still stored as ingested are pooled now when the field
        // pools (right after pooling was turned on, for instance); their
        // pooled counts are known up front, so the document table can be
        // written before the rows.
        let factor = self.config.pool_factor;
        let mut seen = HashSet::new();
        let mut survivors: Vec<Survivor> = Vec::new();
        for (source, reader) in readers.iter().enumerate() {
            for entry in reader.entries() {
                if deletions.is_some_and(|deletions| deletions.is_deleted(entry.doc_id))
                    || !seen.insert(entry.doc_id)
                {
                    continue;
                }
                let pool = factor > 1 && entry.pool_factor == 0;
                let stored = entry.vector_count as usize;
                let entry = if pool {
                    DocEntry {
                        vector_count: pooled_count(stored, factor) as u32,
                        pool_factor: factor,
                        ..entry
                    }
                } else {
                    entry
                };
                survivors.push(Survivor {
                    shrinks: (entry.vector_count as usize) < stored,
                    entry,
                    source,
                    pool,
                    stored,
                });
            }
        }
        survivors.sort_unstable_by_key(|s| s.entry.doc_id);
        let entries: Vec<DocEntry> = survivors.iter().map(|s| s.entry).collect();
        // The rows a survivor is written with: its stored rows, decoded and
        // pooled when it is pooled now.
        let source_rows = |s: &Survivor| {
            readers[s.source].rows(s.entry.doc_id)?.ok_or_else(|| {
                LaurusError::internal("merge source lost a document listed in its table")
            })
        };

        let rows = entries.iter().map(|e| e.vector_count as usize).sum();
        let (target, trained) = self.resolve_target(&mut binary, rows, || {
            training_sample(&survivors, rows, self.config.dimension, |s, out| {
                let values = source_rows(s)?.to_f32();
                match s.pool.then(|| self.pool_rows(&values)).flatten() {
                    Some(pooled) => out.extend_from_slice(&pooled),
                    None => out.extend_from_slice(&values),
                }
                Ok(())
            })
        })?;

        // The survivors pooling shrinks, in write order, grouped into windows
        // whose documents are pooled at the same time; `pooled` holds the
        // current window's rows until they are written.
        let shrinking: Vec<usize> = (0..survivors.len())
            .filter(|&i| survivors[i].shrinks)
            .collect();
        let costs: Vec<usize> = shrinking
            .iter()
            .map(|&i| pooling_bytes(survivors[i].stored, self.config.dimension))
            .collect();
        let mut windows = pool_windows(&costs, pooling_budget).into_iter();
        let mut pooled: VecDeque<(u64, Vec<f32>)> = VecDeque::new();

        let merged_id = self.core.manager().generate_segment_id();
        write_segment(
            self.core.storage().as_ref(),
            &Self::file_name(&merged_id),
            target.as_codec(),
            self.config.dimension,
            &entries,
            |i, sink| {
                let survivor = &survivors[i];
                if survivor.shrinks {
                    if pooled.is_empty() {
                        let window = windows.next().ok_or_else(|| {
                            LaurusError::internal("a merge ran out of documents to pool")
                        })?;
                        pooled = self
                            .pool_window(&readers, &survivors, &shrinking[window])?
                            .into();
                    }
                    let (doc_id, rows) =
                        pooled.pop_front().expect("a pooling window is never empty");
                    if doc_id != survivor.entry.doc_id {
                        return Err(LaurusError::internal(format!(
                            "a merge pooled doc {doc_id} where doc {} belongs",
                            survivor.entry.doc_id
                        )));
                    }
                    return sink.write(&rows);
                }
                // Rows encoded exactly as the target encodes are copied
                // verbatim (no re-quantization, no compounding loss across
                // merges); others (the field's `storage` just changed) are
                // transcoded through f32, which is also how `update_field`
                // migrates every existing segment to the new kind.
                sink.append(&source_rows(survivor)?)
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
        Self::publish(&mut binary, &target, trained);
        for id in &source_ids {
            self.readers.invalidate(id);
        }
        Ok(())
    }

    /// Pool the stored rows of the survivors at `window` (indices into
    /// `survivors`, all of which pooling shrinks), at the same time under the
    /// `native` feature and one after another otherwise.
    ///
    /// # Returns
    ///
    /// Each survivor's doc id and pooled rows, in `window`'s order.
    fn pool_window(
        &self,
        readers: &[Arc<SegmentReader>],
        survivors: &[Survivor],
        window: &[usize],
    ) -> Result<Vec<(u64, Vec<f32>)>> {
        let pooling = self
            .token_pooling()
            .ok_or_else(|| LaurusError::internal("a merge pooled in a field that does not pool"))?;
        let pool_one = |&i: &usize| -> Result<(u64, Vec<f32>)> {
            let survivor = &survivors[i];
            let doc_id = survivor.entry.doc_id;
            let rows = readers[survivor.source].rows(doc_id)?.ok_or_else(|| {
                LaurusError::internal("merge source lost a document listed in its table")
            })?;
            let pooled = pooling
                .pool_rows(&rows.to_f32())
                .ok_or_else(|| LaurusError::internal("a document pooling shrinks did not pool"))?;
            Ok((doc_id, pooled))
        };
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            window.par_iter().map(pool_one).collect()
        }
        #[cfg(not(feature = "native"))]
        {
            window.iter().map(pool_one).collect()
        }
    }
}

/// A document a merge keeps: its entry as the merged segment records it,
/// the reader it comes from, whether the merge pools its rows (and so
/// marks it pooled), how many rows it has stored, and whether pooling
/// shrinks them.
struct Survivor {
    entry: DocEntry,
    source: usize,
    pool: bool,
    stored: usize,
    shrinks: bool,
}

/// Row-major token vectors to learn [`BinaryParams`] from: every
/// `stride`-th document of `docs` (in order, so the sample is
/// deterministic), with the stride chosen so that about
/// [`BinaryParams::MAX_TRAINING_ROWS`] of the `total_rows` are taken, and
/// never more than that. `read` appends one document's rows.
fn training_sample<T>(
    docs: &[T],
    total_rows: usize,
    dimension: usize,
    mut read: impl FnMut(&T, &mut Vec<f32>) -> Result<()>,
) -> Result<Vec<f32>> {
    let stride = total_rows.div_ceil(BinaryParams::MAX_TRAINING_ROWS).max(1);
    let cap = BinaryParams::MAX_TRAINING_ROWS * dimension;
    let mut sample = Vec::new();
    for doc in docs.iter().step_by(stride) {
        read(doc, &mut sample)?;
        if sample.len() >= cap {
            sample.truncate(cap);
            break;
        }
    }
    Ok(sample)
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
        if let Some(reason) = multi_vector_params_error(
            config.dimension,
            config.distance_metric,
            config.storage,
            config.pool_factor,
        ) {
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
                binary: Mutex::new(BinaryState::NotLoaded),
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
        let deletions = self.shared.core.deletion_snapshot()?;
        self.shared
            .merge(candidate.segments, deletions.as_deref())?;
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
            buffer: BTreeMap::new(),
            changed_since_seal: None,
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
        // Filter with a frozen copy of the deletions: a soft delete takes no
        // writer lock and can land while the merge runs (Issue #1518).
        let deletions = self.shared.core.deletion_snapshot()?;
        self.shared.merge(segments, deletions.as_deref())?;
        // Every document the copy marks was dropped by the merge; a delete
        // marked since stays marked.
        self.shared.core.forget_deletions(deletions.as_deref())
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

    fn write_deletions(&self) -> Result<()> {
        self.shared.core.write_deletions()
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

    fn token_pooling(&self, _field: &str) -> Option<TokenPooling> {
        self.shared.token_pooling()
    }
}

/// Active-segment writer of a [`MultiVectorIndex`].
///
/// Buffers each document's token vectors under its doc id, so an upsert
/// costs time proportional to the document's own tokens rather than to the
/// whole buffer, and seals them as one new segment on commit. A sealed
/// writer is done, even after a rollback: committing further changes is
/// rejected, since re-writing a registered segment would keep its
/// generation while newer segments may have been sealed meanwhile.
#[derive(Debug)]
struct MultiVectorWriter {
    shared: Arc<Shared>,
    segment_id: String,
    /// Each buffered document's token vectors in arrival order, keyed (and
    /// so written) by doc id.
    buffer: BTreeMap<u64, Vec<Vector>>,
    /// `None` until the segment is sealed, then whether the buffer has
    /// changed since the seal or the last rollback.
    changed_since_seal: Option<bool>,
    closed: bool,
}

impl MultiVectorWriter {
    /// Record a buffer change; after sealing, it makes `commit` fail.
    fn mark_changed(&mut self) {
        if let Some(changed) = &mut self.changed_since_seal {
            *changed = true;
        }
    }

    /// Reject the first `(doc_id, vector)` whose dimension is not the
    /// index's.
    fn check_dimensions<'a>(
        &self,
        vectors: impl IntoIterator<Item = (u64, &'a Vector)>,
    ) -> Result<()> {
        let dimension = self.shared.config.dimension;
        if let Some((doc_id, vector)) = vectors
            .into_iter()
            .find(|(_, vector)| vector.dimension() != dimension)
        {
            return Err(LaurusError::invalid_argument(format!(
                "token vector of doc {doc_id} has dimension {}, expected {dimension}",
                vector.dimension()
            )));
        }
        Ok(())
    }

    /// Buffer `docs`, each a document's non-empty token vectors in arrival
    /// order (already checked by [`Self::check_dimensions`]) and possibly
    /// the same vectors pooled in advance (Issue #1429): keep the pooled
    /// vectors when they were pooled the way this field pools, and
    /// otherwise normalize the vectors when the metric needs it and pool
    /// them when the field pools; then clear the documents' deletion marks,
    /// and only then replace their buffered entries, so a failure leaves the
    /// buffer untouched.
    ///
    /// Every write passes here exactly once (ingest, WAL replay, embedder
    /// output), so a document is pooled once, deterministically. Vectors
    /// pooled under another configuration (the field changed between
    /// pooling and buffering) are never kept, so the marker
    /// [`Self::write_with`] stamps always describes the stored rows.
    fn buffer_documents(
        &mut self,
        docs: Vec<(u64, Vec<Vector>, Option<PooledTokens>)>,
    ) -> Result<()> {
        let pooling = self.shared.token_pooling();
        let normalize = self.shared.config.distance_metric == DistanceMetric::Cosine;
        let docs: Vec<(u64, Vec<Vector>)> = docs
            .into_iter()
            .map(|(doc_id, mut vectors, pooled)| {
                match pooled {
                    // Kept bit for bit: normalizing pooled rows again can
                    // change them.
                    Some(pooled)
                        if Some(pooled.pooling()) == pooling
                            && pooled.source_len() == vectors.len() =>
                    {
                        vectors = pooled.into_vectors();
                    }
                    _ => match pooling.and_then(|pooling| pooling.apply(&vectors)) {
                        Some(pooled) => vectors = pooled.into_vectors(),
                        None if normalize => vectors.iter_mut().for_each(Vector::normalize),
                        None => {}
                    },
                }
                (doc_id, vectors)
            })
            .collect();
        self.shared
            .core
            .unmark_deleted(docs.iter().map(|(doc_id, _)| *doc_id))?;
        if !docs.is_empty() {
            self.mark_changed();
        }
        for (doc_id, vectors) in docs {
            self.buffer.insert(doc_id, vectors);
        }
        Ok(())
    }

    /// The encoding of this writer's segment (see [`Shared::resolve_target`]),
    /// learning `Binary` parameters from a sample of the buffer when needed.
    fn resolve_target(&self, state: &mut BinaryState) -> Result<(OwnedCodec, bool)> {
        let rows = self.buffer.values().map(Vec::len).sum();
        self.shared.resolve_target(state, rows, || {
            let docs: Vec<&Vec<Vector>> = self.buffer.values().collect();
            training_sample(&docs, rows, self.shared.config.dimension, |doc, out| {
                for vector in doc.iter() {
                    out.extend_from_slice(&vector.data);
                }
                Ok(())
            })
        })
    }

    /// Write the buffered documents as this writer's segment file, encoded
    /// with `codec`.
    fn write_with(&self, codec: &OwnedCodec) -> Result<()> {
        let docs: Vec<&Vec<Vector>> = self.buffer.values().collect();
        let entries: Vec<DocEntry> = self
            .buffer
            .iter()
            .map(|(doc_id, vectors)| DocEntry {
                doc_id: *doc_id,
                vector_count: vectors.len() as u32,
                pool_factor: self.shared.pool_marker(),
            })
            .collect();
        write_segment(
            self.shared.core.storage().as_ref(),
            &Shared::file_name(&self.segment_id),
            codec.as_codec(),
            self.shared.config.dimension,
            &entries,
            |i, sink| {
                for vector in docs[i] {
                    sink.write(vector.data.as_slice())?;
                }
                Ok(())
            },
        )
    }
}

impl VectorIndexWriter for MultiVectorWriter {
    fn next_vector_id(&self) -> u64 {
        self.buffer
            .last_key_value()
            .map_or(0, |(doc_id, _)| doc_id + 1)
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.add_vectors(vectors)
    }

    /// Buffer token vectors. The vectors a call carries for a document
    /// replace any it already buffered, and clear the document's deletion
    /// mark (the second half of an upsert).
    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.check_dimensions(vectors.iter().map(|(doc_id, _, vector)| (*doc_id, vector)))?;
        let mut incoming: BTreeMap<u64, Vec<Vector>> = BTreeMap::new();
        for (doc_id, _, vector) in vectors {
            incoming.entry(doc_id).or_default().push(vector);
        }
        self.buffer_documents(
            incoming
                .into_iter()
                .map(|(doc_id, vectors)| (doc_id, vectors, None))
                .collect(),
        )
    }

    /// Buffer one document's token vectors as [`Self::add_vectors`] does,
    /// without a field name per token (Issue #1406), through
    /// [`Self::add_prepared_document_vectors`] with nothing pooled in
    /// advance.
    fn add_document_vectors(
        &mut self,
        doc_id: u64,
        fields: Vec<(String, Vec<Vector>)>,
    ) -> Result<()> {
        self.add_prepared_document_vectors(
            doc_id,
            fields
                .into_iter()
                .map(|(field, vectors)| FieldVectors::new(field, vectors))
                .collect(),
        )
    }

    /// Buffer one document's token vectors, keeping vectors pooled in
    /// advance when they were pooled the way this field pools (Issue
    /// #1429). This writer serves a single field, so the names are ignored
    /// and every entry's vectors are taken in order, then pooled together
    /// as one document: the first entry's pooled vectors were pooled from
    /// fewer vectors than that, so [`Self::buffer_documents`] does not keep
    /// them.
    fn add_prepared_document_vectors(
        &mut self,
        doc_id: u64,
        fields: Vec<FieldVectors>,
    ) -> Result<()> {
        let mut fields = fields.into_iter();
        let Some(FieldVectors {
            mut vectors,
            pooled,
            ..
        }) = fields.next()
        else {
            return Ok(());
        };
        for more in fields {
            vectors.extend(more.vectors);
        }
        if vectors.is_empty() {
            // Nothing to buffer. Clearing the deletion mark would make the
            // sealed copy that the caller's `delete_document` hid visible
            // again, which `add_vectors` never does for a document it was
            // given no vectors for.
            return Ok(());
        }
        self.check_dimensions(vectors.iter().map(|vector| (doc_id, vector)))?;
        self.buffer_documents(vec![(doc_id, vectors, pooled)])
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn progress(&self) -> f32 {
        if self.changed_since_seal.is_some() {
            1.0
        } else {
            0.0
        }
    }

    fn estimated_memory_usage(&self) -> usize {
        self.buffer
            .values()
            .flatten()
            .map(|vector| vector.dimension() * std::mem::size_of::<f32>() + 64)
            .sum()
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        // Contract mismatch, as in `MultiFieldWriter::vectors`: the buffer
        // is keyed by document, so there is no slice of per-token triples
        // to lend. Committed documents are read through `build_reader`.
        &[]
    }

    /// Write the buffered documents as this writer's segment file (without
    /// registering it; see [`VectorIndexWriter::commit`]). `Binary`
    /// parameters learned here are not kept: only a registered segment
    /// makes them the field's.
    fn write(&self) -> Result<()> {
        let mut binary = self.shared.binary.lock();
        let (codec, _) = self.resolve_target(&mut binary)?;
        self.write_with(&codec)
    }

    fn has_storage(&self) -> bool {
        true
    }

    /// Upsert delete-first: drop the buffered copy and hide the sealed ones.
    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        self.shared.core.mark_deleted(doc_id)?;
        if self.buffer.remove(&doc_id).is_some() {
            self.mark_changed();
        }
        Ok(())
    }

    fn has_pending_changes(&self) -> bool {
        match self.changed_since_seal {
            Some(changed) => changed,
            None => !self.buffer.is_empty(),
        }
    }

    fn commit(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if let Some(changed) = self.changed_since_seal {
            if !changed {
                return Ok(());
            }
            return Err(LaurusError::InvalidOperation(format!(
                "segment '{}' is already sealed; obtain a fresh writer for further changes",
                self.segment_id
            )));
        }
        {
            let mut binary = self.shared.binary.lock();
            let (codec, trained) = self.resolve_target(&mut binary)?;
            self.write_with(&codec)?;
            // Generation 0: the manager stamps max + 1 and measures the size.
            let info =
                ManagedSegmentInfo::new(self.segment_id.clone(), self.buffer.len() as u64, 0, 0);
            self.shared.core.manager().add_segment(info)?;
            Shared::publish(&mut binary, &codec, trained);
        }
        self.changed_since_seal = Some(false);
        self.shared.readers.invalidate(&self.segment_id);
        Ok(())
    }

    /// Discard the changes buffered since the last commit. A sealed writer
    /// stays sealed: its segment is registered, so a later change must still
    /// make `commit` fail rather than rewrite the segment's file and register
    /// its id again.
    fn rollback(&mut self) -> Result<()> {
        // The discarded mutations' WAL records must stay replayable.
        self.shared.core.rollback_pending_wal_seq();
        self.buffer.clear();
        if let Some(changed) = &mut self.changed_since_seal {
            *changed = false;
        }
        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        if self.changed_since_seal.is_some() {
            return 0;
        }
        self.buffer.len() as u64
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
    use std::io::Read;

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

    /// `(doc_id, vector_count)` of every document in the newest segment.
    fn newest_segment_entries(index: &MultiVectorIndex) -> Vec<(u64, u32)> {
        index.shared.readers_newest_first().unwrap()[0]
            .entries()
            .map(|entry| (entry.doc_id, entry.vector_count))
            .collect()
    }

    #[test]
    fn test_delete_drops_the_buffered_copy() {
        let index = open(&storage());
        let mut writer = index.writer().unwrap();
        for doc_id in 1..=3 {
            writer.add_vectors(tokens(doc_id, &[[1.0, 1.0]])).unwrap();
        }
        writer.delete_document(2).unwrap();
        assert_eq!(writer.pending_docs(), 2);
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        assert_eq!(newest_segment_entries(&index), vec![(1, 1), (3, 1)]);
        assert_eq!(vectors_of(&index, 2), None);
    }

    #[test]
    fn test_reput_of_a_buffered_document_keeps_only_the_latest_copy() {
        let index = open(&storage());
        let mut writer = index.writer().unwrap();
        for (doc_id, v) in [(1, [1.0, 0.0]), (2, [2.0, 0.0]), (3, [3.0, 0.0])] {
            writer.delete_document(doc_id).unwrap();
            writer.add_vectors(tokens(doc_id, &[v, v])).unwrap();
        }
        // The store's upsert sequence, on a document between two others.
        writer.delete_document(2).unwrap();
        writer
            .add_vectors(tokens(2, &[[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]]))
            .unwrap();
        assert_eq!(writer.pending_docs(), 3);
        assert_eq!(writer.next_vector_id(), 4);
        writer.commit().unwrap();
        index.persist_deletions().unwrap();

        assert_eq!(newest_segment_entries(&index), vec![(1, 2), (2, 3), (3, 2)]);
        assert_eq!(vectors_of(&index, 1), Some(vec![1.0, 0.0, 1.0, 0.0]));
        assert_eq!(
            vectors_of(&index, 2),
            Some(vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0])
        );
        assert_eq!(vectors_of(&index, 3), Some(vec![3.0, 0.0, 3.0, 0.0]));
    }

    #[test]
    fn test_interleaved_documents_keep_their_token_order() {
        let index = open(&storage());
        let mut writer = index.writer().unwrap();
        let mut vectors = Vec::new();
        for (doc_id, v) in [
            (2, [1.0, 0.0]),
            (1, [2.0, 0.0]),
            (2, [3.0, 0.0]),
            (1, [4.0, 0.0]),
        ] {
            vectors.extend(tokens(doc_id, &[v]));
        }
        writer.add_vectors(vectors).unwrap();
        writer.commit().unwrap();
        assert_eq!(newest_segment_entries(&index), vec![(1, 2), (2, 2)]);
        assert_eq!(vectors_of(&index, 1), Some(vec![2.0, 0.0, 4.0, 0.0]));
        assert_eq!(vectors_of(&index, 2), Some(vec![1.0, 0.0, 3.0, 0.0]));
    }

    #[test]
    fn test_sealed_writer_rejects_a_same_size_replacement() {
        let index = open(&storage());
        let mut writer = index.writer().unwrap();
        writer.add_vectors(tokens(1, &[[1.0, 1.0]])).unwrap();
        writer.commit().unwrap();
        assert!(!writer.has_pending_changes());

        // Same token count as the sealed copy: the buffer size is unchanged.
        writer.delete_document(1).unwrap();
        writer.add_vectors(tokens(1, &[[2.0, 2.0]])).unwrap();
        assert!(writer.has_pending_changes());
        assert!(writer.commit().is_err());
        writer.rollback().unwrap();
    }

    /// `(segment_id, vector_count, generation)` of every registered segment,
    /// and the bytes of the first one's file.
    fn sealed_state(index: &MultiVectorIndex) -> (Vec<(String, u64, u64)>, Vec<u8>) {
        let segments: Vec<(String, u64, u64)> = index
            .shared
            .core
            .manager()
            .list_segments()
            .into_iter()
            .map(|segment| (segment.segment_id, segment.vector_count, segment.generation))
            .collect();
        let mut bytes = Vec::new();
        index
            .shared
            .core
            .storage()
            .open_input(&Shared::file_name(&segments[0].0))
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        (segments, bytes)
    }

    #[test]
    fn test_rollback_after_commit_keeps_the_writer_sealed() {
        let storage = storage();
        let index = open(&storage);
        let mut writer = index.writer().unwrap();
        writer.add_vectors(tokens(1, &[[1.0, 1.0]])).unwrap();
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        let sealed = sealed_state(&index);

        // Re-sealing would rewrite the registered segment's file and
        // register its id a second time.
        writer.rollback().unwrap();
        writer.add_vectors(tokens(2, &[[2.0, 2.0]])).unwrap();
        let err = writer.commit().unwrap_err();
        assert!(err.to_string().contains("already sealed"), "{err}");

        // Rolling back discards the change, so nothing is pending.
        writer.rollback().unwrap();
        assert!(!writer.has_pending_changes());
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        assert_eq!(sealed_state(&index), sealed);

        let reopened = open(&storage);
        assert_eq!(sealed_state(&reopened), sealed);
        assert_eq!(vectors_of(&reopened, 1), Some(vec![1.0, 1.0]));
        assert_eq!(vectors_of(&reopened, 2), None);
        assert_eq!(reopened.stats().unwrap().vector_count, 1);
    }

    #[test]
    fn test_rollback_before_commit_discards_the_buffer() {
        let storage = storage();
        let index = open(&storage);
        let mut writer = index.writer().unwrap();
        writer.add_vectors(tokens(1, &[[1.0, 1.0]])).unwrap();
        writer.rollback().unwrap();
        assert!(!writer.has_pending_changes());
        assert_eq!(writer.pending_docs(), 0);

        writer.add_vectors(tokens(2, &[[2.0, 2.0]])).unwrap();
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        assert_eq!(newest_segment_entries(&index), vec![(2, 1)]);

        let reopened = open(&storage);
        assert_eq!(vectors_of(&reopened, 1), None);
        assert_eq!(vectors_of(&reopened, 2), Some(vec![2.0, 2.0]));
    }

    /// One field's token vectors in the shape `VectorStore::write_vectors`
    /// passes to `add_document_vectors`.
    fn field(vectors: &[[f32; 2]]) -> (String, Vec<Vector>) {
        (
            FIELD.to_string(),
            vectors.iter().map(|v| Vector::new(v.to_vec())).collect(),
        )
    }

    #[test]
    fn test_document_vectors_buffer_like_add_vectors() {
        let index =
            MultiVectorIndex::open_or_create(storage(), "index", config(DistanceMetric::Cosine))
                .unwrap();
        let mut writer = index.writer().unwrap();
        let rows = [[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]];
        writer.add_vectors(tokens(1, &rows)).unwrap();
        writer
            .add_document_vectors(2, vec![field(&rows[..2]), field(&rows[2..])])
            .unwrap();
        writer.commit().unwrap();

        let expected = Some(vec![0.6, 0.8, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(vectors_of(&index, 1), expected);
        assert_eq!(vectors_of(&index, 2), expected);
    }

    #[test]
    fn test_document_vectors_replace_a_sealed_document() {
        let index = open(&storage());
        commit(&index, &[(1, &[[1.0, 1.0]]), (2, &[[2.0, 2.0]])]);

        let mut writer = index.writer().unwrap();
        writer.delete_document(1).unwrap();
        writer
            .add_document_vectors(1, vec![field(&[[7.0, 7.0], [8.0, 8.0]])])
            .unwrap();
        writer.commit().unwrap();
        index.persist_deletions().unwrap();

        assert_eq!(vectors_of(&index, 1), Some(vec![7.0, 7.0, 8.0, 8.0]));
        assert_eq!(vectors_of(&index, 2), Some(vec![2.0, 2.0]));
    }

    #[test]
    fn test_document_vectors_without_vectors_keep_the_deletion_mark() {
        let index = open(&storage());
        commit(&index, &[(1, &[[1.0, 1.0]])]);

        let mut writer = index.writer().unwrap();
        writer.delete_document(1).unwrap();
        writer.add_document_vectors(1, vec![field(&[])]).unwrap();
        writer.add_document_vectors(1, Vec::new()).unwrap();
        assert!(!writer.has_pending_changes());
        writer.commit().unwrap();
        index.persist_deletions().unwrap();

        assert_eq!(vectors_of(&index, 1), None);
    }

    #[test]
    fn test_document_vectors_reject_a_wrong_dimension() {
        let index = open(&storage());
        let mut writer = index.writer().unwrap();
        writer
            .add_document_vectors(1, vec![field(&[[1.0, 1.0]])])
            .unwrap();
        let err = writer
            .add_document_vectors(
                2,
                vec![(
                    FIELD.to_string(),
                    vec![Vector::new(vec![1.0, 1.0]), Vector::new(vec![1.0])],
                )],
            )
            .unwrap_err();
        assert!(err.to_string().contains("dimension 1, expected 2"), "{err}");
        assert_eq!(writer.pending_docs(), 1);
    }

    #[test]
    fn test_sealed_writer_rejects_document_vectors() {
        let index = open(&storage());
        let mut writer = index.writer().unwrap();
        writer
            .add_document_vectors(1, vec![field(&[[1.0, 1.0]])])
            .unwrap();
        writer.commit().unwrap();
        writer
            .add_document_vectors(2, vec![field(&[[2.0, 2.0]])])
            .unwrap();
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

    /// The `Binary` lifecycle: when a field learns its parameters, how it
    /// keeps them, and how merges treat rows encoded under them (#1346).
    mod binary {
        use std::sync::atomic::AtomicBool;

        use super::*;
        use crate::storage::{FileMetadata, StorageInput, StorageOutput};
        use crate::vector::core::multi_vector::{RowCodec, test_binary_params};
        use crate::vector::index::multivector::format::DocEntry;

        const DIM: usize = 4;
        const MIN: usize = BinaryParams::MIN_TRAINING_ROWS;

        fn config(storage: MultiVectorStorage) -> MultiVectorIndexConfig {
            MultiVectorIndexConfig {
                dimension: DIM,
                distance_metric: DistanceMetric::Cosine,
                storage,
                ..MultiVectorIndexConfig::default()
            }
        }

        fn open(storage: &Arc<dyn Storage>, kind: MultiVectorStorage) -> MultiVectorIndex {
            MultiVectorIndex::open_or_create(storage.clone(), "index", config(kind)).unwrap()
        }

        /// Token `i` of `doc_id`: deterministic and anisotropic (every
        /// token leans the same way), like a late-interaction model's.
        fn token(doc_id: u64, i: usize) -> Vec<f32> {
            let h = |k: u64| {
                let x = ((doc_id * 1_000 + i as u64) * 4 + k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            };
            vec![
                0.8 + 0.4 * h(0),
                0.6 + 0.4 * h(1),
                -0.5 + 0.4 * h(2),
                0.4 * h(3),
            ]
        }

        /// `count` documents from `first`, each with `tokens` token vectors.
        fn docs(first: u64, count: u64, tokens: usize) -> Vec<(u64, usize)> {
            (first..first + count).map(|d| (d, tokens)).collect()
        }

        fn add(writer: &mut dyn VectorIndexWriter, docs: &[(u64, usize)]) {
            for &(doc_id, count) in docs {
                let vectors = (0..count)
                    .map(|i| (doc_id, FIELD.to_string(), Vector::new(token(doc_id, i))))
                    .collect();
                writer.add_vectors(vectors).unwrap();
            }
        }

        /// Commit `docs` as one segment.
        fn commit_docs(index: &MultiVectorIndex, docs: &[(u64, usize)]) {
            let mut writer = index.writer().unwrap();
            add(writer.as_mut(), docs);
            writer.commit().unwrap();
        }

        /// The L2-normalized token vectors of `docs`, row-major in doc-id
        /// order: what a flush of them learns from.
        fn normalized_rows(docs: &[(u64, usize)]) -> Vec<f32> {
            let mut sorted = docs.to_vec();
            sorted.sort();
            let mut rows = Vec::new();
            for (doc_id, count) in sorted {
                for i in 0..count {
                    let mut v = Vector::new(token(doc_id, i));
                    v.normalize();
                    rows.extend_from_slice(&v.data);
                }
            }
            rows
        }

        /// `(kind, parameters)` of every sealed segment, newest first; the
        /// kind is `None` for an empty segment.
        fn segments(
            index: &MultiVectorIndex,
        ) -> Vec<(Option<MultiVectorStorage>, Option<BinaryParams>)> {
            index
                .shared
                .readers_newest_first()
                .unwrap()
                .iter()
                .map(|reader| {
                    let kind = reader
                        .entries()
                        .next()
                        .map(|e| reader.rows(e.doc_id).unwrap().unwrap().kind());
                    (kind, reader.binary_params().map(|p| (**p).clone()))
                })
                .collect()
        }

        fn known(index: &MultiVectorIndex) -> Option<BinaryParams> {
            match &*index.shared.binary.lock() {
                BinaryState::Known(params) => Some((**params).clone()),
                BinaryState::NotLoaded | BinaryState::Absent => None,
            }
        }

        fn raw_rows(index: &MultiVectorIndex, doc_id: u64) -> Vec<u8> {
            let snapshot = index.snapshot().unwrap();
            let rows = snapshot.rows(doc_id).unwrap().unwrap();
            rows.as_bytes().to_vec()
        }

        fn decode_with(params: &BinaryParams, values: &[f32]) -> Vec<f32> {
            let codec = RowCodec::Binary(params);
            let mut out = Vec::new();
            for row in values.as_chunks::<DIM>().0 {
                let mut bytes = Vec::new();
                codec.encode_row(row, &mut bytes);
                codec.decode_row(&bytes, DIM, &mut out);
            }
            out
        }

        #[test]
        fn test_a_flush_below_the_threshold_is_interim_and_learns_nothing() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            let mut batch = docs(0, 63, 32);
            batch.push((63, 31));
            assert_eq!(batch.iter().map(|d| d.1).sum::<usize>(), MIN - 1);
            commit_docs(&index, &batch);
            assert_eq!(segments(&index), [(Some(INTERIM_KIND), None)]);
            assert!(matches!(*index.shared.binary.lock(), BinaryState::Absent));
        }

        #[test]
        fn test_a_flush_at_the_threshold_learns_from_its_buffer() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            let batch = docs(0, 64, 32);
            commit_docs(&index, &batch);
            let expected = BinaryParams::train(DIM, &normalized_rows(&batch)).unwrap();
            assert_eq!(
                segments(&index),
                [(Some(MultiVectorStorage::Binary), Some(expected.clone()))]
            );
            assert_eq!(known(&index), Some(expected));
        }

        #[test]
        fn test_write_without_commit_does_not_keep_learned_parameters() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            let mut writer = index.writer().unwrap();
            add(writer.as_mut(), &docs(0, 64, 32));
            writer.write().unwrap();
            assert_eq!(known(&index), None);
        }

        #[test]
        fn test_later_flushes_and_a_reopened_index_reuse_the_parameters() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            commit_docs(&index, &docs(0, 64, 32));
            let params = known(&index).unwrap();
            commit_docs(&index, &docs(100, 1, 1));
            drop(index);

            let reopened = open(&storage, MultiVectorStorage::Binary);
            assert!(matches!(
                *reopened.shared.binary.lock(),
                BinaryState::NotLoaded
            ));
            commit_docs(&reopened, &docs(101, 1, 1));
            let binary = (Some(MultiVectorStorage::Binary), Some(params));
            assert_eq!(
                segments(&reopened),
                [binary.clone(), binary.clone(), binary]
            );
        }

        #[test]
        fn test_interim_segments_below_the_threshold_merge_into_an_interim_segment() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            for first in [0, 10, 20] {
                commit_docs(&index, &docs(first, 10, 32));
            }
            let before: Vec<Vec<u8>> = (0..30).map(|d| raw_rows(&index, d)).collect();
            index.optimize().unwrap();
            assert_eq!(segments(&index), [(Some(INTERIM_KIND), None)]);
            for (doc_id, bytes) in before.iter().enumerate() {
                assert_eq!(&raw_rows(&index, doc_id as u64), bytes);
            }
            assert!(matches!(*index.shared.binary.lock(), BinaryState::Absent));
        }

        #[test]
        fn test_a_merge_reaching_the_threshold_learns_and_converts_interim_segments() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            for first in [0, 25, 50] {
                commit_docs(&index, &docs(first, 25, 32));
            }
            assert!(
                segments(&index)
                    .iter()
                    .all(|s| *s == (Some(INTERIM_KIND), None))
            );
            // The merge learns from the rows as stored: the interim decode.
            let stored: Vec<f32> = (0..75)
                .flat_map(|d| vectors_of(&index, d).unwrap())
                .collect();
            let expected = BinaryParams::train(DIM, &stored).unwrap();

            index.optimize().unwrap();
            assert_eq!(
                segments(&index),
                [(Some(MultiVectorStorage::Binary), Some(expected.clone()))]
            );
            assert_eq!(
                vectors_of(&index, 7).unwrap(),
                decode_with(&expected, &stored[7 * 32 * DIM..8 * 32 * DIM])
            );
            commit_docs(&index, &docs(100, 1, 1));
            assert_eq!(
                segments(&index)[0],
                (Some(MultiVectorStorage::Binary), Some(expected))
            );
        }

        /// Register a one-document segment holding `rows` encoded with
        /// `codec`, as if another flush had written it.
        fn register(index: &MultiVectorIndex, doc_id: u64, codec: RowCodec<'_>, rows: &[f32]) {
            let shared = &index.shared;
            let id = shared.core.manager().generate_segment_id();
            let entry = DocEntry {
                doc_id,
                vector_count: (rows.len() / DIM) as u32,
                pool_factor: 0,
            };
            write_segment(
                shared.core.storage().as_ref(),
                &Shared::file_name(&id),
                codec,
                DIM,
                &[entry],
                |_, sink| sink.write(rows),
            )
            .unwrap();
            shared
                .core
                .manager()
                .add_segment(ManagedSegmentInfo::new(id, 1, 0, 0))
                .unwrap();
        }

        #[test]
        fn test_merging_segments_with_equal_parameters_copies_their_bits() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            commit_docs(&index, &docs(0, 64, 32));
            // Equal parameters in another allocation, as a reopened field
            // would read them back.
            let params = known(&index).unwrap();
            let codec = RowCodec::Binary(&params);
            let decode_bits = |bits: u8| {
                let mut out = Vec::new();
                codec.decode_row(&[bits], DIM, &mut out);
                out
            };
            let encode_bits = |values: &[f32]| {
                let mut out = Vec::new();
                codec.encode_row(values, &mut out);
                out[0]
            };

            // A bit pattern whose decode re-encodes to other bits: only a
            // verbatim copy keeps it. `c` is the midpoint of a pattern's
            // decode and its complement's.
            let pattern = (0u8..16)
                .find(|&p| encode_bits(&decode_bits(p)) != p)
                .expect("these parameters have a pattern that re-encodes differently");
            let center: Vec<f32> = decode_bits(pattern)
                .iter()
                .zip(decode_bits(!pattern & 0x0f))
                .map(|(a, b)| (a + b) / 2.0)
                .collect();
            let row: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    if pattern >> i & 1 == 1 {
                        c + 0.01
                    } else {
                        c - 0.01
                    }
                })
                .collect();
            assert_eq!(encode_bits(&row), pattern);
            register(&index, 2_000, codec, &[row.clone(), row].concat());

            let ids: Vec<u64> = (0..64).chain([2_000]).collect();
            let before: Vec<Vec<u8>> = ids.iter().map(|&d| raw_rows(&index, d)).collect();
            assert_eq!(before[64], [pattern, pattern]);

            index.optimize().unwrap();
            assert_eq!(segments(&index).len(), 1);
            for (doc_id, bytes) in ids.iter().zip(&before) {
                assert_eq!(&raw_rows(&index, *doc_id), bytes, "doc {doc_id}");
            }
        }

        #[test]
        fn test_merging_segments_with_different_parameters_transcodes() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            commit_docs(&index, &docs(0, 64, 32));
            let p1 = known(&index).unwrap();
            let p2 = test_binary_params(DIM);
            assert_ne!(p1, p2);

            // A segment encoded under other parameters.
            let rows: Vec<f32> = (0..3).flat_map(|i| token(500, i)).collect();
            register(&index, 500, RowCodec::Binary(&p2), &rows);
            let under_p2 = vectors_of(&index, 500).unwrap();

            index.optimize().unwrap();
            assert_eq!(
                vectors_of(&index, 500).unwrap(),
                decode_with(&p1, &under_p2)
            );
        }

        #[test]
        fn test_binary_segments_decode_and_migrate_under_another_kind() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            commit_docs(&index, &docs(0, 64, 32));
            let values = vectors_of(&index, 3).unwrap();
            drop(index);

            let f32_index = open(&storage, MultiVectorStorage::F32);
            assert_eq!(vectors_of(&f32_index, 3).unwrap(), values);
            f32_index.optimize().unwrap();
            assert_eq!(
                segments(&f32_index),
                [(Some(MultiVectorStorage::F32), None)]
            );
            assert_eq!(vectors_of(&f32_index, 3).unwrap(), values);
        }

        #[test]
        fn test_migrating_to_binary_learns_from_the_stored_f32_rows() {
            for (count, learns) in [(64, true), (63, false)] {
                let storage = storage();
                let batch = docs(0, count, 32);
                commit_docs(&open(&storage, MultiVectorStorage::F32), &batch);

                let index = open(&storage, MultiVectorStorage::Binary);
                index.optimize().unwrap();
                let expected = if learns {
                    let params = BinaryParams::train(DIM, &normalized_rows(&batch)).unwrap();
                    (Some(MultiVectorStorage::Binary), Some(params))
                } else {
                    (Some(INTERIM_KIND), None)
                };
                assert_eq!(segments(&index), [expected], "{count} documents");
            }
        }

        #[test]
        fn test_deleting_every_document_keeps_the_parameters() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary);
            let batch = docs(0, 64, 32);
            commit_docs(&index, &batch);
            let params = known(&index).unwrap();
            let mut writer = index.writer().unwrap();
            for (doc_id, _) in &batch {
                writer.delete_document(*doc_id).unwrap();
            }
            writer.commit().unwrap();
            index.persist_deletions().unwrap();

            index.optimize().unwrap();
            assert_eq!(segments(&index), [(None, Some(params.clone()))]);
            assert_eq!(known(&index), Some(params));
        }

        /// Fails every `rename_file` onto a segment file while armed, i.e.
        /// the last step of writing a segment.
        #[derive(Debug)]
        struct FailingSegmentRename {
            inner: Arc<dyn Storage>,
            armed: AtomicBool,
        }

        impl Storage for FailingSegmentRename {
            fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
                self.inner.open_input(name)
            }
            fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
                self.inner.create_output(name)
            }
            fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
                self.inner.create_output_append(name)
            }
            fn file_exists(&self, name: &str) -> bool {
                self.inner.file_exists(name)
            }
            fn delete_file(&self, name: &str) -> Result<()> {
                self.inner.delete_file(name)
            }
            fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
                if self.armed.load(Ordering::SeqCst) && new_name.ends_with(LAYOUT.primary) {
                    return Err(LaurusError::storage(format!(
                        "injected failure renaming {old_name}"
                    )));
                }
                self.inner.rename_file(old_name, new_name)
            }
            fn list_files(&self) -> Result<Vec<String>> {
                self.inner.list_files()
            }
            fn file_size(&self, name: &str) -> Result<u64> {
                self.inner.file_size(name)
            }
            fn sync(&self) -> Result<()> {
                self.inner.sync()
            }
            fn metadata(&self, name: &str) -> Result<FileMetadata> {
                self.inner.metadata(name)
            }
            fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
                self.inner.create_temp_output(prefix)
            }
            fn close(&mut self) -> Result<()> {
                Ok(())
            }
        }

        #[test]
        fn test_parameters_learned_by_a_failed_flush_are_not_kept() {
            let failing = Arc::new(FailingSegmentRename {
                inner: storage(),
                armed: AtomicBool::new(true),
            });
            let storage: Arc<dyn Storage> = failing.clone();
            let index = open(&storage, MultiVectorStorage::Binary);
            let mut writer = index.writer().unwrap();
            add(writer.as_mut(), &docs(0, 64, 32));
            assert!(writer.commit().is_err());
            assert_eq!(known(&index), None);
            assert!(segments(&index).is_empty());
        }
    }

    /// Token pooling (#1346): documents are pooled once, at ingest or by
    /// the merge that turns pooling on, and never again.
    mod pooling {
        use super::*;
        use crate::vector::core::multi_vector::BinaryParams;
        use crate::vector::core::token_pooling::{self, PooledTokens, TokenPooling, pooled_count};

        const DIM: usize = 4;

        fn config(storage: MultiVectorStorage, pool_factor: u32) -> MultiVectorIndexConfig {
            MultiVectorIndexConfig {
                dimension: DIM,
                distance_metric: DistanceMetric::Cosine,
                storage,
                pool_factor,
                ..MultiVectorIndexConfig::default()
            }
        }

        fn open(
            storage: &Arc<dyn Storage>,
            kind: MultiVectorStorage,
            pool_factor: u32,
        ) -> MultiVectorIndex {
            MultiVectorIndex::open_or_create(storage.clone(), "index", config(kind, pool_factor))
                .unwrap()
        }

        /// Token `i` of `doc_id`: deterministic and anisotropic.
        fn token(doc_id: u64, i: usize) -> Vec<f32> {
            let h = |k: u64| {
                let x = ((doc_id * 1_000 + i as u64) * 4 + k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            };
            vec![
                0.8 + 0.4 * h(0),
                0.6 + 0.4 * h(1),
                -0.5 + 0.4 * h(2),
                0.4 * h(3),
            ]
        }

        /// Commit `docs` (`(doc id, token count)`) as one segment.
        fn commit_docs(index: &MultiVectorIndex, docs: &[(u64, usize)]) {
            let mut writer = index.writer().unwrap();
            for &(doc_id, count) in docs {
                let vectors = (0..count)
                    .map(|i| (doc_id, FIELD.to_string(), Vector::new(token(doc_id, i))))
                    .collect();
                writer.add_vectors(vectors).unwrap();
            }
            writer.commit().unwrap();
        }

        /// The normalized token vectors of a document, row-major.
        fn normalized(doc_id: u64, count: usize) -> Vec<f32> {
            (0..count)
                .flat_map(|i| {
                    let mut v = Vector::new(token(doc_id, i));
                    v.normalize();
                    v.data.as_ref().clone()
                })
                .collect()
        }

        /// What a factor-`f` Cosine field stores for a document.
        fn pooled(doc_id: u64, count: usize, factor: u32) -> Vec<f32> {
            let rows = normalized(doc_id, count);
            token_pooling::pool(&rows, DIM, factor, true).unwrap_or(rows)
        }

        /// Every document entry of every sealed segment, newest first.
        fn entries(index: &MultiVectorIndex) -> Vec<DocEntry> {
            index
                .shared
                .readers_newest_first()
                .unwrap()
                .iter()
                .flat_map(|reader| reader.entries().collect::<Vec<_>>())
                .collect()
        }

        fn raw_rows(index: &MultiVectorIndex, doc_id: u64) -> Vec<u8> {
            let snapshot = index.snapshot().unwrap();
            let rows = snapshot.rows(doc_id).unwrap().unwrap();
            rows.as_bytes().to_vec()
        }

        /// A document's token vectors as embedded, before normalization.
        fn raw(doc_id: u64, count: usize) -> Vec<Vector> {
            (0..count).map(|i| Vector::new(token(doc_id, i))).collect()
        }

        fn flat(vectors: &[Vector]) -> Vec<f32> {
            vectors
                .iter()
                .flat_map(|v| v.data.iter().copied())
                .collect()
        }

        fn cosine(factor: u32) -> TokenPooling {
            TokenPooling::new(DIM, factor, true)
        }

        /// A document's raw token vectors with the same vectors pooled in
        /// advance under `pooling`.
        fn prepared(doc_id: u64, count: usize, pooling: TokenPooling) -> FieldVectors {
            let vectors = raw(doc_id, count);
            FieldVectors {
                pooled: pooling.apply(&vectors),
                ..FieldVectors::new(FIELD, vectors)
            }
        }

        /// Commit one document, given through `add_prepared_document_vectors`,
        /// as one segment.
        fn commit_prepared(index: &MultiVectorIndex, doc_id: u64, fields: Vec<FieldVectors>) {
            let mut writer = index.writer().unwrap();
            writer
                .add_prepared_document_vectors(doc_id, fields)
                .unwrap();
            writer.commit().unwrap();
        }

        fn binary_params(index: &MultiVectorIndex) -> Option<BinaryParams> {
            index
                .shared
                .readers_newest_first()
                .unwrap()
                .iter()
                .find_map(|reader| reader.binary_params().map(|p| (**p).clone()))
        }

        #[test]
        fn test_ingest_pools_and_marks_documents() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::F32, 2);
            commit_docs(&index, &[(1, 7), (2, 2)]);
            assert_eq!(
                entries(&index),
                [
                    DocEntry {
                        doc_id: 1,
                        vector_count: 4,
                        pool_factor: 2,
                    },
                    // Two rows cannot be reduced, but they were pooled by
                    // factor 2 all the same.
                    DocEntry {
                        doc_id: 2,
                        vector_count: 2,
                        pool_factor: 2,
                    },
                ]
            );
            assert_eq!(vectors_of(&index, 1).unwrap(), pooled(1, 7, 2));
            assert_eq!(vectors_of(&index, 2).unwrap(), normalized(2, 2));
        }

        #[test]
        fn test_unpooled_fields_mark_nothing() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::F32, 1);
            commit_docs(&index, &[(1, 7)]);
            assert_eq!(entries(&index)[0].pool_factor, 0);
            assert_eq!(vectors_of(&index, 1).unwrap(), normalized(1, 7));
        }

        #[test]
        fn test_merges_never_pool_pooled_documents_again() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::F32, 2);
            commit_docs(&index, &[(1, 9), (2, 5)]);
            commit_docs(&index, &[(3, 12)]);
            let before: Vec<Vec<u8>> = (1..=3).map(|d| raw_rows(&index, d)).collect();
            index.optimize().unwrap();
            let after = entries(&index);
            assert_eq!(after.len(), 3);
            assert!(after.iter().all(|e| e.pool_factor == 2), "{after:?}");
            for (doc_id, bytes) in (1..=3).zip(&before) {
                assert_eq!(&raw_rows(&index, doc_id), bytes, "doc {doc_id}");
            }
        }

        #[test]
        fn test_turning_pooling_on_pools_the_stored_vectors_like_ingest() {
            let docs = [(1, 9), (2, 5), (3, 2), (4, 30)];
            let storage = storage();
            {
                let index = open(&storage, MultiVectorStorage::F32, 1);
                commit_docs(&index, &docs[..2]);
                commit_docs(&index, &docs[2..]);
            }
            // What `update_field` does for 1 -> 4: reopen, then optimize.
            let index = open(&storage, MultiVectorStorage::F32, 4);
            index.optimize().unwrap();

            let fresh_storage = self::storage();
            let fresh = open(&fresh_storage, MultiVectorStorage::F32, 4);
            commit_docs(&fresh, &docs);
            for &(doc_id, count) in &docs {
                assert_eq!(
                    raw_rows(&index, doc_id),
                    raw_rows(&fresh, doc_id),
                    "doc {doc_id}"
                );
                assert_eq!(
                    vectors_of(&index, doc_id).unwrap().len(),
                    pooled_count(count, 4) * DIM
                );
            }
            assert!(entries(&index).iter().all(|e| e.pool_factor == 4));
        }

        #[test]
        fn test_a_merge_without_pooling_keeps_each_documents_marker() {
            let storage = storage();
            {
                let index = open(&storage, MultiVectorStorage::F32, 2);
                commit_docs(&index, &[(1, 9)]);
            }
            let index = open(&storage, MultiVectorStorage::F32, 1);
            commit_docs(&index, &[(2, 9)]);
            index.optimize().unwrap();
            let after = entries(&index);
            assert_eq!(
                after.iter().map(|e| e.pool_factor).collect::<Vec<_>>(),
                [2, 0]
            );
            assert_eq!(vectors_of(&index, 1).unwrap(), pooled(1, 9, 2));
            assert_eq!(vectors_of(&index, 2).unwrap(), normalized(2, 9));
        }

        /// 64 documents of 32 tokens: 2048 token vectors as ingested, 1024
        /// after factor-2 pooling.
        fn corpus(first: u64, docs: u64) -> Vec<(u64, usize)> {
            (first..first + docs).map(|d| (d, 32)).collect()
        }

        #[test]
        fn test_binary_counts_and_learns_from_pooled_vectors_at_flush() {
            let storage = storage();
            let index = open(&storage, MultiVectorStorage::Binary, 2);
            commit_docs(&index, &corpus(0, 64));
            assert_eq!(binary_params(&index), None, "1024 pooled rows stay interim");

            let storage = self::storage();
            let index = open(&storage, MultiVectorStorage::Binary, 2);
            let docs = corpus(0, 128);
            commit_docs(&index, &docs);
            let sample: Vec<f32> = docs.iter().flat_map(|&(d, n)| pooled(d, n, 2)).collect();
            assert_eq!(sample.len() / DIM, BinaryParams::MIN_TRAINING_ROWS);
            assert_eq!(
                binary_params(&index),
                Some(BinaryParams::train(DIM, &sample).unwrap())
            );
        }

        #[test]
        fn test_binary_counts_and_learns_from_pooled_vectors_at_merge() {
            let docs = corpus(0, 128);
            let storage = storage();
            {
                let index = open(&storage, MultiVectorStorage::F32, 1);
                commit_docs(&index, &docs);
            }
            let index = open(&storage, MultiVectorStorage::Binary, 2);
            index.optimize().unwrap();
            let sample: Vec<f32> = docs.iter().flat_map(|&(d, n)| pooled(d, n, 2)).collect();
            assert_eq!(
                binary_params(&index),
                Some(BinaryParams::train(DIM, &sample).unwrap())
            );
            assert!(entries(&index).iter().all(|e| e.pool_factor == 2));
        }

        /// A field that, before pooling was turned on, stored documents of
        /// several lengths (two too short to pool), a stale copy and a
        /// deleted document, plus one document a factor-2 field pooled.
        fn mixed_storage() -> Arc<dyn Storage> {
            let storage = storage();
            {
                let index = open(&storage, MultiVectorStorage::F32, 1);
                commit_docs(&index, &[(1, 9), (2, 2), (3, 30), (4, 5)]);
                commit_docs(&index, &[(5, 17), (6, 1), (3, 12)]);
                index.soft_delete_document(4).unwrap();
                index.persist_deletions().unwrap();
            }
            {
                let index = open(&storage, MultiVectorStorage::F32, 2);
                commit_docs(&index, &[(7, 11)]);
            }
            storage
        }

        /// Merge every segment of [`mixed_storage`] into a factor-2 field,
        /// pooling in windows of `budget` bytes on `threads` threads.
        ///
        /// # Returns
        ///
        /// The merged index and its one segment's bytes.
        #[cfg(feature = "native")]
        fn merged_bytes(budget: usize, threads: usize) -> (MultiVectorIndex, Vec<u8>) {
            let storage = mixed_storage();
            let index = open(&storage, MultiVectorStorage::F32, 2);
            let segments = index.shared.core.manager().list_segments();
            let deletions = index.shared.core.deletion_snapshot().unwrap();
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    index
                        .shared
                        .merge_within(segments, deletions.as_deref(), budget)
                })
                .unwrap();
            let merged = index.shared.core.manager().list_segments();
            assert_eq!(merged.len(), 1);
            let mut bytes = Vec::new();
            storage
                .open_input(&Shared::file_name(&merged[0].segment_id))
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            (index, bytes)
        }

        /// Issue #1429: a merge pools window by window, each window's
        /// documents at the same time, and writes the same segment whatever
        /// the windows and the threads: what ingest by factor 2 stores.
        #[cfg(feature = "native")]
        #[test]
        fn test_merges_write_the_same_segment_whatever_the_windows_and_threads() {
            let (index, reference) = merged_bytes(usize::MAX, 1);
            // Documents 1, 3 and 5 shrink; this budget groups 1 and 3.
            let medium = pooling_bytes(9, DIM) + pooling_bytes(17, DIM);
            for budget in [0, medium, usize::MAX] {
                for threads in [1, 4] {
                    assert_eq!(
                        merged_bytes(budget, threads).1,
                        reference,
                        "budget {budget}, {threads} threads"
                    );
                }
            }

            let fresh = open(&storage(), MultiVectorStorage::F32, 2);
            commit_docs(&fresh, &[(1, 9), (2, 2), (3, 12), (5, 17), (6, 1), (7, 11)]);
            for doc_id in [1, 2, 3, 5, 6, 7] {
                assert_eq!(
                    raw_rows(&index, doc_id),
                    raw_rows(&fresh, doc_id),
                    "doc {doc_id}"
                );
            }
            assert_eq!(vectors_of(&index, 4), None);
            assert!(entries(&index).iter().all(|e| e.pool_factor == 2));
        }

        #[test]
        fn test_token_pooling_reports_the_fields_pooling() {
            assert_eq!(
                open(&storage(), MultiVectorStorage::F32, 1).token_pooling(FIELD),
                None
            );
            assert_eq!(
                open(&storage(), MultiVectorStorage::F32, 3).token_pooling(FIELD),
                Some(cosine(3))
            );
        }

        /// Vectors pooled the way the field pools are stored as they are:
        /// neither pooled nor normalized again.
        #[test]
        fn test_vectors_pooled_the_way_the_field_pools_are_kept_bit_for_bit() {
            let index = open(&storage(), MultiVectorStorage::F32, 2);
            // Half the real pooled rows: no longer unit vectors, so
            // normalizing them again changes them, and not what the writer
            // pools from the raw vectors.
            let kept: Vec<Vector> = pooled(1, 7, 2)
                .as_chunks::<DIM>()
                .0
                .iter()
                .map(|row| Vector::new(row.iter().map(|x| x * 0.5).collect()))
                .collect();
            let renormalized: Vec<Vector> = kept.iter().map(Vector::normalized).collect();
            assert_ne!(flat(&renormalized), flat(&kept));
            assert_ne!(flat(&kept), pooled(1, 7, 2));

            let tagged = PooledTokens::tagged(cosine(2), 7, kept.clone());
            commit_prepared(
                &index,
                1,
                vec![FieldVectors {
                    pooled: Some(tagged),
                    ..FieldVectors::new(FIELD, raw(1, 7))
                }],
            );
            assert_eq!(vectors_of(&index, 1).unwrap(), flat(&kept));
            assert_eq!(entries(&index)[0].pool_factor, 2);
        }

        /// What `TokenPooling::apply` pools outside the writer is what the
        /// writer stores when it pools the raw vectors itself.
        #[test]
        fn test_prepared_vectors_store_what_the_writer_pools() {
            let docs = [(1, 9), (2, 2), (3, 30)];
            let index = open(&storage(), MultiVectorStorage::F32, 2);
            for &(doc_id, count) in &docs {
                commit_prepared(&index, doc_id, vec![prepared(doc_id, count, cosine(2))]);
            }
            let fresh = open(&storage(), MultiVectorStorage::F32, 2);
            commit_docs(&fresh, &docs);
            for &(doc_id, _) in &docs {
                assert_eq!(
                    raw_rows(&index, doc_id),
                    raw_rows(&fresh, doc_id),
                    "doc {doc_id}"
                );
            }
        }

        /// Vectors pooled under another factor are pooled again from the
        /// raw vectors, so the marker always describes the rows.
        #[test]
        fn test_vectors_pooled_by_another_factor_are_pooled_again() {
            let index = open(&storage(), MultiVectorStorage::F32, 4);
            commit_prepared(&index, 1, vec![prepared(1, 9, cosine(2))]);
            assert_eq!(vectors_of(&index, 1).unwrap(), pooled(1, 9, 4));
            assert_eq!(entries(&index)[0].pool_factor, 4);

            let index = open(&storage(), MultiVectorStorage::F32, 1);
            commit_prepared(&index, 1, vec![prepared(1, 9, cosine(4))]);
            assert_eq!(vectors_of(&index, 1).unwrap(), normalized(1, 9));
            assert_eq!(entries(&index)[0].pool_factor, 0);
        }

        /// Vectors pooled for a `Cosine` field (normalized) are pooled again
        /// by a field of the same factor that does not normalize.
        #[test]
        fn test_vectors_pooled_for_another_metric_are_pooled_again() {
            let config = MultiVectorIndexConfig {
                distance_metric: DistanceMetric::DotProduct,
                ..config(MultiVectorStorage::F32, 2)
            };
            let index = MultiVectorIndex::open_or_create(storage(), "index", config).unwrap();
            commit_prepared(&index, 1, vec![prepared(1, 9, cosine(2))]);
            let expected = token_pooling::pool(&flat(&raw(1, 9)), DIM, 2, false).unwrap();
            assert_ne!(expected, pooled(1, 9, 2));
            assert_eq!(vectors_of(&index, 1).unwrap(), expected);
        }

        /// Pooled vectors that were pooled from a different number of
        /// vectors than they come with are not kept.
        #[test]
        fn test_vectors_pooled_from_other_vectors_are_pooled_again() {
            let index = open(&storage(), MultiVectorStorage::F32, 2);
            commit_prepared(
                &index,
                1,
                vec![FieldVectors {
                    pooled: cosine(2).apply(&raw(1, 9)),
                    ..FieldVectors::new(FIELD, raw(1, 7))
                }],
            );
            assert_eq!(vectors_of(&index, 1).unwrap(), pooled(1, 7, 2));
        }

        /// Several entries for one document are pooled together, whatever
        /// the first entry was pooled to on its own.
        #[test]
        fn test_several_entries_pool_their_concatenation() {
            let index = open(&storage(), MultiVectorStorage::F32, 2);
            let all = raw(1, 9);
            let first = FieldVectors {
                pooled: cosine(2).apply(&all[..5]),
                ..FieldVectors::new(FIELD, all[..5].to_vec())
            };
            assert!(first.pooled.is_some());
            commit_prepared(
                &index,
                1,
                vec![first, FieldVectors::new(FIELD, all[5..].to_vec())],
            );
            assert_eq!(vectors_of(&index, 1).unwrap(), pooled(1, 9, 2));
        }

        /// A vector of another dimension is rejected even when the document
        /// comes with pooled vectors the writer would keep.
        #[test]
        fn test_vectors_of_another_dimension_are_rejected_even_when_pooled() {
            let index = open(&storage(), MultiVectorStorage::F32, 2);
            let mut vectors = raw(1, 7);
            let pooled = cosine(2).apply(&vectors);
            assert!(pooled.is_some());
            vectors[3] = Vector::new(vec![1.0; DIM + 1]);
            let mut writer = index.writer().unwrap();
            let err = writer
                .add_prepared_document_vectors(
                    1,
                    vec![FieldVectors {
                        pooled,
                        ..FieldVectors::new(FIELD, vectors)
                    }],
                )
                .unwrap_err();
            assert!(err.to_string().contains("dimension"), "{err}");
            assert!(!writer.has_pending_changes());
        }
    }
}
