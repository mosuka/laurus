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
//! A field configured for [`MultiVectorStorage::Binary`] learns its
//! [`BinaryParams`] once, from the first flush or merge that holds at least
//! [`BinaryParams::MIN_TRAINING_ROWS`] token vectors, and encodes every
//! later segment with them. Segments written before that are stored as
//! [`INTERIM_KIND`] and converted by a later merge.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::field::multi_vector_params_error;
use crate::vector::core::multi_vector::{BinaryParams, MultiVectorStorage, OwnedCodec};
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
        let mut binary = self.binary.lock();
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

        let rows = entries.iter().map(|e| e.vector_count as usize).sum();
        let (target, trained) = self.resolve_target(&mut binary, rows, || {
            training_sample(
                &survivors,
                rows,
                self.config.dimension,
                |(entry, source), out| {
                    let doc = readers[*source].rows(entry.doc_id)?.ok_or_else(|| {
                        LaurusError::internal("merge source lost a document listed in its table")
                    })?;
                    out.extend_from_slice(&doc.to_f32());
                    Ok(())
                },
            )
        })?;

        let merged_id = self.core.manager().generate_segment_id();
        write_segment(
            self.core.storage().as_ref(),
            &Self::file_name(&merged_id),
            target.as_codec(),
            self.config.dimension,
            &entries,
            |i, sink| {
                let (entry, source) = survivors[i];
                let rows = readers[source].rows(entry.doc_id)?.ok_or_else(|| {
                    LaurusError::internal("merge source lost a document listed in its table")
                })?;
                // Rows encoded exactly as the target encodes are copied
                // verbatim (no re-quantization, no compounding loss across
                // merges); others (the field's `storage` just changed) are
                // transcoded through f32, which is also how `update_field`
                // migrates every existing segment to the new kind.
                sink.append(&rows)
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
        if let Some(reason) =
            multi_vector_params_error(config.dimension, config.distance_metric, config.storage)
        {
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
    /// order, already checked by [`Self::check_dimensions`]: normalize them
    /// when the metric needs it, clear the documents' deletion marks, and
    /// only then replace their buffered entries, so a failure leaves the
    /// buffer untouched.
    fn buffer_documents(&mut self, docs: &mut [(u64, Vec<Vector>)]) -> Result<()> {
        if self.shared.config.distance_metric == DistanceMetric::Cosine {
            docs.iter_mut()
                .flat_map(|(_, vectors)| vectors.iter_mut())
                .for_each(Vector::normalize);
        }
        self.shared
            .core
            .unmark_deleted(docs.iter().map(|(doc_id, _)| *doc_id))?;
        if !docs.is_empty() {
            self.mark_changed();
        }
        for (doc_id, vectors) in docs {
            self.buffer.insert(*doc_id, std::mem::take(vectors));
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
        let mut docs: Vec<(u64, Vec<Vector>)> = incoming.into_iter().collect();
        self.buffer_documents(&mut docs)
    }

    /// Buffer one document's token vectors as [`Self::add_vectors`] does,
    /// without a field name per token (Issue #1406). This writer serves a
    /// single field, so the names are ignored and every entry's vectors
    /// are taken in order.
    fn add_document_vectors(
        &mut self,
        doc_id: u64,
        fields: Vec<(String, Vec<Vector>)>,
    ) -> Result<()> {
        let mut fields = fields.into_iter();
        let Some((_, mut vectors)) = fields.next() else {
            return Ok(());
        };
        for (_, more) in fields {
            vectors.extend(more);
        }
        if vectors.is_empty() {
            // Nothing to buffer. Clearing the deletion mark would make the
            // sealed copy that the caller's `delete_document` hid visible
            // again, which `add_vectors` never does for a document it was
            // given no vectors for.
            return Ok(());
        }
        self.check_dimensions(vectors.iter().map(|vector| (doc_id, vector)))?;
        self.buffer_documents(&mut [(doc_id, vectors)])
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
}
