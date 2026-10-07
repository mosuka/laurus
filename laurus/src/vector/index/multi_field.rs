//! Multi-field vector index: one independent sub-index per vector field
//! (Issue [#948](https://github.com/mosuka/laurus/issues/948)).
//!
//! # The bug this fixes
//!
//! Before this module existed, [`crate::vector::store::VectorStore`] routed
//! every configured vector field through a *single* [`VectorIndex`]
//! instance (see the old `VectorStore::extract_index_type_config`, which
//! picked one field's config and silently discarded the rest). For HNSW in
//! particular this was silently data-destroying: the graph's node IDs *are*
//! `doc_id`s, so a document with vectors in two fields (e.g. `title_vec` and
//! `body_vec`) landed as two nodes with the same ID in the same graph — the
//! second `add_vectors` call overwrote the first, and which field "won" was
//! non-deterministic across runs.
//!
//! # The fix
//!
//! [`MultiFieldVectorIndex`] gives each vector field its own independent
//! sub-index (typically a [`SegmentedHnswIndex`](crate::vector::index::hnsw::segmented::SegmentedHnswIndex),
//! but any [`VectorIndex`] works), each backed by its own storage namespace
//! via [`PrefixedStorage::new(field_name, storage)`](PrefixedStorage::new)
//! — the same technique [`crate::engine::Engine`] already uses to separate
//! `lexical/` / `vector/` / `documents/` under one physical backend. Because
//! every field gets its own graph, its own `segments.json` manifest, and
//! its own deletion bitmap, two fields sharing a `doc_id` can never collide
//! — the bug's root cause is structurally impossible here, not just
//! patched over.
//!
//! No on-disk format changes: each field's sub-index is a completely
//! ordinary [`SegmentedHnswIndex`]/`SegmentedFlatIndex`/`SegmentedIvfIndex`,
//! read exactly as before. Only the *routing* above them is new.
//!
//! # No migration from the pre-#948 layout
//!
//! [`Self::open_or_create`] does NOT detect or migrate the old
//! single-index layout every prior release used for vector fields (one
//! [`VectorIndex`] shared by all fields, opened under the fixed name
//! `"vector_index"`). Opening a collection that has data in that layout
//! creates fresh, empty per-field sub-indexes instead: the old root-level
//! files are left on disk untouched (not deleted) but become invisible to
//! the new per-field routing, so existing vector search results
//! disappear until the collection is reindexed. This was a deliberate
//! scope decision (not an oversight): the affected data is
//! reconstructible by reindexing, and adding automatic migration would
//! have added substantial, security/correctness-sensitive complexity
//! (detecting the legacy layout, resolving its index type, splitting
//! mixed-field data, crash-resumability) for a pre-1.0 project. Lexical
//! and document data are unaffected either way.
//!
//! # PQ codebook namespacing caveat (for callers of [`MultiFieldVectorIndex::open_or_create`])
//!
//! [`crate::vector::index::config::HnswIndexConfig::pq_codebook_path`] is a
//! path relative to the *root* storage (the one passed to `open_or_create`,
//! not any per-field [`PrefixedStorage`]) — see
//! `laurus-cli/src/commands/train.rs`'s path contract. Callers that resolve
//! a shared PQ codebook (Issue #631) MUST call
//! [`HnswIndexConfig::resolve_pq_codebook`](crate::vector::index::config::HnswIndexConfig::resolve_pq_codebook)
//! against the *root* storage before building the per-field config passed
//! here. `VectorIndexFactory::open_or_create`'s own internal resolution
//! attempt (against the field-scoped storage, where the codebook file does
//! not exist) is a harmless no-op in that case —
//! [`HnswIndexConfig::resolve_pq_codebook`] only overwrites an already-resolved
//! codebook when it finds a *replacement* file, never clears one to `None`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::VectorIndexTypeConfig;
use crate::vector::index::factory::VectorIndexFactory;
use crate::vector::index::multivector::MultiVectorSnapshot;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::{
    SimpleVectorIterator, ValidationReport, VectorIndexMetadata, VectorIndexReader, VectorIterator,
    VectorStats,
};
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher,
};
use crate::vector::writer::VectorIndexWriter;

/// `(doc_id, field_name, Vector)` triples bucketed by field name -- the
/// output of [`MultiFieldWriter::group_by_field`].
type FieldGroupedVectors = BTreeMap<String, Vec<(u64, String, Vector)>>;

/// Fixed sub-index name used inside every field's [`PrefixedStorage`]
/// namespace (as opposed to `"vector_index"`, the historical single-index
/// name). A fixed, non-field-derived name sidesteps
/// [`SegmentedHnswIndex::open_or_create`](crate::vector::index::hnsw::segmented::SegmentedHnswIndex::open_or_create)'s
/// reserved `segment_<digits>` name check: since it is never the field
/// name itself, a field literally named `"segment_0"` still works.
pub(crate) const SUB_INDEX_NAME: &str = "index";

/// Run `f` over every value in `map`, tolerating individual failures until
/// every entry has been attempted.
///
/// Used for the "always try every field" contract this module leans on
/// throughout (`close`, `optimize`, `commit`, soft-delete propagation, ...):
/// a transient failure on one field's storage must not leave a sibling
/// field silently uncommitted or unclosed. Returns the *first* error seen,
/// after every entry has had a chance to run.
fn try_all<'a, V: 'a, F>(values: impl Iterator<Item = &'a V>, mut f: F) -> Result<()>
where
    F: FnMut(&'a V) -> Result<()>,
{
    let mut first_err = None;
    for v in values {
        if let Err(e) = f(v)
            && first_err.is_none()
        {
            first_err = Some(e);
        }
    }
    first_err.map_or(Ok(()), Err)
}

/// Same as [`try_all`] but over mutable references (for writer methods,
/// which take `&mut self`).
fn try_all_mut<'a, V: 'a, F>(values: impl Iterator<Item = &'a mut V>, mut f: F) -> Result<()>
where
    F: FnMut(&'a mut V) -> Result<()>,
{
    let mut first_err = None;
    for v in values {
        if let Err(e) = f(v)
            && first_err.is_none()
        {
            first_err = Some(e);
        }
    }
    first_err.map_or(Ok(()), Err)
}

/// One field's sub-index plus the cached geometry
/// ([`FieldEntry::dimension`] / [`FieldEntry::distance_metric`]) needed for
/// fan-out routing without paying for a `reader()`/`stats()` round trip on
/// every query.
#[derive(Debug, Clone)]
struct FieldEntry {
    index: Arc<dyn VectorIndex>,
    dimension: usize,
    distance_metric: DistanceMetric,
    /// Holds token vectors for late-interaction rescoring (Issue #1177)
    /// rather than searchable vectors.
    multi_vector: bool,
}

impl FieldEntry {
    fn new(index: Box<dyn VectorIndex>, config: &VectorIndexTypeConfig) -> Self {
        Self {
            dimension: config.dimension(),
            distance_metric: config.distance_metric(),
            multi_vector: config.is_multi_vector(),
            index: Arc::from(index),
        }
    }
}

/// A [`VectorIndex`] that routes each vector field to its own independent
/// sub-index (see the module docs for the bug this fixes).
///
/// `fields` is a [`BTreeMap`] (not a `HashMap`): iteration order feeds
/// directly into deterministic aggregate behavior — e.g. [`Self::stats`]'s
/// representative `dimension` and [`MultiFieldReaderFacade`]'s
/// single-value `dimension()`/`distance_metric()` fall back to the
/// *first* field in iteration order, which must not depend on hash seed.
pub struct MultiFieldVectorIndex {
    fields: RwLock<BTreeMap<String, FieldEntry>>,
    /// Root storage namespace (pre-[`PrefixedStorage`]). Exposed via
    /// [`VectorIndex::storage`] because callers such as PQ codebook
    /// training (`Engine::train_pq_codebook`, `laurus-cli`'s `train`
    /// command) write/read root-relative paths — see the module docs'
    /// PQ codebook caveat.
    storage: Arc<dyn Storage>,
    embedder: Arc<dyn Embedder>,
    closed: AtomicBool,
}

impl std::fmt::Debug for MultiFieldVectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Mirrors `HnswIndexConfig`'s `Debug` impl: print the embedder's
        // *name*, not its full contents -- a real embedder (e.g. a loaded
        // BERT model) can hold megabytes of tensor data that would make a
        // derived `Debug` unreadable.
        f.debug_struct("MultiFieldVectorIndex")
            .field("fields", &*self.fields.read())
            .field("embedder", &self.embedder.name())
            .field("closed", &self.is_closed())
            .finish()
    }
}

impl MultiFieldVectorIndex {
    /// Open an existing multi-field index or create a new one, one
    /// sub-index per entry in `field_configs`.
    ///
    /// Does NOT detect or migrate the pre-#948 single-index layout — see
    /// the module docs' "No migration" section.
    ///
    /// # Arguments
    ///
    /// * `storage` - Root storage namespace; each field gets its own
    ///   `PrefixedStorage::new(field_name, storage.clone())` sub-namespace.
    /// * `field_configs` - Per-field index configuration. A `BTreeMap` so
    ///   construction order (and therefore any first-error reporting) is
    ///   deterministic.
    /// * `embedder` - Shared embedder (typically a
    ///   [`PerFieldEmbedder`](crate::embedding::per_field::PerFieldEmbedder))
    ///   used by every field.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        field_configs: &BTreeMap<String, VectorIndexTypeConfig>,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self> {
        let mut fields = BTreeMap::new();
        for (name, config) in field_configs {
            let field_storage: Arc<dyn Storage> =
                Arc::new(PrefixedStorage::new(name.clone(), storage.clone()));
            let index =
                VectorIndexFactory::open_or_create(field_storage, SUB_INDEX_NAME, config.clone())?;
            fields.insert(name.clone(), FieldEntry::new(index, config));
        }

        Ok(Self {
            fields: RwLock::new(fields),
            storage,
            embedder,
            closed: AtomicBool::new(false),
        })
    }

    /// The minimum `last_wal_seq` across every field (see
    /// [`VectorIndex::last_wal_seq`]'s override below for why `min`, not
    /// `max`, is the safe aggregate). `0` when there are no fields yet.
    fn min_wal_seq(fields: &BTreeMap<String, FieldEntry>) -> u64 {
        fields
            .values()
            .map(|f| f.index.last_wal_seq())
            .min()
            .unwrap_or(0)
    }
}

impl VectorIndex for MultiFieldVectorIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        let fields = self.fields.read();
        let mut readers = BTreeMap::new();
        for (name, entry) in fields.iter() {
            readers.insert(name.clone(), entry.index.reader()?);
        }
        Ok(Arc::new(MultiFieldReaderFacade::new(readers)))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        let fields = self.fields.read();
        let mut writers = BTreeMap::new();
        for (name, entry) in fields.iter() {
            writers.insert(name.clone(), entry.index.writer()?);
        }
        Ok(Box::new(MultiFieldWriter::new(writers)))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        try_all(self.fields.read().values(), |f| f.index.close())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        let fields = self.fields.read();
        let mut vector_count = 0u64;
        let mut total_size = 0u64;
        let mut deleted_count = 0u64;
        let mut last_modified = 0u64;
        let mut dimension = 0usize;
        for entry in fields.values() {
            let s = entry.index.stats()?;
            vector_count += s.vector_count;
            total_size += s.total_size;
            deleted_count += s.deleted_count;
            last_modified = last_modified.max(s.last_modified);
            if dimension == 0 {
                // Representative only: a genuinely per-field breakdown is
                // `VectorStore::stats()`'s job (via `field_dimensions` /
                // per-field readers), not this low-level aggregate.
                dimension = s.dimension;
            }
        }
        Ok(VectorIndexStats {
            vector_count,
            dimension,
            total_size,
            deleted_count,
            last_modified,
        })
    }

    fn optimize(&self) -> Result<()> {
        try_all(self.fields.read().values(), |f| f.index.optimize())
    }

    fn refresh(&self) -> Result<()> {
        try_all(self.fields.read().values(), |f| f.index.refresh())
    }

    fn retain_writer_after_commit(&self) -> bool {
        let fields = self.fields.read();
        // Conservative AND: retaining is sound only when every field's
        // sub-index individually certifies it (see the trait doc's two
        // conditions); one field opting out must not let a store keep a
        // stale writer for the others.
        !fields.is_empty()
            && fields
                .values()
                .all(|f| f.index.retain_writer_after_commit())
    }

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        let fields = self.fields.read();
        let mut searchers = BTreeMap::new();
        for (name, entry) in fields.iter() {
            searchers.insert(
                name.clone(),
                SearcherEntry {
                    searcher: entry.index.searcher()?,
                    dimension: entry.dimension,
                    distance_metric: entry.distance_metric,
                    multi_vector: entry.multi_vector,
                },
            );
        }
        Ok(Box::new(MultiFieldFanoutSearcher::new(searchers)))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        self.embedder.clone()
    }

    fn last_wal_seq(&self) -> u64 {
        // The MINIMUM across fields, not the maximum: this value is the
        // replay checkpoint, and a field that lags behind still needs its
        // WAL records replayed. Since upsert is a delete-then-add (Issue
        // #948 plan), re-applying an already-applied record to an
        // up-to-date field is idempotent and safe; the reverse (skipping a
        // record a lagging field still needs) would lose data.
        Self::min_wal_seq(&self.fields.read())
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        try_all(self.fields.read().values(), |f| {
            f.index.set_last_wal_seq(seq)
        })
    }

    fn supports_soft_delete(&self) -> bool {
        let fields = self.fields.read();
        !fields.is_empty() && fields.values().all(|f| f.index.supports_soft_delete())
    }

    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        try_all(self.fields.read().values(), |f| {
            f.index.soft_delete_document(doc_id)
        })
    }

    fn persist_deletions(&self) -> Result<()> {
        try_all(self.fields.read().values(), |f| f.index.persist_deletions())
    }

    fn maybe_auto_compact(&self) -> Result<bool> {
        let fields = self.fields.read();
        let mut compacted_any = false;
        let mut first_err = None;
        for f in fields.values() {
            match f.index.maybe_auto_compact() {
                Ok(true) => compacted_any = true,
                Ok(false) => {}
                Err(e) if first_err.is_none() => first_err = Some(e),
                Err(_) => {}
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(compacted_any),
        }
    }

    fn supports_dynamic_fields(&self) -> bool {
        true
    }

    fn add_field(&self, name: &str, config: VectorIndexTypeConfig) -> Result<()> {
        let mut fields = self.fields.write();
        if fields.contains_key(name) {
            return Err(LaurusError::invalid_argument(format!(
                "vector field '{name}' already exists"
            )));
        }
        let field_storage: Arc<dyn Storage> =
            Arc::new(PrefixedStorage::new(name.to_string(), self.storage.clone()));
        let index =
            VectorIndexFactory::open_or_create(field_storage, SUB_INDEX_NAME, config.clone())?;
        // Seed the new field's checkpoint with the current aggregate
        // (min-across-fields) so it does not drag `last_wal_seq()` (also a
        // min) backwards -- a fresh field starting at 0 would otherwise
        // make replay re-walk WAL records every other field already
        // applied, on every open, forever.
        let current_min = Self::min_wal_seq(&fields);
        index.set_last_wal_seq(current_min)?;
        fields.insert(name.to_string(), FieldEntry::new(index, &config));
        Ok(())
    }

    fn remove_field(&self, name: &str, purge: bool) -> Result<()> {
        // Unregister first (mirrors `VectorStore::delete_field`'s existing
        // contract when `purge` is false: does not delete on-disk data, so
        // re-adding the same field name later recovers it).
        self.fields.write().remove(name);

        if purge {
            // Issue #1080: physically remove the field's files so a
            // same-name re-add under a DIFFERENT type cannot misread old
            // bytes. `list_files` on the field's own `PrefixedStorage`
            // already returns only this field's (prefix-stripped) names.
            let field_storage = PrefixedStorage::new(name.to_string(), self.storage.clone());
            let files = field_storage.list_files()?;
            try_all(files.iter(), |file| field_storage.delete_file(file))?;
        }
        Ok(())
    }

    fn rebuild_field(&self, name: &str, new_config: VectorIndexTypeConfig) -> Result<()> {
        if !self.fields.read().contains_key(name) {
            return Err(LaurusError::invalid_argument(format!(
                "vector field '{name}' does not exist"
            )));
        }

        // Reopen the SAME physical namespace under the new config -- this
        // does not move or lose any data, it only changes which config the
        // existing segments are read/merged under. `open_or_create` always
        // takes the `open` path here (the field's `metadata.json` already
        // exists), so the existing segment manifest is picked up as-is.
        let field_storage: Arc<dyn Storage> =
            Arc::new(PrefixedStorage::new(name.to_string(), self.storage.clone()));
        let index =
            VectorIndexFactory::open_or_create(field_storage, SUB_INDEX_NAME, new_config.clone())?;

        // Force-merge every existing segment into one new segment under
        // the new config. `optimize()` writes the new segment fully before
        // atomically publishing it (see `SegmentedHnswIndex::optimize`), so
        // a failure here never touches `self.fields` below -- the field's
        // existing data is left completely untouched.
        index.optimize()?;

        self.fields
            .write()
            .insert(name.to_string(), FieldEntry::new(index, &new_config));
        Ok(())
    }

    fn field_dimensions(&self) -> BTreeMap<String, usize> {
        self.fields
            .read()
            .iter()
            .map(|(name, entry)| (name.clone(), entry.dimension))
            .collect()
    }

    fn multi_vector_snapshot(&self, field: &str) -> Result<Option<MultiVectorSnapshot>> {
        let index = match self.fields.read().get(field) {
            Some(entry) => entry.index.clone(),
            None => {
                return Err(LaurusError::invalid_argument(format!(
                    "unknown vector field '{field}': no index configured for it"
                )));
            }
        };
        index.multi_vector_snapshot(field)
    }

    fn late_interaction_fields(&self) -> BTreeSet<String> {
        self.fields
            .read()
            .iter()
            .filter(|(_, entry)| entry.multi_vector)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Writer over a [`MultiFieldVectorIndex`]'s per-field sub-writers.
///
/// This is the core of the bug fix: [`Self::add_vectors`] groups the
/// incoming `(doc_id, field_name, Vector)` triples by `field_name` and
/// routes each group to that field's own writer, so vectors for one field
/// are never buffered into the same `doc_id`-keyed structure as another
/// field's vectors.
///
/// No cross-field "poisoning" on a partial [`Self::commit`] failure: each
/// field's own writer (e.g. `SegmentedHnswWriter`) already refuses to
/// re-seal itself once sealed, so mutating and retrying after a failure is
/// exactly as safe here as it is for a single-field writer -- a field that
/// failed to seal still holds its unsealed buffer and simply retries; a
/// field that DID seal is untouched by further mutation unless the caller
/// explicitly adds to it again, in which case ITS OWN next `commit()`
/// surfaces that as an error. Adding an extra poison flag on top would only
/// block the ordinary "commit failed, keep buffering, retry later"
/// workflow that callers (e.g. `VectorStore::commit`'s writer retention on
/// failure) already rely on.
#[derive(Debug)]
struct MultiFieldWriter {
    writers: BTreeMap<String, Box<dyn VectorIndexWriter>>,
    closed: bool,
}

impl MultiFieldWriter {
    fn new(writers: BTreeMap<String, Box<dyn VectorIndexWriter>>) -> Self {
        Self {
            writers,
            closed: false,
        }
    }

    /// Group `vectors` by field name, validating every field name against
    /// this writer's known fields *before* routing any of them -- an
    /// unknown field name rejects the whole batch rather than silently
    /// dropping just that field's vectors while applying the rest.
    fn group_by_field(&self, vectors: Vec<(u64, String, Vector)>) -> Result<FieldGroupedVectors> {
        let mut grouped: FieldGroupedVectors = BTreeMap::new();
        for (doc_id, field_name, vector) in vectors {
            if !self.writers.contains_key(&field_name) {
                return Err(LaurusError::invalid_argument(format!(
                    "unknown vector field '{field_name}': no index configured for it"
                )));
            }
            grouped
                .entry(field_name.clone())
                .or_default()
                .push((doc_id, field_name, vector));
        }
        Ok(grouped)
    }
}

#[async_trait]
impl VectorIndexWriter for MultiFieldWriter {
    fn next_vector_id(&self) -> u64 {
        self.writers
            .values()
            .map(|w| w.next_vector_id())
            .max()
            .unwrap_or(0)
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.add_vectors(vectors)
    }

    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let grouped = self.group_by_field(vectors)?;
        for (field_name, field_vectors) in grouped {
            let writer = self
                .writers
                .get_mut(&field_name)
                .expect("field name validated by group_by_field");
            writer.add_vectors(field_vectors)?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        try_all_mut(self.writers.values_mut(), |w| w.finalize())
    }

    fn progress(&self) -> f32 {
        if self.writers.is_empty() {
            return 1.0;
        }
        self.writers.values().map(|w| w.progress()).sum::<f32>() / self.writers.len() as f32
    }

    fn estimated_memory_usage(&self) -> usize {
        self.writers
            .values()
            .map(|w| w.estimated_memory_usage())
            .sum()
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        // Contract mismatch, by necessity: this method borrows a single
        // contiguous buffer, but a `MultiFieldWriter` holds N independent
        // per-field buffers with no such backing slice to lend out. The
        // only production caller is `ManagedVectorIndex::vectors()`, which
        // is used for the single-index bench/direct-use construction path,
        // never for a `MultiFieldVectorIndex`; callers that need to inspect
        // buffered vectors here should go through `VectorIndex::reader()`
        // instead (via `Self::build_reader`).
        &[]
    }

    fn write(&self) -> Result<()> {
        try_all(self.writers.values(), |w| w.write())
    }

    fn has_storage(&self) -> bool {
        self.writers.values().all(|w| w.has_storage())
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        // A logical document spans every field it was indexed with; delete
        // it from ALL fields, not just whichever one the caller happens to
        // pass vectors for in the same batch.
        try_all_mut(self.writers.values_mut(), |w| w.delete_document(doc_id))
    }

    fn has_pending_changes(&self) -> bool {
        self.writers.values().any(|w| w.has_pending_changes())
    }

    fn delete_documents(&mut self, field: &str, value: &str) -> Result<usize> {
        let mut total = 0usize;
        let mut first_err = None;
        for w in self.writers.values_mut() {
            match w.delete_documents(field, value) {
                Ok(n) => total += n,
                Err(e) if first_err.is_none() => first_err = Some(e),
                Err(_) => {}
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(total),
        }
    }

    fn commit(&mut self) -> Result<()> {
        // Always attempt every field, even once one has failed, so a
        // transient failure on field A's storage does not leave field B's
        // otherwise-successful commit silently undone by an early return.
        let mut first_err = None;
        for w in self.writers.values_mut() {
            if let Err(e) = w.commit()
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    async fn add_value(
        &mut self,
        doc_id: u64,
        field_name: String,
        value: crate::data::DataValue,
    ) -> Result<()> {
        let writer = self.writers.get_mut(&field_name).ok_or_else(|| {
            LaurusError::invalid_argument(format!(
                "unknown vector field '{field_name}': no index configured for it"
            ))
        })?;
        writer.add_value(doc_id, field_name.clone(), value).await
    }

    fn rollback(&mut self) -> Result<()> {
        try_all_mut(self.writers.values_mut(), |w| w.rollback())
    }

    fn pending_docs(&self) -> u64 {
        // An approximation (max across fields), not a true union of
        // pending doc IDs: this is a diagnostic/monitoring value. The
        // correctness-critical boolean is `has_pending_changes`, which is
        // computed precisely above.
        self.writers
            .values()
            .map(|w| w.pending_docs())
            .max()
            .unwrap_or(0)
    }

    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let result = try_all_mut(self.writers.values_mut(), |w| w.close());
        self.closed = true;
        result
    }

    fn is_closed(&self) -> bool {
        self.closed || self.writers.values().all(|w| w.is_closed())
    }

    fn optimize(&mut self) -> Result<()> {
        try_all_mut(self.writers.values_mut(), |w| w.optimize())
    }

    fn build_reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        let mut readers = BTreeMap::new();
        for (name, w) in &self.writers {
            readers.insert(name.clone(), w.build_reader()?);
        }
        Ok(Arc::new(MultiFieldReaderFacade::new(readers)))
    }
}

/// One field's searcher plus the cached geometry needed for fan-out
/// routing (mirrors [`FieldEntry`]).
#[derive(Debug)]
struct SearcherEntry {
    searcher: Box<dyn VectorIndexSearcher>,
    dimension: usize,
    distance_metric: DistanceMetric,
    /// A multi-vector field: rejected when targeted explicitly, never a
    /// fan-out candidate.
    multi_vector: bool,
}

/// Searcher over a [`MultiFieldVectorIndex`]'s per-field sub-searchers.
///
/// A field-targeted query (`request.field_name.is_some()`) is delegated
/// directly to that field's searcher. A field-less query fans out to every
/// field whose configured dimension matches the query vector's, merging
/// results by distance (ascending) when every candidate field shares the
/// same metric, or by similarity (descending) when metrics differ.
///
/// A document present in several candidate fields collapses to a single
/// hit, keeping its best-scoring field (Issue #1343) — so `top_k` counts
/// documents, not `(document, field)` pairs. [`Self::count`] is unaffected
/// and stays per-vector: a document in N fields still counts as N.
#[derive(Debug)]
struct MultiFieldFanoutSearcher {
    fields: BTreeMap<String, SearcherEntry>,
}

impl MultiFieldFanoutSearcher {
    fn new(fields: BTreeMap<String, SearcherEntry>) -> Self {
        Self { fields }
    }

    fn resolve_field(&self, field_name: &str) -> Result<&SearcherEntry> {
        self.fields.get(field_name).ok_or_else(|| {
            LaurusError::invalid_argument(format!(
                "unknown vector field '{field_name}': no index configured for it"
            ))
        })
    }

    /// Searchable fields whose configured dimension matches `dim`, in
    /// deterministic (field-name-sorted) order. Multi-vector fields are
    /// never candidates.
    fn candidates_for_dimension(&self, dim: usize) -> Vec<&SearcherEntry> {
        self.fields
            .values()
            .filter(|e| e.dimension == dim && !e.multi_vector)
            .collect()
    }

    fn search_fanout(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        let dim = request.query.data.len();
        let candidates = self.candidates_for_dimension(dim);

        if candidates.is_empty() {
            return Ok(VectorIndexQueryResults::new());
        }
        if candidates.len() == 1 {
            return candidates[0].searcher.search(request);
        }

        let homogeneous_metric = candidates
            .windows(2)
            .all(|w| w[0].distance_metric == w[1].distance_metric);

        let mut merged = VectorIndexQueryResults::new();
        let mut candidates_examined = 0usize;
        let mut search_time_ms = 0f64;
        for entry in &candidates {
            let r = entry.searcher.search(request)?;
            candidates_examined += r.candidates_examined;
            search_time_ms += r.search_time_ms;
            merged.results.extend(r.results);
            merged.query_metadata.extend(r.query_metadata);
        }
        merged.candidates_examined = candidates_examined;
        merged.search_time_ms = search_time_ms;

        if homogeneous_metric {
            merged.sort_by_distance();
        } else {
            merged.sort_by_similarity();
        }
        // Issue #1343: a document present in several candidate fields must
        // count once, not once per field, otherwise `top_k` is cut at
        // (doc_id, field) granularity and the same document can surface
        // twice. The sort above already orders each doc_id's hits
        // best-first — ties break by field name, since `candidates` walks
        // the fields `BTreeMap` in that order and `sort_by` is stable — so
        // keeping the first occurrence per doc_id keeps its best-scoring
        // field.
        let mut seen_doc_ids = std::collections::HashSet::with_capacity(merged.results.len());
        merged.results.retain(|r| seen_doc_ids.insert(r.doc_id));
        merged.take_top_k(request.params.top_k);
        Ok(merged)
    }
}

impl VectorIndexSearcher for MultiFieldFanoutSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        match &request.field_name {
            Some(field_name) => self.resolve_field(field_name)?.searcher.search(request),
            None => self.search_fanout(request),
        }
    }

    fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        if let Some(field_name) = request.field_name.clone() {
            return self.resolve_field(&field_name)?.searcher.count(request);
        }
        let dim = request.query.data.len();
        let mut total = 0u64;
        for entry in self.candidates_for_dimension(dim) {
            total += entry.searcher.count(request.clone())?;
        }
        Ok(total)
    }

    fn warmup(&mut self) -> Result<()> {
        for entry in self.fields.values_mut() {
            entry.searcher.warmup()?;
        }
        Ok(())
    }

    fn parallel_threshold(&self) -> usize {
        self.fields
            .values()
            .map(|e| e.searcher.parallel_threshold())
            .min()
            .unwrap_or(4)
    }

    fn search_batch_with_threshold(
        &self,
        queries: &[VectorIndexQuery],
        parallel_threshold: usize,
    ) -> Result<Vec<VectorIndexQueryResults>> {
        // Bucket query INDICES by field name (`None` = field-less fan-out),
        // dispatch each field-targeted bucket as one `search_batch_with_threshold`
        // call (preserving per-field batching amortization), then reassemble
        // in the ORIGINAL query order. Order preservation is required:
        // `VectorStore::search_impl` zips this return value against a
        // same-order `query_weights` vector and would silently misattribute
        // scores to the wrong query otherwise.
        let mut buckets: BTreeMap<Option<String>, Vec<usize>> = BTreeMap::new();
        for (i, q) in queries.iter().enumerate() {
            buckets.entry(q.field_name.clone()).or_default().push(i);
        }

        let mut results: Vec<Option<VectorIndexQueryResults>> =
            (0..queries.len()).map(|_| None).collect();

        for (field_name, indices) in buckets {
            match field_name {
                Some(field_name) => {
                    let entry = self.resolve_field(&field_name)?;
                    let bucket_queries: Vec<VectorIndexQuery> =
                        indices.iter().map(|&i| queries[i].clone()).collect();
                    let bucket_results = entry
                        .searcher
                        .search_batch_with_threshold(&bucket_queries, parallel_threshold)?;
                    for (i, r) in indices.into_iter().zip(bucket_results) {
                        results[i] = Some(r);
                    }
                }
                None => {
                    for i in indices {
                        results[i] = Some(self.search_fanout(&queries[i])?);
                    }
                }
            }
        }

        Ok(results
            .into_iter()
            .map(|r| r.expect("every query index is populated by exactly one bucket above"))
            .collect())
    }
}

/// Reader over a [`MultiFieldVectorIndex`]'s per-field sub-readers.
#[derive(Debug)]
struct MultiFieldReaderFacade {
    readers: BTreeMap<String, Arc<dyn VectorIndexReader>>,
}

impl MultiFieldReaderFacade {
    fn new(readers: BTreeMap<String, Arc<dyn VectorIndexReader>>) -> Self {
        Self { readers }
    }
}

impl VectorIndexReader for MultiFieldReaderFacade {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        match self.readers.get(field_name) {
            Some(r) => r.get_vector(doc_id, field_name),
            None => Ok(None),
        }
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut out = Vec::new();
        for r in self.readers.values() {
            out.extend(r.get_vectors_for_doc(doc_id)?);
        }
        Ok(out)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        doc_ids
            .iter()
            .map(|(doc_id, field_name)| self.get_vector(*doc_id, field_name))
            .collect()
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        let mut out = Vec::new();
        for r in self.readers.values() {
            out.extend(r.vector_ids()?);
        }
        Ok(out)
    }

    fn doc_ids_for_field(&self, field_name: &str) -> Arc<[u64]> {
        match self.readers.get(field_name) {
            Some(r) => r.doc_ids_for_field(field_name),
            None => Vec::new().into(),
        }
    }

    fn vector_count(&self) -> usize {
        self.readers.values().map(|r| r.vector_count()).sum()
    }

    fn dimension(&self) -> usize {
        // No single dimension generalizes across heterogeneous fields.
        // This is a legacy single-index accessor kept only for trait-object
        // callers that assume one homogeneous index; callers that need a
        // specific field's dimension should use
        // `MultiFieldVectorIndex::field_dimensions` instead. Falls back to
        // the first field in (deterministic, sorted) iteration order.
        self.readers
            .values()
            .next()
            .map(|r| r.dimension())
            .unwrap_or(0)
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.readers
            .values()
            .next()
            .map(|r| r.distance_metric())
            .unwrap_or(DistanceMetric::Cosine)
    }

    fn stats(&self) -> VectorStats {
        let mut vector_count = 0;
        let mut memory_usage = 0;
        let mut build_time_ms = 0;
        for r in self.readers.values() {
            let s = r.stats();
            vector_count += s.vector_count;
            memory_usage += s.memory_usage;
            build_time_ms = build_time_ms.max(s.build_time_ms);
        }
        VectorStats {
            vector_count,
            dimension: self.dimension(),
            memory_usage,
            build_time_ms,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.readers
            .get(field_name)
            .map(|r| r.contains_vector(doc_id, field_name))
            .unwrap_or(false)
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        let mut out = Vec::new();
        for r in self.readers.values() {
            out.extend(r.get_vector_range(start_doc_id, end_doc_id)?);
        }
        Ok(out)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        match self.readers.get(field_name) {
            Some(r) => r.get_vectors_by_field(field_name),
            None => Ok(Vec::new()),
        }
    }

    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self.readers.keys().cloned().collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        let mut all = Vec::new();
        for r in self.readers.values() {
            let mut it = r.vector_iterator()?;
            while let Some(item) = it.next()? {
                all.push(item);
            }
        }
        Ok(Box::new(SimpleVectorIterator::new(all)))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "MultiField".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1.0".to_string(),
            build_config: serde_json::json!({
                "fields": self.readers.keys().cloned().collect::<Vec<_>>(),
            }),
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut repair_suggestions = Vec::new();
        for (name, r) in &self.readers {
            let report = r.validate()?;
            errors.extend(report.errors.into_iter().map(|e| format!("[{name}] {e}")));
            warnings.extend(report.warnings.into_iter().map(|w| format!("[{name}] {w}")));
            repair_suggestions.extend(report.repair_suggestions);
        }
        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            repair_suggestions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A [`VectorIndexWriter`] whose `commit()` always fails, for
    /// exercising [`MultiFieldWriter::commit`]'s "always attempt every
    /// field, never poison" contract without needing a real storage fault.
    #[derive(Debug)]
    struct FailingWriter {
        commit_attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl VectorIndexWriter for FailingWriter {
        fn next_vector_id(&self) -> u64 {
            0
        }
        fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
            self.add_vectors(vectors)
        }
        fn add_vectors(&mut self, _vectors: Vec<(u64, String, Vector)>) -> Result<()> {
            Ok(())
        }
        fn finalize(&mut self) -> Result<()> {
            Ok(())
        }
        fn progress(&self) -> f32 {
            1.0
        }
        fn estimated_memory_usage(&self) -> usize {
            0
        }
        fn vectors(&self) -> &[(u64, String, Vector)] {
            &[]
        }
        fn write(&self) -> Result<()> {
            Ok(())
        }
        fn has_storage(&self) -> bool {
            true
        }
        fn delete_document(&mut self, _doc_id: u64) -> Result<()> {
            Ok(())
        }
        fn commit(&mut self) -> Result<()> {
            self.commit_attempts.fetch_add(1, Ordering::SeqCst);
            Err(LaurusError::internal("simulated commit failure"))
        }
        fn rollback(&mut self) -> Result<()> {
            Ok(())
        }
        fn pending_docs(&self) -> u64 {
            0
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn build_reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
            Ok(Arc::new(MultiFieldReaderFacade::new(BTreeMap::new())))
        }
    }

    /// A [`VectorIndexWriter`] that always succeeds and counts `commit()`
    /// calls.
    #[derive(Debug)]
    struct SucceedingWriter {
        commit_attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl VectorIndexWriter for SucceedingWriter {
        fn next_vector_id(&self) -> u64 {
            0
        }
        fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
            self.add_vectors(vectors)
        }
        fn add_vectors(&mut self, _vectors: Vec<(u64, String, Vector)>) -> Result<()> {
            Ok(())
        }
        fn finalize(&mut self) -> Result<()> {
            Ok(())
        }
        fn progress(&self) -> f32 {
            1.0
        }
        fn estimated_memory_usage(&self) -> usize {
            0
        }
        fn vectors(&self) -> &[(u64, String, Vector)] {
            &[]
        }
        fn write(&self) -> Result<()> {
            Ok(())
        }
        fn has_storage(&self) -> bool {
            true
        }
        fn delete_document(&mut self, _doc_id: u64) -> Result<()> {
            Ok(())
        }
        fn commit(&mut self) -> Result<()> {
            self.commit_attempts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn rollback(&mut self) -> Result<()> {
            Ok(())
        }
        fn pending_docs(&self) -> u64 {
            0
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn build_reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
            Ok(Arc::new(MultiFieldReaderFacade::new(BTreeMap::new())))
        }
    }

    /// Issue #948: `commit()` must always attempt EVERY field (never
    /// short-circuit on the first failure), report the failure to the
    /// caller, and -- critically -- must NOT poison the writer against
    /// further mutation. A caller that keeps buffering new documents on a
    /// writer whose commit just failed and retries later (exactly what
    /// `VectorStore::commit`'s failure-path writer retention does) must
    /// keep working, the same way a single-field writer already does.
    #[test]
    fn commit_always_attempts_every_field_and_never_poisons() {
        let good_commits = Arc::new(AtomicUsize::new(0));
        let bad_commits = Arc::new(AtomicUsize::new(0));

        let mut writers: BTreeMap<String, Box<dyn VectorIndexWriter>> = BTreeMap::new();
        writers.insert(
            "good".to_string(),
            Box::new(SucceedingWriter {
                commit_attempts: good_commits.clone(),
            }),
        );
        writers.insert(
            "bad".to_string(),
            Box::new(FailingWriter {
                commit_attempts: bad_commits.clone(),
            }),
        );

        let mut writer = MultiFieldWriter::new(writers);
        let err = writer.commit().unwrap_err();
        assert!(format!("{err:?}").contains("simulated commit failure"));

        // BOTH fields were attempted, not just the one that happened to
        // fail (nor short-circuited after it).
        assert_eq!(good_commits.load(Ordering::SeqCst), 1);
        assert_eq!(bad_commits.load(Ordering::SeqCst), 1);

        // The writer must still accept further mutation after the failed
        // commit -- no poisoning -- and a retry must attempt every field
        // again.
        writer
            .add_vectors(vec![(1, "good".to_string(), Vector::new(vec![0.0]))])
            .unwrap();
        writer.commit().unwrap_err();
        assert_eq!(good_commits.load(Ordering::SeqCst), 2);
        assert_eq!(bad_commits.load(Ordering::SeqCst), 2);
    }
}
