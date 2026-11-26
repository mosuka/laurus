//! Per-segment view of an [`InvertedIndexReader`] for the per-segment
//! fanout search path (#476 Phase 1).
//!
//! [`PerSegmentReaderView`] adapts a [`SegmentReader`] reference into
//! a [`LexicalIndexReader`] implementation. It forwards posting / field
//! / document access to the underlying segment, but **injects
//! cross-segment-aggregated** `doc_count`, `max_doc`, and `term_info`
//! `doc_freq` / `total_freq` so the BM25 scorer's IDF reflects the
//! global term rarity rather than the segment-local one.
//!
//! ## Cross-segment scoring semantics
//!
//! | Field | Source | Reason |
//! | ----- | ------ | ------ |
//! | `doc_freq`, `total_freq` (IDF) | global cross-segment aggregate | consistent term-rarity ranking across segments |
//! | `total_docs` (IDF) | global | same |
//! | `avg_field_length` (BM25 TF normaliser) | per-segment | matches the avg the segment's `block_max` factors were anchored against |
//! | `block_max` table | per-segment | computed at write time against the segment's avg, valid as a per-segment scoring bound |
//!
//! Per-segment scoring with cross-segment IDF mirrors how Lucene /
//! Elasticsearch score across shards. Same-doc-id micro-score
//! divergence between segments is accepted: top-K ranking from each
//! segment, then merge.
//!
//! This view exists for *tightness* (keeping `block_max` usable for
//! Block-Max-WAND pruning), not *soundness*: `InvertedIndexReader::term_info`
//! (the cross-segment aggregate this view deliberately avoids for
//! `max_score_factor`/`block_max`) already drops those fields to a safe
//! `0.0`/empty whenever it cannot prove they are a valid bound against the
//! `avg_field_length` a caller would actually score with (#1120) — so a
//! caller using the aggregate reader directly is never unsound, only
//! potentially looser than this per-segment view.
//!
//! ## Why this matters for #476
//!
//! `InvertedIndexReader::term_info`'s cross-segment aggregation
//! returns an empty `block_max` `Vec` when more than one segment
//! matches (PR-D's safety fallback against under-bounded
//! `block_max_score_at`). With this view the per-segment block_max
//! flows through unchanged because the BM25 scorer is built with
//! per-segment avg, restoring PR-F's BMW pivot loop on each segment.

use std::any::Any;
use std::sync::{Arc, RwLock};

use roaring::RoaringTreemap;

use crate::error::Result;
use crate::lexical::core::document::Document;
use crate::lexical::index::inverted::core::automaton::Automaton;
use crate::lexical::index::inverted::reader::{SegmentFieldLengths, SegmentReader};
use crate::lexical::index::inverted::segment::Membership;
use crate::lexical::index::structures::bkd_tree::BKDTree;
use crate::lexical::query::Query;
use crate::lexical::reader::{
    FieldLengths, FieldStats, LexicalIndexReader, PostingIterator, ReaderTermInfo,
};

/// Cross-segment term-info lookup function. Returns the **global**
/// `doc_freq` and `total_freq` for a `(field, term)` pair so per-segment
/// BM25 scorers see the same IDF every segment uses. Returning `None`
/// signals that no segment has the term — the per-segment view will
/// then also report `None` from its `term_info`.
type GlobalTermInfoFn = dyn Fn(&str, &str) -> Result<Option<ReaderTermInfo>> + Send + Sync;

/// Cross-segment matching-doc-ids lookup function (#764). Resolves a query's
/// matched document set against the **cross-segment** reader's snapshot-scoped
/// query/filter cache, so a filter clause evaluated under this per-segment
/// fanout view reuses the same cache as the top-level reader. The set is that
/// of the **whole index**; the view keeps its segment's share of it (Issue
/// #1743). `None` when the reader keeps no cache, so that each segment matches
/// the query itself instead of draining the whole index.
type GlobalMatchingDocIdsFn =
    dyn Fn(&dyn Query) -> Result<Option<Arc<RoaringTreemap>>> + Send + Sync;

/// Adapter that exposes a single [`SegmentReader`] as a
/// [`LexicalIndexReader`] backed by per-segment posting / field data
/// and cross-segment-global IDF inputs.
///
/// Owns the segment via `Arc<RwLock<SegmentReader>>` and acquires a
/// read lock per method call. This keeps the type `'static` so it
/// can satisfy `LexicalIndexReader`'s `as_any` requirement, and the
/// per-method lock acquisition is not a measurable cost in the
/// fanout path because each rayon worker owns a distinct segment.
pub(crate) struct PerSegmentReaderView {
    segment: Arc<RwLock<SegmentReader>>,
    global_doc_count: u64,
    global_max_doc: u64,
    global_term_info_fn: Arc<GlobalTermInfoFn>,
    global_matching_doc_ids_fn: Arc<GlobalMatchingDocIdsFn>,
}

impl PerSegmentReaderView {
    /// Create a view over `segment` that reports the supplied global
    /// `doc_count` / `max_doc`, looks up cross-segment term-info via
    /// `global_term_info_fn`, and resolves cached filter doc-id sets via
    /// `global_matching_doc_ids_fn` (#764).
    pub fn new(
        segment: Arc<RwLock<SegmentReader>>,
        global_doc_count: u64,
        global_max_doc: u64,
        global_term_info_fn: Arc<GlobalTermInfoFn>,
        global_matching_doc_ids_fn: Arc<GlobalMatchingDocIdsFn>,
    ) -> Self {
        PerSegmentReaderView {
            segment,
            global_doc_count,
            global_max_doc,
            global_term_info_fn,
            global_matching_doc_ids_fn,
        }
    }

    /// This segment's share of the cached doc-id set for `query` (#764): the
    /// documents of the whole index's set that this segment holds live.
    ///
    /// The cross-segment reader's cache holds the set of the whole index. A
    /// view must not return it as it is (Issue #1743): the fanout merges the
    /// hits of every segment, so each would return the others' documents too,
    /// and a `MustNot` clause would exclude only the segment's own. The share
    /// is the segment's live documents as
    /// [`SegmentReader::live_doc_ids`] defines them (Issue #1211), so an id
    /// this segment holds deleted is left out even when another segment holds
    /// it live. It costs what the smaller of the set and the segment takes to
    /// read.
    ///
    /// # Arguments
    ///
    /// * `query` - The (cacheable) filter query whose doc-id set is requested.
    ///
    /// # Returns
    ///
    /// This segment's documents of the set, or `None` when the reader keeps
    /// no cache or the segment records no doc ids: the caller then matches
    /// the query on this view instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the set cannot be computed or the segment's live
    /// documents cannot be read.
    pub fn matching_doc_ids(&self, query: &dyn Query) -> Result<Option<Arc<RoaringTreemap>>> {
        // A segment of unknown membership tells its documents from its stored
        // documents, which its own matcher does not need to read.
        if matches!(
            self.segment.read().unwrap().membership(),
            Membership::Unknown
        ) {
            return Ok(None);
        }
        // Not under this segment's lock: computing the set reads every segment.
        let Some(docs) = (self.global_matching_doc_ids_fn)(query)? else {
            return Ok(None);
        };
        let share = self.segment.read().unwrap().live_docs_among(&docs)?;
        Ok(Some(Arc::new(share)))
    }
}

impl std::fmt::Debug for PerSegmentReaderView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerSegmentReaderView")
            .field("global_doc_count", &self.global_doc_count)
            .field("global_max_doc", &self.global_max_doc)
            .finish_non_exhaustive()
    }
}

impl LexicalIndexReader for PerSegmentReaderView {
    fn doc_count(&self) -> u64 {
        // Global cross-segment count so IDF sees the right
        // `total_docs`.
        self.global_doc_count
    }

    fn max_doc(&self) -> u64 {
        self.global_max_doc
    }

    fn is_deleted(&self, doc_id: u64) -> bool {
        // SegmentReader::is_deleted returns Result<bool>; on error or
        // when the bitmap is not loadable we conservatively report
        // `false` (the doc proceeds through scoring) — matching the
        // top-level reader's tolerance for transient I/O issues.
        let seg = self.segment.read().unwrap();
        seg.is_deleted(doc_id).unwrap_or(false)
    }

    fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        let seg = self.segment.read().unwrap();
        seg.document(doc_id)
    }

    fn document_fields(
        &self,
        doc_id: u64,
        field_names: &[&str],
    ) -> Result<Option<std::collections::HashMap<String, crate::data::DataValue>>> {
        // Without this override, the trait default forwards to `document`
        // above and clones every field just to filter it back down --
        // defeating the whole point of `document_fields` (Issue #1047: a
        // DocValues-miss fallback on this per-segment view would otherwise
        // clone the full document per hit).
        let seg = self.segment.read().unwrap();
        seg.document_fields(doc_id, field_names)
    }

    fn doc_ids(&self) -> Result<Vec<u64>> {
        // Segment-local ids (global id values, but only those present in
        // this segment) so stored-document scans stay segment-bounded
        // under the fanout instead of probing `0..global_max_doc` (#994).
        let seg = self.segment.read().unwrap();
        seg.doc_ids()
    }

    fn live_doc_ids(&self) -> Result<Vec<u64>> {
        // This segment's live documents only (Issue #1211). The trait
        // default would probe `scan_doc_ids`, which falls back to
        // `0..global_max_doc` when this segment lists no ids — an empty
        // segment then contributed phantom ids to a MustNot-only query.
        let seg = self.segment.read().unwrap();
        Ok(seg.live_doc_ids()?.iter().collect())
    }

    fn get_doc_value(
        &self,
        field: &str,
        doc_id: u64,
    ) -> Result<Option<crate::lexical::core::field::FieldValue>> {
        // Segment-local DocValues so per-segment field-sorted collection
        // works under the fanout (#944); the trait default returned
        // `None`, which would sort every document as `Null`.
        let seg = self.segment.read().unwrap();
        seg.get_doc_value(field, doc_id)
    }

    fn has_doc_values(&self, field: &str) -> bool {
        let seg = self.segment.read().unwrap();
        seg.has_doc_values(field)
    }

    fn term_info_is_authoritative(&self) -> bool {
        // This view answers `term_info` from its own segment's dictionary
        // (Issue #1196); without one, the segment's documents are only
        // reachable through the stored-document scan.
        let seg = self.segment.read().unwrap();
        seg.has_term_dictionary()
    }

    fn term_info(&self, field: &str, term: &str) -> Result<Option<ReaderTermInfo>> {
        // Combine: global doc_freq / total_freq + per-segment posting
        // offset, max_score_factor, and (critically) the per-segment
        // `block_max` slice so PR-F's BMW pivot loop fires on the
        // per-segment view.
        let local_opt = {
            let seg = self.segment.read().unwrap();
            seg.term_info(field, term)?
        };
        let local = match local_opt {
            Some(li) => li,
            None => return Ok(None),
        };
        let global = (self.global_term_info_fn)(field, term)?;
        let global = match global {
            Some(gi) => gi,
            None => {
                // The segment claims the term but the cross-segment
                // aggregate does not — defensive: fall back to local
                // counts so scoring still produces a result.
                return Ok(Some(ReaderTermInfo {
                    field: field.to_string(),
                    term: term.to_string(),
                    doc_freq: local.doc_frequency,
                    total_freq: local.total_frequency,
                    posting_offset: local.posting_offset,
                    posting_size: local.posting_length,
                    max_score_factor: local.max_score_factor,
                    block_max: local.block_max,
                }));
            }
        };
        Ok(Some(ReaderTermInfo {
            field: field.to_string(),
            term: term.to_string(),
            doc_freq: global.doc_freq,
            total_freq: global.total_freq,
            posting_offset: local.posting_offset,
            posting_size: local.posting_length,
            max_score_factor: local.max_score_factor,
            block_max: local.block_max,
        }))
    }

    fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
        let seg = self.segment.read().unwrap();
        seg.postings(field, term)
    }

    fn docs_of_matching_terms(
        &self,
        field: &str,
        automaton: &dyn Automaton,
    ) -> Result<Option<RoaringTreemap>> {
        // This segment's documents only: the fanout merges the hits of each
        // segment, so another segment's would be returned twice.
        let seg = self.segment.read().unwrap();
        Ok(Some(seg.docs_of_matching_terms(field, automaton)?))
    }

    fn matching_terms(
        &self,
        field: &str,
        automaton: &dyn Automaton,
    ) -> Result<Option<Vec<(String, u64)>>> {
        let seg = self.segment.read().unwrap();
        Ok(Some(seg.matching_terms(field, automaton)?))
    }

    fn field_stats(&self, field: &str) -> Result<Option<FieldStats>> {
        // Per-segment avg so the BM25 normaliser matches the avg
        // each block_max factor was anchored against (#403 PR-D
        // would otherwise under-bound).
        let seg = self.segment.read().unwrap();
        seg.field_stats(field)
    }

    fn field_lengths(&self, field: &str) -> Option<Arc<dyn FieldLengths>> {
        Some(Arc::new(SegmentFieldLengths::new(
            field,
            [(0, u64::MAX, &self.segment)],
        )))
    }

    fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        // Override the trait default (which returns `Ok(None)`) so that
        // BKD-backed queries (geo / geo3d / numeric range) still find
        // hits when the per-segment fanout (#480) routes the query
        // through this view. Without this override the fanout path
        // silently returns 0 hits for every BKD-backed query as soon
        // as the index accumulates two or more segments.
        //
        // Wraps the segment's raw BKD tree in this segment's deletion
        // filter so soft-deleted docs within the segment do not
        // resurface (#400 invariant — must hold per-segment in the
        // fanout path as well).
        let seg = self.segment.read().unwrap();
        seg.get_filtered_bkd_tree(field)
    }

    fn close(&mut self) -> Result<()> {
        // The view does not own the segment — closing is a no-op.
        Ok(())
    }

    fn is_closed(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
