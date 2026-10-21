//! Reproduction for #1017: whether a flushed-but-uncommitted segment is
//! visible to `search()` is currently accidental.
//!
//! `InvertedIndex::load_segments` discovers segments by listing storage for
//! `segment_*.meta`, consulting no commit marker, while `flush_segment` fires
//! automatically at `max_buffered_docs` and writes that `.meta` on the spot.
//! So whether a user sees uncommitted documents depends only on whether the
//! searcher cache happened to be cold — and the repository documents the
//! opposite in roughly twenty-five places ("documents become searchable only
//! after `commit()`").
//!
//! It is worse than early visibility. `segment_info_for` hard-codes
//! `has_deletions: false`, and deletions have been group-committed since
//! #875, so the flag and the `.delmap` only reach storage at commit.
//! `SegmentReader` short-circuits on that flag in `load_deletion_bitmap`,
//! `is_deleted` and `filter_deleted_soa` alike — meaning a searcher that can
//! see such a segment cannot filter its deletions or its superseded upsert
//! versions at all.
//!
//! These landed as `#[ignore]`d RED cases in #1020 and are un-ignored by the
//! fix, so they now stand as the regression gate for the contract.

use std::sync::Arc;

use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, Engine, FieldOption, Schema};
use laurus::{LexicalSearchQuery, SearchRequestBuilder};

/// `InvertedIndexWriterConfig::default().max_buffered_docs`, the point at
/// which the writer flushes a segment on its own.
const MAX_BUFFERED_DOCS: usize = 10_000;

/// Documents written past the flush boundary, so the flush is comfortably
/// behind us while some documents remain buffered.
const TOTAL_DOCS: usize = MAX_BUFFERED_DOCS + 50;

/// A schema with a single `body` text field.
fn body_schema() -> Schema {
    Schema::builder()
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build()
}

/// Build a `(id, doc)` entry. Document 0 gets a distinctive term so a query
/// can single it out.
fn entry(i: usize) -> (String, Document) {
    let body = if i == 0 { "zebra" } else { "alpha" };
    let doc = Document::builder()
        .add_field("body", DataValue::Text(body.into()))
        .build();
    (format!("id{i:06}"), doc)
}

/// Write past the automatic flush boundary **without committing**, and prove
/// the flush actually fired.
///
/// `add_documents` is deliberate: `Engine::index_internal` skips the
/// lookup of previous versions for chunked adds, so nothing warms the
/// searcher cache during ingest. That leaves the cache cold, which is the
/// state the following tests vary.
async fn engine_past_auto_flush() -> laurus::Result<(Engine, Arc<dyn Storage>)> {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let engine = Engine::new(storage.clone(), body_schema()).await?;

    engine
        .add_documents((0..TOTAL_DOCS).map(entry).collect())
        .await?;

    // Without this the tests could pass by failing to set up their own
    // premise: no flush, no flushed segment, nothing to be visible.
    // The evidence is a data file (#1024: `.meta` files no longer exist).
    let flushed = storage
        .list_files()?
        .iter()
        .any(|f| f.contains("segment_") && (f.ends_with(".post") || f.ends_with(".cfs")));
    assert!(
        flushed,
        "automatic flush_segment must fire at max_buffered_docs"
    );

    Ok((engine, storage))
}

/// Search `body` for one term.
async fn search_term(engine: &Engine, term: &str) -> laurus::Result<Vec<laurus::SearchResult>> {
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "body", term,
        ))))
        .limit(10)
        .build();
    engine.search(request).await
}

/// Search for the term carried only by document 0.
async fn search_zebra(engine: &Engine) -> laurus::Result<Vec<laurus::SearchResult>> {
    search_term(engine, "zebra").await
}

/// Search for the term every other document carries.
async fn search_alpha(engine: &Engine) -> laurus::Result<Vec<laurus::SearchResult>> {
    search_term(engine, "alpha").await
}

/// A document written before the automatic flush must not be searchable
/// until commit — with a **cold** searcher cache.
#[tokio::test(flavor = "multi_thread")]
async fn uncommitted_documents_are_invisible_to_search_cold_cache() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    let hits = search_zebra(&engine).await?;
    assert!(
        hits.is_empty(),
        "an uncommitted document must not be searchable, got {} hit(s)",
        hits.len()
    );

    // And after the commit it must be, so the test is not passing because the
    // document was never indexed.
    engine.commit().await?;
    assert_eq!(
        search_zebra(&engine).await?.len(),
        1,
        "the document must be searchable once committed"
    );

    Ok(())
}

/// The same, but with the searcher cache **warmed after the flush**.
///
/// This is the route a real user hits: `get_documents` resolves through
/// `find_doc_ids_by_term`, which runs a search and populates the cache. In an
/// `add_documents` workload nothing warms it during ingest, so the first
/// read-your-writes call after the flush builds a searcher over the flushed
/// segment — and every subsequent search uses it.
///
/// A single-state test would prove nothing here, since the whole defect is
/// that the answer depends on cache state.
#[tokio::test(flavor = "multi_thread")]
async fn uncommitted_documents_are_invisible_to_search_warm_cache() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    // Warm the cache over the flushed segment.
    let resolved = engine.get_documents("id000000").await?;
    assert_eq!(
        resolved.len(),
        1,
        "NRT `_id` resolution must still work before commit (#1016)"
    );

    let hits = search_zebra(&engine).await?;
    assert!(
        hits.is_empty(),
        "an uncommitted document must not be searchable, got {} hit(s)",
        hits.len()
    );

    Ok(())
}

/// A deletion made before commit must not leave the deleted document
/// searchable.
///
/// The delete lands only in the in-memory `DeletionManager`; the segment's
/// `.meta` still says `has_deletions: false` and its `.delmap` does not
/// exist, so a searcher that can see the segment cannot filter it.
#[tokio::test(flavor = "multi_thread")]
async fn uncommitted_deletion_does_not_leave_a_live_hit() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    engine.delete_documents("id000000").await?;

    let hits = search_zebra(&engine).await?;
    assert!(
        hits.is_empty(),
        "a deleted document must not surface as a live hit, got {} hit(s)",
        hits.len()
    );

    engine.commit().await?;
    assert!(
        search_zebra(&engine).await?.is_empty(),
        "and it must stay gone after the commit"
    );

    Ok(())
}

/// An upsert made before commit must not leave both versions searchable.
///
/// This is the sharpest symptom: two hits carrying the same external `_id`,
/// one of them holding superseded field values.
#[tokio::test(flavor = "multi_thread")]
async fn uncommitted_upsert_does_not_duplicate_the_document() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    // Supersede document 0. The replacement gets its own distinctive term so
    // the assertions below can target it exactly, rather than hoping it lands
    // in the top-K of a term ten thousand documents share.
    let replacement = Document::builder()
        .add_field("body", DataValue::Text("quokka".into()))
        .build();
    engine.put_document("id000000", replacement).await?;

    let hits = search_zebra(&engine).await?;
    assert!(
        hits.is_empty(),
        "the superseded version must not remain searchable, got {} hit(s)",
        hits.len()
    );

    engine.commit().await?;

    // The old version is gone and the new one is there, exactly once.
    assert!(
        search_zebra(&engine).await?.is_empty(),
        "the superseded version must not come back at commit"
    );
    let live = search_term(&engine, "quokka").await?;
    assert_eq!(
        live.len(),
        1,
        "exactly one live copy of an `_id` may exist, found {}",
        live.len()
    );
    assert_eq!(live[0].id, "id000000");

    Ok(())
}

/// `count()` and filter-only queries take different code paths from `search`
/// — `count` has an O(1) `term_doc_freq` shortcut with no documents to
/// inspect — so the contract has to hold for them independently.
#[tokio::test(flavor = "multi_thread")]
async fn uncommitted_documents_are_invisible_to_count() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    // `total_hits` is produced inside the collector, not derived from the
    // returned hits, so it is a genuinely separate assertion.
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "body", "alpha",
        ))))
        .limit(10)
        .build();
    let hits = engine.search(request).await?;
    assert!(
        hits.is_empty(),
        "uncommitted documents must not be counted either, got {} hit(s)",
        hits.len()
    );

    engine.commit().await?;
    assert!(
        !search_alpha(&engine).await?.is_empty(),
        "committed documents must be searchable"
    );

    Ok(())
}
