//! Gates for the segment-per-commit HNSW index (#634 PR-3 / #881).
//!
//! The deterministic core gate: committing 1 new document on an N-document
//! base writes O(delta) bytes — a new small segment file plus the manifest —
//! never a rewrite of the existing segments. Plus: config-OFF invariance
//! (the monolithic layout stays byte-identical), legacy-index rejection
//! (zero-copy migration is #882), multi-segment recall parity, and
//! upsert/delete semantics across commits.

mod common;

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

fn doc_vec(i: u64) -> Vector {
    let mut v = vec![0.0f32; 16];
    let t = i as f32 * 0.001;
    v[0] = t.cos();
    v[1] = t.sin();
    v[2] = (t * 2.0).cos();
    v[3] = (t * 3.0).sin();
    Vector::new(v)
}

fn config(segmented: bool) -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: 16,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        segmented,
        ..Default::default()
    }
}

fn storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

/// Add `ids` to a fresh writer and commit (one sealed segment).
fn commit_batch(index: &dyn VectorIndex, ids: std::ops::Range<u64>) {
    let mut writer = index.writer().unwrap();
    let vectors: Vec<_> = ids.map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.commit().unwrap();
}

fn query(id: u64, top_k: usize) -> VectorIndexQuery {
    VectorIndexQuery {
        query: doc_vec(id),
        params: VectorIndexQueryParams {
            top_k,
            ..Default::default()
        },
        field_name: Some("v".to_string()),
        filter: None,
    }
}

/// #881 core gate: a 1-doc commit on a 1000-doc base writes a new segment
/// that is a tiny fraction of the base segment, and never rewrites the base.
#[test]
fn one_doc_commit_writes_o_delta_bytes() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    commit_batch(&index, 0..1000);
    let base_file = "segment_000000.hnsw";
    let base_size = storage.file_size(base_file).unwrap();

    commit_batch(&index, 1000..1001);
    let delta_file = "segment_000001.hnsw";
    let delta_size = storage.file_size(delta_file).unwrap();

    assert!(
        delta_size * 20 < base_size,
        "a 1-doc commit must write O(delta) bytes, got delta={delta_size} vs base={base_size}"
    );
    assert_eq!(
        storage.file_size(base_file).unwrap(),
        base_size,
        "the base segment must never be rewritten by a later commit (#634)"
    );

    // Exactly the two sealed segments exist — no other .hnsw files.
    let hnsw_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".hnsw"))
        .collect();
    assert_eq!(
        hnsw_files.len(),
        2,
        "exactly one new segment per non-empty commit, got {hnsw_files:?}"
    );
}

/// #881: with `segmented: false` (the default) the factory path stays
/// monolithic — no manifest, single `.hnsw` file.
#[test]
fn config_off_keeps_monolithic_layout() {
    let storage = storage();
    let index = HnswIndex::create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(false),
    )
    .unwrap();

    commit_batch(&index, 0..100);
    assert!(
        !storage.file_exists("segments.json"),
        "config OFF must not create a segment manifest"
    );
    assert!(storage.file_exists("vector_index.hnsw"));
}

/// #882: a legacy monolithic index is migrated ZERO-COPY on first segmented
/// open — the existing `.hnsw` becomes segment 0 verbatim (no data
/// movement), search results are identical, and later commits append new
/// segments without ever touching the legacy file.
#[test]
fn legacy_monolithic_index_migrates_zero_copy() {
    let storage = storage();
    {
        let index = HnswIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..100);
    }
    let legacy_size = storage.file_size("vector_index.hnsw").unwrap();

    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    // Migration wrote only the manifest; the legacy file is untouched.
    assert!(storage.file_exists("segments.json"), "manifest created");
    assert_eq!(
        storage.file_size("vector_index.hnsw").unwrap(),
        legacy_size,
        "zero-copy: the legacy file must not be rewritten (#882)"
    );

    // The migrated data is searchable...
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(42, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 42);

    // ...and a later commit appends a NEW segment, still leaving the legacy
    // file untouched.
    commit_batch(&index, 100..101);
    assert_eq!(storage.file_size("vector_index.hnsw").unwrap(), legacy_size);
    let searcher = index.searcher().unwrap();
    assert_eq!(
        searcher.search(&query(100, 1)).unwrap().results[0].doc_id,
        100
    );
    assert_eq!(
        searcher.search(&query(42, 1)).unwrap().results[0].doc_id,
        42
    );

    // Re-open: the migration must not re-run (idempotent — the manifest
    // already exists) and everything stays searchable.
    drop(index);
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    let searcher = index.searcher().unwrap();
    assert_eq!(
        searcher.search(&query(42, 1)).unwrap().results[0].doc_id,
        42
    );
    assert_eq!(
        searcher.search(&query(100, 1)).unwrap().results[0].doc_id,
        100
    );
}

/// #882: the WAL checkpoint is published only by `persist_deletions` (the
/// end of the store's commit ladder) — never by intermediate manifest saves
/// — so recovery can only skip records whose effects are already durable.
#[test]
fn wal_checkpoint_publishes_only_after_persist_deletions() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    index.set_last_wal_seq(7).unwrap();
    assert_eq!(
        index.last_wal_seq(),
        0,
        "a pending seq must not be visible before persist_deletions (#882)"
    );

    // Sealing a segment saves the manifest, but must NOT publish the
    // pending checkpoint (the deletion bitmap for seq<=7 may not be
    // durable yet).
    commit_batch(&index, 0..5);
    assert_eq!(
        index.last_wal_seq(),
        0,
        "sealing must not publish the pending checkpoint (#882)"
    );

    // persist_deletions publishes and persists it.
    index.persist_deletions().unwrap();
    assert_eq!(index.last_wal_seq(), 7);
}

/// #881: self-recall@10 of a 5-segment index matches a monolithic build of
/// the same corpus within tolerance (#872-style gate).
#[test]
fn multi_segment_self_recall_matches_monolithic() {
    let n = 2000u64;
    let per_commit = 400u64;

    let seg_storage = storage();
    let seg_index = SegmentedHnswIndex::open_or_create(
        seg_storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    let mut lo = 0u64;
    while lo < n {
        commit_batch(&seg_index, lo..(lo + per_commit).min(n));
        lo += per_commit;
    }

    let mono_storage = storage();
    let mono_index = HnswIndex::create(
        mono_storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(false),
    )
    .unwrap();
    commit_batch(&mono_index, 0..n);

    let recall = |index: &dyn VectorIndex| -> f32 {
        let searcher = index.searcher().unwrap();
        let mut hits = 0u64;
        for id in 0..n {
            let results = searcher.search(&query(id, 10)).unwrap();
            if results.results.iter().any(|r| r.doc_id == id) {
                hits += 1;
            }
        }
        hits as f32 / n as f32
    };

    let mono = recall(&mono_index);
    let seg = recall(&seg_index);
    assert!(mono > 0.9, "monolithic self-recall sanity, got {mono:.4}");
    assert!(
        seg >= mono - 0.05,
        "multi-segment self-recall ({seg:.4}) must match the monolithic build \
         ({mono:.4}) within tolerance (#881)"
    );
}

/// #881: a same-id upsert across commits resolves to the newest copy exactly
/// once, and a soft delete of a sealed doc is search-invisible and physically
/// reclaimed by optimize().
#[test]
fn upsert_and_soft_delete_across_commits() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    // Commit 1: docs 1..50 (doc 1's OLD embedding = doc_vec(1)).
    commit_batch(&index, 1..50);

    // Commit 2: upsert doc 1 to doc_vec(9000) via delete-first + re-add.
    {
        let mut writer = index.writer().unwrap();
        writer.delete_document(1).unwrap();
        writer
            .add_vectors(vec![(1, "v".to_string(), doc_vec(9000))])
            .unwrap();
        writer.commit().unwrap();
    }

    // The doc resolves via its NEW embedding...
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 1, "newest copy must win");
    // ...and its stale OLD copy must not outrank the true neighbours of the
    // old embedding.
    let results = searcher.search(&query(1, 1)).unwrap();
    assert_ne!(
        results.results[0].doc_id, 1,
        "the stale copy in the older segment must be masked (#880/#881)"
    );

    // Soft delete a sealed doc: search-invisible immediately.
    index.soft_delete_document(10).unwrap();
    index.persist_deletions().unwrap();
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(10, 5)).unwrap();
    assert!(
        results.results.iter().all(|r| r.doc_id != 10),
        "a soft-deleted sealed doc must be search-invisible"
    );

    // Optimize: one merged segment, deletion physically reclaimed, bitmap
    // cleared.
    index.optimize().unwrap();
    let hnsw_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".hnsw"))
        .collect();
    assert_eq!(
        hnsw_files.len(),
        1,
        "optimize must force-merge to one segment"
    );
    assert!(
        !storage.file_exists("vector_index.delmap"),
        "optimize must clear the persisted deletion bitmap"
    );
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(10, 5)).unwrap();
    assert!(results.results.iter().all(|r| r.doc_id != 10));
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(
        results.results[0].doc_id, 1,
        "upserted copy survives the merge"
    );

    let stats = index.stats().unwrap();
    // ids 1..=49 (49 docs; the upsert replaces doc 1 in place) minus the
    // physically reclaimed doc 10.
    assert_eq!(stats.vector_count, 48);
}

/// #881 review (HIGH): the upsert dance can undelete every mark; a previously
/// persisted delmap must then be REMOVED — a stale file would mask the
/// committed upsert in every segment after reopen.
#[test]
fn undelete_to_zero_removes_stale_delmap_and_survives_reopen() {
    let storage = storage();
    {
        let index = SegmentedHnswIndex::open_or_create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(true),
        )
        .unwrap();
        commit_batch(&index, 1..10);

        // Soft-delete doc 3 and persist: the delmap file exists.
        index.soft_delete_document(3).unwrap();
        index.persist_deletions().unwrap();
        assert!(storage.file_exists("vector_index.delmap"));

        // Same-id upsert of doc 3: the re-add clears the only mark.
        let mut writer = index.writer().unwrap();
        writer.delete_document(3).unwrap();
        writer
            .add_vectors(vec![(3, "v".to_string(), doc_vec(9000))])
            .unwrap();
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        assert!(
            !storage.file_exists("vector_index.delmap"),
            "undelete-to-zero must remove the stale delmap (#881)"
        );
    }

    // Reopen: the committed upsert must be visible (a stale delmap would
    // have masked doc 3 in every segment).
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(
        results.results[0].doc_id, 3,
        "the committed upsert must survive a reopen (#881)"
    );
}

/// #881 review (HIGH): a sealed writer must reject further commits (loud
/// error instead of silently resetting the segment's generation), while a
/// post-commit close() stays a clean no-op.
#[test]
fn sealed_writer_rejects_second_commit_and_close_is_noop() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(1))])
        .unwrap();
    writer.commit().unwrap();

    // Post-commit close: no pending changes, no error, no metadata churn.
    assert!(!writer.has_pending_changes());
    writer.close().unwrap();

    // A sealed writer with NEW changes must refuse to commit again.
    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(2, "v".to_string(), doc_vec(2))])
        .unwrap();
    writer.commit().unwrap();
    writer
        .add_vectors(vec![(3, "v".to_string(), doc_vec(3))])
        .unwrap();
    let err = writer.commit();
    assert!(
        err.is_err(),
        "a sealed writer must reject a second commit with new changes (#881)"
    );
}

/// #881 review: count() excludes soft-deleted docs.
#[test]
fn count_excludes_soft_deleted_docs() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    commit_batch(&index, 0..10);
    index.soft_delete_document(4).unwrap();

    let searcher = index.searcher().unwrap();
    let count = searcher.count(query(0, 1)).unwrap();
    assert_eq!(count, 9, "count must exclude soft-deleted docs (#881)");
}

/// Assert a search for every doc returns `doc_id`, then soft delete it and
/// assert a fresh searcher no longer returns it.
///
/// The first search caches the segment readers. Segment readers filter
/// deletions through the bitmap attached when they were loaded, so this
/// fails if readers cached before the delete created the bitmap are reused.
/// Only membership is asserted: HNSW's graph, and so its exact ranking,
/// varies between runs.
fn assert_soft_delete_reaches_cached_readers(index: &SegmentedHnswIndex, doc_id: u64) {
    let all = 20;
    let hits = index
        .searcher()
        .unwrap()
        .search(&query(doc_id, all))
        .unwrap();
    assert!(hits.results.iter().any(|r| r.doc_id == doc_id));

    index.soft_delete_document(doc_id).unwrap();
    let hits = index
        .searcher()
        .unwrap()
        .search(&query(doc_id, all))
        .unwrap();
    assert!(
        hits.results.iter().all(|r| r.doc_id != doc_id),
        "doc {doc_id} was soft deleted after its readers were cached"
    );
}

/// The first soft delete creates the deletion bitmap; readers cached before
/// it must not keep serving the deleted doc.
#[test]
fn first_soft_delete_hides_doc_from_cached_readers() {
    let index = SegmentedHnswIndex::open_or_create(
        storage() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    commit_batch(&index, 0..10);
    commit_batch(&index, 10..20);

    assert_soft_delete_reaches_cached_readers(&index, 4);
}

/// `optimize` drops the deletion bitmap; the next soft delete creates a new
/// one, which readers cached in between must pick up.
#[test]
fn soft_delete_after_optimize_hides_doc_from_cached_readers() {
    let index = SegmentedHnswIndex::open_or_create(
        storage() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    commit_batch(&index, 0..10);
    commit_batch(&index, 10..20);
    index.soft_delete_document(1).unwrap();
    index.optimize().unwrap();

    assert_soft_delete_reaches_cached_readers(&index, 14);
}

/// #882 (CRITICAL review find): the migration must fire through the
/// PRODUCTION factory path. A legacy index always has a `metadata.json`, so
/// it always takes the factory's OPEN arm — which previously had no
/// segmented dispatch, leaving every existing deployment monolithic forever
/// and (worse) flipping an already-migrated directory back to a monolithic
/// view that silently hid post-migration segments.
#[test]
fn factory_open_path_migrates_legacy_index_and_stays_segmented() {
    use laurus::vector::index::config::VectorIndexTypeConfig;
    use laurus::vector::index::factory::VectorIndexFactory;

    let storage = storage();
    {
        let index = HnswIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..50);
    }
    assert!(storage.file_exists("metadata.json"), "legacy precondition");

    // The exact production call (VectorStore::with_index_type_config).
    let index = VectorIndexFactory::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        VectorIndexTypeConfig::HNSW(config(true)),
    )
    .unwrap();
    assert!(
        storage.file_exists("segments.json"),
        "the factory OPEN arm must migrate a legacy index (#882)"
    );
    assert!(
        !storage.file_exists("metadata.json"),
        "the stale monolithic metadata.json must be removed so factory \
         routing can never regress to the monolithic view (#882)"
    );

    // Commit through the migrated index, then reopen through the factory
    // again: the segmented view must persist and serve ALL data.
    commit_batch(index.as_ref(), 50..51);
    drop(index);
    let index = VectorIndexFactory::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        VectorIndexTypeConfig::HNSW(config(true)),
    )
    .unwrap();
    let searcher = index.searcher().unwrap();
    assert_eq!(
        searcher.search(&query(42, 1)).unwrap().results[0].doc_id,
        42
    );
    assert_eq!(
        searcher.search(&query(50, 1)).unwrap().results[0].doc_id,
        50
    );
}

/// #882: opening a segmented directory with `segmented: false` must be
/// rejected loudly — the monolithic view would silently hide every sealed
/// segment.
#[test]
fn factory_rejects_segmented_directory_with_flag_off() {
    use laurus::vector::index::config::VectorIndexTypeConfig;
    use laurus::vector::index::factory::VectorIndexFactory;

    let storage = storage();
    {
        let index = SegmentedHnswIndex::open_or_create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(true),
        )
        .unwrap();
        commit_batch(&index, 0..10);
    }

    let result = VectorIndexFactory::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        VectorIndexTypeConfig::HNSW(config(false)),
    );
    assert!(
        result.is_err(),
        "a segmented directory must not open monolithically (#882)"
    );
}

/// #882: pure-append workloads must not grow the segment count unboundedly
/// — `maybe_auto_compact` merges a policy window once the manager's
/// threshold is exceeded (tiered policy lands with #883).
#[test]
fn append_only_segment_count_is_bounded_by_auto_merge() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    // 101 one-doc commits exceed the default max_segments (100).
    for i in 0..101u64 {
        commit_batch(&index, i..i + 1);
    }
    let before = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".hnsw"))
        .count();
    assert!(before > 100, "precondition: {before} segments");

    let compacted = index.maybe_auto_compact().unwrap();
    assert!(compacted, "the segment-count bound must trigger a merge");
    let after = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".hnsw"))
        .count();
    assert!(
        after < before,
        "the merge must reduce the segment count ({before} -> {after})"
    );

    // Every doc is still searchable through the merged layout.
    let searcher = index.searcher().unwrap();
    for id in [0u64, 50, 100] {
        assert_eq!(
            searcher.search(&query(id, 1)).unwrap().results[0].doc_id,
            id
        );
    }
}

/// #882: serde behavior of the flipped default — a config missing the field
/// deserializes to `true`; an explicit `false` is preserved.
#[test]
fn segmented_flag_serde_default_and_explicit_false() {
    // A pre-#882 config = today's config with the `segmented` key removed.
    let mut value: serde_json::Value = serde_json::to_value(HnswIndexConfig::default()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("segmented")
        .expect("the flag must serialize");
    let config: HnswIndexConfig = serde_json::from_value(value).unwrap();
    assert!(
        config.segmented,
        "a config serialized before the field existed must open segmented (#882)"
    );

    let explicit_false = serde_json::to_string(&HnswIndexConfig {
        segmented: false,
        ..HnswIndexConfig::default()
    })
    .unwrap();
    let config: HnswIndexConfig = serde_json::from_str(&explicit_false).unwrap();
    assert!(
        !config.segmented,
        "an explicit `segmented: false` must be preserved (#882)"
    );
}

/// #883: under sustained ingest with the store-ladder cadence (compact after
/// every commit), the tiered policy keeps the segment count logarithmically
/// bounded instead of one-segment-per-commit.
#[test]
fn sustained_ingest_keeps_segment_count_tiered() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    for i in 0..100u64 {
        commit_batch(&index, i * 5..(i + 1) * 5);
        // The store's commit ladder calls maybe_auto_compact every commit.
        index.maybe_auto_compact().unwrap();
    }

    let segments = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".hnsw"))
        .count();
    assert!(
        segments <= 30,
        "tiered merging must keep the segment count bounded under \
         sustained ingest, got {segments} after 100 commits (#883)"
    );

    // Every doc remains searchable through the merged layout (top-5, since
    // the corpus is a tight curve where adjacent ids are near-duplicates
    // and approximate top-1 can legitimately flip between neighbours).
    let searcher = index.searcher().unwrap();
    for id in [0u64, 250, 499] {
        let results = searcher.search(&query(id, 5)).unwrap();
        assert!(
            results.results.iter().any(|r| r.doc_id == id),
            "doc {id} must stay searchable after tiered merging"
        );
    }
}

/// #883: adaptive refill — a live doc ranked *below a deep band of stale
/// upsert copies inside the same segment* must still surface. This is a
/// red/green gate for the whole refill feature: the stale band (30 copies) is
/// deeper than the first pass's 2x over-fetch AND deeper than any fixed
/// multiplier (the earlier one-shot 8x refill missed it), so the query
/// returns the wrong far doc unless the budget expands geometrically until
/// the live hit is reached.
#[test]
fn adaptive_refill_recovers_live_doc_behind_stale_band() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    // Segment A (older): the stale band (docs 1..=30, tightly clustered
    // around doc_vec(0)) AND the LIVE doc 100 (doc_vec(31), just past the
    // band) are committed together, so doc 100 sits at rank 31 *within the
    // same segment* — behind all 30 copies. This is the construction the
    // refill must handle; committing doc 100 separately would let it surface
    // unmasked from its own segment and never exercise the refill.
    {
        let mut writer = index.writer().unwrap();
        let mut batch: Vec<_> = (1..=30u64)
            .map(|i| (i, "v".to_string(), doc_vec(i)))
            .collect();
        batch.push((100, "v".to_string(), doc_vec(31)));
        writer.add_vectors(batch).unwrap();
        writer.commit().unwrap();
    }

    // Segment C (newest): upserted copies of docs 1..=30, far away from the
    // query — the old copies in segment A become a pure stale band that
    // masking must skip over to reach doc 100.
    {
        let mut writer = index.writer().unwrap();
        for id in 1..=30u64 {
            writer.delete_document(id).unwrap();
            writer
                .add_vectors(vec![(id, "v".to_string(), doc_vec(9000 + id))])
                .unwrap();
        }
        writer.commit().unwrap();
    }

    // Query at the old cluster centre with top_k=1: the 30 hits ranked above
    // doc 100 in segment A are all masked stale copies. A fixed 2x (or 8x)
    // over-fetch never reaches rank 31; only the expanding refill does.
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(0, 1)).unwrap();
    assert_eq!(
        results.results.first().map(|r| r.doc_id),
        Some(100),
        "the live doc behind the deep stale band must surface via the \
         expanding adaptive refill (#883), got {:?}",
        results.results
    );
}

/// #883: the campaign's headline number as a permanent deterministic gate —
/// the auto-commit ingest scenario writes an order of magnitude fewer
/// cumulative bytes under the segmented layout than under the monolithic
/// one (each monolithic commit rewrites and re-quantizes the whole index).
#[test]
fn auto_commit_cumulative_bytes_are_bounded() {
    use common::ByteCountingStorage;

    let run = |segmented: bool| -> u64 {
        let counting = Arc::new(ByteCountingStorage::new(Arc::new(MemoryStorage::new(
            MemoryStorageConfig::default(),
        ))));
        let written = counting.written.clone();
        let index: Box<dyn VectorIndex> = if segmented {
            Box::new(
                SegmentedHnswIndex::open_or_create(
                    counting.clone() as Arc<dyn Storage>,
                    "vector_index",
                    config(true),
                )
                .unwrap(),
            )
        } else {
            Box::new(
                HnswIndex::create(
                    counting.clone() as Arc<dyn Storage>,
                    "vector_index",
                    config(false),
                )
                .unwrap(),
            )
        };
        // 40 commits of 50 docs each = 2000 docs, the auto-commit cadence.
        for i in 0..40u64 {
            commit_batch(index.as_ref(), i * 50..(i + 1) * 50);
            index.maybe_auto_compact().unwrap();
        }
        written.load(std::sync::atomic::Ordering::Relaxed)
    };

    let monolithic = run(false);
    let segmented = run(true);
    eprintln!(
        "auto-commit cumulative bytes: monolithic={monolithic} segmented={segmented} \
         ratio={:.1}x",
        monolithic as f64 / segmented as f64
    );
    assert!(
        segmented * 5 < monolithic,
        "the segmented layout must write at least 5x fewer cumulative bytes \
         under auto-commit ingest, got monolithic={monolithic} vs segmented={segmented} (#883)"
    );
}
