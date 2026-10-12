//! Gates for the segment-per-commit IVF index (Issue #889 PR-6, mirroring
//! `vector_segmented_index_test.rs` (HNSW) and `vector_segmented_flat_test.rs`
//! (Flat)).
//!
//! Two IVF-specific fixture choices, learned while stabilizing this suite:
//!
//! - `doc_vec` uses a well-mixed LCG (like `ivf/writer.rs`'s own
//!   `lcg_vectors` test helper) rather than a smoothly-varying curve. A
//!   slowly-varying curve (e.g. `cos(i * 0.001)`) packs neighboring ids too
//!   closely together once int8-quantized (Scalar8Bit is IVF's default),
//!   producing spurious similarity ties that make exact top-1 self-lookup
//!   flaky — the LCG spreads ids pseudo-randomly across all 16 dimensions,
//!   which is robust against that.
//! - `config()` sets a large `n_probe` so per-segment search is effectively
//!   exhaustive over that segment's own clusters. IVF search only examines
//!   the `n_probe` nearest clusters — unlike Flat's brute-force scan, an
//!   over-restrictive `n_probe` can starve the multi-segment fan-out's
//!   masking/expanding-refill (Issue #883): if a query's nearest cluster in
//!   an older segment is entirely shadowed by a newer segment, expanding
//!   `top_k` within that SAME one probed cluster surfaces nothing new,
//!   since the fan-out layer's refill loop does not also grow `n_probe`.
//!   Structural/functional gates below want that confound removed; the
//!   dedicated `recall_improves_after_merge_for_small_k_segments` test
//!   deliberately overrides `n_probe` back down to exercise genuine
//!   approximate search.

mod common;

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::IvfIndexConfig;
use laurus::vector::index::ivf::IvfIndex;
use laurus::vector::index::ivf::segmented::SegmentedIvfIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

fn doc_vec(i: u64) -> Vector {
    let mut state: u64 = 0x9E3779B97F4A7C15u64.wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let data: Vec<f32> = (0..16)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Vector::new(data)
}

fn config(segmented: bool) -> IvfIndexConfig {
    IvfIndexConfig {
        dimension: 16,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        // Effectively exhaustive per-segment search (clamped internally to
        // the segment's own cluster count) — see the module docs.
        n_probe: 10_000,
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

fn ivf_segment_files(storage: &MemoryStorage) -> Vec<String> {
    storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".ivf"))
        .collect()
}

/// Core gate: a 1-doc commit on a 1000-doc base writes a new segment that is
/// a tiny fraction of the base segment, and never rewrites the base.
#[test]
fn one_doc_commit_writes_o_delta_bytes() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    commit_batch(&index, 0..1000);
    let base_file = "segment_000000.ivf";
    let base_size = storage.file_size(base_file).unwrap();

    commit_batch(&index, 1000..1001);
    let delta_file = "segment_000001.ivf";
    let delta_size = storage.file_size(delta_file).unwrap();

    assert!(
        delta_size * 10 < base_size,
        "a 1-doc commit must write O(delta) bytes, got delta={delta_size} vs base={base_size}"
    );
    assert_eq!(
        storage.file_size(base_file).unwrap(),
        base_size,
        "the base segment must never be rewritten by a later commit"
    );
    assert_eq!(
        ivf_segment_files(&storage).len(),
        2,
        "exactly one new segment per non-empty commit"
    );
}

/// With `segmented: false` (the default for IVF) the factory path stays
/// monolithic — no manifest, single `.ivf` file.
#[test]
fn config_off_keeps_monolithic_layout() {
    let storage = storage();
    let index = IvfIndex::create(
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
    assert!(storage.file_exists("vector_index.ivf"));
}

/// A legacy monolithic index is migrated ZERO-COPY on first segmented open
/// — the existing `.ivf` becomes segment 0 verbatim, search results are
/// identical, and later commits append new segments without ever touching
/// the legacy file.
#[test]
fn legacy_monolithic_index_migrates_zero_copy() {
    let storage = storage();
    {
        let index = IvfIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..100);
    }
    let legacy_size = storage.file_size("vector_index.ivf").unwrap();

    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    assert!(storage.file_exists("segments.json"), "manifest created");
    assert_eq!(
        storage.file_size("vector_index.ivf").unwrap(),
        legacy_size,
        "zero-copy: the legacy file must not be rewritten"
    );

    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(42, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 42);

    commit_batch(&index, 100..101);
    assert_eq!(storage.file_size("vector_index.ivf").unwrap(), legacy_size);
    let searcher = index.searcher().unwrap();
    assert_eq!(
        searcher.search(&query(100, 1)).unwrap().results[0].doc_id,
        100
    );
    assert_eq!(
        searcher.search(&query(42, 1)).unwrap().results[0].doc_id,
        42
    );

    drop(index);
    let index = SegmentedIvfIndex::open_or_create(
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

/// The WAL checkpoint is published only by `persist_deletions` (the end of
/// the store's commit ladder) — never by intermediate manifest saves.
#[test]
fn wal_checkpoint_publishes_only_after_persist_deletions() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    index.set_last_wal_seq(7).unwrap();
    assert_eq!(
        index.last_wal_seq(),
        0,
        "a pending seq must not be visible before persist_deletions"
    );

    commit_batch(&index, 0..5);
    assert_eq!(
        index.last_wal_seq(),
        0,
        "sealing must not publish the pending checkpoint"
    );

    index.persist_deletions().unwrap();
    assert_eq!(index.last_wal_seq(), 7);
}

/// Migrating a legacy monolithic index whose stored dimension does not match
/// the config's dimension must fail loudly instead of silently proceeding
/// (Issue #1409: this guards the dimension check staying in place once IVF
/// shares its migration code with HNSW and Flat).
#[test]
fn legacy_migration_rejects_a_dimension_mismatch() {
    let storage = storage();
    {
        let index = IvfIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..10);
    }

    let mut mismatched = config(true);
    mismatched.dimension = 8;
    let err =
        SegmentedIvfIndex::open_or_create(storage as Arc<dyn Storage>, "vector_index", mismatched);
    assert!(
        err.is_err(),
        "a dimension mismatch during migration must fail"
    );
}

/// A freshly obtained writer that was never mutated reports no pending
/// changes. IVF's inner writer never overrides `has_pending_changes` (the
/// trait default is unconditionally `true`), so the segmented writer only
/// looks at its own buffer, not the inner writer's own flag (Issue #1409:
/// this is the behavior HNSW's writer deliberately does NOT share).
#[test]
fn fresh_writer_has_no_pending_changes_before_any_mutation() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    let writer = index.writer().unwrap();
    assert!(!writer.has_pending_changes());
}

/// Dropping a writer that was never mutated must not roll back a WAL
/// checkpoint: the `Drop` backstop is guarded by `has_pending_changes`,
/// which is `false` for an untouched IVF writer (Issue #1409: HNSW's own
/// writer differs here — see `vector_segmented_index_test.rs`).
#[test]
fn dropping_an_unused_writer_does_not_roll_back_the_wal_checkpoint() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    index.set_last_wal_seq(7).unwrap();
    drop(index.writer().unwrap());
    index.persist_deletions().unwrap();

    assert_eq!(index.last_wal_seq(), 7);
}

/// Dropping a sealed writer after `close()` must not roll back the WAL
/// checkpoint: `close()` empties the inner buffer, which is not a change
/// since the seal, and the sealed segment already holds every buffered
/// record (Issue #1410).
#[test]
fn dropping_a_closed_sealed_writer_keeps_the_wal_checkpoint() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    index.set_last_wal_seq(7).unwrap();
    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(1))])
        .unwrap();
    writer.commit().unwrap();
    writer.close().unwrap();
    drop(writer);
    index.persist_deletions().unwrap();

    assert_eq!(index.last_wal_seq(), 7);
}

/// Dropping a sealed writer after a change was rolled back must not roll back
/// the WAL checkpoint: the rollback discarded every buffered change, so a
/// sequence number set afterwards, here for deleting a sealed document, must
/// be published (Issue #1419).
#[test]
fn dropping_a_rolled_back_sealed_writer_keeps_the_wal_checkpoint() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    index.set_last_wal_seq(7).unwrap();
    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(1))])
        .unwrap();
    writer.commit().unwrap();
    index.persist_deletions().unwrap();

    writer
        .add_vectors(vec![(2, "v".to_string(), doc_vec(2))])
        .unwrap();
    writer.rollback().unwrap();

    // Doc 1 is sealed and no longer buffered: this only marks the deletion
    // bitmap.
    writer.delete_document(1).unwrap();
    index.set_last_wal_seq(9).unwrap();
    drop(writer);
    index.persist_deletions().unwrap();

    assert_eq!(index.last_wal_seq(), 9);
}

/// Self-recall of a multi-segment index matches a monolithic build of the
/// same corpus. IVF self-lookup is always exact (see the module docs), so
/// this pins that the newest-wins containment masking across segments does
/// not accidentally drop or duplicate live docs.
#[test]
fn multi_segment_self_recall_matches_monolithic() {
    let n = 1000u64;
    let per_commit = 200u64;

    let seg_storage = storage();
    let seg_index = SegmentedIvfIndex::open_or_create(
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
    let mono_index = IvfIndex::create(
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
            let results = searcher.search(&query(id, 1)).unwrap();
            if results.results.iter().any(|r| r.doc_id == id) {
                hits += 1;
            }
        }
        hits as f32 / n as f32
    };

    assert_eq!(recall(&mono_index), 1.0, "monolithic self-recall sanity");
    assert_eq!(
        recall(&seg_index),
        1.0,
        "multi-segment self-recall must match the monolithic build"
    );
}

/// A same-id upsert across commits resolves to the newest copy exactly
/// once, and a soft delete of a sealed doc is search-invisible and
/// physically reclaimed by optimize() — the first-time soft-delete/
/// compaction implementation for IVF.
#[test]
fn upsert_and_soft_delete_across_commits() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    commit_batch(&index, 1..50);

    {
        let mut writer = index.writer().unwrap();
        writer.delete_document(1).unwrap();
        writer
            .add_vectors(vec![(1, "v".to_string(), doc_vec(9000))])
            .unwrap();
        writer.commit().unwrap();
    }

    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 1, "newest copy must win");
    let results = searcher.search(&query(1, 1)).unwrap();
    assert_ne!(
        results.results[0].doc_id, 1,
        "the stale copy in the older segment must be masked"
    );

    // Soft delete a doc living in an already-SEALED segment (also exercises
    // the quantized-pool fast-path `is_deleted` fix — Scalar8Bit is IVF's
    // default quantization method, so `IvfSearcher`'s hot path is active).
    index.soft_delete_document(10).unwrap();
    index.persist_deletions().unwrap();
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(10, 5)).unwrap();
    assert!(
        results.results.iter().all(|r| r.doc_id != 10),
        "a soft-deleted sealed doc must be search-invisible"
    );

    index.optimize().unwrap();
    assert_eq!(
        ivf_segment_files(&storage).len(),
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
    assert_eq!(stats.vector_count, 48);
}

/// The upsert dance can undelete every mark; a previously persisted delmap
/// must then be REMOVED.
#[test]
fn undelete_to_zero_removes_stale_delmap_and_survives_reopen() {
    let storage = storage();
    {
        let index = SegmentedIvfIndex::open_or_create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(true),
        )
        .unwrap();
        commit_batch(&index, 1..10);

        index.soft_delete_document(3).unwrap();
        index.persist_deletions().unwrap();
        assert!(storage.file_exists("vector_index.delmap"));

        let mut writer = index.writer().unwrap();
        writer.delete_document(3).unwrap();
        writer
            .add_vectors(vec![(3, "v".to_string(), doc_vec(9000))])
            .unwrap();
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        assert!(
            !storage.file_exists("vector_index.delmap"),
            "undelete-to-zero must remove the stale delmap"
        );
    }

    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(
        results.results[0].doc_id, 3,
        "the committed upsert must survive a reopen"
    );
}

/// A sealed writer must reject further commits, while a post-commit
/// close() stays a clean no-op.
#[test]
fn sealed_writer_rejects_second_commit_and_close_is_noop() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
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

    assert!(!writer.has_pending_changes());
    writer.close().unwrap();

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
        "a sealed writer must reject a second commit with new changes"
    );
}

/// Replacing a sealed writer's buffered doc with the same number of vectors
/// (delete-first, then re-add, as the store's upsert does) leaves the buffer
/// length unchanged; the writer must still report the change and refuse to
/// commit it instead of silently dropping the new version (Issue #1410).
/// Deleting an id that is not buffered changes nothing, and rolling back
/// discards the change without unsealing the writer (Issue #1419).
#[test]
fn sealed_writer_rejects_a_same_size_replacement() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(1))])
        .unwrap();
    writer.commit().unwrap();

    writer.delete_document(2).unwrap();
    assert!(!writer.has_pending_changes());

    writer.delete_document(1).unwrap();
    assert!(writer.has_pending_changes());
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(9000))])
        .unwrap();
    assert!(writer.has_pending_changes());
    let err = writer.commit().unwrap_err();
    assert!(err.to_string().contains("already sealed"), "{err}");

    writer.rollback().unwrap();
    assert!(!writer.has_pending_changes());
    assert_eq!(writer.pending_docs(), 0);

    // Sealing again would overwrite the registered segment.
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(9000))])
        .unwrap();
    let err = writer.commit().unwrap_err();
    assert!(err.to_string().contains("already sealed"), "{err}");
}

/// Rolling back before the first commit discards the buffer and leaves the
/// writer unsealed: the next commit seals only the documents added after the
/// rollback (Issue #1419).
#[test]
fn rollback_before_the_first_commit_discards_the_buffer() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(1))])
        .unwrap();
    writer.rollback().unwrap();
    assert!(!writer.has_pending_changes());
    assert_eq!(writer.pending_docs(), 0);

    writer
        .add_vectors(vec![(2, "v".to_string(), doc_vec(2))])
        .unwrap();
    writer.commit().unwrap();
    index.persist_deletions().unwrap();

    let reopened = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    assert_eq!(
        reopened.reader().unwrap().vector_ids().unwrap(),
        vec![(2, "v".to_string())]
    );
}

/// count() excludes soft-deleted docs.
#[test]
fn count_excludes_soft_deleted_docs() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    commit_batch(&index, 0..10);
    index.soft_delete_document(4).unwrap();

    let searcher = index.searcher().unwrap();
    let count = searcher.count(query(0, 1)).unwrap();
    assert_eq!(count, 9, "count must exclude soft-deleted docs");
}

/// Assert a search for every doc returns `doc_id`, then soft delete it and
/// assert a fresh searcher no longer returns it.
///
/// The first search caches the segment readers. Segment readers filter
/// deletions through the bitmap attached when they were loaded, so this
/// fails if readers cached before the delete created the bitmap are reused.
fn assert_soft_delete_reaches_cached_readers(index: &SegmentedIvfIndex, doc_id: u64) {
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
    let index = SegmentedIvfIndex::open_or_create(
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
    let index = SegmentedIvfIndex::open_or_create(
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

/// The migration must fire through the PRODUCTION factory path.
#[test]
fn factory_open_path_migrates_legacy_index_and_stays_segmented() {
    use laurus::vector::index::config::VectorIndexTypeConfig;
    use laurus::vector::index::factory::VectorIndexFactory;

    let storage = storage();
    {
        let index = IvfIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..50);
    }
    assert!(storage.file_exists("metadata.json"), "legacy precondition");

    let index = VectorIndexFactory::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        VectorIndexTypeConfig::IVF(config(true)),
    )
    .unwrap();
    assert!(
        storage.file_exists("segments.json"),
        "the factory OPEN arm must migrate a legacy index"
    );
    assert!(
        !storage.file_exists("metadata.json"),
        "the stale monolithic metadata.json must be removed"
    );

    commit_batch(index.as_ref(), 50..51);
    drop(index);
    let index = VectorIndexFactory::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        VectorIndexTypeConfig::IVF(config(true)),
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

/// Opening a segmented directory with `segmented: false` must be rejected
/// loudly.
#[test]
fn factory_rejects_segmented_directory_with_flag_off() {
    use laurus::vector::index::config::VectorIndexTypeConfig;
    use laurus::vector::index::factory::VectorIndexFactory;

    let storage = storage();
    {
        let index = SegmentedIvfIndex::open_or_create(
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
        VectorIndexTypeConfig::IVF(config(false)),
    );
    assert!(
        result.is_err(),
        "a segmented directory must not open monolithically"
    );
}

/// Pure-append workloads must not grow the segment count unboundedly.
#[test]
fn append_only_segment_count_is_bounded_by_auto_merge() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    for i in 0..101u64 {
        commit_batch(&index, i..i + 1);
    }
    let before = ivf_segment_files(&storage).len();
    assert!(before > 100, "precondition: {before} segments");

    let compacted = index.maybe_auto_compact().unwrap();
    assert!(compacted, "the segment-count bound must trigger a merge");
    let after = ivf_segment_files(&storage).len();
    assert!(
        after < before,
        "the merge must reduce the segment count ({before} -> {after})"
    );

    let searcher = index.searcher().unwrap();
    for id in [0u64, 50, 100] {
        assert_eq!(
            searcher.search(&query(id, 1)).unwrap().results[0].doc_id,
            id
        );
    }
}

/// serde behavior of the (now on-by-default, Issue #907) flag.
#[test]
fn segmented_flag_serde_default_and_explicit_false() {
    // A pre-#907 config = today's config with the `segmented` key removed.
    let mut value: serde_json::Value = serde_json::to_value(IvfIndexConfig::default()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("segmented")
        .expect("the flag must serialize");
    let config: IvfIndexConfig = serde_json::from_value(value).unwrap();
    assert!(
        config.segmented,
        "a config serialized before the field existed must open segmented (#907)"
    );

    let explicit_false = serde_json::to_string(&IvfIndexConfig {
        segmented: false,
        ..IvfIndexConfig::default()
    })
    .unwrap();
    let config: IvfIndexConfig = serde_json::from_str(&explicit_false).unwrap();
    assert!(
        !config.segmented,
        "an explicit `segmented: false` must be preserved"
    );
}

/// Under sustained ingest with the store-ladder cadence, the tiered policy
/// keeps the segment count logarithmically bounded.
#[test]
fn sustained_ingest_keeps_segment_count_tiered() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    for i in 0..100u64 {
        commit_batch(&index, i * 5..(i + 1) * 5);
        index.maybe_auto_compact().unwrap();
    }

    let segments = ivf_segment_files(&storage).len();
    assert!(
        segments <= 30,
        "tiered merging must keep the segment count bounded, got {segments} after 100 commits"
    );

    let searcher = index.searcher().unwrap();
    for id in [0u64, 250, 499] {
        let results = searcher.search(&query(id, 5)).unwrap();
        assert!(
            results.results.iter().any(|r| r.doc_id == id),
            "doc {id} must stay searchable after tiered merging"
        );
    }
}

/// The campaign's headline number as a permanent deterministic gate.
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
                SegmentedIvfIndex::open_or_create(
                    counting.clone() as Arc<dyn Storage>,
                    "vector_index",
                    config(true),
                )
                .unwrap(),
            )
        } else {
            Box::new(
                IvfIndex::create(
                    counting.clone() as Arc<dyn Storage>,
                    "vector_index",
                    config(false),
                )
                .unwrap(),
            )
        };
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
        segmented * 3 < monolithic,
        "the segmented layout must write materially fewer cumulative bytes under \
         auto-commit ingest, got monolithic={monolithic} vs segmented={segmented}"
    );
}

/// A commit sealing fewer than `n_clusters` new vectors must succeed
/// end-to-end via the adaptive-K clamp (Issue #889 PR-5), through the
/// segmented writer specifically.
#[test]
fn sub_n_clusters_commit_succeeds_via_adaptive_k() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true), // n_clusters defaults to 100
    )
    .unwrap();

    // Only 5 vectors, far below the default 100-cluster ceiling.
    commit_batch(&index, 0..5);

    let searcher = index.searcher().unwrap();
    for id in 0..5u64 {
        assert_eq!(
            searcher.search(&query(id, 1)).unwrap().results[0].doc_id,
            id
        );
    }
}

/// Merge-time re-clustering: commit several small segments (whose own
/// adaptive-K was clamped to their own tiny size), force-merge, and confirm
/// every document is retrievable and the merged segment's cluster count
/// matches the adaptive value for the UNION's size — not the sum, or any
/// individual source's, cluster count.
#[test]
fn merge_time_reclustering_matches_union_adaptive_k() {
    use laurus::vector::index::ivf::reader::IvfIndexReader;

    let storage = storage();
    let cfg = IvfIndexConfig {
        n_clusters: 100,
        ..config(true)
    };
    let index =
        SegmentedIvfIndex::open_or_create(storage.clone() as Arc<dyn Storage>, "vector_index", cfg)
            .unwrap();

    // 5 segments of 10 docs each: each segment's own adaptive K is
    // min(100, 10) = 10 (over-clustered relative to itself).
    for i in 0..5u64 {
        commit_batch(&index, i * 10..(i + 1) * 10);
    }
    assert_eq!(ivf_segment_files(&storage).len(), 5);

    index.optimize().unwrap();
    let remaining = ivf_segment_files(&storage);
    assert_eq!(remaining.len(), 1, "force-merge collapses to one segment");
    let segment_id = remaining[0].trim_end_matches(".ivf").to_string();

    let reader = IvfIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &segment_id,
        DistanceMetric::Cosine,
    )
    .unwrap();
    let (n_clusters, _) = reader.ivf_params();
    assert_eq!(
        n_clusters, 50,
        "the merged segment must retrain adaptive K over the UNION size (50), \
         not sum or inherit any source's own cluster count, got {n_clusters}"
    );

    let searcher = index.searcher().unwrap();
    for id in 0..50u64 {
        assert_eq!(
            searcher.search(&query(id, 1)).unwrap().results[0].doc_id,
            id,
            "doc {id} must be retrievable after the merge"
        );
    }
}

/// A merge window whose every vector is logically deleted must reduce to a
/// valid zero-vector, zero-cluster segment instead of erroring (gates the
/// `IvfIndexWriter::finalize`/`train_centroids` fix — Issue #889 PR-6).
#[test]
fn full_deletion_then_merge_produces_valid_empty_segment() {
    let storage = storage();
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    commit_batch(&index, 0..20);
    for id in 0..20u64 {
        index.soft_delete_document(id).unwrap();
    }
    index.persist_deletions().unwrap();

    // Force-merge a window where every source vector is deleted: must not
    // error, and must leave a valid (empty) index behind.
    index.optimize().unwrap();
    let stats = index.stats().unwrap();
    assert_eq!(stats.vector_count, 0);

    // The index must still accept new commits after collapsing to empty.
    commit_batch(&index, 100..105);
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(100, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 100);
}

/// Small, independently-over-clustered segments have worse *approximate*
/// (non-self-lookup) recall than the same corpus re-clustered as a whole:
/// with `n_clusters` >= each tiny commit's size, k-means degenerates to
/// (near-)one-cluster-per-point, splitting near-identical pairs into
/// different clusters; `n_probe = 1` then misses a pair's partner. Merging
/// re-trains over the full union with the SAME `n_clusters` ceiling, which
/// is now much coarser relative to corpus size, so k-means groups
/// near-identical pairs together again. Asserted as an aggregate rate
/// (not per-pair) to stay robust to individual seeded-k-means boundary
/// cases.
#[test]
fn recall_improves_after_merge_for_small_k_segments() {
    let storage = storage();
    let cfg = IvfIndexConfig {
        n_clusters: 10,
        n_probe: 1,
        ..config(true)
    };
    let index = SegmentedIvfIndex::open_or_create(storage as Arc<dyn Storage>, "vector_index", cfg)
        .unwrap();

    // 40 pair-groups, well separated from each other (an LCG-mixed 16-dim
    // anchor per group — a fixed-dimension modulo scheme would alias
    // distinct groups onto the same anchor); each pair member is a tiny
    // epsilon perturbation of its group's anchor. 5 complete pairs (10
    // vectors) per commit, so each tiny segment's own adaptive K = 10 = its
    // own vector count (the degenerate all-singleton regime).
    let pair_vec = |group: u64, member: u64| -> Vector {
        let mut state: u64 =
            0x9E3779B97F4A7C15u64.wrapping_add(group.wrapping_mul(0xBF58_476D_1CE4_E5B9));
        let mut v: Vec<f32> = (0..16)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2000.0 - 1000.0
            })
            .collect();
        v[0] += member as f32 * 0.01; // tiny epsilon distinguishing the pair
        Vector::new(v)
    };

    let groups_per_commit = 5u64;
    let total_groups = 40u64;
    let mut group = 0u64;
    while group < total_groups {
        let mut writer = index.writer().unwrap();
        let mut batch = Vec::new();
        for g in group..(group + groups_per_commit).min(total_groups) {
            batch.push((g * 2, "v".to_string(), pair_vec(g, 0)));
            batch.push((g * 2 + 1, "v".to_string(), pair_vec(g, 1)));
        }
        writer.add_vectors(batch).unwrap();
        writer.commit().unwrap();
        group += groups_per_commit;
    }

    let pair_found_rate = |index: &dyn VectorIndex| -> f32 {
        let searcher = index.searcher().unwrap();
        let mut both_found = 0u64;
        for g in 0..total_groups {
            let q = VectorIndexQuery {
                query: pair_vec(g, 0),
                params: VectorIndexQueryParams {
                    top_k: 2,
                    ..Default::default()
                },
                field_name: Some("v".to_string()),
                filter: None,
            };
            let results = searcher.search(&q).unwrap();
            let ids: std::collections::HashSet<u64> =
                results.results.iter().map(|r| r.doc_id).collect();
            if ids.contains(&(g * 2)) && ids.contains(&(g * 2 + 1)) {
                both_found += 1;
            }
        }
        both_found as f32 / total_groups as f32
    };

    let before = pair_found_rate(&index);
    index.optimize().unwrap();
    let after = pair_found_rate(&index);

    assert!(
        after > before,
        "merging must improve small-K segments' pair-recall, got before={before:.2} after={after:.2}"
    );
    assert!(
        after >= 0.9,
        "post-merge pair-recall should be high once re-clustered over the full union, got {after:.2}"
    );
}
