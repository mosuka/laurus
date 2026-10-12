//! Gates for the segment-per-commit Flat index (Issue #889 PR-4, mirroring
//! `vector_segmented_index_test.rs`'s HNSW gates).
//!
//! The deterministic core gate: committing 1 new document on an N-document
//! base writes O(delta) bytes — a new small segment file plus the manifest —
//! never a rewrite of the existing segments. Plus: config-OFF invariance
//! (the monolithic layout stays byte-identical), legacy-index rejection,
//! zero-copy migration, multi-segment recall parity, and the first-time
//! soft-delete/compaction lifecycle for Flat.

mod common;

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::FlatIndexConfig;
use laurus::vector::index::flat::FlatIndex;
use laurus::vector::index::flat::segmented::SegmentedFlatIndex;
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

fn config(segmented: bool) -> FlatIndexConfig {
    FlatIndexConfig {
        dimension: 16,
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

/// Core gate: a 1-doc commit on a 1000-doc base writes a new segment that is
/// a tiny fraction of the base segment, and never rewrites the base.
#[test]
fn one_doc_commit_writes_o_delta_bytes() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    commit_batch(&index, 0..1000);
    let base_file = "segment_000000.flat";
    let base_size = storage.file_size(base_file).unwrap();

    commit_batch(&index, 1000..1001);
    let delta_file = "segment_000001.flat";
    let delta_size = storage.file_size(delta_file).unwrap();

    assert!(
        delta_size * 20 < base_size,
        "a 1-doc commit must write O(delta) bytes, got delta={delta_size} vs base={base_size}"
    );
    assert_eq!(
        storage.file_size(base_file).unwrap(),
        base_size,
        "the base segment must never be rewritten by a later commit"
    );

    let flat_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".flat"))
        .collect();
    assert_eq!(
        flat_files.len(),
        2,
        "exactly one new segment per non-empty commit, got {flat_files:?}"
    );
}

/// With `segmented: false` (the default for Flat) the factory path stays
/// monolithic — no manifest, single `.flat` file.
#[test]
fn config_off_keeps_monolithic_layout() {
    let storage = storage();
    let index = FlatIndex::create(
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
    assert!(storage.file_exists("vector_index.flat"));
}

/// A legacy monolithic index is migrated ZERO-COPY on first segmented open
/// — the existing `.flat` becomes segment 0 verbatim (no data movement),
/// search results are identical, and later commits append new segments
/// without ever touching the legacy file.
#[test]
fn legacy_monolithic_index_migrates_zero_copy() {
    let storage = storage();
    {
        let index = FlatIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..100);
    }
    let legacy_size = storage.file_size("vector_index.flat").unwrap();

    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    assert!(storage.file_exists("segments.json"), "manifest created");
    assert_eq!(
        storage.file_size("vector_index.flat").unwrap(),
        legacy_size,
        "zero-copy: the legacy file must not be rewritten"
    );

    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(42, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 42);

    commit_batch(&index, 100..101);
    assert_eq!(storage.file_size("vector_index.flat").unwrap(), legacy_size);
    let searcher = index.searcher().unwrap();
    assert_eq!(
        searcher.search(&query(100, 1)).unwrap().results[0].doc_id,
        100
    );
    assert_eq!(
        searcher.search(&query(42, 1)).unwrap().results[0].doc_id,
        42
    );

    // Re-open: the migration must not re-run (idempotent) and everything
    // stays searchable.
    drop(index);
    let index = SegmentedFlatIndex::open_or_create(
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
/// the store's commit ladder) — never by intermediate manifest saves — so
/// recovery can only skip records whose effects are already durable.
#[test]
fn wal_checkpoint_publishes_only_after_persist_deletions() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
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
/// (Issue #1409: this guards the dimension check staying in place once Flat
/// shares its migration code with HNSW and IVF).
#[test]
fn legacy_migration_rejects_a_dimension_mismatch() {
    let storage = storage();
    {
        let index = FlatIndex::create(
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
        SegmentedFlatIndex::open_or_create(storage as Arc<dyn Storage>, "vector_index", mismatched);
    assert!(
        err.is_err(),
        "a dimension mismatch during migration must fail"
    );
}

/// A freshly obtained writer that was never mutated reports no pending
/// changes. Flat's inner writer never overrides `has_pending_changes` (the
/// trait default is unconditionally `true`), so the segmented writer only
/// looks at its own buffer, not the inner writer's own flag (Issue #1409:
/// this is the behavior HNSW's writer deliberately does NOT share).
#[test]
fn fresh_writer_has_no_pending_changes_before_any_mutation() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
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
/// which is `false` for an untouched Flat writer (Issue #1409: HNSW's own
/// writer differs here — see `vector_segmented_index_test.rs`).
#[test]
fn dropping_an_unused_writer_does_not_roll_back_the_wal_checkpoint() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
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

/// Self-recall@10 of a multi-segment index matches a monolithic build of the
/// same corpus. Flat is exact brute-force search, so both should be at (or
/// very near) 1.0 — this pins that the newest-wins containment masking
/// across segments does not accidentally drop or duplicate live docs.
#[test]
fn multi_segment_self_recall_matches_monolithic() {
    let n = 2000u64;
    let per_commit = 400u64;

    let seg_storage = storage();
    let seg_index = SegmentedFlatIndex::open_or_create(
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
    let mono_index = FlatIndex::create(
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
    assert!(mono > 0.99, "monolithic exact-search sanity, got {mono:.4}");
    assert!(
        seg >= mono - 0.01,
        "multi-segment self-recall ({seg:.4}) must match the monolithic build \
         ({mono:.4}) within tolerance"
    );
}

/// A same-id upsert across commits resolves to the newest copy exactly
/// once, and a soft delete of a sealed doc is search-invisible and
/// physically reclaimed by optimize() — the first-time soft-delete/
/// compaction implementation for Flat.
#[test]
fn upsert_and_soft_delete_across_commits() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
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

    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 1, "newest copy must win");
    let results = searcher.search(&query(1, 1)).unwrap();
    assert_ne!(
        results.results[0].doc_id, 1,
        "the stale copy in the older segment must be masked"
    );

    // Soft delete a doc living in an already-SEALED segment: search-
    // invisible immediately. This is the regression this PR exists to
    // close — the monolithic Flat writer's hard-delete only ever worked
    // because its buffer WAS the whole corpus; under segment-per-commit a
    // hard delete against an empty new-segment buffer would silently no-op.
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
    let flat_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".flat"))
        .collect();
    assert_eq!(
        flat_files.len(),
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
/// must then be REMOVED — a stale file would mask the committed upsert in
/// every segment after reopen.
#[test]
fn undelete_to_zero_removes_stale_delmap_and_survives_reopen() {
    let storage = storage();
    {
        let index = SegmentedFlatIndex::open_or_create(
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

    let index = SegmentedFlatIndex::open_or_create(
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

/// A sealed writer must reject further commits (loud error instead of
/// silently resetting the segment's generation), while a post-commit
/// close() stays a clean no-op.
#[test]
fn sealed_writer_rejects_second_commit_and_close_is_noop() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
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

/// count() excludes soft-deleted docs.
#[test]
fn count_excludes_soft_deleted_docs() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
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

/// Assert `doc_id` is the top hit for its own vector, then soft delete it
/// and assert a fresh searcher no longer returns it.
///
/// The first search caches the segment readers. Segment readers filter
/// deletions through the bitmap attached when they were loaded, so this
/// fails if readers cached before the delete created the bitmap are reused.
fn assert_soft_delete_reaches_cached_readers(index: &SegmentedFlatIndex, doc_id: u64) {
    let hits = index.searcher().unwrap().search(&query(doc_id, 1)).unwrap();
    assert_eq!(hits.results[0].doc_id, doc_id);

    index.soft_delete_document(doc_id).unwrap();
    let hits = index.searcher().unwrap().search(&query(doc_id, 5)).unwrap();
    assert!(
        hits.results.iter().all(|r| r.doc_id != doc_id),
        "doc {doc_id} was soft deleted after its readers were cached"
    );
}

/// The first soft delete creates the deletion bitmap; readers cached before
/// it must not keep serving the deleted doc.
#[test]
fn first_soft_delete_hides_doc_from_cached_readers() {
    let index = SegmentedFlatIndex::open_or_create(
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
    let index = SegmentedFlatIndex::open_or_create(
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

/// The migration must fire through the PRODUCTION factory path. A legacy
/// index always has a `metadata.json`, so it always takes the factory's
/// OPEN arm — which must be exactly where the migration fires.
#[test]
fn factory_open_path_migrates_legacy_index_and_stays_segmented() {
    use laurus::vector::index::config::VectorIndexTypeConfig;
    use laurus::vector::index::factory::VectorIndexFactory;

    let storage = storage();
    {
        let index = FlatIndex::create(
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
        VectorIndexTypeConfig::Flat(config(true)),
    )
    .unwrap();
    assert!(
        storage.file_exists("segments.json"),
        "the factory OPEN arm must migrate a legacy index"
    );
    assert!(
        !storage.file_exists("metadata.json"),
        "the stale monolithic metadata.json must be removed so factory \
         routing can never regress to the monolithic view"
    );

    commit_batch(index.as_ref(), 50..51);
    drop(index);
    let index = VectorIndexFactory::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        VectorIndexTypeConfig::Flat(config(true)),
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
/// loudly — the monolithic view would silently hide every sealed segment.
#[test]
fn factory_rejects_segmented_directory_with_flag_off() {
    use laurus::vector::index::config::VectorIndexTypeConfig;
    use laurus::vector::index::factory::VectorIndexFactory;

    let storage = storage();
    {
        let index = SegmentedFlatIndex::open_or_create(
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
        VectorIndexTypeConfig::Flat(config(false)),
    );
    assert!(
        result.is_err(),
        "a segmented directory must not open monolithically"
    );
}

/// Pure-append workloads must not grow the segment count unboundedly —
/// `maybe_auto_compact` merges a policy window once the manager's
/// threshold is exceeded.
#[test]
fn append_only_segment_count_is_bounded_by_auto_merge() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    for i in 0..101u64 {
        commit_batch(&index, i..i + 1);
    }
    let before = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".flat"))
        .count();
    assert!(before > 100, "precondition: {before} segments");

    let compacted = index.maybe_auto_compact().unwrap();
    assert!(compacted, "the segment-count bound must trigger a merge");
    let after = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".flat"))
        .count();
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

/// serde behavior of the (now on-by-default, Issue #907) flag — a config
/// missing the field deserializes to `true`; an explicit `false` is
/// preserved.
#[test]
fn segmented_flag_serde_default_and_explicit_false() {
    // A pre-#907 config = today's config with the `segmented` key removed.
    let mut value: serde_json::Value = serde_json::to_value(FlatIndexConfig::default()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("segmented")
        .expect("the flag must serialize");
    let config: FlatIndexConfig = serde_json::from_value(value).unwrap();
    assert!(
        config.segmented,
        "a config serialized before the field existed must open segmented (#907)"
    );

    let explicit_false = serde_json::to_string(&FlatIndexConfig {
        segmented: false,
        ..FlatIndexConfig::default()
    })
    .unwrap();
    let config: FlatIndexConfig = serde_json::from_str(&explicit_false).unwrap();
    assert!(
        !config.segmented,
        "an explicit `segmented: false` must be preserved"
    );
}

/// Under sustained ingest with the store-ladder cadence (compact after
/// every commit), the tiered policy keeps the segment count logarithmically
/// bounded instead of one-segment-per-commit.
#[test]
fn sustained_ingest_keeps_segment_count_tiered() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    for i in 0..100u64 {
        commit_batch(&index, i * 5..(i + 1) * 5);
        index.maybe_auto_compact().unwrap();
    }

    let segments = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".flat"))
        .count();
    assert!(
        segments <= 30,
        "tiered merging must keep the segment count bounded under \
         sustained ingest, got {segments} after 100 commits"
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

/// Adaptive refill — a live doc ranked *below a deep band of stale upsert
/// copies inside the same segment* must still surface. Flat's per-segment
/// search is exact brute force, so this pins that the SHARED (#889 PR-2)
/// containment-masking + expanding-refill layer is wired correctly for
/// Flat, not just HNSW.
#[test]
fn adaptive_refill_recovers_live_doc_behind_stale_band() {
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    // Segment A (older): the stale band (docs 1..=30, tightly clustered
    // around doc_vec(0)) AND the LIVE doc 100 (doc_vec(31), just past the
    // band) are committed together, so doc 100 sits at rank 31 *within the
    // same segment* — behind all 30 copies.
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

    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(0, 1)).unwrap();
    assert_eq!(
        results.results.first().map(|r| r.doc_id),
        Some(100),
        "the live doc behind the deep stale band must surface via the \
         expanding adaptive refill, got {:?}",
        results.results
    );
}

/// Flat-specific pin: unlike HNSW (which has an f32 rerank sidecar,
/// Issue #795), Flat's segment merge dequantizes int8 -> re-trains
/// per-segment `ScalarQuantParams` -> re-quantizes, so every merge bakes in
/// one additional generation of quantization error. Recall must still hold
/// after a forced merge — this is the pin for that accepted trade-off.
#[test]
fn recall_holds_after_forced_merge() {
    let n = 500u64;
    let storage = storage();
    let index = SegmentedFlatIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    let mut lo = 0u64;
    while lo < n {
        commit_batch(&index, lo..(lo + 50).min(n));
        lo += 50;
    }
    index.optimize().unwrap();

    let searcher = index.searcher().unwrap();
    let mut hits = 0u64;
    for id in 0..n {
        let results = searcher.search(&query(id, 10)).unwrap();
        if results.results.iter().any(|r| r.doc_id == id) {
            hits += 1;
        }
    }
    let recall = hits as f32 / n as f32;
    assert!(
        recall > 0.99,
        "self-recall must hold after a forced merge despite requantization \
         drift, got {recall:.4}"
    );
}

/// The campaign's headline number as a permanent deterministic gate — the
/// auto-commit ingest scenario writes an order of magnitude fewer
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
                SegmentedFlatIndex::open_or_create(
                    counting.clone() as Arc<dyn Storage>,
                    "vector_index",
                    config(true),
                )
                .unwrap(),
            )
        } else {
            Box::new(
                FlatIndex::create(
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
         under auto-commit ingest, got monolithic={monolithic} vs segmented={segmented}"
    );
}
