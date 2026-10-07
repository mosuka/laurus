use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::DistanceMetric;
use laurus::vector::StoredVector;
use laurus::vector::Vector;
use laurus::vector::VectorFieldConfig;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{FlatIndexConfig, IvfIndexConfig};
use laurus::vector::index::field::{FieldSearchInput, VectorFieldReader, VectorFieldWriter};
use laurus::vector::index::flat::reader::FlatVectorIndexReader;
use laurus::vector::index::flat::segmented::SegmentedFlatIndex;
use laurus::vector::index::hnsw;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::vector::index::ivf::reader::IvfIndexReader;
use laurus::vector::index::ivf::segmented::SegmentedIvfIndex;
use laurus::vector::index::segment::manager::{SegmentManager, SegmentManagerConfig};
use laurus::vector::index::segmented_field::SegmentedVectorField;
use laurus::vector::store::request::QueryVector;
use laurus::vector::{FieldOption, HnswOption};
use std::sync::Arc;

#[tokio::test]
async fn test_segmented_field_manual_merge() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Storage and Manager with small constraints
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,            // Trigger merge when > 2
        merge_factor: 2,            // Merge 2 segments at a time
        min_vectors_per_segment: 1, // Allow small segments
        ..Default::default()
    };

    let manager = Arc::new(SegmentManager::new(
        manager_config,
        storage.clone(),
        hnsw::segment::LAYOUT,
    )?);

    // 2. Setup Field
    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            m: 16, // Default or specific to test? Using defaults or standard values
            ef_construction: 200, // Standard default
            default_ef_search: None,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
            embedder: None,
            pq_codebook_path: None,
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "test_field",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    // 3. Add vectors and flush to create segments
    // Segment 1
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Segment 2
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Segment 3
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Check we have 3 segments
    let segments = manager.list_segments();
    assert_eq!(segments.len(), 3, "Should have 3 segments before merge");

    // 4. Trigger Merge
    // We expect candidates to be found because 3 > max_segments (2).
    // Policy: SimpleMergePolicy sorts by size (all same size 1). Picks 2 smallest (or first 2).

    field.perform_merge()?;

    // 5. Verify Results
    let segments_after = manager.list_segments();
    // merged 2 segments -> 1. Total: 1 (new) + 1 (remaining) = 2.
    assert_eq!(
        segments_after.len(),
        2,
        "Should have 2 segments after merge"
    );

    // Verify Stats
    let stats = field.stats()?; // Should be 3 vectors total
    assert_eq!(stats.vector_count, 3);

    Ok(())
}

/// Regression test for Issue [#660]: `SegmentedVectorField` must reuse
/// a cached `Arc<HnswIndexReader>` across queries against the same
/// segment, and must invalidate the cache entry when the segment is
/// removed via merge.
///
/// [#660]: https://github.com/mosuka/laurus/issues/660
#[tokio::test]
async fn segmented_field_reader_cache_reuses_and_invalidates()
-> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,
        merge_factor: 2,
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(
        manager_config,
        storage.clone(),
        hnsw::segment::LAYOUT,
    )?);

    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            m: 16,
            ef_construction: 200,
            default_ef_search: None,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
            embedder: None,
            pq_codebook_path: None,
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "embedding",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    // Flush three single-vector segments. After this the manager owns
    // 3 segments and the reader cache is still empty (no searches yet).
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    assert_eq!(manager.list_segments().len(), 3);
    assert!(
        field.reader_cache.is_empty(),
        "cache should be empty before the first search"
    );

    let segment_ids: Vec<String> = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .collect();

    let query_input = || FieldSearchInput {
        field: "embedding".to_string(),
        query_vectors: vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            weight: 1.0,
            fields: None,
        }],
        limit: 3,
        allowed_ids: None,
    };

    // First search: populates the cache with one entry per managed segment.
    let _first = field.search(query_input())?;
    assert_eq!(
        field.reader_cache.len(),
        3,
        "cache should hold one reader per managed segment after the first search"
    );
    for id in &segment_ids {
        assert!(
            field.reader_cache.contains(id),
            "cache should contain entry for segment {id}"
        );
    }

    // Second search: cache should still have the same three entries; no
    // additional segments managed and no eviction.
    let _second = field.search(query_input())?;
    assert_eq!(
        field.reader_cache.len(),
        3,
        "cache size should be stable across repeat searches"
    );

    // Trigger a merge — `perform_merge` removes 2 source segments and
    // adds 1 merged segment, so the cache must drop the 2 source entries
    // and report a size of 1.
    field.perform_merge()?;

    let after = manager.list_segments();
    assert_eq!(after.len(), 2, "manager should hold 2 segments after merge");

    // Identify which source ids were merged away — those are the ones
    // whose cache entries should now be gone.
    let remaining: std::collections::HashSet<String> =
        after.iter().map(|s| s.segment_id.clone()).collect();
    let merged_away: Vec<&String> = segment_ids
        .iter()
        .filter(|id| !remaining.contains(*id))
        .collect();
    assert_eq!(
        merged_away.len(),
        2,
        "merge should consume 2 source segments"
    );

    for id in merged_away {
        assert!(
            !field.reader_cache.contains(id),
            "cache entry for merged-away segment {id} must be invalidated"
        );
    }

    assert!(
        field.reader_cache.len() <= 1,
        "cache should have at most the one survivor pre-search; got {}",
        field.reader_cache.len()
    );

    Ok(())
}

/// Regression test for Issue [#790]: `SegmentedVectorField` dropped
/// `rerank_storage` (and the quantizer) when converting `HnswOption`
/// into `HnswIndexConfig`, so neither flushed segments nor merged
/// segments ever emitted the Stage-2 LRS1 sidecar (`<segment>.hnsw.f32`).
///
/// [#790]: https://github.com/mosuka/laurus/issues/790
#[tokio::test]
async fn segmented_flush_and_merge_emit_rerank_sidecar() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,
        merge_factor: 2,
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(
        manager_config,
        storage.clone(),
        hnsw::segment::LAYOUT,
    )?);

    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            rerank_storage: Some(RerankStorageKind::F32),
            ..HnswOption::default()
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "embedding",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Every flushed segment must carry its LRS1 sidecar (Issue #790:
    // the active-segment config previously dropped rerank_storage).
    let before_ids: Vec<String> = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .collect();
    assert_eq!(before_ids.len(), 3);
    for id in &before_ids {
        let sidecar = format!("{id}.hnsw.f32");
        assert!(
            storage.file_exists(&sidecar),
            "flushed segment must emit {sidecar}"
        );
    }

    field.perform_merge()?;

    // Locate the merged segment (the id that was not present before).
    let after = manager.list_segments();
    assert_eq!(after.len(), 2);
    let merged_id = after
        .iter()
        .map(|s| s.segment_id.clone())
        .find(|id| !before_ids.contains(id))
        .expect("merge must create a new segment");

    let merged_sidecar = format!("{merged_id}.hnsw.f32");
    assert!(
        storage.file_exists(&merged_sidecar),
        "merged segment must re-emit {merged_sidecar} (Issue #790: the \
         merge-engine config previously dropped rerank_storage)"
    );

    // The merged sidecar must load into the rerank pool (Eager mode)
    // and contain exactly the two merged vectors — the anti-fallback
    // guard from the #788 test.
    let reader = HnswIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &merged_id,
        DistanceMetric::Cosine,
    )?;
    let pool = reader
        .rerank_storage()
        .expect("merged sidecar must load into the rerank pool");
    let merged_doc_count = [1u64, 2, 3]
        .iter()
        .filter(|doc_id| pool.contains(**doc_id, "embedding"))
        .count();
    assert_eq!(
        merged_doc_count, 2,
        "the merged pool must contain exactly the 2 merged vectors"
    );

    Ok(())
}

/// Regression test for the latent merge-config bug fixed with Issue
/// [#790]: `perform_merge_with_policy` built its `HnswIndexConfig`
/// without `distance_metric` (default Cosine) and without
/// `normalize_vectors` (default `true`), so merging a **Euclidean**
/// segmented field silently L2-normalized the merged vectors. The
/// merged vectors must keep their original (non-unit) norms.
///
/// [#790]: https://github.com/mosuka/laurus/issues/790
#[tokio::test]
async fn segmented_merge_keeps_vectors_unnormalized_for_euclidean()
-> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,
        merge_factor: 2,
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(
        manager_config,
        storage.clone(),
        hnsw::segment::LAYOUT,
    )?);

    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            m: 16,
            ef_construction: 200,
            rerank_storage: Some(RerankStorageKind::F32),
            ..HnswOption::default()
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "embedding",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    // All vectors have norms far from 1 so an accidental L2
    // normalization during merge is unambiguously detectable.
    let originals: [(u64, [f32; 4]); 3] = [
        (1, [2.0, 0.0, 0.0, 0.0]),
        (2, [0.0, 3.0, 0.0, 0.0]),
        (3, [0.0, 0.0, 4.0, 0.0]),
    ];
    for (doc_id, vec) in &originals {
        field
            .add_stored_vector(*doc_id, &StoredVector::new(vec.to_vec()), 0)
            .await?;
        field.flush().await?;
    }

    let before_ids: Vec<String> = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .collect();
    field.perform_merge()?;

    let merged_id = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .find(|id| !before_ids.contains(id))
        .expect("merge must create a new segment");

    let reader = HnswIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &merged_id,
        DistanceMetric::Euclidean,
    )?;
    let pool = reader
        .rerank_storage()
        .expect("merged sidecar must load into the rerank pool");

    // Whichever two of the three docs were merged, their vectors must
    // keep the original norms (2/3/4) within int8 dequantization
    // tolerance — a normalized vector would have norm 1.0.
    let mut checked = 0;
    for (doc_id, original) in &originals {
        if let Some(slice) = pool.get_f32_slice(*doc_id, "embedding") {
            let norm: f32 = slice.iter().map(|v| v * v).sum::<f32>().sqrt();
            let expected: f32 = original.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - expected).abs() < 0.1,
                "merged vector for doc {doc_id} must keep its original norm \
                 {expected} (got {norm}); norm 1.0 means the merge config \
                 normalized a Euclidean field (Issue #790 latent bug)"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 2, "the merged pool must contain 2 of the 3 docs");

    Ok(())
}

/// Issue [#795]: a segment merge must rebuild the merged `.hnsw.f32`
/// rerank sidecar from the source segments' **original f32** sidecars,
/// not from int8-dequantized values. Otherwise each merge bakes one
/// round of scalar-quantization error (~`scale/2`, on the order of
/// `1e-3` for these vectors) into the "lossless" rerank payload.
///
/// The test uses a Euclidean field (so #794 does not normalize the
/// stored vectors) with components that do not land on the int8
/// quantization grid, then asserts the merged pool's f32 equals the
/// original input within `f32::EPSILON` — which only holds if the merge
/// read the source sidecar rather than the dequantized int8 segment.
///
/// [#795]: https://github.com/mosuka/laurus/issues/795
#[tokio::test]
async fn segmented_merge_preserves_rerank_sidecar_f32_losslessly()
-> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,
        merge_factor: 2,
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(
        manager_config,
        storage.clone(),
        hnsw::segment::LAYOUT,
    )?);

    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            m: 16,
            ef_construction: 200,
            rerank_storage: Some(RerankStorageKind::F32),
            ..HnswOption::default()
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "embedding",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    // Each segment shares one (offset, scale) computed from all of its
    // vectors. Pairing each probe vector with a large-magnitude anchor
    // widens the int8 range so the probe's fractional components land
    // well off the quantization grid (dequant error ~0.05, far above
    // f32::EPSILON). With int8-rebuilt sidecars the merged probe would
    // be that coarse; only reading the source f32 sidecar preserves it.
    let probes: [(u64, [f32; 4]); 3] = [
        (1, [0.137, 0.642, 0.319, 0.808]),
        (2, [0.251, 0.563, 0.174, 0.926]),
        (3, [0.488, 0.071, 0.655, 0.302]),
    ];
    let anchor = [40.0_f32, 40.0, 40.0, 40.0];
    for (i, (doc_id, vec)) in probes.iter().enumerate() {
        field
            .add_stored_vector(*doc_id, &StoredVector::new(vec.to_vec()), 0)
            .await?;
        field
            .add_stored_vector(100 + i as u64, &StoredVector::new(anchor.to_vec()), 0)
            .await?;
        field.flush().await?;
    }

    let before_ids: Vec<String> = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .collect();
    field.perform_merge()?;

    let merged_id = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .find(|id| !before_ids.contains(id))
        .expect("merge must create a new segment");

    let reader = HnswIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &merged_id,
        DistanceMetric::Euclidean,
    )?;
    let pool = reader
        .rerank_storage()
        .expect("merged sidecar must load into the rerank pool");

    // Whichever two segments were merged, the probe vectors they
    // contain must equal the original input exactly (the source sidecar
    // carried the original f32). int8-dequantized values would differ
    // by ~0.05 here.
    let mut checked = 0;
    for (doc_id, original) in &probes {
        if let Some(slice) = pool.get_f32_slice(*doc_id, "embedding") {
            for (i, (got, want)) in slice.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= f32::EPSILON,
                    "probe doc {doc_id} component {i}: merged f32 {got} != original {want} \
                     (diff {:.2e}); a coarse difference means the merge rebuilt the \
                     sidecar from int8-dequantized values (Issue #795)",
                    (got - want).abs()
                );
            }
            checked += 1;
        }
    }
    assert_eq!(
        checked, 2,
        "the merged pool must contain the probes from the 2 merged segments"
    );

    Ok(())
}

/// Issue #1348: `flat::segment::LAYOUT` previously declared no sidecar
/// suffix, so `SegmentManager::delete_segment_files` and the orphan sweep
/// never recognized a merged-away source segment's `.flat.f32` rerank
/// sidecar — it leaked on every merge. `SegmentedVectorField` is HNSW-only,
/// so this drives `SegmentedFlatIndex` directly through `VectorIndex`,
/// mirroring `vector_segmented_flat_test.rs`'s style.
#[test]
fn flat_merge_deletes_source_rerank_sidecars() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = FlatIndexConfig {
        dimension: 4,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config,
    )
    .unwrap();

    for (doc_id, vec) in [
        (1u64, vec![1.0, 0.0, 0.0, 0.0]),
        (2, vec![0.0, 1.0, 0.0, 0.0]),
        (3, vec![0.0, 0.0, 1.0, 0.0]),
    ] {
        let mut writer = index.writer().unwrap();
        writer
            .add_vectors(vec![(doc_id, "embedding".to_string(), Vector::new(vec))])
            .unwrap();
        writer.commit().unwrap();
    }

    let before_sidecars: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".flat.f32"))
        .collect();
    assert_eq!(
        before_sidecars.len(),
        3,
        "each committed segment must emit its own sidecar"
    );

    index.optimize().unwrap();

    let after_main: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".flat"))
        .collect();
    let after_sidecars: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".flat.f32"))
        .collect();
    assert_eq!(
        after_main.len(),
        1,
        "optimize must force-merge to one segment"
    );
    assert_eq!(
        after_sidecars.len(),
        1,
        "the 3 source sidecars must not survive the merge, only the \
         merged segment's own"
    );
    for sidecar in &before_sidecars {
        assert!(
            !storage.file_exists(sidecar),
            "merge must delete the source segment's rerank sidecar \
             ({sidecar}), not leak it forever"
        );
    }

    // Acceptance criterion: no `.f32` files remain after merge + reopen --
    // re-open to also exercise the orphan sweep (`SegmentManager::new` runs
    // it unconditionally) on the manifest the merge just published.
    drop(index);
    let reopened = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        FlatIndexConfig {
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: false,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        },
    )
    .unwrap();
    drop(reopened);
    let sidecars_after_reopen: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".flat.f32"))
        .collect();
    assert_eq!(
        sidecars_after_reopen.len(),
        1,
        "reopening must not resurrect or leave behind any source sidecar"
    );
    for sidecar in &before_sidecars {
        assert!(!storage.file_exists(sidecar));
    }
}

/// Issue #1348: `flat::segment::merge_engine::MergeEngine` read every
/// source vector through `vector_iterator()` (int8-dequantized) and never
/// consulted `rerank_storage()`, so a merge rebuilt the sidecar from
/// already-quantized values. Mirrors HNSW's
/// `segmented_merge_preserves_rerank_sidecar_f32_losslessly` above, driven
/// through `SegmentedFlatIndex` (`SegmentedVectorField` is HNSW-only).
#[test]
fn flat_merge_preserves_rerank_sidecar_f32_losslessly() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = FlatIndexConfig {
        dimension: 4,
        distance_metric: DistanceMetric::Euclidean,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let index = SegmentedFlatIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config,
    )
    .unwrap();

    // Same anchor trick as the HNSW test: pairing each probe with a
    // large-magnitude vector widens the int8 range so the probe's
    // fractional components land well off the quantization grid
    // (dequant error ~0.05, far above f32::EPSILON).
    let probes: [(u64, [f32; 4]); 3] = [
        (1, [0.137, 0.642, 0.319, 0.808]),
        (2, [0.251, 0.563, 0.174, 0.926]),
        (3, [0.488, 0.071, 0.655, 0.302]),
    ];
    let anchor = [40.0_f32, 40.0, 40.0, 40.0];
    for (i, (doc_id, vec)) in probes.iter().enumerate() {
        let mut writer = index.writer().unwrap();
        writer
            .add_vectors(vec![
                (*doc_id, "embedding".to_string(), Vector::new(vec.to_vec())),
                (
                    100 + i as u64,
                    "embedding".to_string(),
                    Vector::new(anchor.to_vec()),
                ),
            ])
            .unwrap();
        writer.commit().unwrap();
    }

    index.optimize().unwrap();

    let merged_file = storage
        .list_files()
        .unwrap()
        .into_iter()
        .find(|f| f.ends_with(".flat"))
        .expect("optimize must leave exactly one merged segment");
    let merged_id = merged_file.strip_suffix(".flat").unwrap().to_string();

    let reader = FlatVectorIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &merged_id,
        DistanceMetric::Euclidean,
    )
    .unwrap();
    let pool = reader
        .rerank_storage()
        .expect("merged sidecar must load into the rerank pool");

    for (doc_id, original) in &probes {
        let slice = pool
            .get_f32_slice(*doc_id, "embedding")
            .unwrap_or_else(|| panic!("doc {doc_id} must be in the merged sidecar"));
        for (i, (got, want)) in slice.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() <= f32::EPSILON,
                "probe doc {doc_id} component {i}: merged f32 {got} != original {want} \
                 (diff {:.2e}); a coarse difference means the merge rebuilt the \
                 sidecar from int8-dequantized values (Issue #1348)",
                (got - want).abs()
            );
        }
    }
}

/// Issue #1348: `ivf::segment::LAYOUT` equivalent of
/// `flat_merge_deletes_source_rerank_sidecars` above.
#[test]
fn ivf_merge_deletes_source_rerank_sidecars() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = IvfIndexConfig {
        dimension: 4,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config,
    )
    .unwrap();

    for (doc_id, vec) in [
        (1u64, vec![1.0, 0.0, 0.0, 0.0]),
        (2, vec![0.0, 1.0, 0.0, 0.0]),
        (3, vec![0.0, 0.0, 1.0, 0.0]),
    ] {
        let mut writer = index.writer().unwrap();
        writer
            .add_vectors(vec![(doc_id, "embedding".to_string(), Vector::new(vec))])
            .unwrap();
        writer.commit().unwrap();
    }

    let before_sidecars: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".ivf.f32"))
        .collect();
    assert_eq!(
        before_sidecars.len(),
        3,
        "each committed segment must emit its own sidecar"
    );

    index.optimize().unwrap();

    let after_main: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".ivf"))
        .collect();
    let after_sidecars: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".ivf.f32"))
        .collect();
    assert_eq!(
        after_main.len(),
        1,
        "optimize must force-merge to one segment"
    );
    assert_eq!(
        after_sidecars.len(),
        1,
        "the 3 source sidecars must not survive the merge, only the \
         merged segment's own"
    );
    for sidecar in &before_sidecars {
        assert!(
            !storage.file_exists(sidecar),
            "merge must delete the source segment's rerank sidecar \
             ({sidecar}), not leak it forever"
        );
    }

    // Acceptance criterion: no `.f32` files remain after merge + reopen --
    // re-open to also exercise the orphan sweep (`SegmentManager::new` runs
    // it unconditionally) on the manifest the merge just published.
    drop(index);
    let reopened = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        IvfIndexConfig {
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: false,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        },
    )
    .unwrap();
    drop(reopened);
    let sidecars_after_reopen: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".ivf.f32"))
        .collect();
    assert_eq!(
        sidecars_after_reopen.len(),
        1,
        "reopening must not resurrect or leave behind any source sidecar"
    );
    for sidecar in &before_sidecars {
        assert!(!storage.file_exists(sidecar));
    }
}

/// Issue #1348: `ivf::segment::merge_engine::MergeEngine` equivalent of
/// `flat_merge_preserves_rerank_sidecar_f32_losslessly` above.
#[test]
fn ivf_merge_preserves_rerank_sidecar_f32_losslessly() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = IvfIndexConfig {
        dimension: 4,
        distance_metric: DistanceMetric::Euclidean,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let index = SegmentedIvfIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config,
    )
    .unwrap();

    let probes: [(u64, [f32; 4]); 3] = [
        (1, [0.137, 0.642, 0.319, 0.808]),
        (2, [0.251, 0.563, 0.174, 0.926]),
        (3, [0.488, 0.071, 0.655, 0.302]),
    ];
    let anchor = [40.0_f32, 40.0, 40.0, 40.0];
    for (i, (doc_id, vec)) in probes.iter().enumerate() {
        let mut writer = index.writer().unwrap();
        writer
            .add_vectors(vec![
                (*doc_id, "embedding".to_string(), Vector::new(vec.to_vec())),
                (
                    100 + i as u64,
                    "embedding".to_string(),
                    Vector::new(anchor.to_vec()),
                ),
            ])
            .unwrap();
        writer.commit().unwrap();
    }

    index.optimize().unwrap();

    let merged_file = storage
        .list_files()
        .unwrap()
        .into_iter()
        .find(|f| f.ends_with(".ivf"))
        .expect("optimize must leave exactly one merged segment");
    let merged_id = merged_file.strip_suffix(".ivf").unwrap().to_string();

    let reader = IvfIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &merged_id,
        DistanceMetric::Euclidean,
    )
    .unwrap();
    let pool = reader
        .rerank_storage()
        .expect("merged sidecar must load into the rerank pool");

    for (doc_id, original) in &probes {
        let slice = pool
            .get_f32_slice(*doc_id, "embedding")
            .unwrap_or_else(|| panic!("doc {doc_id} must be in the merged sidecar"));
        for (i, (got, want)) in slice.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() <= f32::EPSILON,
                "probe doc {doc_id} component {i}: merged f32 {got} != original {want} \
                 (diff {:.2e}); a coarse difference means the merge rebuilt the \
                 sidecar from int8-dequantized values (Issue #1348)",
                (got - want).abs()
            );
        }
    }
}

/// Build a Cosine-metric `SegmentedVectorField` over in-memory storage
/// for the active-segment (NRT, unflushed) search tests (Issue #640).
///
/// # Arguments
///
/// * `dimension` - Vector dimension for the field.
///
/// # Returns
///
/// The field plus its backing `SegmentManager` (kept alive by the caller).
fn cosine_field(
    dimension: usize,
) -> Result<(SegmentedVectorField, Arc<SegmentManager>), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let manager = Arc::new(SegmentManager::new(
        SegmentManagerConfig {
            max_segments: 8,
            merge_factor: 2,
            min_vectors_per_segment: 1,
            ..Default::default()
        },
        storage.clone(),
        hnsw::segment::LAYOUT,
    )?);
    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension,
            distance: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            default_ef_search: None,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
            embedder: None,
            pq_codebook_path: None,
        })),
        lexical: None,
    };
    let field = SegmentedVectorField::create(
        "embedding",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;
    Ok((field, manager))
}

/// One-query search helper for the active-segment tests.
fn top_k_input(query: Vec<f32>, limit: usize) -> FieldSearchInput {
    FieldSearchInput {
        field: "embedding".to_string(),
        query_vectors: vec![QueryVector {
            vector: Vector::new(query),
            weight: 1.0,
            fields: None,
        }],
        limit,
        allowed_ids: None,
    }
}

/// Issue #640: searching BEFORE any flush must return correctly ranked
/// hits from the active (unflushed) segment. Prior to this test the
/// active brute-force path had no direct coverage — every existing test
/// flushed before searching.
#[tokio::test]
async fn search_before_flush_returns_active_hits() -> Result<(), Box<dyn std::error::Error>> {
    let (field, _manager) = cosine_field(4)?;

    // Cosine similarities against query [1, 0, 0, 0]:
    // doc 1 -> 1.0 (identical direction)
    // doc 2 -> 0.8 ([0.8, 0.6] direction)
    // doc 3 -> 0.0 (orthogonal)
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.8, 0.6, 0.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;

    // NO flush: hits must come from the active segment scan.
    let results = field.search(top_k_input(vec![1.0, 0.0, 0.0, 0.0], 2))?;
    let ids: Vec<u64> = results.hits.iter().map(|h| h.doc_id).collect();
    assert_eq!(
        ids,
        vec![1, 2],
        "top-2 must be doc 1 (cos 1.0) then doc 2 (cos 0.8), got {ids:?}"
    );
    assert!(
        results.hits[0].score > results.hits[1].score,
        "scores must be ranked descending: {} vs {}",
        results.hits[0].score,
        results.hits[1].score
    );
    Ok(())
}

/// Issue #640: hits from the active (unflushed) segment must merge with
/// hits from managed (flushed) segments in one search.
#[tokio::test]
async fn search_merges_active_and_managed_hits() -> Result<(), Box<dyn std::error::Error>> {
    let (field, _manager) = cosine_field(4)?;

    // doc 1 goes to a managed segment via flush...
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    // ...doc 2 stays in the active segment (no flush).
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.9, 0.1, 0.0, 0.0]), 0)
        .await?;

    let results = field.search(top_k_input(vec![1.0, 0.0, 0.0, 0.0], 3))?;
    let mut ids: Vec<u64> = results.hits.iter().map(|h| h.doc_id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2],
        "search must merge managed (doc 1) and active (doc 2) hits, got {ids:?}"
    );
    Ok(())
}

/// Issue #640: the active-segment scan must stay correct above the
/// parallel-scan threshold (2048 candidates), where the rayon path runs.
/// A single distinguished best match keeps the assertion tie-free.
#[tokio::test]
async fn search_active_segment_above_parallel_threshold() -> Result<(), Box<dyn std::error::Error>>
{
    let (field, _manager) = cosine_field(4)?;

    // 2500 vectors (> PARALLEL_SCAN_THRESHOLD = 2048), all orthogonal to
    // the query except doc 777, the unique best match.
    for doc_id in 1..=2500u64 {
        let v = if doc_id == 777 {
            vec![1.0, 0.0, 0.0, 0.0]
        } else {
            vec![0.0, 1.0, 0.0, 0.0]
        };
        field
            .add_stored_vector(doc_id, &StoredVector::new(v), 0)
            .await?;
    }

    let results = field.search(top_k_input(vec![1.0, 0.0, 0.0, 0.0], 1))?;
    assert_eq!(results.hits.len(), 1, "top-1 must return exactly one hit");
    assert_eq!(
        results.hits[0].doc_id, 777,
        "the unique aligned vector must win the unflushed scan"
    );
    Ok(())
}
