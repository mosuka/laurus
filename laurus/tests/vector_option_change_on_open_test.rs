//! Issue #1547: a vector field opens only under the dimension and distance
//! its files were written with, and an index kind that reads them.
//!
//! The CLI, the server and the bindings read the schema from `schema.toml`,
//! which a user can edit while the index is closed, and `Engine::new` opens
//! each field under the schema it is given. A field opened under another
//! dimension used to panic or fail on every search, under another kind it
//! failed every search with "File not found", and under another distance it
//! returned wrong scores; a put and a commit under the new option still
//! succeeded and mixed segments of both into the field. The field's
//! `segments.json` now records the three, and the open fails instead.
//! Every other option, such as the quantizer, may still change.
//!
//! Issue #1579: HNSW, Flat and IVF read each other's segments, so a field
//! opens under another of the three; `engine_update_field_kind_test.rs`
//! tests what it then holds. MultiVector reads only its own.
//!
//! Issue #1536: `update_field` changes the dimension, the distance, and the
//! kind to or from MultiVector as a Destructive change, which recreates the
//! field empty.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use tempfile::TempDir;

use laurus::storage::Storage;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::manifest::{load_checksummed_json, save_checksummed_json};
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::{
    Document, Engine, FieldChangeKind, FieldOption, FlatOption, HnswOption, IvfOption, LaurusError,
    MultiVectorOption, MultiVectorStorage, QuantizationMethod, QueryVector, RerankStorageKind,
    Schema, SearchRequestBuilder, TextOption, UpdateFieldOptions, VectorSearchQuery,
};

const DIM: usize = 8;

const MANIFEST: &str = "vector/vec/segments.json";

const REINDEX: UpdateFieldOptions = UpdateFieldOptions {
    reindex: true,
    dry_run: false,
};

const DRY_RUN: UpdateFieldOptions = UpdateFieldOptions {
    reindex: false,
    dry_run: true,
};

fn flat(dimension: usize) -> FlatOption {
    FlatOption::new(dimension)
}

fn hnsw(dimension: usize) -> HnswOption {
    HnswOption::new(dimension)
}

fn ivf(dimension: usize) -> IvfOption {
    IvfOption::new(dimension).n_clusters(2).n_probe(2)
}

fn multi_vector(dimension: usize) -> MultiVectorOption {
    MultiVectorOption::new(dimension)
}

/// A schema with `option` as field `vec`, and a text field `title`.
fn schema_with(option: FieldOption) -> Schema {
    let mut schema = Schema::new();
    schema.fields.insert("vec".to_string(), option);
    schema.fields.insert(
        "title".to_string(),
        FieldOption::Text(TextOption::default()),
    );
    schema
}

fn create_storage(dir: &TempDir) -> Arc<dyn Storage> {
    let config = StorageConfig::File(FileStorageConfig::new(dir.path()));
    StorageFactory::create(config).unwrap()
}

fn open_storage(dir: &TempDir) -> Arc<dyn Storage> {
    let config = StorageConfig::File(FileStorageConfig::new(dir.path()));
    StorageFactory::open(config).unwrap()
}

fn vector(seed: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|i| ((seed * 7 + i) % 11) as f32 + 1.0)
        .collect()
}

fn dimension_of(option: &FieldOption) -> usize {
    option.to_vector().unwrap().dimension()
}

/// Document `seed`, with a vector or, for a multi-vector field, two token
/// vectors of `option`'s dimension in `vec`.
fn doc(option: &FieldOption, seed: usize) -> Document {
    let dimension = dimension_of(option);
    let builder = Document::builder().add_text("title", "rust");
    match option {
        FieldOption::MultiVector(_) => builder.add_vector_array(
            "vec",
            vec![vector(seed, dimension), vector(seed + 1, dimension)],
        ),
        _ => builder.add_vector("vec", vector(seed, dimension)),
    }
    .build()
}

/// Create an index whose field `vec` is `option`, with eight documents in
/// one commit.
async fn create_index(dir: &TempDir, option: &FieldOption) {
    let engine = Engine::new(create_storage(dir), schema_with(option.clone()))
        .await
        .unwrap();
    for i in 0..8 {
        engine
            .put_document(&format!("d{i}"), doc(option, i))
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
}

/// The hits, with their scores, of a search of `vec` for the vector of
/// seed 1.
async fn search(engine: &Engine, dimension: usize) -> Vec<(String, f32)> {
    try_search(engine, dimension).await.unwrap()
}

/// [`search`], returning the error of a failed search.
async fn try_search(engine: &Engine, dimension: usize) -> laurus::Result<Vec<(String, f32)>> {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vector(1, dimension)),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]))
        .limit(20)
        .build();
    Ok(engine
        .search(request)
        .await?
        .into_iter()
        .map(|hit| (hit.id, hit.score))
        .collect())
}

/// Every file under the field's directory, with its bytes.
fn field_files(dir: &TempDir) -> BTreeMap<String, Vec<u8>> {
    fn collect(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(root, &path, out);
            } else {
                let name = path.strip_prefix(root).unwrap().display().to_string();
                out.insert(name, std::fs::read(&path).unwrap());
            }
        }
    }
    let mut files = BTreeMap::new();
    collect(dir.path(), &dir.path().join("vector/vec"), &mut files);
    files
}

fn assert_rejected(result: laurus::Result<Engine>, reason: &str) {
    let expected = format!("vector field 'vec': {reason}");
    match result {
        Err(LaurusError::InvalidArgument(message)) => assert_eq!(message, expected),
        Err(other) => panic!("expected InvalidArgument({expected:?}), got {other:?}"),
        Ok(_) => panic!("expected InvalidArgument({expected:?}), but the index opened"),
    }
}

/// Each change of the dimension or the distance, or of the kind to or from
/// MultiVector, fails the open, changes none of the field's files, and the
/// index still opens under the option it was written with.
#[tokio::test(flavor = "multi_thread")]
async fn reopening_with_another_kind_dimension_or_distance_fails_at_the_open() {
    let cosine = |kind_and_dimension: &str| {
        format!("its files were written with {kind_and_dimension} and distance Cosine")
    };
    let cases: Vec<(FieldOption, FieldOption, String)> = vec![
        (
            FieldOption::Flat(flat(DIM)),
            FieldOption::MultiVector(multi_vector(DIM)),
            format!(
                "{}, so it cannot be opened with kind MultiVector",
                cosine("kind Flat, dimension 8")
            ),
        ),
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::MultiVector(multi_vector(DIM)),
            format!(
                "{}, so it cannot be opened with kind MultiVector",
                cosine("kind Hnsw, dimension 8")
            ),
        ),
        (
            FieldOption::Ivf(ivf(DIM)),
            FieldOption::MultiVector(multi_vector(DIM)),
            format!(
                "{}, so it cannot be opened with kind MultiVector",
                cosine("kind Ivf, dimension 8")
            ),
        ),
        (
            FieldOption::MultiVector(multi_vector(DIM)),
            FieldOption::Flat(flat(DIM)),
            format!(
                "{}, so it cannot be opened with kind Flat",
                cosine("kind MultiVector, dimension 8")
            ),
        ),
        (
            FieldOption::MultiVector(multi_vector(DIM)),
            FieldOption::Hnsw(hnsw(DIM)),
            format!(
                "{}, so it cannot be opened with kind Hnsw",
                cosine("kind MultiVector, dimension 8")
            ),
        ),
        (
            FieldOption::MultiVector(multi_vector(DIM)),
            FieldOption::Ivf(ivf(DIM)),
            format!(
                "{}, so it cannot be opened with kind Ivf",
                cosine("kind MultiVector, dimension 8")
            ),
        ),
        // Another of the three kinds that read each other does not count
        // among what the open rejects.
        (
            FieldOption::Ivf(ivf(DIM)),
            FieldOption::Hnsw(hnsw(4)),
            format!(
                "{}, so it cannot be opened with dimension 4",
                cosine("kind Ivf, dimension 8")
            ),
        ),
        (
            FieldOption::Flat(flat(DIM)),
            FieldOption::Flat(flat(4)),
            format!(
                "{}, so it cannot be opened with dimension 4",
                cosine("kind Flat, dimension 8")
            ),
        ),
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::Hnsw(hnsw(64)),
            format!(
                "{}, so it cannot be opened with dimension 64",
                cosine("kind Hnsw, dimension 8")
            ),
        ),
        (
            FieldOption::Ivf(ivf(DIM)),
            FieldOption::Ivf(ivf(4)),
            format!(
                "{}, so it cannot be opened with dimension 4",
                cosine("kind Ivf, dimension 8")
            ),
        ),
        (
            FieldOption::MultiVector(multi_vector(DIM)),
            FieldOption::MultiVector(multi_vector(4)),
            format!(
                "{}, so it cannot be opened with dimension 4",
                cosine("kind MultiVector, dimension 8")
            ),
        ),
        (
            FieldOption::Flat(flat(DIM)),
            FieldOption::Flat(flat(DIM).distance(DistanceMetric::DotProduct)),
            format!(
                "{}, so it cannot be opened with distance DotProduct",
                cosine("kind Flat, dimension 8")
            ),
        ),
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::Hnsw(hnsw(DIM).distance(DistanceMetric::Euclidean)),
            format!(
                "{}, so it cannot be opened with distance Euclidean",
                cosine("kind Hnsw, dimension 8")
            ),
        ),
        (
            FieldOption::Ivf(ivf(DIM)),
            FieldOption::Ivf(ivf(DIM).distance(DistanceMetric::Euclidean)),
            format!(
                "{}, so it cannot be opened with distance Euclidean",
                cosine("kind Ivf, dimension 8")
            ),
        ),
        (
            FieldOption::MultiVector(multi_vector(DIM)),
            FieldOption::MultiVector(multi_vector(DIM).distance(DistanceMetric::DotProduct)),
            format!(
                "{}, so it cannot be opened with distance DotProduct",
                cosine("kind MultiVector, dimension 8")
            ),
        ),
        (
            FieldOption::Flat(flat(DIM)),
            FieldOption::Hnsw(hnsw(4).distance(DistanceMetric::Euclidean)),
            format!(
                "{}, so it cannot be opened with dimension 4 and distance Euclidean",
                cosine("kind Flat, dimension 8")
            ),
        ),
        (
            FieldOption::Flat(flat(DIM)),
            FieldOption::MultiVector(multi_vector(4).distance(DistanceMetric::DotProduct)),
            format!(
                "{}, so it cannot be opened with kind MultiVector, dimension 4 and distance \
                 DotProduct",
                cosine("kind Flat, dimension 8")
            ),
        ),
    ];
    for (written, changed, reason) in cases {
        let dir = TempDir::new().unwrap();
        create_index(&dir, &written).await;
        let files = field_files(&dir);

        // Checked before any search: under another dimension a search
        // would hit a `debug_assert!` first.
        assert_rejected(
            Engine::new(open_storage(&dir), schema_with(changed)).await,
            &reason,
        );
        assert_eq!(field_files(&dir), files, "{reason}");

        let engine = Engine::new(open_storage(&dir), schema_with(written.clone()))
            .await
            .unwrap();
        if !matches!(written, FieldOption::MultiVector(_)) {
            assert_eq!(search(&engine, DIM).await.len(), 8, "{reason}");
        }
    }
}

/// A change of an option the files do not depend on still opens the field,
/// which finds what it found before and takes new documents.
#[tokio::test(flavor = "multi_thread")]
async fn reopening_with_another_quantizer_or_graph_parameter_keeps_the_field() {
    let cases: Vec<(FieldOption, FieldOption)> = vec![
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::Hnsw(hnsw(DIM).quantizer(QuantizationMethod::Float16)),
        ),
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::Hnsw(
                hnsw(DIM).quantizer(QuantizationMethod::ProductQuantization { subvector_count: 2 }),
            ),
        ),
        (
            FieldOption::Flat(flat(DIM)),
            FieldOption::Flat(flat(DIM).quantizer(QuantizationMethod::Float16)),
        ),
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::Hnsw(hnsw(DIM).m(4).ef_construction(50)),
        ),
        (
            FieldOption::Hnsw(hnsw(DIM)),
            FieldOption::Hnsw(hnsw(DIM).rerank_storage(RerankStorageKind::F32)),
        ),
        // A segment keeps the centroids it was written with; `n_probe` is
        // left alone, since probing fewer clusters changes the hits.
        (
            FieldOption::Ivf(ivf(DIM)),
            FieldOption::Ivf(ivf(DIM).n_clusters(3)),
        ),
        (
            FieldOption::MultiVector(multi_vector(DIM)),
            FieldOption::MultiVector(multi_vector(DIM).storage(MultiVectorStorage::F16)),
        ),
    ];
    for (written, changed) in cases {
        let label = format!("{written:?} -> {changed:?}");
        let dir = TempDir::new().unwrap();
        create_index(&dir, &written).await;
        let multi_vector = matches!(written, FieldOption::MultiVector(_));

        let engine = Engine::new(open_storage(&dir), schema_with(written.clone()))
            .await
            .unwrap();
        let before = if multi_vector {
            Vec::new()
        } else {
            search(&engine, DIM).await
        };
        drop(engine);

        let engine = Engine::new(open_storage(&dir), schema_with(changed.clone()))
            .await
            .unwrap();
        if !multi_vector {
            assert_eq!(search(&engine, DIM).await, before, "{label}");
        }
        engine.put_document("d9", doc(&changed, 9)).await.unwrap();
        engine.commit().await.unwrap();
        drop(engine);

        let engine = Engine::new(open_storage(&dir), schema_with(changed))
            .await
            .unwrap();
        if !multi_vector {
            assert_eq!(search(&engine, DIM).await.len(), 9, "{label}");
        }
    }
}

/// A field that holds no vector yet reopens under another dimension, whose
/// vectors it then takes and records.
#[tokio::test(flavor = "multi_thread")]
async fn a_field_with_no_vectors_reopens_with_another_dimension() {
    let dir = TempDir::new().unwrap();
    let engine = Engine::new(
        create_storage(&dir),
        schema_with(FieldOption::Flat(flat(DIM))),
    )
    .await
    .unwrap();
    let text_only = Document::builder().add_text("title", "rust").build();
    engine.put_document("t0", text_only).await.unwrap();
    engine.commit().await.unwrap();
    drop(engine);
    // The commit saved a manifest with no segment for the field.
    assert!(dir.path().join(MANIFEST).exists());

    let fixed = FieldOption::Flat(flat(4));
    let engine = Engine::new(open_storage(&dir), schema_with(fixed.clone()))
        .await
        .unwrap();
    engine.put_document("d0", doc(&fixed, 0)).await.unwrap();
    engine.commit().await.unwrap();
    drop(engine);

    let engine = Engine::new(open_storage(&dir), schema_with(fixed))
        .await
        .unwrap();
    assert_eq!(search(&engine, 4).await.len(), 1);
    drop(engine);
    assert_rejected(
        Engine::new(
            open_storage(&dir),
            schema_with(FieldOption::Flat(flat(DIM))),
        )
        .await,
        "its files were written with kind Flat, dimension 4 and distance Cosine, so it cannot \
         be opened with dimension 8",
    );
}

/// The field's manifest, as JSON.
fn manifest(dir: &TempDir) -> serde_json::Value {
    load_checksummed_json(open_storage(dir).as_ref(), MANIFEST, None)
        .unwrap()
        .unwrap()
        .0
}

/// An index written before the record existed is checked against what its
/// files prove (its segments' kind and dimension), opens otherwise, and has
/// its record written by its next commit that changes the index, even one
/// that leaves the field alone.
#[tokio::test(flavor = "multi_thread")]
async fn an_index_without_a_record_is_checked_against_its_files() {
    let dir = TempDir::new().unwrap();
    let written = FieldOption::Hnsw(hnsw(DIM));
    create_index(&dir, &written).await;
    let mut unrecorded = manifest(&dir);
    assert_eq!(
        unrecorded["field"],
        serde_json::json!({"kind": "hnsw", "dimension": 8, "distance": "Cosine"})
    );
    unrecorded.as_object_mut().unwrap().remove("field");
    save_checksummed_json(open_storage(&dir).as_ref(), MANIFEST, None, &unrecorded).unwrap();

    assert_rejected(
        Engine::new(open_storage(&dir), schema_with(FieldOption::Hnsw(hnsw(4)))).await,
        "its files were written with dimension 8, so it cannot be opened with dimension 4",
    );
    // The dimension is read from the header of the kind that was written.
    for other in [FieldOption::Flat(flat(4)), FieldOption::Ivf(ivf(4))] {
        assert_rejected(
            Engine::new(open_storage(&dir), schema_with(other)).await,
            "its files were written with dimension 8, so it cannot be opened with dimension 4",
        );
    }
    assert_rejected(
        Engine::new(
            open_storage(&dir),
            schema_with(FieldOption::MultiVector(multi_vector(DIM))),
        )
        .await,
        "its files were written with kind Hnsw, so it cannot be opened with kind MultiVector",
    );
    // No file records the distance, so a change of it opens.
    let euclidean = FieldOption::Hnsw(hnsw(DIM).distance(DistanceMetric::Euclidean));
    drop(
        Engine::new(open_storage(&dir), schema_with(euclidean.clone()))
            .await
            .unwrap(),
    );
    assert!(manifest(&dir).get("field").is_none());

    // A commit that moves the WAL checkpoint saves the manifest of every
    // field, so a document without the field writes the record too.
    let engine = Engine::new(open_storage(&dir), schema_with(written))
        .await
        .unwrap();
    let text_only = Document::builder().add_text("title", "rust").build();
    engine.put_document("t9", text_only).await.unwrap();
    engine.commit().await.unwrap();
    drop(engine);
    assert_eq!(
        manifest(&dir)["field"],
        serde_json::json!({"kind": "hnsw", "dimension": 8, "distance": "Cosine"})
    );
    assert_rejected(
        Engine::new(open_storage(&dir), schema_with(euclidean)).await,
        "its files were written with kind Hnsw, dimension 8 and distance Cosine, so it cannot \
         be opened with distance Euclidean",
    );
}

/// An index that an older build opened under another dimension and wrote
/// to holds segments of both dimensions and no record. No dimension opens
/// it, and removing the field from the schema, opening and adding the field
/// again recovers the index, with the field empty.
#[tokio::test(flavor = "multi_thread")]
async fn an_index_with_segments_of_two_dimensions_needs_the_field_added_again() {
    let dir = TempDir::new().unwrap();
    let eight = FieldOption::Flat(flat(DIM));
    create_index(&dir, &eight).await;
    let other = TempDir::new().unwrap();
    let four = FieldOption::Flat(flat(4));
    create_index(&other, &four).await;

    // Register the 4-dimensional index's segment as a second segment.
    std::fs::copy(
        other.path().join("vector/vec/segment_000000.flat"),
        dir.path().join("vector/vec/segment_000001.flat"),
    )
    .unwrap();
    let mut mixed = manifest(&dir);
    let mut second = manifest(&other)["segments"][0].clone();
    second["segment_id"] = "segment_000001".into();
    second["generation"] = 2.into();
    mixed["segments"].as_array_mut().unwrap().push(second);
    mixed["next_segment_id"] = 2.into();
    mixed.as_object_mut().unwrap().remove("field");
    save_checksummed_json(open_storage(&dir).as_ref(), MANIFEST, None, &mixed).unwrap();

    assert_rejected(
        Engine::new(open_storage(&dir), schema_with(eight)).await,
        "its files were written with dimension 4, so it cannot be opened with dimension 8",
    );
    assert_rejected(
        Engine::new(open_storage(&dir), schema_with(four.clone())).await,
        "its files were written with dimension 8, so it cannot be opened with dimension 4",
    );

    let mut without_the_field = schema_with(four.clone());
    without_the_field.fields.remove("vec");
    let engine = Engine::new(open_storage(&dir), without_the_field)
        .await
        .unwrap();
    engine.add_field("vec", four.clone()).await.unwrap();
    engine.put_document("d9", doc(&four, 9)).await.unwrap();
    engine.commit().await.unwrap();
    assert_eq!(search(&engine, 4).await.len(), 1);
    drop(engine);

    let engine = Engine::new(open_storage(&dir), schema_with(four))
        .await
        .unwrap();
    assert_eq!(search(&engine, 4).await.len(), 1);
}

/// The ids, sorted, of the hits of [`try_search`].
async fn found_ids(engine: &Engine) -> Result<Vec<String>, String> {
    let mut ids: Vec<String> = try_search(engine, DIM)
        .await
        .map_err(|e| format!("search failed: {e}"))?
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    Ok(ids)
}

/// A change between the three kinds that read each other keeps the field's
/// vectors (Issue #1579), on disk too: the index opens under the new schema
/// and under the old one, and no file of the old kind is left once it was
/// opened again. On Windows a file that a reader has mapped cannot be
/// deleted before.
#[tokio::test(flavor = "multi_thread")]
async fn an_update_field_between_the_three_kinds_keeps_the_vectors_on_disk() {
    let kinds = [
        ("flat", FieldOption::Flat(flat(DIM))),
        ("hnsw", FieldOption::Hnsw(hnsw(DIM))),
        ("ivf", FieldOption::Ivf(ivf(DIM))),
    ];
    for (old_suffix, written) in &kinds {
        for (new_suffix, changed) in &kinds {
            if old_suffix == new_suffix {
                continue;
            }
            let label = format!("{old_suffix} -> {new_suffix}");
            let dir = TempDir::new().unwrap();
            create_index(&dir, written).await;
            let engine = Engine::new(open_storage(&dir), schema_with(written.clone()))
                .await
                .unwrap();
            let before = found_ids(&engine).await.unwrap();
            assert_eq!(before.len(), 8, "{label}");
            let outcome = engine
                .update_field("vec", changed.clone(), REINDEX)
                .await
                .unwrap();
            assert_eq!(outcome.classification, FieldChangeKind::Reindex, "{label}");
            assert_eq!(found_ids(&engine).await.unwrap(), before, "{label}");
            drop(engine);

            for schema in [changed, written] {
                let engine = Engine::new(open_storage(&dir), schema_with(schema.clone()))
                    .await
                    .unwrap();
                assert_eq!(found_ids(&engine).await.unwrap(), before, "{label}");
                drop(engine);
                let left: Vec<String> = field_files(&dir)
                    .into_keys()
                    .filter(|name| name.contains(&format!(".{old_suffix}")))
                    .collect();
                assert!(left.is_empty(), "{label}: {left:?}");
            }
        }
    }
}

/// Change field `vec` from `written` to `changed`, one of which is
/// MultiVector, with `update_field` and check each step, returning the
/// first that went wrong.
async fn change_kind(written: &FieldOption, changed: &FieldOption) -> Result<(), String> {
    let dir = TempDir::new().unwrap();
    create_index(&dir, written).await;
    let files = field_files(&dir);
    let engine = Engine::new(open_storage(&dir), schema_with(written.clone()))
        .await
        .map_err(|e| format!("open: {e}"))?;

    // The dry run reports the classification and changes nothing.
    let outcome = engine
        .update_field("vec", changed.clone(), DRY_RUN)
        .await
        .map_err(|e| format!("dry run: {e}"))?;
    if outcome.classification != FieldChangeKind::Destructive {
        return Err(format!(
            "the dry run classified it as {:?}",
            outcome.classification
        ));
    }
    if format!("{:?}", engine.schema().fields["vec"]) != format!("{written:?}") {
        return Err("the dry run changed the schema".to_string());
    }
    if field_files(&dir) != files {
        return Err("the dry run changed the field's files".to_string());
    }

    // Applying it discards the field's data.
    let outcome = engine
        .update_field("vec", changed.clone(), REINDEX)
        .await
        .map_err(|e| format!("update_field: {e}"))?;
    if outcome.classification != FieldChangeKind::Destructive {
        return Err(format!("it was applied as {:?}", outcome.classification));
    }
    let schema = engine.schema();
    if format!("{:?}", schema.fields["vec"]) != format!("{changed:?}") {
        return Err(format!("the schema has {:?}", schema.fields["vec"]));
    }
    if !schema.pending_reindex.contains("vec") {
        return Err("pending_reindex does not name the field".to_string());
    }
    let left: Vec<String> = field_files(&dir)
        .into_keys()
        .filter(|name| name.contains("segment_"))
        .collect();
    if !left.is_empty() {
        return Err(format!("segment files are left: {left:?}"));
    }

    // Documents put afterwards are indexed and searched under the new kind,
    // also after a reopen. Only a single-vector field is searched.
    for i in 10..13 {
        engine
            .put_document(&format!("n{i}"), doc(changed, i))
            .await
            .map_err(|e| format!("put: {e}"))?;
    }
    engine.commit().await.map_err(|e| format!("commit: {e}"))?;
    drop(engine);
    let engine = Engine::new(open_storage(&dir), schema_with(changed.clone()))
        .await
        .map_err(|e| format!("reopen: {e}"))?;
    if !matches!(changed, FieldOption::MultiVector(_)) {
        let found = found_ids(&engine).await?;
        if found != ["n10", "n11", "n12"] {
            return Err(format!("the search after the reopen found {found:?}"));
        }
    }
    Ok(())
}

/// A change between MultiVector and a single-vector kind is Destructive,
/// since they read only their own segments (Issue #1536). The dry run says
/// so, and applying it discards the field's data and recreates the field
/// empty under the new kind, marked in `pending_reindex`. Documents put
/// afterwards are searched under the new kind, also after a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn an_update_field_to_or_from_multi_vector_recreates_the_field_empty() {
    let multi_vector = FieldOption::MultiVector(multi_vector(DIM));
    let mut failures = Vec::new();
    for single in [
        FieldOption::Flat(flat(DIM)),
        FieldOption::Hnsw(hnsw(DIM)),
        FieldOption::Ivf(ivf(DIM)),
    ] {
        for (written, changed) in [(&single, &multi_vector), (&multi_vector, &single)] {
            if let Err(failure) = change_kind(written, changed).await {
                failures.push(format!("{written:?} -> {changed:?}: {failure}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
