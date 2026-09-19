//! Integration test for preserving BKD points of index-only numeric fields
//! across a merge (Issue #758, follow-up of #753).
//!
//! A numeric field configured `indexed = true, stored = false` lives only in
//! the BKD tree (not in the stored `.docs`). The merge must reconstruct point
//! values from the source segments' BKD trees — not from stored fields — or
//! range queries on such a field stop matching after a merge.

use std::sync::Arc;

use laurus::DataValue;
use laurus::Document;
use laurus::lexical::core::field::{FieldOption, FloatOption, IntegerOption};
use laurus::lexical::query::NumericRangeQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn price_doc(price: i64) -> Document {
    Document::builder()
        .add_field("price", DataValue::Int64(price))
        .build()
}

fn count_in_range(store: &LexicalStore, lower: i64, upper: i64) -> usize {
    let query = Box::new(NumericRangeQuery::i64_range(
        "price",
        Some(lower),
        Some(upper),
    ));
    store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .len()
}

#[test]
fn merge_preserves_bkd_points_for_index_only_numeric_field() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    // `price` is indexed (BKD) but NOT stored — so its points exist only in the
    // BKD tree, the case the old stored-field-derived merge dropped.
    let config = LexicalIndexConfig::builder()
        .add_field(
            "price",
            FieldOption::Integer(IntegerOption {
                indexed: true,
                stored: false,
                multi_valued: false,
                doc_values: true,
            }),
        )
        .build();
    let store = LexicalStore::new(storage, config).unwrap();

    // Two segments (two commits).
    store.upsert_document(1, price_doc(10)).unwrap();
    store.upsert_document(2, price_doc(20)).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, price_doc(30)).unwrap();
    store.commit().unwrap();

    // Sanity: range query works before merge (per-segment BKD).
    assert_eq!(
        count_in_range(&store, 0, 100),
        3,
        "all docs in range pre-merge"
    );

    // Merge: the index-only field's BKD points must survive (#758). With the
    // old stored-field derivation these were lost (stored=false), so the merged
    // segment had no BKD and this returned 0.
    store.optimize().unwrap();
    assert_eq!(
        count_in_range(&store, 0, 100),
        3,
        "all BKD points survive the merge for a stored=false field"
    );
    // The actual values survive, not just the count: only price=20 is in [15,25].
    assert_eq!(
        count_in_range(&store, 15, 25),
        1,
        "merged BKD holds the real point values"
    );
}

fn float_doc(v: f64) -> Document {
    Document::builder()
        .add_field("v", DataValue::Float64(v))
        .build()
}

fn count_in_f64_range(store: &LexicalStore, lower: Option<f64>, upper: Option<f64>) -> usize {
    let query = Box::new(NumericRangeQuery::f64_range("v", lower, upper));
    store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .len()
}

/// Regression test for a BKD-leaf bit-packing blocker (Issue #549):
/// `compute_aabb` used to compare with plain IEEE `<`/`>`, which does not
/// distinguish `-0.0` from `+0.0`, so a leaf's `leaf_min`/`leaf_max` could
/// fail to bound one of its own points in the sortable-integer order the
/// packer relies on. `MergeEngine`'s point-collecting visitor always forces
/// a full descent (`CellRelation::Crosses` unconditionally) and feeds every
/// decoded point straight back into `BKDWriter::write`, which rejects NaN —
/// so if that bound is wrong, a `-0.0`/`+0.0` mix (or `Infinity`, which
/// pins the same `leaf_min`/`leaf_max` comparison at its widest) makes the
/// merge itself fail, not just return a slightly wrong query result.
#[test]
fn merge_preserves_bkd_points_with_signed_zero_and_infinity() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "v",
            FieldOption::Float(FloatOption {
                indexed: true,
                stored: true,
                multi_valued: false,
                doc_values: true,
            }),
        )
        .build();
    let store = LexicalStore::new(storage, config).unwrap();

    // Segment 1: +0.0 then -0.0 then a positive value, in the exact order
    // that defeats a naive IEEE `<`/`>` AABB computation.
    store.upsert_document(1, float_doc(0.0)).unwrap();
    store.upsert_document(2, float_doc(-0.0)).unwrap();
    store.upsert_document(3, float_doc(5.0)).unwrap();
    store.commit().unwrap();

    // Segment 2: both infinities.
    store
        .upsert_document(4, float_doc(f64::NEG_INFINITY))
        .unwrap();
    store.upsert_document(5, float_doc(f64::INFINITY)).unwrap();
    store.commit().unwrap();

    // The merge itself must not error (see doc comment above).
    store.optimize().unwrap();

    assert_eq!(
        count_in_f64_range(&store, None, None),
        5,
        "all 5 docs survive the merge"
    );
    // +0.0 and -0.0 compare numerically equal, so both match [0.0, 0.0].
    assert_eq!(
        count_in_f64_range(&store, Some(0.0), Some(0.0)),
        2,
        "both signed zeros survive and match a [0.0, 0.0] range"
    );
    assert_eq!(
        count_in_f64_range(&store, Some(f64::NEG_INFINITY), Some(f64::NEG_INFINITY)),
        1
    );
    assert_eq!(
        count_in_f64_range(&store, Some(f64::INFINITY), Some(f64::INFINITY)),
        1
    );
}
