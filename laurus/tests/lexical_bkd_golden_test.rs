//! Byte-for-byte guard on the `.bkd` files the writer produces (Issue #1165).
//!
//! Each scenario builds an index from a fixed corpus and pins every `.bkd`
//! file's name, length and 64-bit FNV-1a hash. The pins were recorded before the writer's
//! buffered points moved from per-document maps to per-field columns, so a
//! refactor of how points are buffered or fed to `BKDWriter` must reproduce
//! them exactly. A change that alters the on-disk BKD bytes on purpose (a new
//! `BKD_VERSION`, a different tree layout) has to re-record them: the failure
//! message prints the actual list in source form.
//!
//! The corpus is chosen to make ordering mistakes visible:
//!
//! - documents arrive in an order unrelated to their ids;
//! - many points share a value, so stable-sort ties decide the layout;
//! - multi-valued fields repeat values inside one document;
//! - documents are re-upserted, deleted, and deleted then re-upserted while
//!   still buffered, and some of the deleted versions carry extreme values
//!   and NaN that must not leak into the written trees;
//! - one field's points are all deleted, so it must produce no `.bkd`;
//! - one id is buffered twice through `upsert_analyzed_document`, whose two
//!   entries both reach the tree;
//! - point counts exceed a 512-point leaf, so the trees split.
//!
//! Every value is supplied raw (no trigonometry), so the bytes are the same
//! on every platform. Flushes and merge rollovers are triggered by document
//! count or a 1-byte budget only, never by the buffered-memory estimate,
//! which is allowed to change.

use std::sync::Arc;

use ahash::AHashMap;
use laurus::lexical::core::analyzed::AnalyzedDocument;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::config::InvertedIndexConfig;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::{InvertedIndexWriter, InvertedIndexWriterConfig};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{Document, GeoPoint};

/// Deterministic pseudo-random stream (64-bit LCG, Knuth's constants).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A document with every point-bearing field type, whose values depend on
/// `id` and `version` (so a re-upsert changes them) and repeat often.
///
/// Two fields exist to make the order of a document's own points visible,
/// which a split 1D field hides (every split re-sorts by value, and a
/// document's equal values are indistinguishable): `few` stays under one
/// 512-point leaf, so it is never split and keeps insertion order; `spots`
/// gives each document two 2D points tied on latitude, so a split on
/// latitude keeps their insertion order.
fn point_doc(id: u64, version: u64) -> Document {
    let key = id.wrapping_mul(31).wrapping_add(version);
    let floats = [-0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, 1.5, -2.25];
    let tag_count = (key % 4 + 1) as usize;
    let tags: Vec<i64> = (0..tag_count)
        .map(|k| ((key + k as u64 * 3) % 7) as i64)
        .collect();
    let lat = (key % 5) as f64 * 10.0;
    let mut builder = Document::builder()
        .add_integer("n", (key % 37) as i64)
        .add_float("f", floats[(key % floats.len() as u64) as usize])
        .add_geo(
            "loc",
            (key % 50) as f64 - 25.0,
            (key % 70) as f64 * 2.0 - 70.0,
        )
        .add_int64_array("tags", tags)
        .add_geo_ecef(
            "ecef",
            (key % 11) as f64 * 1000.0,
            (key % 13) as f64 * -500.0,
            (key % 5) as f64 * 250.0,
        )
        .add_geo_array(
            "spots",
            vec![
                GeoPoint::new(lat, (key % 90) as f64),
                GeoPoint::new(lat, -((key % 90) as f64) - 1.0),
            ],
        );
    if key.is_multiple_of(10) {
        builder = builder.add_float64_array("few", vec![key as f64, -(key as f64), 0.5]);
    }
    builder.build()
}

/// A version of `id` whose values must never reach a written tree: extreme
/// integers, NaN, and a field (`gone`) that only deleted documents carry.
fn doomed_doc(id: u64) -> Document {
    Document::builder()
        .add_integer(
            "n",
            if id.is_multiple_of(2) {
                i64::MAX
            } else {
                i64::MIN
            },
        )
        .add_float("f", f64::NAN)
        .add_int64_array("tags", vec![i64::MAX, i64::MIN, 0])
        .add_integer("gone", id as i64)
        .build()
}

/// Points only, entered through `upsert_analyzed_document` for an id that is
/// already buffered: the writer keeps both entries (Issue #1210).
fn analyzed_points(id: u64) -> AnalyzedDocument {
    let mut point_values = AHashMap::new();
    point_values.insert("n".to_string(), vec![vec![(id % 37) as f64]]);
    point_values.insert("tags".to_string(), vec![vec![3.0], vec![1.0], vec![3.0]]);
    AnalyzedDocument {
        field_terms: AHashMap::new(),
        stored_fields: AHashMap::new(),
        field_lengths: AHashMap::new(),
        point_values,
    }
}

/// 64-bit FNV-1a. Not CRC32: every `.bkd` ends with a CRC32 of the bytes
/// before it, and the CRC32 of data followed by its own CRC32 is the same
/// constant for every file.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// `(name, length, FNV-1a)` of every `.bkd` file in `storage`, by name.
fn bkd_digests(storage: &Arc<dyn Storage>) -> Vec<(String, u64, u64)> {
    let mut names: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".bkd"))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let mut input = storage.open_input(&name).unwrap();
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
            let hash = fnv1a64(&bytes);
            (name, bytes.len() as u64, hash)
        })
        .collect()
}

fn assert_digests(scenario: &str, actual: &[(String, u64, u64)], expected: &[(&str, u64, u64)]) {
    let matches = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|((an, al, ac), (en, el, ec))| an == en && al == el && ac == ec);
    if !matches {
        let mut source = String::new();
        for (name, len, hash) in actual {
            source.push_str(&format!("    (\"{name}\", {len}, 0x{hash:016x}),\n"));
        }
        panic!(
            "{scenario}: .bkd files differ from the pinned bytes.\n\
             Actual (paste over the expected list only for an intended \
             on-disk change):\n{source}"
        );
    }
}

const DOCS: u64 = 1500;

/// A permutation of `0..DOCS` (7919 is coprime to 1500), so push order is
/// unrelated to doc-id order.
fn shuffled_id(i: u64) -> u64 {
    (i * 7919) % DOCS
}

const COMMIT_PATH: &[(&str, u64, u64)] = &[
    ("segment_000000.ecef.bkd", 10570, 0x6db5bb66aff4ce17),
    ("segment_000000.f.bkd", 3769, 0x140549c71986d793),
    ("segment_000000.few.bkd", 1357, 0x46cb2f3081e95447),
    ("segment_000000.loc.bkd", 7249, 0xc37138abdbc2cd1e),
    ("segment_000000.n.bkd", 4118, 0xd344f47cba2690ad),
    ("segment_000000.spots.bkd", 13855, 0x343de22a87fced06),
    ("segment_000000.tags.bkd", 9691, 0x594857ffc7381155),
    ("segment_000001.ecef.bkd", 11100, 0xd3c04f6de99a5291),
    ("segment_000001.f.bkd", 3956, 0x0ce4e19f3ce2074a),
    ("segment_000001.few.bkd", 1239, 0x0d5b9716b52e9fef),
    ("segment_000001.loc.bkd", 7612, 0x49b01791925f43f6),
    ("segment_000001.n.bkd", 4352, 0x56f0180431fa9c69),
    ("segment_000001.spots.bkd", 14547, 0x44761b4593c3e64d),
    ("segment_000001.tags.bkd", 10318, 0xeb44a11be13ae819),
    ("segment_000002.ecef.bkd", 10882, 0x774acc22a453b9c5),
    ("segment_000002.f.bkd", 3879, 0x859069f1a7d320f7),
    ("segment_000002.few.bkd", 1117, 0x72c8242bb2f3e289),
    ("segment_000002.loc.bkd", 7463, 0x5b40718abaa6d5a2),
    ("segment_000002.n.bkd", 4252, 0x4cdea5d76ca2274e),
    ("segment_000002.spots.bkd", 14262, 0x5d71a12614a0a8e9),
    ("segment_000002.tags.bkd", 9947, 0xc26deaa6ffc62a02),
];

/// The standalone writer's commit path, committing every 500 documents so
/// the corpus spans several segments. (Explicit commits rather than
/// count-triggered auto-flushes: an auto-flush could land between a doomed
/// version's upsert and its delete and write its NaN, which the writer
/// rightly rejects. Both reach the same `flush_segment`.)
#[test]
fn commit_path_bkd_bytes_are_pinned() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = InvertedIndexWriterConfig {
        use_compound: false,
        max_buffered_docs: 1_000_000,
        max_buffer_memory: usize::MAX,
        ..Default::default()
    };
    let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

    // Buffered, then all deleted: the commit finds an empty buffer.
    for id in 0..5 {
        writer.upsert_document(id, doomed_doc(id)).unwrap();
    }
    for id in 0..5 {
        writer.delete_document(id).unwrap();
    }
    writer.commit().unwrap();

    let mut rng = Lcg(1165);
    for i in 0..DOCS {
        let id = shuffled_id(i);
        match rng.below(10) {
            // Deleted while buffered, carrying values that must not leak.
            0 => {
                writer.upsert_document(id, doomed_doc(id)).unwrap();
                writer.delete_document(id).unwrap();
            }
            // Deleted, then re-upserted while still buffered.
            1 => {
                writer.upsert_document(id, doomed_doc(id)).unwrap();
                writer.delete_document(id).unwrap();
                writer.upsert_document(id, point_doc(id, 1)).unwrap();
            }
            // Re-upserted in place: the first version must be replaced.
            2 => {
                writer.upsert_document(id, doomed_doc(id)).unwrap();
                writer.upsert_document(id, point_doc(id, 2)).unwrap();
            }
            // A second buffered entry for the same id.
            3 => {
                writer.upsert_document(id, point_doc(id, 0)).unwrap();
                writer
                    .upsert_analyzed_document(id, analyzed_points(id))
                    .unwrap();
            }
            _ => writer.upsert_document(id, point_doc(id, 0)).unwrap(),
        }
        if i % 500 == 499 {
            writer.commit().unwrap();
        }
    }
    writer.commit().unwrap();

    assert_digests("commit path", &bkd_digests(&storage), COMMIT_PATH);
}

/// Three committed segments with ids interleaved across them, where later
/// segments re-upsert and delete documents of earlier ones.
fn three_segment_index() -> (Arc<dyn Storage>, InvertedIndex) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = InvertedIndexConfig {
        use_compound: false,
        max_buffered_docs: 1_000_000,
        max_buffer_memory: usize::MAX,
        max_segments: 1000,
        ..Default::default()
    };
    let index = InvertedIndex::create(storage.clone(), config).unwrap();
    let mut writer = index.writer().unwrap();

    let mut rng = Lcg(1144);
    for segment in 0..3u64 {
        for i in 0..DOCS {
            let id = shuffled_id(i);
            if id % 3 != segment {
                continue;
            }
            writer.upsert_document(id, point_doc(id, segment)).unwrap();
        }
        if segment > 0 {
            // Touch documents committed in earlier segments.
            for _ in 0..60 {
                let id = rng.below(DOCS);
                if id % 3 >= segment {
                    continue;
                }
                if rng.below(2) == 0 {
                    writer
                        .upsert_document(id, point_doc(id, 10 + segment))
                        .unwrap();
                } else {
                    writer.delete_document(id).unwrap();
                }
            }
        }
        writer.commit().unwrap();
    }
    drop(writer);
    (storage, index)
}

const OPTIMIZE_PATH: &[(&str, u64, u64)] = &[
    ("merged_3.ecef.bkd", 34561, 0xdf0ba2d40249b7de),
    ("merged_3.f.bkd", 12669, 0x121ddb08cb020e14),
    ("merged_3.few.bkd", 3935, 0xf7bf3b9f56fc7dae),
    ("merged_3.loc.bkd", 23475, 0x4f17229147e944f7),
    ("merged_3.n.bkd", 11281, 0xa9f98cc81d94aa52),
    ("merged_3.spots.bkd", 45748, 0xc416f4c177a18211),
    ("merged_3.tags.bkd", 22432, 0x90367e0bb7c97381),
];

/// A force-merge replays every source segment's points into one writer.
#[test]
fn optimize_path_bkd_bytes_are_pinned() {
    let (storage, index) = three_segment_index();
    index.optimize().unwrap();
    assert_digests("optimize path", &bkd_digests(&storage), OPTIMIZE_PATH);
}

const BOUNDED_OPTIMIZE_PATH: &[(&str, u64, u64)] = &[
    ("merged_3.ecef.bkd", 11330, 0x35ca437ec8acc8f3),
    ("merged_3.f.bkd", 3942, 0x48398bbb131493b5),
    ("merged_3.few.bkd", 1313, 0xf161401a964b8fd6),
    ("merged_3.loc.bkd", 7752, 0x7d4868877df88ea5),
    ("merged_3.n.bkd", 3942, 0xa06a04917e14a884),
    ("merged_3.spots.bkd", 14807, 0x93f7d52c96967bf5),
    ("merged_3.tags.bkd", 8937, 0xb311e91fa9cc54cd),
    ("merged_4.ecef.bkd", 12114, 0xad0a1afe5a19ca6f),
    ("merged_4.f.bkd", 4313, 0xaec4708d0a064d43),
    ("merged_4.few.bkd", 1392, 0xae2e0ec1eaf78148),
    ("merged_4.loc.bkd", 8305, 0x82a0fe0435f163fe),
    ("merged_4.n.bkd", 4252, 0x2077b7d8b42d62ae),
    ("merged_4.spots.bkd", 15376, 0xfcaf4ffded1a9aa1),
    ("merged_4.tags.bkd", 9499, 0x4ffdb810ddc485aa),
    ("merged_5.ecef.bkd", 12553, 0xac41d377f33b720f),
    ("merged_5.f.bkd", 4445, 0x805149c73ff6f8b5),
    ("merged_5.few.bkd", 1477, 0x95d5f54b9ba09cd5),
    ("merged_5.loc.bkd", 8774, 0x5d0ded5d40414f5c),
    ("merged_5.n.bkd", 4152, 0x36ff63e64668add2),
    ("merged_5.spots.bkd", 16771, 0xb97c47fbdec92040),
    ("merged_5.tags.bkd", 9815, 0x1bbb2f53d2a59ee5),
];

/// A 1-byte budget rolls the merge over after every source segment.
#[test]
fn bounded_optimize_path_bkd_bytes_are_pinned() {
    let (storage, index) = three_segment_index();
    index.optimize_within_budget(1).unwrap();
    assert_digests(
        "bounded optimize path",
        &bkd_digests(&storage),
        BOUNDED_OPTIMIZE_PATH,
    );
}
