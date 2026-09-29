//! Issue #1214: every `StructWriter` file ends in an 8-byte footer —
//! `[crc32 of every byte before it: u32 LE]["LCRC": u32 LE]` — instead of a
//! 4-byte trailer holding the CRC-32 of the file's last write only.
//!
//! End to end through `LexicalStore`: an index written with legacy trailers
//! still opens, searches and merges; a merge refuses a source whose
//! random-access parts (`.post`, `.bkd`) fail their footer; and a corrupted
//! `.delmap` is an error, never "no deletions" (which resurrected deleted
//! documents) nor a file the writer silently replaces.

use std::io::{Read, Write};
use std::sync::Arc;

use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::{
    InvertedIndexConfig, LexicalIndexConfig, LexicalSearchRequest, LexicalStore, NumericRangeQuery,
    Query, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{Document, LaurusError, Result as LaurusResult};

/// The footer's trailing magic, as it sits on disk.
const MAGIC: [u8; 4] = *b"LCRC";

/// Footer length: the CRC-32, then the magic.
const FOOTER_LEN: usize = 8;

/// Doc ids of the two segments `build_index` commits.
const FIRST_SEGMENT: [u64; 3] = [1, 2, 3];
const SECOND_SEGMENT: [u64; 3] = [4, 5, 6];

/// The document `build_index` deletes from the first segment.
const DELETED: u64 = 2;

/// The `StructWriter` parts every segment of `doc`s has (`.dv` is not one).
const STRUCT_PARTS: [&str; 6] = ["dict", "post", "docs", "norms", "ids", "rank.bkd"];

fn inverted_config(use_compound: bool, merging: bool) -> InvertedIndexConfig {
    InvertedIndexConfig {
        use_compound,
        // Merging: past one segment, every segment is merged at once
        // (`merge_factor` is clamped to the segment count).
        max_segments: if merging { 1 } else { 1000 },
        merge_factor: 100,
        ..Default::default()
    }
}

fn config(use_compound: bool, merging: bool) -> LexicalIndexConfig {
    LexicalIndexConfig::Inverted(inverted_config(use_compound, merging))
}

/// A text field (`.dict`/`.post`) and a numeric one (`.rank.bkd`).
fn doc(id: u64) -> Document {
    Document::builder()
        .add_text("body", format!("word{id} common"))
        .add_integer("rank", id as i64 * 10)
        .build()
}

/// Two committed segments with merging disabled; with `delete`, `DELETED`
/// is then deleted from the first, which writes its `.delmap`. Built through
/// `InvertedIndex`, since `LexicalStore` has no public delete.
fn build_index(use_compound: bool, delete: bool) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index =
        InvertedIndex::create(storage.clone(), inverted_config(use_compound, false)).unwrap();
    let mut writer = index.writer().unwrap();
    for ids in [FIRST_SEGMENT, SECOND_SEGMENT] {
        for id in ids {
            writer.upsert_document(id, doc(id)).unwrap();
        }
        writer.commit().unwrap();
    }
    if delete {
        writer.delete_document(DELETED).unwrap();
        writer.commit().unwrap();
    }
    storage
}

fn read_file(storage: &Arc<dyn Storage>, name: &str) -> Vec<u8> {
    let mut input = storage.open_input(name).unwrap();
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
    bytes
}

fn write_file(storage: &Arc<dyn Storage>, name: &str, bytes: &[u8]) {
    let mut output = storage.create_output(name).unwrap();
    output.write_all(bytes).unwrap();
    output.close().unwrap();
}

/// A LEB128 varint at the start of `bytes`: `(value, encoded length)`.
fn decode_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, byte) in bytes.iter().enumerate().take(10) {
        value |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// The last write of a file whose legacy trailer is checked against it: a
/// framed manifest (`varint(len) || json`) or a `.ids` part (`"SIDS" | u16
/// version | varint(len) || set`). `None` when the bytes are not so framed.
fn checked_last_write<'a>(name: &str, bytes: &'a [u8]) -> Option<&'a [u8]> {
    let payload_end = bytes.len().checked_sub(FOOTER_LEN)?;
    let header = if name.ends_with(".ids") { 6 } else { 0 };
    let (len, varint_len) = decode_varint(bytes.get(header..)?)?;
    let start = header + varint_len;
    (start + len as usize == payload_end).then(|| &bytes[start..payload_end])
}

/// Rewrite every file ending in the footer magic into its pre-#1214 form:
/// footer dropped, 4-byte trailer appended. Returns the files converted.
fn rewrite_as_legacy(storage: &Arc<dyn Storage>) -> Vec<String> {
    let mut converted = Vec::new();
    for name in storage.list_files().unwrap() {
        let bytes = read_file(storage, &name);
        if !bytes.ends_with(&MAGIC) {
            continue;
        }
        let payload = &bytes[..bytes.len() - FOOTER_LEN];
        let trailer = match checked_last_write(&name, &bytes) {
            Some(last_write) => crc32fast::hash(last_write),
            None => {
                assert!(
                    !name.ends_with(".json") && !name.ends_with(".ids"),
                    "{name}: expected a framed payload"
                );
                // Nothing checks these trailers; any non-magic value will do.
                crc32fast::hash(&payload[payload.len() - 4..])
            }
        };
        assert_ne!(
            trailer.to_le_bytes(),
            MAGIC,
            "{name}: trailer reads as magic"
        );
        write_file(storage, &name, &[payload, &trailer.to_le_bytes()].concat());
        converted.push(name);
    }
    converted.sort();
    converted
}

/// The segment ids the manifest — the sole publication record — lists.
fn list_manifest_ids(storage: &Arc<dyn Storage>) -> Vec<String> {
    let bytes = read_file(storage, "segments.json");
    let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| {
        let (len, start) = decode_varint(&bytes).unwrap();
        serde_json::from_slice(&bytes[start..start + len as usize]).unwrap()
    });
    let mut ids: Vec<String> = payload["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["segment_id"].as_str().unwrap().to_string())
        .collect();
    ids.sort();
    ids
}

fn hits(store: &LexicalStore, query: Box<dyn Query>) -> LaurusResult<Vec<u64>> {
    let results = store.search(LexicalSearchRequest::new(query).limit(100))?;
    let mut ids: Vec<u64> = results.hits.iter().map(|hit| hit.doc_id).collect();
    ids.sort_unstable();
    Ok(ids)
}

fn term_hits(store: &LexicalStore, term: &str) -> LaurusResult<Vec<u64>> {
    hits(store, Box::new(TermQuery::new("body", term)))
}

fn range_hits(store: &LexicalStore) -> LaurusResult<Vec<u64>> {
    hits(
        store,
        Box::new(NumericRangeQuery::i64_range("rank", Some(0), Some(1000))),
    )
}

/// Where `{segment}.{suffix}` lives: `(file, offset, len)`, parsing the
/// `.cfs` part table (`compound.rs`) for a compound segment.
fn locate_part(storage: &Arc<dyn Storage>, segment: &str, suffix: &str) -> (String, usize, usize) {
    let container = format!("{segment}.cfs");
    if !storage.file_exists(&container) {
        let name = format!("{segment}.{suffix}");
        let len = read_file(storage, &name).len();
        return (name, 0, len);
    }
    let bytes = read_file(storage, &container);
    // Trailer: u64 table_offset | u32 table_crc | u32 version | u32 "CFND".
    let trailer_start = bytes.len() - 20;
    let table_offset =
        u64::from_le_bytes(bytes[trailer_start..trailer_start + 8].try_into().unwrap()) as usize;
    // Table: varint count, then { varint suffix_len, suffix, u64 offset, u64 len }.
    let table = &bytes[table_offset..trailer_start];
    let (count, mut cursor) = decode_varint(table).unwrap();
    for _ in 0..count {
        let (suffix_len, n) = decode_varint(&table[cursor..]).unwrap();
        cursor += n;
        let name = &table[cursor..cursor + suffix_len as usize];
        cursor += suffix_len as usize;
        let field = |at: usize| u64::from_le_bytes(table[at..at + 8].try_into().unwrap()) as usize;
        let (offset, len) = (field(cursor), field(cursor + 8));
        cursor += 16;
        if name == suffix.as_bytes() {
            return (container, offset, len);
        }
    }
    panic!("{container} has no {suffix} part");
}

/// Flip one bit in the middle of a part's payload, clear of its footer.
fn corrupt_part(storage: &Arc<dyn Storage>, segment: &str, suffix: &str) {
    let (file, offset, len) = locate_part(storage, segment, suffix);
    let mut bytes = read_file(storage, &file);
    assert_eq!(
        bytes[offset + len - 4..offset + len],
        MAGIC,
        "{segment}.{suffix}: the part must end in a footer"
    );
    bytes[offset + (len - FOOTER_LEN) / 2] ^= 0x01;
    write_file(storage, &file, &bytes);
}

/// Rewrite the deleted id in `segment`'s `.delmap` as `DELETED + 1`: the
/// bitmap still decodes, so only the footer can tell. Returns the file name
/// and its corrupted bytes.
fn corrupt_delmap(storage: &Arc<dyn Storage>, segment: &str) -> (String, Vec<u8>) {
    let name = format!("{segment}.delmap");
    let mut bytes = read_file(storage, &name);
    // The payload ends with the Roaring array container's last u16 value.
    let at = bytes.len() - FOOTER_LEN - 2;
    assert_eq!(
        u16::from_le_bytes([bytes[at], bytes[at + 1]]),
        DELETED as u16,
        "precondition: the bitmap's last value is the deleted id"
    );
    bytes[at] = (DELETED + 1) as u8;
    write_file(storage, &name, &bytes);
    (name, bytes)
}

/// Recompute a footed file's CRC, so its current bytes read as intact.
fn restamp_footer(storage: &Arc<dyn Storage>, name: &str) {
    let mut bytes = read_file(storage, name);
    let payload_end = bytes.len() - FOOTER_LEN;
    let crc = crc32fast::hash(&bytes[..payload_end]);
    bytes[payload_end..payload_end + 4].copy_from_slice(&crc.to_le_bytes());
    write_file(storage, name, &bytes);
}

/// Reopen with the merging config, add a document and commit: the merge
/// must fail, publishing only the new document's own segment. Returns the
/// commit's error.
fn assert_merge_is_refused(storage: &Arc<dyn Storage>, use_compound: bool) -> LaurusError {
    let store = LexicalStore::new(storage.clone(), config(use_compound, true)).unwrap();
    let before = list_manifest_ids(storage);
    store.upsert_document(7, doc(7)).unwrap();
    let err = store
        .commit()
        .expect_err("a corrupted source segment must abort the merge");
    drop(store);

    let after = list_manifest_ids(storage);
    let added: Vec<&String> = after.iter().filter(|id| !before.contains(id)).collect();
    assert_eq!(added.len(), 1, "only the new document's segment: {added:?}");
    assert!(!added[0].starts_with("merged_"), "{added:?}");
    assert!(
        before.iter().all(|id| after.contains(id)),
        "the source segments must stay published: {before:?} -> {after:?}"
    );
    err
}

fn assert_corruption(err: &LaurusError) {
    let msg = err.to_string();
    assert!(
        msg.contains("checksum mismatch") || msg.contains("corrupted"),
        "expected a corruption error, got: {msg}"
    );
}

/// An index whose every part and manifest carries a pre-#1214 trailer
/// opens, answers term and range queries, and merges into footed parts.
#[test]
fn an_index_written_before_the_footer_still_opens_searches_and_merges() {
    let storage = build_index(false, true);
    let sources = list_manifest_ids(&storage);
    assert_eq!(sources.len(), 2, "{sources:?}");
    let converted = rewrite_as_legacy(&storage);
    let mut expected = vec![
        "segments.json".to_string(),
        "metadata.json".to_string(),
        format!("{}.delmap", sources[0]),
    ];
    for segment in &sources {
        for suffix in STRUCT_PARTS {
            expected.push(format!("{segment}.{suffix}"));
        }
    }
    for name in &expected {
        assert!(
            converted.contains(name),
            "{name} not converted: {converted:?}"
        );
    }
    for name in storage.list_files().unwrap() {
        assert!(
            !read_file(&storage, &name).ends_with(&MAGIC),
            "{name} still ends in the magic"
        );
    }

    let live = [1, 3, 4, 5, 6];
    let store = LexicalStore::new(storage.clone(), config(false, false)).unwrap();
    assert_eq!(term_hits(&store, "common").unwrap(), live);
    assert_eq!(term_hits(&store, "word2").unwrap(), [] as [u64; 0]);
    assert_eq!(term_hits(&store, "word5").unwrap(), [5]);
    assert_eq!(range_hits(&store).unwrap(), live);
    drop(store);

    let store = LexicalStore::new(storage.clone(), config(false, true)).unwrap();
    store.upsert_document(7, doc(7)).unwrap();
    store.commit().unwrap();

    let segments = list_manifest_ids(&storage);
    assert_eq!(segments.len(), 1, "every segment must merge: {segments:?}");
    let merged = &segments[0];
    assert!(merged.starts_with("merged_"), "{merged}");
    let live = [1, 3, 4, 5, 6, 7];
    assert_eq!(term_hits(&store, "common").unwrap(), live);
    assert_eq!(term_hits(&store, "word2").unwrap(), [] as [u64; 0]);
    assert_eq!(range_hits(&store).unwrap(), live);

    for suffix in STRUCT_PARTS {
        let name = format!("{merged}.{suffix}");
        assert!(read_file(&storage, &name).ends_with(&MAGIC), "{name}");
    }
    for manifest in ["segments.json", "metadata.json"] {
        assert!(
            read_file(&storage, manifest).ends_with(&MAGIC),
            "{manifest}"
        );
    }
}

/// A flipped `.post` byte in a merge source aborts the merge, in both
/// layouts, before it can be rewritten under a fresh checksum.
#[test]
fn a_corrupted_source_postings_part_aborts_the_merge() {
    for use_compound in [false, true] {
        let storage = build_index(use_compound, false);
        let first = list_manifest_ids(&storage)[0].clone();
        corrupt_part(&storage, &first, "post");
        let err = assert_merge_is_refused(&storage, use_compound);
        assert_corruption(&err);
        assert!(
            err.to_string().contains(".post"),
            "compound={use_compound}: {err}"
        );
    }
}

/// A flipped `.bkd` byte in a merge source aborts the merge, in both layouts.
#[test]
fn a_corrupted_source_bkd_part_aborts_the_merge() {
    for use_compound in [false, true] {
        let storage = build_index(use_compound, false);
        let first = list_manifest_ids(&storage)[0].clone();
        corrupt_part(&storage, &first, "rank.bkd");
        let err = assert_merge_is_refused(&storage, use_compound);
        assert_corruption(&err);
        assert!(
            err.to_string().contains(".bkd"),
            "compound={use_compound}: {err}"
        );
    }
}

/// A corrupted `.delmap` never serves or publishes its deleted document as
/// live, and the writer never overwrites it with a fresh bitmap.
#[test]
fn a_corrupted_deletion_bitmap_is_not_treated_as_no_deletions() {
    for use_compound in [false, true] {
        // Control: restamped, the corrupted bitmap loads and DELETED is
        // live again — the footer is the only guard against it.
        let storage = build_index(use_compound, true);
        let first = list_manifest_ids(&storage)[0].clone();
        let (delmap, _) = corrupt_delmap(&storage, &first);
        restamp_footer(&storage, &delmap);
        let store = LexicalStore::new(storage.clone(), config(use_compound, false)).unwrap();
        assert_eq!(term_hits(&store, "word2").unwrap(), [DELETED]);
        drop(store);

        // (a) The store opens, but a query touching the segment fails (the
        // bitmap loads on first use), and so does the merge.
        let storage = build_index(use_compound, true);
        corrupt_delmap(&storage, &first);
        let store = LexicalStore::new(storage.clone(), config(use_compound, false)).unwrap();
        for result in [term_hits(&store, "word2"), range_hits(&store)] {
            let err = result.expect_err("a query must not answer past a corrupted .delmap");
            assert_corruption(&err);
        }
        drop(store);
        let err = assert_merge_is_refused(&storage, use_compound);
        assert_corruption(&err);
        assert!(err.to_string().contains("deletion bitmap"), "{err}");

        // (b) A deletion in that segment refuses to replace the bitmap.
        let storage = build_index(use_compound, true);
        let (delmap, corrupted) = corrupt_delmap(&storage, &first);
        let store = LexicalStore::new(storage.clone(), config(use_compound, false)).unwrap();
        let err = store
            .upsert_document(DELETED + 1, doc(DELETED + 1))
            .expect_err("deleting from a segment with a corrupted .delmap must fail");
        assert_corruption(&err);
        // Whatever the commit and the writer's drop do, they must not
        // rewrite the bitmap.
        let _ = store.commit();
        drop(store);
        assert_eq!(
            read_file(&storage, &delmap),
            corrupted,
            "compound={use_compound}: the corrupted .delmap must not be overwritten"
        );
    }
}
