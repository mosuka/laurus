//! `{segment}.ids` — the exact set of doc ids a segment holds (Issue #1210).
//!
//! Deletions address documents by global doc id, and a segment's id range
//! `[min_doc_id, max_doc_id]` can contain ids the segment does not hold: a
//! merge of non-adjacent segments spans the ones it left out, concurrent puts
//! can reach the writer out of id order, and callers of the lower-level APIs
//! may choose ids themselves. Before setting a deletion bit, the writer asks
//! this set whether the segment really holds the id, so a deletion can no
//! longer mark a document another segment owns.
//!
//! ```text
//! magic "SIDS"(u32 LE) | version(u16 LE) | payload: varint(len) + RoaringTreemap bytes
//! -- trailer: u32 CRC-32 (StructWriter::close) --
//! ```
//!
//! The payload is the file's last write, so `StructWriter`'s trailer — which
//! covers only the most recent write — covers exactly the payload, and
//! [`StructReader::verify_checksum`] checks it on read. This is the framing
//! the checksummed segment manifest uses.

use roaring::RoaringTreemap;

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::compound::open_part;
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{Storage, StorageInput, StorageOutput};
use crate::util::alloc_bounds::checked_len;

/// Part suffix of the doc-id set: `{segment}.ids`.
pub(crate) const DOC_ID_SET_SUFFIX: &str = "ids";

const MAGIC: u32 = u32::from_le_bytes(*b"SIDS");
const VERSION: u16 = 1;

/// Write a segment's doc-id set.
///
/// # Arguments
///
/// * `output` - The part output to write into; it is closed on success.
/// * `doc_ids` - The segment's doc ids, sorted and deduplicated.
///
/// # Errors
///
/// Returns an error if `doc_ids` is not strictly increasing or the write
/// fails.
pub(crate) fn write_doc_id_set<W: StorageOutput>(output: W, doc_ids: &[u64]) -> Result<()> {
    let set = RoaringTreemap::from_sorted_iter(doc_ids.iter().copied()).map_err(|e| {
        LaurusError::index(format!(
            "segment doc ids must be sorted and deduplicated: {e}"
        ))
    })?;
    let mut payload = Vec::with_capacity(set.serialized_size());
    set.serialize_into(&mut payload)
        .map_err(|e| LaurusError::index(format!("Failed to serialize segment doc ids: {e}")))?;

    let mut writer = StructWriter::new(output);
    writer.write_u32(MAGIC)?;
    writer.write_u16(VERSION)?;
    // Last write: the trailer checksum covers exactly this payload.
    writer.write_bytes(&payload)?;
    writer.close()
}

/// Read a segment's doc-id set, compound or loose.
///
/// # Returns
///
/// `Ok(None)` when the segment has no `.ids` part — one written before
/// Issue #1210.
///
/// # Errors
///
/// Returns an error when the part exists but is unreadable, has a foreign
/// magic or version, fails its checksum, or does not decode.
pub(crate) fn read_doc_id_set(
    storage: &dyn Storage,
    segment_id: &str,
) -> Result<Option<RoaringTreemap>> {
    let Some(input) = open_part(storage, segment_id, DOC_ID_SET_SUFFIX)? else {
        return Ok(None);
    };
    decode(input, segment_id).map(Some)
}

/// A segment's doc-id set: its `.ids` part, or — for a segment written
/// before that part existed — the `.norms` slot map, which records the same
/// ids (Issue #1210).
///
/// # Returns
///
/// `Ok(None)` when the segment has neither part (a pre-#555 segment).
///
/// # Errors
///
/// Returns an error when the part that exists is unreadable or corrupt.
pub(crate) fn load_segment_doc_ids(
    storage: &dyn Storage,
    segment_id: &str,
) -> Result<Option<RoaringTreemap>> {
    if let Some(ids) = read_doc_id_set(storage, segment_id)? {
        return Ok(Some(ids));
    }
    super::norms::read_doc_ids(storage, segment_id)
}

/// A segment's doc-id set, read through the segment's own storage — its
/// compound facade, or the index storage for a loose segment (Issue #1211).
///
/// [`load_segment_doc_ids`] opens a compound container from scratch, which
/// on an eager in-memory backend copies the whole container; a
/// `SegmentReader` already holds a facade whose part windows share one
/// buffered copy, so it reads through that instead. Existence is checked
/// first: the facade passes a missing part through to the inner storage,
/// whose `open_input` fails.
///
/// # Returns
///
/// `Ok(None)` when the segment has neither `.ids` nor `.norms`.
///
/// # Errors
///
/// Returns an error when the part that exists is unreadable or corrupt.
pub(crate) fn load_doc_ids_from_segment_storage(
    storage: &dyn Storage,
    segment_id: &str,
) -> Result<Option<RoaringTreemap>> {
    let ids = format!("{segment_id}.{DOC_ID_SET_SUFFIX}");
    if storage.file_exists(&ids) {
        return decode(storage.open_input(&ids)?, segment_id).map(Some);
    }
    let norms = format!("{segment_id}.norms");
    if storage.file_exists(&norms) {
        return super::norms::read_doc_ids_from(storage.open_input(&norms)?).map(Some);
    }
    Ok(None)
}

fn decode<R: StorageInput>(input: R, segment_id: &str) -> Result<RoaringTreemap> {
    let mut reader = StructReader::new(input)?;
    let magic = reader.read_u32()?;
    if magic != MAGIC {
        return Err(LaurusError::index(format!(
            "{segment_id}.{DOC_ID_SET_SUFFIX}: bad magic {magic:#010x}"
        )));
    }
    let version = reader.read_u16()?;
    if version != VERSION {
        return Err(LaurusError::index(format!(
            "{segment_id}.{DOC_ID_SET_SUFFIX}: unsupported version {version}"
        )));
    }
    let len = reader.read_varint()? as usize;
    let available = reader.size().saturating_sub(reader.position());
    let len = checked_len(len, available, "segment doc-id set")?;
    let payload = reader.read_raw(len)?;
    if !reader.verify_checksum()? {
        return Err(LaurusError::index(format!(
            "{segment_id}.{DOC_ID_SET_SUFFIX}: checksum mismatch — the part is corrupted"
        )));
    }
    RoaringTreemap::deserialize_from(&payload[..]).map_err(|e| {
        LaurusError::index(format!(
            "{segment_id}.{DOC_ID_SET_SUFFIX}: failed to decode the doc-id set: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn storage() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    #[test]
    fn round_trips_a_sparse_set() {
        let storage = storage();
        let ids = [0u64, 2, 3, 10, 1 << 33];
        write_doc_id_set(storage.create_output("seg.ids").unwrap(), &ids).unwrap();

        let set = read_doc_id_set(storage.as_ref(), "seg").unwrap().unwrap();
        assert_eq!(set.iter().collect::<Vec<_>>(), ids);
    }

    #[test]
    fn a_missing_part_reads_as_none() {
        assert!(
            read_doc_id_set(storage().as_ref(), "seg")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rejects_unsorted_ids() {
        let storage = storage();
        let output = storage.create_output("seg.ids").unwrap();
        assert!(write_doc_id_set(output, &[3, 1]).is_err());
    }

    /// A flipped payload byte fails the checksum instead of decoding into a
    /// wrong set — trusting a wrong set would skip a real deletion.
    #[test]
    fn rejects_a_corrupted_payload() {
        let storage = storage();
        write_doc_id_set(storage.create_output("seg.ids").unwrap(), &[1, 2, 3]).unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut storage.open_input("seg.ids").unwrap(), &mut bytes)
            .unwrap();
        // Past magic (4) + version (2) + length varint (1): inside the payload.
        bytes[8] ^= 0xFF;
        let mut output = storage.create_output("seg.ids").unwrap();
        std::io::Write::write_all(&mut output, &bytes).unwrap();
        output.close().unwrap();

        let err = read_doc_id_set(storage.as_ref(), "seg").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
    }
}
