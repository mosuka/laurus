//! Crash-atomic, checksummed persistence for small JSON control files
//! (#1022).
//!
//! Several subsystems keep a small file that decides what the rest of the
//! index means — the vector segment manifest, the document store's manifest,
//! the per-index metadata. Losing one of those to a torn write is not the
//! loss of one file: the index stops opening, or opens describing state that
//! no longer exists.
//!
//! They had each grown their own copy of the same five steps, and the copies
//! had drifted — one omitted the `sync()` that makes the rename durable,
//! another wrote a checksum it never verified on the way back in. This module
//! is the single implementation:
//!
//! 1. serialize to JSON
//! 2. write it to `<name>.tmp` through [`StructWriter`], which accumulates a
//!    CRC-32 of every byte and emits it in a footer on `close`
//! 3. `close`, which fsyncs the payload
//! 4. rename `<name>.tmp` over `<name>` — the atomic step
//! 5. `sync()` the storage, so the *directory entry* is durable too
//!
//! A crash before step 4 leaves the previous file untouched; a crash after it
//! leaves the new one. There is no window in which a reader sees half a file.
//!
//! # Legacy files
//!
//! Call sites that predate the framing wrote bare JSON in place. The loader
//! detects that by its leading byte and hands it back unverified, so existing
//! indexes keep opening; the next save rewrites them in the framed form.
//!
//! Framed files written before Issue #1214 end in a 4-byte trailer holding
//! the CRC-32 of the payload alone, rather than the footer that now covers
//! the whole file. The payload was the file's last write, so that trailer is
//! still checked against it.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::structured::{FOOTER_LEN, FOOTER_MAGIC, StructReader, StructWriter};
use crate::util::varint::decode_u64;

/// Suffix used for the staging file a save writes before renaming.
const TMP_SUFFIX: &str = ".tmp";

/// What [`load_checksummed_json`] found on storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestFormat {
    /// The framed form: length-prefixed JSON with a verified CRC-32 footer
    /// (or, written before Issue #1214, a verified trailer over the payload).
    Checksummed,
    /// Bare JSON written before framing existed. Read as-is; nothing verifies
    /// it, and the next save upgrades it.
    Legacy,
}

/// Write `value` as JSON, atomically and with a checksum.
///
/// Safe to call while holding a lock over the state being serialized — the
/// caller should serialize under the lock and then release it, since this
/// performs I/O.
///
/// # Arguments
///
/// * `storage` - Storage to write into.
/// * `name` - Final file name. `<name>.tmp` is used as the staging file and
///   is renamed over it.
/// * `magic` - Optional leading `u32` marker, written before the payload.
///   Pass `Some` where a call site already emits one; the same value must be
///   given to [`load_checksummed_json`].
/// * `value` - The value to serialize.
///
/// # Errors
///
/// Returns [`LaurusError`] if serialization, either write, the rename, or the
/// sync fails. On failure the previous file is left intact.
pub fn save_checksummed_json<T: Serialize>(
    storage: &dyn Storage,
    name: &str,
    magic: Option<u32>,
    value: &T,
) -> Result<()> {
    let json = serde_json::to_vec(value)
        .map_err(|e| LaurusError::index(format!("failed to serialize {name}: {e}")))?;
    save_checksummed(storage, name, magic, &json)
}

/// Write raw `payload` bytes, atomically and with a checksum.
///
/// The byte-level form exists for callers that produce their JSON separately
/// — for instance to serialize it under a lock they must not hold across I/O.
///
/// # Arguments
///
/// * `storage` - Storage to write into.
/// * `name` - Final file name.
/// * `magic` - Optional leading `u32` marker.
/// * `payload` - Bytes to store.
///
/// # Errors
///
/// Returns [`LaurusError`] if either write, the rename, or the sync fails.
pub fn save_checksummed(
    storage: &dyn Storage,
    name: &str,
    magic: Option<u32>,
    payload: &[u8],
) -> Result<()> {
    let tmp_name = format!("{name}{TMP_SUFFIX}");

    // The writer accumulates the CRC-32 of every byte; `close` writes it in
    // the file's footer and fsyncs.
    let output = storage.create_output(&tmp_name)?;
    let mut writer = StructWriter::new(output);
    if let Some(magic) = magic {
        writer.write_u32(magic)?;
    }
    writer.write_bytes(payload)?;
    writer.close()?;

    storage.rename_file(&tmp_name, name)?;
    // The rename itself is only durable once the directory entry is. Without
    // this a crash can lose the rename and leave the *previous* file — safe,
    // but not what the caller was told happened.
    storage.sync()?;

    Ok(())
}

/// Read a JSON value written by [`save_checksummed_json`].
///
/// Returns `Ok(None)` when the file does not exist or is empty, which callers
/// treat as "nothing persisted yet".
///
/// # Arguments
///
/// * `storage` - Storage to read from.
/// * `name` - File name.
/// * `magic` - The marker given at save time, if any. When present it is
///   consumed before the payload; a file lacking it is treated as legacy.
///
/// # Errors
///
/// Returns [`LaurusError`] if the file cannot be read, if the checksum does
/// not match, or if the payload does not deserialize. A checksum mismatch is
/// an error rather than a silent fallback: the whole point of the trailer is
/// that corruption is refused, not read as though it were valid.
pub fn load_checksummed_json<T: DeserializeOwned>(
    storage: &dyn Storage,
    name: &str,
    magic: Option<u32>,
) -> Result<Option<(T, ManifestFormat)>> {
    let Some((payload, format)) = load_checksummed(storage, name, magic)? else {
        return Ok(None);
    };
    let value = serde_json::from_slice(&payload)
        .map_err(|e| LaurusError::index(format!("failed to deserialize {name}: {e}")))?;
    Ok(Some((value, format)))
}

/// Read raw bytes written by [`save_checksummed`].
///
/// # Arguments
///
/// * `storage` - Storage to read from.
/// * `name` - File name.
/// * `magic` - The marker given at save time, if any.
///
/// # Returns
///
/// The payload and which on-disk form it was in, or `None` when the file is
/// absent or empty.
///
/// # Errors
///
/// Returns [`LaurusError`] if the file cannot be read or its checksum does not
/// match.
pub fn load_checksummed(
    storage: &dyn Storage,
    name: &str,
    magic: Option<u32>,
) -> Result<Option<(Vec<u8>, ManifestFormat)>> {
    use std::io::Read;

    let Ok(mut input) = storage.open_input(name) else {
        return Ok(None);
    };

    let mut content = Vec::new();
    input.read_to_end(&mut content)?;
    if content.is_empty() {
        return Ok(None);
    }

    if is_legacy_json(&content, magic) {
        return Ok(Some((content, ManifestFormat::Legacy)));
    }

    let input = storage.open_input(name)?;
    let mut reader = StructReader::new(input)?;
    if let Some(expected) = magic {
        let found = reader.read_u32()?;
        if found != expected {
            return Err(LaurusError::index(format!(
                "{name} has magic 0x{found:08X}, expected 0x{expected:08X}"
            )));
        }
    }
    let payload = reader.read_bytes()?;
    // The payload is the file's last write, so a legacy trailer — its CRC —
    // is still checked in full.
    if !reader.verify_checksum()?.vouches_for(&payload) {
        return Err(LaurusError::index(format!(
            "{name} checksum mismatch — the file is corrupted"
        )));
    }

    Ok(Some((payload, ManifestFormat::Checksummed)))
}

/// Whether `content` is bare JSON from before this framing existed.
///
/// With a magic marker the test is exact — framed files start with it, and a
/// file that starts with something else is legacy only if it actually looks
/// like JSON. Anything else is treated as framed so the magic check reports
/// the mismatch instead of a confusing deserialize error.
///
/// Without a marker the test is structural: a framed file is exactly
/// `varint(len) || payload || trailer`, so its length is determined by its
/// own header. Checking that is far sturdier than looking at the first byte,
/// which collides whenever a payload happens to be 91 or 123 bytes long — the
/// varint for those encodes as `[` and `{`.
///
/// # Arguments
///
/// * `content` - The file's bytes.
/// * `magic` - The marker expected for framed files, if any.
///
/// # Returns
///
/// `true` when the bytes should be parsed as bare JSON.
fn is_legacy_json(content: &[u8], magic: Option<u32>) -> bool {
    if let Some(expected) = magic {
        if content.len() >= 4 {
            let found = u32::from_le_bytes([content[0], content[1], content[2], content[3]]);
            if found == expected {
                return false;
            }
        }
        return looks_like_json(content);
    }

    !framing_is_consistent(content)
}

/// Whether `content`'s length matches what its own varint header claims.
///
/// The trailer is a footer (8 bytes, ending in [`FOOTER_MAGIC`]) or, for a
/// file written before Issue #1214, a 4-byte CRC. A footer is accepted only
/// with its magic in place: JSON never ends in the magic's last byte (`C`),
/// so a bare JSON file whose leading byte happens to read as a varint that
/// leaves exactly 8 bytes over is still recognised as legacy.
///
/// # Arguments
///
/// * `content` - The file's bytes.
///
/// # Returns
///
/// `true` when the bytes are shaped like `varint(len) || payload || trailer`.
fn framing_is_consistent(content: &[u8]) -> bool {
    let Ok((payload_len, header_len)) = decode_u64(content) else {
        return false;
    };
    let Some(framed_len) = usize::try_from(payload_len)
        .ok()
        .and_then(|n| n.checked_add(header_len))
    else {
        return false;
    };
    let ends_in_footer = content.ends_with(&FOOTER_MAGIC.to_le_bytes());
    let trailer_len = if ends_in_footer {
        FOOTER_LEN as usize
    } else {
        4
    };
    framed_len
        .checked_add(trailer_len)
        .is_some_and(|total| total == content.len())
}

/// Whether `content` starts like a JSON object or array.
///
/// # Arguments
///
/// * `content` - The file's bytes.
///
/// # Returns
///
/// `true` for bytes whose first non-whitespace character is `{` or `[`.
fn looks_like_json(content: &[u8]) -> bool {
    let first = content
        .iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace())
        .unwrap_or(0);
    first == b'{' || first == b'['
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use serde::Deserialize;
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Sample {
        name: String,
        count: u64,
    }

    fn sample() -> Sample {
        Sample {
            name: "alpha".to_string(),
            count: 42,
        }
    }

    fn storage() -> Arc<MemoryStorage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    /// A value survives a round trip and is reported as framed.
    #[test]
    fn round_trips_and_reports_the_framed_format() {
        let storage = storage();
        save_checksummed_json(storage.as_ref(), "m.json", None, &sample()).unwrap();

        let (loaded, format): (Sample, _) = load_checksummed_json(storage.as_ref(), "m.json", None)
            .unwrap()
            .expect("the file was just written");
        assert_eq!(loaded, sample());
        assert_eq!(format, ManifestFormat::Checksummed);
    }

    /// The magic marker round-trips and a mismatched one is refused rather
    /// than parsed as something else.
    #[test]
    fn magic_is_checked() {
        let storage = storage();
        save_checksummed_json(storage.as_ref(), "m.json", Some(0xABCD_1234), &sample()).unwrap();

        let (loaded, _): (Sample, _) =
            load_checksummed_json(storage.as_ref(), "m.json", Some(0xABCD_1234))
                .unwrap()
                .unwrap();
        assert_eq!(loaded, sample());

        let err = load_checksummed_json::<Sample>(storage.as_ref(), "m.json", Some(0xDEAD_BEEF))
            .unwrap_err();
        assert!(err.to_string().contains("magic"), "unexpected error: {err}");
    }

    /// The staging file must not survive a successful save.
    #[test]
    fn leaves_no_temp_file_behind() {
        let storage = storage();
        save_checksummed_json(storage.as_ref(), "m.json", None, &sample()).unwrap();
        assert!(!storage.file_exists("m.json.tmp"));
        assert!(storage.file_exists("m.json"));
    }

    /// Absent and empty files both read back as "nothing persisted".
    #[test]
    fn absent_and_empty_read_as_none() {
        let storage = storage();
        assert!(
            load_checksummed_json::<Sample>(storage.as_ref(), "missing.json", None)
                .unwrap()
                .is_none()
        );

        storage
            .create_output("empty.json")
            .unwrap()
            .close()
            .unwrap();
        assert!(
            load_checksummed_json::<Sample>(storage.as_ref(), "empty.json", None)
                .unwrap()
                .is_none()
        );
    }

    /// Bare JSON from before the framing existed still loads, and is reported
    /// as legacy so callers can upgrade it.
    #[test]
    fn legacy_bare_json_still_loads() {
        use std::io::Write;

        let storage = storage();
        let json = serde_json::to_vec(&sample()).unwrap();
        let mut output = storage.create_output("m.json").unwrap();
        output.write_all(&json).unwrap();
        output.close().unwrap();

        let (loaded, format): (Sample, _) = load_checksummed_json(storage.as_ref(), "m.json", None)
            .unwrap()
            .unwrap();
        assert_eq!(loaded, sample());
        assert_eq!(format, ManifestFormat::Legacy);
    }

    /// A legacy bare array is recognised too — the vector manifest's old form.
    #[test]
    fn legacy_bare_array_still_loads() {
        use std::io::Write;

        let storage = storage();
        let json = serde_json::to_vec(&vec![sample()]).unwrap();
        let mut output = storage.create_output("m.json").unwrap();
        output.write_all(&json).unwrap();
        output.close().unwrap();

        let (loaded, format): (Vec<Sample>, _) =
            load_checksummed_json(storage.as_ref(), "m.json", None)
                .unwrap()
                .unwrap();
        assert_eq!(loaded, vec![sample()]);
        assert_eq!(format, ManifestFormat::Legacy);
    }

    /// Corruption is refused, not read as valid data. This is the property
    /// the trailer exists for, and the document store was writing a checksum
    /// without ever checking it.
    #[test]
    fn corruption_is_refused() {
        use std::io::{Read, Write};

        let storage = storage();
        save_checksummed_json(storage.as_ref(), "m.json", None, &sample()).unwrap();

        let mut bytes = Vec::new();
        storage
            .open_input("m.json")
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        // Flip a byte inside the payload, leaving the framing intact.
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        let mut output = storage.create_output("m.json").unwrap();
        output.write_all(&bytes).unwrap();
        output.close().unwrap();

        let err = load_checksummed_json::<Sample>(storage.as_ref(), "m.json", None).unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "corruption must be refused, got: {err}"
        );
    }

    /// A framed payload of exactly 123 bytes has a varint header of `0x7B`,
    /// which is `{` — the byte a leading-byte heuristic reads as "legacy
    /// JSON". The same collision exists at 91 bytes for `[`.
    ///
    /// This is why the format test is structural rather than a peek at the
    /// first byte. Both collisions are exercised here because both are real
    /// payload sizes for a small manifest.
    #[test]
    fn framed_payload_is_not_mistaken_for_legacy_json() {
        for payload_len in [91usize, 123] {
            let storage = storage();
            let payload = vec![b'x'; payload_len];
            save_checksummed(storage.as_ref(), "m.json", None, &payload).unwrap();

            let (loaded, format) = load_checksummed(storage.as_ref(), "m.json", None)
                .unwrap()
                .expect("just written");
            assert_eq!(
                format,
                ManifestFormat::Checksummed,
                "a {payload_len}-byte payload must still read as framed"
            );
            assert_eq!(loaded, payload);
        }
    }

    fn put(storage: &MemoryStorage, name: &str, bytes: &[u8]) {
        use std::io::Write;

        let mut output = storage.create_output(name).unwrap();
        output.write_all(bytes).unwrap();
        output.close().unwrap();
    }

    /// A framed file written before Issue #1214 ends in the CRC of its
    /// payload alone. It still loads as checksummed, because that trailer is
    /// still checked against the payload — and a corrupted payload is still
    /// refused.
    #[test]
    fn a_legacy_trailer_is_still_checked_against_the_payload() {
        let storage = storage();
        let json = serde_json::to_vec(&sample()).unwrap();
        for magic in [None, Some(0xABCD_1234u32)] {
            let mut framed = Vec::new();
            if let Some(magic) = magic {
                framed.extend_from_slice(&magic.to_le_bytes());
            }
            framed.extend(crate::util::varint::encode_u64(json.len() as u64));
            framed.extend_from_slice(&json);
            framed.extend_from_slice(&crc32fast::hash(&json).to_le_bytes());

            put(&storage, "m.json", &framed);
            let (loaded, format): (Sample, _) =
                load_checksummed_json(storage.as_ref(), "m.json", magic)
                    .unwrap()
                    .unwrap();
            assert_eq!(loaded, sample());
            assert_eq!(format, ManifestFormat::Checksummed);

            let mid = framed.len() / 2;
            framed[mid] ^= 0x01;
            put(&storage, "m.json", &framed);
            let err =
                load_checksummed_json::<Sample>(storage.as_ref(), "m.json", magic).unwrap_err();
            assert!(err.to_string().contains("checksum mismatch"), "{err}");
        }
    }

    /// Bare JSON whose leading `[` or `{`, read as a varint, would leave
    /// exactly a footer's 8 bytes over is still legacy: a footer needs its
    /// magic, and JSON never ends in it.
    #[test]
    fn bare_json_of_a_footed_frames_length_is_still_legacy() {
        let storage = storage();
        // `[` is 91 and `{` is 123: header byte + payload + 8.
        for (json, len) in [("[1]", 1 + 91 + 8), ("{\"a\":1}", 1 + 123 + 8)] {
            let bare = format!("{json:<len$}");
            put(&storage, "m.json", bare.as_bytes());

            let (loaded, format) = load_checksummed(storage.as_ref(), "m.json", None)
                .unwrap()
                .unwrap();
            assert_eq!(format, ManifestFormat::Legacy, "{len}-byte {json}");
            assert_eq!(loaded, bare.as_bytes());
        }
    }

    /// Overwriting keeps the newest value and still leaves no staging file.
    #[test]
    fn overwrite_replaces_the_previous_value() {
        let storage = storage();
        save_checksummed_json(storage.as_ref(), "m.json", None, &sample()).unwrap();

        let updated = Sample {
            name: "beta".to_string(),
            count: 7,
        };
        save_checksummed_json(storage.as_ref(), "m.json", None, &updated).unwrap();

        let (loaded, _): (Sample, _) = load_checksummed_json(storage.as_ref(), "m.json", None)
            .unwrap()
            .unwrap();
        assert_eq!(loaded, updated);
        assert!(!storage.file_exists("m.json.tmp"));
    }
}
