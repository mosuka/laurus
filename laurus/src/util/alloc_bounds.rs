//! Allocation bounds for on-disk segment parsing (Issue #806, moved here
//! from `vector::index` in Issue #1047 so lexical loaders can share it).
//!
//! Generalizes the Issue #791 technique. A segment reader/loader takes
//! element counts and byte lengths straight from an on-disk header that
//! has **not** yet been integrity-checked. For vector segments, the
//! `.hnsw` CRC footer (Issue #786) is verified before the reader's
//! structural parse, but *legacy footer-less segments* — and every writer
//! load path, which does not run the footer verification at all — reach
//! these counts unverified. Lexical `.dv` (DocValues) segments have no
//! footer verification at all. A single flipped byte can turn a small
//! `num_vectors` / `node_count` / `field_name_len` / `num_fields` into a
//! multi-GiB allocation request that aborts the process through
//! `handle_alloc_error` (OOM) instead of surfacing a clean "corrupted
//! segment" error.
//!
//! These helpers reject impossible sizes up front by comparing the
//! header-declared size against ground truth: the true byte length of the
//! file, obtained from [`crate::storage::StorageInput::size`]. Because the
//! bytes a count or buffer describes must physically exist in the file, any
//! size larger than the bytes left in the relevant section is provably
//! corruption and is rejected before a single byte is allocated.
//!
//! Loaders capture `file_size = input.size()?` once and pass the bytes
//! still available (`file_size - stream_position`) as `available`; per-record
//! checks reuse the section's starting `available` so no extra `seek` /
//! `stream_position` syscall is added inside the hot per-record loops.

use crate::error::{LaurusError, Result};

/// Bound a header-declared element `count` against the bytes available in
/// the file before it is used to reserve a `Vec` / `HashMap` capacity.
///
/// Each element occupies at least `min_stride` bytes on disk, so a region of
/// `available` bytes can hold at most `available / min_stride` of them. A
/// `count` larger than that cannot be backed by real data and is treated as
/// corruption — preventing a `with_capacity(count)` that would request
/// gigabytes from a flipped byte and abort via `handle_alloc_error`.
///
/// # Arguments
///
/// * `count` - The element count parsed from the (unverified) header.
/// * `min_stride` - A lower bound on the on-disk byte size of one element.
///   Pass the smallest number of bytes every element is guaranteed to
///   occupy; `0` is treated as `1` so the division never traps.
/// * `available` - The number of bytes the file can still supply for these
///   elements (typically `file_size - current_position`).
/// * `what` - A short label naming the count, used in the error message.
///
/// # Returns
///
/// `count` unchanged when it fits.
///
/// # Errors
///
/// [`LaurusError::Index`] when `count` exceeds what `available` bytes can
/// hold (the segment is corrupted).
pub(crate) fn checked_capacity(
    count: usize,
    min_stride: u64,
    available: u64,
    what: &str,
) -> Result<usize> {
    checked_capacity_u64(count as u64, min_stride, available, what)
}

/// [`checked_capacity`] for a `count` still in its on-disk `u64` form
/// (Issue #1218).
///
/// The count is compared before it is narrowed to `usize`: on a 32-bit
/// target, `count as usize` would first truncate a corrupt count into a
/// small, plausible one that the bound then lets through.
///
/// # Errors
///
/// [`LaurusError::Index`] when `count` exceeds what `available` bytes can
/// hold (the segment is corrupted).
pub(crate) fn checked_capacity_u64(
    count: u64,
    min_stride: u64,
    available: u64,
    what: &str,
) -> Result<usize> {
    let stride = min_stride.max(1);
    let max_elements = available / stride;
    if count > max_elements {
        return Err(LaurusError::index(format!(
            "{what}: header declares {count} elements but at most {max_elements} can fit in the \
             {available} bytes left in the file — segment is corrupted"
        )));
    }
    narrow(count, what)
}

/// Bound a header-declared byte `len` against the bytes available in the
/// file before it is used to size a `vec![0u8; len]` (or equivalent) read
/// buffer.
///
/// A single record's buffer can never be larger than the bytes remaining in
/// the file, so a `len` that exceeds `available` is corruption and is
/// rejected before the buffer is allocated.
///
/// # Arguments
///
/// * `len` - The byte length parsed from the (unverified) header.
/// * `available` - The number of bytes the file can still supply (typically
///   `file_size - current_position`).
/// * `what` - A short label naming the length, used in the error message.
///
/// # Returns
///
/// `len` unchanged when it fits.
///
/// # Errors
///
/// [`LaurusError::Index`] when `len` exceeds `available` (the segment is
/// corrupted).
pub(crate) fn checked_len(len: usize, available: u64, what: &str) -> Result<usize> {
    checked_len_u64(len as u64, available, what)
}

/// [`checked_len`] for a `len` still in its on-disk `u64` form (Issue
/// #1218), compared before it is narrowed to `usize` for the same reason as
/// [`checked_capacity_u64`].
///
/// # Errors
///
/// [`LaurusError::Index`] when `len` exceeds `available` (the segment is
/// corrupted).
pub(crate) fn checked_len_u64(len: u64, available: u64, what: &str) -> Result<usize> {
    if len > available {
        return Err(LaurusError::index(format!(
            "{what}: header declares {len} bytes but only {available} bytes are left in the file \
             — segment is corrupted"
        )));
    }
    narrow(len, what)
}

/// Narrow a size already bounded by the bytes left in the file. It cannot
/// fail for an input the platform can address; a size that still does not
/// fit is reported as corruption rather than truncated.
fn narrow(value: u64, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        LaurusError::index(format!(
            "{what}: {value} exceeds this platform's address space — segment is corrupted"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_capacity_accepts_a_count_that_fits() {
        // 10 elements * 4 bytes = 40 <= 40 available.
        assert_eq!(checked_capacity(10, 4, 40, "n").unwrap(), 10);
    }

    #[test]
    fn checked_capacity_accepts_exact_fit() {
        assert_eq!(checked_capacity(5, 8, 40, "n").unwrap(), 5);
    }

    #[test]
    fn checked_capacity_rejects_a_count_that_overflows_the_file() {
        // A flipped byte turning the count huge while the file holds only
        // a handful of bytes must be rejected, not allocated.
        let err = checked_capacity(1usize << 40, 8, 64, "num_vectors").unwrap_err();
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("num_vectors"), "message should name the count");
                assert!(msg.contains("corrupted"), "message should flag corruption");
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn checked_capacity_treats_zero_stride_as_one() {
        // A zero stride must not divide-by-zero; it degrades to a 1-byte
        // lower bound so the count is bounded by `available`.
        assert!(checked_capacity(64, 0, 64, "n").is_ok());
        assert!(checked_capacity(65, 0, 64, "n").is_err());
    }

    #[test]
    fn checked_len_accepts_a_length_that_fits() {
        assert_eq!(checked_len(16, 16, "field_name_len").unwrap(), 16);
        assert_eq!(checked_len(0, 16, "field_name_len").unwrap(), 0);
    }

    #[test]
    fn u64_bounds_accept_what_fits_and_reject_what_does_not() {
        assert_eq!(checked_len_u64(16, 16, "len").unwrap(), 16);
        assert_eq!(checked_len_u64(0, 0, "len").unwrap(), 0);
        assert!(checked_len_u64(17, 16, "len").is_err());
        assert!(checked_len_u64(u64::MAX, 16, "len").is_err());

        assert_eq!(checked_capacity_u64(5, 8, 40, "n").unwrap(), 5);
        assert!(checked_capacity_u64(6, 8, 40, "n").is_err());
        let err = checked_capacity_u64(u64::MAX, 1, 40, "num_entries").unwrap_err();
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("num_entries"));
                assert!(
                    msg.contains(&u64::MAX.to_string()),
                    "the real count is reported"
                );
                assert!(msg.contains("corrupted"));
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn checked_len_rejects_a_length_beyond_the_file() {
        let err = checked_len(1usize << 31, 16, "field_name_len").unwrap_err();
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("field_name_len"));
                assert!(msg.contains("corrupted"));
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }
}
