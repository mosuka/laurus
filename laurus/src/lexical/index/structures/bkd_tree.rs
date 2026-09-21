//! BKD tree implementation for axis-aligned numeric range and visitor-driven
//! spatial queries.
//!
//! Modeled on Apache Lucene's BKD-tree, the on-disk layout described on
//! [`BKD_VERSION`] stores per-node and per-leaf axis-aligned bounding boxes
//! (AABBs) so the reader can prune subtrees with Inside / Outside / Crosses
//! logic. The trait [`BKDTree`] exposes the low-level
//! [`intersect`](BKDTree::intersect) primitive plus a default
//! [`range_search`](BKDTree::range_search) wrapper.

use super::aabb::AABB;
use super::visitor::{CellRelation, IntersectVisitor, RangeQueryVisitor};
use crate::error::Result;
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{Storage, StorageInput, StorageOutput};
use crate::util::alloc_bounds::{checked_capacity, checked_len};
use std::io::SeekFrom;
use std::sync::Arc;

/// Trait for BKD Tree implementations (in-memory or disk-based).
///
/// Implementations expose two query primitives:
///
/// - [`BKDTree::intersect`] is the low-level Lucene-style traversal: the
///   reader walks the tree once, calling the visitor's `compare` method on
///   each subtree's AABB (Inside / Outside / Crosses) and either pruning,
///   collecting, or descending accordingly. This is the building block for
///   sphere queries, k-NN, and any custom shape that fits the visitor API.
/// - [`BKDTree::range_search`] is the legacy axis-aligned range API. It is
///   provided as a default method that builds a [`RangeQueryVisitor`] and
///   delegates to `intersect`, so concrete `BKDTree` implementations only
///   need to supply `intersect`.
pub trait BKDTree: Send + Sync + std::fmt::Debug {
    /// Walk the tree, dispatching subtree pruning decisions and per-point
    /// candidates to `visitor`.
    ///
    /// Implementations are expected to honor the visitor's `compare` result
    /// faithfully:
    /// - `CellRelation::Outside` cells are skipped.
    /// - `CellRelation::Inside` cells contribute every doc id beneath them
    ///   via `visit_inside` (the visitor does not need the point bytes).
    /// - `CellRelation::Crosses` leaves expose every (doc_id, point) pair
    ///   via `visit` so the visitor can perform the final per-point check.
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()>;

    /// This tree's global value range on dimension `dim`, plus the
    /// number of points it covers, without walking the tree.
    ///
    /// Returns `(min, max, total_point_count)`. Implementations that
    /// cannot answer cheaply return `None`, and callers must then treat
    /// the range as unknown — the default does exactly that (#944).
    ///
    /// The range covers every point written at flush time, including
    /// those of documents deleted since, so it is a superset of the live
    /// values. Callers using it to prune therefore only ever fail to
    /// prune; they never prune something that could have matched.
    ///
    /// # Arguments
    ///
    /// * `dim` - Zero-based dimension index.
    ///
    /// # Returns
    ///
    /// `Some((min, max, total_point_count))` when known, else `None`.
    fn value_range(&self, dim: usize) -> Option<(f64, f64, u64)> {
        let _ = dim;
        None
    }

    /// Axis-aligned range search returning the matching doc ids in sorted
    /// and deduplicated order.
    ///
    /// `mins[d]` / `maxs[d]` may be `None` to leave a dimension unbounded.
    /// `include_min` / `include_max` control whether the boundary itself
    /// matches.
    ///
    /// The default implementation builds a [`RangeQueryVisitor`] and
    /// delegates to [`BKDTree::intersect`].
    fn range_search(
        &self,
        mins: &[Option<f64>],
        maxs: &[Option<f64>],
        include_min: bool,
        include_max: bool,
    ) -> Result<Vec<u64>> {
        let mut visitor = RangeQueryVisitor::new(mins, maxs, include_min, include_max);
        self.intersect(&mut visitor)?;
        let mut hits = visitor.into_hits();
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }
}

/// Magic number for BKD Tree files: ASCII "BKDT" in little-endian.
pub const BKD_MAGIC: u32 = 0x54444B42;

/// Current on-disk format version.
///
/// Version 4 (this revision, Issue #1142): within each leaf, points are now
/// stored in **doc_id-ascending order** (a stable sort — see
/// [`BKDWriter::write_leaf_block`] for why stability matters), and
/// `doc_ids` are packed as **consecutive deltas** (`doc_id[i] - doc_id[i-1]`,
/// with `doc_id[-1] := doc_id_base`, so the first packed delta is always
/// `0`) instead of independent deltas from `doc_id_base`. This shrinks
/// `doc_id_bits` to `bits_needed(max consecutive gap)` rather than
/// `bits_needed(max - min)`, which is never larger and often much smaller
/// for locally-clustered doc_id distributions. `doc_id_base`/`doc_id_bits`
/// keep the same on-disk types and positions as version 3; only what they
/// mean changes. The file header also now carries `block_size`, used to
/// bound a leaf's `count` on read (version 3 lacked this bound).
///
/// Version 3 (Issue #549): leaf blocks bit-pack `points` and `doc_ids`
/// instead of storing them as raw `f64`/`u64`. Each dimension's per-point
/// values are packed at a fixed bit width derived from `leaf_min`/`leaf_max`
/// (delta-from-min in IEEE-754 total-order space, never stored on disk since
/// writer and reader compute it identically); `doc_ids` used the same
/// scheme, independently anchored at `doc_id_base` per point (no sort, no
/// running delta). The previous version 2 layout (raw, unpacked leaves) is
/// no longer supported, and version 3 itself is no longer supported by this
/// revision — laurus is pre-release, so the format is broken intentionally
/// rather than dual-supported (Issue #1040 tracks these one-way breaks for
/// release notes).
///
/// File layout (all integers little-endian):
///
/// ```text
/// Header (fixed-prefix + 2 * num_dims * 8 bytes):
///   magic               u32
///   version             u32
///   num_dims            u32
///   bytes_per_dim       u32   (always 8 today: f64)
///   total_point_count   u64
///   num_blocks          u64
///   block_size          u32   (max points per leaf; bounds a leaf's `count`)
///   global_min          [f64; num_dims]
///   global_max          [f64; num_dims]
///   index_start_offset  u64
///   root_node_offset    u64
///
/// Leaf Block (points stored in doc_id-ascending order):
///   count               u32
///   leaf_min            [f64; num_dims]
///   leaf_max            [f64; num_dims]
///   doc_id_base         u64   (= the leaf's minimum doc_id)
///   doc_id_bits         u8    (0..=64; validated on read; width of the
///                              largest consecutive doc_id gap in the leaf)
///   packed_dim[0]       ceil(count * bits[0] / 8) bytes, byte-aligned
///   packed_dim[1..]     ...  (one section per dimension; bits[d] is
///                             derived from leaf_min[d]/leaf_max[d], not
///                             stored)
///   packed_doc_ids      ceil(count * doc_id_bits / 8) bytes, byte-aligned;
///                       value i is `doc_id[i-1] + delta[i]` with
///                       `doc_id[-1] := doc_id_base`, so `delta[0]` is
///                       always 0 (the reader treats a nonzero leading
///                       delta as corruption)
///
/// Internal Index Node (size = 28 + 32 * num_dims bytes, unchanged from v2):
///   split_dim           u32
///   split_value         f64
///   left_min            [f64; num_dims]
///   left_max            [f64; num_dims]
///   right_min           [f64; num_dims]
///   right_max           [f64; num_dims]
///   left_offset         u64
///   right_offset        u64
/// ```
pub const BKD_VERSION: u32 = 4;

/// BKD Tree File Header
#[derive(Debug, Clone)]
pub struct BKDFileHeader {
    pub magic: u32,
    pub version: u32,
    pub num_dims: u32,
    pub bytes_per_dim: u32,
    pub total_point_count: u64,
    pub num_blocks: u64,
    /// Max points per leaf, as configured on the writer (`BKDWriter::
    /// with_block_size`, default 512). Bounds a leaf's `count` on read
    /// (Issue #1142).
    pub block_size: u32,
    pub min_values: Vec<f64>,
    pub max_values: Vec<f64>,
    pub index_start_offset: u64,
    pub root_node_offset: u64,
}

/// Writer for BKD Trees.
pub struct BKDWriter<W: StorageOutput> {
    writer: StructWriter<W>,
    block_size: usize,
    num_blocks: u64,
    num_dims: u32,
    min_values: Vec<f64>,
    max_values: Vec<f64>,
    index_nodes: Vec<IndexNode>,
}

/// Internal index node for navigation.
///
/// Each node remembers the axis-aligned bounding box (`*_min`/`*_max`) of the
/// two child subtrees in addition to the split dimension and value, enabling
/// readers to prune entire subtrees when their AABB lies fully inside or
/// outside the query region.
#[derive(Debug, Clone)]
struct IndexNode {
    split_dim: u32,
    split_value: f64,
    left_min: Vec<f64>,
    left_max: Vec<f64>,
    right_min: Vec<f64>,
    right_max: Vec<f64>,
    left_offset: u64,
    right_offset: u64,
    // Helper to back-patch offsets during writing
    left_child_idx: Option<usize>,
    right_child_idx: Option<usize>,
}

/// Information returned by `BKDWriter::build_subtree` so the caller can fold
/// per-child AABBs into the parent index node.
struct SubtreeInfo {
    /// `Some(idx)` when the subtree is rooted at the internal node at index
    /// `idx` in `index_nodes`; `None` when the subtree is a single leaf
    /// (whose file offset was captured by the caller before recursion).
    node_idx: Option<usize>,
    /// Per-dimension minimum coordinates of all points in this subtree.
    min: Vec<f64>,
    /// Per-dimension maximum coordinates of all points in this subtree.
    max: Vec<f64>,
}

/// Borrowed view over the caller's flat point/doc_id buffers used during
/// recursive subtree construction. Holding only references avoids deep-copying
/// the input data while the builder permutes its private index array.
struct BuildContext<'a> {
    points: &'a [f64],
    doc_ids: &'a [u64],
    num_dims: usize,
}

impl BuildContext<'_> {
    /// Return the d-th coordinate of the point at slot `i` in the original
    /// (unpermuted) buffer.
    #[inline]
    fn value(&self, i: u32, d: usize) -> f64 {
        self.points[i as usize * self.num_dims + d]
    }
}

/// Return whichever of `a`/`b` is smaller under IEEE-754 total order
/// (`f64::total_cmp`), not plain IEEE `<`.
///
/// This matters because IEEE `<`/`>` treat `-0.0` and `+0.0` as equal, while
/// the leaf bit-packing in [`BKDWriter::write_leaf_block`] needs `leaf_min`
/// to be the true total-order minimum of the leaf's points — otherwise a
/// point equal to the "wrong" zero can fall outside the `[leaf_min, leaf_max]`
/// range the packer assumes, corrupting its reconstructed bit pattern (see
/// the module-level format notes on [`BKD_VERSION`]).
#[inline]
fn total_min(a: f64, b: f64) -> f64 {
    if a.total_cmp(&b).is_le() { a } else { b }
}

/// Return whichever of `a`/`b` is larger under IEEE-754 total order. See
/// [`total_min`].
#[inline]
fn total_max(a: f64, b: f64) -> f64 {
    if a.total_cmp(&b).is_ge() { a } else { b }
}

/// Compute the per-dimension axis-aligned bounding box that encloses every
/// point referenced by `indices` in the underlying buffer.
///
/// The returned `(min, max)` Vecs have length `ctx.num_dims`. Callers must
/// pass a non-empty `indices` slice; an empty slice would leave the bounds at
/// their `INFINITY` / `NEG_INFINITY` sentinels and propagate degenerate
/// AABBs into the index, so the caller is responsible for the precondition.
fn compute_aabb(ctx: &BuildContext<'_>, indices: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let mut min = vec![f64::INFINITY; ctx.num_dims];
    let mut max = vec![f64::NEG_INFINITY; ctx.num_dims];
    for &i in indices {
        let base = i as usize * ctx.num_dims;
        for d in 0..ctx.num_dims {
            let v = ctx.points[base + d];
            min[d] = total_min(min[d], v);
            max[d] = total_max(max[d], v);
        }
    }
    (min, max)
}

/// Pick the dimension whose `(max - min)` range is the widest.
///
/// Ties are broken by lower dimension index (stable, deterministic). The
/// caller must pass equal-length `min` / `max` slices of at least one
/// element — empty AABBs have no defined "widest axis".
///
/// Returning `u32` matches the on-disk `split_dim` encoding so the caller
/// can drop the result straight into an `IndexNode`.
fn widest_axis(min: &[f64], max: &[f64]) -> u32 {
    debug_assert_eq!(min.len(), max.len());
    debug_assert!(!min.is_empty());
    let mut best = 0usize;
    let mut best_range = max[0] - min[0];
    for d in 1..min.len() {
        let r = max[d] - min[d];
        if r > best_range {
            best = d;
            best_range = r;
        }
    }
    best as u32
}

/// Map an `f64` to a `u64` whose *unsigned* ordering matches the value's
/// IEEE-754 total order — the same order [`f64::total_cmp`] computes. More
/// negative values map to smaller `u64`s, `-0.0` sorts just below `+0.0`, and
/// `NEG_INFINITY`/`INFINITY` map to the smallest/largest non-NaN results.
///
/// `BKDWriter::write` rejects NaN coordinates up front, so this never has to
/// define (or preserve) an ordering for NaN bit patterns. The unsigned form
/// (as opposed to the signed/arithmetic-shift variant some implementations
/// use) is deliberate: it lets [`write_leaf_block`](BKDWriter::write_leaf_block)
/// and its reader counterpart compute `max - min` with plain `u64`
/// subtraction, which never overflows because total order guarantees
/// `sortable(leaf_min) <= sortable(leaf_max) <= sortable(any point in the leaf)`.
#[inline]
fn f64_to_sortable_u64(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

/// Inverse of [`f64_to_sortable_u64`].
#[inline]
fn sortable_u64_to_f64(bits: u64) -> f64 {
    let bits = if bits & (1 << 63) != 0 {
        bits & !(1u64 << 63)
    } else {
        !bits
    };
    f64::from_bits(bits)
}

/// Number of bits needed to represent `v` in unsigned binary (`0` for `v ==
/// 0`), i.e. `ceil(log2(v + 1))`.
#[inline]
fn bits_needed(v: u64) -> u8 {
    (64 - v.leading_zeros()) as u8
}

/// Byte length of a packed run of `count` values at `bits` bits each.
#[inline]
fn packed_byte_len(count: usize, bits: u8) -> usize {
    (count * bits as usize).div_ceil(8)
}

/// Validates that `len` (a point/doc_id count) fits in `u32`, the width
/// [`BKDWriter::write`] uses for its index permutation.
///
/// Extracted from `write` as its own function so the `u32::MAX` boundary can
/// be unit-tested directly on the `usize` value, without allocating the
/// multi-gigabyte buffer a real out-of-range call would require.
fn checked_point_count_u32(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        crate::error::LaurusError::index(format!(
            "BKD point count {len} exceeds u32::MAX; cannot build index permutation"
        ))
    })
}

/// Sortable-space anchor and bit width for one dimension's delta-from-min
/// packing, derived from that dimension's `leaf_min`/`leaf_max`.
///
/// Both [`BKDWriter::write_leaf_block`] and
/// [`BKDReader::read_leaf_fixed_header`] call this — a dimension's bit width
/// is never stored on disk, so having the two sides share one formula
/// (rather than two independently written copies of the same arithmetic) is
/// what guarantees they can't drift apart.
#[inline]
fn dim_point_bits(leaf_min_d: f64, leaf_max_d: f64) -> (u64, u8) {
    let base = f64_to_sortable_u64(leaf_min_d);
    let top = f64_to_sortable_u64(leaf_max_d);
    (base, bits_needed(top.wrapping_sub(base)))
}

/// LSB-first fixed-width bit packer used for both leaf point deltas and
/// doc_id deltas (see the [`BKD_VERSION`] format notes).
///
/// Widths `0..=64` are handled uniformly with no special-casing at either
/// boundary because the accumulator is `u128`: `(1u128 << width) - 1` never
/// overflows for `width` up to 64, unlike the `u64`-mask arithmetic a naive
/// implementation would reach for (which panics at `width == 64` and needs an
/// explicit branch at `width == 0`).
struct BitWriter {
    buf: Vec<u8>,
    acc: u128,
    acc_bits: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            buf: Vec::new(),
            acc: 0,
            acc_bits: 0,
        }
    }

    /// Append the low `width` bits of `value`. `width == 0` is a no-op.
    fn write(&mut self, value: u64, width: u8) {
        let mask = (1u128 << width) - 1;
        self.acc |= (value as u128 & mask) << self.acc_bits;
        self.acc_bits += width as u32;
        while self.acc_bits >= 8 {
            self.buf.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.acc_bits -= 8;
        }
    }

    /// Flush any partial trailing byte and return the packed buffer.
    fn finish(mut self) -> Vec<u8> {
        if self.acc_bits > 0 {
            self.buf.push((self.acc & 0xFF) as u8);
        }
        self.buf
    }
}

/// Mirrors [`BitWriter`]; reads back exactly the sequence of `write` calls
/// that produced `data`.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    acc: u128,
    acc_bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            acc: 0,
            acc_bits: 0,
        }
    }

    /// Read and return the next `width` bits (`0` for `width == 0`, no bytes
    /// consumed). The caller must supply a `data` slice with enough bytes
    /// for every `read` it intends to make — callers in this module size it
    /// exactly via [`packed_byte_len`], validated up-front by
    /// [`BKDReader::read_leaf_fixed_header`].
    fn read(&mut self, width: u8) -> u64 {
        while self.acc_bits < width as u32 {
            self.acc |= (self.data[self.byte_pos] as u128) << self.acc_bits;
            self.acc_bits += 8;
            self.byte_pos += 1;
        }
        let mask = (1u128 << width) - 1;
        let v = (self.acc & mask) as u64;
        self.acc >>= width as u32;
        self.acc_bits -= width as u32;
        v
    }
}

impl<W: StorageOutput> BKDWriter<W> {
    pub fn new(writer: W, num_dims: u32) -> Self {
        BKDWriter {
            writer: StructWriter::new(writer),
            block_size: 512,
            num_blocks: 0,
            num_dims,
            min_values: vec![f64::MAX; num_dims as usize],
            max_values: vec![f64::MIN; num_dims as usize],
            index_nodes: Vec::new(),
        }
    }

    /// Set custom block size
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Write a BKD tree from flat point/doc_id buffers.
    ///
    /// The `points` buffer is laid out as a row-major matrix of
    /// `doc_ids.len()` rows by `num_dims` columns: the d-th coordinate of the
    /// i-th point lives at `points[i * num_dims + d]`. `points.len()` must
    /// therefore equal `doc_ids.len() * num_dims`.
    ///
    /// Internally the builder sorts an index permutation rather than the
    /// point/doc_id buffers themselves, so no per-point heap allocation is
    /// performed regardless of point count.
    ///
    /// # Numeric robustness
    ///
    /// Coordinates must be totally orderable. `NaN` is rejected at write
    /// time with `LaurusError::index` because it has no defined ordering and
    /// would otherwise corrupt the BKD's split decisions and per-node AABB
    /// containment invariants. `f64::INFINITY` and `f64::NEG_INFINITY` are
    /// both accepted: they sort consistently against every finite value
    /// (`NEG_INFINITY < x < INFINITY`) and act as natural sentinels for
    /// "unbounded" semantics in queries (compare with [`AABB::unbounded`]).
    ///
    /// # Arguments
    /// - `points`: flat row-major buffer of f64 coordinates.
    /// - `doc_ids`: parallel buffer of document ids.
    ///
    /// # Returns
    /// `Ok(())` on success, otherwise a `LaurusError::index` describing the
    /// dimensional mismatch, the NaN position, or an underlying I/O error.
    pub fn write(&mut self, points: &[f64], doc_ids: &[u64]) -> Result<()> {
        let num_dims = self.num_dims as usize;
        let expected = doc_ids.len().checked_mul(num_dims).ok_or_else(|| {
            crate::error::LaurusError::index(
                "Point count overflows when multiplied by num_dims".to_string(),
            )
        })?;
        if points.len() != expected {
            return Err(crate::error::LaurusError::index(format!(
                "Point buffer size mismatch: expected {} doc_ids * {} dims = {} f64s, got {}",
                doc_ids.len(),
                num_dims,
                expected,
                points.len()
            )));
        }

        // The index permutation built below is `Vec<u32>`; reject inputs that
        // wouldn't fit before any I/O happens rather than silently truncating.
        let point_count = checked_point_count_u32(doc_ids.len())?;

        if doc_ids.is_empty() {
            // Write basic header for empty tree
            self.write_header(0, 0, 0)?;
            return Ok(());
        }

        // Reject any NaN coordinate up-front. NaN's `partial_cmp` is `None`,
        // so silently allowing it would corrupt sort order and AABB
        // containment in subtle, query-dependent ways.
        for (offset, &v) in points.iter().enumerate() {
            if v.is_nan() {
                let doc_idx = offset / num_dims;
                let dim = offset % num_dims;
                return Err(crate::error::LaurusError::index(format!(
                    "Point at doc index {doc_idx} dim {dim} is NaN; BKD requires \
                     totally-ordered values (NaN has no defined ordering)"
                )));
            }
        }

        // Calculate global min/max. Uses total-order comparison (not `f64::min`/
        // `f64::max`, which treat -0.0 and +0.0 as interchangeable) for the same
        // reason `compute_aabb` does — see `total_min`/`total_max`.
        for i in 0..doc_ids.len() {
            let base = i * num_dims;
            for d in 0..num_dims {
                let v = points[base + d];
                self.min_values[d] = total_min(self.min_values[d], v);
                self.max_values[d] = total_max(self.max_values[d], v);
            }
        }

        let total_count = doc_ids.len() as u64;

        // Reserve space for header:
        // Magic(4) + Version(4) + num_dims(4) + bytes_per_dim(4) + total_count(8) + num_blocks(8)
        // + block_size(4) + min_values(num_dims * 8) + max_values(num_dims * 8) + index_start(8) + root_offset(8)
        let header_size = 4 + 4 + 4 + 4 + 8 + 8 + 4 + (self.num_dims as u64 * 8 * 2) + 8 + 8;

        self.writer.write_u32(0)?; // Placeholder
        self.writer.seek(SeekFrom::Start(header_size))?;

        // Sort an index permutation instead of the data: this keeps the
        // point/doc_id buffers immutable and avoids per-point allocations.
        let mut indices: Vec<u32> = (0..point_count).collect();
        let ctx = BuildContext {
            points,
            doc_ids,
            num_dims,
        };
        let root_info = self.build_subtree(&ctx, &mut indices)?;

        // Write index section after all leaves
        let index_start_offset = self.writer.stream_position()?;
        self.write_index()?;

        let node_size = Self::node_size(self.num_dims);
        let root_node_offset = if let Some(idx) = root_info.node_idx {
            index_start_offset + (idx as u64) * node_size
        } else {
            // Single-leaf tree: the leaf was written immediately after the
            // header, so the "root" address is just past the header bytes.
            header_size
        };

        // Go back and write real header
        self.writer.seek(SeekFrom::Start(0))?;
        self.write_header(total_count, index_start_offset, root_node_offset)?;

        // Go back to end
        self.writer.seek(SeekFrom::End(0))?;

        Ok(())
    }

    fn write_header(&mut self, total_count: u64, index_start: u64, root_offset: u64) -> Result<()> {
        self.writer.write_u32(BKD_MAGIC)?;
        self.writer.write_u32(BKD_VERSION)?;
        self.writer.write_u32(self.num_dims)?;
        self.writer.write_u32(8)?; // Bytes per dim (f64)
        self.writer.write_u64(total_count)?;
        self.writer.write_u64(self.num_blocks)?;
        // Issue #1142: bounds a leaf's `count` on read against the writer's
        // actual per-leaf cap, tightening the pre-existing (and, since this
        // revision shrinks `doc_id_bits`, more reachable) gap where a leaf
        // with zero point bits and zero doc_id bits left `count` completely
        // unchecked.
        self.writer.write_u32(self.block_size as u32)?;
        for &v in &self.min_values {
            self.writer.write_f64(v)?;
        }
        for &v in &self.max_values {
            self.writer.write_f64(v)?;
        }
        self.writer.write_u64(index_start)?;
        self.writer.write_u64(root_offset)?;
        Ok(())
    }

    /// Returns the on-disk byte size of one internal index node.
    ///
    /// Each dimension contributes 32 bytes (left_min, left_max, right_min,
    /// right_max — four f64 values per dimension) on top of the fixed 28-byte
    /// split / offset header, matching the layout documented on
    /// [`BKD_VERSION`].
    #[inline]
    fn node_size(num_dims: u32) -> u64 {
        28 + 32 * num_dims as u64
    }

    /// Recursively build a subtree, writing leaves on the fly and recording
    /// internal nodes in `self.index_nodes` for back-patching. The slice
    /// `indices` is a permutation of point ids that this call owns and is
    /// allowed to reorder; recursion proceeds on the two halves of the
    /// permutation around the split position.
    ///
    /// The AABB of the points covered by `indices` is computed up-front and
    /// reused for two purposes: a leaf call writes it as `leaf_min`/`leaf_max`,
    /// and an internal call uses it both to pick the widest axis as the split
    /// dimension and to populate its own [`SubtreeInfo`] without unioning the
    /// children afterwards.
    fn build_subtree(
        &mut self,
        ctx: &BuildContext<'_>,
        indices: &mut [u32],
    ) -> Result<SubtreeInfo> {
        if indices.is_empty() {
            // The recursion only descends into non-empty halves (we only split
            // when len > block_size, where the smaller half has at least one
            // element), so reaching this branch indicates a programmer error.
            return Err(crate::error::LaurusError::index(
                "build_subtree called with empty indices".to_string(),
            ));
        }

        let (subtree_min, subtree_max) = compute_aabb(ctx, indices);

        if indices.len() <= self.block_size {
            self.write_leaf_block(ctx, indices, &subtree_min, &subtree_max)?;
            self.num_blocks += 1;
            return Ok(SubtreeInfo {
                node_idx: None,
                min: subtree_min,
                max: subtree_max,
            });
        }

        // Split on the axis with the widest range, mirroring Lucene BKD.
        // For uniformly distributed data this collapses to the previous
        // round-robin (`depth % num_dims`) pattern, but for skewed data
        // (e.g. lat/lon paired with a tiny altitude) it concentrates splits
        // on the axis where they actually shrink the search box.
        let split_dim = widest_axis(&subtree_min, &subtree_max);
        let split_dim_us = split_dim as usize;

        // Sort the permutation by the split dimension to find the median.
        // The underlying point/doc_id buffers stay immutable; only `indices`
        // is reordered. `total_cmp` is safe here because `BKDWriter::write`
        // has already rejected NaN coordinates, so every f64 in `ctx.points`
        // is totally ordered.
        indices.sort_by(|&a, &b| {
            ctx.value(a, split_dim_us)
                .total_cmp(&ctx.value(b, split_dim_us))
        });

        // Internal nodes are written AFTER all leaves; we track tree structure
        // in `index_nodes` here and back-patch offsets in `write_index`.
        let mid = indices.len() / 2;
        let (left_indices, right_indices) = indices.split_at_mut(mid);
        let split_value = ctx.value(right_indices[0], split_dim_us);

        // Reserve a slot for this internal node now so the index_nodes vector
        // is stable across recursive calls. AABB / offset / child fields are
        // back-patched once both children have been built. The AABB Vec
        // placeholders use `Vec::new()` (a const, no-alloc constructor)
        // because they are immediately overwritten by moves from the
        // children's `SubtreeInfo` once recursion returns — preallocating
        // capacity here would just be discarded.
        let node_idx = self.index_nodes.len();
        self.index_nodes.push(IndexNode {
            split_dim,
            split_value,
            left_min: Vec::new(),
            left_max: Vec::new(),
            right_min: Vec::new(),
            right_max: Vec::new(),
            left_offset: 0,
            right_offset: 0,
            left_child_idx: None,
            right_child_idx: None,
        });

        let left_file_pos_before = self.writer.stream_position()?;
        let left_info = self.build_subtree(ctx, left_indices)?;
        let left_is_leaf = left_info.node_idx.is_none();

        let right_file_pos_before = self.writer.stream_position()?;
        let right_info = self.build_subtree(ctx, right_indices)?;
        let right_is_leaf = right_info.node_idx.is_none();

        // Update the previously reserved node slot. The parent's AABB was
        // computed up-front so we no longer need to union the child AABBs.
        let node = &mut self.index_nodes[node_idx];
        node.left_child_idx = left_info.node_idx;
        node.right_child_idx = right_info.node_idx;
        node.left_min = left_info.min;
        node.left_max = left_info.max;
        node.right_min = right_info.min;
        node.right_max = right_info.max;
        if left_is_leaf {
            node.left_offset = left_file_pos_before;
        }
        if right_is_leaf {
            node.right_offset = right_file_pos_before;
        }

        Ok(SubtreeInfo {
            node_idx: Some(node_idx),
            min: subtree_min,
            max: subtree_max,
        })
    }

    fn write_leaf_block(
        &mut self,
        ctx: &BuildContext<'_>,
        indices: &mut [u32],
        leaf_min: &[f64],
        leaf_max: &[f64],
    ) -> Result<()> {
        let count = indices.len() as u32;
        self.writer.write_u32(count)?;

        // Per-leaf AABB, used by the reader for subtree pruning starting
        // from #292, and (since #549) as the anchor for delta-from-min
        // bit-packing below. Computed by `compute_aabb` over `indices` in
        // `build_subtree` *before* this function's doc_id sort below runs
        // (order-independent min/max scan), so the sort cannot affect it.
        for &v in leaf_min {
            self.writer.write_f64(v)?;
        }
        for &v in leaf_max {
            self.writer.write_f64(v)?;
        }

        // Issue #1142: sort this leaf's indices by doc_id so doc_ids can be
        // packed as consecutive deltas (`bits_needed(max consecutive gap)`)
        // instead of independent deltas from a single anchor
        // (`bits_needed(max - min)`) -- the former is never larger and often
        // much smaller for locally-clustered doc_id distributions, since
        // deltas telescope-sum to exactly `max - min`.
        //
        // This MUST be a stable sort, not `sort_unstable_by_key`: when the
        // same doc_id contributes more than one point to this leaf (a
        // multi-valued field), `GeoBoxPointsVisitor::into_candidates`
        // (`lexical/query/geo.rs`) dedups by "first one seen in leaf
        // traversal order wins". A stable sort preserves each doc's
        // duplicate points in their original (split-dimension-sorted)
        // relative order, so which point wins is unchanged by this
        // revision; an unstable sort could silently change it.
        //
        // The point-packing loop below iterates `indices` after this sort,
        // so it inherits doc_id order automatically -- nothing else reads
        // `indices` between here and the end of this function.
        indices.sort_by_key(|&i| ctx.doc_ids[i as usize]);

        // doc_id_base / doc_id_bits: the one on-disk bit-width field (every
        // other width is derived from leaf_min/leaf_max on both sides, see
        // `BKD_VERSION`'s doc comment). `indices` is now sorted ascending by
        // doc_id, so the endpoints give min/max in O(1).
        let doc_id_min = ctx.doc_ids[indices[0] as usize];
        let mut max_delta = 0u64;
        for w in indices.windows(2) {
            let d = ctx.doc_ids[w[1] as usize] - ctx.doc_ids[w[0] as usize];
            max_delta = max_delta.max(d);
        }
        let doc_id_bits = bits_needed(max_delta);
        self.writer.write_u64(doc_id_min)?;
        self.writer.write_u8(doc_id_bits)?;

        // Per-dimension packed points: delta from `leaf_min[d]` in sortable
        // (total-order) space, fixed width. A constant dimension needs 0
        // bits and contributes no bytes at all. Iterates the now doc_id
        // -sorted `indices`, so points are stored in doc_id-ascending order.
        for d in 0..ctx.num_dims {
            let (base, bits) = dim_point_bits(leaf_min[d], leaf_max[d]);
            if bits == 0 {
                continue;
            }
            let mut bw = BitWriter::new();
            for &i in indices.iter() {
                let v = f64_to_sortable_u64(ctx.value(i, d));
                bw.write(v - base, bits);
            }
            self.writer.write_raw(&bw.finish())?;
        }

        // Packed doc_ids as consecutive deltas: value 0 is always
        // `doc_id_min - doc_id_min == 0` (the leading delta the reader
        // treats as a corruption signal if it's ever nonzero), and each
        // subsequent value is the gap from its immediate predecessor.
        if doc_id_bits > 0 {
            let mut bw = BitWriter::new();
            let mut prev = doc_id_min;
            for &i in indices.iter() {
                let d = ctx.doc_ids[i as usize];
                debug_assert!(
                    d >= prev,
                    "write_leaf_block: indices must be sorted ascending by doc_id \
                     before this loop runs"
                );
                bw.write(d - prev, doc_id_bits);
                prev = d;
            }
            self.writer.write_raw(&bw.finish())?;
        }

        Ok(())
    }

    fn write_index(&mut self) -> Result<()> {
        let start_pos = self.writer.stream_position()?;
        let node_size = Self::node_size(self.num_dims);

        for i in 0..self.index_nodes.len() {
            let left_idx = self.index_nodes[i].left_child_idx;
            if let Some(idx) = left_idx {
                self.index_nodes[i].left_offset = start_pos + (idx as u64) * node_size;
            }

            let right_idx = self.index_nodes[i].right_child_idx;
            if let Some(idx) = right_idx {
                self.index_nodes[i].right_offset = start_pos + (idx as u64) * node_size;
            }
        }

        // Write nodes in the layout documented on `BKD_VERSION`.
        for node in &self.index_nodes {
            self.writer.write_u32(node.split_dim)?;
            self.writer.write_f64(node.split_value)?;
            for &v in &node.left_min {
                self.writer.write_f64(v)?;
            }
            for &v in &node.left_max {
                self.writer.write_f64(v)?;
            }
            for &v in &node.right_min {
                self.writer.write_f64(v)?;
            }
            for &v in &node.right_max {
                self.writer.write_f64(v)?;
            }
            self.writer.write_u64(node.left_offset)?;
            self.writer.write_u64(node.right_offset)?;
        }

        Ok(())
    }

    /// Finish writing and return the underlying writer.
    pub fn finish(self) -> Result<()> {
        self.writer.close()
    }
}

/// Reader for BKD Trees.
#[derive(Debug)]
pub struct BKDReader {
    header: BKDFileHeader,
    storage: Arc<dyn Storage>,
    path: String,
}

impl BKDReader {
    /// Borrow the file header parsed at `open` time.
    ///
    /// Useful for callers that need to inspect the dimensionality, point
    /// count, or global AABB of an existing tree without performing a
    /// query (e.g. integration tests, schema introspection tooling).
    pub fn header(&self) -> &BKDFileHeader {
        &self.header
    }
}

impl BKDReader {
    /// Open a BKD tree from storage and path.
    pub fn open(storage: Arc<dyn Storage>, path: &str) -> Result<Self> {
        let input = storage.open_input(path)?;
        let mut reader = StructReader::new(input)?;

        // Read header
        let magic = reader.read_u32()?;
        if magic != BKD_MAGIC {
            return Err(crate::error::LaurusError::storage(format!(
                "Invalid BKD magic: {:x}",
                magic
            )));
        }

        let version = reader.read_u32()?;
        if version != BKD_VERSION {
            return Err(crate::error::LaurusError::storage(format!(
                "Unsupported BKD version: {} (expected {}). Pre-release format \
                 changes do not support older revisions; rebuild the index.",
                version, BKD_VERSION
            )));
        }
        let num_dims = reader.read_u32()?;
        let bytes_per_dim = reader.read_u32()?;
        if bytes_per_dim != 8 {
            return Err(crate::error::LaurusError::storage(format!(
                "Unsupported BKD bytes_per_dim: {bytes_per_dim} (expected 8) — segment is corrupted"
            )));
        }
        // Each dimension contributes at least 8 bytes to `min_values` and 8
        // to `max_values` before the header is fully read; bound `num_dims`
        // against the file's remaining bytes so a corrupted header can't
        // drive a multi-gigabyte `Vec::with_capacity` below.
        let available = reader.size().saturating_sub(reader.position());
        let num_dims = checked_capacity(num_dims as usize, 16, available, "BKD num_dims")? as u32;
        let total_point_count = reader.read_u64()?;
        let num_blocks = reader.read_u64()?;
        let block_size = reader.read_u32()?;
        let mut min_values = Vec::with_capacity(num_dims as usize);
        for _ in 0..num_dims {
            min_values.push(reader.read_f64()?);
        }
        let mut max_values = Vec::with_capacity(num_dims as usize);
        for _ in 0..num_dims {
            max_values.push(reader.read_f64()?);
        }
        let index_start_offset = reader.read_u64()?;
        let root_node_offset = reader.read_u64()?;

        let header = BKDFileHeader {
            magic,
            version,
            num_dims,
            bytes_per_dim,
            total_point_count,
            num_blocks,
            block_size,
            min_values,
            max_values,
            index_start_offset,
            root_node_offset,
        };

        Ok(BKDReader {
            header,
            storage,
            path: path.to_string(),
        })
    }

    /// Read `num_dims` `f64` values for `min` followed by `num_dims` for
    /// `max`, returning the constructed AABB.
    fn read_child_aabb<R: StorageInput>(
        reader: &mut StructReader<R>,
        num_dims: usize,
    ) -> Result<AABB> {
        let mut min = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            min.push(reader.read_f64()?);
        }
        let mut max = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            max.push(reader.read_f64()?);
        }
        AABB::new(min, max)
    }

    /// Walk the subtree rooted at `offset`, dispatching pruning decisions
    /// to `visitor`. Internal nodes consult `visitor.compare` on each
    /// child's AABB; leaves either short-circuit (Outside / Inside) or
    /// stream every (doc_id, point) candidate through `visitor.visit`.
    ///
    /// The `scratch` buffer is reused across every leaf visited in this
    /// query, so steady-state queries on similarly-sized leaves run
    /// allocation-free after the first `Crosses` leaf.
    fn intersect_subtree<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
        scratch: &mut IntersectScratch,
    ) -> Result<()> {
        if offset < self.header.index_start_offset {
            return self.intersect_leaf(reader, offset, visitor, scratch);
        }
        let num_dims = self.header.num_dims as usize;
        reader.seek(SeekFrom::Start(offset))?;
        let _split_dim = reader.read_u32()?;
        let _split_value = reader.read_f64()?;
        let left_aabb = Self::read_child_aabb(reader, num_dims)?;
        let right_aabb = Self::read_child_aabb(reader, num_dims)?;
        let left_offset = reader.read_u64()?;
        let right_offset = reader.read_u64()?;

        match visitor.compare(&left_aabb) {
            CellRelation::Outside => {}
            CellRelation::Inside => self.collect_subtree(reader, left_offset, visitor)?,
            CellRelation::Crosses => {
                self.intersect_subtree(reader, left_offset, visitor, scratch)?
            }
        }
        match visitor.compare(&right_aabb) {
            CellRelation::Outside => {}
            CellRelation::Inside => self.collect_subtree(reader, right_offset, visitor)?,
            CellRelation::Crosses => {
                self.intersect_subtree(reader, right_offset, visitor, scratch)?
            }
        }
        Ok(())
    }

    /// Walk a leaf at `offset`, classifying it via `visitor.compare` on the
    /// stored leaf AABB and dispatching points accordingly.
    ///
    /// On `Crosses`, leaf points are decoded into the caller-supplied
    /// `scratch.points` buffer (grown only on the first leaf large enough
    /// to need it), so steady-state queries on similarly-sized leaves run
    /// allocation-free after the first `Crosses` leaf.
    fn intersect_leaf<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
        scratch: &mut IntersectScratch,
    ) -> Result<()> {
        reader.seek(SeekFrom::Start(offset))?;
        let num_dims = self.header.num_dims as usize;
        let leaf_header = Self::read_leaf_fixed_header(
            reader,
            num_dims,
            self.header.total_point_count,
            self.header.block_size,
        )?;

        match visitor.compare(&leaf_header.leaf_aabb) {
            CellRelation::Outside => Ok(()),
            CellRelation::Inside => {
                Self::skip_points(reader, &leaf_header)?;
                Self::decode_doc_ids_and_dispatch(reader, &leaf_header, None, num_dims, visitor)
            }
            CellRelation::Crosses => {
                let points_buf = scratch.point_slice(leaf_header.count * num_dims);
                Self::decode_points_into(reader, &leaf_header, num_dims, points_buf)?;
                Self::decode_doc_ids_and_dispatch(
                    reader,
                    &leaf_header,
                    Some(&*points_buf),
                    num_dims,
                    visitor,
                )
            }
        }
    }

    /// Walk a subtree whose root the caller has already classified as
    /// `Inside`. No `compare` calls are made — every doc is reported via
    /// `visit_inside` and the leaf point bytes are skipped entirely.
    fn collect_subtree<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        if offset < self.header.index_start_offset {
            return self.collect_leaf(reader, offset, visitor);
        }
        let num_dims = self.header.num_dims as usize;
        reader.seek(SeekFrom::Start(offset))?;
        let _split_dim = reader.read_u32()?;
        let _split_value = reader.read_f64()?;
        // Skip both child AABBs.
        let aabb_bytes = (num_dims as u64) * 16 * 2;
        reader.seek(SeekFrom::Current(aabb_bytes as i64))?;
        let left_offset = reader.read_u64()?;
        let right_offset = reader.read_u64()?;
        self.collect_subtree(reader, left_offset, visitor)?;
        self.collect_subtree(reader, right_offset, visitor)?;
        Ok(())
    }

    /// Walk a leaf whose enclosing cell has already been classified as
    /// `Inside`. The leaf AABB and point bytes are skipped; only doc ids
    /// are decoded and reported via `visit_inside`.
    fn collect_leaf<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        reader.seek(SeekFrom::Start(offset))?;
        let num_dims = self.header.num_dims as usize;
        let leaf_header = Self::read_leaf_fixed_header(
            reader,
            num_dims,
            self.header.total_point_count,
            self.header.block_size,
        )?;
        Self::skip_points(reader, &leaf_header)?;
        Self::decode_doc_ids_and_dispatch(reader, &leaf_header, None, num_dims, visitor)
    }

    /// Read a leaf's fixed-size header fields (`count`, AABB, doc_id
    /// anchor), derive each dimension's bit width from the AABB, compute
    /// every packed section's exact byte length, and bounds-check the total
    /// against the bytes actually left in the file before any of it is
    /// used to size a read or a buffer.
    fn read_leaf_fixed_header<R: StorageInput>(
        reader: &mut StructReader<R>,
        num_dims: usize,
        total_point_count: u64,
        block_size: u32,
    ) -> Result<LeafHeader> {
        let count = reader.read_u32()? as usize;
        if count as u64 > total_point_count {
            return Err(crate::error::LaurusError::index(format!(
                "BKD leaf: declares {count} points but the tree has only \
                 {total_point_count} total — segment is corrupted"
            )));
        }
        // Issue #1142: independent of the packed-length bound below, which
        // this revision's smaller `doc_id_bits` values weaken (a leaf with
        // zero point bits and zero doc_id bits has `total_packed_len == 0`,
        // leaving `count` otherwise unchecked against physical file size).
        if count as u64 > block_size as u64 {
            return Err(crate::error::LaurusError::index(format!(
                "BKD leaf: declares {count} points but block_size is only \
                 {block_size} — segment is corrupted"
            )));
        }
        let leaf_aabb = Self::read_child_aabb(reader, num_dims)?;
        let doc_id_base = reader.read_u64()?;
        let doc_id_bits = reader.read_u8()?;
        if doc_id_bits > 64 {
            return Err(crate::error::LaurusError::index(format!(
                "BKD leaf: doc_id_bits = {doc_id_bits} exceeds 64 — segment is corrupted"
            )));
        }

        let mut point_base = Vec::with_capacity(num_dims);
        let mut point_bits = Vec::with_capacity(num_dims);
        let mut point_section_lens = Vec::with_capacity(num_dims);
        for d in 0..num_dims {
            let (base, bits) = dim_point_bits(leaf_aabb.min()[d], leaf_aabb.max()[d]);
            point_base.push(base);
            point_bits.push(bits);
            point_section_lens.push(packed_byte_len(count, bits));
        }
        let doc_id_section_len = packed_byte_len(count, doc_id_bits);

        let total_packed_len: usize = point_section_lens.iter().sum::<usize>() + doc_id_section_len;
        let available = reader.size().saturating_sub(reader.position());
        checked_len(total_packed_len, available, "BKD leaf packed data")?;

        Ok(LeafHeader {
            count,
            leaf_aabb,
            doc_id_base,
            doc_id_bits,
            point_base,
            point_bits,
            point_section_lens,
            doc_id_section_len,
        })
    }

    /// Skip past a leaf's packed point sections in a single seek (the
    /// `Inside` / `collect_leaf` path, which only needs doc ids).
    fn skip_points<R: StorageInput>(
        reader: &mut StructReader<R>,
        header: &LeafHeader,
    ) -> Result<()> {
        let total: usize = header.point_section_lens.iter().sum();
        if total > 0 {
            reader.seek(SeekFrom::Current(total as i64))?;
        }
        Ok(())
    }

    /// Decode every dimension's packed point section into `points_buf`
    /// (point-major, matching `IntersectScratch`'s layout).
    fn decode_points_into<R: StorageInput>(
        reader: &mut StructReader<R>,
        header: &LeafHeader,
        num_dims: usize,
        points_buf: &mut [f64],
    ) -> Result<()> {
        let count = header.count;
        for d in 0..num_dims {
            let len = header.point_section_lens[d];
            let bits = header.point_bits[d];
            let base = header.point_base[d];
            if len == 0 {
                // bits == 0: every point in this dimension equals leaf_min[d].
                let v = sortable_u64_to_f64(base);
                for i in 0..count {
                    points_buf[i * num_dims + d] = v;
                }
                continue;
            }
            reader.read_raw_with(len, |raw: &[u8]| {
                let mut bit_reader = BitReader::new(raw);
                for i in 0..count {
                    let delta = bit_reader.read(bits);
                    points_buf[i * num_dims + d] = sortable_u64_to_f64(base.wrapping_add(delta));
                }
            })?;
        }
        Ok(())
    }

    /// Decode a leaf's packed doc_id section and report each doc id via
    /// `visitor.visit` (when `points_buf` is `Some`, i.e. the `Crosses`
    /// path) or `visitor.visit_inside` (the `Inside` path, `points_buf`
    /// is `None`).
    fn decode_doc_ids_and_dispatch<R: StorageInput>(
        reader: &mut StructReader<R>,
        header: &LeafHeader,
        points_buf: Option<&[f64]>,
        num_dims: usize,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        let count = header.count;
        let base = header.doc_id_base;
        let bits = header.doc_id_bits;
        let len = header.doc_id_section_len;

        let dispatch = |visitor: &mut dyn IntersectVisitor, doc_id: u64, i: usize| match points_buf
        {
            Some(buf) => visitor.visit(doc_id, &buf[i * num_dims..(i + 1) * num_dims]),
            None => visitor.visit_inside(doc_id),
        };

        if len == 0 {
            // bits == 0: every doc_id in this leaf equals doc_id_base.
            for i in 0..count {
                dispatch(visitor, base, i);
            }
            return Ok(());
        }

        reader.read_raw_with(len, |raw: &[u8]| -> Result<()> {
            let mut bit_reader = BitReader::new(raw);
            let mut running = base;
            for i in 0..count {
                let delta = bit_reader.read(bits);
                // Issue #1142: doc_ids are packed as consecutive deltas
                // (value i is `doc_id[i-1] + delta`, with `doc_id[-1] :=
                // base`), so decoding accumulates rather than adding to a
                // fixed anchor. The very first delta is always 0 by
                // construction (the leaf's smallest doc_id, sorted to
                // position 0, has nothing before it to differ from) --
                // a nonzero value here can only mean the file is
                // corrupted or was written by a non-conforming encoder,
                // since a conforming writer can never produce one.
                if i == 0 && delta != 0 {
                    return Err(crate::error::LaurusError::index(
                        "BKD leaf: first packed doc_id delta must be 0 — segment is corrupted"
                            .to_string(),
                    ));
                }
                running = running.wrapping_add(delta);
                dispatch(visitor, running, i);
            }
            Ok(())
        })?
    }
}

/// Fixed-size fields read from a leaf's header, plus the per-section byte
/// lengths derived from them. See
/// [`BKDReader::read_leaf_fixed_header`].
struct LeafHeader {
    count: usize,
    leaf_aabb: AABB,
    doc_id_base: u64,
    doc_id_bits: u8,
    /// Sortable-space anchor per dimension (`f64_to_sortable_u64(leaf_min[d])`).
    point_base: Vec<u64>,
    /// Bit width per dimension, derived from `leaf_aabb` (never stored on
    /// disk — writer and reader compute it with the same formula, so the
    /// two sides can't drift apart).
    point_bits: Vec<u8>,
    /// Byte length of each dimension's packed section, in order.
    point_section_lens: Vec<usize>,
    /// Byte length of the packed doc_id section.
    doc_id_section_len: usize,
}

/// Per-query scratch buffer reused across every leaf visited by
/// [`BKDReader::intersect`]. Holding the buffer outside the recursion lets
/// `Crosses` leaves reuse the same allocation instead of allocating a fresh
/// `Vec<f64>` of `count * num_dims` floats per leaf.
struct IntersectScratch {
    /// Backing storage for leaf point bytes. Grown on demand to the largest
    /// leaf encountered, never shrunk during one query.
    points: Vec<f64>,
}

impl IntersectScratch {
    fn new() -> Self {
        IntersectScratch { points: Vec::new() }
    }

    /// Return a mutable slice with at least `len` elements, growing the
    /// backing buffer with `Vec::resize` if needed. Returned slice always
    /// has exactly `len` elements.
    fn point_slice(&mut self, len: usize) -> &mut [f64] {
        if self.points.len() < len {
            self.points.resize(len, 0.0);
        }
        &mut self.points[..len]
    }
}

impl BKDTree for BKDReader {
    fn value_range(&self, dim: usize) -> Option<(f64, f64, u64)> {
        // The global AABB is computed at write time and parsed with the
        // header at `open`, so this costs nothing beyond two indexed
        // reads (#944).
        if self.header.total_point_count == 0 {
            return None;
        }
        let min = *self.header.min_values.get(dim)?;
        let max = *self.header.max_values.get(dim)?;
        Some((min, max, self.header.total_point_count))
    }

    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        if self.header.total_point_count == 0 {
            return Ok(());
        }
        let input = self.storage.open_input(&self.path)?;
        let mut reader = StructReader::new(input)?;
        let root_offset = self.header.root_node_offset;
        let mut scratch = IntersectScratch::new();
        if root_offset < self.header.index_start_offset {
            // Single-leaf tree: the root "address" is just past the header.
            self.intersect_leaf(&mut reader, root_offset, visitor, &mut scratch)
        } else {
            self.intersect_subtree(&mut reader, root_offset, visitor, &mut scratch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;

    #[test]
    fn test_bkd_writer_reader_2d() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // Flat row-major buffer: [pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y]
        let points: Vec<f64> = vec![10.0, 20.0, 15.0, 25.0, 20.0, 30.0];
        let doc_ids: Vec<u64> = vec![1, 2, 3];

        // Write
        {
            let output = storage.create_output("test_2d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        // Read
        {
            let reader = BKDReader::open(storage.clone(), "test_2d.bkd").unwrap();
            assert_eq!(reader.header.num_dims, 2);

            // Search [10, 10] to [15, 25]
            let results = reader
                .range_search(
                    &[Some(10.0), Some(10.0)],
                    &[Some(15.0), Some(25.0)],
                    true,
                    true,
                )
                .unwrap();
            assert_eq!(results, vec![1, 2]);
        }
    }

    #[test]
    fn test_bkd_writer_empty() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let points: Vec<f64> = vec![];
        let doc_ids: Vec<u64> = vec![];

        {
            let output = storage.create_output("empty.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "empty.bkd").unwrap();
        assert_eq!(reader.header.total_point_count, 0);
        let results = reader
            .range_search(&[None, None], &[None, None], true, true)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bkd_writer_size_mismatch_rejected() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // 2 doc_ids in 2D would require 4 f64s, but we pass 3.
        let points: Vec<f64> = vec![1.0, 2.0, 3.0];
        let doc_ids: Vec<u64> = vec![10, 20];

        let output = storage.create_output("bad.bkd").unwrap();
        let mut writer = BKDWriter::new(output, 2);
        let err = writer.write(&points, &doc_ids).unwrap_err();
        assert!(
            format!("{err:?}").contains("Point buffer size mismatch"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_bkd_writer_reader_1d_multi_block() {
        // Exercise the recursive build path with more points than the leaf
        // block size so the index/leaf split is actually visited.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 2_000;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();

        {
            let output = storage.create_output("range1d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(128);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "range1d.bkd").unwrap();
        let results = reader
            .range_search(&[Some(100.0)], &[Some(200.0)], true, true)
            .unwrap();
        let expected: Vec<u64> = (100u64..=200u64).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_bkd_writer_reader_3d_multi_block() {
        // 3D round-trip with multiple leaf blocks: the new per-node /
        // per-leaf AABB layout must round-trip without misaligning offsets.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 1_000;
        let mut points: Vec<f64> = Vec::with_capacity(n * 3);
        let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f64;
            points.push(v);
            points.push(v + 1000.0);
            points.push(v + 2000.0);
            doc_ids.push(i as u64);
        }

        {
            let output = storage.create_output("range3d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 3).with_block_size(64);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "range3d.bkd").unwrap();
        assert_eq!(reader.header.num_dims, 3);
        assert_eq!(reader.header.version, BKD_VERSION);

        // Half-open bound on the first axis with an upper-only bound on the
        // second axis to make sure visit_node consumes the AABB bytes
        // correctly even with mixed bounded/unbounded dimensions.
        let results = reader
            .range_search(
                &[Some(100.0), None, None],
                &[Some(150.0), Some(1200.0), None],
                true,
                true,
            )
            .unwrap();
        let expected: Vec<u64> = (100u64..=150u64)
            .filter(|&i| (i as f64) + 1000.0 <= 1200.0)
            .collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_bkd_reader_rejects_version_mismatch() {
        // Hand-craft a header that claims version 3 (the independent-delta
        // doc_id layout this revision, #1142, retires in favor of
        // consecutive deltas) and confirm the reader refuses to open it.
        // The version check runs immediately after reading `version` --
        // before `block_size` or anything else -- so this file need not
        // carry a `block_size` field to exercise it.
        use crate::storage::structured::StructWriter;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("v3.bkd").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(BKD_MAGIC).unwrap();
            writer.write_u32(3).unwrap(); // retired version (pre-#1142)
            writer.write_u32(2).unwrap(); // num_dims
            writer.write_u32(8).unwrap(); // bytes_per_dim
            writer.write_u64(0).unwrap(); // total_count
            writer.write_u64(0).unwrap(); // num_blocks
            writer.write_f64(0.0).unwrap(); // global_min[0]
            writer.write_f64(0.0).unwrap(); // global_min[1]
            writer.write_f64(0.0).unwrap(); // global_max[0]
            writer.write_f64(0.0).unwrap(); // global_max[1]
            writer.write_u64(0).unwrap(); // index_start
            writer.write_u64(0).unwrap(); // root_offset
            writer.close().unwrap();
        }

        let err = BKDReader::open(storage.clone(), "v3.bkd").unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Unsupported BKD version"),
            "unexpected error: {msg}"
        );
    }

    /// Visitor that segregates hits by which BKD code path produced them
    /// (`visit_inside` vs `visit`), used to assert that `Inside` cells avoid
    /// per-point filtering and `Crosses` leaves go through it.
    struct TracingVisitor {
        query: AABB,
        inside_hits: Vec<u64>,
        crosses_hits: Vec<u64>,
    }

    impl TracingVisitor {
        fn new(query: AABB) -> Self {
            Self {
                query,
                inside_hits: Vec::new(),
                crosses_hits: Vec::new(),
            }
        }
    }

    impl IntersectVisitor for TracingVisitor {
        fn compare(&self, cell: &AABB) -> CellRelation {
            // Conservative compare: cell vs query (closed intervals).
            let qmin = self.query.min();
            let qmax = self.query.max();
            let cmin = cell.min();
            let cmax = cell.max();
            for d in 0..cell.num_dims() {
                if cmax[d] < qmin[d] || cmin[d] > qmax[d] {
                    return CellRelation::Outside;
                }
            }
            for d in 0..cell.num_dims() {
                if cmin[d] < qmin[d] || cmax[d] > qmax[d] {
                    return CellRelation::Crosses;
                }
            }
            CellRelation::Inside
        }
        fn visit_inside(&mut self, doc_id: u64) {
            self.inside_hits.push(doc_id);
        }
        fn visit(&mut self, doc_id: u64, point: &[f64]) {
            if self.query.contains_point(point) {
                self.crosses_hits.push(doc_id);
            }
        }
    }

    /// A wrapper that also records every `compare` outcome (including
    /// `Outside`) by using a `Cell` for interior mutability.
    struct RecordingVisitor {
        query: AABB,
        relations: std::cell::RefCell<Vec<CellRelation>>,
        hits: Vec<u64>,
    }

    impl RecordingVisitor {
        fn new(query: AABB) -> Self {
            Self {
                query,
                relations: std::cell::RefCell::new(Vec::new()),
                hits: Vec::new(),
            }
        }
    }

    impl IntersectVisitor for RecordingVisitor {
        fn compare(&self, cell: &AABB) -> CellRelation {
            let qmin = self.query.min();
            let qmax = self.query.max();
            let cmin = cell.min();
            let cmax = cell.max();
            let mut relation = CellRelation::Inside;
            for d in 0..cell.num_dims() {
                if cmax[d] < qmin[d] || cmin[d] > qmax[d] {
                    relation = CellRelation::Outside;
                    break;
                }
            }
            if !matches!(relation, CellRelation::Outside) {
                for d in 0..cell.num_dims() {
                    if cmin[d] < qmin[d] || cmax[d] > qmax[d] {
                        relation = CellRelation::Crosses;
                        break;
                    }
                }
            }
            self.relations.borrow_mut().push(relation);
            relation
        }
        fn visit_inside(&mut self, doc_id: u64) {
            self.hits.push(doc_id);
        }
        fn visit(&mut self, doc_id: u64, point: &[f64]) {
            // For Crosses cells, accept the point only if it actually lies
            // inside the query.
            if self.query.contains_point(point) {
                self.hits.push(doc_id);
            }
        }
    }

    #[test]
    fn widest_axis_picks_largest_range() {
        // Free-function smoke test (doesn't go through the writer).
        assert_eq!(widest_axis(&[0.0, 0.0], &[10.0, 100.0]), 1);
        assert_eq!(widest_axis(&[0.0, 0.0], &[100.0, 10.0]), 0);
        // Tie: lower-index dimension wins (deterministic).
        assert_eq!(widest_axis(&[0.0, 0.0], &[5.0, 5.0]), 0);
        // 3D, middle axis widest.
        assert_eq!(widest_axis(&[0.0, 0.0, 0.0], &[1.0, 50.0, 10.0]), 1);
    }

    #[test]
    fn build_subtree_root_split_is_widest_axis() {
        // 2D dataset where dim 0 spans 0..n and dim 1 stays in [0, 1).
        // The widest-axis policy must pick dim 0 for the root split,
        // unlike the previous round-robin which would also pick dim 0
        // at depth 0 by accident — so we confirm by also testing the
        // mirrored dataset where dim 1 is widest.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 256;

        // Wider on dim 0.
        let mut points: Vec<f64> = Vec::with_capacity(n * 2);
        let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            points.push(i as f64);
            points.push(0.0); // narrow: every point shares the same dim 1
            doc_ids.push(i as u64);
        }
        {
            let output = storage.create_output("wide_dim0.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2).with_block_size(32);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        // Mirrored: wider on dim 1.
        points.clear();
        doc_ids.clear();
        for i in 0..n {
            points.push(0.0);
            points.push(i as f64);
            doc_ids.push(i as u64);
        }
        {
            let output = storage.create_output("wide_dim1.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2).with_block_size(32);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        // Helper that reads the root index node's split_dim straight from
        // disk — the root sits at `index_start_offset` because it is the
        // first node pushed into `index_nodes`.
        fn root_split_dim(storage: &Arc<MemoryStorage>, path: &str) -> u32 {
            let reader = BKDReader::open(storage.clone(), path).unwrap();
            let index_start = reader.header.index_start_offset;
            let input = storage.open_input(path).unwrap();
            let mut sr = StructReader::new(input).unwrap();
            sr.seek(SeekFrom::Start(index_start)).unwrap();
            sr.read_u32().unwrap()
        }

        assert_eq!(
            root_split_dim(&storage, "wide_dim0.bkd"),
            0,
            "root should split on dim 0 when dim 0 is widest"
        );
        assert_eq!(
            root_split_dim(&storage, "wide_dim1.bkd"),
            1,
            "root should split on dim 1 when dim 1 is widest"
        );
    }

    #[test]
    fn build_subtree_skewed_data_round_trip() {
        // End-to-end correctness on a heavily skewed 3D dataset: dim 0
        // spans [0, n), dim 1 spans [0, 1), dim 2 spans [0, 0.001).
        // Widest-axis splitting must still produce a tree that returns
        // exactly the expected doc ids for an axis-aligned query.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 1_000;
        let mut points: Vec<f64> = Vec::with_capacity(n * 3);
        let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f64;
            points.push(v); // dim 0: wide
            points.push(v / (n as f64)); // dim 1: narrow [0, 1)
            points.push(v / (n as f64 * 1000.0)); // dim 2: very narrow
            doc_ids.push(i as u64);
        }
        {
            let output = storage.create_output("skewed.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 3).with_block_size(64);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "skewed.bkd").unwrap();
        let results = reader
            .range_search(
                &[Some(100.0), None, None],
                &[Some(200.0), None, None],
                true,
                true,
            )
            .unwrap();
        assert_eq!(results, (100u64..=200u64).collect::<Vec<_>>());
    }

    #[test]
    fn intersect_scratch_reuse_across_many_crosses_leaves() {
        // Build a tree with many small leaves and run a query that crosses
        // every leaf boundary, forcing the Crosses branch in intersect_leaf
        // to be taken once per leaf. The shared `IntersectScratch.points`
        // buffer must be reused without losing data across leaf reads.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 4_096;
        let block_size: usize = 32; // → ~128 leaves
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("scratch.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(block_size);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "scratch.bkd").unwrap();

        // Pick a query whose bounds (10.5 / (n - 10).5) sit *inside* leaf
        // blocks rather than on their boundaries, guaranteeing many leaves
        // hit the Crosses branch.
        let lower = 10.5;
        let upper = (n - 10) as f64 + 0.5;
        let results = reader
            .range_search(&[Some(lower)], &[Some(upper)], true, true)
            .unwrap();
        let expected: Vec<u64> = (11u64..=(n as u64 - 10)).collect();
        assert_eq!(results, expected);

        // Re-run the query — the second call uses a fresh scratch but
        // should also be deterministic. This guards against any cross-call
        // state leakage.
        let results2 = reader
            .range_search(&[Some(lower)], &[Some(upper)], true, true)
            .unwrap();
        assert_eq!(results2, expected);
    }

    #[test]
    fn intersect_inside_avoids_per_point_filter() {
        // Build a 1D tree with 4 leaf blocks; query the entire range so the
        // root subtree is `Inside` and every doc is reported via
        // `visit_inside`, never `visit`.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 256;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("inside.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(32);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "inside.bkd").unwrap();
        let query = AABB::new(vec![-1e9], vec![1e9]).unwrap();
        let mut v = TracingVisitor::new(query);
        reader.intersect(&mut v).unwrap();

        // Every hit came through visit_inside: the query bounds wholly
        // enclose every cell, so no point ever needed per-coordinate
        // filtering.
        assert_eq!(v.inside_hits.len(), n);
        assert!(v.crosses_hits.is_empty());
        v.inside_hits.sort_unstable();
        let expected: Vec<u64> = (0..n as u64).collect();
        assert_eq!(v.inside_hits, expected);
    }

    #[test]
    fn intersect_outside_prunes_subtree() {
        // Query that lies entirely above every point; expect zero hits and
        // at least one Outside compare result.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 128;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("outside.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(16);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "outside.bkd").unwrap();
        let query = AABB::new(vec![1000.0], vec![2000.0]).unwrap();
        let mut v = RecordingVisitor::new(query);
        reader.intersect(&mut v).unwrap();

        assert!(v.hits.is_empty());
        assert!(
            v.relations
                .borrow()
                .iter()
                .any(|r| matches!(r, CellRelation::Outside)),
            "expected at least one Outside compare, got {:?}",
            v.relations.borrow()
        );
    }

    #[test]
    fn intersect_crosses_filters_per_point() {
        // Query that overlaps a leaf boundary; expect Crosses leaves and
        // hits accumulated via visit (per-point filtering).
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 200;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("crosses.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(16);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "crosses.bkd").unwrap();
        let query = AABB::new(vec![50.5], vec![100.5]).unwrap();
        let mut v = TracingVisitor::new(query);
        reader.intersect(&mut v).unwrap();

        let expected: Vec<u64> = (51u64..=100u64).collect();
        let mut got = v.crosses_hits.clone();
        got.append(&mut v.inside_hits.clone());
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, expected);
        // At least some hits arrived via the `Crosses` path because the
        // query bounds (50.5 / 100.5) cut through leaf blocks.
        assert!(!v.crosses_hits.is_empty());
    }

    #[test]
    fn range_search_default_impl_matches_legacy_semantics() {
        // The trait's default `range_search` should still produce the same
        // sorted/deduped doc-id list it always has, now via `intersect`.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 500;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("legacy.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(64);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "legacy.bkd").unwrap();

        // Inclusive bounds.
        let inclusive = reader
            .range_search(&[Some(100.0)], &[Some(200.0)], true, true)
            .unwrap();
        assert_eq!(inclusive, (100u64..=200u64).collect::<Vec<_>>());

        // Exclusive bounds: 100 < x < 200.
        let exclusive = reader
            .range_search(&[Some(100.0)], &[Some(200.0)], false, false)
            .unwrap();
        assert_eq!(exclusive, (101u64..=199u64).collect::<Vec<_>>());

        // Unbounded upper.
        let lower_only = reader
            .range_search(&[Some(490.0)], &[None], true, true)
            .unwrap();
        assert_eq!(lower_only, (490u64..n as u64).collect::<Vec<_>>());
    }

    #[test]
    fn test_bkd_writer_reader_2d_single_leaf_aabb() {
        // Single-leaf tree: exercises the leaf-only write/read path that
        // skips the index section entirely. The new leaf AABB must still be
        // written and consumed.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let points: Vec<f64> = vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0];
        let doc_ids: Vec<u64> = vec![10, 20, 30];

        {
            let output = storage.create_output("single.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "single.bkd").unwrap();
        let results = reader
            .range_search(
                &[Some(2.0), Some(150.0)],
                &[Some(3.0), Some(250.0)],
                true,
                true,
            )
            .unwrap();
        assert_eq!(results, vec![20]);
    }

    // Note: the legacy `test_bkd_tree_creation`, `test_empty_tree`, and
    // `test_range_search_exact_bounds` were removed in #295 along with the
    // in-memory `SimpleBKDTree` they exercised. Equivalent coverage is
    // provided by `test_bkd_writer_empty`, `test_bkd_writer_reader_*`,
    // and `range_search_default_impl_matches_legacy_semantics` above.

    #[test]
    fn write_rejects_nan_coordinate() {
        // NaN has no defined ordering and would corrupt the BKD's split
        // decisions; the writer must reject it up-front with an index
        // error pointing at the offending dimension.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let points: Vec<f64> = vec![1.0, 2.0, f64::NAN, 4.0];
        let doc_ids: Vec<u64> = vec![10, 20];

        let output = storage.create_output("nan.bkd").unwrap();
        let mut writer = BKDWriter::new(output, 2);
        let err = writer.write(&points, &doc_ids).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("NaN"), "unexpected error: {msg}");
        // Offending position: doc 1 (second doc), dim 0.
        assert!(msg.contains("doc index 1"), "unexpected error: {msg}");
        assert!(msg.contains("dim 0"), "unexpected error: {msg}");
    }

    #[test]
    fn write_accepts_infinity_and_round_trips() {
        // ±Infinity sort consistently against every finite f64, so the
        // writer must accept them and the reader must surface them.
        //
        // This is also the canary for a single-leaf tree spanning
        // -INFINITY..+INFINITY: in sortable-u64 space that's the full
        // width, i.e. `bits_needed(...) == 64` for this dimension — the
        // exact boundary a naive `1u64 << width` bit-packer would panic on
        // (see `BitWriter`/`BitReader`'s `u128` accumulator).
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let points: Vec<f64> = vec![f64::NEG_INFINITY, -10.0, 0.0, 10.0, f64::INFINITY];
        let doc_ids: Vec<u64> = vec![100, 200, 300, 400, 500];
        {
            let output = storage.create_output("inf.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "inf.bkd").unwrap();

        // Unbounded query: every doc, including the infinities.
        let mut all = reader.range_search(&[None], &[None], true, true).unwrap();
        all.sort_unstable();
        assert_eq!(all, vec![100, 200, 300, 400, 500]);

        // Bounded query that excludes both infinities.
        let finite = reader
            .range_search(&[Some(-100.0)], &[Some(100.0)], true, true)
            .unwrap();
        assert_eq!(finite, vec![200, 300, 400]);

        // Lower bound at NEG_INFINITY (closed): includes the NEG_INFINITY
        // doc as well as every finite doc up to (and including) 0.0.
        let lower_inf = reader
            .range_search(&[Some(f64::NEG_INFINITY)], &[Some(0.0)], true, true)
            .unwrap();
        assert_eq!(lower_inf, vec![100, 200, 300]);

        // Upper bound at INFINITY (closed): includes the INFINITY doc.
        let upper_inf = reader
            .range_search(&[Some(0.0)], &[Some(f64::INFINITY)], true, true)
            .unwrap();
        assert_eq!(upper_inf, vec![300, 400, 500]);
    }

    // ---- Issue #549: leaf bit-packing primitives ----

    #[test]
    fn bit_writer_reader_round_trip_various_widths() {
        let widths: [u8; 8] = [0, 1, 7, 8, 9, 33, 63, 64];
        for &width in &widths {
            let values: Vec<u64> = if width == 64 {
                vec![0, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX]
            } else if width == 0 {
                vec![0, 0, 0]
            } else {
                let max_value: u64 = (1u64 << width) - 1;
                (0..64u64)
                    .map(|i| i.wrapping_mul(2_654_435_761) % (max_value + 1))
                    .collect()
            };
            let mut w = BitWriter::new();
            for &v in &values {
                w.write(v, width);
            }
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            for &expected in &values {
                assert_eq!(r.read(width), expected, "width={width} mismatch");
            }
        }
    }

    #[test]
    fn bit_writer_reader_width_zero_consumes_no_bits() {
        let mut w = BitWriter::new();
        w.write(0, 0);
        w.write(0, 0);
        w.write(0, 0);
        let bytes = w.finish();
        assert!(bytes.is_empty(), "width-0 writes must not emit any bytes");

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(0), 0);
        assert_eq!(r.read(0), 0);
    }

    #[test]
    fn bit_writer_reader_width_64_round_trips_full_range() {
        let mut w = BitWriter::new();
        w.write(0, 64);
        w.write(u64::MAX, 64);
        let bytes = w.finish();
        assert_eq!(bytes.len(), 16);
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(64), 0);
        assert_eq!(r.read(64), u64::MAX);
    }

    #[test]
    fn sortable_u64_round_trips_and_preserves_total_order() {
        let values: [f64; 10] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324, // smallest positive subnormal
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for &v in &values {
            let bits = f64_to_sortable_u64(v);
            let back = sortable_u64_to_f64(bits);
            assert_eq!(back.to_bits(), v.to_bits(), "bit-exact round trip for {v}");
        }
        // Order preservation: total_cmp order matches unsigned u64 order.
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let sortable: Vec<u64> = sorted.iter().map(|&v| f64_to_sortable_u64(v)).collect();
        let mut expected = sortable.clone();
        expected.sort_unstable();
        assert_eq!(
            sortable, expected,
            "sortable_u64 must preserve total_cmp order"
        );
    }

    #[test]
    fn bits_needed_matches_expected_widths() {
        assert_eq!(bits_needed(0), 0);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(255), 8);
        assert_eq!(bits_needed(256), 9);
        assert_eq!(bits_needed(u64::MAX), 64);
    }

    // ---- Issue #549: leaf format round trips ----

    /// Visitor that forces every leaf through the `Crosses` path (mirroring
    /// `MergeEngine::CollectPointsVisitor`) and records exact bit patterns,
    /// for round-trip tests that must distinguish `-0.0` from `+0.0`.
    #[derive(Default)]
    struct CollectAllVisitor {
        entries: Vec<(u64, Vec<f64>)>,
    }

    impl IntersectVisitor for CollectAllVisitor {
        fn compare(&self, _cell: &AABB) -> CellRelation {
            CellRelation::Crosses
        }
        fn visit_inside(&mut self, _doc_id: u64) {
            unreachable!("compare always returns Crosses");
        }
        fn visit(&mut self, doc_id: u64, point: &[f64]) {
            self.entries.push((doc_id, point.to_vec()));
        }
    }

    #[test]
    fn leaf_with_negative_and_positive_zero_round_trips_exactly() {
        // `+0.0` then `-0.0` then a positive value (and the mirrored
        // order): naive `<`/`>` comparison in `compute_aabb` would keep
        // `leaf_min = +0.0` even though `-0.0` (smaller in total order) is
        // present, corrupting the delta-from-min reconstruction. Regression
        // test for the design-review blocker (see `total_min`/`total_max`).
        for points in [vec![0.0_f64, -0.0, 5.0], vec![-5.0_f64, -0.0, 0.0]] {
            let doc_ids: Vec<u64> = (0..points.len() as u64).collect();
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            {
                let output = storage.create_output("zero.bkd").unwrap();
                let mut writer = BKDWriter::new(output, 1).with_block_size(16);
                writer.write(&points, &doc_ids).unwrap();
                writer.finish().unwrap();
            }
            let reader = BKDReader::open(storage.clone(), "zero.bkd").unwrap();
            let mut visitor = CollectAllVisitor::default();
            reader.intersect(&mut visitor).unwrap();
            visitor.entries.sort_by_key(|(id, _)| *id);
            assert_eq!(visitor.entries.len(), points.len());
            for (doc_id, point) in &visitor.entries {
                let expected = points[*doc_id as usize];
                assert_eq!(
                    point[0].to_bits(),
                    expected.to_bits(),
                    "doc {doc_id}: expected bit pattern {:x}, got {:x} (input {points:?})",
                    expected.to_bits(),
                    point[0].to_bits(),
                );
            }
        }
    }

    #[test]
    fn leaf_with_constant_dimension_round_trips_through_crosses() {
        // dim 1 is constant (5.0) across all points, so its derived bit
        // width is 0 and its packed section is 0 bytes. Forced through
        // `Crosses` (not `Inside`), the path that actually decodes point
        // coordinates, unlike `build_subtree_root_split_is_widest_axis`
        // which only exercises constant dimensions via the index node.
        let n = 20usize;
        let mut points = Vec::with_capacity(n * 2);
        for i in 0..n {
            points.push(i as f64);
            points.push(5.0);
        }
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("const_dim.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2).with_block_size(n);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let reader = BKDReader::open(storage.clone(), "const_dim.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        reader.intersect(&mut visitor).unwrap();
        assert_eq!(visitor.entries.len(), n);
        for (doc_id, point) in &visitor.entries {
            assert_eq!(point[0], *doc_id as f64);
            assert_eq!(point[1], 5.0, "constant dimension must reconstruct exactly");
        }
    }

    #[test]
    fn leaf_with_extreme_doc_id_range_round_trips() {
        let points = vec![0.0_f64, 1.0];
        let doc_ids = vec![0u64, u64::MAX];
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("extreme_doc_id.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(16);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let reader = BKDReader::open(storage.clone(), "extreme_doc_id.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        reader.intersect(&mut visitor).unwrap();
        let mut ids: Vec<u64> = visitor.entries.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, u64::MAX]);
    }

    /// Issue #1142: `GeoBoxPointsVisitor::into_candidates` (`lexical/
    /// query/geo.rs`) dedups multiple points sharing a doc_id by "first
    /// one seen in leaf traversal order wins" -- the new doc_id sort in
    /// `write_leaf_block` MUST be stable (`sort_by_key`, not
    /// `sort_unstable_by_key`) or this convention could silently change.
    /// Uses enough elements and interleaved duplicate keys that a small
    /// slice's tiny-array insertion-sort fallback can't accidentally
    /// mask an unstable sort (a handful of elements can stay in order by
    /// luck even under `sort_unstable_by_key`).
    #[test]
    fn duplicate_doc_ids_preserve_relative_point_order() {
        let n = 60usize;
        // doc_id cycles through {10, 20, 30}; the coordinate directly
        // encodes each point's original insertion index so the expected
        // post-sort order can be computed independently of the visitor.
        let doc_ids: Vec<u64> = (0..n).map(|i| 10 * ((i % 3) as u64 + 1)).collect();
        let points: Vec<f64> = (0..n as u64).map(|i| i as f64).collect();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("dup_stable.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(n);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let reader = BKDReader::open(storage.clone(), "dup_stable.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        reader.intersect(&mut visitor).unwrap();
        assert_eq!(visitor.entries.len(), n);

        // Expected: stable sort by doc_id, ties broken by original
        // insertion order (the index encoded in the coordinate).
        let mut expected: Vec<(u64, f64)> = (0..n)
            .map(|i| (10 * ((i % 3) as u64 + 1), i as f64))
            .collect();
        expected.sort_by_key(|&(doc_id, _)| doc_id);

        // A single leaf (block_size == n, no splitting), so the visitor's
        // arrival order reflects the on-disk point order directly.
        let actual: Vec<(u64, f64)> = visitor.entries.iter().map(|(id, p)| (*id, p[0])).collect();
        assert_eq!(
            actual, expected,
            "doc_id sort must be stable: points sharing a doc_id must keep \
             their original relative order"
        );
    }

    /// Issue #1142: pins the new format invariant directly -- a leaf's
    /// points are always decoded in doc_id-ascending order, regardless of
    /// insertion order.
    #[test]
    fn leaf_points_are_emitted_in_doc_id_ascending_order() {
        let n = 50usize;
        // Insert in reverse order so ascending output can only come from
        // the writer's sort, never from insertion order happening to
        // already be sorted.
        let doc_ids: Vec<u64> = (0..n as u64).rev().collect();
        let points: Vec<f64> = (0..n as u64).map(|i| i as f64).collect();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("ascending.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(n);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let reader = BKDReader::open(storage.clone(), "ascending.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        reader.intersect(&mut visitor).unwrap();
        let ids: Vec<u64> = visitor.entries.iter().map(|(id, _)| *id).collect();
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        assert_eq!(
            ids, sorted_ids,
            "leaf points must be emitted in doc_id-ascending order"
        );
    }

    /// Issue #1142: all doc_ids equal in a leaf must still round-trip
    /// exactly, and must pack strictly fewer bytes than an otherwise
    /// identical leaf needing at least 1 doc_id bit (confirms
    /// `doc_id_bits == 0` -- i.e. a genuinely zero-byte packed doc_id
    /// section -- rather than merely "small").
    #[test]
    fn leaf_with_all_identical_doc_ids_packs_zero_bits() {
        let n = 10usize;
        let points: Vec<f64> = (0..n as u64).map(|i| i as f64).collect();

        let all_same: Vec<u64> = vec![42u64; n];
        let storage_a = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage_a.create_output("same.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(n);
            writer.write(&points, &all_same).unwrap();
            writer.finish().unwrap();
        }
        let reader = BKDReader::open(storage_a.clone(), "same.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        reader.intersect(&mut visitor).unwrap();
        assert_eq!(visitor.entries.len(), n);
        for (doc_id, _) in &visitor.entries {
            assert_eq!(*doc_id, 42);
        }

        let mut one_different = all_same;
        one_different[n - 1] = 43;
        let storage_b = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage_b.create_output("diff.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(n);
            writer.write(&points, &one_different).unwrap();
            writer.finish().unwrap();
        }
        let same_size = storage_a.metadata("same.bkd").unwrap().size;
        let diff_size = storage_b.metadata("diff.bkd").unwrap().size;
        assert!(
            same_size < diff_size,
            "all-identical doc_ids should pack strictly fewer bytes than a leaf \
             needing at least 1 doc_id bit: same={same_size}, diff={diff_size}"
        );
    }

    /// Issue #1142: small consecutive gaps among nine values, then one
    /// huge jump -- the one shape the empirical experiment found doesn't
    /// improve over the old fixed-width-from-range scheme (the outlier
    /// gap nearly spans the leaf's full range, so `bits_needed(max_delta)`
    /// and `bits_needed(max - min)` end up close). Must still round-trip
    /// exactly.
    #[test]
    fn leaf_with_one_large_gap_among_small_gaps() {
        let doc_ids: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 1_000_000];
        let n = doc_ids.len();
        let points: Vec<f64> = (0..n as u64).map(|i| i as f64).collect();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("gap.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(n);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let reader = BKDReader::open(storage.clone(), "gap.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        reader.intersect(&mut visitor).unwrap();
        let mut ids: Vec<u64> = visitor.entries.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        let mut expected = doc_ids.clone();
        expected.sort_unstable();
        assert_eq!(ids, expected);
    }

    #[test]
    fn bkd_leaf_bytes_shrink_relative_to_raw_format() {
        // A correlated field (monotonically increasing, like a timestamp)
        // should compress well under bit-packing. Compares the on-disk
        // file size against the pre-#549 raw-format byte count for the
        // same point+doc_id data (ignoring header/index overhead,
        // identical either way and negligible at this scale).
        //
        // Issue #1142 tightens this bound: consecutive-delta doc_id
        // packing measures ~42.5% (34026 bytes) for this exact corpus,
        // versus the pre-#1142 (v3, independent-delta) ~53% this test
        // used to bound against (60% threshold). 45% leaves headroom for
        // build non-determinism (e.g. a future k-means-style leaf
        // assignment change) without being as loose as the old bound.
        let n: u64 = 5000;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n).collect();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("shrink.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let on_disk = storage.metadata("shrink.bkd").unwrap().size;
        let raw_points_and_doc_ids = n * (8 + 8); // pre-#549: f64 point + u64 doc_id per point
        assert!(
            on_disk < raw_points_and_doc_ids * 45 / 100,
            "expected the packed leaf format to use well under 45% of the raw \
             point+doc_id bytes for a correlated field: on_disk={on_disk}, \
             raw={raw_points_and_doc_ids}"
        );
    }

    /// Issue #1142: a fixed corpus's on-disk size must never exceed a
    /// golden upper bound, pinning the consecutive-delta doc_id packing's
    /// measured output size so a future regression (e.g. accidentally
    /// reverting to independent-delta packing) is caught immediately
    /// rather than only via a loose ratio check.
    #[test]
    fn bkd_file_size_has_golden_upper_bound() {
        let n: u64 = 5000;
        // Uniform-ish 1D data (not perfectly correlated with doc_id, unlike
        // `bkd_leaf_bytes_shrink_relative_to_raw_format`'s corpus) so this
        // pins a second, independent point on the compression curve.
        let points: Vec<f64> = (0..n).map(|i| ((i * 2654435761) % 100000) as f64).collect();
        let doc_ids: Vec<u64> = (0..n).collect();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("golden.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let on_disk = storage.metadata("golden.bkd").unwrap().size;
        // Measured at 36990 bytes when this test was written (Issue #1142).
        // A future change that legitimately shrinks this further should
        // lower GOLDEN_V4; one that grows it should be treated as a
        // regression unless deliberately justified.
        const GOLDEN_V4: u64 = 37200;
        assert!(
            on_disk <= GOLDEN_V4,
            "on-disk size {on_disk} exceeds the golden upper bound {GOLDEN_V4} -- \
             if this is an intentional format change, update GOLDEN_V4"
        );
    }

    // ---- Issue #549: corruption / bounds-check rejection ----

    /// Hand-craft a single-leaf v3 BKD file with an explicit (possibly
    /// invalid) leaf header, for exercising corruption-rejection paths a
    /// well-formed `BKDWriter` would never itself produce. `index_start_offset`
    /// is fixed at `u64::MAX` (no internal index nodes are ever written), so
    /// `root_node_offset` (`header_size`) always compares as a leaf.
    #[allow(clippy::too_many_arguments)]
    fn write_hand_crafted_single_leaf_file(
        storage: &Arc<MemoryStorage>,
        path: &str,
        num_dims: u32,
        total_point_count: u64,
        block_size: u32,
        leaf_min: &[f64],
        leaf_max: &[f64],
        leaf_count: u32,
        doc_id_base: u64,
        doc_id_bits: u8,
        packed_point_bytes: &[u8],
        packed_doc_id_bytes: &[u8],
    ) {
        use crate::storage::structured::StructWriter;
        let output = storage.create_output(path).unwrap();
        let mut writer = StructWriter::new(output);

        let header_size = 4 + 4 + 4 + 4 + 8 + 8 + 4 + (num_dims as u64 * 8 * 2) + 8 + 8;
        writer.write_u32(BKD_MAGIC).unwrap();
        writer.write_u32(BKD_VERSION).unwrap();
        writer.write_u32(num_dims).unwrap();
        writer.write_u32(8).unwrap();
        writer.write_u64(total_point_count).unwrap();
        writer.write_u64(1).unwrap(); // num_blocks
        writer.write_u32(block_size).unwrap();
        for &v in leaf_min {
            writer.write_f64(v).unwrap();
        }
        for &v in leaf_max {
            writer.write_f64(v).unwrap();
        }
        writer.write_u64(u64::MAX).unwrap(); // index_start_offset
        writer.write_u64(header_size).unwrap(); // root_node_offset

        writer.write_u32(leaf_count).unwrap();
        for &v in leaf_min {
            writer.write_f64(v).unwrap();
        }
        for &v in leaf_max {
            writer.write_f64(v).unwrap();
        }
        writer.write_u64(doc_id_base).unwrap();
        writer.write_u8(doc_id_bits).unwrap();
        writer.write_raw(packed_point_bytes).unwrap();
        writer.write_raw(packed_doc_id_bytes).unwrap();

        writer.close().unwrap();
    }

    #[test]
    fn rejects_leaf_with_doc_id_bits_exceeding_64() {
        // `doc_id_bits = 200` claims a 25-byte packed section for a single
        // value (`ceil(200/8)`); the file supplies exactly that many bytes
        // so `checked_len` alone would let it through — isolating this test
        // to the dedicated `doc_id_bits <= 64` check, not the general
        // packed-length bound (see `rejects_leaf_whose_packed_length_overruns_the_file`
        // for that one). Without the dedicated check, `BitReader::read(200)`
        // would panic on `1u128 << 200` (shift amount >= 128).
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        write_hand_crafted_single_leaf_file(
            &storage,
            "bad_doc_id_bits.bkd",
            1,
            1,
            512,
            &[0.0],
            &[0.0],
            1,
            0,
            200,
            &[],
            &[0u8; 25],
        );
        let reader = BKDReader::open(storage.clone(), "bad_doc_id_bits.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        let err = reader.intersect(&mut visitor).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("doc_id_bits"), "unexpected error: {msg}");
    }

    #[test]
    fn rejects_leaf_with_count_exceeding_total_point_count() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        write_hand_crafted_single_leaf_file(
            &storage,
            "bad_count.bkd",
            1,
            1,
            512,
            &[0.0],
            &[0.0],
            1000,
            0,
            0,
            &[],
            &[],
        );
        let reader = BKDReader::open(storage.clone(), "bad_count.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        let err = reader.intersect(&mut visitor).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("total"), "unexpected error: {msg}");
    }

    #[test]
    fn rejects_leaf_whose_packed_length_overruns_the_file() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // leaf_min=0.0, leaf_max=1.0 gives a wide bit width; count=1_000_000
        // claims a multi-megabyte packed section, but the file only
        // actually contains a handful of bytes after the header.
        write_hand_crafted_single_leaf_file(
            &storage,
            "overrun.bkd",
            1,
            1_000_000,
            1_000_000,
            &[0.0],
            &[1.0],
            1_000_000,
            0,
            0,
            &[0u8; 4],
            &[],
        );
        let reader = BKDReader::open(storage.clone(), "overrun.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        let err = reader.intersect(&mut visitor).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("corrupted"), "unexpected error: {msg}");
    }

    /// Issue #1142: shrinking `doc_id_bits` (this revision's whole point)
    /// weakens `rejects_leaf_whose_packed_length_overruns_the_file`'s
    /// packed-length bound proportionally -- a leaf with zero point bits
    /// and zero doc_id bits leaves that bound completely unable to reject
    /// an oversized `count` at all. `block_size` closes this independent
    /// of the packed-length arithmetic.
    #[test]
    fn rejects_leaf_with_count_exceeding_block_size() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // leaf_min == leaf_max (0 point bits) and doc_id_bits == 0 together
        // make `total_packed_len == 0`, so only the `block_size` bound (not
        // the packed-length one) can reject this.
        write_hand_crafted_single_leaf_file(
            &storage,
            "count_exceeds_block_size.bkd",
            1,
            1_000_000,
            512,
            &[0.0],
            &[0.0],
            1_000_000,
            0,
            0,
            &[],
            &[],
        );
        let reader = BKDReader::open(storage.clone(), "count_exceeds_block_size.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        let err = reader.intersect(&mut visitor).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("block_size"), "unexpected error: {msg}");
    }

    /// Issue #1142: the first packed doc_id delta must always be 0 by
    /// construction (the leaf's smallest doc_id, sorted to position 0,
    /// has nothing before it to differ from). A hand-crafted leaf with a
    /// nonzero leading delta can only be corrupt or written by a
    /// non-conforming encoder, and must be rejected rather than silently
    /// decoded into a wrong doc_id (which would also offset every
    /// subsequent doc_id in the leaf via the running accumulator).
    #[test]
    fn rejects_leaf_with_nonzero_leading_doc_id_delta() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        write_hand_crafted_single_leaf_file(
            &storage,
            "bad_leading_delta.bkd",
            1,
            2,
            512,
            &[0.0],
            &[0.0], // constant dimension -> 0 point bits, no packed point bytes needed
            2,
            100, // doc_id_base
            8,   // doc_id_bits (1 byte/value)
            &[],
            &[5u8, 3u8], // first delta = 5 (must be 0), second = 3
        );
        let reader = BKDReader::open(storage.clone(), "bad_leading_delta.bkd").unwrap();
        let mut visitor = CollectAllVisitor::default();
        let err = reader.intersect(&mut visitor).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("first packed doc_id delta"),
            "unexpected error: {msg}"
        );
    }

    /// Issue #1159: `checked_point_count_u32` must accept every length that
    /// actually fits in a `u32`, including the boundary value itself.
    #[test]
    fn checked_point_count_u32_accepts_representable_lengths() {
        assert_eq!(checked_point_count_u32(0).unwrap(), 0);
        assert_eq!(checked_point_count_u32(5).unwrap(), 5);
        assert_eq!(
            checked_point_count_u32(u32::MAX as usize).unwrap(),
            u32::MAX
        );
    }

    /// Issue #1159: a length one past `u32::MAX` must be rejected with an
    /// error, not silently truncated by an `as u32` cast.
    #[test]
    fn checked_point_count_u32_rejects_lengths_beyond_u32_max() {
        let err = checked_point_count_u32(u32::MAX as usize + 1).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("exceeds u32::MAX"), "unexpected error: {msg}");
    }
}
