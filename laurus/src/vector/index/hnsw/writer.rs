//! HNSW (Hierarchical Navigable Small World) index builder for approximate search.

use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::util::alloc_bounds::checked_capacity;
use crate::vector::core::rerank::RerankStorageKind;
use crate::vector::core::vector::Vector;
use crate::vector::index::HnswIndexConfig;
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::format::{
    QuantHeader, VERSION_FIELD_DICT, VERSION_ORDINAL_GRAPH, VectorSegmentHeader, build_field_dict,
    record_prefix_size,
};
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::quantized_io::{
    quantize_segment, quantized_record_payload_size, read_dequantized_vector,
    write_quantized_record,
};
use crate::vector::index::rerank_sidecar::{read_sidecar, write_sidecar};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use bit_vec::BitVec;
use parking_lot::RwLock;
use rand::{RngExt, SeedableRng};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Fixed seed for the HNSW level-selection RNG (Issue #841).
///
/// Level selection needs no secret randomness — what matters is the
/// geometric level distribution, not its unpredictability — so the
/// build uses a deterministic generator seeded with this constant.
/// This makes graph topology reproducible for a given insertion order:
/// segment builds, merges, and topology-sensitive tests all become
/// deterministic instead of flaking on unlucky layouts. Precedent:
/// Lucene's `HnswGraphBuilder` builds with `DEFAULT_RAND_SEED = 42`.
const LEVEL_RNG_SEED: u64 = 42;

/// Derive an independent, deterministic RNG seed for `doc_id`'s level
/// assignment (Issue #637).
///
/// Advancing one shared sequential RNG across every node (the pre-#637
/// approach) makes node N's draw depend on how many random numbers every
/// prior node happened to consume, which serializes level assignment onto
/// the main thread before the parallel insertion phase can even start.
/// Seeding a fresh RNG per doc_id from a well-mixed hash of
/// `(LEVEL_RNG_SEED, doc_id)` instead makes each node's level computable
/// independently — and therefore in parallel via `rayon::par_iter` — while
/// staying deterministic for that doc_id regardless of thread scheduling
/// (preserving Issue #841's invariant under a different exact RNG stream;
/// the specific levels a given corpus produces change, but build-to-build
/// reproducibility for the same corpus does not).
///
/// The mixing step is the SplitMix64 finalizer: a plain `SEED ^ doc_id`
/// would leave sequential doc_ids (0, 1, 2, ...) producing seeds that
/// differ only in a few low bits, which does not scramble a PRNG's early
/// output well.
#[inline]
fn level_rng_seed_for(doc_id: u64) -> u64 {
    let mut z = LEVEL_RNG_SEED ^ doc_id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Minimum vector count for training a PQ codebook (Issue #880).
///
/// PQ k-means fits 256 centroids per sub-quantizer; training on fewer
/// vectors than centroids yields a degenerate codebook. Segments below this
/// threshold are written as Scalar8Bit instead — the LVS1 header is
/// self-describing, so readers dispatch on the stored kind and a later
/// (larger) merged segment picks PQ back up automatically.
const PQ_MIN_TRAIN_VECTORS: usize = 256;

/// Minimum vector count for training a PQ FastScan codebook (Issue #880).
///
/// FastScan trains K=16 centroids per sub-quantizer (4-bit codes), so its
/// degenerate-training floor is 16 — far below the PQ-256 threshold.
#[cfg(feature = "pq-fastscan")]
const PQ_FASTSCAN_MIN_TRAIN_VECTORS: usize = 16;

/// Abstract trait to allow reading from both HnswGraph (serial) and ConcurrentHnswGraph (parallel)
trait GraphView {
    /// Copy `doc_id`'s neighbor list at `level` into `out` (`out` is always
    /// cleared first, regardless of whether the node/level is found — an
    /// absent list simply leaves `out` empty, which every caller already
    /// treats identically to "nothing to iterate").
    ///
    /// Takes a caller-owned buffer instead of returning `Option<Vec<u64>>`
    /// (Issue #1137): the build-time traversal loops that call this
    /// (`search_layer`'s candidate loop, `insert_one`'s Phase A descent) do
    /// so far more often per node insertion than the containers Issue #632
    /// addressed, so a fresh `Vec` per call was actually the dominant
    /// allocation source on this hot path.
    fn copy_neighbors_into(&self, doc_id: u64, level: usize, out: &mut Vec<u64>);

    /// Whether a node may be traversed/selected during a build-time search.
    ///
    /// Defaults to `true` (every node visible), so query-time searches over a
    /// finished [`HnswGraph`] are unaffected. [`ConcurrentHnswGraph`] overrides
    /// it to hide nodes that are not yet fully linked (Issue #868 / #621): a
    /// concurrent inserter must never select a node whose forward edges are
    /// still being written, or it would connect to a half-built dead end.
    #[inline]
    fn is_visible(&self, _doc_id: u64) -> bool {
        true
    }
}

impl GraphView for HnswGraph {
    fn copy_neighbors_into(&self, doc_id: u64, level: usize, out: &mut Vec<u64>) {
        out.clear();
        if let Some(neighbors) = self.get_neighbors(doc_id, level) {
            out.extend_from_slice(neighbors);
        }
    }
}

/// A node's per-level neighbor lists plus a "fully linked" flag used to gate
/// build-time visibility (Issue #868 / #621).
struct NodeEntry {
    /// `false` until this node's forward edges have been written at **all** of
    /// its levels; set once, with `Release`, as the last step of inserting the
    /// node. Readers load it with `Acquire`, so observing `true` guarantees
    /// every `set_neighbors` write for this node is visible.
    linked: std::sync::atomic::AtomicBool,
    /// One RwLock-protected neighbor list per level `0..=level`.
    layers: Vec<RwLock<Vec<u64>>>,
}

/// A thread-safe view of the HNSW graph during construction
struct ConcurrentHnswGraph {
    max_level: usize,
    // Map from doc_id to its NodeEntry (per-level neighbor lists + linked flag)
    nodes: HashMap<u64, NodeEntry>,
}

impl ConcurrentHnswGraph {
    /// Build an entry with empty neighbor lists for levels `0..=level` and the
    /// given initial `linked` state (`true` for pre-existing/seed nodes that
    /// are already searchable, `false` for new nodes still to be inserted).
    fn new_entry(level: usize, linked: bool) -> NodeEntry {
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(RwLock::new(Vec::new()));
        }
        NodeEntry {
            linked: std::sync::atomic::AtomicBool::new(linked),
            layers,
        }
    }

    fn new(nodes_with_levels: Vec<(u64, usize)>, max_level: usize) -> Self {
        let mut nodes = HashMap::new();
        for (doc_id, level) in nodes_with_levels {
            // New nodes start unlinked (invisible to build-time search until
            // their forward edges are set).
            nodes.insert(doc_id, Self::new_entry(level, false));
        }

        Self { max_level, nodes }
    }

    /// Whether `doc_id` is fully linked and therefore visible to build-time
    /// search (Issue #868 / #621). Acquire-loads the flag so all of the node's
    /// `set_neighbors` writes are visible once this returns `true`.
    #[inline]
    fn is_linked(&self, doc_id: u64) -> bool {
        self.nodes
            .get(&doc_id)
            .map(|e| e.linked.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(false)
    }

    /// Publish `doc_id` as fully linked. Must be called **after** every
    /// `set_neighbors` for the node; the `Release` store pairs with the
    /// `Acquire` in [`Self::is_linked`].
    #[inline]
    fn mark_linked(&self, doc_id: u64) {
        if let Some(e) = self.nodes.get(&doc_id) {
            e.linked.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    fn set_neighbors(&self, doc_id: u64, level: usize, new_neighbors: Vec<u64>) {
        if let Some(entry) = self.nodes.get(&doc_id)
            && let Some(lock) = entry.layers.get(level)
        {
            *lock.write() = new_neighbors;
        }
    }

    fn add_neighbor_with_pruning(
        &self,
        doc_id: u64,
        level: usize,
        neighbor_id: u64,
        max_conn: usize,
        writer: &HnswIndexWriter,
    ) -> Result<()> {
        if let Some(entry) = self.nodes.get(&doc_id)
            && let Some(lock) = entry.layers.get(level)
        {
            // Push + prune under a SINGLE held write lock (Issue #868). The
            // previous version dropped the lock between snapshotting the list
            // and writing the pruned result back, so a concurrent thread's
            // back-edge push in that window was clobbered by the stale
            // overwrite — losing the only in-edge of some node made it
            // unreachable from the entry point (silent recall loss). Pruning
            // needs only the immutable `self.vectors` / `self.doc_id_map`
            // (via `prune_neighbors`) and touches no node lock, so holding
            // the write lock across it is deadlock-free; the extra work is an
            // O(max_conn) distance pass that runs only when a node exceeds
            // its degree bound.
            let mut neighbors = lock.write();
            if !neighbors.contains(&neighbor_id) {
                neighbors.push(neighbor_id);
            }
            if neighbors.len() > max_conn {
                let pruned = writer.prune_neighbors(doc_id, neighbors.clone(), max_conn)?;
                *neighbors = pruned;
            }
        }
        Ok(())
    }

    fn from_hnsw_graph(graph: HnswGraph, extended_max_level: usize) -> Self {
        let mut nodes = HashMap::with_capacity(graph.node_count());
        for (doc_id, layered_neighbors) in graph.into_iter_nodes() {
            let layers = layered_neighbors.into_iter().map(RwLock::new).collect();
            // Nodes loaded from a finished graph are already fully linked and
            // searchable.
            nodes.insert(
                doc_id,
                NodeEntry {
                    linked: std::sync::atomic::AtomicBool::new(true),
                    layers,
                },
            );
        }

        Self {
            max_level: extended_max_level,
            nodes,
        }
    }

    fn add_nodes(&mut self, nodes_with_levels: Vec<(u64, usize)>) {
        for (doc_id, level) in nodes_with_levels {
            if self.nodes.contains_key(&doc_id) {
                continue;
            }
            // New nodes start unlinked (invisible until inserted).
            self.nodes.insert(doc_id, Self::new_entry(level, false));
        }
    }
}

impl GraphView for ConcurrentHnswGraph {
    fn copy_neighbors_into(&self, doc_id: u64, level: usize, out: &mut Vec<u64>) {
        out.clear();
        if let Some(lock) = self
            .nodes
            .get(&doc_id)
            .and_then(|entry| entry.layers.get(level))
        {
            out.extend_from_slice(&lock.read());
        }
    }

    #[inline]
    fn is_visible(&self, doc_id: u64) -> bool {
        self.is_linked(doc_id)
    }
}

/// Builder for HNSW vector indexes (approximate search).
#[derive(Debug)]
pub struct HnswIndexWriter {
    index_config: HnswIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    path: String,
    _ml: f64, // Level normalization factor
    vectors: Vec<(u64, String, Vector)>,
    // Map from doc_id to index in vectors for fast access
    doc_id_map: HashMap<u64, usize>,
    #[allow(dead_code)] // Maintained during build but not yet read; reserved for future use
    levels: Vec<Vec<u64>>,
    entry_point: Option<u64>,
    graph: Option<HnswGraph>,
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
    next_vec_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    id: u64,
    distance: f32,
    similarity: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (nearest first) or max-heap depending on usage.
        // For keeping top-K nearest, we usually want max-heap to pop largest distance.
        // But let's define standard ordering: smaller distance = smaller.
        // Wait, for BinaryHeap in Rust, it's a max-heap.
        // If we want smallest distance at top, we need reverse.
        // If we want largest distance at top (to remove worst candidate), we use standard.
        self.distance.total_cmp(&other.distance)
    }
}

/// A node in `search_layer`'s to-visit frontier (min-heap by distance).
///
/// Hoisted out of `search_layer` (Issue #632) so it can be a field type of
/// [`SearchLayerArena`] below.
#[derive(Debug, Clone, PartialEq)]
struct VisitorCandidate {
    id: u64,
    distance: f32,
}
impl Eq for VisitorCandidate {}
impl Ord for VisitorCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance > larger distance
        other.distance.total_cmp(&self.distance)
    }
}
impl PartialOrd for VisitorCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `search_layer`-only scratch state, reused across calls on the same
/// thread instead of allocating a fresh `HashSet` + two `BinaryHeap`s every
/// call (Issue #632: at M=16/ef_construction=200/10M nodes, `search_layer`
/// runs `O(top_level)` times per inserted node, so those three containers'
/// growth-from-empty allocations add up to hundreds of millions of
/// allocator calls under a parallel build).
///
/// Design notes:
/// - `visited` is a `BitVec` indexed by the dense index `doc_id_map` maps
///   each `doc_id` to (not the `doc_id` itself), mirroring the convention
///   the read-side `HnswSearcher` (`searcher.rs`) already uses for its own
///   visited bitmap.
/// - Deliberately NOT the generation-counter `Vec<u32>` this issue
///   originally proposed: that shape is retained per OS thread for the
///   life of the process (rayon's global pool threads never die), so at
///   10M nodes it would hold ~40 MB/thread (hundreds of MB across a
///   thread pool) forever, versus the `HashSet` it replaces which only
///   ever held the handful of actually-visited ids and was freed after
///   each call. A `BitVec` (1 bit/node) plus `touched` — the list of
///   indices set *this call*, used to undo exactly those bits — costs
///   ~1.25 MB/thread at 10M nodes and resets in time proportional to what
///   was actually visited, not the whole node count. It also has no
///   wraparound case to handle, unlike a generation counter.
/// - `reset` runs on entry, not on exit: if a previous call returned early
///   via `?`, the next call still starts from a clean, fully-undone state
///   without needing a cleanup-on-every-exit-path discipline.
/// - Invariant this design depends on: `search_layer` never reenters
///   itself on the same thread (no nested `rayon::join`/`par_iter` inside
///   it). Verified for the current body: `calc_dist`/`calc_dist_by_idx` is
///   pure SIMD arithmetic and `GraphView::copy_neighbors_into` only takes a
///   `parking_lot` read lock (never yields to a scheduler), so neither can
///   trigger a reentrant call. If `search_layer` ever gains internal
///   parallelism, revisit this — `RefCell::borrow_mut()` below would
///   panic on reentry.
struct SearchLayerArena {
    visited: BitVec,
    touched: Vec<usize>,
    to_visit: BinaryHeap<VisitorCandidate>,
    found: BinaryHeap<Candidate>,
    /// Scratch buffer for `GraphView::copy_neighbors_into` (Issue #1137),
    /// reused across calls instead of letting each call allocate its own
    /// `Vec<u64>`. Cleared and refilled by `copy_neighbors_into` itself on
    /// every use, so `reset` doesn't need to touch it.
    neighbor_buf: Vec<u64>,
}

impl SearchLayerArena {
    fn new() -> Self {
        Self {
            visited: BitVec::new(),
            touched: Vec::new(),
            to_visit: BinaryHeap::new(),
            found: BinaryHeap::new(),
            neighbor_buf: Vec::new(),
        }
    }

    /// Prepare the arena for a new `search_layer` call: grow `visited` if
    /// the vector set has grown since last use, undo exactly the bits this
    /// thread's *previous* call set (via `touched`), and clear the heaps.
    fn reset(&mut self, node_capacity: usize) {
        if self.visited.len() < node_capacity {
            let grow_by = node_capacity - self.visited.len();
            self.visited.grow(grow_by, false);
        }
        for idx in self.touched.drain(..) {
            self.visited.set(idx, false);
        }
        self.to_visit.clear();
        self.found.clear();
    }

    /// Same semantics as `HashSet::insert`: `true` if `idx` was not
    /// already visited (and it is now), `false` if it already was.
    fn mark_visited(&mut self, idx: usize) -> bool {
        if self.visited.get(idx).unwrap_or(false) {
            false
        } else {
            self.visited.set(idx, true);
            self.touched.push(idx);
            true
        }
    }
}

thread_local! {
    static SEARCH_LAYER_ARENA: RefCell<SearchLayerArena> = RefCell::new(SearchLayerArena::new());
}

/// Result of parsing a serialized graph block back into the writer's
/// doc_id-keyed shape: `(entry_point, max_level, doc_id -> layered
/// neighbour doc ids)`.
type ParsedGraphBlock = (Option<u64>, usize, HashMap<u64, Vec<Vec<u64>>>);

impl HnswIndexWriter {
    /// Create a new HNSW index builder.
    pub fn new(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
    ) -> Result<Self> {
        if index_config.m < 2 {
            return Err(crate::error::LaurusError::InvalidOperation(
                "HNSW parameter m must be >= 2".to_string(),
            ));
        }
        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            path: path.into(),
            _ml,
            levels: vec![Vec::new(); max_level + 1],
            entry_point: None,
            vectors: Vec::new(),
            doc_id_map: HashMap::new(),
            graph: None,
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Create a new HNSW index builder with storage.
    ///
    /// If an existing index file is found on disk, its vectors are loaded
    /// into the writer so that the next commit preserves them. This
    /// prevents data loss across multiple commit cycles.
    pub fn with_storage(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let path = path.into();
        let file_name = format!("{}.hnsw", path);
        if storage.file_exists(&file_name) {
            return Self::load(index_config, writer_config, storage, &path);
        }

        if index_config.m < 2 {
            return Err(crate::error::LaurusError::InvalidOperation(
                "HNSW parameter m must be >= 2".to_string(),
            ));
        }
        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path,
            _ml,
            levels: vec![Vec::new(); max_level + 1],
            entry_point: None,
            vectors: Vec::new(),
            doc_id_map: HashMap::new(),
            graph: None,
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Convert this writer into a doc-centric field writer adapter.
    pub fn into_field_writer(self, field_name: impl Into<String>) -> LegacyVectorFieldWriter<Self> {
        LegacyVectorFieldWriter::new(field_name, self)
    }

    /// Parse a v1 (doc_id-encoded) graph block into the writer's
    /// doc_id-keyed representation. The caller has already consumed the
    /// leading `has_graph = 1` byte.
    ///
    /// # Arguments
    ///
    /// * `input` - Stream positioned after the `has_graph` byte.
    /// * `file_size` - Total file size, for allocation bounding (#806).
    ///
    /// # Returns
    ///
    /// `(entry_point, max_level, doc_id → layered neighbour doc ids)`.
    fn load_graph_block_v1(
        input: &mut dyn crate::storage::StorageInput,
        file_size: u64,
    ) -> Result<ParsedGraphBlock> {
        // Read entry point
        let mut entry_point_buf = [0u8; 8];
        input.read_exact(&mut entry_point_buf)?;
        let entry_point_raw = u64::from_le_bytes(entry_point_buf);
        let entry_point = if entry_point_raw == u64::MAX {
            None
        } else {
            Some(entry_point_raw)
        };

        // Read max level
        let mut max_level_buf = [0u8; 4];
        input.read_exact(&mut max_level_buf)?;
        let max_level = u32::from_le_bytes(max_level_buf) as usize;

        // Read nodes (u64 to match the v1 write format)
        let mut node_count_buf = [0u8; 8];
        input.read_exact(&mut node_count_buf)?;
        let node_count = u64::from_le_bytes(node_count_buf) as usize;

        // Bound every graph allocation by the bytes left in the file
        // (Issue #806). Reused for the inner layer / neighbor counts so
        // no extra syscall is taken inside the per-node / per-layer
        // loops.
        let graph_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        // Each node serializes at least doc_id (8) + layer_count (4).
        checked_capacity(node_count, 12, graph_remaining, "hnsw node_count")?;
        let mut nodes = HashMap::with_capacity(node_count);

        for _ in 0..node_count {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);

            let mut layer_count_buf = [0u8; 4];
            input.read_exact(&mut layer_count_buf)?;
            let layer_count = u32::from_le_bytes(layer_count_buf) as usize;

            // Each layer serializes at least its neighbor_count (4).
            checked_capacity(layer_count, 4, graph_remaining, "hnsw layer_count")?;
            let mut layers = Vec::with_capacity(layer_count);

            for _ in 0..layer_count {
                let mut neighbor_count_buf = [0u8; 4];
                input.read_exact(&mut neighbor_count_buf)?;
                let neighbor_count = u32::from_le_bytes(neighbor_count_buf) as usize;

                // Each v1 neighbor serializes as a u64 (8 bytes).
                checked_capacity(neighbor_count, 8, graph_remaining, "hnsw neighbor_count")?;
                let mut neighbors = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    let mut neighbor_buf = [0u8; 8];
                    input.read_exact(&mut neighbor_buf)?;
                    neighbors.push(u64::from_le_bytes(neighbor_buf));
                }
                layers.push(neighbors);
            }
            nodes.insert(doc_id, layers);
        }

        Ok((entry_point, max_level, nodes))
    }

    /// Parse a v2 (ordinal-encoded, Issue #686) graph block and translate
    /// it back to the writer's doc_id-keyed representation. The caller
    /// has already consumed the leading `has_graph = 1` byte.
    ///
    /// # Arguments
    ///
    /// * `input` - Stream positioned after the `has_graph` byte.
    /// * `file_size` - Total file size, for allocation bounding (#806).
    /// * `vectors` - The record `(doc_id, field, vector)` triples in
    ///   on-disk (doc_id-ascending) order; the ordinal table is their
    ///   deduplicated id sequence.
    ///
    /// # Returns
    ///
    /// `(entry_point, max_level, doc_id → layered neighbour doc ids)`,
    /// or an error on any count/ordinal inconsistency (corrupt segment).
    fn load_graph_block_v2(
        input: &mut dyn crate::storage::StorageInput,
        file_size: u64,
        vectors: &[(u64, String, Vector)],
    ) -> Result<ParsedGraphBlock> {
        let mut unique_ids: Vec<u64> = Vec::with_capacity(vectors.len());
        for (doc_id, _, _) in vectors {
            match unique_ids.last() {
                Some(&last) if *doc_id == last => {}
                Some(&last) if *doc_id < last => {
                    return Err(LaurusError::index(format!(
                        "HNSW segment corrupt: record doc ids not sorted \
                         ({doc_id} follows {last})"
                    )));
                }
                _ => unique_ids.push(*doc_id),
            }
        }
        let doc_of = |ord: u32| -> Result<u64> {
            unique_ids.get(ord as usize).copied().ok_or_else(|| {
                LaurusError::index(format!(
                    "HNSW v2 graph corrupt: ordinal {ord} out of range \
                     (node count {})",
                    unique_ids.len()
                ))
            })
        };

        let mut entry_point_buf = [0u8; 4];
        input.read_exact(&mut entry_point_buf)?;
        let entry_point_raw = u32::from_le_bytes(entry_point_buf);
        let entry_point = if entry_point_raw == u32::MAX {
            None
        } else {
            Some(doc_of(entry_point_raw)?)
        };

        let mut max_level_buf = [0u8; 4];
        input.read_exact(&mut max_level_buf)?;
        let max_level = u32::from_le_bytes(max_level_buf) as usize;

        let mut node_count_buf = [0u8; 4];
        input.read_exact(&mut node_count_buf)?;
        let node_count = u32::from_le_bytes(node_count_buf) as usize;
        if node_count != unique_ids.len() {
            return Err(LaurusError::index(format!(
                "HNSW v2 graph corrupt: node_count {node_count} does not match \
                 the segment's {} unique record doc ids",
                unique_ids.len()
            )));
        }

        // Allocation bounding (#806); v2 strides are 4 bytes per node
        // minimum (layer_count) and 4 bytes per neighbour ordinal.
        let graph_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        checked_capacity(node_count, 4, graph_remaining, "hnsw node_count")?;

        let mut nodes = HashMap::with_capacity(node_count);
        // node_count == unique_ids.len() (validated above), so iterating
        // the ordinal table walks exactly the serialized node sequence.
        for &doc_id in unique_ids.iter() {
            let mut layer_count_buf = [0u8; 4];
            input.read_exact(&mut layer_count_buf)?;
            let layer_count = u32::from_le_bytes(layer_count_buf) as usize;

            checked_capacity(layer_count, 4, graph_remaining, "hnsw layer_count")?;
            let mut layers = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let mut neighbor_count_buf = [0u8; 4];
                input.read_exact(&mut neighbor_count_buf)?;
                let neighbor_count = u32::from_le_bytes(neighbor_count_buf) as usize;

                checked_capacity(neighbor_count, 4, graph_remaining, "hnsw neighbor_count")?;
                let mut neighbors = Vec::with_capacity(neighbor_count);
                let mut neighbor_buf = [0u8; 4];
                for _ in 0..neighbor_count {
                    input.read_exact(&mut neighbor_buf)?;
                    neighbors.push(doc_of(u32::from_le_bytes(neighbor_buf))?);
                }
                layers.push(neighbors);
            }
            nodes.insert(doc_id, layers);
        }

        Ok((entry_point, max_level, nodes))
    }

    /// Load an existing HNSW index from storage.
    pub fn load(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut input = storage.open_input(&file_name)?;

        // Ground truth for bounding allocations sized from unverified header
        // counts below (Issue #806). Unlike the reader, this writer load path
        // runs no checksum footer verification, so every count — including
        // those of footer-carrying segments — reaches its allocation unverified.
        let file_size = input.size()?;

        // Read metadata (vector count stored as u64)
        let mut num_vectors_buf = [0u8; 8];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u64::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut m_buf = [0u8; 4];
        input.read_exact(&mut m_buf)?;
        let _m = u32::from_le_bytes(m_buf) as usize;

        let mut ef_construction_buf = [0u8; 4];
        input.read_exact(&mut ef_construction_buf)?;
        let _ef_construction = u32::from_le_bytes(ef_construction_buf) as usize;

        if dimension != index_config.dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read the Issue #481 Stage 1 / Stage 3 vector segment header
        // (LVS1). Pre-Stage-1 segments are rejected with
        // IncompatibleFormat by the reader. Both Scalar8Bit and
        // ProductQuantization (Stage 3, #481) variants are reconstituted
        // back to f32 for the writer's in-memory state — the on-disk
        // form is rebuilt from scratch in `write()` once add_vector /
        // delete_document calls have replayed.
        // Issue #921: pass the bytes physically left in the file so the
        // header's PQ codebook allocation is bounded before it reserves.
        let header_available =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let header = VectorSegmentHeader::read_from(&mut input, header_available)?;

        // Read quantized vectors and dequantize back to f32 for the
        // in-memory writer state. The dequantized values are a lossy
        // approximation of the originals; if a Stage 2 sidecar is
        // present we'll overwrite them below with the lossless f32
        // payload.
        // Bytes left for the per-vector records section (Issue #806). Each
        // record is at least doc_id (8) + field_name_len (4) + the per-kind
        // fixed payload, so `record_stride` also bounds the per-record payload
        // read decoded below.
        let records_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let min_payload = match &header.quant {
            QuantHeader::Scalar8Bit(_) => quantized_record_payload_size(dimension) as u64,
            QuantHeader::ProductQuantization { params, .. } => params.m as u64,
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { .. } => 1,
        };
        let record_stride = record_prefix_size(header.version) + min_payload;
        checked_capacity(
            num_vectors,
            record_stride,
            records_remaining,
            "hnsw num_vectors",
        )?;
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);

            // Field reference: dictionary id (v3+) or inline name.
            let field_name =
                header.read_record_field(&mut input, records_remaining, "hnsw field_name_len")?;

            // Decode the per-vector payload according to the segment's
            // quantization kind.
            let values = match &header.quant {
                QuantHeader::Scalar8Bit(params) => {
                    read_dequantized_vector(&mut input, dimension, params)?
                }
                QuantHeader::ProductQuantization { params, codebook } => {
                    crate::vector::index::pq_io::read_dequantized_pq_vector(
                        &mut input, *params, codebook,
                    )?
                }
                #[cfg(feature = "pq-fastscan")]
                QuantHeader::ProductQuantizationFastScan { params, codebook } => {
                    crate::vector::index::pq_fastscan_io::read_dequantized_pq_fastscan_vector(
                        &mut input, *params, codebook,
                    )?
                }
            };

            vectors.push((doc_id, field_name, Vector::new(values)));
        }

        // Stage 2 (Issue #481): if the LRS1 sidecar exists alongside
        // the main `.hnsw` file, prefer its lossless f32 payload over
        // the dequantized int8 values. This keeps the in-memory writer
        // state byte-exact across load -> add -> write cycles, so a
        // re-emitted sidecar does not slowly bleed precision through
        // repeated dequant -> requantize round-trips.
        let sidecar_name = format!("{}.f32", file_name);
        if storage.file_exists(&sidecar_name) {
            let mut sidecar_in = storage.open_input(&sidecar_name)?;
            let sidecar_size = sidecar_in.size()?;
            let (header, payload) = read_sidecar(&mut sidecar_in, sidecar_size)?;
            if header.dim as usize != dimension {
                return Err(LaurusError::InvalidOperation(format!(
                    "rerank sidecar dim mismatch: LVS1 segment uses {dimension}, sidecar uses {}",
                    header.dim
                )));
            }
            if header.vector_count as usize != vectors.len() {
                return Err(LaurusError::InvalidOperation(format!(
                    "rerank sidecar vector_count mismatch: LVS1 segment has {} vectors, \
                     sidecar has {}",
                    vectors.len(),
                    header.vector_count
                )));
            }
            match header.storage_kind {
                RerankStorageKind::F32 => {
                    let bytes_per_vec = dimension * 4;
                    for (i, (_, _, vec)) in vectors.iter_mut().enumerate() {
                        let start = i * bytes_per_vec;
                        let mut data = Vec::with_capacity(dimension);
                        for j in 0..dimension {
                            let lo = start + j * 4;
                            data.push(f32::from_le_bytes([
                                payload[lo],
                                payload[lo + 1],
                                payload[lo + 2],
                                payload[lo + 3],
                            ]));
                        }
                        *vec = Vector::new(data);
                    }
                }
            }
        }

        // Rebuild doc_id_map
        let mut doc_id_map = HashMap::new();
        for (i, (doc_id, _, _)) in vectors.iter().enumerate() {
            doc_id_map.insert(*doc_id, i);
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        if index_config.m < 2 {
            return Err(LaurusError::InvalidOperation(
                "HNSW parameter m must be >= 2".to_string(),
            ));
        }
        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        // Read graph data if present. The writer keeps a doc_id-keyed
        // in-memory graph, so a v2 (ordinal-encoded, Issue #686) block is
        // translated back to doc ids via the record order; a v1 block is
        // read verbatim.
        let mut has_graph_buf = [0u8; 1];
        let graph = if input.read_exact(&mut has_graph_buf).is_ok() {
            if has_graph_buf[0] == 1 {
                let (entry_point, max_level, nodes) = if header.version >= VERSION_ORDINAL_GRAPH {
                    Self::load_graph_block_v2(&mut input, file_size, &vectors)?
                } else {
                    Self::load_graph_block_v1(&mut input, file_size)?
                };

                Some(HnswGraph::new(
                    entry_point,
                    max_level,
                    nodes,
                    index_config.m,
                    index_config.m,
                    index_config.m * 2,
                    index_config.ef_construction,
                    _ml,
                ))
            } else {
                None
            }
        } else {
            None
        };

        // If we loaded a graph, we are not "finalized" in the sense that we can't append.
        // We want to support append, so we should allow modifications if loaded.
        // Previously, is_finalized=true prevented modifications.
        // For append support, we set is_finalized=false.

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path: path.to_string(),
            _ml,
            levels: vec![Vec::new(); max_level + 1], // Still re-init levels, but they are conceptually in the graph
            entry_point: graph.as_ref().and_then(|g| g.entry_point),
            vectors,
            is_finalized: false, // Changed to false to allow appending
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
            doc_id_map,
            graph,
        })
    }

    /// Set HNSW-specific parameters.
    pub fn with_hnsw_params(mut self, m: usize, ef_construction: usize) -> Self {
        self.index_config.m = m;
        self.index_config.ef_construction = ef_construction;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
    }

    /// Calculate the layer for a new vector.
    ///
    /// # Arguments
    ///
    /// * `rng` - A caller-supplied RNG. Callers seed one independently per
    ///   node via [`level_rng_seed_for`] so level assignment parallelizes
    ///   across nodes (Issue #637) while staying deterministic for a given
    ///   doc_id (Issue #841).
    ///
    /// # Returns
    ///
    /// The layer index, geometrically distributed with ratio `_ml` and
    /// capped at 16.
    fn select_layer(&self, rng: &mut impl RngExt) -> usize {
        let mut layer = 0;

        while rng.random_range(0.0..1.0) < self._ml && layer < 16 {
            layer += 1;
        }

        layer
    }

    /// Assign a level to each doc_id in `doc_ids`, in parallel when not
    /// targeting wasm32 (Issue #637). `rayon` has no thread pool on
    /// `wasm32-unknown-unknown`, so that target falls back to a plain
    /// sequential loop; both paths use the same per-doc-id-seeded scheme
    /// (`level_rng_seed_for`), so the result is identical either way, just
    /// computed with or without threads.
    fn assign_levels(&self, doc_ids: &[u64]) -> Vec<(u64, usize)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            doc_ids
                .par_iter()
                .map(|&doc_id| {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(level_rng_seed_for(doc_id));
                    (doc_id, self.select_layer(&mut rng))
                })
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            doc_ids
                .iter()
                .map(|&doc_id| {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(level_rng_seed_for(doc_id));
                    (doc_id, self.select_layer(&mut rng))
                })
                .collect()
        }
    }

    /// Calculate the maximum level based on M and ef_construction.
    /// This is a heuristic, often 1/ln(M) or 1/ln(2) is used for probability.
    /// For simplicity, we can cap it or use a fixed formula.
    /// A common formula for max_level is based on the number of elements and M.
    /// For now, let's use a simple heuristic or a fixed max.
    fn calculate_max_level(_m: usize, _ef_construction: usize) -> usize {
        // A common heuristic is to have max_level around log_M(N) or a fixed small number.
        // For now, let's use a fixed small number or a simple formula.
        // The original code used 1/ln(2) for probability, which implies levels grow with log_2(N).
        // Let's set a reasonable cap, e.g., 16 or 32.
        // Or, based on the probability p = 1/ln(M), the expected max level for N elements is log_p(N).
        // For simplicity, let's use a fixed max level for now, or a simple calculation.
        // The `select_layer` uses `1.0 / (self.index_config.m as f64).ln()` as probability.
        // Let's assume a max level that allows for a reasonable number of layers.
        // For example, if M=16, 1/ln(16) approx 0.36.
        // A max level of 16-32 is common.
        16 // A reasonable default max level
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &Vec<(u64, String, Vector)>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, _, vector) in vectors {
            if vector.dimension() != self.index_config.dimension {
                return Err(LaurusError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(LaurusError::InvalidOperation(format!(
                    "Vector {doc_id} contains invalid values (NaN or infinity)"
                )));
            }
        }

        Ok(())
    }

    /// Normalize vectors if configured to do so.
    /// Normalize vectors if configured to do so.
    #[allow(unused_variables)]
    fn normalize_vectors_internal(
        index_config: &HnswIndexConfig,
        writer_config: &VectorIndexWriterConfig,
        vectors: &mut Vec<(u64, String, Vector)>,
    ) {
        if !index_config.normalize_vectors {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        if writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
            return;
        }

        for (_, _, vector) in vectors {
            vector.normalize();
        }
    }

    /// Initialize lookups for fast vector access
    fn rebuild_doc_id_map(&mut self) {
        self.doc_id_map.clear();
        for (idx, (doc_id, _, _)) in self.vectors.iter().enumerate() {
            self.doc_id_map.insert(*doc_id, idx);
        }
    }

    /// Build the HNSW graph structure.
    fn build_hnsw_graph(&mut self) -> Result<()> {
        let count = self.vectors.len();
        if count == 0 {
            return Ok(());
        }

        // TODO: replace with tracing::info! when a logging crate is added
        // "Building HNSW graph with {count} vectors (parallel), M={m}, efConstruction={ef}"

        // Ensure doc_id_map is up to date
        self.rebuild_doc_id_map();

        let m = self.index_config.m;
        let m_max = m;
        let m_max_0 = m * 2;
        let ef_construction = self.index_config.ef_construction;

        // Determine which vectors are new and need insertion
        let mut new_doc_ids_in_order = Vec::new();

        // Rebuild from scratch instead of appending when the loaded base is
        // too small to serve as a routing hierarchy for the new nodes (Issue
        // #872). A base below `ef_construction` has a near-flat hierarchy (max
        // level ~0-1), so appending many higher-level nodes onto it can only
        // route through that low-level core and fragments the graph. Since
        // `self.vectors` already holds the existing + new vectors and such a
        // base is by definition small, discarding it and doing a full build is
        // both correct and cheap — and yields fresh-build quality. A larger
        // base keeps the incremental path (its hierarchy is real and a rebuild
        // would be wasteful).
        let existing = self
            .graph
            .take()
            .filter(|g| g.node_count() >= ef_construction);

        // Check if we have an existing graph to append to
        let (graph, entry_point, max_level, search_entry_point, promoted_ep) =
            if let Some(existing_graph) = existing {
                // Identify new vectors
                for (doc_id, _, _) in &self.vectors {
                    if !existing_graph.contains_node(doc_id) {
                        new_doc_ids_in_order.push(*doc_id);
                    }
                }
                new_doc_ids_in_order.sort_unstable();

                // Assign levels to new vectors (Issue #637).
                let new_node_levels: Vec<(u64, usize)> = self.assign_levels(&new_doc_ids_in_order);

                let current_max_level = existing_graph.max_level;
                let new_max_level = new_node_levels.iter().map(|(_, l)| *l).max().unwrap_or(0);
                let total_max_level = current_max_level.max(new_max_level);

                let old_ep = existing_graph.entry_point;
                let mut ep = old_ep;

                // If new nodes reach a higher level than the loaded graph, the
                // entry point is promoted to one of them. `promoted_ep` records
                // that new top-level node so it can be inserted first and used
                // as the build search start (Issue #872) — otherwise searches
                // start from the low-level `old_ep` and can't route through the
                // upper layers, funneling every insert through the small base.
                let mut promoted_ep = None;
                if new_max_level > current_max_level
                    && let Some(new_top) = new_node_levels
                        .iter()
                        .find(|(_, l)| *l == total_max_level)
                        .map(|(id, _)| *id)
                {
                    ep = Some(new_top);
                    promoted_ep = Some(new_top);
                }

                // Convert to ConcurrentHnswGraph and extend
                let mut concurrent_graph =
                    ConcurrentHnswGraph::from_hnsw_graph(existing_graph, total_max_level);
                concurrent_graph.add_nodes(new_node_levels.clone());

                // Insert searches start from the loaded graph's entry point so
                // new nodes connect to the existing component; the promoted top
                // node (if any) is inserted first from here, then becomes the
                // start (see the loop below).
                let search_ep = old_ep.or(ep);

                (
                    concurrent_graph,
                    ep,
                    total_max_level,
                    search_ep,
                    promoted_ep,
                )
            } else {
                // Full build
                let mut doc_ids_in_order: Vec<u64> =
                    self.vectors.iter().map(|(id, _, _)| *id).collect();
                doc_ids_in_order.sort_unstable();

                // Assign levels to all vectors (Issue #637).
                let new_node_levels: Vec<(u64, usize)> = self.assign_levels(&doc_ids_in_order);

                let max_level = new_node_levels.iter().map(|(_, l)| *l).max().unwrap_or(0);
                let ep = new_node_levels
                    .iter()
                    .find(|(_, l)| *l == max_level)
                    .map(|(id, _)| *id);

                new_doc_ids_in_order = doc_ids_in_order;

                let concurrent_graph = ConcurrentHnswGraph::new(new_node_levels.clone(), max_level);
                // Full build: the seed `ep` is already the max-level node, so
                // the search start needs no promotion.
                (concurrent_graph, ep, max_level, ep, None)
            };

        // 3. Concurrent insertion (Issues #868 / #621).
        //
        // Every new node is pre-populated into the graph with an empty
        // neighbor list but marked *unlinked*; a node becomes visible to
        // build-time search only after `mark_linked` is called at the end of
        // its own insertion (once its forward edges are set at all levels).
        // This is the fix for #868: without it, a concurrent `search_layer`
        // could select a not-yet-inserted node (empty list, a dead end) as a
        // neighbor, connect far away, get its back-edge pruned, and leave the
        // node unreachable from the entry point — silent recall loss that at
        // scale disconnected ~96% of the index. The visibility gate makes each
        // concurrent insert see only fully-linked nodes, matching serial HNSW
        // quality; a serial bootstrap warms a connected core so the first
        // parallel inserts do not all pile onto the lone seed; and a
        // connectivity-repair pass (below) is the hard backstop that
        // guarantees full reachability regardless of interleaving.
        let writer_ref = &*self;

        // The initial search-start node must be visible before any worker runs
        // so traversals that reach it are not filtered out. For an incremental
        // build it is the (already-linked) old entry point; for a full build
        // it is the seed, which has no edges yet but is a valid, linked start.
        if let Some(sp) = search_entry_point {
            graph.mark_linked(sp);
        }

        let insert_one = |doc_id: u64, start_node: u64| -> Result<()> {
            // Skip insertion of the search start node itself (the seed / entry
            // point — other nodes link TO it via bidirectional edges).
            if start_node == doc_id {
                return Ok(());
            }

            let doc_vector_idx = *writer_ref.doc_id_map.get(&doc_id).ok_or_else(|| {
                LaurusError::internal(format!("Doc ID {} not found in doc_id_map", doc_id))
            })?;
            let vector = &writer_ref.vectors[doc_vector_idx].2;

            // Determine the assigned level from the pre-populated graph
            let layers_len = graph
                .nodes
                .get(&doc_id)
                .map(|e| e.layers.len())
                .unwrap_or(0);
            if layers_len == 0 {
                return Ok(());
            }
            let level = layers_len - 1;

            let max_level = graph.max_level;
            let mut curr_obj = start_node;
            let mut dist = writer_ref.calc_dist(vector, curr_obj)?;

            // Scratch buffer for `copy_neighbors_into` (Issue #1137), reused
            // across every iteration of the descent loop below instead of
            // allocating a fresh `Vec<u64>` per level. A plain local
            // suffices here (no thread-local needed, unlike `search_layer`'s
            // `SearchLayerArena`): Phase A never calls `search_layer`, so
            // this buffer's lifetime never needs to span or nest with that
            // arena's own borrow.
            let mut neighbor_buf: Vec<u64> = Vec::new();

            // Phase A: Greedy descent from top layer down to level + 1
            for lc in (level + 1..=max_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    graph.copy_neighbors_into(curr_obj, lc, &mut neighbor_buf);
                    for &neighbor_id in &neighbor_buf {
                        // Skip nodes still being inserted (Issue #868).
                        if !graph.is_visible(neighbor_id) {
                            continue;
                        }
                        let d = writer_ref.calc_dist(vector, neighbor_id)?;
                        if d < dist {
                            dist = d;
                            curr_obj = neighbor_id;
                            changed = true;
                        }
                    }
                }
            }

            // Phase B: Search & connect from min(max_level, level) down to 0
            let top_level = usize::min(max_level, level);
            for lc in (0..=top_level).rev() {
                let candidates =
                    writer_ref.search_layer(&graph, curr_obj, vector, ef_construction, lc)?;

                if let Some(min_cand) = candidates
                    .iter()
                    .min_by(|a, b| a.distance.total_cmp(&b.distance))
                {
                    curr_obj = min_cand.id;
                }

                let neighbors = writer_ref.select_neighbors(&candidates, m, lc, m_max, m_max_0);

                graph.set_neighbors(doc_id, lc, neighbors.clone());

                for neighbor_id in neighbors {
                    let current_m_max = if lc == 0 { m_max_0 } else { m_max };
                    graph.add_neighbor_with_pruning(
                        neighbor_id,
                        lc,
                        doc_id,
                        current_m_max,
                        writer_ref,
                    )?;
                }
            }

            // Publish this node as fully linked — MUST be the last step, after
            // every `set_neighbors`, so a concurrent reader observing the flag
            // (Acquire) sees all forward edges (Issue #868).
            graph.mark_linked(doc_id);
            Ok(())
        };

        // Serial bootstrap: insert the first `bootstrap_count` new nodes
        // sequentially to warm a connected core before going parallel, so the
        // first parallel inserts fill a full search frontier from mature nodes
        // instead of all piling onto a cold seed and pruning each other off
        // (Issue #868 / #621). The size adapts to the ALREADY-linked core so
        // it covers every case the build type alone would misjudge: a fresh
        // full build has only the lone seed linked and needs a full ef-sized
        // warm-up; an incremental append onto a large base needs none; and an
        // incremental append onto a *small* base (e.g. seed-then-bulk-load)
        // still needs a warm-up despite being "incremental". `existing_linked`
        // is the count of already-linked nodes (loaded core, or ~0 for a fresh
        // build); bootstrap fills the gap up to `ef_construction`.
        //
        // NB: warming the core does not fully fix seed-then-bulk-load — that
        // pattern also suffers from a low-level entry point, handled (as a
        // quality follow-up) in Issue #872; the connectivity repair below is
        // the correctness backstop for it regardless.
        //
        // The `effective_start` is the node every insert searches from. When
        // the append promoted the entry point to a new top-level node (#872),
        // that node is inserted FIRST (searching from the low-level `old_ep`,
        // so it joins the existing component) and then becomes the start —
        // matching the full-build regime where searches descend from the
        // max-level node. Otherwise it stays the loaded/seed entry point.
        if let Some(start) = search_entry_point {
            let effective_start = match promoted_ep {
                Some(pep) => {
                    insert_one(pep, start)?;
                    pep
                }
                None => start,
            };

            const BOOTSTRAP_FLOOR: usize = 32;
            let new_count = new_doc_ids_in_order.len();
            let existing_linked = graph.nodes.len().saturating_sub(new_count);
            let bootstrap_count = ef_construction
                .max(BOOTSTRAP_FLOOR)
                .saturating_sub(existing_linked)
                .min(new_count);
            let parallel_ids: Vec<u64> = new_doc_ids_in_order.split_off(bootstrap_count);
            for doc_id in new_doc_ids_in_order {
                insert_one(doc_id, effective_start)?;
            }

            #[cfg(not(target_arch = "wasm32"))]
            parallel_ids
                .into_par_iter()
                .try_for_each(|doc_id| insert_one(doc_id, effective_start))?;
            #[cfg(target_arch = "wasm32")]
            for doc_id in parallel_ids {
                insert_one(doc_id, effective_start)?;
            }
        }

        // 4. Convert ConcurrentGraph to HnswGraph
        let mut final_nodes = HashMap::new();
        let mut final_levels_map = HashMap::new();

        for (doc_id, entry) in graph.nodes {
            let mut vec_layers = Vec::with_capacity(entry.layers.len());
            for lock in entry.layers {
                vec_layers.push(lock.into_inner()); // Consume RwLock
            }
            final_levels_map.insert(doc_id, vec_layers.len() - 1);
            final_nodes.insert(doc_id, vec_layers);
        }

        // 5. Connectivity repair (Issue #868). The concurrent build produces a
        // near-serial-quality graph, but interleaving can still leave a few
        // nodes with no in-edge and thus unreachable from the entry point.
        // Guarantee full layer-0 reachability by reconnecting any residual
        // stragglers to their nearest reachable node. For a healthy build this
        // touches ~0 nodes.
        if let Some(ep) = entry_point {
            self.repair_layer0_connectivity(ep, &mut final_nodes)?;
        }

        self.graph = Some(HnswGraph::new(
            entry_point,
            max_level,
            final_nodes,
            m,
            m_max,
            m_max_0,
            ef_construction,
            1.0 / (self.index_config.m as f64).ln(),
        ));
        self.entry_point = entry_point;

        // Rebuild self.levels
        let mut levels_vec = vec![Vec::new(); max_level + 1];
        for (doc_id, level) in final_levels_map {
            if level < levels_vec.len() {
                levels_vec[level].push(doc_id);
            }
        }
        self.levels = levels_vec;

        Ok(())
    }

    /// Calculate distance between a query vector and a document already
    /// resolved to its dense `vectors`/`doc_id_map` index. Split out of
    /// [`Self::calc_dist`] (Issue #632) so hot loops that already have the
    /// index (e.g. `search_layer`'s neighbor traversal) don't pay for a
    /// second `doc_id_map` lookup.
    fn calc_dist_by_idx(&self, query: &Vector, idx: usize) -> Result<f32> {
        let target = &self.vectors[idx].2;
        self.index_config
            .distance_metric
            .distance(&query.data, &target.data)
    }

    // Calculates distance between a query vector and a document in the index
    fn calc_dist(&self, query: &Vector, doc_id: u64) -> Result<f32> {
        let idx = *self
            .doc_id_map
            .get(&doc_id)
            .ok_or_else(|| LaurusError::internal(format!("Doc ID {} not found in map", doc_id)))?;
        self.calc_dist_by_idx(query, idx)
    }

    /// Search for nearest neighbors in a specific layer.
    ///
    /// Uses the thread-local [`SearchLayerArena`] (Issue #632) instead of
    /// allocating a fresh visited-set and two heaps on every call.
    fn search_layer<G: GraphView>(
        &self,
        graph: &G,
        entry_point: u64,
        query: &Vector,
        ef: usize,
        level: usize,
    ) -> Result<Vec<Candidate>> {
        SEARCH_LAYER_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();
            arena.reset(self.vectors.len());

            // We use min-heap for "results" to keep track of nearest found?
            // No, HNSW "v" list (candidates to visit) is min-heap (nearest first).
            // "C" list (found candidates) is max-heap (furthest first) to keep ef smallest.

            let entry_idx = *self.doc_id_map.get(&entry_point).ok_or_else(|| {
                LaurusError::internal(format!("Doc ID {} not found in map", entry_point))
            })?;
            let dist = self.calc_dist_by_idx(query, entry_idx)?;

            arena.to_visit.push(VisitorCandidate {
                id: entry_point,
                distance: dist,
            });
            arena.found.push(Candidate {
                id: entry_point,
                distance: dist,
                similarity: 0.0,
            });
            arena.mark_visited(entry_idx);

            while let Some(curr) = arena.to_visit.pop() {
                // If closest candidate to visit is further than the furthest found candidate, and we found enough, stop
                if let Some(furthest_found) = arena.found.peek()
                    && curr.distance > furthest_found.distance
                    && arena.found.len() >= ef
                {
                    break;
                }

                graph.copy_neighbors_into(curr.id, level, &mut arena.neighbor_buf);
                // Indexed rather than `for x in &arena.neighbor_buf`: `arena`
                // is a `RefMut`, so every field access goes through
                // `DerefMut` and Rust's field-disjoint-borrow splitting does
                // not apply — an iterator borrowing `neighbor_buf` would
                // conflict with `arena.mark_visited`/`arena.found.push`
                // below, which need a mutable borrow of the whole struct.
                // Indexing only borrows for the instant of reading a `Copy`
                // `u64`, which ends immediately (NLL), so it coexists fine.
                for i in 0..arena.neighbor_buf.len() {
                    let neighbor_id = arena.neighbor_buf[i];
                    // Skip nodes still being inserted (Issue #868). Checked
                    // before the visited check so a node hidden now can still
                    // be discovered later in the same search if it becomes
                    // linked. A no-op for query-time / finished graphs
                    // (`is_visible` defaults to `true`).
                    if !graph.is_visible(neighbor_id) {
                        continue;
                    }

                    // Resolve the dense index once and reuse it for both the
                    // visited-check and the distance calculation (Issue
                    // #632) — this keeps the already-visited case at one
                    // lookup (was zero) but the newly-visited case at one
                    // lookup (was two: `visited.insert` here plus
                    // `calc_dist`'s own lookup), so it doesn't regress net
                    // lookup count. Resolved before the visited check so a
                    // missing id still errors on first encounter, matching
                    // today's behaviour.
                    let idx = *self.doc_id_map.get(&neighbor_id).ok_or_else(|| {
                        LaurusError::internal(format!(
                            "Doc ID {} not found in doc_id_map",
                            neighbor_id
                        ))
                    })?;
                    if !arena.mark_visited(idx) {
                        continue;
                    }

                    let neighbor_dist = self.calc_dist_by_idx(query, idx)?;
                    let furthest_dist = arena.found.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < furthest_dist || arena.found.len() < ef {
                        arena.found.push(Candidate {
                            id: neighbor_id,
                            distance: neighbor_dist,
                            similarity: 0.0,
                        });
                        arena.to_visit.push(VisitorCandidate {
                            id: neighbor_id,
                            distance: neighbor_dist,
                        });

                        if arena.found.len() > ef {
                            arena.found.pop();
                        }
                    }
                }
            }

            Ok(arena.found.iter().cloned().collect())
        })
    }

    fn select_neighbors(
        &self,
        candidates: &[Candidate],
        m: usize,
        _level: usize,
        _m_max: usize,
        _m_max_0: usize,
    ) -> Vec<u64> {
        // Simple heuristic: take M nearest.
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        sorted.truncate(m);
        sorted.into_iter().map(|c| c.id).collect()
    }

    fn prune_neighbors(
        &self,
        doc_id: u64,
        neighbors: Vec<u64>,
        max_conn: usize,
    ) -> Result<Vec<u64>> {
        if neighbors.len() <= max_conn {
            return Ok(neighbors);
        }

        // Sort by distance from doc_id
        let idx = *self.doc_id_map.get(&doc_id).ok_or_else(|| {
            LaurusError::internal(format!(
                "Doc ID {} not found in doc_id_map during pruning",
                doc_id
            ))
        })?;
        let doc_vec = &self.vectors[idx].2;

        let mut candidates = Vec::new();
        for nid in neighbors {
            let dist = self.calc_dist(doc_vec, nid)?;
            candidates.push(Candidate {
                id: nid,
                distance: dist,
                similarity: 0.0,
            });
        }

        // We want to keep nearest. Move to min-heap or just sort.
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.truncate(max_conn);

        Ok(candidates.into_iter().map(|c| c.id).collect())
    }

    /// Guarantee that every node is reachable from `entry` over the layer-0
    /// adjacency (Issue #868 / #621).
    ///
    /// The concurrent build is high quality but not guaranteed connected —
    /// interleaving can leave a few nodes with no in-edge. This is the hard
    /// backstop: BFS from `entry`, and for each still-unreachable node (in
    /// ascending doc_id order for reproducibility) add an in-edge from its
    /// nearest reachable node **without pruning**, so the repaired node is
    /// guaranteed to survive; then extend the reachable set through it so one
    /// repair edge fixes an entire disconnected component. For a healthy build
    /// the repair count is ~0.
    ///
    /// # Parameters
    ///
    /// - `entry` - The graph entry point (BFS root) doc id.
    /// - `nodes` - The final `doc_id -> per-level neighbor lists` map, mutated
    ///   in place at level 0.
    ///
    /// # Returns
    ///
    /// The number of repair edges added (a build-quality signal; ~0 expected).
    ///
    /// # Errors
    ///
    /// Returns an error if a doc id is missing from `doc_id_map` during the
    /// nearest-node distance scan.
    fn repair_layer0_connectivity(
        &self,
        entry: u64,
        nodes: &mut HashMap<u64, Vec<Vec<u64>>>,
    ) -> Result<usize> {
        use std::collections::VecDeque;

        // BFS from `entry` over layer-0 adjacency to collect the reachable set.
        fn bfs_from(start: u64, nodes: &HashMap<u64, Vec<Vec<u64>>>, reachable: &mut HashSet<u64>) {
            let mut queue = VecDeque::new();
            if reachable.insert(start) {
                queue.push_back(start);
            }
            while let Some(cur) = queue.pop_front() {
                if let Some(layers) = nodes.get(&cur)
                    && let Some(level0) = layers.first()
                {
                    for &nb in level0 {
                        if reachable.insert(nb) {
                            queue.push_back(nb);
                        }
                    }
                }
            }
        }

        let mut reachable = HashSet::new();
        bfs_from(entry, nodes, &mut reachable);

        // Deterministic order over the residual unreachable nodes.
        let mut remaining: Vec<u64> = nodes
            .keys()
            .copied()
            .filter(|id| !reachable.contains(id))
            .collect();
        remaining.sort_unstable();

        // Bound the total repair cost. A healthy concurrent build leaves only
        // a handful of components (~13/20000 measured), so scanning the whole
        // reachable set for each is cheap and gives the best-quality
        // reconnection. But a pathological build (e.g. bulk-appending far more
        // nodes than an existing tiny base — the cold-start regime) can
        // fragment into thousands of components; a per-component full scan
        // would then be O(components × |reachable|) ≈ O(N²). So the first
        // `FULL_SCAN_BUDGET` components get the exact nearest-reachable node,
        // and any beyond that attach straight to the entry point (always
        // reachable) — connectivity is still guaranteed, at O(1) each, keeping
        // the whole pass linear. Reaching the budget is itself a signal that
        // the build fragmented; see the seed-then-bulk-load follow-up.
        const FULL_SCAN_BUDGET: usize = 512;
        let mut repairs = 0usize;
        for u in remaining {
            // May have been pulled in by an earlier repair's cascade.
            if reachable.contains(&u) {
                continue;
            }

            // Pick the in-edge source V: the exact nearest reachable node while
            // within the scan budget, else the entry point (bounded fallback).
            let v = if repairs < FULL_SCAN_BUDGET {
                let u_vec = {
                    let idx = *self.doc_id_map.get(&u).ok_or_else(|| {
                        LaurusError::internal(format!(
                            "Doc ID {u} not found during connectivity repair"
                        ))
                    })?;
                    &self.vectors[idx].2
                };
                let mut best: Option<(f32, u64)> = None;
                for &v in &reachable {
                    let d = self.calc_dist(u_vec, v)?;
                    match best {
                        Some((bd, bid)) if !(d < bd || (d == bd && v < bid)) => {}
                        _ => best = Some((d, v)),
                    }
                }
                match best {
                    Some((_, v)) => v,
                    // `reachable` is non-empty (contains `entry`), so this is
                    // unreachable in practice; skip defensively.
                    None => continue,
                }
            } else {
                entry
            };

            // Add the in-edge V -> U at level 0 without pruning, so U cannot be
            // dropped again.
            if let Some(layers) = nodes.get_mut(&v)
                && let Some(level0) = layers.first_mut()
                && !level0.contains(&u)
            {
                level0.push(u);
            }
            repairs += 1;

            // Extend the reachable set through U (fixes its whole component).
            bfs_from(u, nodes, &mut reachable);
        }

        Ok(repairs)
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(LaurusError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    /// Whether a document is currently buffered in this writer (O(1)).
    ///
    /// Used by the segmented search path's newest-source-wins masking
    /// (Issue #880) to decide whether the active buffer shadows a sealed
    /// segment's copy of the same document.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID to probe.
    ///
    /// # Returns
    ///
    /// `true` when the buffer holds at least one vector for `doc_id`.
    pub fn contains_doc(&self, doc_id: u64) -> bool {
        self.doc_id_map.contains_key(&doc_id)
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.index_config.m, self.index_config.ef_construction)
    }
}

#[async_trait::async_trait]
impl VectorIndexWriter for HnswIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;

        self.vectors = vectors;
        Self::normalize_vectors_internal(
            &self.index_config,
            &self.writer_config,
            &mut self.vectors,
        );
        self.rebuild_doc_id_map();

        // Update next_vec_id
        if let Some((max_id, _, _)) = self.vectors.iter().max_by_key(|(id, _, _)| id)
            && *max_id >= self.next_vec_id
        {
            self.next_vec_id = *max_id + 1;
        }

        self.total_vectors_to_add = Some(self.vectors.len());

        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }

        self.validate_vectors(&vectors)?;
        Self::normalize_vectors_internal(&self.index_config, &self.writer_config, &mut vectors);

        // Ensure doc_id_map is up to date
        self.rebuild_doc_id_map();

        for (doc_id, field, vector) in vectors {
            if let Some(&idx) = self.doc_id_map.get(&doc_id) {
                // Update existing vector
                self.vectors[idx] = (doc_id, field, vector);
            } else {
                // Add new vector
                let idx = self.vectors.len();
                self.vectors.push((doc_id, field, vector));
                self.doc_id_map.insert(doc_id, idx);
            }
        }

        // Update next_vec_id
        if let Some((max_id, _, _)) = self.vectors.iter().max_by_key(|(id, _, _)| id)
            && *max_id >= self.next_vec_id
        {
            self.next_vec_id = *max_id + 1;
        }

        self.check_memory_limit()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.is_finalized {
            return Ok(());
        }

        // Build the actual HNSW graph structure
        self.build_hnsw_graph()?;

        self.is_finalized = true;
        Ok(())
    }

    fn progress(&self) -> f32 {
        if let Some(total) = self.total_vectors_to_add {
            if total == 0 {
                if self.is_finalized { 1.0 } else { 0.0 }
            } else {
                let current = self.vectors.len() as u64 as f32;
                let progress = current / total as f32;
                if self.is_finalized {
                    1.0
                } else {
                    progress.min(0.99) // Never report 100% until finalized
                }
            }
        } else if self.is_finalized {
            1.0
        } else {
            0.0
        }
    }

    fn estimated_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len()
            * (
                8 + // doc_id (tuple element)
            32 + // field_name string overhead (approx)
            self.index_config.dimension * 4
                // f32 values
            );

        // HNSW graph overhead (rough estimate)
        // Each vector can have up to M connections per layer
        // Average layers per vector is approximately 1/(1-p) where p=0.5
        let avg_layers = 2.0;
        let graph_memory =
            self.vectors.len() * (self.index_config.m as f32 * avg_layers * 8.0) as usize;

        let metadata_memory = self.vectors.len() * 128; // Increased for graph structure

        vector_memory + graph_memory + metadata_memory
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self) -> Result<()> {
        use std::io::Write;

        if !self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| LaurusError::InvalidOperation("No storage configured".to_string()))?;

        // Write to a temp file and atomically rename into place (Issue #784)
        // so a crash mid-write leaves the previously committed `.hnsw` intact
        // instead of a truncated, unreadable segment.
        let file_name = format!("{}.hnsw", self.path);
        let tmp_name = format!("{}.hnsw.tmp", self.path);
        // Wrap the output so a CRC-32 accumulates over the segment bytes as
        // they are written; a checksum footer is appended below (Issue #786).
        let mut output =
            crate::storage::checksum::CrcWriter::new(storage.create_output(&tmp_name)?);

        // Write metadata (vector count as u64 to avoid truncation)
        output.write_all(&(self.vectors.len() as u64).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.m as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.ef_construction as u32).to_le_bytes())?;

        // Write vectors using the Issue #481 quantized format. The
        // HNSW-specific 28-byte preamble above (count / dim / m / ef)
        // stays unchanged so the graph parameters are still readable
        // first; the vector payload is quantized to int8 (Stage 1) or
        // PQ codes (Stage 3, #481) according to the field's
        // `quantization_method`, prefixed by `VectorSegmentHeader`
        // (LVS1).

        // Sort by doc_id for deterministic serialization.
        let mut sorted_vectors: Vec<_> = self.vectors.iter().collect();
        sorted_vectors.sort_by_key(|(doc_id, _, _)| *doc_id);

        // Per-segment field-name dictionary (Issue #633): ids assigned in
        // first-appearance order over the exact emission order below.
        let (field_dict, field_ids) =
            build_field_dict(sorted_vectors.iter().map(|(_, f, _)| f.as_str()))?;

        let f32_vectors: Vec<Vector> = sorted_vectors
            .iter()
            .map(|(_, _, v)| (*v).clone())
            .collect();

        // PQ min-train guard (Issue #880): PQ k-means trains 256 centroids
        // per sub-quantizer (16 for FastScan) — training on fewer vectors
        // than centroids produces a degenerate codebook with meaningless
        // recall (a 1-doc auto-commit segment would "train" k-means on one
        // point). Segments below the threshold are written as Scalar8Bit
        // instead; the LVS1 header is self-describing, so readers dispatch
        // on the stored quant kind regardless of the configured method, and
        // a merged (large) segment picks PQ back up automatically. Empty
        // segments keep the configured method's header for uniform
        // dispatch.
        //
        // This guard does not apply when a shared PQ codebook (Issue #631)
        // is configured: nothing is trained in that case, so the
        // degenerate-codebook rationale above does not hold, and small
        // segment-per-commit flushes can stay PQ instead of degrading to
        // Scalar8Bit. A configured-but-unresolved `pq_codebook_path` (set
        // in the schema, but not trained yet) must ALSO skip the guard —
        // otherwise a small flush would silently degrade to Scalar8Bit
        // here instead of reaching the loud "train one first" error in
        // the PQ arm below, hiding the missing-codebook misconfiguration
        // exactly when it is cheapest to surface (#918's failure policy:
        // no silent fallback).
        let has_shared_pq_codebook =
            self.index_config.pq_codebook.is_some() || self.index_config.pq_codebook_path.is_some();
        let effective_quantization = {
            use crate::vector::core::quantization::QuantizationMethod as Qm;
            let n = f32_vectors.len();
            match self.index_config.quantization_method {
                Qm::ProductQuantization { subvector_count }
                    if n > 0 && n < PQ_MIN_TRAIN_VECTORS && !has_shared_pq_codebook =>
                {
                    // The fallback must not skip config validation: an
                    // invalid PQ geometry (subvector_count not dividing the
                    // dimension) is rejected here exactly as the training
                    // path would, so acceptance never depends on corpus
                    // size.
                    crate::vector::core::quantization::PqParams::from_dim_and_m(
                        self.index_config.dimension,
                        subvector_count.max(1),
                    )?;
                    Qm::Scalar8Bit
                }
                #[cfg(feature = "pq-fastscan")]
                Qm::ProductQuantizationFastScan { subvector_count }
                    if n > 0 && n < PQ_FASTSCAN_MIN_TRAIN_VECTORS && !has_shared_pq_codebook =>
                {
                    // Same geometry validation as the PQ arm above; the
                    // shared-codebook exemption also mirrors it (Issue
                    // #920): with a shared codebook nothing is trained, so
                    // small flushes stay FastScan, and a configured-but-
                    // untrained path must reach the loud error in the
                    // FastScan arm below instead of degrading silently.
                    crate::vector::core::quantization::PqParams::from_dim_and_m(
                        self.index_config.dimension,
                        subvector_count.max(1),
                    )?;
                    Qm::Scalar8Bit
                }
                method => method,
            }
        };

        match effective_quantization {
            crate::vector::core::quantization::QuantizationMethod::Scalar8Bit => {
                // Empty segments fall back to neutral params (0.0, 1.0)
                // since there is nothing to train on; the LVS1 header
                // is still emitted so readers can dispatch on
                // quant_kind uniformly.
                let (params, records) = if f32_vectors.is_empty() {
                    (
                        crate::vector::core::quantization::ScalarQuantParams {
                            offset: 0.0,
                            scale: 1.0,
                        },
                        Vec::new(),
                    )
                } else {
                    quantize_segment(&f32_vectors, self.index_config.dimension)?
                };
                VectorSegmentHeader::scalar_8bit(params)
                    .with_version(VERSION_FIELD_DICT)
                    .with_field_dict(field_dict.clone())
                    .write_to(&mut output)?;
                for ((doc_id, field_name, _), (int8, meta)) in
                    sorted_vectors.iter().zip(records.iter())
                {
                    output.write_all(&doc_id.to_le_bytes())?;
                    output.write_all(&field_ids[field_name.as_str()].to_le_bytes())?;
                    write_quantized_record(&mut output, int8, *meta)?;
                }
            }
            crate::vector::core::quantization::QuantizationMethod::ProductQuantization {
                subvector_count,
            } => {
                if f32_vectors.is_empty() {
                    // An empty segment still needs a well-formed LVS1
                    // header so the reader can dispatch on quant_kind.
                    // We pick a minimal (m=1, sub_dim=dim) codebook of
                    // a single zero centroid per sub-vector — readers
                    // will never index into it because there are no
                    // codes after it.
                    let params = crate::vector::core::quantization::PqParams::from_dim_and_m(
                        self.index_config.dimension,
                        subvector_count.max(1),
                    )?;
                    let codebook = vec![0.0_f32; params.codebook_len()];
                    VectorSegmentHeader::product_quantization(params, codebook)
                        .with_version(VERSION_FIELD_DICT)
                        .with_field_dict(field_dict.clone())
                        .write_to(&mut output)?;
                } else {
                    // Issue #631: a configured-but-unresolved shared
                    // codebook (the path was set, but no file existed
                    // there when the index was opened) must fail loudly
                    // here rather than silently falling through to
                    // per-segment training below -- an invisible
                    // regression to the multi-second training cost would
                    // defeat the entire point of configuring a shared
                    // codebook. This is the ONLY place in the writer that
                    // can catch it: `resolve_pq_codebook` (index-open
                    // time) is deliberately lenient about a not-yet-
                    // trained codebook, since erroring there would make
                    // `create`/`open` fail for a schema whose codebook
                    // simply hasn't been trained yet.
                    if self.index_config.pq_codebook_path.is_some()
                        && self.index_config.pq_codebook.is_none()
                    {
                        return Err(LaurusError::InvalidOperation(format!(
                            "HNSW field configures pq_codebook_path = {:?} but no codebook \
                             has been trained there yet; train one first (e.g. `laurus train \
                             pq-codebook`) before committing, or clear pq_codebook_path to use \
                             per-segment training",
                            self.index_config.pq_codebook_path.as_deref().unwrap_or_default()
                        )));
                    }
                    let (params, codebook, codes) =
                        crate::vector::index::pq_io::quantize_segment_pq(
                            &f32_vectors,
                            self.index_config.dimension,
                            subvector_count,
                            self.index_config.pq_codebook.as_deref(),
                        )?;
                    VectorSegmentHeader::product_quantization(params, codebook)
                        .with_version(VERSION_FIELD_DICT)
                        .with_field_dict(field_dict.clone())
                        .write_to(&mut output)?;
                    for ((doc_id, field_name, _), codes_i) in
                        sorted_vectors.iter().zip(codes.iter())
                    {
                        output.write_all(&doc_id.to_le_bytes())?;
                        output.write_all(&field_ids[field_name.as_str()].to_le_bytes())?;
                        crate::vector::index::pq_io::write_pq_record(&mut output, codes_i)?;
                    }
                }
            }
            #[cfg(feature = "pq-fastscan")]
            crate::vector::core::quantization::QuantizationMethod::ProductQuantizationFastScan {
                subvector_count,
            } => {
                if f32_vectors.is_empty() {
                    // Empty segment: emit a well-formed LVS1 header with a
                    // minimal zero-centroid K=16 codebook so the reader can
                    // dispatch on quant_kind. Mirrors the PQ-256 empty path.
                    let m = subvector_count.max(1);
                    let sub_dim = self.index_config.dimension / m;
                    let params = crate::vector::core::quantization::PqParams::new(
                        m as u16,
                        16,
                        sub_dim as u16,
                    )?;
                    let codebook = vec![0.0_f32; params.codebook_len()];
                    VectorSegmentHeader::product_quantization_fastscan(params, codebook)
                        .with_version(VERSION_FIELD_DICT)
                        .with_field_dict(field_dict.clone())
                        .write_to(&mut output)?;
                } else {
                    // Issue #920: same failure policy as the standard PQ
                    // arm above — a configured-but-unresolved shared
                    // codebook must fail loudly here, not silently fall
                    // through to per-segment training.
                    if self.index_config.pq_codebook_path.is_some()
                        && self.index_config.pq_codebook.is_none()
                    {
                        return Err(LaurusError::InvalidOperation(format!(
                            "HNSW field configures pq_codebook_path = {:?} but no codebook \
                             has been trained there yet; train one first (e.g. `laurus train \
                             pq-codebook`) before committing, or clear pq_codebook_path to use \
                             per-segment training",
                            self.index_config.pq_codebook_path.as_deref().unwrap_or_default()
                        )));
                    }
                    let (params, codebook, codes) =
                        crate::vector::index::pq_fastscan_io::quantize_segment_pq_fastscan(
                            &f32_vectors,
                            self.index_config.dimension,
                            subvector_count,
                            self.index_config.pq_codebook.as_deref(),
                        )?;
                    VectorSegmentHeader::product_quantization_fastscan(params, codebook)
                        .with_version(VERSION_FIELD_DICT)
                        .with_field_dict(field_dict.clone())
                        .write_to(&mut output)?;
                    for ((doc_id, field_name, _), codes_i) in
                        sorted_vectors.iter().zip(codes.iter())
                    {
                        output.write_all(&doc_id.to_le_bytes())?;
                        output.write_all(&field_ids[field_name.as_str()].to_le_bytes())?;
                        crate::vector::index::pq_fastscan_io::write_pq_fastscan_record(
                            &mut output,
                            codes_i,
                        )?;
                    }
                }
            }
        }

        // Write Graph Data — v2 ordinal encoding (Issue #686). Neighbours
        // and the entry point are stored as segment-local u32 ordinals
        // (the rank of a doc id in the ascending unique record id
        // sequence); the per-node doc id is dropped because node order is
        // the same rank. The reader reconstructs doc ids from the record
        // section, which is written doc_id-sorted above.
        if let Some(graph) = &self.graph {
            output.write_all(&[1u8])?;

            // Ordinal table from the already-sorted record sequence.
            let mut unique_ids: Vec<u64> = Vec::with_capacity(sorted_vectors.len());
            for (doc_id, _, _) in &sorted_vectors {
                if unique_ids.last() != Some(doc_id) {
                    unique_ids.push(*doc_id);
                }
            }
            if unique_ids.len() >= u32::MAX as usize {
                return Err(LaurusError::InvalidOperation(format!(
                    "HNSW segment has {} unique doc ids — the v2 ordinal graph \
                     format supports at most u32::MAX - 1 nodes per segment",
                    unique_ids.len()
                )));
            }
            // Defensive: the graph's node set must be exactly the unique
            // record id set, otherwise the ordinal encoding would be
            // corrupt on disk. This never fires for writer-built graphs
            // (the graph is derived from `self.vectors`).
            if graph.node_count() != unique_ids.len() {
                return Err(LaurusError::InvalidOperation(format!(
                    "HNSW graph has {} nodes but the segment has {} unique \
                     record doc ids; refusing to write a corrupt v2 graph block",
                    graph.node_count(),
                    unique_ids.len()
                )));
            }
            let ord_of: ahash::AHashMap<u64, u32> = unique_ids
                .iter()
                .enumerate()
                .map(|(ord, &id)| (id, ord as u32))
                .collect();
            let ord_of_doc = |doc_id: u64| -> Result<u32> {
                ord_of.get(&doc_id).copied().ok_or_else(|| {
                    LaurusError::InvalidOperation(format!(
                        "HNSW graph references doc id {doc_id} that has no \
                         record in the segment; refusing to write a corrupt \
                         v2 graph block"
                    ))
                })
            };

            // Entry point as an ordinal; u32::MAX = None.
            let entry_ord = match graph.entry_point {
                Some(id) => ord_of_doc(id)?,
                None => u32::MAX,
            };
            output.write_all(&entry_ord.to_le_bytes())?;
            output.write_all(&(graph.max_level as u32).to_le_bytes())?;
            output.write_all(&(unique_ids.len() as u32).to_le_bytes())?;

            // Sort nodes by doc_id: node order == ordinal order.
            let sorted_nodes = graph.sorted_nodes();

            for (ord, (doc_id, layers)) in sorted_nodes.into_iter().enumerate() {
                if unique_ids[ord] != doc_id {
                    return Err(LaurusError::InvalidOperation(format!(
                        "HNSW graph node order diverges from the record order \
                         at ordinal {ord} (graph {doc_id}, records {}); \
                         refusing to write a corrupt v2 graph block",
                        unique_ids[ord]
                    )));
                }

                let layer_count = layers.len() as u32;
                output.write_all(&layer_count.to_le_bytes())?;

                for neighbors in layers {
                    let neighbor_count = neighbors.len() as u32;
                    output.write_all(&neighbor_count.to_le_bytes())?;
                    for &neighbor in neighbors {
                        output.write_all(&ord_of_doc(neighbor)?.to_le_bytes())?;
                    }
                }
            }
        } else {
            // No graph built
            output.write_all(&[0u8])?;
        }

        // Append the CRC-32 footer (magic + checksum over all preceding bytes)
        // so a corrupted segment is rejected on load (Issue #786).
        let content_crc = output.checksum();
        output.write_all(&crate::vector::index::hnsw::HNSW_FOOTER_MAGIC.to_le_bytes())?;
        output.write_all(&content_crc.to_le_bytes())?;
        // Close with an fsync BEFORE the rename (#882 review): the rename
        // publishes the segment (and, in the segmented layout, a manifest
        // whose WAL checkpoint covers it may follow) — a flush alone leaves
        // the content in the page cache, so a power loss could surface a
        // published-but-hollow segment file.
        let mut inner = output.into_inner();
        inner.close()?;
        storage.rename_file(&tmp_name, &file_name)?;

        // Stage 2 (Issue #481): emit the optional LRS1 rerank sidecar
        // alongside the main int8 segment. The sidecar's payload order
        // matches `sorted_vectors` (the same doc_id ordering used for
        // the LVS1 records above), which keeps a (sidecar position) ->
        // (LVS1 position) mapping at the identity. Sidecar is written
        // only when explicitly enabled per field; absence keeps Stage 1
        // (int8-only) behavior intact.
        if let Some(rerank_kind) = self.index_config.rerank_storage {
            // Same temp-then-rename atomicity for the rerank sidecar (#784).
            let sidecar_name = format!("{}.f32", file_name);
            let sidecar_tmp = format!("{}.f32.tmp", file_name);
            let mut sidecar_out = storage.create_output(&sidecar_tmp)?;
            let mut payload: Vec<f32> =
                Vec::with_capacity(sorted_vectors.len() * self.index_config.dimension);
            for (_, _, v) in &sorted_vectors {
                payload.extend_from_slice(&v.data);
            }
            write_sidecar(
                &mut sidecar_out,
                rerank_kind,
                self.index_config.dimension as u32,
                &payload,
            )?;
            sidecar_out.flush()?;
            drop(sidecar_out);
            storage.rename_file(&sidecar_tmp, &sidecar_name)?;
        }

        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    fn has_pending_changes(&self) -> bool {
        // `finalize()` sets the flag and every mutation (add_vectors,
        // delete_document/s, build) clears it, so a finalized writer's
        // in-memory state has already been captured by the finalize+write
        // pair and dropping it loses nothing. Note the load path constructs
        // writers with `is_finalized: false`, so a freshly loaded writer
        // conservatively reports pending changes.
        !self.is_finalized
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }

        // Logical deletion from buffer
        let initial_len = self.vectors.len();
        self.vectors.retain(|(id, _, _)| *id != doc_id);

        if self.vectors.len() < initial_len {
            self.rebuild_doc_id_map();
            // Invalidate the HNSW graph — it still contains edges
            // referencing the deleted doc_id.  The graph will be rebuilt
            // on the next finalize().
            self.graph = None;
        }
        Ok(())
    }

    fn delete_documents(&mut self, _field: &str, _value: &str) -> Result<usize> {
        if self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Cannot delete documents from finalized index".to_string(),
            ));
        }

        // Vectors no longer carry metadata; field-based deletion is not supported.
        // Use delete_document(doc_id) for document-level deletion.
        Ok(0)
    }

    fn rollback(&mut self) -> Result<()> {
        self.vectors.clear();
        self.doc_id_map.clear();
        self.graph = None;
        self.is_finalized = false;
        self.next_vec_id = 0;
        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        if self.is_finalized {
            0
        } else {
            self.vectors.len() as u64
        }
    }

    fn close(&mut self) -> Result<()> {
        self.vectors.clear();
        self.doc_id_map.clear();
        self.graph = None;
        self.is_finalized = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.is_finalized && self.vectors.is_empty()
    }

    fn build_reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        use crate::vector::index::hnsw::reader::HnswIndexReader;

        let storage = self.storage.as_ref().ok_or_else(|| {
            LaurusError::InvalidOperation("Cannot build reader: storage not configured".to_string())
        })?;

        let reader = HnswIndexReader::load(
            storage.clone(),
            &self.path,
            self.index_config.distance_metric,
        )?;

        Ok(Arc::new(reader))
    }
}

/// Unit tests for the `SearchLayerArena` thread-local reuse introduced by
/// Issue #632. `search_layer`/`SearchLayerArena`/`Candidate` are all private
/// to this module, so these tests live here rather than in the sibling
/// `hnsw::tests` module (`hnsw/tests.rs`, declared in `hnsw.rs`), which
/// cannot reach them. Broader HNSW correctness (recall, reachability,
/// incremental-append behaviour) is unaffected by this change and is
/// covered by the existing `hnsw::tests`, `hnsw_reachability_test`,
/// `hnsw_coldstart_recall_test`, and `vector_recall_test` suites — those
/// are the regression net for "search results are byte-for-byte the same
/// as before"; these tests specifically target bugs the arena-reuse
/// refactor itself could introduce.
#[cfg(test)]
mod search_layer_arena_tests {
    use super::*;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::index::HnswIndexConfig;
    use crate::vector::index::hnsw::graph::HnswGraph;

    /// A minimal, storage-less writer with `vectors`/`doc_id_map` populated
    /// directly (bypassing `build`/`add_vectors`, which need a `Storage`).
    fn make_writer(vectors: Vec<(u64, String, Vector)>) -> HnswIndexWriter {
        let config = HnswIndexConfig {
            dimension: vectors.first().map(|(_, _, v)| v.data.len()).unwrap_or(2),
            m: 4,
            ef_construction: 8,
            distance_metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let mut writer = HnswIndexWriter::new(config, VectorIndexWriterConfig::default(), "test")
            .expect("writer construction must succeed");
        writer.vectors = vectors;
        writer.rebuild_doc_id_map();
        writer
    }

    /// A single-level (level 0) line graph over `node_ids` in order:
    /// `node_ids[0] - node_ids[1] - node_ids[2] - ...`.
    fn line_graph(node_ids: &[u64]) -> HnswGraph {
        let mut nodes = HashMap::new();
        for (i, &id) in node_ids.iter().enumerate() {
            let mut neighbors = Vec::new();
            if i > 0 {
                neighbors.push(node_ids[i - 1]);
            }
            if i + 1 < node_ids.len() {
                neighbors.push(node_ids[i + 1]);
            }
            nodes.insert(id, vec![neighbors]);
        }
        HnswGraph::new(Some(node_ids[0]), 0, nodes, 4, 4, 8, 8, 1.0)
    }

    #[test]
    fn search_layer_reuses_arena_without_leaking_visited_state() -> Result<()> {
        let vectors = vec![
            (1, "t".to_string(), Vector::new(vec![0.0, 0.0])),
            (2, "t".to_string(), Vector::new(vec![1.0, 0.0])),
            (3, "t".to_string(), Vector::new(vec![2.0, 0.0])),
            (4, "t".to_string(), Vector::new(vec![3.0, 0.0])),
        ];
        let writer = make_writer(vectors);
        let graph = line_graph(&[1, 2, 3, 4]);

        let query_near_1 = Vector::new(vec![0.1, 0.0]);
        let result1 = writer.search_layer(&graph, 1, &query_near_1, 10, 0)?;
        assert_eq!(
            result1.len(),
            4,
            "first call must discover all 4 connected nodes"
        );

        // Same thread, second call: if the arena's visited bits from the
        // first call were not undone, this call would incorrectly treat
        // every node as already visited and find only the entry point.
        let query_near_4 = Vector::new(vec![2.9, 0.0]);
        let result2 = writer.search_layer(&graph, 1, &query_near_4, 10, 0)?;
        assert_eq!(
            result2.len(),
            4,
            "second call must not be polluted by the first call's visited state"
        );
        Ok(())
    }

    #[test]
    fn search_layer_grows_visited_buffer_when_vector_set_grows() -> Result<()> {
        let vectors = vec![
            (1, "t".to_string(), Vector::new(vec![0.0, 0.0])),
            (2, "t".to_string(), Vector::new(vec![1.0, 0.0])),
        ];
        let mut writer = make_writer(vectors);
        let small_graph = line_graph(&[1, 2]);
        let query = Vector::new(vec![0.5, 0.0]);
        let result = writer.search_layer(&small_graph, 1, &query, 10, 0)?;
        assert_eq!(result.len(), 2);

        // Grow the vector set as a real incremental build would, adding a
        // node whose dense index exceeds the arena's previously-sized
        // BitVec, then search a graph that includes it.
        writer
            .vectors
            .push((3, "t".to_string(), Vector::new(vec![2.0, 0.0])));
        writer.rebuild_doc_id_map();
        let bigger_graph = line_graph(&[1, 2, 3]);
        let result2 = writer.search_layer(&bigger_graph, 1, &query, 10, 0)?;
        assert_eq!(
            result2.len(),
            3,
            "the newly added node's dense index must be reachable after growth"
        );
        Ok(())
    }

    #[test]
    fn search_layer_errors_when_neighbor_missing_from_doc_id_map() {
        let vectors = vec![(1, "t".to_string(), Vector::new(vec![0.0, 0.0]))];
        let writer = make_writer(vectors);
        // Node 1's neighbor list references doc_id 999, which has no entry
        // in `vectors`/`doc_id_map` — this must error, not be silently
        // skipped, matching pre-refactor behaviour.
        let mut nodes = HashMap::new();
        nodes.insert(1u64, vec![vec![999u64]]);
        let graph = HnswGraph::new(Some(1), 0, nodes, 4, 4, 8, 8, 1.0);
        let query = Vector::new(vec![0.0, 0.0]);
        let err = writer
            .search_layer(&graph, 1, &query, 10, 0)
            .expect_err("a neighbor missing from doc_id_map must error");
        assert!(err.to_string().contains("999"), "{err}");
    }

    #[test]
    fn search_layer_arena_is_per_thread() {
        // Each spawned OS thread gets its own fresh `SearchLayerArena` via
        // `thread_local!`; this exercises that a brand-new (empty
        // `BitVec`/`touched`) arena resets and searches correctly, and that
        // two independently driven writers running concurrently on
        // different threads don't panic or interfere with each other.
        let handles: Vec<_> = (0..2)
            .map(|offset: u64| {
                std::thread::spawn(move || -> Result<usize> {
                    let base = offset * 10;
                    let vectors = vec![
                        (base + 1, "t".to_string(), Vector::new(vec![0.0, 0.0])),
                        (base + 2, "t".to_string(), Vector::new(vec![1.0, 0.0])),
                        (base + 3, "t".to_string(), Vector::new(vec![2.0, 0.0])),
                    ];
                    let writer = make_writer(vectors);
                    let graph = line_graph(&[base + 1, base + 2, base + 3]);
                    let query = Vector::new(vec![1.0, 0.0]);
                    let result = writer.search_layer(&graph, base + 1, &query, 10, 0)?;
                    Ok(result.len())
                })
            })
            .collect();
        for handle in handles {
            let len = handle
                .join()
                .expect("spawned thread must not panic")
                .expect("search_layer must succeed");
            assert_eq!(len, 3);
        }
    }

    // ── GraphView::copy_neighbors_into buffer reuse (Issue #1137) ───────────

    #[test]
    fn hnsw_graph_copy_neighbors_into_clears_buffer_between_calls() {
        let mut nodes = HashMap::new();
        nodes.insert(1u64, vec![vec![10u64, 20, 30]]);
        nodes.insert(2u64, vec![vec![99u64]]);
        let graph = HnswGraph::new(Some(1), 0, nodes, 4, 4, 8, 8, 1.0);

        let mut buf = Vec::new();
        graph.copy_neighbors_into(1, 0, &mut buf);
        assert_eq!(buf, vec![10, 20, 30]);

        // Reusing the same buffer for a node with FEWER neighbors must not
        // retain any of the previous call's entries.
        graph.copy_neighbors_into(2, 0, &mut buf);
        assert_eq!(
            buf,
            vec![99],
            "buffer must be cleared, not appended to, between calls"
        );
    }

    #[test]
    fn concurrent_hnsw_graph_copy_neighbors_into_clears_buffer_between_calls() {
        let graph = ConcurrentHnswGraph::new(vec![(1, 0), (2, 0)], 0);
        graph.set_neighbors(1, 0, vec![10, 20, 30]);
        graph.set_neighbors(2, 0, vec![99]);

        let mut buf = Vec::new();
        graph.copy_neighbors_into(1, 0, &mut buf);
        assert_eq!(buf, vec![10, 20, 30]);

        graph.copy_neighbors_into(2, 0, &mut buf);
        assert_eq!(
            buf,
            vec![99],
            "buffer must be cleared, not appended to, between calls"
        );
    }
}
