//! HNSW vector searcher for approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::distance::{DistanceMetric, PreparedQuery};
#[cfg(feature = "pq-fastscan")]
use crate::vector::core::distance_pq_fastscan::PqFastScanQuery;
use crate::vector::core::distance_quantized::{
    PqQuery, QuantizedQuery, distance_pq_adc, distance_quantized,
};
use crate::vector::core::vector::Vector;
use crate::vector::index::hnsw::graph::OrdinalHnswGraph;
use crate::vector::index::hnsw::reader::HnswIndexReader;
#[cfg(feature = "pq-fastscan")]
use crate::vector::index::pq_fastscan_avx2::distance_pq_fastscan_block;
#[cfg(feature = "pq-fastscan")]
use crate::vector::index::pq_fastscan_storage::{BLOCK_SIZE, PqFastScanPool};
use crate::vector::index::pq_storage::PqVectorPool;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResult, VectorIndexQueryResults,
};
use bit_vec::BitVec;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Upper bound on visited nodes for a filter-aware traversal, expressed as
/// a multiple of `max(ef_search, top_k)` (Issue #645).
///
/// A filtered search keeps only matching documents in its result heap, so
/// when the filter is selective the result heap fills slowly and the search
/// would otherwise walk most of the graph chasing matches. This cap bounds
/// that worst case, trading recall for a latency ceiling; the unfiltered
/// path is unaffected (it uses no cap). The chosen factor of `16` lets a
/// default `ef_search = 50` visit ~800 nodes before giving up.
const MAX_VISIT_FACTOR: usize = 16;

/// Per-search state for the quantized hot path (Issue #481 Stage 1 +
/// Stage 3).
///
/// Built once at the start of `search_graph` when the reader exposes
/// either a [`QuantizedVectorPool`] (Stage 1, int8) or a
/// [`PqVectorPool`] (Stage 3, PQ codes + codebook). Threaded through
/// `calc_dist` so each per-candidate call is one O(1)
/// `field_idx.get` plus a quantized distance kernel call — no per-call
/// allocation, no `String` clone.
enum QuantizedSearchCtx {
    /// Stage 1: int8 hot path.
    Scalar8Bit {
        /// Quantized query (int8 + cached norm + offset/scale),
        /// prepared once per search via [`QuantizedQuery::prepare`].
        prepared: QuantizedQuery,
        /// The reader's in-memory int8 storage. Cloned `Arc` so the
        /// pool stays alive even if the reader is dropped mid-search
        /// (it isn't, but the borrow checker needs the lifetime
        /// extension).
        pool: Arc<QuantizedVectorPool>,
        /// Per-field doc_id -> vector position in `pool.data`. Cached
        /// so the hot loop is a HashMap probe, not a per-field-name
        /// indirection.
        field_idx: Arc<HashMap<u64, u32>>,
        /// Distance metric, cached so the hot loop skips the
        /// `reader.distance_metric()` indirection.
        metric: DistanceMetric,
    },
    /// Stage 3: PQ ADC hot path.
    Pq {
        /// Per-query LUT (M × K floats) prepared once per search via
        /// [`PqQuery::prepare`].
        prepared: PqQuery,
        /// The reader's in-memory PQ pool (codes + codebook + index).
        pool: Arc<PqVectorPool>,
        /// Per-field doc_id -> vector position in `pool.data`.
        field_idx: Arc<HashMap<u64, u32>>,
        /// Distance metric, cached for the hot loop.
        metric: DistanceMetric,
    },
    /// PQ FastScan hot path (Issue #695 / #702, experimental).
    ///
    /// The kernel computes 32 distances per call via
    /// [`distance_pq_fastscan_block`] (AVX2 / NEON / scalar dispatch),
    /// so `distance()` evaluates one block and returns the in-block
    /// offset. For dense HNSW search this wastes 31/32 of the block
    /// computation, but it keeps the per-doc interface used by the
    /// graph traversal — fully batched block evaluation (one block
    /// per HNSW hop's neighbour list) is a future optimisation.
    #[cfg(feature = "pq-fastscan")]
    PqFastScan {
        /// Per-query state with the FastScan u8 / f32 LUTs prepared
        /// once via [`PqFastScanQuery::prepare`].
        prepared: PqFastScanQuery,
        /// The reader's in-memory FastScan pool (block-transposed
        /// 4-bit codes + K=16 codebook + per-field doc-id index).
        pool: Arc<PqFastScanPool>,
        /// Per-field doc_id -> vector position in `pool.packed`
        /// (block-transposed, so `pos / BLOCK_SIZE` is the block and
        /// `pos % BLOCK_SIZE` is the in-block offset).
        field_idx: Arc<HashMap<u64, u32>>,
        /// Distance metric, cached for the hot loop.
        metric: DistanceMetric,
    },
}

impl QuantizedSearchCtx {
    /// Compute distance from the prepared query to the candidate at
    /// pool position `pos` (Issue #686 ordinal hot path — no hash
    /// probe; the caller has already translated ordinal → position).
    ///
    /// # Arguments
    ///
    /// * `pos` - The candidate's pool position (`< vector_count`,
    ///   guaranteed by the reader's load-time validation).
    ///
    /// # Returns
    ///
    /// The quantized distance.
    #[inline]
    fn distance_at(&self, pos: u32) -> f32 {
        match self {
            Self::Scalar8Bit {
                prepared,
                pool,
                metric,
                ..
            } => {
                let (int8, meta) = pool.record_at(pos);
                distance_quantized(*metric, prepared, int8, meta)
            }
            Self::Pq {
                prepared,
                pool,
                metric,
                ..
            } => {
                let codes = pool.codes_at(pos);
                distance_pq_adc(*metric, prepared, codes)
            }
            #[cfg(feature = "pq-fastscan")]
            Self::PqFastScan {
                prepared,
                pool,
                metric,
                ..
            } => {
                let pos = pos as usize;
                let block_idx = pos / BLOCK_SIZE;
                let in_block = pos % BLOCK_SIZE;
                let stride = pool.block_stride();
                let block_base = block_idx * stride;
                let packed_block = &pool.packed[block_base..block_base + stride];
                let distances = distance_pq_fastscan_block(*metric, prepared, packed_block);
                distances[in_block]
            }
        }
    }

    /// Compute distance from the prepared query to the candidate at
    /// segment ordinal `ord` (Issue #686).
    ///
    /// # Arguments
    ///
    /// * `ord` - The candidate's segment ordinal (`< node_count`).
    /// * `ord_to_pos` - Ordinal → pool-position table, `None` for the
    ///   identity mapping (single-field segments — the common case).
    ///
    /// # Returns
    ///
    /// The quantized distance, or `f32::MAX` when the table marks the
    /// doc absent from the searched field (#676 semantics).
    #[inline]
    fn distance_ord(&self, ord: u32, ord_to_pos: Option<&[u32]>) -> f32 {
        let pos = match ord_to_pos {
            None => ord,
            Some(table) => {
                let pos = table[ord as usize];
                if pos == u32::MAX {
                    return f32::MAX;
                }
                pos
            }
        };
        self.distance_at(pos)
    }

    /// Compute distance from the prepared query to the candidate at
    /// `doc_id`. Returns `f32::MAX` if the candidate is missing, which
    /// is what HNSW's calc_dist expects for absent neighbours.
    ///
    /// Cold-path variant (the #738 brute-force mode and the entry-point
    /// probe): pays one `field_idx` hash probe per call. The graph
    /// traversal itself uses [`Self::distance_ord`].
    #[inline]
    fn distance_doc(&self, doc_id: u64) -> f32 {
        let field_idx = match self {
            Self::Scalar8Bit { field_idx, .. } => field_idx,
            Self::Pq { field_idx, .. } => field_idx,
            #[cfg(feature = "pq-fastscan")]
            Self::PqFastScan { field_idx, .. } => field_idx,
        };
        match field_idx.get(&doc_id) {
            Some(&pos) => self.distance_at(pos),
            None => f32::MAX,
        }
    }

    /// Base address and record stride for direct-address software
    /// prefetch (Issue #686), when this ctx's pool layout supports it.
    ///
    /// Only the int8 SQ pool benefits: PQ records are 8–32 bytes (the
    /// LUT is the important cache occupant) and FastScan streams whole
    /// blocks sequentially, so both return `None` — matching the
    /// pre-#686 behaviour where their prefetch maps were never built.
    ///
    /// # Returns
    ///
    /// `Some((base_address, stride_bytes))` for the SQ pool, else `None`.
    fn prefetch_base_stride(&self) -> Option<(usize, usize)> {
        match self {
            Self::Scalar8Bit { pool, .. } => Some((pool.int8_data.as_ptr() as usize, pool.pad_dim)),
            _ => None,
        }
    }
}

/// Fallback `ef_search` used when neither the per-query
/// [`VectorIndexQueryParams::ef_search`] nor the schema-level
/// [`crate::vector::core::field::HnswOption::default_ef_search`] is set.
///
/// Issue [#644](https://github.com/mosuka/laurus/issues/644).
pub(crate) const HNSW_DEFAULT_EF_SEARCH: usize = 50;

/// HNSW vector searcher that performs approximate nearest neighbor search.
///
/// The searcher's `default_ef_search` field holds the schema-level
/// fallback for the `ef_search` parameter. Per-query callers can override
/// it via [`VectorIndexQueryParams::ef_search`]; the effective value used
/// by the graph traversal is computed by [`Self::effective_ef`] for each
/// search request.
#[derive(Debug)]
pub struct HnswSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    default_ef_search: usize,
}

impl HnswSearcher {
    /// Create a new HNSW searcher with the built-in fallback `ef_search`
    /// of [`HNSW_DEFAULT_EF_SEARCH`].
    ///
    /// For schemas that opt into a higher schema-level default, use
    /// [`Self::with_default_ef_search`] instead. Per-query overrides are
    /// honoured regardless of how the searcher was constructed.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        Ok(Self {
            index_reader,
            default_ef_search: HNSW_DEFAULT_EF_SEARCH,
        })
    }

    /// Create a new HNSW searcher with an explicit schema-level
    /// `default_ef_search`. Pass `None` to use the built-in fallback
    /// ([`HNSW_DEFAULT_EF_SEARCH`]).
    ///
    /// Issue [#644](https://github.com/mosuka/laurus/issues/644).
    pub fn with_default_ef_search(
        index_reader: Arc<dyn VectorIndexReader>,
        default_ef_search: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            index_reader,
            default_ef_search: default_ef_search.unwrap_or(HNSW_DEFAULT_EF_SEARCH),
        })
    }

    /// Override the schema-level default `ef_search`. Equivalent to
    /// constructing the searcher with [`Self::with_default_ef_search`]
    /// after the fact.
    ///
    /// Per-query [`VectorIndexQueryParams::ef_search`] overrides this
    /// value at search time.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.default_ef_search = ef_search;
    }

    /// Compute the `ef_search` used for a specific request.
    ///
    /// Precedence:
    /// 1. Per-query [`VectorIndexQueryParams::ef_search`] (highest)
    /// 2. The searcher's schema-level `default_ef_search`
    /// 3. The built-in fallback [`HNSW_DEFAULT_EF_SEARCH`] (= `50`)
    ///
    /// In all cases the result is lifted to at least
    /// `max(top_k, top_k * rerank_factor.unwrap_or(1))` so the
    /// candidate heap is never undersized for the requested `top_k`
    /// (Issue [#644](https://github.com/mosuka/laurus/issues/644)).
    #[inline]
    fn effective_ef(&self, request: &VectorIndexQuery) -> usize {
        let params = &request.params;
        let user_ef = params.ef_search.unwrap_or(self.default_ef_search);
        let rerank = params.rerank_factor.unwrap_or(1).max(1);
        user_ef
            .max(params.top_k.saturating_mul(rerank))
            .max(params.top_k)
    }
}

impl VectorIndexSearcher for HnswSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        use crate::util::time::Timer;

        // Stage 2 (Issue #481): rerank_factor is honored on the HNSW
        // graph path when the reader has a rerank storage pool loaded
        // (`reader.rerank_storage().is_some()`). Otherwise the value
        // is silently ignored: there is no f32 information to recover
        // for Stage 1 segments or for the brute-force fallback below
        // (which already runs against the dequantized f32 vectors).

        let start = Timer::now();

        // correct approach: usage of downcast_ref to check if we can use graph search
        if let Some(reader) = self.index_reader.as_any().downcast_ref::<HnswIndexReader>()
            && let Some(graph) = &reader.graph
            && let Some(ref field_name) = request.field_name
        {
            // Perform Graph Search
            let mut results = self.search_graph(reader, graph, request, field_name)?;
            results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(results);
        }

        // Fallback to Linear Scan (brute-force over all vectors)
        let mut results = VectorIndexQueryResults::new();
        let metric = self.index_reader.distance_metric();
        // Cache the query-side norm once per search (#414): for Cosine /
        // Angular this skips one `||query||²` accumulation per
        // candidate; other metrics fall back to the unprepared path.
        let prepared_query = metric.prepare_query(&request.query.data);

        if let Some(ref field_name) = request.field_name {
            // Field-filtered path: fetch the per-field doc-id slice from the
            // reader's pre-built index (#405 — O(1) Arc clone, avoids the full
            // `Vec<(u64, String)>` clone and the linear retain scan). Every
            // candidate shares the same `field_name`, so it is not stored
            // per-candidate; it is cloned only when emitting the top_k
            // results.
            let ids = self.index_reader.doc_ids_for_field(field_name);
            results.candidates_examined = ids.len();

            let mut candidates: Vec<(u64, f32, f32, Vector)> = Vec::with_capacity(ids.len());
            for &doc_id in ids.iter() {
                if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, field_name) {
                    // Compute distance once and derive similarity from it.
                    // `similarity()` would otherwise re-run the SIMD distance
                    // kernel a second time on the same pair.
                    let distance = metric.distance_with_prepared(&prepared_query, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    candidates.push((doc_id, similarity, distance, vector));
                }
            }

            // Sort ascending by distance with a doc-id tiebreak (#933):
            // similarity's `exp(-d)` underflows to 0.0 at long range,
            // collapsing distant candidates into ties whose unstable order
            // would make top-k membership arbitrary; distance stays precise.
            candidates.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));

            let top_k = request.params.top_k.min(candidates.len());
            for (doc_id, similarity, distance, vector) in candidates.into_iter().take(top_k) {
                if similarity < request.params.min_similarity {
                    break;
                }

                let vector_output = if request.params.include_vectors {
                    Some(vector)
                } else {
                    None
                };

                results
                    .results
                    .push(crate::vector::search::searcher::VectorIndexQueryResult {
                        doc_id,
                        field_name: field_name.clone(),
                        similarity,
                        distance,
                        vector: vector_output,
                    });
            }
        } else {
            // Unfiltered path: docs may belong to different fields, so the
            // field name must travel with each candidate.
            let candidates_list = self.index_reader.vector_ids()?;
            results.candidates_examined = candidates_list.len();

            let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
                Vec::with_capacity(candidates_list.len());
            for (doc_id, field_name) in candidates_list.iter() {
                if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name) {
                    let distance = metric.distance_with_prepared(&prepared_query, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    candidates.push((*doc_id, field_name.clone(), similarity, distance, vector));
                }
            }

            candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

            let top_k = request.params.top_k.min(candidates.len());
            for (doc_id, field_name, similarity, distance, vector) in
                candidates.into_iter().take(top_k)
            {
                if similarity < request.params.min_similarity {
                    break;
                }

                let vector_output = if request.params.include_vectors {
                    Some(vector)
                } else {
                    None
                };

                results
                    .results
                    .push(crate::vector::search::searcher::VectorIndexQueryResult {
                        doc_id,
                        field_name,
                        similarity,
                        distance,
                        vector: vector_output,
                    });
            }
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(results)
    }

    fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        // Field-filtered counts use the pre-built per-field index (#405);
        // avoids allocating + linear-filtering the full `vector_ids`.
        if let Some(ref field_name) = request.field_name {
            Ok(self.index_reader.doc_ids_for_field(field_name).len() as u64)
        } else {
            // Issue #672: `vector_ids()` materializes a String per record
            // just to be counted; `vector_count()` is the same number (one
            // entry per (doc, field) record) with no allocation.
            Ok(self.index_reader.vector_count() as u64)
        }
    }

    /// Pre-fault the on-disk vector data into the OS page cache (Issue #677).
    ///
    /// Only the [`OnDemand`](crate::vector::index::storage::VectorStorage::OnDemand)
    /// (`Mmap` / lazy) storage benefits: without warming, the first query pays
    /// a page fault for every candidate vector it reads. Touching each stored
    /// vector once moves that cost to startup. The `Owned*` variants are
    /// already heap-resident after the reader load that
    /// [`VectorStore::warmup`](crate::vector::VectorStore::warmup) forces, so
    /// this is a no-op for them. The HNSW graph is always loaded into memory
    /// eagerly, so only the vector data needs warming.
    ///
    /// Individual read failures are skipped rather than aborting startup —
    /// warming is a best-effort optimisation, and a genuinely unreadable vector
    /// would surface on the real query regardless.
    fn warmup(&mut self) -> Result<()> {
        let Some(reader) = self.index_reader.as_any().downcast_ref::<HnswIndexReader>() else {
            return Ok(());
        };
        if !matches!(
            reader.vectors(),
            crate::vector::index::storage::VectorStorage::OnDemand { .. }
        ) {
            return Ok(());
        }
        // Read every stored vector so its backing page is faulted in. The
        // accumulator (kept live via `black_box`) stops the loop from being
        // optimised away as dead code. The interned iterator (#672) avoids
        // materializing one `String` per record just to name the field.
        let mut acc = 0u64;
        for (doc_id, field) in reader.interned_vector_ids() {
            if let Ok(Some(vector)) = reader.get_vector(doc_id, field)
                && let Some(first) = vector.data.first()
            {
                acc = acc.wrapping_add(first.to_bits() as u64);
            }
        }
        std::hint::black_box(acc);
        Ok(())
    }
}

/// Frontier heap entry for graph traversal (Issue #686): carries the
/// segment-local u32 ordinal, packing the entry into 8 bytes (vs 16 for
/// the former `{u64, f32}` shape) — half the heap traffic per push/pop.
#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    ord: u32,
    distance: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance > larger distance for Visitor (nearest first)
        // But for Result (Found), we might want Max-heap (furthest first) to keep ef smallest.
        // HNSW logic typically uses Min-heap for "candidates to visit" and Max-heap for "dynamic list of found nearest"
        // Here we define one Candidate struct. Let's assume standard PartialOrd (smaller < larger).
        // Then BinaryHeap is MaxHeap (largest at top).

        // This impl makes BinaryHeap a MIN-HEAP (smallest distance at top)
        other.distance.total_cmp(&self.distance)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ResultCandidate {
    id: u64,
    distance: f32,
}

impl Eq for ResultCandidate {}
impl Ord for ResultCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance at top (to remove worst)
        self.distance.total_cmp(&other.distance)
    }
}
impl PartialOrd for ResultCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswSearcher {
    fn search_graph(
        &self,
        reader: &HnswIndexReader,
        graph: &OrdinalHnswGraph,
        request: &VectorIndexQuery,
        field_name: &str,
    ) -> Result<VectorIndexQueryResults> {
        // Ordinal traversal (Issue #686): the graph is addressed by
        // segment-local u32 ordinals throughout; doc ids materialise only
        // at admission/emission via the `graph.doc_id(ord)` array read.
        let entry_ord = match graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(VectorIndexQueryResults::new()),
        };

        let query = &request.query;
        let ef_search = self.effective_ef(request);

        // Prepare the quantized hot path according to the segment's
        // storage kind:
        // * Stage 1 (`OwnedQuantized`): int8 SIMD via
        //   [`distance_quantized`].
        // * Stage 3 (`OwnedPq`, Issue #481 PQ): ADC LUT via
        //   [`distance_pq_adc`].
        // Other storage kinds (`OnDemand`, `Owned`) fall through to
        // the f32 reference path in `calc_dist`.
        let metric = reader.distance_metric();
        let quant_ctx: Option<QuantizedSearchCtx> =
            if let Some(pool) = reader.vectors().quantized_pool() {
                pool.field_position_index(field_name).map(|field_idx| {
                    let prepared = QuantizedQuery::prepare(&query.data, &pool.params);
                    QuantizedSearchCtx::Scalar8Bit {
                        prepared,
                        pool: pool.clone(),
                        field_idx,
                        metric,
                    }
                })
            } else if let Some(pool) = reader.vectors().pq_pool() {
                pool.field_position_index(field_name).map(|field_idx| {
                    let prepared = PqQuery::prepare(&query.data, pool.params, &pool.codebook);
                    QuantizedSearchCtx::Pq {
                        prepared,
                        pool: pool.clone(),
                        field_idx,
                        metric,
                    }
                })
            } else {
                #[cfg(feature = "pq-fastscan")]
                {
                    if let Some(pool) = reader.vectors().pq_fastscan_pool() {
                        let field_idx = pool.field_position_index(field_name);
                        let prepared =
                            PqFastScanQuery::prepare(&query.data, pool.params, &pool.codebook)?;
                        field_idx.map(|field_idx| QuantizedSearchCtx::PqFastScan {
                            prepared,
                            pool: pool.clone(),
                            field_idx,
                            metric,
                        })
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "pq-fastscan"))]
                {
                    None
                }
            };

        // f32 reference path (no quantized context): cache the query norm once
        // so `calc_dist` uses `distance_with_prepared` (candidate-norm-only
        // kernel) instead of recomputing `‖query‖²` per candidate (#835). The
        // quantized contexts already carry their own prepared query.
        let prepared_query = quant_ctx
            .is_none()
            .then(|| metric.prepare_query(&query.data));

        // Ordinal → pool-position table (Issue #686): `None` means the
        // identity holds (single-field segment — every current writer),
        // so the ordinal doubles as the pool position and the hot loop
        // pays no translation at all.
        let ord_to_pos = reader.field_ord_to_pos(field_name);

        // Prefetch setup. The int8 SQ hot path computes each neighbour's
        // record address directly as `base + pos * stride` (Issue #686 —
        // no doc_id-keyed map probe); the legacy f32 `Owned` storage
        // keeps its doc_id-keyed address map. PQ/FastScan never prefetch
        // (records are tiny / block-streamed respectively).
        let sq_prefetch = quant_ctx
            .as_ref()
            .and_then(QuantizedSearchCtx::prefetch_base_stride);
        let field_prefetch = if quant_ctx.is_none() {
            reader.field_prefetch_index(field_name)
        } else {
            None
        };
        let prefetch_n_bytes = match &sq_prefetch {
            // The padded int8 record; its meta lives in separate SoA
            // arrays, so the int8 stride alone is what the loop streams.
            Some((_, stride)) => *stride,
            None => reader.dimension() * std::mem::size_of::<f32>(),
        };

        // One prefetch hint per unvisited neighbour ordinal. Kept as a
        // plain stack closure so both traversal branches share one body
        // without perturbing their codegen (#645 discipline).
        // Whether any prefetch source exists at all. Hoisted so the
        // no-prefetch storages (OnDemand / PQ / FastScan) skip the
        // pass-1 loops entirely, exactly like the pre-#686 structure
        // (`if let Some(idx) = field_prefetch` around the loop).
        let prefetch_enabled = sq_prefetch.is_some() || field_prefetch.is_some();
        let prefetch_ord = |ord: u32| {
            if let Some((base, stride)) = sq_prefetch {
                let pos = match ord_to_pos.as_deref() {
                    None => ord,
                    Some(table) => table[ord as usize],
                };
                if pos != u32::MAX {
                    // SAFETY (address computation only): `base` was taken
                    // from the SQ pool's `int8_data`, which the ctx's Arc
                    // keeps alive for this search; `pos` is a validated
                    // pool position, so the address stays in-bounds.
                    // Prefetch is a pure hint and never dereferences.
                    Self::prefetch_addr(base + pos as usize * stride, prefetch_n_bytes);
                }
            } else if let Some(idx) = field_prefetch {
                Self::prefetch_neighbor(idx, graph.doc_id(ord), prefetch_n_bytes);
            }
        };

        // Issue #656: speculative two-hop lookahead. Peeks the frontier
        // heap's current top -- state as of the end of the *previous*
        // iteration, since this iteration hasn't pushed anything yet -- and
        // prefetches *its* neighbours' vector data, overlapping that fetch
        // with the current iteration's own neighbour processing below.
        //
        // `candidates`/`visited` are taken as explicit parameters rather
        // than captured: both are mutated elsewhere in the same loop
        // (`candidates.push`, `visited.set`), and a closure holding `&`
        // references to them for its whole lifetime would conflict with
        // those later `&mut` accesses -- exactly why `prefetch_ord` above
        // takes `ord` as a parameter instead of capturing "the current
        // neighbour".
        //
        // This can never change search results: it only issues hardware
        // prefetch hints (no dereference, no observable side effect on
        // `candidates`/`found`/`visited`), and the popped candidate's own
        // Pass 1 below still runs unconditionally as the ground-truth
        // prefetch for whatever actually gets processed next iteration. A
        // wrong guess here (the heap's top can change once this iteration's
        // own discoveries are pushed) just touches cache lines that may or
        // may not be used later -- never a correctness concern.
        let prefetch_lookahead = |candidates: &BinaryHeap<Candidate>, visited: &BitVec| {
            if let Some(next) = candidates.peek()
                && let Some(next_neighbors) = graph.neighbors(next.ord, 0)
            {
                for &neighbor_ord in next_neighbors {
                    if !visited.get(neighbor_ord as usize).unwrap_or(false) {
                        prefetch_ord(neighbor_ord);
                    }
                }
            }
        };

        // Cardinality-driven mode (Issue #738): when the filter is selective
        // enough that fewer documents are allowed than the candidate-list size
        // (`ef_search`), scoring those documents directly is both cheaper and
        // exact — it touches exactly `cardinality` documents, never more than
        // the graph walk's `ef_search`, and computes the true distance to
        // every match (no approximation). The graph walk's job is to *find*
        // near neighbours among many; when the allow-set is already tiny there
        // is nothing to find.
        if let Some(filter) = request.filter.as_deref()
            && filter.len() <= ef_search as u64
        {
            let mut found = BinaryHeap::new();
            for doc_id in filter.iter() {
                // Skip logically deleted docs (Issue #665): the brute path
                // would otherwise score and return them, since nothing
                // downstream re-checks deletion.
                if reader.is_deleted(doc_id) {
                    continue;
                }
                let d = self.calc_dist(
                    reader,
                    query,
                    quant_ctx.as_ref(),
                    prepared_query.as_ref(),
                    doc_id,
                    field_name,
                )?;
                // Skip docs with no vector in this field (Issue #676);
                // `finalize_graph_results` also guards this, but skipping here
                // keeps the heap small.
                if d == f32::MAX {
                    continue;
                }
                found.push(ResultCandidate {
                    id: doc_id,
                    distance: d,
                });
            }
            return self.finalize_graph_results(
                reader,
                query,
                request,
                field_name,
                found,
                filter.len() as usize,
            );
        }

        // 1. Start from entry point at max_level. The entry is assumed
        // to belong to `field_name` (HnswIndex is single-field); a
        // missing field yields `f32::MAX` and the descent degrades
        // gracefully (#676 semantics).
        let mut curr_ord = entry_ord;
        let mut dist = self.calc_dist_ord(
            reader,
            query,
            quant_ctx.as_ref(),
            prepared_query.as_ref(),
            ord_to_pos.as_deref(),
            graph,
            curr_ord,
            field_name,
        )?;

        // 2. Greedy descent
        for lc in (1..=graph.max_level()).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = graph.neighbors(curr_ord, lc) {
                    // Pass 1: issue prefetch hints for all neighbors before computing
                    // distances.  For datasets larger than L3 cache this hides the
                    // memory latency of loading the candidate records.
                    if prefetch_enabled {
                        for &neighbor_ord in neighbors {
                            prefetch_ord(neighbor_ord);
                        }
                    }
                    // Pass 2: compute distances (data is being fetched in the background).
                    for &neighbor_ord in neighbors {
                        let d = self.calc_dist_ord(
                            reader,
                            query,
                            quant_ctx.as_ref(),
                            prepared_query.as_ref(),
                            ord_to_pos.as_deref(),
                            graph,
                            neighbor_ord,
                            field_name,
                        )?;
                        if d < dist {
                            dist = d;
                            curr_ord = neighbor_ord;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 3. Search at layer 0 with ef_search
        // Issue #680: pre-size both heaps instead of growing geometrically
        // from empty on every query. `found` never holds more than
        // `ef_search + 1` entries (pushed then immediately popped back down
        // once it overflows, below); `candidates` has no hard cap, but stays
        // in the same order of magnitude in practice, so the same estimate
        // is used for both.
        let mut candidates = BinaryHeap::with_capacity(ef_search * 2); // Min-heap (nearest first)
        let mut found = BinaryHeap::with_capacity(ef_search * 2); // Max-heap (furthest first)

        // A node enters the result heap only if it satisfies the admission
        // predicate; the frontier (`candidates`) always expands through every
        // node to preserve connectivity. Bookkeeping (the allow-set probe and
        // the per-neighbour deletion check) is needed when a filter is present
        // (Issue #645) OR the reader has deletions (Issue #665). When neither
        // holds, the pristine `else` branch below runs unchanged. `check_deletions`
        // is hoisted so the filter-only path never pays for `is_deleted` calls.
        let check_deletions = reader.has_deletions();
        let needs_bookkeeping = request.filter.is_some() || check_deletions;

        candidates.push(Candidate {
            ord: curr_ord,
            distance: dist,
        });
        // Seed the result heap with the entry only if it is admissible. The
        // pre-#665 code admitted the entry unconditionally, which let a deleted
        // (or, under #645, filter-rejected) entry leak into results.
        let curr_doc = graph.doc_id(curr_ord);
        let entry_admitted = !needs_bookkeeping
            || (request
                .filter
                .as_deref()
                .is_none_or(|f| f.contains(curr_doc))
                && !(check_deletions && reader.is_deleted(curr_doc)));
        if entry_admitted {
            found.push(ResultCandidate {
                id: curr_doc,
                distance: dist,
            });
        }

        // Visited set as a dense bitmap indexed by segment ordinal
        // (Issue #686): exactly `node_count` bits, independent of the
        // global doc-id space — a long-lived store whose ids have grown
        // far past this segment's node count no longer pays a
        // proportionally inflated allocation + zeroing per query (the
        // #647 premise). `BitVec::get` / `BitVec::set` are a single
        // array index + bit op, materially cheaper than `HashSet<u64>`'s
        // hash + bucket lookup that the audit (#406) flagged for
        // ef_search graph traversals.
        let mut visited = BitVec::from_elem(graph.node_count(), false);
        visited.set(curr_ord as usize, true);

        // Bookkeeping traversal (Issues #645 and #665). The result heap
        // (`found`) admits a node only if it passes the admission predicate —
        // it matches the filter (if any) AND is not logically deleted — while
        // the frontier (`candidates`) still expands through every neighbour so
        // the search can cross rejected regions to reach admissible clusters.
        // A plain post-filter cannot (its slots are already spent), which is
        // why a selective filter or a high deletion ratio could otherwise
        // return far fewer hits than exist (or none). `max_visits` bounds the
        // worst case where admissible nodes are rare.
        //
        // The two paths are split deliberately: the pristine `else` branch is
        // byte-for-byte the pre-#645 loop, so this bookkeeping cannot change
        // the codegen (and thus the latency) of the common search that has
        // neither a filter nor deletions — the dominant production case. The
        // bookkeeping branch carries the extra per-neighbour work (`n_visited`,
        // the allow-set probe, the deletion check) that the pristine path must
        // not pay for.
        if needs_bookkeeping {
            let filter = request.filter.as_deref();
            // Deliberately NOT clamped to the node count: `n_visited` rises at
            // most once per node (the `visited` guard), so when
            // `ef_search * MAX_VISIT_FACTOR >= N` the cap is simply never hit
            // and the traversal runs to completion. Clamping to `N` instead
            // made the cap fire exactly as the last node was visited, dropping
            // whichever match happened to sit last in the (graph-shape- and
            // platform-dependent) traversal order.
            let max_visits = ef_search
                .max(request.params.top_k)
                .saturating_mul(MAX_VISIT_FACTOR);
            let mut n_visited = 1usize; // entry point already marked visited

            while let Some(curr) = candidates.pop() {
                if let Some(furthest) = found.peek()
                    && curr.distance > furthest.distance
                    && found.len() >= ef_search
                {
                    break;
                }
                if n_visited >= max_visits {
                    break;
                }

                if prefetch_enabled {
                    prefetch_lookahead(&candidates, &visited);
                }

                if let Some(neighbors) = graph.neighbors(curr.ord, 0) {
                    if prefetch_enabled {
                        for &neighbor_ord in neighbors {
                            if !visited.get(neighbor_ord as usize).unwrap_or(false) {
                                prefetch_ord(neighbor_ord);
                            }
                        }
                    }

                    for &neighbor_ord in neighbors {
                        let nbr_idx = neighbor_ord as usize;
                        if visited.get(nbr_idx).unwrap_or(false) {
                            continue;
                        }
                        visited.set(nbr_idx, true);
                        n_visited += 1;

                        let d = self.calc_dist_ord(
                            reader,
                            query,
                            quant_ctx.as_ref(),
                            prepared_query.as_ref(),
                            ord_to_pos.as_deref(),
                            graph,
                            neighbor_ord,
                            field_name,
                        )?;
                        let furthest_dist = found.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                        if d < furthest_dist || found.len() < ef_search {
                            // Frontier expands through every neighbour, even
                            // ones the filter rejects or that are deleted, to
                            // preserve connectivity.
                            candidates.push(Candidate {
                                ord: neighbor_ord,
                                distance: d,
                            });
                            // Result heap keeps only admissible docs: matching
                            // the filter (if any) AND not deleted (Issue #665).
                            // `check_deletions` short-circuits the `is_deleted`
                            // call away on the filter-only path. The doc id
                            // materialises here — once per candidate that
                            // reaches admission, via one array read.
                            let neighbor_doc = graph.doc_id(neighbor_ord);
                            let admitted = filter.is_none_or(|f| f.contains(neighbor_doc))
                                && !(check_deletions && reader.is_deleted(neighbor_doc));
                            if admitted {
                                found.push(ResultCandidate {
                                    id: neighbor_doc,
                                    distance: d,
                                });
                                if found.len() > ef_search {
                                    found.pop();
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Pristine path (no filter, no deletions) — byte-for-byte the
            // pre-#645 loop, so neither filtering (#645) nor deletion-awareness
            // (#665) can perturb its codegen or latency.
            while let Some(curr) = candidates.pop() {
                if let Some(furthest) = found.peek()
                    && curr.distance > furthest.distance
                    && found.len() >= ef_search
                {
                    break;
                }

                if prefetch_enabled {
                    prefetch_lookahead(&candidates, &visited);
                }

                if let Some(neighbors) = graph.neighbors(curr.ord, 0) {
                    // Pass 1: issue prefetch hints for unvisited neighbors.
                    // O(1) per neighbor (direct address computation, no
                    // allocation, no hash probe on the SQ hot path).
                    if prefetch_enabled {
                        for &neighbor_ord in neighbors {
                            if !visited.get(neighbor_ord as usize).unwrap_or(false) {
                                prefetch_ord(neighbor_ord);
                            }
                        }
                    }

                    // Pass 2: compute distances for unvisited neighbors (data
                    // loading overlaps with the prefetch hints issued above).
                    for &neighbor_ord in neighbors {
                        let nbr_idx = neighbor_ord as usize;
                        if visited.get(nbr_idx).unwrap_or(false) {
                            continue;
                        }
                        visited.set(nbr_idx, true);

                        let d = self.calc_dist_ord(
                            reader,
                            query,
                            quant_ctx.as_ref(),
                            prepared_query.as_ref(),
                            ord_to_pos.as_deref(),
                            graph,
                            neighbor_ord,
                            field_name,
                        )?;
                        let furthest_dist = found.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                        if d < furthest_dist || found.len() < ef_search {
                            candidates.push(Candidate {
                                ord: neighbor_ord,
                                distance: d,
                            });
                            found.push(ResultCandidate {
                                id: graph.doc_id(neighbor_ord),
                                distance: d,
                            });

                            if found.len() > ef_search {
                                found.pop();
                            }
                        }
                    }
                }
            }
        }

        self.finalize_graph_results(reader, query, request, field_name, found, visited.len())
    }

    /// Turn the result heap from a graph search (or the brute-force scan, see
    /// [`Self::search_graph`]'s `#738` mode) into ranked results.
    ///
    /// Shared tail of both HNSW search modes: applies the optional Stage 2
    /// rerank (Issue #481), drops field-missing candidates (`f32::MAX`, Issue
    /// #676), filters by `min_similarity`, sorts by similarity, and truncates
    /// to `top_k`. Lives outside the per-neighbour hot loop, so factoring it
    /// out does not affect graph-traversal latency.
    ///
    /// # Arguments
    ///
    /// * `found` - The candidate heap (int8 / quantized distances).
    /// * `candidates_examined` - Number of candidates the caller scored (the
    ///   visited-node count for a graph search, or the filter cardinality for
    ///   the brute-force scan); reported back for diagnostics.
    //
    // `#[inline]` so the graph-search call site folds this back in: extracting
    // the shared tail must not change the codegen (and thus latency) of the
    // unfiltered graph path, which is the dominant production case (Issue #645
    // showed how sensitive that path is to function-shape changes).
    #[inline]
    fn finalize_graph_results(
        &self,
        reader: &HnswIndexReader,
        query: &Vector,
        request: &VectorIndexQuery,
        field_name: &str,
        found: BinaryHeap<ResultCandidate>,
        candidates_examined: usize,
    ) -> Result<VectorIndexQueryResults> {
        // Stage 2 (Issue #481, refactored by #650 PR-1 / #931): when the
        // query asks for rerank (`rerank_factor`) and the reader has the
        // LRS1 sidecar loaded, run the shared `RerankPipeline` — for this
        // searcher normally a single `F32SidecarStage` that widens the
        // int8/ADC candidate set to `top_k * rerank_factor` and rescores
        // against the original f32 vectors. When either prerequisite is
        // missing we silently fall through to Stage 1 ranking — Stage 1
        // segments cannot recover the f32 information that was discarded
        // at index time, so there's nothing better to do.
        //
        // Issue #673 (#650 PR-3): on a PQ/PQ-FastScan segment, the graph
        // traversal already computed `effective_ef` ADC candidates, but
        // only the leading `top_k * rerank_factor` of them feed the exact
        // f32 stage — the rest are discarded unscored. When `ef` exceeds
        // that budget, insert a cheap int8 stage (`SqRerankStage`, derived
        // from the same f32 sidecar `F32SidecarStage` reads — see
        // `RerankStoragePool::int8_view`) ahead of the exact stage to
        // rescore that surplus instead of throwing it away. When `ef`
        // does not exceed the budget the surplus is empty, so the SQ
        // stage would narrow nothing and is skipped to avoid pure
        // overhead.
        use crate::vector::search::rerank::{
            F32SidecarStage, RerankCandidates, RerankPipeline, RerankStage, SqRerankStage,
        };
        let pipeline = match (request.params.rerank_factor, reader.rerank_storage()) {
            (Some(factor), Some(pool)) => {
                let metric = reader.distance_metric();
                let f32_stage: Box<dyn RerankStage> =
                    Box::new(F32SidecarStage::new(Arc::clone(pool), field_name, metric));

                let is_pq_segment = {
                    #[cfg(feature = "pq-fastscan")]
                    {
                        reader.vectors().pq_pool().is_some()
                            || reader.vectors().pq_fastscan_pool().is_some()
                    }
                    #[cfg(not(feature = "pq-fastscan"))]
                    {
                        reader.vectors().pq_pool().is_some()
                    }
                };
                let top_k = request.params.top_k;
                let ef = self.effective_ef(request);
                let sq_view = (is_pq_segment && ef > top_k.saturating_mul(factor))
                    .then(|| pool.int8_view())
                    .flatten();

                match sq_view {
                    Some(int8_pool) => Some(RerankPipeline::new(
                        vec![
                            Box::new(SqRerankStage::new(
                                Arc::clone(int8_pool),
                                field_name,
                                metric,
                            )),
                            f32_stage,
                        ],
                        vec![ef.div_ceil(top_k.max(1)), factor],
                    )),
                    None => Some(RerankPipeline::new(vec![f32_stage], vec![factor])),
                }
            }
            _ => None,
        };

        // Issue #927: when the pipeline's final stage actually applies,
        // the scores are `distance(raw_query, true_f32_vector)` — an
        // exact, cross-segment-comparable basis the multi-segment
        // fan-out must not overwrite with its (approximate) dequantized
        // rescore.
        let mut rerank_applied = false;
        let candidates_for_results: Vec<ResultCandidate> = match pipeline {
            Some(pipeline) => {
                // Ascending int8/ADC order so the pipeline's widening
                // prefix matches the pre-refactor `into_sorted_vec().take(..)`.
                let int8_sorted: Vec<ResultCandidate> = found.into_sorted_vec();
                let mut candidates = RerankCandidates::with_capacity(int8_sorted.len());
                for c in &int8_sorted {
                    candidates.push(c.id, c.distance);
                }
                rerank_applied = pipeline.run(query, &mut candidates, request.params.top_k)?;
                candidates
                    .doc_ids
                    .iter()
                    .zip(&candidates.distances)
                    .map(|(&id, &distance)| ResultCandidate { id, distance })
                    .collect()
            }
            None => found.into_iter().collect(),
        };

        // Convert candidate set to results.
        let field_name_owned = field_name.to_string();
        let mut final_results = Vec::new();
        for c in candidates_for_results {
            // Skip candidates that have no vector in the searched field
            // (Issue #676). The single HNSW graph mixes documents from every
            // field; `calc_dist` returns `f32::MAX` for a doc that lacks a
            // vector in `field_name`. Such docs must not leak into a
            // field-routed query's results — without this guard they would
            // surface whenever the result set is smaller than `top_k`.
            if c.distance == f32::MAX {
                continue;
            }

            // Convert cached distance to similarity without re-reading vectors.
            let similarity = reader.distance_metric().distance_to_similarity(c.distance);

            // Apply min_score filter.
            if similarity < request.params.min_similarity {
                continue;
            }

            // Only load vector data if explicitly requested.
            let vector = if request.params.include_vectors {
                reader.get_vector(c.id, field_name)?
            } else {
                None
            };

            final_results.push(VectorIndexQueryResult {
                doc_id: c.id,
                field_name: field_name_owned.clone(),
                similarity,
                distance: c.distance,
                vector,
            });
        }

        // Sort ascending by distance with a doc-id tiebreak (#933):
        // similarity's `exp(-d)` underflows to 0.0 at long range, collapsing
        // distant candidates into ties whose unstable order would make top-k
        // membership arbitrary; distance stays precise at any range.
        final_results.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.doc_id.cmp(&b.doc_id))
        });

        // Top K
        let top_k = request.params.top_k.min(final_results.len());
        final_results.truncate(top_k);

        let mut query_metadata = std::collections::HashMap::new();
        if rerank_applied {
            query_metadata.insert(
                crate::vector::search::searcher::SCORE_BASIS_METADATA_KEY.to_string(),
                crate::vector::search::searcher::SCORE_BASIS_F32_RERANK.to_string(),
            );
        }
        Ok(VectorIndexQueryResults {
            results: final_results,
            candidates_examined,
            search_time_ms: 0.0, // Set by caller
            query_metadata,
        })
    }

    /// Distance to the candidate at segment ordinal `ord` (Issue #686).
    ///
    /// The quantized hot path resolves the pool position from the
    /// ordinal (identity or one table read — no hash probe); the f32
    /// fallback translates the ordinal to a doc id with one array read
    /// and delegates to [`Self::calc_dist`].
    ///
    /// # Arguments
    ///
    /// * `ord_to_pos` - Ordinal → pool-position table (`None` = identity).
    /// * `graph` - The ordinal graph, for the f32 fallback translation.
    /// * `ord` - The candidate's segment ordinal.
    ///
    /// # Returns
    ///
    /// The distance, or `f32::MAX` when the doc has no vector in
    /// `field_name` (#676 semantics).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn calc_dist_ord(
        &self,
        reader: &HnswIndexReader,
        query: &Vector,
        quant_ctx: Option<&QuantizedSearchCtx>,
        prepared: Option<&PreparedQuery<'_>>,
        ord_to_pos: Option<&[u32]>,
        graph: &OrdinalHnswGraph,
        ord: u32,
        field_name: &str,
    ) -> Result<f32> {
        if let Some(ctx) = quant_ctx {
            return Ok(ctx.distance_ord(ord, ord_to_pos));
        }
        self.calc_dist(
            reader,
            query,
            quant_ctx,
            prepared,
            graph.doc_id(ord),
            field_name,
        )
    }

    fn calc_dist(
        &self,
        reader: &HnswIndexReader,
        query: &Vector,
        quant_ctx: Option<&QuantizedSearchCtx>,
        prepared: Option<&PreparedQuery<'_>>,
        doc_id: u64,
        field_name: &str,
    ) -> Result<f32> {
        // Issue #481 Stage 1, Step 6: prefer the int8 hot path when
        // the reader exposes a QuantizedVectorPool. The fallback
        // remains f32 for backward compatibility (OnDemand mode and
        // legacy f32 Owned).
        if let Some(ctx) = quant_ctx {
            return Ok(ctx.distance_doc(doc_id));
        }
        if let Some(target) = reader.get_vector(doc_id, field_name)? {
            // f32 reference path. Prefer the prepared query (#835): it caches
            // the query norm once per search so Cosine/Angular skip the
            // per-candidate `‖query‖²` accumulation (`simd_dot_and_norm_b`
            // instead of `simd_dot_and_norms`).
            let metric = reader.distance_metric();
            match prepared {
                Some(prepared) => metric.distance_with_prepared(prepared, &target.data),
                None => metric.distance(&query.data, &target.data),
            }
        } else {
            // Vector not found in this field?
            // Should return max distance or error?
            // Since graph contains doc_id, it should exist.
            // But if mixed fields, it might not exist in *this* field.
            Ok(f32::MAX)
        }
    }

    /// Issue software prefetch hints for the vector identified by `doc_id`.
    ///
    /// Performs an O(1) `u64` lookup in `idx` (no `String` allocation) to
    /// retrieve the base address of the vector's `f32` data, then emits one
    /// prefetch instruction per 64-byte cache line.  This lets the CPU start
    /// fetching the data from RAM before the distance computation begins,
    /// reducing memory-latency stalls on datasets larger than L3 cache.
    ///
    /// # Safety
    ///
    /// The addresses in `idx` were recorded from `Vec<f32>::as_ptr()` at reader
    /// construction time.  The backing `Arc<Vec<f32>>` is kept alive by
    /// `VectorStorage::Owned` inside the same `HnswIndexReader`, so every
    /// pointer is valid for the entire lifetime of the search.
    /// `_mm_prefetch` / `prfm` are pure hints that never dereference the pointer.
    #[inline]
    fn prefetch_neighbor(idx: &HashMap<u64, usize>, doc_id: u64, n_bytes: usize) {
        if let Some(&addr) = idx.get(&doc_id) {
            Self::prefetch_addr(addr, n_bytes);
        }
    }

    /// Issue software prefetch hints for `n_bytes` starting at `addr`
    /// (one hint per 64-byte cache line).
    ///
    /// Shared tail of [`Self::prefetch_neighbor`] (doc_id-keyed f32
    /// map) and the Issue #686 direct-address SQ path, which computes
    /// `addr` as `pool base + pos * stride` without any map probe.
    ///
    /// # Safety
    ///
    /// Callers must pass an address whose backing allocation outlives
    /// the search (both callers derive it from pools kept alive by the
    /// reader / search ctx). `_mm_prefetch` / `prfm` are pure hints
    /// that never dereference the pointer.
    ///
    /// # Arguments
    ///
    /// * `addr` - Base address of the record to prefetch.
    /// * `n_bytes` - Number of bytes the upcoming access will stream.
    ///
    /// No-ops on targets other than x86_64/aarch64 (e.g. wasm32, Issue
    /// #657): there is no software-prefetch intrinsic there, so callers on
    /// those targets would otherwise run this loop's bounds-check/increment
    /// for zero benefit on every neighbor visited during search.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[inline]
    fn prefetch_addr(addr: usize, n_bytes: usize) {
        let base_ptr = addr as *const i8;
        let mut offset = 0;
        while offset < n_bytes {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: see method doc comment.
            unsafe {
                use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(offset));
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: see method doc comment.
            unsafe {
                std::arch::asm!(
                    "prfm pldl1keep, [{p}]",
                    p = in(reg) base_ptr.add(offset),
                    options(nostack, readonly),
                );
            }
            offset += 64;
        }
    }

    /// See the primary definition above (Issue #657).
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline]
    fn prefetch_addr(_addr: usize, _n_bytes: usize) {}
}

#[cfg(test)]
mod ef_search_tests {
    //! Unit tests for `HnswSearcher::effective_ef` (Issue #644).
    //!
    //! These tests verify the precedence and `max` formula in isolation —
    //! integration tests in `tests.rs` cover the end-to-end flow.

    use super::*;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::vector::Vector;
    use crate::vector::reader::SimpleVectorReader;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};

    fn make_searcher(default_ef: Option<usize>) -> HnswSearcher {
        let reader = Arc::new(
            SimpleVectorReader::new(
                vec![(1u64, "f".to_string(), Vector::new(vec![1.0, 0.0]))],
                2,
                DistanceMetric::Cosine,
            )
            .expect("reader"),
        );
        HnswSearcher::with_default_ef_search(reader, default_ef).expect("searcher")
    }

    fn req(top_k: usize, ef: Option<usize>, rerank: Option<usize>) -> VectorIndexQuery {
        VectorIndexQuery {
            query: Vector::new(vec![1.0, 0.0]),
            params: VectorIndexQueryParams {
                top_k,
                ef_search: ef,
                rerank_factor: rerank,
                ..Default::default()
            },
            field_name: Some("f".to_string()),
            filter: None,
        }
    }

    #[test]
    fn fallback_default_is_50_when_no_override() {
        let s = make_searcher(None);
        // top_k below the fallback => the fallback (50) wins.
        assert_eq!(s.effective_ef(&req(10, None, None)), 50);
    }

    #[test]
    fn lifts_to_top_k_when_top_k_exceeds_default() {
        let s = make_searcher(None);
        // top_k = 100 > 50 fallback => effective_ef is lifted to top_k.
        assert_eq!(s.effective_ef(&req(100, None, None)), 100);
    }

    #[test]
    fn schema_default_takes_precedence_over_fallback() {
        let s = make_searcher(Some(300));
        // Schema default 300 wins over the 50 fallback.
        assert_eq!(s.effective_ef(&req(10, None, None)), 300);
    }

    #[test]
    fn per_query_override_beats_schema_default_and_fallback() {
        let s = make_searcher(Some(300));
        // Per-query 200 wins over schema default 300 *only when it is >= the
        // top_k floor*. With top_k = 10 the formula returns 200 since 200 > 10.
        assert_eq!(s.effective_ef(&req(10, Some(200), None)), 200);
    }

    #[test]
    fn rerank_factor_lifts_effective_ef() {
        let s = make_searcher(None);
        // top_k * rerank = 10 * 10 = 100 > 50 fallback => 100 wins.
        assert_eq!(s.effective_ef(&req(10, None, Some(10))), 100);
    }

    #[test]
    fn user_ef_still_wins_if_larger_than_top_k_times_rerank() {
        let s = make_searcher(None);
        // top_k * rerank = 100, user ef = 500 => 500 wins.
        assert_eq!(s.effective_ef(&req(10, Some(500), Some(10))), 500);
    }

    #[test]
    fn top_k_zero_is_safe() {
        let s = make_searcher(None);
        // top_k = 0 is degenerate; effective_ef should at least equal the fallback (50).
        assert_eq!(s.effective_ef(&req(0, None, None)), 50);
    }

    #[test]
    fn rerank_zero_is_treated_as_one() {
        let s = make_searcher(None);
        // rerank_factor = Some(0) is treated as 1 (defensive) so we never
        // collapse the candidate widening to zero.
        assert_eq!(s.effective_ef(&req(10, None, Some(0))), 50);
    }
}

#[cfg(test)]
mod nan_ordering_tests {
    //! Issue #667: the HNSW candidate / result heaps order by an `f32`
    //! `distance`. The previous `partial_cmp(...).unwrap_or(Equal)` made a
    //! NaN distance compare equal to everything — a non-total order, which
    //! `BinaryHeap` / `sort_unstable` forbid (silent reorder, or a panic on
    //! recent std). `total_cmp` restores a total order so a NaN is handled
    //! deterministically without losing or misordering the finite entries.

    use super::{Candidate, ResultCandidate};
    use std::collections::BinaryHeap;

    #[test]
    fn candidate_min_heap_handles_nan_without_panic() {
        // `Candidate` is a min-heap by distance (nearest pops first).
        let mut heap = BinaryHeap::new();
        for d in [3.0_f32, 1.0, f32::NAN, 2.0] {
            heap.push(Candidate {
                ord: 0,
                distance: d,
            });
        }
        let popped: Vec<f32> = std::iter::from_fn(|| heap.pop().map(|c| c.distance)).collect();
        assert_eq!(popped.len(), 4, "no candidate is lost");
        let finite: Vec<f32> = popped.iter().copied().filter(|d| !d.is_nan()).collect();
        assert_eq!(
            finite,
            vec![1.0, 2.0, 3.0],
            "finite distances pop nearest-first regardless of the NaN"
        );
        assert_eq!(
            popped.iter().filter(|d| d.is_nan()).count(),
            1,
            "the NaN is retained, not silently dropped"
        );
    }

    #[test]
    fn result_candidate_max_heap_handles_nan_without_panic() {
        // `ResultCandidate` is a max-heap by distance (furthest pops first).
        let mut heap = BinaryHeap::new();
        for d in [3.0_f32, 1.0, f32::NAN, 2.0] {
            heap.push(ResultCandidate { id: 0, distance: d });
        }
        let popped: Vec<f32> = std::iter::from_fn(|| heap.pop().map(|c| c.distance)).collect();
        assert_eq!(popped.len(), 4, "no candidate is lost");
        let finite: Vec<f32> = popped.iter().copied().filter(|d| !d.is_nan()).collect();
        assert_eq!(
            finite,
            vec![3.0, 2.0, 1.0],
            "finite distances pop furthest-first regardless of the NaN"
        );
    }
}
