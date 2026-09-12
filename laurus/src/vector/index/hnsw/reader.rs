//! HNSW vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::QuantizedVectorMeta;
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{
    FieldInterner, QuantHeader, VERSION_ORDINAL_GRAPH, VectorSegmentHeader, record_prefix_size,
};
use crate::vector::index::hnsw::graph::OrdinalHnswGraph;
use crate::vector::index::quantized_io::quantized_record_payload_size;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::index::rerank_storage::RerankStoragePool;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};

use crate::maintenance::deletion::DeletionBitmap;
/// Storage for vectors (in-memory or on-demand).
use crate::vector::index::storage::VectorStorage;

/// Build a `field_name → Arc<[u64]>` lookup from a flat `Vec<(u64, String)>`.
///
/// The grouping happens once at reader load so search-time
/// `doc_ids_for_field` is a single HashMap lookup + Arc clone.
fn build_vector_ids_by_field(
    vector_ids: &[(u64, u16)],
    field_dict: &[Arc<str>],
) -> HashMap<String, Arc<[u64]>> {
    let mut by_field: Vec<Vec<u64>> = vec![Vec::new(); field_dict.len()];
    for &(doc_id, fid) in vector_ids {
        by_field[fid as usize].push(doc_id);
    }
    field_dict
        .iter()
        .zip(by_field)
        .map(|(field, ids)| (field.to_string(), Arc::<[u64]>::from(ids)))
        .collect()
}

/// Reader for HNSW (Hierarchical Navigable Small World) vector indexes.
#[derive(Debug)]
pub struct HnswIndexReader {
    vectors: VectorStorage,
    /// `(doc_id, field_id)` per record; ids index [`Self::field_dict`]
    /// (Issue #633 PR-B — interned, no per-record heap `String`).
    vector_ids: Vec<(u64, u16)>,
    /// Per-segment field-name dictionary (synthesized at load for
    /// v1/v2 segments, taken from the header for v3).
    field_dict: Arc<[Arc<str>]>,
    dimension: usize,
    distance_metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
    /// Ordinal-addressed search graph (Issue #686), `None` when the
    /// segment was written without a graph block. Built from either the
    /// v1 (doc_id-encoded) or v2 (ordinal-encoded) graph block.
    pub graph: Option<Arc<OrdinalHnswGraph>>,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
    /// Per-field ordinal → pool-position tables (Issue #686).
    ///
    /// Empty when the identity `ordinal == position` holds — the segment
    /// is single-field with one record per doc id, which is what every
    /// current writer produces. A table is built only for legacy
    /// multi-field segments; `u32::MAX` marks a doc absent from the
    /// field (feeds the existing `f32::MAX` missing-distance semantics).
    ord_to_pos: HashMap<String, Arc<[u32]>>,
    /// Pre-built lookup table: `field_name → (doc_id → Vec<f32> base address as usize)`.
    ///
    /// Populated at load time for in-memory (`Owned`) storage; empty for
    /// on-demand (disk-backed) storage.  Enables zero-allocation, O(1) software
    /// prefetch hints on the HNSW search hot-path without the per-call
    /// `String` allocation that a direct `HashMap<(u64, String), _>` lookup
    /// would require.
    ///
    /// The address is stored as `usize` so that the struct remains `Send + Sync`.
    /// The pointer is valid for the lifetime of `self` because the backing
    /// `Arc<Vec<f32>>` is kept alive by `VectorStorage::Owned`.
    prefetch_index: HashMap<String, HashMap<u64, usize>>,

    /// Pre-built per-field doc-id list (`field_name → Arc<[u64]>`).
    ///
    /// Built once at load time so `doc_ids_for_field` returns a refcount-
    /// shared slice (no per-call allocation, no `Vec<(u64, String)>` clone,
    /// no linear filter). #405 (per-field `vector_ids` cache).
    vector_ids_by_field: HashMap<String, Arc<[u64]>>,

    /// Optional Stage 2 rerank storage pool (Issue #481).
    ///
    /// `Some(_)` only when (a) the LRS1 sidecar exists alongside the
    /// main `.hnsw` file and (b) the storage loading mode is Eager.
    /// Lazy mode silently skips the sidecar so the on-disk read pattern
    /// stays small; Stage 1 segments without a sidecar always yield
    /// `None`. The HNSW searcher reads this to decide whether to
    /// activate the two-stage rerank flow.
    pub rerank_storage: Option<Arc<RerankStoragePool>>,
}

impl HnswIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(LaurusError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Probe the trailing CRC-32 footer of a `.hnsw` segment (Issue #786) and
    /// return the stored checksum if a valid footer is present, leaving `input`
    /// rewound to offset 0.
    ///
    /// Unlike a full verification this reads **only** the 8-byte footer, not the
    /// whole file: the Eager load path folds the actual CRC computation into its
    /// single structural pass (Issue #789), so it just needs the expected value
    /// up front. New segments end with `[magic u32][crc-32 u32]` over all
    /// preceding bytes (written via [`crate::storage::checksum::CrcWriter`]); the
    /// footer is detected by the magic at `size - 8`. Legacy footer-less
    /// segments return `None` (verification is skipped, back-compat).
    ///
    /// # Arguments
    ///
    /// * `input` - The opened `.hnsw` segment stream.
    /// * `size` - The total file size (footer included).
    ///
    /// # Returns
    ///
    /// `Some(stored_crc)` when a footer is present, otherwise `None`.
    fn read_footer_crc(
        input: &mut dyn crate::storage::StorageInput,
        size: u64,
    ) -> Result<Option<u32>> {
        use std::io::SeekFrom;

        let mut stored = None;
        if size >= crate::vector::index::hnsw::HNSW_FOOTER_LEN {
            let content_len = size - crate::vector::index::hnsw::HNSW_FOOTER_LEN;
            input.seek(SeekFrom::Start(content_len))?;
            let mut footer = [0u8; 8];
            input.read_exact(&mut footer)?;
            let magic = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
            if magic == crate::vector::index::hnsw::HNSW_FOOTER_MAGIC {
                stored = Some(u32::from_le_bytes([
                    footer[4], footer[5], footer[6], footer[7],
                ]));
            }
        }
        input.seek(SeekFrom::Start(0))?;
        Ok(stored)
    }

    /// Verify that the `content_len` bytes from offset 0 hash to `expected`, in
    /// an independent sequential pass, then rewind to 0.
    ///
    /// Used on the Lazy / OnDemand load path (Issue #789): that parse seeks over
    /// the vector payload and so cannot fold the checksum into its structural
    /// pass the way the Eager path does, so the integrity guarantee from
    /// Issue #786 is preserved here with a dedicated pass.
    ///
    /// # Arguments
    ///
    /// * `input` - The opened `.hnsw` segment stream.
    /// * `content_len` - Number of payload bytes preceding the footer.
    /// * `expected` - The footer's stored CRC-32.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::index`] if the recomputed CRC does not match
    /// `expected` (corruption), or on I/O failure.
    fn verify_footer_content(
        input: &mut dyn crate::storage::StorageInput,
        content_len: u64,
        expected: u32,
    ) -> Result<()> {
        use std::io::{Read, SeekFrom};

        input.seek(SeekFrom::Start(0))?;
        let mut crc_in = crate::storage::checksum::CrcReader::new(&mut *input);
        let mut remaining = content_len;
        let mut buf = [0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            crc_in.read_exact(&mut buf[..want])?;
            remaining -= want as u64;
        }
        let computed = crc_in.checksum();
        input.seek(SeekFrom::Start(0))?;
        if computed != expected {
            return Err(LaurusError::index(
                "HNSW segment checksum mismatch: .hnsw file is corrupted",
            ));
        }
        Ok(())
    }

    /// Derive the segment's ordinal table from its record id sequence.
    ///
    /// The ordinal of a vector is its rank in the ascending,
    /// deduplicated record doc-id sequence (Issue #686). Records are
    /// always written sorted by doc id, so a consecutive dedup suffices;
    /// a decreasing id means the segment is corrupt.
    ///
    /// # Arguments
    ///
    /// * `vector_ids` - The `(doc_id, field_name)` pairs in on-disk
    ///   record order.
    ///
    /// # Returns
    ///
    /// The strictly ascending unique doc-id table, or an error if the
    /// record ids are not non-decreasing.
    fn unique_sorted_doc_ids(vector_ids: &[(u64, u16)]) -> Result<Arc<[u64]>> {
        let mut unique: Vec<u64> = Vec::with_capacity(vector_ids.len());
        for (doc_id, _) in vector_ids {
            match unique.last() {
                Some(&last) if *doc_id == last => {}
                Some(&last) if *doc_id < last => {
                    return Err(LaurusError::index(format!(
                        "HNSW segment corrupt: record doc ids not sorted \
                         ({doc_id} follows {last})"
                    )));
                }
                _ => unique.push(*doc_id),
            }
        }
        Ok(Arc::from(unique))
    }

    /// Parse a v1 (doc_id-encoded) graph block and translate it to
    /// ordinals (Issue #686 back-compat path).
    ///
    /// Layout: `has_graph u8, entry_point u64 (u64::MAX = None),
    /// max_level u32, node_count u64, per node {doc_id u64,
    /// layer_count u32, per layer [neighbor_count u32, neighbors u64…]}`.
    ///
    /// Nodes or neighbours whose doc id has no record are dropped with a
    /// warning (only reachable on corrupt files — the writer always
    /// emits graph nodes for exactly the record id set); a dangling
    /// entry point degrades to `None` (empty search results).
    ///
    /// # Arguments
    ///
    /// * `input` - Stream positioned at the graph block.
    /// * `file_size` - Total file size, for allocation bounding (#806).
    /// * `doc_ids` - The segment's ordinal table.
    ///
    /// # Returns
    ///
    /// The ordinal graph, or `None` when the block is absent.
    fn read_graph_block_v1(
        input: &mut dyn crate::storage::StorageInput,
        file_size: u64,
        doc_ids: Arc<[u64]>,
    ) -> Result<Option<Arc<OrdinalHnswGraph>>> {
        use crate::util::alloc_bounds::checked_capacity;
        use ahash::AHashMap;

        let mut has_graph_buf = [0u8; 1];
        if input.read_exact(&mut has_graph_buf).is_err() || has_graph_buf[0] != 1 {
            return Ok(None);
        }

        let mut entry_point_buf = [0u8; 8];
        input.read_exact(&mut entry_point_buf)?;
        let entry_point_raw = u64::from_le_bytes(entry_point_buf);
        let entry_point_doc = if entry_point_raw == u64::MAX {
            None
        } else {
            Some(entry_point_raw)
        };

        let mut max_level_buf = [0u8; 4];
        input.read_exact(&mut max_level_buf)?;
        let max_level = u32::from_le_bytes(max_level_buf) as usize;

        let mut node_count_buf = [0u8; 8];
        input.read_exact(&mut node_count_buf)?;
        let node_count = u64::from_le_bytes(node_count_buf) as usize;

        // Bound every graph allocation by the bytes left in the file
        // (Issue #806). The graph trails the vector payload, so this
        // remaining count is tight: a corrupt count on a small graph can
        // no longer drive a huge `with_capacity`. Reused for the inner
        // layer / neighbor counts so no extra syscall is taken inside
        // the per-node / per-layer loops.
        let graph_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        // Each v1 node serializes at least doc_id (8) + layer_count (4).
        checked_capacity(node_count, 12, graph_remaining, "hnsw node_count")?;

        let rank: AHashMap<u64, u32> = doc_ids
            .iter()
            .enumerate()
            .map(|(ord, &id)| (id, ord as u32))
            .collect();
        let mut nodes: Vec<Vec<Vec<u32>>> = vec![Vec::new(); doc_ids.len()];

        for _ in 0..node_count {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);
            let node_ord = rank.get(&doc_id).copied();
            if node_ord.is_none() {
                log::warn!(
                    "HNSW v1 graph node {doc_id} has no record in the segment; \
                     dropping it (corrupt segment?)"
                );
            }

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
                    let neighbor_id = u64::from_le_bytes(neighbor_buf);
                    match rank.get(&neighbor_id) {
                        Some(&ord) => neighbors.push(ord),
                        None => log::warn!(
                            "HNSW v1 graph neighbour {neighbor_id} has no record in \
                             the segment; dropping the edge (corrupt segment?)"
                        ),
                    }
                }
                layers.push(neighbors);
            }
            if let Some(ord) = node_ord {
                nodes[ord as usize] = layers;
            }
        }

        let entry_point = match entry_point_doc {
            Some(id) => {
                let ord = rank.get(&id).copied();
                if ord.is_none() {
                    log::warn!(
                        "HNSW v1 graph entry point {id} has no record in the segment; \
                         searches on this segment will return no graph results"
                    );
                }
                ord
            }
            None => None,
        };

        Ok(Some(Arc::new(OrdinalHnswGraph::from_parts(
            entry_point,
            max_level,
            doc_ids,
            nodes,
        )?)))
    }

    /// Parse a v2 (ordinal-encoded) graph block (Issue #686).
    ///
    /// Layout: `has_graph u8, entry_point u32 (u32::MAX = None),
    /// max_level u32, node_count u32, per node (ordinal implicit by
    /// order, doc_id dropped) {layer_count u32, per layer
    /// [neighbor_count u32, neighbor ordinals u32…]}`.
    ///
    /// # Arguments
    ///
    /// * `input` - Stream positioned at the graph block.
    /// * `file_size` - Total file size, for allocation bounding (#806).
    /// * `doc_ids` - The segment's ordinal table; `node_count` must
    ///   match its length exactly.
    ///
    /// # Returns
    ///
    /// The ordinal graph, or `None` when the block is absent; an error
    /// on any count/ordinal inconsistency (corrupt segment).
    fn read_graph_block_v2(
        input: &mut dyn crate::storage::StorageInput,
        file_size: u64,
        doc_ids: Arc<[u64]>,
    ) -> Result<Option<Arc<OrdinalHnswGraph>>> {
        use crate::util::alloc_bounds::checked_capacity;

        let mut has_graph_buf = [0u8; 1];
        if input.read_exact(&mut has_graph_buf).is_err() || has_graph_buf[0] != 1 {
            return Ok(None);
        }

        let mut entry_point_buf = [0u8; 4];
        input.read_exact(&mut entry_point_buf)?;
        let entry_point_raw = u32::from_le_bytes(entry_point_buf);
        let entry_point = if entry_point_raw == u32::MAX {
            None
        } else {
            Some(entry_point_raw)
        };

        let mut max_level_buf = [0u8; 4];
        input.read_exact(&mut max_level_buf)?;
        let max_level = u32::from_le_bytes(max_level_buf) as usize;

        let mut node_count_buf = [0u8; 4];
        input.read_exact(&mut node_count_buf)?;
        let node_count = u32::from_le_bytes(node_count_buf) as usize;
        if node_count != doc_ids.len() {
            return Err(LaurusError::index(format!(
                "HNSW v2 graph corrupt: node_count {node_count} does not match \
                 the segment's {} unique record doc ids",
                doc_ids.len()
            )));
        }

        // Allocation bounding (#806); v2 strides are 4 bytes per node
        // minimum (layer_count) and 4 bytes per neighbour ordinal.
        let graph_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        checked_capacity(node_count, 4, graph_remaining, "hnsw node_count")?;

        let mut nodes: Vec<Vec<Vec<u32>>> = Vec::with_capacity(node_count);
        for _ in 0..node_count {
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
                    neighbors.push(u32::from_le_bytes(neighbor_buf));
                }
                layers.push(neighbors);
            }
            nodes.push(layers);
        }

        // `from_parts` validates the entry point and every neighbour
        // ordinal against `node_count`.
        Ok(Some(Arc::new(OrdinalHnswGraph::from_parts(
            entry_point,
            max_level,
            doc_ids,
            nodes,
        )?)))
    }

    /// Load an HNSW vector index from storage.
    ///
    /// # Arguments
    ///
    /// * `storage` - Shared storage backend (cloned into `OnDemand` for concurrent reads).
    /// * `path` - Base path/name for the index file (`.hnsw` extension is appended).
    /// * `distance_metric` - Distance metric used for similarity computations.
    ///
    /// # Returns
    ///
    /// A new `HnswIndexReader` instance.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O or format errors.
    pub fn load(
        storage: Arc<dyn Storage>,
        path: &str,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        use crate::util::alloc_bounds::{checked_capacity, checked_len};
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut raw_input = storage.open_input(&file_name)?;

        // Ground truth for bounding allocations sized from the (for legacy
        // footer-less segments, unverified) header counts below (Issue #806).
        let file_size = raw_input.size()?;

        // Probe the CRC-32 footer (Issue #786) cheaply: read only the 8-byte
        // footer to learn the expected checksum, not the whole file.
        let stored_crc = Self::read_footer_crc(&mut *raw_input, file_size)?;
        let content_len = match stored_crc {
            Some(_) => file_size - crate::vector::index::hnsw::HNSW_FOOTER_LEN,
            None => file_size,
        };

        // Fold the integrity check into the single Eager structural pass
        // (Issue #789): that pass reads the whole content sequentially, so a CRC
        // accumulated *during* it verifies the footer with no extra read. The
        // Lazy / OnDemand parse seeks over the vector payload and so cannot
        // fold; verify it up front with a dedicated pass instead (the #786
        // behavior). Footer-less legacy segments have nothing to verify.
        let eager = matches!(storage.loading_mode(), crate::storage::LoadingMode::Eager);
        let fold = eager && stored_crc.is_some();
        if !fold && let Some(expected) = stored_crc {
            Self::verify_footer_content(&mut *raw_input, content_len, expected)?;
        }

        // Wrap the input so the structural parse below accumulates the content
        // CRC as it reads (only when `fold`; otherwise this is a thin
        // position-tracking pass-through). The wrapper implements `StorageInput`,
        // so the parse — including the Lazy seeks — is unchanged.
        let mut input = crate::storage::checksum::ChecksumTrackingInput::new(raw_input, fold);

        // Read metadata (vector count stored as u64)
        let mut num_vectors_buf = [0u8; 8];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u64::from_le_bytes(num_vectors_buf) as usize;

        // We already have dimension from argument, but file has it too.
        // Let's read it to advance cursor, and verify?
        // Or strictly trust file? FlatIndexReader reads it.
        // Let's read it.
        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut m_buf = [0u8; 4];
        input.read_exact(&mut m_buf)?;
        let m = u32::from_le_bytes(m_buf) as usize;

        let mut ef_construction_buf = [0u8; 4];
        input.read_exact(&mut ef_construction_buf)?;
        let ef_construction = u32::from_le_bytes(ef_construction_buf) as usize;

        // Read the Issue #481 vector segment header (LVS1). Pre-Stage-1
        // segments are rejected with IncompatibleFormat. Both Scalar8Bit
        // (Stage 1) and ProductQuantization (Stage 3) variants are
        // handled here; Lazy mode silently degrades PQ to "not
        // supported" because the OnDemand path's offsets table only
        // carries Scalar8Bit params today.
        // Issue #921: pass the bytes physically left in the file so the
        // header's PQ codebook allocation is bounded before it reserves.
        let header_available =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let header = VectorSegmentHeader::read_from(&mut input, header_available)?;

        // Graph-block parser, version-dispatched (Issue #686). The graph
        // physically trails the records, so by the time each quant branch
        // calls this its `vector_ids` are fully collected — the ordinal
        // table (ascending unique record doc ids) is derived from them.
        let graph_version = header.version;

        // Interned field ids (Issue #633 PR-B): one shared dictionary per
        // segment instead of one heap `String` per record.
        let mut interner = FieldInterner::from_header(&header);
        let read_graph = |input: &mut dyn crate::storage::StorageInput,
                          vector_ids: &[(u64, u16)]|
         -> Result<Option<Arc<OrdinalHnswGraph>>> {
            let doc_ids = Self::unique_sorted_doc_ids(vector_ids)?;
            if graph_version >= VERSION_ORDINAL_GRAPH {
                Self::read_graph_block_v2(input, file_size, doc_ids)
            } else {
                Self::read_graph_block_v1(input, file_size, doc_ids)
            }
        };

        // Bytes left for the per-vector records section, captured once at its
        // start (Issue #806). Reused by the per-record `field_name_len` /
        // payload checks below so the hot loop adds no extra syscall.
        let records_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);

        let (vectors, vector_ids, graph, field_dict) = match (&header.quant, storage.loading_mode())
        {
            (QuantHeader::Scalar8Bit(params), crate::storage::LoadingMode::Eager) => {
                // Step 6 of #481 Stage 1: load vectors as int8 + meta
                // directly into a QuantizedVectorPool so the search
                // hot loop can use distance_quantized without per-call
                // dequantization. The legacy
                // VectorIndexReader::get_vector API still works via
                // VectorStorage::OwnedQuantized's dequantize-on-get.
                // Each record is at least doc_id (8) + field_name_len (4) +
                // the fixed quantized payload (dim int8 + 8 meta). Bounding
                // `num_vectors` by that stride also bounds the per-record
                // `dimension`-sized int8 read (Issue #806).
                let record_stride = record_prefix_size(header.version)
                    + quantized_record_payload_size(dimension) as u64;
                checked_capacity(
                    num_vectors,
                    record_stride,
                    records_remaining,
                    "hnsw num_vectors",
                )?;
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>, QuantizedVectorMeta)> =
                    Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let fid = interner.read_record_field_id(
                        &header,
                        &mut input,
                        records_remaining,
                        "hnsw field_name_len",
                    )?;

                    let mut int8 = vec![0u8; dimension];
                    input.read_exact(&mut int8)?;
                    let mut sum_q_buf = [0u8; 4];
                    let mut norm_q_buf = [0u8; 4];
                    input.read_exact(&mut sum_q_buf)?;
                    input.read_exact(&mut norm_q_buf)?;
                    let meta = QuantizedVectorMeta {
                        sum_q: u32::from_le_bytes(sum_q_buf),
                        norm_q: f32::from_le_bytes(norm_q_buf),
                    };

                    vector_ids.push((doc_id, fid));
                    // Transient clone for the pool's String-shaped build
                    // input; the pool retains only per-field keys.
                    records.push((doc_id, interner.name(fid).to_string(), int8, meta));
                }
                let graph = read_graph(&mut input, &vector_ids)?;
                let pool = QuantizedVectorPool::build(*params, dimension, records);
                (
                    VectorStorage::OwnedQuantized(Arc::new(pool)),
                    vector_ids,
                    graph,
                    interner.into_dict(),
                )
            }
            (QuantHeader::Scalar8Bit(params), crate::storage::LoadingMode::Lazy) => {
                let record_stride = record_prefix_size(header.version)
                    + quantized_record_payload_size(dimension) as u64;
                checked_capacity(
                    num_vectors,
                    record_stride,
                    records_remaining,
                    "hnsw num_vectors",
                )?;
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                // Seek to the start of the per-vector entries: HNSW preamble
                // (count u64 + dim u32 + m u32 + ef u32 = 20 bytes) followed
                // by the parsed header's real size (which includes the v3
                // field dictionary, Issue #633 — reconstructing a fresh
                // header here would omit it).
                let start_pos = 20u64 + header.serialized_size() as u64;
                input
                    .seek(std::io::SeekFrom::Start(start_pos))
                    .map_err(LaurusError::Io)?;

                let quant_payload_size = quantized_record_payload_size(dimension) as i64;

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let fid = interner.read_record_field_id(
                        &header,
                        &mut input,
                        records_remaining,
                        "hnsw field_name_len",
                    )?;

                    // Offsets point at the payload start (right after the
                    // record prefix), so `VectorStorage::get` seeks straight
                    // to the int8 data (Issue #633).
                    let payload_offset = input.stream_position().map_err(LaurusError::Io)?;
                    offsets.insert((doc_id, fid), payload_offset);
                    vector_ids.push((doc_id, fid));

                    input
                        .seek(std::io::SeekFrom::Current(quant_payload_size))
                        .map_err(LaurusError::Io)?;
                }
                let graph = read_graph(&mut input, &vector_ids)?;

                let field_dict = interner.into_dict();
                (
                    VectorStorage::OnDemand {
                        storage: storage.clone(),
                        file_name: file_name.clone(),
                        offsets: Arc::new(offsets),
                        field_dict: field_dict.clone(),
                        quant_params: Some(*params),
                        cached_input: Arc::new(std::sync::RwLock::new(None)),
                    },
                    vector_ids,
                    graph,
                    field_dict,
                )
            }
            (
                QuantHeader::ProductQuantization {
                    params: pq_params,
                    codebook,
                },
                _loading_mode,
            ) => {
                // Stage 3 (#481): read M-byte codes per vector into a
                // PqVectorPool. Lazy mode is not yet supported for PQ
                // segments (the OnDemand path's offsets table only
                // carries Scalar8Bit params), so we eagerly load the
                // codes regardless of `loading_mode`.
                let codes_size = pq_params.m as usize;
                // Each PQ record is at least the version-dependent prefix
                // (doc_id + field reference, Issue #633) + `codes_size`
                // bytes of codes (Issue #806).
                checked_capacity(
                    num_vectors,
                    record_prefix_size(header.version) + codes_size as u64,
                    records_remaining,
                    "hnsw num_vectors",
                )?;
                checked_len(codes_size, records_remaining, "hnsw pq codes")?;
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>)> = Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let fid = interner.read_record_field_id(
                        &header,
                        &mut input,
                        records_remaining,
                        "hnsw field_name_len",
                    )?;

                    let mut codes = vec![0u8; codes_size];
                    input.read_exact(&mut codes)?;

                    vector_ids.push((doc_id, fid));
                    records.push((doc_id, interner.name(fid).to_string(), codes));
                }
                let graph = read_graph(&mut input, &vector_ids)?;
                let pool = crate::vector::index::pq_storage::PqVectorPool::build(
                    *pq_params,
                    codebook.clone(),
                    records,
                );
                (
                    VectorStorage::OwnedPq(Arc::new(pool)),
                    vector_ids,
                    graph,
                    interner.into_dict(),
                )
            }
            #[cfg(feature = "pq-fastscan")]
            (
                QuantHeader::ProductQuantizationFastScan {
                    params: pq_params,
                    codebook,
                },
                _loading_mode,
            ) => {
                // Mirror the K=256 PQ load path but read 4-bit packed
                // codes via `pq_fastscan_io::read_pq_fastscan_record`
                // and build a `PqFastScanPool` for the SIMD-friendly
                // block-transposed in-memory layout.
                // Each record is at least the version-dependent prefix
                // (doc_id + field reference, Issue #633) + one byte of
                // packed codes (Issue #806).
                checked_capacity(
                    num_vectors,
                    record_prefix_size(header.version) + 1,
                    records_remaining,
                    "hnsw num_vectors",
                )?;
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>)> = Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let fid = interner.read_record_field_id(
                        &header,
                        &mut input,
                        records_remaining,
                        "hnsw field_name_len",
                    )?;

                    let codes = crate::vector::index::pq_fastscan_io::read_pq_fastscan_record(
                        &mut input, *pq_params,
                    )?;

                    vector_ids.push((doc_id, fid));
                    records.push((doc_id, interner.name(fid).to_string(), codes));
                }
                let graph = read_graph(&mut input, &vector_ids)?;
                let pool = crate::vector::index::pq_fastscan_storage::PqFastScanPool::build(
                    *pq_params,
                    codebook.clone(),
                    records,
                )?;
                (
                    VectorStorage::OwnedPqFastScan(Arc::new(pool)),
                    vector_ids,
                    graph,
                    interner.into_dict(),
                )
            }
        };

        // Finalize the folded CRC verification (Issue #789). On the Eager path
        // the structural parse above read every content byte sequentially
        // through `input`, accumulating the CRC as it went; `absorb_to` covers
        // any residual bytes (the parse normally stops exactly at the footer, so
        // this is a no-op), then the running CRC is compared against the
        // footer's stored value. This replaces the separate full read that
        // Issue #786 used, so corruption is still detected with no extra I/O.
        // `is_sequential()` guards the running CRC: the Eager parse never seeks
        // backward, but if that ever changes the wrapper degrades gracefully to
        // a dedicated verification pass so the #786 guarantee can never silently
        // weaken. `fold` is only set in Eager mode with a footer present, so
        // Lazy/OnDemand and footer-less legacy segments skip this block (they
        // were verified up front or have nothing to verify).
        if fold && let Some(expected) = stored_crc {
            if input.is_sequential() {
                input.absorb_to(content_len).map_err(LaurusError::Io)?;
                if input.checksum() != expected {
                    return Err(LaurusError::index(
                        "HNSW segment checksum mismatch: .hnsw file is corrupted",
                    ));
                }
            } else {
                Self::verify_footer_content(&mut input, content_len, expected)?;
            }
        }

        // Build zero-allocation prefetch lookup. For OwnedQuantized
        // (Step 6+), the address points to the int8 record start
        // inside the contiguous AoS buffer; the int8 payload + 8 bytes
        // of meta will be prefetched per neighbour. For Owned (legacy
        // f32, kept for symmetry), the address points to the Vec<f32>
        // data. Empty for OnDemand where CPU cache hints don't apply.
        let prefetch_index: HashMap<String, HashMap<u64, usize>> = match &vectors {
            VectorStorage::Owned(map) => {
                let mut idx: HashMap<String, HashMap<u64, usize>> = HashMap::new();
                for ((doc_id, field_name), vector) in map.iter() {
                    idx.entry(field_name.clone())
                        .or_default()
                        .insert(*doc_id, vector.data.as_ptr() as usize);
                }
                idx
            }
            VectorStorage::OwnedQuantized(_) => {
                // The ordinal search path (Issue #686) computes int8
                // prefetch addresses directly from the pool as
                // `base + pos * pad_dim` — a doc_id-keyed address map
                // would only add a hash probe back to the hot loop, so
                // none is built.
                HashMap::new()
            }
            VectorStorage::OwnedPq(_) => {
                // PQ records are M bytes each (8-32 bytes) — small
                // enough that prefetching adds no benefit; the LUT is
                // the more important cache occupant.
                HashMap::new()
            }
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                // FastScan uses 4-bit packed codes; the SIMD kernel
                // streams entire 32-vector blocks (~16M bytes/block
                // for typical M) so per-vector prefetch hints add
                // nothing on top of the natural sequential access.
                HashMap::new()
            }
            VectorStorage::OnDemand { .. } => HashMap::new(),
        };

        // Per-field doc-id lookup (#405). Built from `vector_ids` so it
        // reflects the same set the search path used to filter at every
        // call; finalised into `Arc<[u64]>` for cheap clone semantics.
        let vector_ids_by_field = build_vector_ids_by_field(&vector_ids, &field_dict);

        // Per-field ordinal → pool-position tables (Issue #686). The
        // identity `ordinal == position` holds exactly when the segment
        // has a single field and one record per doc id (what every
        // current writer produces): both orderings are then the same
        // doc_id-ascending record sequence. Only legacy multi-field
        // segments pay for explicit tables.
        let ord_to_pos: HashMap<String, Arc<[u32]>> = {
            let unique = Self::unique_sorted_doc_ids(&vector_ids)?;
            let identity = vector_ids_by_field.len() <= 1 && vector_ids.len() == unique.len();
            if identity {
                HashMap::new()
            } else {
                let pos_index_for = |field: &str| match &vectors {
                    VectorStorage::OwnedQuantized(pool) => pool.field_position_index(field),
                    VectorStorage::OwnedPq(pool) => pool.field_position_index(field),
                    #[cfg(feature = "pq-fastscan")]
                    VectorStorage::OwnedPqFastScan(pool) => pool.field_position_index(field),
                    _ => None,
                };
                let mut tables = HashMap::new();
                for field in vector_ids_by_field.keys() {
                    if let Some(pos_map) = pos_index_for(field) {
                        let table: Vec<u32> = unique
                            .iter()
                            .map(|id| pos_map.get(id).copied().unwrap_or(u32::MAX))
                            .collect();
                        tables.insert(field.clone(), Arc::<[u32]>::from(table));
                    }
                }
                tables
            }
        };

        // Stage 2 rerank sidecar (Issue #481), via the shared loader
        // (#650 PR-2 / #554): loads whenever the sidecar exists,
        // regardless of loading mode — the old Eager-only gate here never
        // fired behind the engine (PrefixedStorage misreported Eager until
        // #554 made it delegate), so "rerank works on mmap" is the
        // shipped, tested behavior. The pool's vector positions pair with
        // `vector_ids` (the same order as the LVS1 segment).
        let rerank_storage = crate::vector::index::rerank_sidecar::load_rerank_sidecar(
            storage.as_ref(),
            &file_name,
            dimension,
            &vector_ids,
            &field_dict,
        )?;

        Ok(Self {
            vectors,
            vector_ids,
            field_dict,
            dimension,
            distance_metric,
            m,
            ef_construction,
            graph,
            deletion_bitmap: None,
            ord_to_pos,
            prefetch_index,
            vector_ids_by_field,
            rerank_storage,
        })
    }

    /// Per-field ordinal → pool-position table (Issue #686).
    ///
    /// Returns `None` when the identity `ordinal == position` holds for
    /// `field_name` (single-field segment with one record per doc id —
    /// what every current writer produces), in which case the searcher
    /// uses the ordinal directly as the pool position. A `Some` table is
    /// only present for legacy multi-field segments; `u32::MAX` entries
    /// mark docs absent from the field.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The vector field being searched.
    ///
    /// # Returns
    ///
    /// The translation table, or `None` for the identity mapping.
    pub(crate) fn field_ord_to_pos(&self, field_name: &str) -> Option<Arc<[u32]>> {
        self.ord_to_pos.get(field_name).cloned()
    }

    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Returns whether `doc_id` has been logically deleted.
    ///
    /// `pub(crate)` so the HNSW searcher can consult it during graph traversal
    /// (Issue #665): a deleted node must not be admitted to the result heap,
    /// otherwise it consumes an `ef_search` slot and leaks into results (the
    /// quantized distance path never calls `get_vector`, the only place
    /// deletions were previously honoured).
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Internal document ID to test.
    ///
    /// # Returns
    ///
    /// `true` if a deletion bitmap is attached and marks `doc_id` deleted;
    /// `false` when no bitmap is attached.
    pub(crate) fn is_deleted(&self, doc_id: u64) -> bool {
        if let Some(bitmap) = &self.deletion_bitmap {
            bitmap.is_deleted(doc_id)
        } else {
            false
        }
    }

    /// Returns whether this reader has any logically deleted documents.
    ///
    /// Used by [`HnswSearcher`](crate::vector::index::hnsw::HnswSearcher) to
    /// decide whether graph traversal needs the per-neighbour deletion
    /// bookkeeping at all. When no bitmap is attached, or it is attached but
    /// empty (e.g. a freshly initialized segment), this returns `false` so the
    /// pristine, no-bookkeeping search path is preserved unchanged.
    ///
    /// # Returns
    ///
    /// `true` if a deletion bitmap is attached and holds at least one deleted
    /// document; `false` otherwise.
    pub(crate) fn has_deletions(&self) -> bool {
        self.deletion_bitmap
            .as_ref()
            .is_some_and(|bitmap| bitmap.deleted_count.load(Ordering::Relaxed) > 0)
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.m, self.ef_construction)
    }

    /// Returns a reference to the per-field prefetch lookup table for `field_name`.
    ///
    /// The returned map provides O(1), zero-allocation access from `doc_id` to
    /// the base address of the corresponding `Vec<f32>` data, allowing
    /// [`HnswSearcher`] to issue software prefetch hints on the search hot-path
    /// without the `String` allocation that a direct `HashMap<(u64, String), _>`
    /// lookup would require.
    ///
    /// Returns `None` for on-demand (disk-backed) storage where prefetching is
    /// not applicable.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The name of the vector field.
    pub(crate) fn field_prefetch_index(&self, field_name: &str) -> Option<&HashMap<u64, usize>> {
        self.prefetch_index.get(field_name)
    }

    /// Borrow the underlying
    /// [`crate::vector::index::storage::VectorStorage`].
    ///
    /// Intended for the HNSW searcher to detect the
    /// [`crate::vector::index::storage::VectorStorage::OwnedQuantized`]
    /// variant and switch to the int8 hot path (Issue #481 Stage 1
    /// Step 6).
    pub fn vectors(&self) -> &crate::vector::index::storage::VectorStorage {
        &self.vectors
    }

    /// Iterate the interned `(doc_id, field_name)` records without
    /// materializing a `String` per record (Issue #672).
    ///
    /// The trait-level [`VectorIndexReader::vector_ids`] must rehydrate
    /// owned `String`s at every call (its signature predates the #633
    /// interning); callers that already hold a concrete
    /// `HnswIndexReader` — e.g. the warmup page-fault pass — can borrow
    /// the dictionary-backed names instead.
    ///
    /// # Returns
    ///
    /// An iterator over `(doc_id, &field_name)` in record order.
    pub(crate) fn interned_vector_ids(&self) -> impl Iterator<Item = (u64, &str)> {
        self.vector_ids
            .iter()
            .map(|&(id, fid)| (id, &*self.field_dict[fid as usize]))
    }

    /// Borrow the optional Stage 2 rerank storage pool.
    ///
    /// Returns `Some(_)` only when this reader was loaded against a
    /// segment whose LRS1 sidecar was present and the storage loading
    /// mode allowed eager sidecar load. The HNSW searcher consults
    /// this to decide whether the two-stage rerank flow is available.
    pub fn rerank_storage(&self) -> Option<&Arc<RerankStoragePool>> {
        self.rerank_storage.as_ref()
    }
}

impl VectorIndexReader for HnswIndexReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        if self.is_deleted(doc_id) {
            return Ok(None);
        }
        self.vectors.get(doc_id, field_name, self.dimension)
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut result = Vec::new();
        for &(id, fid) in &self.vector_ids {
            let field = &self.field_dict[fid as usize];
            if id == doc_id
                && !self.is_deleted(id)
                && let Some(vec) = self.vectors.get(id, field, self.dimension)?
            {
                result.push((field.to_string(), vec));
            }
        }
        Ok(result)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        let mut result = Vec::with_capacity(doc_ids.len());
        for (id, field) in doc_ids {
            if self.is_deleted(*id) {
                result.push(None);
            } else {
                result.push(self.vectors.get(*id, field, self.dimension)?);
            }
        }
        Ok(result)
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        // Rehydrated at the trait boundary (Issue #633 PR-B).
        Ok(self
            .vector_ids
            .iter()
            .map(|&(id, fid)| (id, self.field_dict[fid as usize].to_string()))
            .collect())
    }

    fn doc_ids_for_field(&self, field_name: &str) -> Arc<[u64]> {
        // O(1) HashMap lookup + Arc clone (refcount bump). Compared to
        // the default impl this avoids the per-call `Vec<(u64, String)>`
        // clone of the full corpus and the linear filter scan. #405.
        self.vector_ids_by_field
            .get(field_name)
            .cloned()
            .unwrap_or_else(|| Vec::<u64>::new().into())
    }

    fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        let memory_usage = match &self.vectors {
            VectorStorage::Owned(vectors) => vectors.len() * (8 + self.dimension * 4),
            VectorStorage::OwnedQuantized(pool) => pool.heap_size(),
            VectorStorage::OwnedPq(pool) => pool.data.len() + pool.codebook.len() * 4,
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.packed.len() + pool.codebook.len() * 4,
            VectorStorage::OnDemand { offsets, .. } => {
                // Estimate memory for offsets map + ID list
                offsets.len() * (8 + 32 + 8) // Key + Valid + Offset roughly
            }
        };

        VectorStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            memory_usage,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.vectors.contains(doc_id, field_name)
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        let mut result = Vec::new();
        for &(id, fid) in &self.vector_ids {
            let field = &self.field_dict[fid as usize];
            if id >= start_doc_id
                && id < end_doc_id
                && !self.is_deleted(id)
                && let Some(vec) = self.vectors.get(id, field, self.dimension)?
            {
                result.push((id, field.to_string(), vec));
            }
        }
        Ok(result)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        // One dictionary resolve, then integer compares per record.
        let Some(target) =
            crate::vector::index::format::resolve_field_id(&self.field_dict, field_name)
        else {
            return Ok(Vec::new());
        };
        let mut result = Vec::new();
        for &(id, fid) in &self.vector_ids {
            if fid == target
                && !self.is_deleted(id)
                && let Some(vec) = self.vectors.get(id, field_name, self.dimension)?
            {
                result.push((id, vec));
            }
        }
        Ok(result)
    }

    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self.field_dict.iter().map(|f| f.to_string()).collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(HnswVectorIterator {
            storage: self.vectors.clone(),
            keys: self.vector_ids.clone(),
            field_dict: self.field_dict.clone(),
            current: 0,
            dimension: self.dimension,
            deletion_bitmap: self.deletion_bitmap.clone(),
        }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "hnsw".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1".to_string(),
            build_config: serde_json::json!({}),
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if self.vector_ids.len() != self.vectors.len() {
            errors.push(format!(
                "Mismatch between vector_ids count ({}) and vectors count ({})",
                self.vector_ids.len(),
                self.vectors.len()
            ));
        }

        match &self.vectors {
            VectorStorage::Owned(map) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    let (id, field) = (&id, &field.to_string());
                    if let Some(vector) = map.get(&(*id, field.clone())) {
                        if vector.dimension() != self.dimension {
                            errors.push(format!(
                                "Vector {}:{} has dimension {}, expected {}",
                                id,
                                field,
                                vector.dimension(),
                                self.dimension
                            ));
                        }
                        if !vector.is_valid() {
                            errors.push(format!(
                                "Vector {}:{} contains invalid values (NaN or infinity)",
                                id, field
                            ));
                        }
                    } else {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in storage",
                            id, field
                        ));
                    }
                }
            }
            VectorStorage::OwnedQuantized(pool) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    if !pool.contains(id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in quantized pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedQuantized mode: dimension / NaN checks skipped (int8 storage \
                     guarantees finite values within [offset, offset + 255*scale])"
                        .to_string(),
                );
            }
            VectorStorage::OwnedPq(pool) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    if !pool.contains(id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in PQ pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedPq mode: dimension / NaN checks skipped (codes index into \
                     the trained codebook which is bounded by construction)"
                        .to_string(),
                );
            }
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    if !pool.contains(id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in PQ FastScan pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedPqFastScan mode: dimension / NaN checks skipped (4-bit codes \
                     index into the trained K=16 codebook which is bounded by construction)"
                        .to_string(),
                );
            }
            VectorStorage::OnDemand { offsets, .. } => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    if !offsets.contains_key(&(id, fid)) {
                        errors.push(format!(
                            "Vector {}:{} in ids but missing in storage",
                            id, field
                        ));
                    }
                }
                warnings.push("OnDemand mode: Deep vector validation skipped".to_string());
            }
        }

        // HNSW-specific validation
        if self.m == 0 {
            warnings.push("HNSW parameter M is 0, this may indicate a corrupted index".to_string());
        }
        if self.ef_construction == 0 {
            warnings.push(
                "HNSW parameter ef_construction is 0, this may indicate a corrupted index"
                    .to_string(),
            );
        }

        Ok(ValidationReport {
            repair_suggestions: Vec::new(),
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
}

/// Iterator for HNSW vector index.
struct HnswVectorIterator {
    storage: VectorStorage,
    keys: Vec<(u64, u16)>,
    field_dict: Arc<[Arc<str>]>,
    current: usize,
    dimension: usize,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl VectorIterator for HnswVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        // Use a loop instead of recursion to avoid stack overflow when
        // many consecutive entries are deleted.
        while self.current < self.keys.len() {
            let (doc_id, fid) = self.keys[self.current];
            let field = &self.field_dict[fid as usize];

            // Skip deleted entries
            if let Some(bitmap) = &self.deletion_bitmap
                && bitmap.is_deleted(doc_id)
            {
                self.current += 1;
                continue;
            }

            if let Some(vec) = self.storage.get(doc_id, field, self.dimension)? {
                self.current += 1;
                return Ok(Some((doc_id, field.to_string(), vec)));
            } else {
                return Err(LaurusError::internal(format!(
                    "Vector {}:{} found in keys but missing in storage",
                    doc_id, field
                )));
            }
        }
        Ok(None)
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.current < self.keys.len() {
            let (id, fid) = self.keys[self.current];
            let field = &self.field_dict[fid as usize];
            if id > doc_id || (id == doc_id && field.as_ref() as &str >= field_name) {
                return Ok(true);
            }
            self.current += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.current < self.keys.len() {
            let (id, fid) = self.keys[self.current];
            (id, self.field_dict[fid as usize].to_string())
        } else {
            (u64::MAX, String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }
}

#[cfg(test)]
mod alloc_bound_tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::quantization::ScalarQuantParams;
    use std::io::Write;

    fn storage_with(name: &str, bytes: Vec<u8>) -> Arc<dyn Storage> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output(name).unwrap();
        out.write_all(&bytes).unwrap();
        out.flush_and_sync().unwrap();
        Arc::new(storage)
    }

    fn neutral_header_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        VectorSegmentHeader::scalar_8bit(ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        })
        .write_to(&mut buf)
        .unwrap();
        buf
    }

    /// The HNSW preamble: num_vectors (u64) + dimension/m/ef (u32 each).
    fn preamble(num_vectors: u64, dimension: u32, m: u32, ef: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&num_vectors.to_le_bytes());
        buf.extend_from_slice(&dimension.to_le_bytes());
        buf.extend_from_slice(&m.to_le_bytes());
        buf.extend_from_slice(&ef.to_le_bytes());
        buf
    }

    #[test]
    fn load_rejects_oversized_num_vectors_on_footerless_segment() {
        // The residual exposure for `.hnsw` is a *legacy footer-less* segment:
        // it skips the Issue #786 checksum and reaches the per-vector
        // allocation with an unverified header. A corrupt `num_vectors` over a
        // record-less file must be rejected, never aborted (Issue #806).
        let mut bytes = preamble(u64::MAX, 4, 16, 200);
        bytes.extend_from_slice(&neutral_header_bytes()); // no records, no footer

        let storage = storage_with("corrupt.hnsw", bytes);
        let err = HnswIndexReader::load(storage, "corrupt", DistanceMetric::Cosine)
            .expect_err("oversized num_vectors must be rejected as corruption");
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("num_vectors"), "got: {msg}");
                assert!(msg.contains("corrupted"), "got: {msg}");
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_oversized_graph_node_count_on_footerless_segment() {
        // The graph trails the (empty) vector payload. A corrupt `node_count`
        // must be bounded by the bytes left for the graph, not allocated up
        // front (Issue #806).
        let mut bytes = preamble(0, 4, 16, 200);
        bytes.extend_from_slice(&neutral_header_bytes()); // zero vector records
        bytes.push(1u8); // has_graph = true
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // entry_point (== None)
        bytes.extend_from_slice(&0u32.to_le_bytes()); // max_level
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // node_count (corrupt)

        let storage = storage_with("corrupt_graph.hnsw", bytes);
        let err = HnswIndexReader::load(storage, "corrupt_graph", DistanceMetric::Cosine)
            .expect_err("oversized node_count must be rejected as corruption");
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("node_count"), "got: {msg}");
                assert!(msg.contains("corrupted"), "got: {msg}");
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }
}

/// Format-migration tests for the Issue #686 v1 → v2 (ordinal) graph
/// block. There was no legacy-format read test before this module; the
/// v1 fixtures are hand-written bytes, following the `.delmap` legacy
/// test precedent (`maintenance/deletion.rs`) and the byte builders of
/// `alloc_bound_tests` above. All fixtures are footer-less, which also
/// keeps the legacy no-CRC read path covered.
#[cfg(test)]
mod ordinal_migration_tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::quantization::ScalarQuantParams;
    use crate::vector::index::HnswIndexConfig;
    use crate::vector::index::format::{CURRENT_VERSION, VERSION_FIELD_DICT};
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::index::hnsw::writer::HnswIndexWriter;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
    use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
    use std::io::Write;

    const DIM: usize = 4;

    fn storage_with(name: &str, bytes: Vec<u8>) -> Arc<dyn Storage> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output(name).unwrap();
        out.write_all(&bytes).unwrap();
        out.flush_and_sync().unwrap();
        Arc::new(storage)
    }

    /// The HNSW preamble: num_vectors (u64) + dimension/m/ef (u32 each).
    fn preamble(num_vectors: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&num_vectors.to_le_bytes());
        buf.extend_from_slice(&(DIM as u32).to_le_bytes());
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&200u32.to_le_bytes());
        buf
    }

    /// Neutral (offset 0, scale 1) LVS1 header at the given version.
    fn header_bytes(version: u16) -> Vec<u8> {
        let mut buf = Vec::new();
        VectorSegmentHeader::scalar_8bit(ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        })
        .with_version(version)
        .write_to(&mut buf)
        .unwrap();
        buf
    }

    /// One SQ record with self-consistent meta under the neutral params
    /// (dequantized value == quantized byte), so distances stay finite.
    fn sq_record(doc_id: u64, field: &str, values: [u8; DIM]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&doc_id.to_le_bytes());
        buf.extend_from_slice(&(field.len() as u32).to_le_bytes());
        buf.extend_from_slice(field.as_bytes());
        buf.extend_from_slice(&values);
        let sum_q: u32 = values.iter().map(|&v| v as u32).sum();
        let norm_q: f32 = values
            .iter()
            .map(|&v| (v as f32) * (v as f32))
            .sum::<f32>()
            .sqrt();
        buf.extend_from_slice(&sum_q.to_le_bytes());
        buf.extend_from_slice(&norm_q.to_le_bytes());
        buf
    }

    /// A v1 graph block over doc ids `[10, 20, 30]`: entry 10, one
    /// layer each, ring adjacency expressed as doc ids (u64).
    fn v1_graph_block() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1u8); // has_graph
        buf.extend_from_slice(&10u64.to_le_bytes()); // entry_point doc id
        buf.extend_from_slice(&0u32.to_le_bytes()); // max_level
        buf.extend_from_slice(&3u64.to_le_bytes()); // node_count
        for (doc_id, neighbors) in [
            (10u64, [20u64, 30u64]),
            (20u64, [10u64, 30u64]),
            (30u64, [10u64, 20u64]),
        ] {
            buf.extend_from_slice(&doc_id.to_le_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // layer_count
            buf.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
            for n in neighbors {
                buf.extend_from_slice(&n.to_le_bytes());
            }
        }
        buf
    }

    fn three_doc_records() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&sq_record(10, "f", [100, 1, 1, 1]));
        buf.extend_from_slice(&sq_record(20, "f", [1, 100, 1, 1]));
        buf.extend_from_slice(&sq_record(30, "f", [1, 1, 100, 1]));
        buf
    }

    /// v3 LVS1 header bytes: neutral SQ params + the given dictionary.
    fn v3_header_bytes(dict: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        VectorSegmentHeader::scalar_8bit(ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        })
        .with_version(VERSION_FIELD_DICT)
        .with_field_dict(dict.iter().map(|s| s.to_string()).collect())
        .write_to(&mut buf)
        .unwrap();
        buf
    }

    /// One v3 SQ record: `[doc_id u64][field_id u16][int8][meta]`.
    fn sq_record_v3(doc_id: u64, field_id: u16, values: [u8; DIM]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&doc_id.to_le_bytes());
        buf.extend_from_slice(&field_id.to_le_bytes());
        buf.extend_from_slice(&values);
        let sum_q: u32 = values.iter().map(|&v| v as u32).sum();
        let norm_q: f32 = values
            .iter()
            .map(|&v| (v as f32) * (v as f32))
            .sum::<f32>()
            .sqrt();
        buf.extend_from_slice(&sum_q.to_le_bytes());
        buf.extend_from_slice(&norm_q.to_le_bytes());
        buf
    }

    /// A v2-encoding (ordinal) graph block over 3 nodes: entry ordinal 0,
    /// one layer each, ring adjacency. Valid for any version >= 2 fixture.
    fn ordinal_graph_block() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1u8); // has_graph
        buf.extend_from_slice(&0u32.to_le_bytes()); // entry ordinal
        buf.extend_from_slice(&0u32.to_le_bytes()); // max_level
        buf.extend_from_slice(&3u32.to_le_bytes()); // node_count
        for neighbors in [[1u32, 2u32], [0, 2], [0, 1]] {
            buf.extend_from_slice(&1u32.to_le_bytes()); // layer_count
            buf.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
            for n in neighbors {
                buf.extend_from_slice(&n.to_le_bytes());
            }
        }
        buf
    }

    fn v3_fixture_bytes() -> Vec<u8> {
        let mut bytes = preamble(3);
        bytes.extend_from_slice(&v3_header_bytes(&["f"]));
        bytes.extend_from_slice(&sq_record_v3(10, 0, [100, 1, 1, 1]));
        bytes.extend_from_slice(&sq_record_v3(20, 0, [1, 100, 1, 1]));
        bytes.extend_from_slice(&sq_record_v3(30, 0, [1, 1, 100, 1]));
        bytes.extend_from_slice(&ordinal_graph_block());
        bytes
    }

    fn v1_fixture_bytes() -> Vec<u8> {
        let mut bytes = preamble(3);
        bytes.extend_from_slice(&header_bytes(CURRENT_VERSION));
        bytes.extend_from_slice(&three_doc_records());
        bytes.extend_from_slice(&v1_graph_block());
        bytes
    }

    #[test]
    fn reads_v3_fixture_and_searches() {
        let storage = storage_with("v3_fixture.hnsw", v3_fixture_bytes());
        let reader = HnswIndexReader::load(storage, "v3_fixture", DistanceMetric::Cosine).unwrap();

        let graph = reader.graph.as_ref().expect("v3 graph must load");
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.entry_point(), Some(0));
        // Single-field, unique ids → identity ord→pos mapping.
        assert!(reader.field_ord_to_pos("f").is_none());

        let searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
        let results = searcher
            .search(
                &VectorIndexQuery::new(Vector::new(vec![1.0, 1.0, 100.0, 1.0]))
                    .top_k(3)
                    .field_name("f".to_string()),
            )
            .unwrap();
        let mut ids: Vec<u64> = results.results.iter().map(|r| r.doc_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![10, 20, 30]);
        assert_eq!(results.results[0].doc_id, 30);
    }

    #[test]
    fn rejects_v3_record_field_id_out_of_range() {
        let mut bytes = preamble(3);
        bytes.extend_from_slice(&v3_header_bytes(&["f"]));
        bytes.extend_from_slice(&sq_record_v3(10, 0, [100, 1, 1, 1]));
        bytes.extend_from_slice(&sq_record_v3(20, 7, [1, 100, 1, 1])); // id 7 >= dict len 1
        bytes.extend_from_slice(&sq_record_v3(30, 0, [1, 1, 100, 1]));
        bytes.extend_from_slice(&ordinal_graph_block());

        let storage = storage_with("bad_fid.hnsw", bytes);
        let err = HnswIndexReader::load(storage, "bad_fid", DistanceMetric::Cosine)
            .expect_err("out-of-range field_id must be rejected");
        assert!(
            err.to_string().contains("out of dictionary range"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_v3_empty_dict_with_records() {
        let mut bytes = preamble(1);
        bytes.extend_from_slice(&v3_header_bytes(&[]));
        bytes.extend_from_slice(&sq_record_v3(10, 0, [100, 1, 1, 1]));
        bytes.push(0u8); // has_graph = false

        let storage = storage_with("empty_dict.hnsw", bytes);
        let err = HnswIndexReader::load(storage, "empty_dict", DistanceMetric::Cosine)
            .expect_err("records referencing an empty dictionary must be rejected");
        assert!(
            err.to_string().contains("out of dictionary range"),
            "got: {err}"
        );
    }

    #[test]
    fn v3_writer_output_has_exact_size_and_v1_delta() {
        // Deterministic headline gate for #633: byte-exact size formula
        // plus the delta a v1 layout would have produced.
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = HnswIndexConfig {
            dimension: DIM,
            m: 4,
            ef_construction: 16,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let field = "embedding";
        let n = 5u64;
        let mut writer = HnswIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            "sized",
            Arc::clone(&storage),
        )
        .unwrap();
        let vectors: Vec<(u64, String, Vector)> = (0..n)
            .map(|i| {
                (
                    i,
                    field.to_string(),
                    Vector::new(vec![i as f32 + 1.0, 1.0, 1.0, 1.0]),
                )
            })
            .collect();
        writer.add_vectors(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();

        let input = storage.open_input("sized.hnsw").unwrap();
        let actual = input.size().unwrap();

        // Reload to measure the graph block exactly (its size depends on
        // the built topology): total = preamble(20) + header(24 + dict)
        // + records + graph + footer(8).
        let reader =
            HnswIndexReader::load(Arc::clone(&storage), "sized", DistanceMetric::Cosine).unwrap();
        let graph = reader.graph.as_ref().unwrap();
        let mut graph_block = 1u64 + 4 + 4 + 4; // has_graph + entry + max_level + node_count
        for (_, layers) in graph.iter_nodes() {
            graph_block += 4; // layer_count
            for neighbors in layers {
                graph_block += 4 + 4 * neighbors.len() as u64;
            }
        }
        let dict_bytes = 2 + (2 + field.len()) as u64;
        let record_bytes = n * (10 + DIM as u64 + 8);
        let expected = 20 + (24 + dict_bytes) + record_bytes + graph_block + 8;
        assert_eq!(actual, expected, "v3 .hnsw file size must be byte-exact");

        // v1 would have spent (4 + field.len()) per record on the name and
        // carried no dictionary: delta = n*(len-2+4) - (2 + (2+len)).
        let v1_records = n * (12 + field.len() as u64 + DIM as u64 + 8);
        let saved = (v1_records + 24) - (record_bytes + 24 + dict_bytes);
        assert_eq!(
            saved,
            n * (field.len() as u64 + 2) - dict_bytes,
            "the v1-vs-v3 record-section delta formula must hold"
        );
        assert!(saved > 0, "v3 must be smaller for n=5, k=9");
    }

    #[test]
    fn reads_v1_graph_fixture_as_ordinals_and_searches() {
        let storage = storage_with("v1_fixture.hnsw", v1_fixture_bytes());
        let reader = HnswIndexReader::load(storage, "v1_fixture", DistanceMetric::Cosine).unwrap();

        let graph = reader.graph.as_ref().expect("v1 graph must load");
        assert_eq!(graph.node_count(), 3);
        // Ordinals are doc-id ranks: 10 → 0, 20 → 1, 30 → 2.
        assert_eq!(graph.entry_point(), Some(0));
        assert_eq!(graph.doc_id(1), 20);
        assert_eq!(graph.neighbors(0, 0), Some(&[1u32, 2][..]));
        assert_eq!(graph.neighbors(2, 0), Some(&[0u32, 1][..]));
        // Single-field, unique ids → identity ord→pos mapping.
        assert!(reader.field_ord_to_pos("f").is_none());

        let searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
        let results = searcher
            .search(
                &VectorIndexQuery::new(Vector::new(vec![100.0, 1.0, 1.0, 1.0]))
                    .top_k(3)
                    .field_name("f".to_string()),
            )
            .unwrap();
        let mut ids: Vec<u64> = results.results.iter().map(|r| r.doc_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![10, 20, 30], "all three docs must be reachable");
        assert_eq!(
            results.results[0].doc_id, 10,
            "nearest neighbour of doc 10's own vector must rank first"
        );
    }

    #[test]
    fn writer_loads_v1_and_rewrites_v3_that_reader_loads() {
        let storage = storage_with("upgrade.hnsw", v1_fixture_bytes());
        let config = HnswIndexConfig {
            dimension: DIM,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        // Writer reads the v1 segment (compaction/append entry point)…
        let mut writer = HnswIndexWriter::load(
            config,
            VectorIndexWriterConfig::default(),
            Arc::clone(&storage),
            "upgrade",
        )
        .unwrap();
        // …and rewrites it, emitting the v2 ordinal format. `load` leaves
        // the writer appendable, so finalize first (no-op append here).
        writer.finalize().unwrap();
        writer.write().unwrap();

        // The rewritten header must carry the v2 version stamp (bytes
        // 4-5 of the LVS1 header, which follows the 20-byte preamble).
        let mut input = storage.open_input("upgrade.hnsw").unwrap();
        let mut head = vec![0u8; 26];
        input.read_exact(&mut head).unwrap();
        assert_eq!(
            u16::from_le_bytes([head[24], head[25]]),
            VERSION_FIELD_DICT,
            "rewritten segment must stamp the v3 header version (Issue #633)"
        );

        let reader = HnswIndexReader::load(storage, "upgrade", DistanceMetric::Cosine).unwrap();
        let graph = reader.graph.as_ref().expect("v2 graph must load");
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.entry_point().map(|o| graph.doc_id(o)), Some(10));
        assert_eq!(graph.neighbors(0, 0), Some(&[1u32, 2][..]));

        let searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
        let results = searcher
            .search(
                &VectorIndexQuery::new(Vector::new(vec![1.0, 100.0, 1.0, 1.0]))
                    .top_k(3)
                    .field_name("f".to_string()),
            )
            .unwrap();
        assert_eq!(results.results[0].doc_id, 20);
    }

    #[test]
    fn rejects_v2_out_of_range_neighbor_ordinal() {
        let mut bytes = preamble(3);
        bytes.extend_from_slice(&header_bytes(VERSION_ORDINAL_GRAPH));
        bytes.extend_from_slice(&three_doc_records());
        bytes.push(1u8); // has_graph
        bytes.extend_from_slice(&0u32.to_le_bytes()); // entry ordinal
        bytes.extend_from_slice(&0u32.to_le_bytes()); // max_level
        bytes.extend_from_slice(&3u32.to_le_bytes()); // node_count
        for _ in 0..3 {
            bytes.extend_from_slice(&1u32.to_le_bytes()); // layer_count
            bytes.extend_from_slice(&1u32.to_le_bytes()); // neighbor_count
            bytes.extend_from_slice(&7u32.to_le_bytes()); // ordinal out of range
        }

        let storage = storage_with("bad_ord.hnsw", bytes);
        let err = HnswIndexReader::load(storage, "bad_ord", DistanceMetric::Cosine)
            .expect_err("out-of-range neighbour ordinal must be rejected");
        assert!(err.to_string().contains("neighbour ordinal"), "got: {err}");
    }

    #[test]
    fn rejects_v2_node_count_mismatch_with_records() {
        let mut bytes = preamble(3);
        bytes.extend_from_slice(&header_bytes(VERSION_ORDINAL_GRAPH));
        bytes.extend_from_slice(&three_doc_records());
        bytes.push(1u8); // has_graph
        bytes.extend_from_slice(&0u32.to_le_bytes()); // entry ordinal
        bytes.extend_from_slice(&0u32.to_le_bytes()); // max_level
        bytes.extend_from_slice(&1u32.to_le_bytes()); // node_count ≠ 3 unique ids

        let storage = storage_with("bad_count.hnsw", bytes);
        let err = HnswIndexReader::load(storage, "bad_count", DistanceMetric::Cosine)
            .expect_err("node_count mismatch must be rejected");
        assert!(err.to_string().contains("node_count"), "got: {err}");
    }

    #[test]
    fn v2_empty_graph_block_loads_and_search_returns_empty() {
        let mut bytes = preamble(0);
        bytes.extend_from_slice(&header_bytes(VERSION_ORDINAL_GRAPH));
        bytes.push(1u8); // has_graph
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // entry = None
        bytes.extend_from_slice(&0u32.to_le_bytes()); // max_level
        bytes.extend_from_slice(&0u32.to_le_bytes()); // node_count

        let storage = storage_with("empty_graph.hnsw", bytes);
        let reader = HnswIndexReader::load(storage, "empty_graph", DistanceMetric::Cosine).unwrap();
        let graph = reader.graph.as_ref().expect("empty graph must load");
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.entry_point(), None);

        let searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
        let results = searcher
            .search(
                &VectorIndexQuery::new(Vector::new(vec![1.0; DIM]))
                    .top_k(3)
                    .field_name("f".to_string()),
            )
            .unwrap();
        assert!(results.results.is_empty());
    }

    #[test]
    fn multi_field_segment_builds_ord_to_pos_and_searches_per_field() {
        // Legacy multi-field shape: two fields sharing doc ids, so
        // ordinal ≠ per-field pool position and the non-identity
        // `ord_to_pos` path must engage.
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = HnswIndexConfig {
            dimension: DIM,
            m: 4,
            ef_construction: 16,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut writer = HnswIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            "multi_field",
            Arc::clone(&storage),
        )
        .unwrap();
        // Distinct doc ids per field — the shape real multi-field
        // segments have (the writer upserts by doc_id, so one doc id
        // cannot carry two fields' records).
        let mut vectors = Vec::new();
        for i in 0..8u64 {
            let t = i as f32;
            vectors.push((
                i,
                "a".to_string(),
                Vector::new(vec![t + 1.0, 1.0, 1.0, 1.0]),
            ));
            vectors.push((
                8 + i,
                "b".to_string(),
                Vector::new(vec![1.0, t + 1.0, 1.0, 1.0]),
            ));
        }
        writer.add_vectors(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();

        let reader = HnswIndexReader::load(storage, "multi_field", DistanceMetric::Cosine).unwrap();
        assert!(
            reader.field_ord_to_pos("a").is_some(),
            "multi-field segment must build an ord_to_pos table"
        );
        let graph = reader.graph.as_ref().expect("graph must load");
        assert_eq!(graph.node_count(), 16, "one node per unique doc id");

        let searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
        for field in ["a", "b"] {
            let results = searcher
                .search(
                    &VectorIndexQuery::new(Vector::new(vec![1.0; DIM]))
                        .top_k(4)
                        .field_name(field.to_string()),
                )
                .unwrap();
            assert!(
                !results.results.is_empty(),
                "field {field} must return hits through the ord_to_pos path"
            );
            for r in &results.results {
                assert!(
                    r.distance != f32::MAX,
                    "admitted results must carry finite distances"
                );
                // Field routing must hold: docs of the other field are
                // u32::MAX in the table -> f32::MAX -> dropped (#676).
                let in_field = if field == "a" {
                    r.doc_id < 8
                } else {
                    r.doc_id >= 8
                };
                assert!(in_field, "doc {} leaked into field {field}", r.doc_id);
            }
        }
    }
}
