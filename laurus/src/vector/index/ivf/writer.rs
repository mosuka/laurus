//! IVF (Inverted File) index builder for memory-efficient search.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::util::alloc_bounds::checked_capacity;
use crate::vector::core::quantization::ScalarQuantParams;
use crate::vector::core::vector::Vector;
use crate::vector::index::IvfIndexConfig;
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::format::{
    QuantHeader, VERSION_FIELD_DICT, VectorSegmentHeader, build_field_dict, record_prefix_size,
};
use crate::vector::index::quantized_io::{
    quantize_segment, quantized_record_payload_size, read_dequantized_vector,
    write_quantized_record,
};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use serde::{Deserialize, Serialize};

/// Fixed seed for the IVF k-means RNG (Issue #847).
///
/// k-means++ initialization and cluster splitting need randomness only
/// for their statistical properties, not unpredictability, so training
/// uses a deterministic generator seeded with this constant — the same
/// rationale as the HNSW level RNG (Issue #841 / #842, following
/// Lucene's `DEFAULT_RAND_SEED = 42`). Same input → same centroids and
/// assignments (bitwise on the serial path; parallel accumulation from
/// Issue #843 may regroup f32 sums), making builds reproducible and
/// construction benchmarks stable.
const KMEANS_RNG_SEED: u64 = 42;

#[derive(Debug)]
/// Builder for IVF vector indexes (memory-efficient search).
pub struct IvfIndexWriter {
    index_config: IvfIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    path: String,
    centroids: Vec<Vector>,                          // Cluster centroids
    inverted_lists: Vec<Vec<(u64, String, Vector)>>, // Inverted lists for each cluster
    vectors: Vec<(u64, String, Vector)>,             // All vectors (used during construction)
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
    next_vec_id: u64,
    /// The cluster-count ceiling the caller configured (via
    /// [`IvfIndexConfig::n_clusters`], [`Self::with_ivf_params`], or
    /// [`Self::set_expected_vector_count`]) — never mutated by training.
    /// `train_centroids` clamps the *effective* `index_config.n_clusters`
    /// down to `min(configured_n_clusters, vectors.len())` for training and
    /// serialization, but this field is what that clamp is always computed
    /// from, so the effective count recovers (rather than staying
    /// permanently shrunk) once the corpus grows past it again (Issue #889
    /// PR-5).
    configured_n_clusters: usize,
}

impl IvfIndexWriter {
    /// Create a new IVF index builder.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index configuration
    /// * `n_clusters` - Number of clusters (cells) to create (typical: sqrt(n_vectors))
    /// * `n_probe` - Number of clusters to search (typical: 1-10, higher = more accurate but slower)
    pub fn new(
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
    ) -> Result<Self> {
        let configured_n_clusters = index_config.n_clusters;
        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            path: path.into(),

            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
            configured_n_clusters,
        })
    }

    /// Create a new IVF index builder with storage.
    /// Create a new IVF index builder with storage.
    ///
    /// If an existing index file is found on disk, its vectors are loaded
    /// into the writer so that the next commit preserves them. This
    /// prevents data loss across multiple commit cycles.
    pub fn with_storage(
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let path = path.into();
        let file_name = format!("{}.ivf", path);
        if storage.file_exists(&file_name) {
            return Self::load(index_config, writer_config, storage, &path);
        }

        let configured_n_clusters = index_config.n_clusters;
        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path,
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
            configured_n_clusters,
        })
    }

    /// Convert this writer into a doc-centric field writer adapter.
    pub fn into_field_writer(self, field_name: impl Into<String>) -> LegacyVectorFieldWriter<Self> {
        LegacyVectorFieldWriter::new(field_name, self)
    }

    /// Load an existing IVF index from storage.
    pub fn load(
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.ivf", path);
        let mut input = storage.open_input(&file_name)?;

        // Ground truth for bounding allocations sized from unverified header
        // counts below (Issue #806). This writer load path runs no checksum
        // verification, so every count is unverified.
        let file_size = input.size()?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut n_clusters_buf = [0u8; 4];
        input.read_exact(&mut n_clusters_buf)?;
        let n_clusters = u32::from_le_bytes(n_clusters_buf) as usize;

        let mut n_probe_buf = [0u8; 4];
        input.read_exact(&mut n_probe_buf)?;
        let _n_probe = u32::from_le_bytes(n_probe_buf) as usize;

        if dimension != index_config.dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read centroids. Each centroid serializes as `dimension` f32 values
        // (4 bytes each), so bounding `n_clusters` by that stride also bounds
        // each per-centroid `vec![0.0f32; dimension]` allocation (Issue #806).
        let centroids_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        checked_capacity(
            n_clusters,
            (dimension as u64).saturating_mul(4),
            centroids_remaining,
            "ivf centroids",
        )?;
        let mut centroids = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }
            centroids.push(Vector::new(values));
        }

        // Read the Issue #481 Stage 1 vector segment header (LVS1).
        // Pre-Stage-1 segments are rejected with IncompatibleFormat.
        // Matched by reference so `header` (version + field dictionary,
        // Issue #633) stays alive for the record parse below.
        // Issue #921: pass the bytes physically left in the file so the
        // header's PQ codebook allocation is bounded before it reserves.
        let header_available =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let header = VectorSegmentHeader::read_from(&mut input, header_available)?;
        let params = match &header.quant {
            QuantHeader::Scalar8Bit(p) => *p,
            QuantHeader::ProductQuantization { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "Product quantization (Issue #481 Stage 3) is HNSW-only; \
                     the IVF writer does not support PQ segments yet"
                        .to_string(),
                ));
            }
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "PQ FastScan (#695) is HNSW-only; the IVF writer does not \
                     support PQ FastScan segments"
                        .to_string(),
                ));
            }
        };

        // Read inverted lists, dequantizing each record back to f32
        // for the in-memory writer state.
        // Bytes left for the inverted-list section (Issue #806). Each record
        // is at least doc_id (8) + field_name_len (4) + the fixed quantized
        // payload (dim int8 + 8 meta); each cluster serializes at least its
        // list_size (4 bytes).
        let lists_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let record_stride =
            record_prefix_size(header.version) + quantized_record_payload_size(dimension) as u64;
        checked_capacity(n_clusters, 4, lists_remaining, "ivf cluster lists")?;
        let mut inverted_lists = vec![Vec::new(); n_clusters];
        for list in &mut inverted_lists {
            let mut list_size_buf = [0u8; 4];
            input.read_exact(&mut list_size_buf)?;
            let list_size = u32::from_le_bytes(list_size_buf) as usize;
            checked_capacity(list_size, record_stride, lists_remaining, "ivf list_size")?;

            for _ in 0..list_size {
                let mut doc_id_buf = [0u8; 8];
                input.read_exact(&mut doc_id_buf)?;
                let doc_id = u64::from_le_bytes(doc_id_buf);

                // Field reference: dictionary id (v3+) or inline name.
                let field_name =
                    header.read_record_field(&mut input, lists_remaining, "ivf field_name_len")?;

                // Read quantized payload + dequantize.
                let values = read_dequantized_vector(&mut input, dimension, &params)?;

                list.push((doc_id, field_name, Vector::new(values)));
            }
        }

        // Reconstruct vectors from inverted lists. `num_vectors` is a separate
        // header field and is still unverified, so bound it against the same
        // record section before reserving (Issue #806).
        checked_capacity(
            num_vectors,
            record_stride,
            lists_remaining,
            "ivf num_vectors",
        )?;
        let mut vectors = Vec::with_capacity(num_vectors);
        for list in &inverted_lists {
            vectors.extend(list.iter().cloned());
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        let configured_n_clusters = index_config.n_clusters;
        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path: path.to_string(),

            centroids,
            inverted_lists,
            vectors,
            is_finalized: true,
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
            configured_n_clusters,
        })
    }

    /// Set IVF-specific parameters.
    pub fn with_ivf_params(mut self, n_clusters: usize, n_probe: usize) -> Self {
        self.index_config.n_clusters = n_clusters;
        self.configured_n_clusters = n_clusters;
        self.index_config.n_probe = n_probe;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    ///
    /// Also updates the cluster-count ceiling ([`Self::with_ivf_params`]'s
    /// `n_clusters`, or the configured default) to a size-appropriate
    /// default, since the caller is stating a new expectation about the
    /// eventual corpus size.
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
        // Adjust number of clusters based on dataset size
        self.index_config.n_clusters = Self::compute_default_clusters(count);
        self.configured_n_clusters = self.index_config.n_clusters;
    }

    /// Compute default number of clusters based on dataset size.
    fn compute_default_clusters(n_vectors: usize) -> usize {
        // Rule of thumb: sqrt(n_vectors), with reasonable min/max bounds
        let clusters = (n_vectors as f64).sqrt() as usize;
        clusters.clamp(10, 10000)
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, String, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, _field_name, vector) in vectors {
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
    fn normalize_vectors(&self, vectors: &mut [(u64, String, Vector)]) {
        if !self.index_config.normalize_vectors {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
            return;
        }

        for (_, _, vector) in vectors {
            vector.normalize();
        }
    }

    /// Train centroids using k-means clustering.
    fn train_centroids(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            // Issue #889 PR-6: a segmented force-merge can legitimately
            // reduce a merge window to zero surviving vectors (every
            // document in it was logically deleted). Represent that as
            // zero clusters rather than erroring — `build_inverted_lists`,
            // `write`, and every reader/searcher already handle
            // `n_clusters == 0` gracefully (their loops just run zero
            // times), so there is nothing downstream left to special-case.
            self.index_config.n_clusters = 0;
            self.centroids.clear();
            return Ok(());
        }

        // Issue #889 PR-5: clamp to the available corpus instead of
        // hard-erroring. `configured_n_clusters` is the ceiling the caller
        // asked for and is never mutated by training, so the effective
        // count recovers as the corpus grows across successive
        // `add_vectors` calls rather than staying permanently shrunk once a
        // small commit clamps it down.
        self.index_config.n_clusters = self.configured_n_clusters.min(self.vectors.len()).max(1);

        // Initialize centroids with k-means++
        self.init_centroids_kmeans_plus_plus()?;

        // Run k-means iterations
        let max_iterations = 100;
        let convergence_threshold = 1e-6;

        for _iteration in 0..max_iterations {
            let old_centroids = self.centroids.clone();

            // Assign vectors to clusters
            let assignments = self.assign_vectors_to_clusters();

            // Update centroids
            self.update_centroids(&assignments)?;

            // Check for convergence
            let convergence = self.compute_convergence(&old_centroids);
            if convergence < convergence_threshold {
                break;
            }
        }

        Ok(())
    }

    /// Initialize centroids using k-means++ algorithm.
    fn init_centroids_kmeans_plus_plus(&mut self) -> Result<()> {
        use rand::prelude::*;
        // Deterministic training RNG (Issue #847); see KMEANS_RNG_SEED.
        let mut rng = rand::rngs::StdRng::seed_from_u64(KMEANS_RNG_SEED);

        self.centroids.clear();

        // Choose first centroid randomly
        let first_idx = rng.random_range(0..self.vectors.len());
        self.centroids.push(self.vectors[first_idx].2.clone());

        // Running minimum distance from each vector to its nearest
        // chosen centroid (Issue #622). The textbook k-means++
        // formulation folds only the NEWEST centroid into the running
        // minimum on each pick — O(n·d) per pick, O(n·k·d) total —
        // replacing the previous full recompute over all chosen
        // centroids per pick, which was O(n·k²·d). `f32::min` over the
        // same distance values is exact, so the per-pick weights (and
        // therefore the RNG sampling sequence and the chosen centroids)
        // are bit-identical to the previous implementation.
        let mut min_dists: Vec<f32> = vec![f32::INFINITY; self.vectors.len()];

        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..self.index_config.n_clusters {
            // Fold the most recently chosen centroid into the running
            // minima (the only centroid the previous minima have not
            // seen yet).
            let newest = self
                .centroids
                .last()
                .expect("k-means++ always has at least the first centroid")
                .clone();
            self.fold_min_dists(&newest, &mut min_dists);

            // Weights are accumulated serially in vector order so the
            // floating-point sums match the previous implementation.
            let mut distances = Vec::with_capacity(self.vectors.len());
            let mut total_weight = 0.0;
            for &min_dist in &min_dists {
                let weight = min_dist * min_dist;
                distances.push(weight);
                total_weight += weight;
            }

            if total_weight == 0.0 {
                // Fallback to random selection
                let idx = rng.random_range(0..self.vectors.len());
                self.centroids.push(self.vectors[idx].2.clone());
                continue;
            }

            // Select next centroid based on weighted probability
            let target = rng.random::<f32>() * total_weight;
            let mut cumsum = 0.0;
            let mut selected = false;

            for (i, &weight) in distances.iter().enumerate() {
                cumsum += weight;
                if cumsum >= target {
                    self.centroids.push(self.vectors[i].2.clone());
                    selected = true;
                    break;
                }
            }

            // Fallback: floating-point precision may prevent cumsum from reaching target
            if !selected {
                self.centroids
                    .push(self.vectors[self.vectors.len() - 1].2.clone());
            }
        }

        Ok(())
    }

    /// Fold `centroid` into the per-vector running minimum distances
    /// used by k-means++ initialization (Issue #622).
    ///
    /// Runs the per-vector scan in parallel under the same gate as
    /// [`Self::assign_vectors_to_clusters`] (`parallel_build` and more
    /// than 1000 buffered vectors; always serial on wasm32). Each
    /// element update is independent, so the parallel result is
    /// bit-identical to the serial one.
    ///
    /// # Arguments
    ///
    /// * `centroid` - The newest chosen centroid.
    /// * `min_dists` - Per-vector running minima, updated in place.
    fn fold_min_dists(&self, centroid: &Vector, min_dists: &mut [f32]) {
        let metric = self.index_config.distance_metric;
        // Distance errors contribute `INFINITY`, which leaves the
        // running minimum unchanged — the same effect the previous
        // implementation's `unwrap_or(f32::INFINITY)` had.
        let fold = |(min_dist, (_, _, vector)): (&mut f32, &(u64, String, Vector))| {
            let dist = metric
                .distance(&vector.data, &centroid.data)
                .unwrap_or(f32::INFINITY);
            *min_dist = min_dist.min(dist);
        };

        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build && self.vectors.len() as u64 > 1000 {
            min_dists
                .par_iter_mut()
                .zip(self.vectors.par_iter())
                .for_each(fold);
            return;
        }

        min_dists.iter_mut().zip(self.vectors.iter()).for_each(fold);
    }

    /// Assign each vector to its nearest cluster.
    fn assign_vectors_to_clusters(&self) -> Vec<usize> {
        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build && self.vectors.len() as u64 > 1000 {
            return self
                .vectors
                .par_iter()
                .map(|(_, _, vector)| self.find_nearest_centroid(vector))
                .collect();
        }

        self.vectors
            .iter()
            .map(|(_, _, vector)| self.find_nearest_centroid(vector))
            .collect()
    }

    /// Find the index of the nearest centroid for a vector.
    fn find_nearest_centroid(&self, vector: &Vector) -> usize {
        let mut best_cluster = 0;
        let mut best_distance = f32::INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            if let Ok(distance) = self
                .index_config
                .distance_metric
                .distance(&vector.data, &centroid.data)
                && distance < best_distance
            {
                best_distance = distance;
                best_cluster = i;
            }
        }

        best_cluster
    }

    /// Update centroids based on cluster assignments.
    fn update_centroids(&mut self, assignments: &[usize]) -> Result<()> {
        let (cluster_sums, cluster_counts) = self.accumulate_cluster_sums(assignments);

        // Compute new centroids as averages
        for (i, (sum, count)) in cluster_sums.iter().zip(cluster_counts.iter()).enumerate() {
            if *count == 0 {
                // Keep the old centroid if no vectors assigned
                continue;
            }

            let centroid_data: Vec<f32> = sum.iter().map(|&s| s / *count as f32).collect();

            self.centroids[i] = Vector::new(centroid_data);
        }

        Ok(())
    }

    /// Accumulate per-cluster `(sums, counts)` over the buffered
    /// vectors (Issue #622).
    ///
    /// Runs a rayon `fold`/`reduce` with per-thread partial sums under
    /// the same gate as [`Self::assign_vectors_to_clusters`]
    /// (`parallel_build` and more than 1000 buffered vectors; always
    /// serial on wasm32). The parallel path regroups f32 additions, so
    /// centroid low-order bits may differ from the serial order —
    /// acceptable because k-means++ initialization is already
    /// RNG-nondeterministic across runs.
    ///
    /// # Arguments
    ///
    /// * `assignments` - Cluster index per buffered vector (parallel
    ///   array to `self.vectors`).
    ///
    /// # Returns
    ///
    /// `(cluster_sums, cluster_counts)`: per-cluster component sums
    /// (`n_clusters × dimension`) and member counts.
    fn accumulate_cluster_sums(&self, assignments: &[usize]) -> (Vec<Vec<f32>>, Vec<usize>) {
        let dim = self.index_config.dimension;
        let n_clusters = self.index_config.n_clusters;

        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build && self.vectors.len() as u64 > 1000 {
            return self
                .vectors
                .par_iter()
                .zip(assignments.par_iter())
                .fold(
                    || {
                        (
                            vec![vec![0.0_f32; dim]; n_clusters],
                            vec![0_usize; n_clusters],
                        )
                    },
                    |(mut sums, mut counts), ((_, _, vector), &cluster)| {
                        counts[cluster] += 1;
                        for (j, &value) in vector.data.iter().enumerate() {
                            sums[cluster][j] += value;
                        }
                        (sums, counts)
                    },
                )
                .reduce(
                    || {
                        (
                            vec![vec![0.0_f32; dim]; n_clusters],
                            vec![0_usize; n_clusters],
                        )
                    },
                    |(mut sums_a, mut counts_a), (sums_b, counts_b)| {
                        for (sum_a, sum_b) in sums_a.iter_mut().zip(sums_b) {
                            for (a, b) in sum_a.iter_mut().zip(sum_b) {
                                *a += b;
                            }
                        }
                        for (a, b) in counts_a.iter_mut().zip(counts_b) {
                            *a += b;
                        }
                        (sums_a, counts_a)
                    },
                );
        }

        let mut cluster_sums = vec![vec![0.0_f32; dim]; n_clusters];
        let mut cluster_counts = vec![0_usize; n_clusters];
        for (i, (_, _, vector)) in self.vectors.iter().enumerate() {
            let cluster = assignments[i];
            cluster_counts[cluster] += 1;
            for (j, &value) in vector.data.iter().enumerate() {
                cluster_sums[cluster][j] += value;
            }
        }
        (cluster_sums, cluster_counts)
    }

    /// Compute convergence metric between old and new centroids.
    fn compute_convergence(&self, old_centroids: &[Vector]) -> f32 {
        if old_centroids.len() != self.centroids.len() {
            return f32::INFINITY;
        }

        let mut total_movement = 0.0;

        for (old, new) in old_centroids.iter().zip(self.centroids.iter()) {
            if let Ok(distance) = self
                .index_config
                .distance_metric
                .distance(&old.data, &new.data)
            {
                total_movement += distance;
            }
        }

        total_movement / self.centroids.len() as f32
    }

    /// Build inverted lists by assigning vectors to clusters.
    fn build_inverted_lists(&mut self) -> Result<()> {
        self.inverted_lists = vec![Vec::new(); self.index_config.n_clusters];

        for (doc_id, field_name, vector) in &self.vectors {
            let cluster = self.find_nearest_centroid(vector);
            self.inverted_lists[cluster].push((*doc_id, field_name.clone(), vector.clone()));
        }

        // Sort each inverted list by document ID
        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build {
            self.inverted_lists.par_iter_mut().for_each(|list| {
                list.sort_by_key(|(doc_id, _, _)| *doc_id);
            });
            return Ok(());
        }

        for list in &mut self.inverted_lists {
            list.sort_by_key(|(doc_id, _, _)| *doc_id);
        }

        Ok(())
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

    /// Get IVF parameters.
    pub fn ivf_params(&self) -> (usize, usize) {
        (self.index_config.n_clusters, self.index_config.n_probe)
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vector] {
        &self.centroids
    }

    /// Get inverted lists.
    pub fn inverted_lists(&self) -> &[Vec<(u64, String, Vector)>] {
        &self.inverted_lists
    }
}

type SplitClusterResult = (
    Vector,
    Vec<(u64, String, Vector)>,
    Vector,
    Vec<(u64, String, Vector)>,
);

/// Statistics for a single IVF cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    pub cluster_id: usize,
    pub count: usize,
    pub avg_distance: f32,
}

impl IvfIndexWriter {
    /// Get statistics for each cluster in the index.
    pub fn get_cluster_stats(&self) -> Vec<ClusterStats> {
        if !self.is_finalized || self.centroids.is_empty() {
            return Vec::new();
        }

        let mut stats = Vec::with_capacity(self.centroids.len());

        for (i, (centroid, list)) in self
            .centroids
            .iter()
            .zip(self.inverted_lists.iter())
            .enumerate()
        {
            let count = list.len();
            let total_dist: f32 = list
                .iter()
                .map(|(_, _, vec)| {
                    self.index_config
                        .distance_metric
                        .distance(&vec.data, &centroid.data)
                        .unwrap_or(0.0)
                })
                .sum();

            stats.push(ClusterStats {
                cluster_id: i,
                count,
                avg_distance: if count > 0 {
                    total_dist / count as f32
                } else {
                    0.0
                },
            });
        }

        stats
    }

    /// Merge sparse clusters into their nearest neighbors.
    pub fn merge_sparse_clusters(&mut self, threshold: usize) -> Result<usize> {
        if !self.is_finalized || self.centroids.is_empty() {
            return Ok(0);
        }

        let stats = self.get_cluster_stats();
        let sparse_indices: Vec<usize> = stats
            .iter()
            .filter(|s| s.count < threshold)
            .map(|s| s.cluster_id)
            .collect();

        if sparse_indices.is_empty() || sparse_indices.len() == self.centroids.len() {
            return Ok(0);
        }

        let non_sparse_indices: Vec<usize> = stats
            .iter()
            .filter(|s| s.count >= threshold)
            .map(|s| s.cluster_id)
            .collect();

        let merged_count = sparse_indices.len();
        let mut moves = Vec::new();

        for &sparse_idx in &sparse_indices {
            let sparse_centroid = &self.centroids[sparse_idx];
            let mut best_target = non_sparse_indices[0];
            let mut best_dist = f32::INFINITY;

            for &target_idx in &non_sparse_indices {
                if let Ok(dist) = self
                    .index_config
                    .distance_metric
                    .distance(&sparse_centroid.data, &self.centroids[target_idx].data)
                    && dist < best_dist
                {
                    best_dist = dist;
                    best_target = target_idx;
                }
            }
            moves.push((sparse_idx, best_target));
        }

        // Apply moves
        for (sparse_idx, target_idx) in moves {
            let mut vectors_to_move = std::mem::take(&mut self.inverted_lists[sparse_idx]);
            self.inverted_lists[target_idx].append(&mut vectors_to_move);
        }

        // Rebuild centroids and inverted lists
        let mut new_centroids = Vec::new();
        let mut new_inverted_lists = Vec::new();

        for i in 0..self.centroids.len() {
            if !sparse_indices.contains(&i) {
                new_centroids.push(self.centroids[i].clone());
                new_inverted_lists.push(std::mem::take(&mut self.inverted_lists[i]));
            }
        }

        self.centroids = new_centroids;
        self.inverted_lists = new_inverted_lists;
        self.index_config.n_clusters = self.centroids.len();

        // Update centroids for the merged clusters (re-average)
        for (i, list) in self.inverted_lists.iter().enumerate() {
            if !list.is_empty() {
                let dim = self.index_config.dimension;
                let mut sum = vec![0.0; dim];
                for (_, _, vec) in list {
                    for (j, &val) in vec.data.iter().enumerate() {
                        sum[j] += val;
                    }
                }
                let new_data: Vec<f32> = sum.iter().map(|&s| s / list.len() as f32).collect();
                self.centroids[i] = Vector::new(new_data);
            }
        }

        Ok(merged_count)
    }

    /// Split dense clusters into multiple clusters using K-means (k=2).
    pub fn split_dense_clusters(&mut self, threshold: usize) -> Result<usize> {
        if !self.is_finalized || self.centroids.is_empty() {
            return Ok(0);
        }

        let stats = self.get_cluster_stats();
        let dense_indices: Vec<usize> = stats
            .iter()
            .filter(|s| s.count > threshold)
            .map(|s| s.cluster_id)
            .collect();

        if dense_indices.is_empty() {
            return Ok(0);
        }

        let mut additional_clusters = 0;
        let mut new_centroids = Vec::new();
        let mut new_inverted_lists = Vec::new();

        for i in 0..self.centroids.len() {
            if dense_indices.contains(&i) {
                let list = std::mem::take(&mut self.inverted_lists[i]);
                if list.len() < 2 {
                    // Cannot split
                    new_centroids.push(self.centroids[i].clone());
                    new_inverted_lists.push(list);
                    continue;
                }

                // Perform k=2 split
                let (c1, l1, c2, l2) = self.split_cluster_kmeans_k2(list)?;
                new_centroids.push(c1);
                new_inverted_lists.push(l1);
                new_centroids.push(c2);
                new_inverted_lists.push(l2);
                additional_clusters += 1;
            } else {
                new_centroids.push(self.centroids[i].clone());
                new_inverted_lists.push(std::mem::take(&mut self.inverted_lists[i]));
            }
        }

        self.centroids = new_centroids;
        self.inverted_lists = new_inverted_lists;
        self.index_config.n_clusters = self.centroids.len();

        Ok(additional_clusters)
    }

    /// Split a cluster into two using K-means.
    fn split_cluster_kmeans_k2(
        &self,
        vectors: Vec<(u64, String, Vector)>,
    ) -> Result<SplitClusterResult> {
        use rand::prelude::*;
        // Deterministic training RNG (Issue #847); see KMEANS_RNG_SEED.
        let mut rng = rand::rngs::StdRng::seed_from_u64(KMEANS_RNG_SEED);

        // Pick two initial centroids
        let idx1 = rng.random_range(0..vectors.len());
        let mut idx2 = rng.random_range(0..vectors.len());
        while idx1 == idx2 && vectors.len() > 1 {
            idx2 = rng.random_range(0..vectors.len());
        }

        let mut c1 = vectors[idx1].2.clone();
        let mut c2 = vectors[idx2].2.clone();

        let mut l1 = Vec::new();
        let mut l2 = Vec::new();

        // Simple 10 iterations of K-means
        for _ in 0..10 {
            l1.clear();
            l2.clear();

            for (_, _, vec) in &vectors {
                let d1 = self
                    .index_config
                    .distance_metric
                    .distance(&vec.data, &c1.data)
                    .unwrap_or(f32::INFINITY);
                let d2 = self
                    .index_config
                    .distance_metric
                    .distance(&vec.data, &c2.data)
                    .unwrap_or(f32::INFINITY);

                if d1 < d2 {
                    l1.push((0, String::new(), vec.clone())); // We'll restore the actual IDs later
                } else {
                    l2.push((0, String::new(), vec.clone()));
                }
            }

            // Update centroids
            if !l1.is_empty() {
                c1 = self.calculate_mean_vector(&l1);
            }
            if !l2.is_empty() {
                c2 = self.calculate_mean_vector(&l2);
            }
        }

        // Final assignment with original vectors to preserve IDs
        l1.clear();
        l2.clear();
        for item in vectors {
            let d1 = self
                .index_config
                .distance_metric
                .distance(&item.2.data, &c1.data)
                .unwrap_or(f32::INFINITY);
            let d2 = self
                .index_config
                .distance_metric
                .distance(&item.2.data, &c2.data)
                .unwrap_or(f32::INFINITY);

            if d1 < d2 {
                l1.push(item);
            } else {
                l2.push(item);
            }
        }

        Ok((c1, l1, c2, l2))
    }

    /// Calculate the mean vector for a list of vectors.
    fn calculate_mean_vector(&self, list: &[(u64, String, Vector)]) -> Vector {
        let dim = self.index_config.dimension;
        if list.is_empty() {
            return Vector::new(vec![0.0; dim]);
        }
        let mut sum = vec![0.0; dim];
        for (_, _, vec) in list {
            for (j, &val) in vec.data.iter().enumerate() {
                sum[j] += val;
            }
        }
        let data: Vec<f32> = sum.iter().map(|&s| s / list.len() as f32).collect();
        Vector::new(data)
    }
    // optimize method moved to VectorIndexWriter trait implementation
}

#[async_trait::async_trait]
impl VectorIndexWriter for IvfIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    fn build(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors = vectors;
        self.total_vectors_to_add = Some(self.vectors.len());

        // Cluster-count clamping now happens uniformly in `train_centroids`
        // (Issue #889 PR-5), regardless of whether vectors arrived via
        // `build` or `add_vectors`.
        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors.extend(vectors);
        self.check_memory_limit()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.is_finalized {
            return Ok(());
        }

        // `train_centroids` handles the empty-vectors case itself (Issue
        // #889 PR-6: zero clusters, not an error) — no separate guard
        // needed here.
        self.train_centroids()?;

        // Build inverted lists
        self.build_inverted_lists()?;

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
                8 + // doc_id
            self.index_config.dimension * 4 + // f32 values
            std::mem::size_of::<Vector>()
                // Vector struct overhead
            );

        // Centroid memory
        let centroid_memory = self.centroids.len()
            * (self.index_config.dimension * 4 + std::mem::size_of::<Vector>());

        // Inverted list overhead (pointers and metadata)
        let inverted_list_memory =
            self.inverted_lists.len() * (std::mem::size_of::<Vec<(u64, String, Vector)>>() + 64); // Rough estimate

        let metadata_memory = self.vectors.len() * 64;

        vector_memory + centroid_memory + inverted_list_memory + metadata_memory
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

        // Write to a temp file and atomically rename into place (Issue #889,
        // matching HNSW's #784 pattern) so a crash mid-write leaves the
        // previously committed `.ivf` intact instead of a truncated,
        // unreadable segment.
        let file_name = format!("{}.ivf", self.path);
        let tmp_name = format!("{}.ivf.tmp", self.path);
        let mut output = storage.create_output(&tmp_name)?;

        // Write metadata
        let vector_count: u32 = self.vectors.len().try_into().map_err(|_| {
            LaurusError::InvalidOperation(format!(
                "Vector count {} exceeds u32::MAX",
                self.vectors.len()
            ))
        })?;
        output.write_all(&vector_count.to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.n_clusters as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.n_probe as u32).to_le_bytes())?;

        // Write centroids (kept as f32 - they're cluster centers used
        // for partition assignment at search time, not part of the
        // segment vector data the LVS1 header describes).
        for centroid in &self.centroids {
            for value in centroid.data.iter() {
                output.write_all(&value.to_le_bytes())?;
            }
        }

        // Issue #481 Stage 1, Step 7: train per-segment SQ params on
        // ALL vectors across ALL inverted lists (segment-wide single
        // (offset, scale) pair) and emit the LVS1 header before the
        // inverted-list data.
        let all_vectors: Vec<Vector> = self
            .inverted_lists
            .iter()
            .flat_map(|list| list.iter().map(|(_, _, v)| v.clone()))
            .collect();
        let (params, records) = if all_vectors.is_empty() {
            (
                ScalarQuantParams {
                    offset: 0.0,
                    scale: 1.0,
                },
                Vec::new(),
            )
        } else {
            quantize_segment(&all_vectors, self.index_config.dimension)?
        };
        // Per-segment field-name dictionary (Issue #633): ids assigned in
        // first-appearance order over the cluster-grouped emission order
        // below (the same order the flatten above walked).
        let (field_dict, field_ids) = build_field_dict(
            self.inverted_lists
                .iter()
                .flat_map(|list| list.iter().map(|(_, f, _)| f.as_str())),
        )?;
        VectorSegmentHeader::scalar_8bit(params)
            .with_version(VERSION_FIELD_DICT)
            .with_field_dict(field_dict)
            .write_to(&mut output)?;

        // Write inverted lists with quantized records. The records
        // were produced in flatten order, so we step through them in
        // the same order while emitting each list.
        let mut record_iter = records.into_iter();
        for list in &self.inverted_lists {
            output.write_all(&(list.len() as u32).to_le_bytes())?;
            for (doc_id, field_name, _) in list {
                output.write_all(&doc_id.to_le_bytes())?;
                output.write_all(&field_ids[field_name.as_str()].to_le_bytes())?;

                // Write quantized payload (dim int8 + sum_q + norm_q).
                let (int8, meta) = record_iter
                    .next()
                    .expect("records vector length matches the sum of inverted_lists lengths");
                write_quantized_record(&mut output, &int8, meta)?;
            }
        }

        // Close with an fsync BEFORE the rename (mirrors HNSW's #882 review
        // fix): a flush alone leaves the content in the page cache, so a
        // power loss could surface a published-but-hollow segment file.
        output.close()?;
        storage.rename_file(&tmp_name, &file_name)?;

        // Stage 2 (Issue #481, extended to IVF by #650 PR-2 / #932): emit
        // the optional LRS1 rerank sidecar alongside the main int8 segment.
        // The payload reuses `all_vectors` — materialized above in the same
        // cluster-grouped flatten order the records were emitted in — so
        // the reader's (sidecar position) -> (record position) mapping is
        // the identity, mirroring HNSW.
        if let Some(rerank_kind) = self.index_config.rerank_storage {
            let sidecar_name = format!("{}.f32", file_name);
            let sidecar_tmp = format!("{}.f32.tmp", file_name);
            let mut sidecar_out = storage.create_output(&sidecar_tmp)?;
            let mut payload: Vec<f32> =
                Vec::with_capacity(all_vectors.len() * self.index_config.dimension);
            for v in &all_vectors {
                payload.extend_from_slice(&v.data);
            }
            crate::vector::index::rerank_sidecar::write_sidecar(
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

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }
        self.vectors.retain(|(id, _, _)| *id != doc_id);
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
        self.is_finalized = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.is_finalized && self.vectors.is_empty()
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // Rebalance clusters if they're too uneven
        let total_vectors = self.vectors.len();
        let avg_vectors_per_cluster = total_vectors / self.index_config.n_clusters.max(1);
        let sparse_threshold = avg_vectors_per_cluster / 4;
        let dense_threshold = avg_vectors_per_cluster * 4;

        self.merge_sparse_clusters(sparse_threshold.max(2))?;
        self.split_dense_clusters(dense_threshold)?;

        // For now, just compact memory
        self.vectors.shrink_to_fit();
        self.centroids.shrink_to_fit();
        for list in &mut self.inverted_lists {
            list.shrink_to_fit();
        }

        Ok(())
    }

    fn build_reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        use crate::vector::index::ivf::reader::IvfIndexReader;

        let storage = self.storage.as_ref().ok_or_else(|| {
            LaurusError::InvalidOperation("Cannot build reader: storage not configured".to_string())
        })?;

        let reader = IvfIndexReader::load(
            storage.clone(),
            &self.path,
            self.index_config.distance_metric,
        )?;

        Ok(Arc::new(reader))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vector corpus (no thread RNG).
    fn lcg_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        (0..count)
            .map(|i| {
                let data: Vec<f32> = (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                    })
                    .collect();
                (i as u64, "f".to_string(), Vector::new(data))
            })
            .collect()
    }

    /// Issue #847: with the seeded k-means RNG, two trainings over the
    /// same input produce bitwise-identical centroids and identical
    /// cluster assignments.
    ///
    /// Uses n = 500 (below the 1000-vector parallel gate) so the whole
    /// Lloyd loop runs serially: at larger n the parallel accumulation
    /// from Issue #843 may regroup f32 sums across threads, so parallel
    /// builds are deterministic only up to summation order. This test
    /// pins exactly what the seed guarantees — the same RNG picks, the
    /// same k-means++ starts, the same convergence trajectory —
    /// mirroring how #842's HNSW test pins level assignment only.
    #[test]
    fn kmeans_training_is_deterministic_with_seeded_rng() {
        fn train_once() -> (Vec<Vec<u32>>, Vec<Vec<u64>>) {
            let config = IvfIndexConfig {
                dimension: 16,
                n_clusters: 8,
                n_probe: 2,
                ..Default::default()
            };
            let mut writer =
                IvfIndexWriter::new(config, VectorIndexWriterConfig::default(), "determinism")
                    .unwrap();
            writer.add_vectors(lcg_vectors(500, 16)).unwrap();
            writer.finalize().unwrap();

            let centroids: Vec<Vec<u32>> = writer
                .centroids
                .iter()
                .map(|c| c.data.iter().map(|f| f.to_bits()).collect())
                .collect();
            let assignments: Vec<Vec<u64>> = writer
                .inverted_lists
                .iter()
                .map(|list| {
                    let mut ids: Vec<u64> = list.iter().map(|(id, _, _)| *id).collect();
                    ids.sort_unstable();
                    ids
                })
                .collect();
            (centroids, assignments)
        }

        let (centroids_a, assignments_a) = train_once();
        let (centroids_b, assignments_b) = train_once();
        assert_eq!(
            centroids_a, centroids_b,
            "two seeded trainings must produce bitwise-identical centroids"
        );
        assert_eq!(
            assignments_a, assignments_b,
            "cluster assignments must be identical across runs"
        );
    }

    /// Issue #889 PR-6: finalizing a writer with zero buffered vectors must
    /// succeed with zero clusters instead of erroring — the segmented
    /// force-merge path can legitimately reduce a merge window to zero
    /// surviving vectors when every document in it was deleted.
    #[test]
    fn finalize_on_empty_vectors_succeeds_with_zero_clusters() {
        let config = IvfIndexConfig {
            dimension: 4,
            n_clusters: 8,
            n_probe: 1,
            ..Default::default()
        };
        let mut writer =
            IvfIndexWriter::new(config, VectorIndexWriterConfig::default(), "empty_merge").unwrap();

        writer.finalize().unwrap();
        assert_eq!(writer.centroids().len(), 0);
        assert_eq!(writer.ivf_params().0, 0, "n_clusters must be reset to 0");
    }
}
