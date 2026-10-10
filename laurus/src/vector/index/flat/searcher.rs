//! Flat vector searcher for exact search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::distance_quantized::{QuantizedQuery, distance_quantized};
use crate::vector::core::vector::Vector;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryResults};

/// Flat vector searcher that performs exact (brute force) search.
#[derive(Debug)]
pub struct FlatVectorSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
}

impl FlatVectorSearcher {
    /// Create a new flat vector searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        Ok(Self { index_reader })
    }
}

impl VectorIndexSearcher for FlatVectorSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        use crate::util::time::Timer;

        // `request.filter` (Issue #645 allow-set) is honoured inline (Issue
        // #740): candidates whose `doc_id` is not in the set are skipped before
        // the distance kernel, saving the distance computation for a selective
        // filter. The store's post-filter still runs but becomes a no-op for
        // the already-filtered results, so recall is unchanged.

        // Issue #481 Stage 2 rerank (extended to Flat by #650 PR-2 /
        // #932): honored on the field-filtered path below when the reader
        // has the `.f32` sidecar loaded. On the unfiltered path (or when
        // the sidecar is absent) `rerank_factor` silently falls back to
        // Stage 1 ranking — the same convention as HNSW, where a missing
        // prerequisite (here: a single field to key the sidecar position
        // index) cannot be recovered at query time. Issue #1388: a
        // field-less request can still take this path when the reader
        // unambiguously belongs to one field (see
        // `FlatVectorIndexReader::sole_field_name`).

        let start = Timer::now();
        let mut results = VectorIndexQueryResults::new();
        let metric = self.index_reader.distance_metric();
        // Cache the query-side norm once per search (#414): for Cosine /
        // Angular this skips one `||query||²` accumulation per
        // candidate; for the other metrics the prepared variant
        // forwards to `distance` and the cached value is unused.
        let prepared_query = metric.prepare_query(&request.query.data);

        // Issue #481 Stage 1, Step 7: try the int8 hot path for the
        // field-filtered case if the reader holds an OwnedQuantized
        // pool. Build per-search QuantizedQuery + per-field position
        // index once before the candidate loop.
        let flat_reader = self
            .index_reader
            .as_any()
            .downcast_ref::<FlatVectorIndexReader>();
        let quant_pool = flat_reader.and_then(|r| r.vectors().quantized_pool().cloned());

        // Filter-aware allow-set honoured inline (Issue #740). Borrowed once
        // before the scan; `&FilterSet` is `Send + Sync`, so it composes with
        // the rayon-parallel candidate loops below.
        let filter = request.filter.as_deref();

        // Issue #1388: a field-less request (`field_name: None`) can still
        // take the field-filtered path when this reader unambiguously
        // belongs to one field (see `FlatVectorIndexReader::sole_field_name`).
        // Falls through to the unfiltered path below only when the field
        // cannot be resolved (empty or genuinely multi-field reader -- the
        // legacy, non-multi-field single-index layout).
        let field_name = request
            .field_name
            .as_deref()
            .or_else(|| flat_reader.and_then(|r| r.sole_field_name()));

        if let Some(field_name) = field_name {
            // Field-filtered path: fetch the per-field doc-id slice from the
            // reader's pre-built index (#405 — O(1) Arc clone, avoids the full
            // `Vec<(u64, String)>` clone and the linear filter scan). Since
            // every candidate shares the same `field_name`, do not store it
            // per-candidate; clone it only when constructing the top_k
            // results.
            let ids = self.index_reader.doc_ids_for_field(field_name);
            results.candidates_examined = ids.len();

            // Step-7 hot path: prepare quantized query once and look
            // up per-field doc_id -> position once per search.
            let quant_ctx = quant_pool.as_ref().and_then(|pool| {
                pool.field_position_index(field_name).map(|idx| {
                    let prepared = QuantizedQuery::prepare(&request.query.data, &pool.params);
                    (prepared, pool.clone(), idx)
                })
            });

            // Distance scan, parallelised across candidates above
            // PARALLEL_SCAN_THRESHOLD (#662). The quantized hot path and the
            // f32 fallback both run inside the per-candidate closure; a
            // missing vector yields `Ok(None)` (skipped) and a dimension
            // mismatch propagates as `Err`.
            let mut candidates: Vec<(u64, f32, f32, Vector)> =
                crate::vector::search::searcher::parallel_scan(&ids[..], |&doc_id| {
                    // Skip non-matching candidates before the distance kernel.
                    if let Some(allowed) = filter
                        && !allowed.contains(doc_id)
                    {
                        return Ok(None);
                    }
                    if let Some((prepared, pool, idx)) = &quant_ctx
                        && let Some(&pos) = idx.get(&doc_id)
                        && !flat_reader.is_some_and(|r| r.is_deleted(doc_id))
                    {
                        let (int8, meta) = pool.record_at(pos);
                        let distance = distance_quantized(metric, prepared, int8, meta);
                        let similarity = metric.distance_to_similarity(distance);
                        // include_vectors path still needs the f32 vector;
                        // dequantize lazily only when requested.
                        let vector = if request.params.include_vectors {
                            pool.dequantize_to_vector(doc_id, field_name)
                                .unwrap_or_else(|| Vector::new(Vec::new()))
                        } else {
                            Vector::new(Vec::new())
                        };
                        return Ok(Some((doc_id, similarity, distance, vector)));
                    }
                    if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, field_name) {
                        let distance =
                            metric.distance_with_prepared(&prepared_query, &vector.data)?;
                        let similarity = metric.distance_to_similarity(distance);
                        return Ok(Some((doc_id, similarity, distance, vector)));
                    }
                    Ok(None)
                })?;

            // With an inline allow-set the scored count (post-skip) is the
            // meaningful "candidates examined" figure (Issue #740); the
            // unfiltered figure (`ids.len()`) is kept when no filter is set.
            if filter.is_some() {
                results.candidates_examined = candidates.len();
            }

            // Sort ascending by distance with a doc-id tiebreak (#933):
            // similarity's `exp(-d)` underflows to 0.0 at long range,
            // collapsing distant candidates into ties whose unstable order
            // would make top-k membership arbitrary; distance stays precise.
            candidates.sort_unstable_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));

            // Stage 2 (Issue #481 / #932): run the shared rerank pipeline
            // over the sorted quantized candidates. When it applies, the
            // survivors carry exact f32 distances — stamped below as the
            // fan-out's `score_basis` (#927).
            let mut rerank_applied = false;
            if let (Some(factor), Some(pool)) = (
                request.params.rerank_factor,
                flat_reader.and_then(|r| r.rerank_storage()),
            ) {
                use crate::vector::search::rerank::{
                    F32SidecarStage, RerankCandidates, RerankPipeline,
                };
                let pipeline = RerankPipeline::new(
                    vec![Box::new(F32SidecarStage::new(
                        std::sync::Arc::clone(pool),
                        field_name,
                        metric,
                    ))],
                    vec![factor],
                );
                let mut rc = RerankCandidates::with_capacity(candidates.len());
                for (doc_id, _, distance, _) in &candidates {
                    rc.push(*doc_id, *distance);
                }
                rerank_applied = pipeline.run(&request.query, &mut rc, request.params.top_k)?;
                if rerank_applied {
                    // Rebuild on the exact basis; vectors are re-fetched in
                    // the result loop only when `include_vectors` asks.
                    candidates = rc
                        .doc_ids
                        .iter()
                        .zip(&rc.distances)
                        .map(|(&doc_id, &distance)| {
                            (
                                doc_id,
                                metric.distance_to_similarity(distance),
                                distance,
                                Vector::new(Vec::new()),
                            )
                        })
                        .collect();
                }
            }

            let top_k = request.params.top_k.min(candidates.len());
            for (doc_id, similarity, distance, vector) in candidates.into_iter().take(top_k) {
                if similarity < request.params.min_similarity {
                    break;
                }

                let vector_output = if request.params.include_vectors {
                    if rerank_applied {
                        // The rescored tuple carries a placeholder; fetch
                        // the real vector only for the final results.
                        self.index_reader.get_vector(doc_id, field_name)?
                    } else {
                        Some(vector)
                    }
                } else {
                    None
                };

                results
                    .results
                    .push(crate::vector::search::searcher::VectorIndexQueryResult {
                        doc_id,
                        field_name: field_name.to_string(),
                        similarity,
                        distance,
                        vector: vector_output,
                    });
            }

            // Issue #927: exact-f32 scores must be flagged so the
            // multi-segment fan-out keeps them instead of overwriting
            // with its dequantized rescore.
            if rerank_applied {
                results.query_metadata.insert(
                    crate::vector::search::searcher::SCORE_BASIS_METADATA_KEY.to_string(),
                    crate::vector::search::searcher::SCORE_BASIS_F32_RERANK.to_string(),
                );
            }
        } else {
            // Unfiltered path: each doc may belong to a different field, so the
            // field name must travel with each candidate.
            let candidates_list = self.index_reader.vector_ids()?;
            results.candidates_examined = self.index_reader.vector_count();

            // Distance scan, parallelised across candidates above
            // PARALLEL_SCAN_THRESHOLD (#662). Each candidate may belong to a
            // different field, so the field name travels with the result.
            let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
                crate::vector::search::searcher::parallel_scan(
                    &candidates_list[..],
                    |(doc_id, field_name)| {
                        // Skip non-matching candidates before the distance kernel.
                        if let Some(allowed) = filter
                            && !allowed.contains(*doc_id)
                        {
                            return Ok(None);
                        }
                        if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name)
                        {
                            let distance =
                                metric.distance_with_prepared(&prepared_query, &vector.data)?;
                            let similarity = metric.distance_to_similarity(distance);
                            return Ok(Some((
                                *doc_id,
                                field_name.clone(),
                                similarity,
                                distance,
                                vector,
                            )));
                        }
                        Ok(None)
                    },
                )?;

            // With an inline allow-set the scored count (post-skip) is the
            // meaningful "candidates examined" figure (Issue #740); the
            // unfiltered figure (`vector_count()`) is kept when no filter is set.
            if filter.is_some() {
                results.candidates_examined = candidates.len();
            }

            // Sort ascending by distance with a doc-id tiebreak (#933):
            // similarity's `exp(-d)` underflows to 0.0 at long range,
            // collapsing distant candidates into ties whose unstable order
            // would make top-k membership arbitrary; distance stays precise.
            candidates.sort_unstable_by(|a, b| a.3.total_cmp(&b.3).then(a.0.cmp(&b.0)));

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
        // For a field-filtered count, use the pre-built per-field index
        // (#405); avoids allocating + iterating the full `vector_ids`.
        if let Some(ref field_name) = request.field_name {
            Ok(self.index_reader.doc_ids_for_field(field_name).len() as u64)
        } else {
            // Issue #672: `vector_ids()` materializes a String per record
            // just to be counted; `vector_count()` is the same number (one
            // entry per (doc, field) record) with no allocation.
            Ok(self.index_reader.vector_count() as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::index::FlatIndexConfig;
    use crate::vector::index::flat::writer::FlatIndexWriter;
    use crate::vector::search::filter_set::FilterSet;
    use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

    /// Build a 20-vector flat index (all in field `f`) on a line so the
    /// nearest-to-query ordering is deterministic.
    fn build_flat_reader(name: &str) -> Arc<dyn VectorIndexReader> {
        let storage = Arc::new(MemoryStorage::default());
        let config = FlatIndexConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
            normalize_vectors: false,
            ..FlatIndexConfig::default()
        };
        let mut writer = FlatIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            name,
            storage,
        )
        .unwrap();
        let vectors: Vec<(u64, String, Vector)> = (0..20)
            .map(|i| (i as u64, "f".to_string(), Vector::new(vec![i as f32, 0.0])))
            .collect();
        writer.build(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();
        writer.build_reader().unwrap()
    }

    /// The flat searcher must skip non-matching candidates before the distance
    /// kernel when an allow-set filter is supplied (Issue #740), on both the
    /// unfiltered and field-filtered scan paths, while leaving the no-filter
    /// path unchanged.
    #[test]
    fn test_flat_searcher_honors_filter_inline() {
        let reader = build_flat_reader("test_flat_filter_inline");
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = Vector::new(vec![3.0, 0.0]);
        let allow: Arc<FilterSet> = Arc::new(FilterSet::Hash([3u64, 7, 15].into_iter().collect()));

        // No filter (unfiltered path): every vector is scored — unchanged.
        let unfiltered = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(20))
            .unwrap();
        assert_eq!(unfiltered.candidates_examined, 20);

        // Unfiltered path + inline allow-set: only the 3 allowed docs scored.
        let filtered = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(20)
                    .filter(allow.clone()),
            )
            .unwrap();
        assert_eq!(
            filtered.candidates_examined, 3,
            "only allowed docs should reach the distance kernel"
        );
        for r in &filtered.results {
            assert!(
                allow.contains(r.doc_id),
                "result {} not in allow-set",
                r.doc_id
            );
        }

        // Field-filtered path (field_name set) + inline allow-set.
        let field_filtered = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(20)
                    .field_name("f".to_string())
                    .filter(allow.clone()),
            )
            .unwrap();
        assert_eq!(field_filtered.candidates_examined, 3);
        for r in &field_filtered.results {
            assert!(
                allow.contains(r.doc_id),
                "result {} not in allow-set",
                r.doc_id
            );
        }
        // Closest allowed vector to the query (3, 0) is doc 3.
        assert_eq!(field_filtered.results[0].doc_id, 3);
    }

    /// The Flat scan must return the same results whether the allow-set is a
    /// `Hash` or a `Bitmap` (Issue #739) — the representation is an internal
    /// detail that must not change which documents match.
    #[test]
    fn filter_hash_and_bitmap_agree() {
        let reader = build_flat_reader("test_flat_filter_repr_agree");
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = Vector::new(vec![3.0, 0.0]);
        let ids = [3u64, 7, 15];

        let hash = Arc::new(FilterSet::Hash(ids.into_iter().collect()));
        let bitmap = Arc::new(FilterSet::from_bitmap(Arc::new(ids.into_iter().collect())));

        let run = |fs: Arc<FilterSet>| -> Vec<u64> {
            let mut got: Vec<u64> = searcher
                .search(&VectorIndexQuery::new(query.clone()).top_k(20).filter(fs))
                .unwrap()
                .results
                .into_iter()
                .map(|r| r.doc_id)
                .collect();
            got.sort_unstable();
            got
        };

        assert_eq!(
            run(hash),
            run(bitmap),
            "Hash and Bitmap allow-sets must yield identical results"
        );
    }
}
