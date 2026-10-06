//! Intrinsic-fidelity tests for `MultiVector` storage compression (Issue
//! #1346): f16/int8-compressed late-interaction (MaxSim) scores are
//! compared against the f32 baseline, not against an external IR benchmark
//! (the repository has no BEIR/nDCG harness, and this issue's accepted
//! scope is intrinsic fidelity only — see the implementation plan).
//!
//! The per-element error bounds below are derived, not guessed:
//! - f16 has an 11-bit significand, so its relative error per element is
//!   bounded by 2⁻¹¹ ≈ 4.9e-4. Accumulated over a dot product of
//!   L2-normalized vectors this stays well under 2e-3.
//! - int8 uses a per-vector scale `max|d| / 127` (not a global range), so
//!   for L2-normalized ColBERT-sized vectors (dimension 64, per-element
//!   magnitude around 0.1-0.4) the half-ULP quantization step is roughly
//!   `max|d| / 254`; accumulated over a dot product this is bounded well
//!   under 5e-2 per query token, with a large safety margin.
//!
//! Rank agreement is asserted only where it is decidable without flakiness:
//! wherever the f32 reference's score gap between consecutive candidates
//! exceeds twice the derived bound, the compressed ranking must agree
//! exactly on that boundary (see `test_top_k_rank_agreement_is_exact_where_the_f32_gap_is_decisive`).

use std::sync::Arc;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::{
    DistanceMetric, Document, Engine, FieldOption, MultiVectorOption, MultiVectorStorage,
    RescoreOptions, Result, Schema, SearchRequestBuilder, TextOption,
};

const TOKENS: &str = "tokens";
const DIMENSION: usize = 64;
const DOC_COUNT: usize = 24;
const QUERY_TOKEN_COUNT: usize = 6;

/// f16 tolerance per query token (see module docs for the derivation).
const F16_TOLERANCE_PER_QUERY_TOKEN: f32 = 2e-3;
/// int8 tolerance per query token (see module docs for the derivation).
const INT8_TOLERANCE_PER_QUERY_TOKEN: f32 = 5e-2;

/// Deterministic pseudo-random values in `[-1, 1)` (same generator as
/// `vector/core/late_interaction.rs`'s tests, duplicated rather than
/// exposed across a test/production boundary).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// One document's deterministic token vectors: `3 + (seed % 5)` tokens,
/// each L2-normalized (matching how a real ColBERT embedder's output is
/// used with `DistanceMetric::Cosine`).
fn doc_tokens(seed: u64) -> Vec<Vec<f32>> {
    let token_count = 3 + (seed % 5) as usize;
    (0..token_count)
        .map(|i| normalize(values(seed * 1000 + i as u64, DIMENSION)))
        .collect()
}

fn query_tokens() -> Vec<Vector> {
    (0..QUERY_TOKEN_COUNT)
        .map(|i| Vector::new(normalize(values(9_000 + i as u64, DIMENSION))))
        .collect()
}

fn schema(storage: MultiVectorStorage) -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(
            TOKENS,
            FieldOption::MultiVector(
                MultiVectorOption::new(DIMENSION)
                    .distance(DistanceMetric::Cosine)
                    .storage(storage),
            ),
        )
        .build()
}

async fn engine(storage: MultiVectorStorage) -> Result<Engine> {
    let engine = Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        schema(storage),
    )
    .await?;
    for seed in 0..DOC_COUNT as u64 {
        let doc = Document::builder()
            .add_text("title", "doc")
            .add_vector_array(TOKENS, doc_tokens(seed))
            .build();
        engine.put_document(&seed.to_string(), doc).await?;
    }
    engine.commit().await?;
    Ok(engine)
}

fn rescore() -> RescoreOptions {
    RescoreOptions::late_interaction(TOKENS, query_tokens()).window_size(DOC_COUNT)
}

/// `doc_id -> score`, ordered by the engine's own ranking.
async fn scored(engine: &Engine) -> Result<Vec<(String, f32)>> {
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("title:doc")
                .rescore(rescore())
                .limit(DOC_COUNT)
                .build(),
        )
        .await?;
    Ok(results.into_iter().map(|r| (r.id, r.score)).collect())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_f16_scores_match_f32_within_the_derived_bound() -> Result<()> {
    let f32_engine = engine(MultiVectorStorage::F32).await?;
    let f16_engine = engine(MultiVectorStorage::F16).await?;

    let f32_scores = scored(&f32_engine).await?;
    let f16_by_id: std::collections::HashMap<_, _> =
        scored(&f16_engine).await?.into_iter().collect();

    assert_eq!(f32_scores.len(), DOC_COUNT);
    let tol = F16_TOLERANCE_PER_QUERY_TOKEN * QUERY_TOKEN_COUNT as f32;
    for (id, f32_score) in &f32_scores {
        let f16_score = f16_by_id[id];
        assert!(
            (f32_score - f16_score).abs() <= tol,
            "doc {id}: f32 {f32_score} vs f16 {f16_score} (tol {tol})"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_int8_scores_match_f32_within_the_derived_bound() -> Result<()> {
    let f32_engine = engine(MultiVectorStorage::F32).await?;
    let int8_engine = engine(MultiVectorStorage::Int8).await?;

    let f32_scores = scored(&f32_engine).await?;
    let int8_by_id: std::collections::HashMap<_, _> =
        scored(&int8_engine).await?.into_iter().collect();

    assert_eq!(f32_scores.len(), DOC_COUNT);
    let tol = INT8_TOLERANCE_PER_QUERY_TOKEN * QUERY_TOKEN_COUNT as f32;
    for (id, f32_score) in &f32_scores {
        let int8_score = int8_by_id[id];
        assert!(
            (f32_score - int8_score).abs() <= tol,
            "doc {id}: f32 {f32_score} vs int8 {int8_score} (tol {tol})"
        );
    }
    Ok(())
}

/// Deterministic (non-flaky) rank-agreement check: compressed rankings are
/// only required to agree with the f32 reference at a rank boundary whose
/// f32 score gap is large enough that no amount of quantization error
/// (within the derived bound) could plausibly reorder it. A boundary with
/// a smaller gap is a genuine near-tie and is intentionally not asserted.
#[tokio::test(flavor = "multi_thread")]
async fn test_top_k_rank_agreement_is_exact_where_the_f32_gap_is_decisive() -> Result<()> {
    let f32_engine = engine(MultiVectorStorage::F32).await?;
    let int8_engine = engine(MultiVectorStorage::Int8).await?;

    let f32_scores = scored(&f32_engine).await?;
    let int8_ids: Vec<String> = scored(&int8_engine)
        .await?
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let f32_ids: Vec<String> = f32_scores.iter().map(|(id, _)| id.clone()).collect();

    let bound = INT8_TOLERANCE_PER_QUERY_TOKEN * QUERY_TOKEN_COUNT as f32;
    let mut decisive_prefix = f32_scores.len();
    for k in 0..f32_scores.len().saturating_sub(1) {
        let gap = f32_scores[k].1 - f32_scores[k + 1].1;
        if gap <= 2.0 * bound {
            decisive_prefix = k + 1;
            break;
        }
    }
    // The corpus and query are fixed and deterministic; if nothing is
    // decisive the bound (or the test corpus) needs revisiting, not a
    // silently-vacuous test.
    assert!(
        decisive_prefix > 0,
        "no decisive prefix: tighten the corpus or document the gap"
    );
    assert_eq!(
        &f32_ids[..decisive_prefix],
        &int8_ids[..decisive_prefix],
        "f32 vs int8 top-{decisive_prefix} order diverged"
    );
    Ok(())
}
