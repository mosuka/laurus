//! The rescore stage of [`Engine::search`](super::Engine::search)
//! (Issue #1345).
//!
//! The first stage ranks candidates as usual; this stage reorders the top
//! `window_size` of them before the page is cut. Options are validated
//! against the schema before any search runs, so a bad request fails fast.
//!
//! Preparing takes two steps (Issue #1349): [`RescorePlan::new`] checks the
//! options synchronously under the schema lock, and
//! [`RescorePlan::prepare`] embeds a text query outside it, so a bad
//! request fails before any model runs.

use std::cell::Cell;
use std::sync::Arc;

use crate::embedding::cache::{EmbeddingCache, embed_tokens_with_cache};
use crate::embedding::embedder::{EmbedInput, EmbedRole, Embedder};
use crate::embedding::per_field::resolve_field_embedder;
use crate::engine::schema::{FieldOption, Schema};
use crate::engine::search::{LateInteractionQuery, RescoreOptions, Rescorer};
use crate::error::{LaurusError, Result};
use crate::lexical::search::searcher::SortField;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::multivector::MultiVectorSnapshot;
use crate::vector::store::VectorStore;

/// Most query token vectors a late-interaction rescore accepts (ColBERT
/// queries are typically 32 tokens).
const MAX_QUERY_VECTORS: usize = 1024;

/// Largest scratch, in bytes, that a thread keeps for its next candidate
/// (Issue #1438): 2,730 decoded rows at 96 dimensions, 2,048 at 128. A
/// larger scratch is dropped after its candidate, so a thread holds at most
/// this much between searches.
const MAX_KEPT_SCRATCH_BYTES: usize = 1 << 20;

thread_local! {
    /// This thread's scratch for [`MaxSimQuery::max_sim`]: an `F16` or
    /// `Int8` document's decoded rows, or a `Binary` one's per-row factors.
    ///
    /// [`MaxSimQuery::max_sim`]: crate::vector::core::late_interaction::MaxSimQuery::max_sim
    static SCRATCH: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
}

/// Run `f` with this thread's scratch, reused across candidates and
/// searches instead of allocated per candidate, then keep the scratch
/// unless it grew past [`MAX_KEPT_SCRATCH_BYTES`].
///
/// The scratch is taken out of the thread-local while `f` runs, so a nested
/// call gets an empty one instead of aliasing it.
fn with_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    let mut scratch = SCRATCH.take();
    let result = f(&mut scratch);
    if scratch.capacity() * size_of::<f32>() <= MAX_KEPT_SCRATCH_BYTES {
        SCRATCH.set(scratch);
    }
    result
}

/// Rescore options checked against the schema, with a text query not yet
/// embedded.
pub(super) struct RescorePlan {
    window_size: usize,
    field: String,
    dimension: usize,
    /// Whether query vectors are L2-normalized (a cosine field).
    normalize: bool,
    query: PlannedQuery,
}

enum PlannedQuery {
    Vectors(Vec<Vector>),
    /// Text to embed with the field's embedder, as routed by `embedder`.
    Text {
        text: String,
        embedder: Arc<dyn Embedder>,
    },
}

impl RescorePlan {
    /// Check `options` against the schema and the rest of the request.
    ///
    /// Embeds nothing, so it can run under the schema lock.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The engine's vector embedder, which embeds a text
    ///   query.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when the window is outside
    /// `1..=RescoreOptions::MAX_WINDOW_SIZE`, the request also sorts by a
    /// field, the target field is not a multi-vector field, or a text query
    /// is blank or has no token-level embedder for the field.
    pub(super) fn new(
        schema: &Schema,
        embedder: Arc<dyn Embedder>,
        options: RescoreOptions,
        sort_by: &SortField,
    ) -> Result<Self> {
        if !(1..=RescoreOptions::MAX_WINDOW_SIZE).contains(&options.window_size) {
            return Err(invalid(format!(
                "window_size must be between 1 and {}, got {}",
                RescoreOptions::MAX_WINDOW_SIZE,
                options.window_size
            )));
        }
        if matches!(sort_by, SortField::Field { .. }) {
            return Err(invalid(
                "cannot be combined with sort_by, which orders results by a field instead of \
                 by score"
                    .to_string(),
            ));
        }

        let Rescorer::LateInteraction { field, query } = options.rescorer;
        let field_option = match schema.fields.get(&field) {
            Some(FieldOption::MultiVector(option)) => option,
            Some(_) => {
                return Err(invalid(format!(
                    "late interaction needs a MultiVector field, but '{field}' is not one"
                )));
            }
            None => return Err(invalid(format!("unknown field '{field}'"))),
        };

        let query = match query {
            LateInteractionQuery::Vectors(vectors) => PlannedQuery::Vectors(vectors),
            LateInteractionQuery::Text(text) => {
                if text.trim().is_empty() {
                    return Err(invalid("the query text is empty".to_string()));
                }
                let field_embedder = resolve_field_embedder(&embedder, &field);
                if field_embedder.as_token_embedder().is_none() {
                    return Err(invalid(format!(
                        "field '{field}' has no token-level embedder to embed the query text \
                         (its embedder is '{}')",
                        field_embedder.name()
                    )));
                }
                PlannedQuery::Text { text, embedder }
            }
        };

        Ok(Self {
            window_size: options.window_size,
            dimension: field_option.dimension,
            normalize: field_option.distance == DistanceMetric::Cosine,
            field,
            query,
        })
    }

    /// Embed a text query and check the query vectors.
    ///
    /// Must not run under the schema lock: it awaits the embedder.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when the query does not hold
    /// `1..=1024` finite vectors of the field's dimension, and the
    /// embedder's error when embedding a text query fails.
    pub(super) async fn prepare(
        self,
        cache: Option<&Arc<EmbeddingCache>>,
    ) -> Result<PreparedRescore> {
        let query = match &self.query {
            PlannedQuery::Vectors(vectors) => self.flatten(vectors)?,
            PlannedQuery::Text { text, embedder } => {
                let tokens = embed_tokens_with_cache(
                    cache,
                    embedder,
                    &self.field,
                    &EmbedInput::Text(text),
                    EmbedRole::Query,
                )
                .await?;
                self.flatten(&tokens)?
            }
        };
        Ok(PreparedRescore {
            window_size: self.window_size,
            field: self.field,
            query,
            dimension: self.dimension,
        })
    }

    /// Check the query vectors and lay them out row-major, normalized for a
    /// cosine field.
    fn flatten(&self, vectors: &[Vector]) -> Result<Vec<f32>> {
        if vectors.is_empty() || vectors.len() > MAX_QUERY_VECTORS {
            return Err(invalid(format!(
                "late interaction needs between 1 and {MAX_QUERY_VECTORS} query vectors, got {}",
                vectors.len()
            )));
        }
        let (field, dimension) = (&self.field, self.dimension);
        let mut flat = Vec::with_capacity(vectors.len() * dimension);
        for (i, vector) in vectors.iter().enumerate() {
            if vector.dimension() != dimension {
                return Err(invalid(format!(
                    "query vector {i} has dimension {}, but field '{field}' has {dimension}",
                    vector.dimension()
                )));
            }
            if !vector.is_valid() {
                return Err(invalid(format!(
                    "query vector {i} contains a non-finite value"
                )));
            }
            if self.normalize {
                flat.extend_from_slice(&vector.normalized().data);
            } else {
                flat.extend_from_slice(&vector.data);
            }
        }
        Ok(flat)
    }
}

/// Late-interaction scores kept across calls of [`PreparedRescore::apply`]
/// within one search (Issue #1297).
#[derive(Debug, Default)]
pub(super) struct RescoreScores {
    /// The field's snapshot, taken on the first call that has candidates.
    snapshot: Option<MultiVectorSnapshot>,
    /// Each scored candidate's score; `None` for one without token vectors.
    scores: ahash::AHashMap<u64, Option<f32>>,
    /// How many candidates were scored over every call.
    #[cfg(test)]
    pub(super) scored: usize,
}

/// Validated rescore options, ready to apply.
#[derive(Debug)]
pub(super) struct PreparedRescore {
    window_size: usize,
    field: String,
    /// Query token vectors, row-major; L2-normalized for a cosine field.
    query: Vec<f32>,
    dimension: usize,
}

impl PreparedRescore {
    /// How many first-stage candidates to rank: the window, or the
    /// requested page when it reaches further.
    pub(super) fn depth(&self, fetch_count: usize) -> usize {
        self.window_size.max(fetch_count)
    }

    /// Reorder the top of `ranked` (best first).
    ///
    /// The window is sorted by late-interaction score (ties by doc id).
    /// Window candidates without token vectors follow in their first-stage
    /// order with their first-stage score, then every candidate beyond the
    /// window, unchanged.
    ///
    /// Scores go through `scores`: a candidate that an earlier call with the
    /// same `scores` already scored keeps that score, and every call reads
    /// the field's snapshot taken by the first one. A grouped search (Issue
    /// #1297) reranks a growing candidate pool over several rounds, and its
    /// window shifts as the first stage widens, so only candidates new to the
    /// window are scored. Any other search passes fresh `scores`.
    ///
    /// # Errors
    ///
    /// Returns an error when the field's segments fail to load or no longer
    /// match the dimension the query was validated against.
    pub(super) fn apply(
        &self,
        store: &VectorStore,
        mut ranked: Vec<(u64, f32)>,
        scores: &mut RescoreScores,
    ) -> Result<Vec<(u64, f32)>> {
        let tail = ranked.split_off(self.window_size.min(ranked.len()));
        if ranked.is_empty() {
            return Ok(tail);
        }

        let snapshot = match scores.snapshot.take() {
            Some(snapshot) => snapshot,
            None => {
                let taken = store.multi_vector_snapshot(&self.field)?;
                if taken.dimension() != self.dimension {
                    return Err(LaurusError::InvalidOperation(format!(
                        "field '{}' changed dimension during the search",
                        self.field
                    )));
                }
                taken
            }
        };
        let snapshot = &*scores.snapshot.insert(snapshot);
        let unseen: Vec<(u64, f32)> = ranked
            .iter()
            .filter(|(doc_id, _)| !scores.scores.contains_key(doc_id))
            .copied()
            .collect();
        if !unseen.is_empty() {
            for ((doc_id, _), score) in unseen.iter().zip(self.score(snapshot, &unseen)?) {
                scores.scores.insert(*doc_id, score);
            }
        }
        #[cfg(test)]
        {
            scores.scored += unseen.len();
        }

        let mut rescored = Vec::with_capacity(ranked.len() + tail.len());
        let mut unscored = Vec::new();
        for (doc_id, first_stage) in ranked {
            match scores.scores.get(&doc_id).copied().flatten() {
                Some(score) => rescored.push((doc_id, score)),
                None => unscored.push((doc_id, first_stage)),
            }
        }
        rescored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        rescored.extend(unscored);
        rescored.extend(tail);
        Ok(rescored)
    }

    /// Late-interaction score of each candidate, `None` for one without
    /// token vectors. The query is prepared for the snapshot once, then
    /// candidates are scored in parallel on native builds, each thread
    /// reusing its own scratch ([`with_scratch`]).
    fn score(
        &self,
        snapshot: &MultiVectorSnapshot,
        candidates: &[(u64, f32)],
    ) -> Result<Vec<Option<f32>>> {
        let query = snapshot.max_sim_query(&self.query);
        let score_one = |&(doc_id, _): &(u64, f32)| -> Result<Option<f32>> {
            let Some(document) = snapshot.rows(doc_id)? else {
                return Ok(None);
            };
            Ok(Some(with_scratch(|scratch| {
                query.max_sim(&document, scratch)
            })))
        };
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            candidates.par_iter().map(score_one).collect()
        }
        #[cfg(not(feature = "native"))]
        {
            candidates.iter().map(score_one).collect()
        }
    }
}

fn invalid(reason: String) -> LaurusError {
    LaurusError::invalid_argument(format!("rescore: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #1438: a scratch up to the cap is handed to the next call on
    /// the same thread, capacity included.
    #[test]
    fn test_with_scratch_keeps_a_scratch_up_to_the_cap() {
        let floats = MAX_KEPT_SCRATCH_BYTES / size_of::<f32>();
        with_scratch(|scratch| scratch.reserve_exact(floats));
        let kept = with_scratch(|scratch| scratch.capacity());
        assert!(kept >= floats, "kept a capacity of {kept} floats");
    }

    /// Issue #1438: a scratch grown past the cap is dropped, so the thread
    /// does not hold it until its next rescore.
    #[test]
    fn test_with_scratch_drops_a_scratch_past_the_cap() {
        let floats = MAX_KEPT_SCRATCH_BYTES / size_of::<f32>() + 1;
        with_scratch(|scratch| scratch.reserve_exact(floats));
        assert_eq!(with_scratch(|scratch| scratch.capacity()), 0);
    }

    /// A nested call gets an empty scratch of its own, not the outer one.
    #[test]
    fn test_with_scratch_nested_call_gets_an_empty_scratch() {
        with_scratch(|outer| {
            outer.extend([1.0, 2.0]);
            with_scratch(|inner| assert!(inner.is_empty()));
            assert_eq!(outer, &[1.0, 2.0]);
        });
    }
}
