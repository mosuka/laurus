//! The rescore stage of [`Engine::search`](super::Engine::search)
//! (Issue #1345).
//!
//! The first stage ranks candidates as usual; this stage reorders the top
//! `window_size` of them before the page is cut. Options are validated
//! against the schema before any search runs, so a bad request fails fast.

use crate::engine::schema::{FieldOption, Schema};
use crate::engine::search::{LateInteractionQuery, RescoreOptions, Rescorer};
use crate::error::{LaurusError, Result};
use crate::lexical::search::searcher::SortField;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::late_interaction::max_sim;
use crate::vector::index::multivector::MultiVectorSnapshot;
use crate::vector::store::VectorStore;

/// Most query token vectors a late-interaction rescore accepts (ColBERT
/// queries are typically 32 tokens).
const MAX_QUERY_VECTORS: usize = 1024;

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
    /// Check `options` against the schema and the rest of the request.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when the window is outside
    /// `1..=RescoreOptions::MAX_WINDOW_SIZE`, the request also sorts by a
    /// field, the target field is not a multi-vector field, or the query
    /// does not hold `1..=1024` finite vectors of the field's dimension.
    pub(super) fn prepare(
        schema: &Schema,
        options: &RescoreOptions,
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

        let Rescorer::LateInteraction { field, query } = &options.rescorer;
        let field_option = match schema.fields.get(field) {
            Some(FieldOption::MultiVector(option)) => option,
            Some(_) => {
                return Err(invalid(format!(
                    "late interaction needs a MultiVector field, but '{field}' is not one"
                )));
            }
            None => return Err(invalid(format!("unknown field '{field}'"))),
        };
        let LateInteractionQuery::Vectors(vectors) = query;
        if vectors.is_empty() || vectors.len() > MAX_QUERY_VECTORS {
            return Err(invalid(format!(
                "late interaction needs between 1 and {MAX_QUERY_VECTORS} query vectors, got {}",
                vectors.len()
            )));
        }

        let dimension = field_option.dimension;
        let normalize = field_option.distance == DistanceMetric::Cosine;
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
            if normalize {
                flat.extend_from_slice(&vector.normalized().data);
            } else {
                flat.extend_from_slice(&vector.data);
            }
        }

        Ok(Self {
            window_size: options.window_size,
            field: field.clone(),
            query: flat,
            dimension,
        })
    }

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
    /// # Errors
    ///
    /// Returns an error when the field's segments fail to load or no longer
    /// match the dimension the query was validated against.
    pub(super) fn apply(
        &self,
        store: &VectorStore,
        mut ranked: Vec<(u64, f32)>,
    ) -> Result<Vec<(u64, f32)>> {
        let tail = ranked.split_off(self.window_size.min(ranked.len()));
        if ranked.is_empty() {
            return Ok(tail);
        }

        let snapshot = store.multi_vector_snapshot(&self.field)?;
        if snapshot.dimension() != self.dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "field '{}' changed dimension during the search",
                self.field
            )));
        }
        let scores = self.score(&snapshot, &ranked)?;

        let mut rescored = Vec::with_capacity(ranked.len() + tail.len());
        let mut unscored = Vec::new();
        for ((doc_id, first_stage), score) in ranked.into_iter().zip(scores) {
            match score {
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
    /// token vectors. Candidates are scored in parallel on native builds.
    fn score(
        &self,
        snapshot: &MultiVectorSnapshot,
        candidates: &[(u64, f32)],
    ) -> Result<Vec<Option<f32>>> {
        let score_one = |&(doc_id, _): &(u64, f32)| -> Result<Option<f32>> {
            Ok(snapshot
                .vectors(doc_id)?
                .map(|document| max_sim(&self.query, &document, self.dimension)))
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
