//! Conversion between search-related laurus domain types and protobuf types.
//!
//! [`from_proto`] builds a [`laurus::SearchRequest`] from the incoming proto
//! message, mapping the `query` field to [`SearchQuery::Dsl`] so the engine
//! can parse unified query DSL (including vector clauses) internally.
//! [`result_to_proto`] converts engine results back to proto.

use std::collections::HashMap;

use laurus::vector::Vector;
use laurus::vector::store::request::FieldSelector;
use laurus::{
    FusionAlgorithm, HighlightConfig, QueryVector, RescoreOptions, SearchRequestBuilder,
    SearchResult, SortField, SortOrder, VectorScoreMode, VectorSearchQuery,
};

use crate::convert::document;
use crate::proto::laurus::v1;

/// Build a laurus SearchRequest from a proto SearchRequest.
///
/// The proto `query` field is always the unified query DSL
/// ([`SearchQuery::Dsl`](laurus::SearchQuery::Dsl)): the engine parses its
/// lexical and vector clauses. `query_vectors` are added to its vector part
/// (Issue #1366). `lexical_params` and `field_boosts` set the lexical options
/// of a request that has a `query`.
#[allow(clippy::result_large_err)]
pub fn from_proto(proto: &v1::SearchRequest) -> Result<laurus::SearchRequest, tonic::Status> {
    let mut builder = SearchRequestBuilder::new();

    if !proto.query.is_empty() {
        builder = builder.query_dsl(proto.query.clone());

        for (field, boost) in &proto.field_boosts {
            builder = builder.add_field_boost(field.clone(), *boost);
        }

        if let Some(p) = &proto.lexical_params {
            builder = builder.lexical_min_score(p.min_score);
            if let Some(timeout_ms) = p.timeout_ms
                && timeout_ms > 0
            {
                builder = builder.lexical_timeout_ms(timeout_ms);
            }
            if p.parallel {
                builder = builder.lexical_parallel(true);
            }
            if let Some(spec) = &p.sort_by
                && !spec.field.is_empty()
            {
                let order = match v1::SortOrder::try_from(spec.order) {
                    Ok(v1::SortOrder::Desc) => SortOrder::Desc,
                    _ => SortOrder::Asc,
                };
                builder = builder.sort_by(SortField::Field {
                    name: spec.field.clone(),
                    order,
                });
            }
        }
    }

    // Explicit pre-embedded vectors
    if !proto.query_vectors.is_empty() {
        let query_vectors: Vec<QueryVector> = proto
            .query_vectors
            .iter()
            .map(|qv| QueryVector {
                vector: Vector::new(qv.vector.clone()),
                weight: if qv.weight == 0.0 { 1.0 } else { qv.weight },
                fields: if qv.fields.is_empty() {
                    None
                } else {
                    Some(qv.fields.clone())
                },
            })
            .collect();

        builder = builder.vector_query(VectorSearchQuery::Vectors(query_vectors));
    }

    if let Some(vp) = &proto.vector_params {
        builder = apply_vector_params(builder, vp, proto)?;
    }

    // Limit and offset
    if proto.limit > 0 {
        builder = builder.limit(proto.limit as usize);
    }
    builder = builder.offset(proto.offset as usize);

    // Fusion
    if let Some(fusion) = &proto.fusion
        && let Some(alg) = &fusion.algorithm
    {
        let fusion_alg = match alg {
            v1::fusion_algorithm::Algorithm::Rrf(rrf) => FusionAlgorithm::RRF { k: rrf.k },
            v1::fusion_algorithm::Algorithm::WeightedSum(ws) => FusionAlgorithm::WeightedSum {
                lexical_weight: ws.lexical_weight,
                vector_weight: ws.vector_weight,
            },
        };
        builder = builder.fusion_algorithm(fusion_alg);
    }

    // Highlighting (Issue #1134). Independent of has_lexical_overrides: it
    // applies to whichever lexical query the request carries (DSL or the
    // overridden LexicalSearchQuery::Dsl above).
    if let Some(highlight) = &proto.highlight {
        if highlight.fields.is_empty() {
            return Err(tonic::Status::invalid_argument(
                "highlight.fields must not be empty",
            ));
        }
        builder = builder.highlight(highlight.fields.clone());
        if has_highlight_config(highlight) {
            builder = builder.highlight_config(highlight_config_from_proto(highlight)?);
        }
    }

    // Rescoring (Issue #1351). The engine validates the window, the field
    // and the query; only the message shape is checked here.
    if let Some(rescore) = &proto.rescore {
        builder = builder.rescore(rescore_from_proto(rescore)?);
    }

    Ok(builder.build())
}

/// Apply proto `VectorParams` to the builder (Issue #1342).
///
/// Every field either takes effect or fails the request: nothing is
/// silently ignored. proto3 cannot tell an unset `float` from `0.0`, so a
/// zero `min_score` sets no threshold and a zero `overfetch` keeps the
/// engine default.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` when the request has no vector part, when
/// `fields` would reach no query vector, or when a value is out of range.
#[allow(clippy::result_large_err)]
fn apply_vector_params(
    mut builder: SearchRequestBuilder,
    vp: &v1::VectorParams,
    proto: &v1::SearchRequest,
) -> Result<SearchRequestBuilder, tonic::Status> {
    let invalid =
        |message: String| tonic::Status::invalid_argument(format!("vector_params: {message}"));

    // The vector part is `query_vectors` plus any vector clauses of `query`
    // (Issue #1366); the server cannot tell whether `query` has vector
    // clauses without parsing it, so the params are applied to them.
    if proto.query_vectors.is_empty() && proto.query.is_empty() {
        return Err(invalid(
            "the request has no vector part (no query_vectors and no query)".to_string(),
        ));
    }

    let score_mode = match v1::VectorScoreMode::try_from(vp.score_mode) {
        Ok(v1::VectorScoreMode::WeightedSum) => VectorScoreMode::WeightedSum,
        Ok(v1::VectorScoreMode::MaxSim) => VectorScoreMode::MaxSim,
        Ok(v1::VectorScoreMode::LateInteraction) => VectorScoreMode::LateInteraction,
        Err(_) => return Err(invalid(format!("unknown score_mode {}", vp.score_mode))),
    };
    builder = builder.vector_score_mode(score_mode);

    if !vp.min_score.is_finite() {
        return Err(invalid(format!(
            "min_score must be finite, got {}",
            vp.min_score
        )));
    }
    if vp.min_score > 0.0 {
        builder = builder.vector_min_score(vp.min_score);
    }

    // Issue #481 Stage 2: HNSW fields with rerank_storage honor the
    // factor; other fields fall back to Stage 1 ranking. A zero value
    // disables rerank, matching `None`.
    if let Some(factor) = vp.rerank_factor
        && factor > 0
    {
        builder = builder.vector_rerank_factor(factor as usize);
    }

    match vp.ef_search {
        Some(0) => return Err(invalid("ef_search must be greater than 0".to_string())),
        Some(ef) => builder = builder.vector_ef_search(ef as usize),
        None => {}
    }

    if !vp.overfetch.is_finite() || vp.overfetch < 0.0 {
        return Err(invalid(format!(
            "overfetch must be a finite, non-negative number, got {}",
            vp.overfetch
        )));
    }
    if vp.overfetch > 0.0 {
        builder = builder.vector_overfetch(vp.overfetch);
    }

    if !vp.fields.is_empty() {
        // A query vector's own fields win, and every DSL vector clause
        // names its field, so `fields` reaches only query_vectors entries
        // that name none.
        if !proto.query_vectors.iter().any(|qv| qv.fields.is_empty()) {
            return Err(invalid(
                "fields applies only to query_vectors entries that name no field; \
                 DSL vector clauses and query_vectors with their own fields ignore it"
                    .to_string(),
            ));
        }
        builder = builder.vector_fields(
            vp.fields
                .iter()
                .cloned()
                .map(FieldSelector::Exact)
                .collect(),
        );
    }

    Ok(builder)
}

/// Build [`RescoreOptions`] from proto `RescoreParams`.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` when no rescorer or no late-interaction query
/// is set.
#[allow(clippy::result_large_err)]
fn rescore_from_proto(params: &v1::RescoreParams) -> Result<RescoreOptions, tonic::Status> {
    let Some(v1::rescore_params::Rescorer::LateInteraction(late)) = &params.rescorer else {
        return Err(tonic::Status::invalid_argument(
            "rescore.late_interaction must be set",
        ));
    };
    let options = match &late.query {
        Some(v1::late_interaction_rescore::Query::Vectors(vectors)) => {
            let vectors = document::vector_array_from_proto(vectors)
                .into_iter()
                .map(Vector::new)
                .collect();
            RescoreOptions::late_interaction(late.field.clone(), vectors)
        }
        Some(v1::late_interaction_rescore::Query::Text(text)) => {
            RescoreOptions::late_interaction_text(late.field.clone(), text.clone())
        }
        None => {
            return Err(tonic::Status::invalid_argument(
                "rescore.late_interaction needs vectors or text",
            ));
        }
    };
    Ok(match params.window_size {
        Some(window_size) => options.window_size(window_size as usize),
        None => options,
    })
}

/// Parse the JSON form of [`v1::RescoreParams`], shared by the HTTP
/// gateway's `POST /v1/search` and the MCP `search` tool (Issue #1351):
///
/// ```json
/// {"window_size": 100,
///  "late_interaction": {"field": "body_colbert", "vectors": [[0.1, 0.2], [0.3, 0.4]]}}
/// ```
///
/// `late_interaction` takes exactly one of `vectors` (equal-length numeric
/// arrays) and `text`; `window_size` is optional. Unlike the other search
/// options, a malformed value is an error rather than ignored: the caller
/// asked for a rescore explicitly, and dropping it would silently return the
/// first-stage ranking.
///
/// # Errors
///
/// Returns a message describing the first malformed part.
pub fn rescore_params_from_json(json: &serde_json::Value) -> Result<v1::RescoreParams, String> {
    use serde_json::Value;

    let obj = json.as_object().ok_or("rescore must be an object")?;
    let window_size = match obj.get("window_size") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("rescore.window_size must be a non-negative integer")?,
        ),
    };
    let late = obj
        .get("late_interaction")
        .ok_or("rescore.late_interaction is required")?
        .as_object()
        .ok_or("rescore.late_interaction must be an object")?;
    let field = late
        .get("field")
        .and_then(Value::as_str)
        .ok_or("rescore.late_interaction.field must be a string")?
        .to_string();
    let query = match (late.get("vectors"), late.get("text")) {
        (Some(vectors), None) => {
            v1::late_interaction_rescore::Query::Vectors(token_vectors_from_json(vectors)?)
        }
        (None, Some(text)) => v1::late_interaction_rescore::Query::Text(
            text.as_str()
                .ok_or("rescore.late_interaction.text must be a string")?
                .to_string(),
        ),
        _ => {
            return Err(
                "rescore.late_interaction needs exactly one of vectors and text".to_string(),
            );
        }
    };
    Ok(v1::RescoreParams {
        window_size,
        rescorer: Some(v1::rescore_params::Rescorer::LateInteraction(
            v1::LateInteractionRescore {
                field,
                query: Some(query),
            },
        )),
    })
}

/// Pack a JSON array of equal-length numeric arrays row-major.
///
/// An empty array packs to no vectors, which the engine rejects with its
/// own message.
fn token_vectors_from_json(json: &serde_json::Value) -> Result<v1::VectorArrayValue, String> {
    const WHAT: &str = "rescore.late_interaction.vectors";
    let rows = json
        .as_array()
        .ok_or_else(|| format!("{WHAT} must be an array of numeric arrays"))?;
    let dimension = rows
        .first()
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let mut values = Vec::with_capacity(rows.len() * dimension);
    for (i, row) in rows.iter().enumerate() {
        let row = row
            .as_array()
            .ok_or_else(|| format!("{WHAT}: element {i} is not an array"))?;
        if row.is_empty() || row.len() != dimension {
            return Err(format!(
                "{WHAT}: vector {i} has {} values, expected {dimension}",
                row.len()
            ));
        }
        for value in row {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("{WHAT}: vector {i} holds a non-number"))?;
            values.push(number as f32);
        }
    }
    Ok(v1::VectorArrayValue {
        dimension: dimension as u32,
        values,
    })
}

/// Whether `params` sets any field beyond `fields`, i.e. whether a
/// non-default [`HighlightConfig`] needs to be built at all.
fn has_highlight_config(params: &v1::HighlightParams) -> bool {
    params.max_fragments.is_some()
        || params.fragment_size.is_some()
        || params.tag.as_deref().is_some_and(|s| !s.is_empty())
        || params.css_class.as_deref().is_some_and(|s| !s.is_empty())
        || params.require_field_match.is_some()
        || params.max_analyzed_chars.is_some()
        || params.return_entire_field_if_no_highlight.is_some()
}

/// Build a [`HighlightConfig`] from proto `HighlightParams`, layering set
/// fields onto [`HighlightConfig::default`]. `max_fragments` and
/// `fragment_size` reject zero (a zero-sized budget can never highlight
/// anything, which almost certainly indicates a client bug); an empty `tag`
/// or `css_class` is treated as unset rather than rejected, since an empty
/// tag is comparatively harmless and DSL-adjacent tooling may round-trip an
/// unset optional string as `""`.
#[allow(clippy::result_large_err)]
fn highlight_config_from_proto(
    params: &v1::HighlightParams,
) -> Result<HighlightConfig, tonic::Status> {
    let mut config = HighlightConfig::default();

    if let Some(max_fragments) = params.max_fragments {
        if max_fragments == 0 {
            return Err(tonic::Status::invalid_argument(
                "highlight.max_fragments must be greater than zero",
            ));
        }
        config = config.max_fragments(max_fragments as usize);
    }
    if let Some(fragment_size) = params.fragment_size {
        if fragment_size == 0 {
            return Err(tonic::Status::invalid_argument(
                "highlight.fragment_size must be greater than zero",
            ));
        }
        config = config.fragment_size(fragment_size as usize);
    }
    if let Some(tag) = &params.tag
        && !tag.is_empty()
    {
        config = config.tag(tag.clone());
    }
    if let Some(css_class) = &params.css_class
        && !css_class.is_empty()
    {
        config = config.css_class(css_class.clone());
    }
    if let Some(require_field_match) = params.require_field_match {
        config = config.require_field_match(require_field_match);
    }
    if let Some(max_analyzed_chars) = params.max_analyzed_chars {
        config.max_analyzed_chars = max_analyzed_chars as usize;
    }
    if let Some(return_entire) = params.return_entire_field_if_no_highlight {
        config.return_entire_field_if_no_highlight = return_entire;
    }

    Ok(config)
}

/// Convert a laurus SearchResult into a proto SearchResult.
pub fn result_to_proto(result: &SearchResult) -> v1::SearchResult {
    v1::SearchResult {
        id: result.id.clone(),
        score: result.score,
        document: result.document.as_ref().map(document::to_proto),
        highlights: result
            .highlights
            .iter()
            .map(|(field, fragments)| {
                (
                    field.clone(),
                    v1::Highlights {
                        fragments: fragments.clone(),
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request(highlight: Option<v1::HighlightParams>) -> v1::SearchRequest {
        v1::SearchRequest {
            query: "title:rust".to_string(),
            highlight,
            ..Default::default()
        }
    }

    #[test]
    fn from_proto_without_highlight_leaves_it_unset() {
        let request = from_proto(&base_request(None)).unwrap();
        assert!(request.lexical_options.highlight.is_none());
    }

    fn late_interaction(
        window_size: Option<u32>,
        query: Option<v1::late_interaction_rescore::Query>,
    ) -> v1::SearchRequest {
        v1::SearchRequest {
            rescore: Some(v1::RescoreParams {
                window_size,
                rescorer: Some(v1::rescore_params::Rescorer::LateInteraction(
                    v1::LateInteractionRescore {
                        field: "tokens".to_string(),
                        query,
                    },
                )),
            }),
            ..base_request(None)
        }
    }

    /// #1351: packed token vectors and query text become the engine's
    /// rescore options, with the default window when none is given.
    #[test]
    fn from_proto_builds_late_interaction_rescore() {
        let vectors = v1::late_interaction_rescore::Query::Vectors(v1::VectorArrayValue {
            dimension: 2,
            values: vec![1.0, 0.0, 0.0, 1.0],
        });
        let request = from_proto(&late_interaction(None, Some(vectors))).unwrap();
        let rescore = request.rescore.expect("rescore is set");
        assert_eq!(rescore.window_size, RescoreOptions::DEFAULT_WINDOW_SIZE);
        match rescore.rescorer {
            laurus::Rescorer::LateInteraction {
                field,
                query: laurus::LateInteractionQuery::Vectors(vectors),
            } => {
                assert_eq!(field, "tokens");
                assert_eq!(vectors.len(), 2);
                assert_eq!(vectors[1].data.as_slice(), [0.0, 1.0]);
            }
            other => panic!("unexpected rescorer {other:?}"),
        }

        let text = v1::late_interaction_rescore::Query::Text("rust".to_string());
        let rescore = from_proto(&late_interaction(Some(7), Some(text)))
            .unwrap()
            .rescore
            .unwrap();
        assert_eq!(rescore.window_size, 7);
        assert!(matches!(
            rescore.rescorer,
            laurus::Rescorer::LateInteraction {
                query: laurus::LateInteractionQuery::Text(ref t),
                ..
            } if t == "rust"
        ));
    }

    #[test]
    fn from_proto_rejects_a_rescore_without_rescorer_or_query() {
        let mut no_rescorer = base_request(None);
        no_rescorer.rescore = Some(v1::RescoreParams {
            window_size: None,
            rescorer: None,
        });
        for request in [no_rescorer, late_interaction(None, None)] {
            let Err(status) = from_proto(&request) else {
                panic!("expected INVALID_ARGUMENT for {request:?}");
            };
            assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
        }
    }

    #[test]
    fn rescore_params_from_json_parses_vectors_and_text() {
        let params = rescore_params_from_json(&serde_json::json!({
            "window_size": 50,
            "late_interaction": {"field": "tokens", "vectors": [[1, 0.5], [0, 1]]}
        }))
        .unwrap();
        assert_eq!(params.window_size, Some(50));
        let Some(v1::rescore_params::Rescorer::LateInteraction(late)) = params.rescorer else {
            panic!("late interaction expected");
        };
        assert_eq!(late.field, "tokens");
        assert_eq!(
            late.query,
            Some(v1::late_interaction_rescore::Query::Vectors(
                v1::VectorArrayValue {
                    dimension: 2,
                    values: vec![1.0, 0.5, 0.0, 1.0],
                }
            ))
        );

        let params = rescore_params_from_json(&serde_json::json!({
            "late_interaction": {"field": "tokens", "text": "rust"}
        }))
        .unwrap();
        assert_eq!(params.window_size, None);
    }

    #[test]
    fn rescore_params_from_json_rejects_malformed_values() {
        let cases = [
            (serde_json::json!([]), "must be an object"),
            (serde_json::json!({}), "late_interaction is required"),
            (
                serde_json::json!({"window_size": -1, "late_interaction": {"field": "t", "text": "q"}}),
                "window_size",
            ),
            (
                serde_json::json!({"late_interaction": {"text": "q"}}),
                "field must be a string",
            ),
            (
                serde_json::json!({"late_interaction": {"field": "t"}}),
                "exactly one of vectors and text",
            ),
            (
                serde_json::json!({"late_interaction": {"field": "t", "text": "q", "vectors": [[1]]}}),
                "exactly one of vectors and text",
            ),
            (
                serde_json::json!({"late_interaction": {"field": "t", "vectors": [[1, 0], [1]]}}),
                "vector 1 has 1 values, expected 2",
            ),
            (
                serde_json::json!({"late_interaction": {"field": "t", "vectors": [[1, "x"]]}}),
                "non-number",
            ),
            (
                serde_json::json!({"late_interaction": {"field": "t", "vectors": [1, 2]}}),
                "element 0 is not an array",
            ),
        ];
        for (json, expected) in cases {
            let err = rescore_params_from_json(&json).unwrap_err();
            assert!(err.contains(expected), "{json}: {err}");
        }
    }

    #[test]
    fn from_proto_with_highlight_sets_the_builder_option() {
        let proto = base_request(Some(v1::HighlightParams {
            fields: vec!["title".to_string(), "body".to_string()],
            ..Default::default()
        }));
        let request = from_proto(&proto).unwrap();
        let options = request.lexical_options.highlight.expect("highlight set");
        assert_eq!(options.fields, ["title", "body"]);
        // No config fields were set on the proto, so the default HighlightConfig applies.
        assert_eq!(options.config.tag, "mark");
        assert!(options.config.require_field_match);
    }

    #[test]
    fn from_proto_rejects_empty_highlight_fields() {
        let proto = base_request(Some(v1::HighlightParams::default()));
        match from_proto(&proto) {
            Err(err) => assert_eq!(err.code(), tonic::Code::InvalidArgument),
            Ok(_) => panic!("expected an error for empty highlight.fields"),
        }
    }

    #[test]
    fn from_proto_highlight_config_layers_settings_onto_defaults() {
        let proto = base_request(Some(v1::HighlightParams {
            fields: vec!["body".to_string()],
            max_fragments: Some(2),
            fragment_size: Some(80),
            tag: Some("em".to_string()),
            css_class: Some("hl".to_string()),
            require_field_match: Some(false),
            max_analyzed_chars: Some(500),
            return_entire_field_if_no_highlight: Some(true),
        }));
        let request = from_proto(&proto).unwrap();
        let options = request.lexical_options.highlight.expect("highlight set");
        assert_eq!(options.config.max_fragments, 2);
        assert_eq!(options.config.fragment_size, 80);
        assert_eq!(options.config.tag, "em");
        assert_eq!(options.config.css_class.as_deref(), Some("hl"));
        assert!(!options.config.require_field_match);
        assert_eq!(options.config.max_analyzed_chars, 500);
        assert!(options.config.return_entire_field_if_no_highlight);
    }

    #[test]
    fn from_proto_rejects_zero_max_fragments_and_zero_fragment_size() {
        let zero_fragments = base_request(Some(v1::HighlightParams {
            fields: vec!["body".to_string()],
            max_fragments: Some(0),
            ..Default::default()
        }));
        match from_proto(&zero_fragments) {
            Err(err) => assert_eq!(err.code(), tonic::Code::InvalidArgument),
            Ok(_) => panic!("expected an error for max_fragments = 0"),
        }

        let zero_size = base_request(Some(v1::HighlightParams {
            fields: vec!["body".to_string()],
            fragment_size: Some(0),
            ..Default::default()
        }));
        match from_proto(&zero_size) {
            Err(err) => assert_eq!(err.code(), tonic::Code::InvalidArgument),
            Ok(_) => panic!("expected an error for fragment_size = 0"),
        }
    }

    #[test]
    fn from_proto_treats_empty_tag_and_css_class_as_unset() {
        let proto = base_request(Some(v1::HighlightParams {
            fields: vec!["body".to_string()],
            tag: Some(String::new()),
            css_class: Some(String::new()),
            ..Default::default()
        }));
        let request = from_proto(&proto).unwrap();
        let options = request.lexical_options.highlight.expect("highlight set");
        assert_eq!(options.config.tag, "mark");
        assert_eq!(options.config.css_class, None);
    }

    #[test]
    fn result_to_proto_fills_the_highlights_map() {
        let mut highlights = HashMap::new();
        highlights.insert(
            "body".to_string(),
            vec!["<mark>Rust</mark> is great".to_string()],
        );
        let result = SearchResult {
            id: "doc1".to_string(),
            score: 1.5,
            document: None,
            highlights,
        };

        let proto = result_to_proto(&result);
        assert_eq!(
            proto.highlights["body"].fragments,
            ["<mark>Rust</mark> is great"]
        );
    }

    #[test]
    fn result_to_proto_omits_empty_highlights() {
        let result = SearchResult {
            id: "doc1".to_string(),
            score: 1.5,
            document: None,
            highlights: HashMap::new(),
        };

        let proto = result_to_proto(&result);
        assert!(proto.highlights.is_empty());
    }

    // ── vector_params (Issue #1342) ─────────────────────────────────────

    fn query_vector(fields: &[&str]) -> v1::QueryVector {
        v1::QueryVector {
            vector: vec![1.0, 0.0],
            weight: 1.0,
            fields: fields.iter().map(|f| f.to_string()).collect(),
        }
    }

    /// Every field set to a value distinct from its default.
    fn every_vector_param() -> v1::VectorParams {
        v1::VectorParams {
            fields: vec!["a".to_string()],
            score_mode: v1::VectorScoreMode::MaxSim as i32,
            overfetch: 3.0,
            min_score: 0.25,
            rerank_factor: Some(4),
            ef_search: Some(80),
        }
    }

    fn assert_fields(request: &laurus::SearchRequest, expected: &[&str]) {
        let fields = request
            .vector_options
            .fields
            .as_ref()
            .expect("fields must be set");
        let names: Vec<&str> = fields
            .iter()
            .map(|f| match f {
                FieldSelector::Exact(name) => name.as_str(),
                other => panic!("unexpected selector {other:?}"),
            })
            .collect();
        assert_eq!(names, expected);
    }

    fn assert_non_field_params(request: &laurus::SearchRequest) {
        let options = &request.vector_options;
        assert!(matches!(options.score_mode, VectorScoreMode::MaxSim));
        assert_eq!(options.min_score, 0.25);
        assert_eq!(options.rerank_factor, Some(4));
        assert_eq!(options.ef_search, Some(80));
        assert_eq!(options.overfetch, 3.0);
    }

    #[test]
    fn from_proto_applies_every_vector_param_to_query_vectors() {
        let request = from_proto(&v1::SearchRequest {
            query_vectors: vec![query_vector(&[])],
            vector_params: Some(every_vector_param()),
            ..Default::default()
        })
        .unwrap();
        assert_non_field_params(&request);
        assert_fields(&request, &["a"]);
    }

    /// The vector part of a DSL query gets `vector_params` too; `fields`
    /// is left out because a DSL vector clause names its own field.
    #[test]
    fn from_proto_applies_vector_params_to_a_dsl_query() {
        let request = from_proto(&v1::SearchRequest {
            query: "vec:\"rust\"".to_string(),
            vector_params: Some(v1::VectorParams {
                fields: Vec::new(),
                ..every_vector_param()
            }),
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(request.query, laurus::SearchQuery::Dsl(_)));
        assert_non_field_params(&request);
        assert!(request.vector_options.fields.is_none());
    }

    /// `fields` is accepted when some query vector names no field, also
    /// alongside a DSL `query` with or without lexical overrides (#1366).
    #[test]
    fn from_proto_accepts_fields_when_a_query_vector_names_none() {
        let mixed = v1::SearchRequest {
            query_vectors: vec![query_vector(&["b"]), query_vector(&[])],
            vector_params: Some(every_vector_param()),
            ..Default::default()
        };
        let with_overrides = v1::SearchRequest {
            query: "title:rust".to_string(),
            lexical_params: Some(v1::LexicalParams::default()),
            query_vectors: vec![query_vector(&[])],
            vector_params: Some(every_vector_param()),
            ..Default::default()
        };
        let without_overrides = v1::SearchRequest {
            query: "title:rust".to_string(),
            query_vectors: vec![query_vector(&[])],
            vector_params: Some(every_vector_param()),
            ..Default::default()
        };
        for proto in [mixed, with_overrides, without_overrides] {
            let request = from_proto(&proto).unwrap();
            assert_fields(&request, &["a"]);
        }
    }

    /// Issue #1366: `query_vectors` sent with a DSL `query` are kept and
    /// added to the DSL's vector part.
    #[test]
    fn from_proto_keeps_query_vectors_with_a_dsl_query() {
        let request = from_proto(&v1::SearchRequest {
            query: "title:rust".to_string(),
            query_vectors: vec![query_vector(&[])],
            ..Default::default()
        })
        .unwrap();
        assert!(
            matches!(
                request.query,
                laurus::SearchQuery::DslWithVectors {
                    ref dsl,
                    vectors: VectorSearchQuery::Vectors(ref v),
                } if dsl == "title:rust" && v.len() == 1
            ),
            "{:?}",
            request.query
        );
    }

    /// Issue #1366: `lexical_params` / `field_boosts` no longer turn `query`
    /// into a lexical-only query; it stays a unified DSL query and the
    /// lexical options still apply.
    #[test]
    fn from_proto_parses_query_as_dsl_with_lexical_overrides() {
        let request = from_proto(&v1::SearchRequest {
            query: "title:rust".to_string(),
            lexical_params: Some(v1::LexicalParams {
                min_score: 0.5,
                ..Default::default()
            }),
            field_boosts: [("title".to_string(), 2.0)].into_iter().collect(),
            ..Default::default()
        })
        .unwrap();
        assert!(
            matches!(request.query, laurus::SearchQuery::Dsl(ref d) if d == "title:rust"),
            "{:?}",
            request.query
        );
        assert_eq!(request.lexical_options.min_score, 0.5);
        assert_eq!(
            request.lexical_options.field_boosts.get("title"),
            Some(&2.0)
        );

        // vector_params without `fields` now apply to the DSL's vector clauses.
        let with_vector_params = from_proto(&v1::SearchRequest {
            query: "title:rust".to_string(),
            lexical_params: Some(v1::LexicalParams::default()),
            vector_params: Some(v1::VectorParams {
                fields: Vec::new(),
                ..every_vector_param()
            }),
            ..Default::default()
        })
        .unwrap();
        assert_non_field_params(&with_vector_params);
    }

    /// proto3 cannot tell an unset `float` from `0.0`: `overfetch == 0`
    /// keeps the engine default and `min_score == 0` sets no threshold.
    #[test]
    fn from_proto_treats_zero_floats_as_unset() {
        let request = from_proto(&v1::SearchRequest {
            query_vectors: vec![query_vector(&[])],
            vector_params: Some(v1::VectorParams::default()),
            ..Default::default()
        })
        .unwrap();
        let defaults = laurus::VectorSearchOptions::default();
        assert_eq!(request.vector_options.overfetch, defaults.overfetch);
        assert_eq!(request.vector_options.min_score, defaults.min_score);
        assert_eq!(request.vector_options.ef_search, None);
        assert!(request.vector_options.fields.is_none());
    }

    #[test]
    fn from_proto_rejects_vector_params_that_would_be_ignored() {
        let with = |f: fn(&mut v1::SearchRequest)| {
            let mut proto = v1::SearchRequest {
                query_vectors: vec![query_vector(&[])],
                vector_params: Some(every_vector_param()),
                ..Default::default()
            };
            f(&mut proto);
            proto
        };
        let cases = [
            ("no query at all", with(|p| p.query_vectors.clear())),
            (
                "fields with a DSL-only request",
                with(|p| {
                    p.query = "vec:\"rust\"".to_string();
                    p.query_vectors.clear();
                }),
            ),
            (
                "fields when every query vector names its own",
                with(|p| p.query_vectors = vec![query_vector(&["b"])]),
            ),
            (
                "unknown score_mode",
                with(|p| p.vector_params.as_mut().unwrap().score_mode = 99),
            ),
            (
                "NaN min_score",
                with(|p| p.vector_params.as_mut().unwrap().min_score = f32::NAN),
            ),
            (
                "negative overfetch",
                with(|p| p.vector_params.as_mut().unwrap().overfetch = -1.0),
            ),
            (
                "NaN overfetch",
                with(|p| p.vector_params.as_mut().unwrap().overfetch = f32::NAN),
            ),
            (
                "infinite overfetch",
                with(|p| p.vector_params.as_mut().unwrap().overfetch = f32::INFINITY),
            ),
            (
                "zero ef_search",
                with(|p| p.vector_params.as_mut().unwrap().ef_search = Some(0)),
            ),
        ];
        for (case, proto) in cases {
            let Err(status) = from_proto(&proto) else {
                panic!("{case}: expected INVALID_ARGUMENT");
            };
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "{case}: {status:?}"
            );
            assert!(
                status.message().contains("vector_params"),
                "{case}: {status:?}"
            );
        }
    }
}
