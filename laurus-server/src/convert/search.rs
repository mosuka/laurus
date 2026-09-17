//! Conversion between search-related laurus domain types and protobuf types.
//!
//! [`from_proto`] builds a [`laurus::SearchRequest`] from the incoming proto
//! message, mapping the `query` field to [`SearchQuery::Dsl`] so the engine
//! can parse unified query DSL (including vector clauses) internally.
//! [`result_to_proto`] converts engine results back to proto.

use std::collections::HashMap;

use laurus::vector::Vector;
use laurus::{
    FusionAlgorithm, HighlightConfig, LexicalSearchQuery, QueryVector, SearchRequestBuilder,
    SearchResult, SortField, SortOrder, VectorScoreMode, VectorSearchQuery,
};

use crate::convert::document;
use crate::proto::laurus::v1;

/// Build a laurus SearchRequest from a proto SearchRequest.
///
/// The proto `query` field is mapped to [`SearchQuery::Dsl`] so the engine
/// handles unified query DSL parsing (including vector clauses) internally.
///
/// When `lexical_params` or `field_boosts` are provided, the query is
/// wrapped as [`LexicalSearchQuery::Dsl`] and lexical options are set
/// directly on the builder.
#[allow(clippy::result_large_err)]
pub fn from_proto(proto: &v1::SearchRequest) -> Result<laurus::SearchRequest, tonic::Status> {
    let mut builder = SearchRequestBuilder::new();

    let has_lexical_overrides = proto.lexical_params.is_some() || !proto.field_boosts.is_empty();

    if !proto.query.is_empty() {
        if has_lexical_overrides {
            // Wrap query as LexicalSearchQuery::Dsl so that lexical options
            // are preserved via builder methods.
            builder = builder.lexical_query(LexicalSearchQuery::Dsl(proto.query.clone()));

            // Apply field boosts
            for (field, boost) in &proto.field_boosts {
                builder = builder.add_field_boost(field.clone(), *boost);
            }

            // Apply lexical params
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
        } else {
            // Use the DSL variant — engine will parse with UnifiedQueryParser
            builder = builder.query_dsl(proto.query.clone());
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

        // Apply vector params
        if let Some(vp) = &proto.vector_params {
            let score_mode = match v1::VectorScoreMode::try_from(vp.score_mode) {
                Ok(v1::VectorScoreMode::MaxSim) => VectorScoreMode::MaxSim,
                Ok(v1::VectorScoreMode::LateInteraction) => VectorScoreMode::LateInteraction,
                _ => VectorScoreMode::WeightedSum,
            };
            builder = builder.vector_score_mode(score_mode);
            if vp.min_score > 0.0 {
                builder = builder.vector_min_score(vp.min_score);
            }
            // Issue #481 Stage 2: forward rerank_factor to the engine
            // (which forwards it to the HNSW searcher). Per-field
            // capability checks happen later: HNSW fields with
            // rerank_storage configured honor the value; everything
            // else silently ignores it. A zero value disables rerank
            // (defensive: matches `None` semantics).
            if let Some(factor) = vp.rerank_factor
                && factor > 0
            {
                builder = builder.vector_rerank_factor(factor as usize);
            }
        }
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

    Ok(builder.build())
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
}
