//! Unified query parser for hybrid search.
//!
//! Parses a single DSL string containing both lexical and vector query clauses
//! into a [`SearchRequest`] for the Engine.
//!
//! # Syntax
//!
//! Lexical and vector clauses can be freely mixed in a single query string.
//! Vector fields are identified by schema — any `field:value` clause where
//! `field` is a vector field is routed to the vector parser:
//!
//! - **Lexical**: Standard query syntax (`title:hello`, `"phrase"`, `AND`/`OR`, etc.)
//! - **Vector**: `field:"text"` or `field:text` where `field` is a vector field
//!
//! # Examples
//!
//! ```ignore
//! use laurus::engine::query::UnifiedQueryParser;
//!
//! let parser = UnifiedQueryParser::new(lexical_parser, vector_parser, vector_fields);
//!
//! // Hybrid search
//! let request = parser.parse(r#"title:hello content:"cute kitten"^0.8"#).await?;
//! assert!(matches!(request.query, SearchQuery::Hybrid { .. }));
//!
//! // Lexical only
//! let request = parser.parse("title:hello AND body:world").await?;
//! assert!(matches!(request.query, SearchQuery::Lexical(_)));
//!
//! // Vector only
//! let request = parser.parse(r#"content:"cats" image:"dogs"^0.5"#).await?;
//! assert!(matches!(request.query, SearchQuery::Vector(_)));
//! ```

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use crate::engine::search::{FusionAlgorithm, HybridMode, SearchQuery, SearchRequest};
use crate::error::{LaurusError, Result};
use crate::lexical::query::parser::LexicalQueryParser;
use crate::lexical::search::searcher::LexicalSearchQuery;
use crate::vector::query::parser::VectorQueryParser;

/// Unified query parser that composes lexical and vector parsers.
///
/// Parses a single DSL string into a [`SearchRequest`] by splitting the input
/// into lexical and vector portions and delegating to the appropriate sub-parser.
///
/// Vector clauses are identified by their field name: any `field:value` clause
/// where `field` matches a known vector field (from the schema) is routed to
/// the vector parser. This approach is unambiguous because field names are
/// unique within a schema and each field has a single type.
///
/// After vector clauses are extracted, any leftover dangling boolean operators
/// (`AND`, `OR`) at the edges or in consecutive positions are cleaned up
/// before the lexical portion is parsed.
pub struct UnifiedQueryParser {
    lexical_parser: LexicalQueryParser,
    vector_parser: VectorQueryParser,
    vector_fields: HashSet<String>,
    /// All field names declared in the schema (lexical + vector). When
    /// non-empty, any `field:value` clause referencing a field outside this
    /// set is rejected at parse time with a clear error.
    known_fields: HashSet<String>,
    default_fusion: FusionAlgorithm,
}

impl UnifiedQueryParser {
    /// Create a new `UnifiedQueryParser` with the given sub-parsers.
    ///
    /// The default fusion algorithm for hybrid queries is
    /// [`FusionAlgorithm::RRF { k: 60.0 }`](FusionAlgorithm::RRF).
    ///
    /// # Parameters
    ///
    /// - `lexical_parser` - Parser for lexical (text) query clauses.
    /// - `vector_parser` - Parser for vector query clauses.
    /// - `vector_fields` - Set of field names that are vector fields in the schema.
    pub fn new(
        lexical_parser: LexicalQueryParser,
        vector_parser: VectorQueryParser,
        vector_fields: HashSet<String>,
    ) -> Self {
        Self {
            lexical_parser,
            vector_parser,
            vector_fields,
            known_fields: HashSet::new(),
            default_fusion: FusionAlgorithm::RRF { k: 60.0 },
        }
    }

    /// Provide the set of all field names declared in the schema.
    ///
    /// When set (and non-empty), parsing a query that references a field name
    /// outside this set produces an error. This catches typos such as
    /// `titl:hello` early instead of returning silently-empty results.
    ///
    /// Callers should pass the union of lexical and vector field names. If
    /// the set is left empty (the default), no field-existence check is
    /// performed.
    ///
    /// # Arguments
    ///
    /// * `known_fields` - All declared field names.
    pub fn with_known_fields(mut self, known_fields: HashSet<String>) -> Self {
        self.known_fields = known_fields;
        self
    }

    /// Set the default fusion algorithm for hybrid queries.
    ///
    /// The fusion algorithm is only applied when the parsed query contains
    /// **both** lexical and vector clauses. For queries with only one type
    /// of clause, no fusion is performed.
    pub fn with_fusion(mut self, fusion: FusionAlgorithm) -> Self {
        self.default_fusion = fusion;
        self
    }

    /// Parse a unified query string into a [`SearchRequest`].
    ///
    /// The query string may contain both lexical and vector clauses:
    /// - Vector clauses: `field:"text"`, `field:text`, `field:"text"^0.8`
    ///   (where `field` is a vector field)
    /// - Lexical clauses: everything else (`title:hello`, `AND`, `"phrase"`, etc.)
    ///
    /// Vector text is embedded into vectors at parse time via the
    /// `VectorQueryParser`'s embedder, so this method is `async`.
    ///
    /// When the parsed query contains both lexical and vector clauses, the
    /// returned `SearchRequest` will have its `fusion_algorithm` set to the
    /// parser's default (configurable via [`with_fusion`](Self::with_fusion)).
    ///
    /// # Parameters
    ///
    /// - `query_str` - The unified query DSL string to parse.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::Other`] (invalid argument) if the query string
    /// is empty or consists only of whitespace.
    ///
    /// Returns [`LaurusError::Other`] (invalid argument) if, after splitting,
    /// no valid lexical or vector clause could be parsed from the input.
    pub async fn parse(&self, query_str: &str) -> Result<SearchRequest> {
        let query_str = query_str.trim();
        if query_str.is_empty() {
            return Err(LaurusError::invalid_argument(
                "Query string must not be empty",
            ));
        }

        let (lexical_str, vector_str, hybrid_mode) = self.split_query(query_str)?;

        let lexical = if let Some(ref s) = lexical_str {
            Some(self.lexical_parser.parse(s)?)
        } else {
            None
        };

        // Walk the parsed lexical tree and verify every field reference is
        // declared in the schema. This runs after the (cheap) lexical parse
        // but before the (expensive, embedding-driven) vector parse, so a
        // typo'd query is rejected before any embedder cost is incurred.
        //
        // Vector clauses do not need a separate walk: `split_query` only
        // routes a clause to the vector parser when its field is in the
        // known vector-field set, so any unknown field name would have
        // remained in the lexical portion and been caught here.
        if let Some(ref lex) = lexical {
            self.validate_field_refs(lex.as_ref())?;
        }

        let vector = if let Some(ref s) = vector_str {
            Some(self.vector_parser.parse(s).await?)
        } else {
            None
        };

        if lexical.is_none() && vector.is_none() {
            return Err(LaurusError::invalid_argument(
                "Query must contain at least one lexical or vector clause",
            ));
        }

        let fusion = if lexical.is_some() && vector.is_some() {
            Some(self.default_fusion)
        } else {
            None
        };

        let query = match (lexical, vector) {
            (Some(lex_query), Some(vec_req)) => SearchQuery::Hybrid {
                lexical: LexicalSearchQuery::Obj(lex_query),
                vector: vec_req.query,
                mode: hybrid_mode,
            },
            (Some(lex_query), None) => SearchQuery::Lexical(LexicalSearchQuery::Obj(lex_query)),
            (None, Some(vec_req)) => SearchQuery::Vector(vec_req.query),
            (None, None) => unreachable!(),
        };

        Ok(SearchRequest {
            query,
            fusion_algorithm: fusion,
            ..Default::default()
        })
    }

    /// Split a query string into lexical and vector portions.
    ///
    /// Uses the set of known vector field names to identify vector clauses.
    /// A clause is considered a vector clause if it starts with a known
    /// vector field name followed by `:`, e.g. `embedding:"text"` or
    /// `embedding:python^0.8`.
    ///
    /// A `+` prefix before a vector field clause (e.g. `+embedding:"text"`)
    /// triggers [`HybridMode::Intersection`], requiring documents to appear
    /// in both lexical and vector results.
    ///
    /// # Errors
    ///
    /// Returns an error if a vector field clause uses lexical-only syntax
    /// such as proximity/fuzzy modifiers (`~`) or range queries (`[`/`{`).
    ///
    /// # Returns
    ///
    /// `(lexical, vector, mode)` where either string may be `None`.
    fn split_query(&self, input: &str) -> Result<(Option<String>, Option<String>, HybridMode)> {
        if self.vector_fields.is_empty() {
            // No vector fields → everything is lexical
            return Ok((Some(input.to_string()), None, HybridMode::Union));
        }

        let fields_pattern: String = self
            .vector_fields
            .iter()
            .map(|f| regex::escape(f))
            .collect::<Vec<_>>()
            .join("|");

        // Detect lexical-only syntax on vector fields and reject with
        // clear error messages.
        self.check_unsupported_vector_syntax(input, &fields_pattern)?;

        // Match vector field clauses with an optional leading `+` prefix.
        // Group 1: optional `+` prefix
        // Group 2: the vector clause itself (field:value[^boost])
        let clause_pattern = format!(
            r#"(\+)?({fields})(?::(?:"[^"]*"|[^\s"^~\[\{{]+)(?:\^[\d]+(?:\.[\d]+)?)?)"#,
            fields = fields_pattern,
        );
        let vector_re = Regex::new(&clause_pattern).unwrap();

        let mut vector_clauses: Vec<String> = Vec::new();
        let mut has_required = false;

        for caps in vector_re.captures_iter(input) {
            if caps.get(1).is_some() {
                has_required = true;
            }
            // Group 0 minus the `+` prefix = the actual vector clause
            let full_match = caps.get(0).unwrap().as_str();
            let clause = full_match.strip_prefix('+').unwrap_or(full_match);
            vector_clauses.push(clause.to_string());
        }

        // Remove the full matches (including `+`) from input to get lexical part
        let lexical_raw = vector_re.replace_all(input, " ");
        let lexical_cleaned = clean_lexical_string(&lexical_raw);

        let lexical = if lexical_cleaned.is_empty() {
            None
        } else {
            Some(lexical_cleaned)
        };

        let vector = if vector_clauses.is_empty() {
            None
        } else {
            Some(vector_clauses.join(" "))
        };

        let mode = if has_required {
            HybridMode::Intersection
        } else {
            HybridMode::Union
        };

        Ok((lexical, vector, mode))
    }

    /// Reject field references that name a field outside the schema.
    ///
    /// Walks the parsed lexical query tree using
    /// [`Query::collect_field_refs`](crate::lexical::query::Query::collect_field_refs),
    /// gathers every field name reachable from the root (including those
    /// nested inside boolean / span / advanced composite queries), and
    /// verifies each against [`known_fields`](Self::known_fields). When
    /// `known_fields` is empty (the default, used by unit tests) no
    /// validation is performed, so callers that did not supply a field set
    /// are unaffected.
    ///
    /// Unlike the previous regex-based heuristic, this approach:
    ///
    /// - Sees only resolved field references — no false positives from
    ///   `identifier:` tokens that appear inside quoted phrase literals or
    ///   range expressions like `[A TO Z]`.
    /// - Detects every leaf reference inside arbitrarily-nested boolean and
    ///   span constructs.
    /// - Allows users to declare fields named `AND`, `OR`, `NOT`, `TO` (or
    ///   any other former reserved keyword) without the validator silently
    ///   skipping them.
    ///
    /// # Arguments
    ///
    /// * `query` - The parsed lexical query whose field references should
    ///   be validated.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] listing the unknown field
    /// names found, in deterministic (sorted) order.
    fn validate_field_refs(&self, query: &dyn crate::lexical::query::Query) -> Result<()> {
        if self.known_fields.is_empty() {
            return Ok(());
        }

        let mut refs: HashSet<String> = HashSet::new();
        query.collect_field_refs(&mut refs);

        let mut unknown: Vec<String> = refs
            .into_iter()
            .filter(|f| !self.known_fields.contains(f))
            .collect();

        if unknown.is_empty() {
            Ok(())
        } else {
            unknown.sort();
            Err(LaurusError::invalid_argument(format!(
                "query references unknown field(s) {unknown:?}; \
                 declare them in the schema or check for typos"
            )))
        }
    }

    /// Check for lexical-only syntax used on vector fields and return a
    /// descriptive error if found.
    ///
    /// Detects:
    /// - Proximity/fuzzy modifiers: `content:"text"~2`, `content:word~`
    /// - Range queries: `content:[A TO Z]`, `content:{100 TO 500}`
    fn check_unsupported_vector_syntax(&self, input: &str, fields_pattern: &str) -> Result<()> {
        // Proximity/fuzzy: vector_field:value~[digits]
        let tilde_pattern = format!(
            r#"((?:{fields})(?::(?:"[^"]*"|[^\s"^~]+)(?:\^[\d]+(?:\.[\d]+)?)?))(~[\d]*)"#,
            fields = fields_pattern,
        );
        let tilde_re = Regex::new(&tilde_pattern).unwrap();
        if let Some(caps) = tilde_re.captures(input) {
            let clause = caps.get(1).unwrap().as_str();
            let modifier = caps.get(2).unwrap().as_str();
            return Err(LaurusError::invalid_argument(format!(
                "Proximity/fuzzy modifier '{modifier}' is not supported on vector field \
                 clause '{clause}'. The '~' modifier is only valid for lexical queries \
                 (e.g. \"term~2\" for fuzzy, '\"phrase\"~10' for proximity)."
            )));
        }

        // Range queries: vector_field:[...] or vector_field:{...}
        let range_pattern = format!(r#"({fields}):(\[|\{{)"#, fields = fields_pattern,);
        let range_re = Regex::new(&range_pattern).unwrap();
        if let Some(caps) = range_re.captures(input) {
            let field = caps.get(1).unwrap().as_str();
            let bracket = caps.get(2).unwrap().as_str();
            let kind = if bracket == "[" {
                "inclusive range ([...TO...])"
            } else {
                "exclusive range ({...TO...})"
            };
            return Err(LaurusError::invalid_argument(format!(
                "Range query on vector field '{field}' is not supported. \
                 The {kind} syntax is only valid for lexical fields."
            )));
        }

        Ok(())
    }
}

/// Clean up a lexical query string after vector clause removal.
///
/// Handles:
/// 1. Collapse multiple whitespace into single space
/// 2. Remove empty groups (`( )`, optionally `+`/`-` prefixed) left behind
///    when a vector clause is extracted from inside a boolean group, e.g.
///    `+(embedding:"...")` → `+( )` (GitHub issue #983)
/// 3. Remove leading/trailing boolean operators (AND, OR)
/// 4. Collapse consecutive boolean operators (AND AND → AND)
fn clean_lexical_string(s: &str) -> String {
    static WHITESPACE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
    static EMPTY_GROUP_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[+-]?\(\s*\)").unwrap());

    // Collapse multiple whitespace
    let mut s = WHITESPACE_RE.replace_all(s, " ").into_owned();

    // Strip empty groups, looping so nested leftovers like `+( ( ) )`
    // collapse completely. Non-empty parentheses (boolean groups,
    // `geo_bbox(...)` arguments) never match.
    loop {
        let stripped = EMPTY_GROUP_RE.replace_all(&s, " ");
        if stripped == s {
            break;
        }
        s = stripped.into_owned();
    }

    let s = s.trim();

    if s.is_empty() {
        return String::new();
    }

    // Split into tokens and filter out dangling boolean operators
    let tokens: Vec<&str> = s.split_whitespace().collect();
    let mut result: Vec<&str> = Vec::new();

    for token in &tokens {
        if token.eq_ignore_ascii_case("AND") || token.eq_ignore_ascii_case("OR") {
            // Only add boolean operator if there's a preceding non-boolean token
            // and the previous token is not already a boolean operator.
            if !result.is_empty() {
                let last = result.last().unwrap();
                if !last.eq_ignore_ascii_case("AND") && !last.eq_ignore_ascii_case("OR") {
                    result.push(token);
                }
                // else: skip consecutive boolean operator
            }
            // else: skip leading boolean operator
        } else {
            result.push(token);
        }
    }

    // Remove trailing boolean operator
    if let Some(last) = result.last()
        && (last.eq_ignore_ascii_case("AND") || last.eq_ignore_ascii_case("OR"))
    {
        result.pop();
    }

    result.join(" ")
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::*;
    use crate::analysis::analyzer::standard::StandardAnalyzer;
    use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
    use crate::engine::search::VectorSearchQuery;
    use crate::error::Result as IrisResult;
    use crate::vector::core::vector::Vector;
    use crate::vector::store::request::QueryVector;

    /// Mock embedder that returns a zero vector of dimension 4.
    #[derive(Debug)]
    struct MockEmbedder;

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, _input: &EmbedInput<'_>) -> IrisResult<Vector> {
            Ok(Vector::new(vec![0.0; 4]))
        }
        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }
        fn name(&self) -> &str {
            "mock"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Assert that the request contains a lexical-only query.
    fn assert_lexical_only(request: &SearchRequest) {
        assert!(matches!(request.query, SearchQuery::Lexical(_)));
    }

    /// Assert that the request contains a vector-only query and return a
    /// reference to the vector query.
    fn assert_vector_only(request: &SearchRequest) -> &VectorSearchQuery {
        match &request.query {
            SearchQuery::Vector(v) => v,
            _ => panic!("Expected SearchQuery::Vector"),
        }
    }

    /// Assert that the request contains a hybrid query and return references
    /// to the lexical and vector components along with the hybrid mode.
    fn assert_hybrid(
        request: &SearchRequest,
    ) -> (&LexicalSearchQuery, &VectorSearchQuery, &HybridMode) {
        match &request.query {
            SearchQuery::Hybrid {
                lexical,
                vector,
                mode,
            } => (lexical, vector, mode),
            _ => panic!("Expected SearchQuery::Hybrid"),
        }
    }

    /// Extract the query vectors from a vector search query.
    fn get_vectors(vq: &VectorSearchQuery) -> &Vec<QueryVector> {
        match vq {
            VectorSearchQuery::Vectors(v) => v,
            _ => panic!("Expected VectorSearchQuery::Vectors"),
        }
    }

    fn make_parser() -> UnifiedQueryParser {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let lexical = LexicalQueryParser::new(analyzer).with_default_field("title");
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let vector = VectorQueryParser::new(embedder).with_default_field("content");
        let vector_fields: HashSet<String> = ["content".to_string()].into_iter().collect();
        UnifiedQueryParser::new(lexical, vector, vector_fields)
    }

    #[tokio::test]
    async fn test_lexical_only() {
        let parser = make_parser();
        let request = parser.parse("title:hello").await.unwrap();

        assert_lexical_only(&request);
        assert!(request.fusion_algorithm.is_none());
    }

    /// A vector clause extracted from inside a boolean group must not
    /// leave an unparsable empty group (`+( )`) behind in the lexical
    /// portion (GitHub issue #983).
    #[tokio::test]
    async fn test_vector_clause_inside_group_leaves_valid_lexical() {
        let parser = make_parser();
        let request = parser
            .parse(r#"+(content:"cats") +title:hello"#)
            .await
            .unwrap();

        assert!(
            matches!(request.query, SearchQuery::Hybrid { .. }),
            "group-wrapped vector clause plus lexical clause must parse as hybrid"
        );
    }

    /// A query that is nothing but a group-wrapped vector clause reduces
    /// to a pure vector query once the empty group is stripped.
    #[tokio::test]
    async fn test_vector_clause_alone_inside_group_is_pure_vector() {
        let parser = make_parser();
        let request = parser.parse(r#"+(content:"cats")"#).await.unwrap();

        assert_vector_only(&request);
    }

    #[tokio::test]
    async fn test_vector_only_quoted() {
        let parser = make_parser();
        let request = parser.parse(r#"content:"cats""#).await.unwrap();

        let vq = assert_vector_only(&request);
        assert!(request.fusion_algorithm.is_none());

        let vecs = get_vectors(vq);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "content");
    }

    #[tokio::test]
    async fn test_vector_only_unquoted() {
        let parser = make_parser();
        let request = parser.parse("content:cats").await.unwrap();

        let vq = assert_vector_only(&request);
        let vecs = get_vectors(vq);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "content");
    }

    #[tokio::test]
    async fn test_hybrid() {
        let parser = make_parser();
        let request = parser.parse(r#"title:hello content:"cats""#).await.unwrap();

        assert_hybrid(&request);
        assert!(request.fusion_algorithm.is_some());

        // Fusion defaults to RRF
        if let Some(FusionAlgorithm::RRF { k }) = request.fusion_algorithm {
            assert!((k - 60.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected RRF fusion");
        }
    }

    #[tokio::test]
    async fn test_hybrid_unquoted_vector() {
        let parser = make_parser();
        let request = parser.parse("title:hello content:cats").await.unwrap();

        assert_hybrid(&request);
    }

    #[tokio::test]
    async fn test_vector_with_boost() {
        let parser = make_parser();
        let request = parser.parse(r#"content:"text"^0.8"#).await.unwrap();

        let vq = assert_vector_only(&request);
        let vecs = get_vectors(vq);
        assert!((vecs[0].weight - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_unquoted_vector_with_boost() {
        let parser = make_parser();
        let request = parser.parse("content:python^0.8").await.unwrap();

        let vq = assert_vector_only(&request);
        let vecs = get_vectors(vq);
        assert!((vecs[0].weight - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_multiple_vector_clauses() {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let lexical = LexicalQueryParser::new(analyzer).with_default_field("title");
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let vector = VectorQueryParser::new(embedder);
        let vector_fields: HashSet<String> =
            ["a".to_string(), "b".to_string()].into_iter().collect();
        let parser = UnifiedQueryParser::new(lexical, vector, vector_fields);

        let request = parser.parse(r#"a:"x" b:"y"^0.5"#).await.unwrap();

        let vq = assert_vector_only(&request);
        let vecs = get_vectors(vq);
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "a");
        assert_eq!(vecs[1].fields.as_ref().unwrap()[0], "b");
        assert!((vecs[1].weight - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_lexical_and_with_vector() {
        let parser = make_parser();
        let request = parser
            .parse(r#"title:hello AND title:world content:"cats""#)
            .await
            .unwrap();

        assert_hybrid(&request);
        assert!(request.fusion_algorithm.is_some());
    }

    #[tokio::test]
    async fn test_vector_between_and() {
        let parser = make_parser();
        // After removing content:"cats", we get "title:hello AND AND title:world"
        // which should be cleaned to "title:hello AND title:world"
        let request = parser
            .parse(r#"title:hello AND content:"cats" AND title:world"#)
            .await
            .unwrap();

        assert_hybrid(&request);
    }

    #[tokio::test]
    async fn test_empty_query_error() {
        let parser = make_parser();
        assert!(parser.parse("").await.is_err());
        assert!(parser.parse("   ").await.is_err());
    }

    #[tokio::test]
    async fn test_custom_fusion() {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let lexical = LexicalQueryParser::new(analyzer).with_default_field("title");
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let vector = VectorQueryParser::new(embedder).with_default_field("content");
        let vector_fields: HashSet<String> = ["content".to_string()].into_iter().collect();
        let parser = UnifiedQueryParser::new(lexical, vector, vector_fields).with_fusion(
            FusionAlgorithm::WeightedSum {
                lexical_weight: 0.7,
                vector_weight: 0.3,
            },
        );

        let request = parser.parse(r#"title:hello content:"cats""#).await.unwrap();

        if let Some(FusionAlgorithm::WeightedSum {
            lexical_weight,
            vector_weight,
        }) = request.fusion_algorithm
        {
            assert!((lexical_weight - 0.7).abs() < f32::EPSILON);
            assert!((vector_weight - 0.3).abs() < f32::EPSILON);
        } else {
            panic!("Expected WeightedSum fusion");
        }
    }

    #[tokio::test]
    async fn test_unicode_vector_text() {
        let parser = make_parser();
        let request = parser.parse(r#"content:"日本語テスト""#).await.unwrap();

        let vq = assert_vector_only(&request);
        let vecs = get_vectors(vq);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "content");
        assert_eq!(vecs[0].vector.dimension(), 4);
    }

    #[tokio::test]
    async fn test_no_vector_fields_all_lexical() {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let lexical = LexicalQueryParser::new(analyzer).with_default_field("title");
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let vector = VectorQueryParser::new(embedder);
        let parser = UnifiedQueryParser::new(lexical, vector, HashSet::new());

        let request = parser.parse("title:hello").await.unwrap();
        assert_lexical_only(&request);
    }

    // -- Tests for proximity/fuzzy modifier rejection on vector fields --

    #[tokio::test]
    async fn test_vector_field_with_proximity_error() {
        let parser = make_parser();
        let result = parser.parse(r#"content:"hello world"~2"#).await;
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("Proximity/fuzzy modifier"), "got: {msg}");
        assert!(msg.contains("content:"), "got: {msg}");
    }

    #[tokio::test]
    async fn test_vector_field_with_fuzzy_error() {
        let parser = make_parser();
        let result = parser.parse("content:python~2").await;
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("Proximity/fuzzy modifier"), "got: {msg}");
    }

    #[tokio::test]
    async fn test_vector_field_with_tilde_only_error() {
        let parser = make_parser();
        let result = parser.parse("content:python~").await;
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("Proximity/fuzzy modifier"), "got: {msg}");
    }

    #[tokio::test]
    async fn test_vector_field_with_inclusive_range_error() {
        let parser = make_parser();
        let result = parser.parse("content:[A TO Z]").await;
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("Range query"), "got: {msg}");
        assert!(msg.contains("content"), "got: {msg}");
        assert!(msg.contains("inclusive"), "got: {msg}");
    }

    #[tokio::test]
    async fn test_vector_field_with_exclusive_range_error() {
        let parser = make_parser();
        let result = parser.parse("content:{100 TO 500}").await;
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("Range query"), "got: {msg}");
        assert!(msg.contains("exclusive"), "got: {msg}");
    }

    #[tokio::test]
    async fn test_lexical_field_range_still_works() {
        // title is a lexical field — range should work fine
        let parser = make_parser();
        let request = parser.parse("title:[A TO Z]").await.unwrap();
        assert_lexical_only(&request);
    }

    #[tokio::test]
    async fn test_lexical_field_fuzzy_still_works() {
        // title is a lexical field — fuzzy should work fine
        let parser = make_parser();
        let request = parser.parse("title:hello~2").await.unwrap();
        assert_lexical_only(&request);
    }

    #[tokio::test]
    async fn test_tilde_inside_quotes_is_valid_vector_text() {
        // ~2 is inside quotes — treated as literal text for embedding
        let parser = make_parser();
        let request = parser.parse(r#"content:"python~2""#).await.unwrap();
        let vq = assert_vector_only(&request);
        let vecs = get_vectors(vq);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "content");
    }

    // -- Tests for HybridMode (AND/OR semantics) --

    #[tokio::test]
    async fn test_hybrid_union_by_default() {
        let parser = make_parser();
        let request = parser.parse(r#"title:hello content:"cats""#).await.unwrap();
        let (_, _, mode) = assert_hybrid(&request);
        assert_eq!(*mode, HybridMode::Union);
    }

    #[tokio::test]
    async fn test_hybrid_intersection_with_plus_vector() {
        let parser = make_parser();
        let request = parser
            .parse(r#"title:hello +content:"cats""#)
            .await
            .unwrap();
        let (_, _, mode) = assert_hybrid(&request);
        assert_eq!(*mode, HybridMode::Intersection);
    }

    #[tokio::test]
    async fn test_hybrid_intersection_plus_on_both() {
        // + on lexical field is preserved for the lexical parser;
        // + on vector field triggers Intersection mode.
        let parser = make_parser();
        let request = parser
            .parse(r#"+title:hello +content:"cats""#)
            .await
            .unwrap();
        let (_, _, mode) = assert_hybrid(&request);
        assert_eq!(*mode, HybridMode::Intersection);
    }

    #[tokio::test]
    async fn test_hybrid_intersection_unquoted_vector() {
        let parser = make_parser();
        let request = parser.parse("title:hello +content:cats").await.unwrap();
        let (_, _, mode) = assert_hybrid(&request);
        assert_eq!(*mode, HybridMode::Intersection);
    }

    #[tokio::test]
    async fn test_vector_only_with_plus_stays_vector() {
        // + on a vector-only query (no lexical part) → still vector-only, not hybrid
        let parser = make_parser();
        let request = parser.parse(r#"+content:"cats""#).await.unwrap();
        // No lexical component, so it's vector-only (mode is irrelevant)
        assert_vector_only(&request);
    }

    // -- Tests for clean_lexical_string helper --

    #[test]
    fn test_clean_leading_boolean() {
        assert_eq!(clean_lexical_string("AND hello"), "hello");
        assert_eq!(clean_lexical_string("OR hello"), "hello");
    }

    #[test]
    fn test_clean_trailing_boolean() {
        assert_eq!(clean_lexical_string("hello AND"), "hello");
        assert_eq!(clean_lexical_string("hello OR"), "hello");
    }

    #[test]
    fn test_clean_consecutive_boolean() {
        assert_eq!(
            clean_lexical_string("hello AND AND world"),
            "hello AND world"
        );
        assert_eq!(clean_lexical_string("hello OR OR world"), "hello OR world");
        assert_eq!(
            clean_lexical_string("hello AND OR world"),
            "hello AND world"
        );
    }

    #[test]
    fn test_clean_multiple_spaces() {
        assert_eq!(clean_lexical_string("hello   world"), "hello world");
    }

    #[test]
    fn test_clean_empty() {
        assert_eq!(clean_lexical_string(""), "");
        assert_eq!(clean_lexical_string("   "), "");
        assert_eq!(clean_lexical_string("AND"), "");
        assert_eq!(clean_lexical_string("AND OR"), "");
    }

    /// Empty groups left behind by vector-clause extraction must be
    /// stripped so the remaining lexical string still parses
    /// (GitHub issue #983).
    #[test]
    fn test_clean_empty_group() {
        assert_eq!(
            clean_lexical_string("+( ) +location:geo_bbox(35.6, 139.5, 35.8, 140.0)"),
            "+location:geo_bbox(35.6, 139.5, 35.8, 140.0)"
        );
        assert_eq!(clean_lexical_string("( )"), "");
        assert_eq!(clean_lexical_string("-( ) foo"), "foo");
    }

    /// Nested leftovers like `+( ( ) )` must collapse completely.
    #[test]
    fn test_clean_nested_empty_group() {
        assert_eq!(clean_lexical_string("foo +( ( ) )"), "foo");
        assert_eq!(clean_lexical_string("+( ( ) )"), "");
    }

    /// Removing an empty group can expose a dangling trailing operator,
    /// which must also be dropped.
    #[test]
    fn test_clean_empty_group_exposes_trailing_operator() {
        assert_eq!(clean_lexical_string("foo AND ( )"), "foo");
    }

    // -- Field-reference validation (parse-tree walking) --

    /// Build a parser that knows the given lexical field set, so
    /// `validate_field_refs` is active.
    fn make_parser_with_known_fields(fields: &[&str]) -> UnifiedQueryParser {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let lexical = LexicalQueryParser::new(analyzer).with_default_field(fields[0]);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let vector = VectorQueryParser::new(embedder);
        let known_fields: HashSet<String> = fields.iter().map(|s| s.to_string()).collect();
        UnifiedQueryParser::new(lexical, vector, HashSet::new()).with_known_fields(known_fields)
    }

    #[tokio::test]
    async fn validate_accepts_declared_field() {
        let parser = make_parser_with_known_fields(&["title", "body"]);
        parser.parse("title:hello").await.unwrap();
        parser.parse("body:world").await.unwrap();
    }

    #[tokio::test]
    async fn validate_rejects_typo_with_sorted_message() {
        let parser = make_parser_with_known_fields(&["title", "body"]);
        let err = match parser.parse("titl:hello").await {
            Ok(_) => panic!("expected error for typo'd field"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("titl"), "{msg}");
        assert!(msg.contains("unknown field"), "{msg}");
    }

    #[tokio::test]
    async fn validate_collects_all_unknowns_sorted() {
        let parser = make_parser_with_known_fields(&["title"]);
        let err = match parser.parse("foo:hello AND bar:world").await {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        let msg = err.to_string();
        // Both unknown fields appear, in sorted order (`bar` before `foo`).
        let foo_pos = msg
            .find("foo")
            .unwrap_or_else(|| panic!("foo present: {msg}"));
        let bar_pos = msg
            .find("bar")
            .unwrap_or_else(|| panic!("bar present: {msg}"));
        assert!(bar_pos < foo_pos, "expected sorted order: {msg}");
    }

    #[tokio::test]
    async fn validate_recurses_into_boolean_clauses() {
        let parser = make_parser_with_known_fields(&["title", "body", "tag"]);
        // Nested boolean: should walk every leaf.
        parser
            .parse("(title:foo AND body:bar) OR tag:baz")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn validate_detects_unknown_inside_nested_boolean() {
        let parser = make_parser_with_known_fields(&["title", "body"]);
        let err = match parser.parse("(title:foo AND body:bar) OR tag:baz").await {
            Ok(_) => panic!("expected error for nested unknown"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("tag"), "{err}");
    }

    #[tokio::test]
    async fn validate_does_not_inspect_quoted_phrases() {
        // The phrase contains `bar:` literally, which the previous regex
        // pre-stripped. With AST walking it never reaches the field check
        // because the parsed query targets only the surrounding `title`.
        let parser = make_parser_with_known_fields(&["title"]);
        parser.parse(r#"title:"foo bar:baz qux""#).await.unwrap();
    }

    #[tokio::test]
    async fn validate_allows_field_named_and() {
        // A field literally named `AND` (or `OR` / `NOT` / `TO`) was
        // skipped by the old regex via its keyword allow-list, which made
        // typos in those names invisible. The AST walk treats them as
        // ordinary identifiers and validates them like any other field.
        let parser = make_parser_with_known_fields(&["AND", "title"]);
        parser.parse("AND:hello").await.unwrap();
    }

    #[tokio::test]
    async fn validate_allows_field_named_to() {
        let parser = make_parser_with_known_fields(&["TO", "title"]);
        parser.parse("TO:hello").await.unwrap();
    }

    #[tokio::test]
    async fn validate_does_not_misinterpret_range_to_keyword() {
        // `[A TO Z]` is a range literal; `TO` inside it is a separator,
        // not a field reference. The walker never sees `TO` as a field
        // name because it walks the parsed RangeQuery, not the source
        // string.
        let parser = make_parser_with_known_fields(&["title"]);
        parser.parse("title:[A TO Z]").await.unwrap();
    }

    /// Numeric ranges parse to `NumericRangeQuery` and `?`/`*` terms to
    /// `WildcardQuery`; both must report their field through
    /// `Query::field()` so a misspelled field is rejected exactly like it
    /// is for a plain term (#1131).
    #[tokio::test]
    async fn validate_rejects_unknown_field_in_numeric_range_and_wildcard_terms() {
        let parser = make_parser_with_known_fields(&["title", "price"]);
        for dsl in ["pric:[1 TO 10]", "titl:fo?"] {
            let err = match parser.parse(dsl).await {
                Ok(_) => panic!("expected an unknown-field error for {dsl}"),
                Err(e) => e,
            };
            assert!(err.to_string().contains("unknown field"), "{dsl}: {err}");
        }
        // The declared spellings still parse.
        parser.parse("price:[1 TO 10]").await.unwrap();
        parser.parse("title:fo?").await.unwrap();
    }

    #[tokio::test]
    async fn validate_skipped_when_known_fields_empty() {
        // The default parser has an empty `known_fields` set, so the
        // unknown field name `anything_goes` does not raise a validation
        // error. (Whether the underlying lexical parser succeeds is
        // independent — using a known-default field keeps the test
        // focused on the validation skip behaviour.)
        let parser = make_parser();
        // `make_parser` uses default field "title", so this query parses
        // successfully and the lexical AST targets only `title`.
        parser.parse("title:anything_goes").await.unwrap();
    }
}
