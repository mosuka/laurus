//! Text highlighting functionality for search results.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::token::{Token, flatten_token_graph, token_positions};
use crate::error::Result;
use crate::lexical::index::inverted::core::automaton::{Automaton, LevenshteinAutomaton};
use crate::lexical::query::{HighlightTerm, PhraseArc, Query};

/// Configuration for text highlighting.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// HTML tag to wrap highlighted terms (e.g., "mark", "em", "strong").
    pub tag: String,
    /// CSS class to add to highlight tags.
    pub css_class: Option<String>,
    /// Maximum number of fragments to return.
    pub max_fragments: usize,
    /// Length of each fragment in characters.
    pub fragment_size: usize,
    /// Number of characters to overlap between fragments.
    pub fragment_overlap: usize,
    /// Separator between fragments.
    pub fragment_separator: String,
    /// Whether to return the entire field if no highlights are found.
    pub return_entire_field_if_no_highlight: bool,
    /// Maximum length of returned text.
    pub max_analyzed_chars: usize,
    /// Whether only query terms targeting the highlighted field are used
    /// (`true`, the default, as in Elasticsearch's `require_field_match`).
    /// With `false`, terms from every field in the query highlight.
    pub require_field_match: bool,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        HighlightConfig {
            tag: "mark".to_string(),
            css_class: None,
            max_fragments: 5,
            fragment_size: 150,
            fragment_overlap: 20,
            fragment_separator: " ... ".to_string(),
            return_entire_field_if_no_highlight: false,
            max_analyzed_chars: 1_000_000,
            require_field_match: true,
        }
    }
}

impl HighlightConfig {
    /// Create a new highlight configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTML tag for highlighting.
    pub fn tag(mut self, tag: String) -> Self {
        self.tag = tag;
        self
    }

    /// Set the CSS class for highlight tags.
    pub fn css_class(mut self, css_class: String) -> Self {
        self.css_class = Some(css_class);
        self
    }

    /// Set the maximum number of fragments.
    pub fn max_fragments(mut self, max_fragments: usize) -> Self {
        self.max_fragments = max_fragments;
        self
    }

    /// Set the fragment size.
    pub fn fragment_size(mut self, fragment_size: usize) -> Self {
        self.fragment_size = fragment_size;
        self
    }

    /// Set whether only query terms targeting the highlighted field are used.
    pub fn require_field_match(mut self, require_field_match: bool) -> Self {
        self.require_field_match = require_field_match;
        self
    }

    /// Build the opening HTML tag.
    pub fn opening_tag(&self) -> String {
        if let Some(ref css_class) = self.css_class {
            format!("<{} class=\"{}\">", self.tag, css_class)
        } else {
            format!("<{}>", self.tag)
        }
    }

    /// Build the closing HTML tag.
    pub fn closing_tag(&self) -> String {
        format!("</{}>", self.tag)
    }
}

/// Represents a highlighted fragment of text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightFragment {
    /// The highlighted text fragment.
    pub text: String,
    /// Starting position in the original text.
    pub start_offset: usize,
    /// Ending position in the original text.
    pub end_offset: usize,
    /// Score indicating relevance of this fragment.
    pub score: f32,
}

impl HighlightFragment {
    /// Create a new highlight fragment.
    pub fn new(text: String, start_offset: usize, end_offset: usize, score: f32) -> Self {
        HighlightFragment {
            text,
            start_offset,
            end_offset,
            score,
        }
    }
}

/// Represents highlight information for a field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHighlight {
    /// Field name.
    pub field_name: String,
    /// Highlighted fragments.
    pub fragments: Vec<HighlightFragment>,
    /// Whether the entire field content was returned.
    pub is_entire_field: bool,
}

impl FieldHighlight {
    /// Create a new field highlight.
    pub fn new(field_name: String) -> Self {
        FieldHighlight {
            field_name,
            fragments: Vec::new(),
            is_entire_field: false,
        }
    }

    /// Add a fragment to this field highlight.
    pub fn add_fragment(&mut self, fragment: HighlightFragment) {
        self.fragments.push(fragment);
    }

    /// Get the best fragment (highest score).
    pub fn best_fragment(&self) -> Option<&HighlightFragment> {
        self.fragments
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
    }

    /// Combine all fragments into a single string.
    pub fn combined_text(&self, separator: &str) -> String {
        self.fragments
            .iter()
            .map(|f| &f.text)
            .cloned()
            .collect::<Vec<_>>()
            .join(separator)
    }
}

/// Text range with highlighting information.
#[derive(Debug, Clone)]
struct HighlightSpan {
    /// Range in the original text.
    range: Range<usize>,
    /// Whether this span should be highlighted.
    highlight: bool,
    /// Score for this span (higher = more important).
    score: f32,
}

impl HighlightSpan {
    fn new(range: Range<usize>, highlight: bool, score: f32) -> Self {
        HighlightSpan {
            range,
            highlight,
            score,
        }
    }
}

/// One fragment candidate: the run of spans that overlaps `window`.
#[derive(Debug)]
struct FragmentCandidate {
    /// Index range into the span slice the candidate was built from.
    spans: Range<usize>,
    /// Byte window in the original text.
    window: Range<usize>,
}

/// Main highlighter that can highlight text based on search queries.
pub struct Highlighter {
    /// Configuration for highlighting.
    config: HighlightConfig,
    /// Text analyzer for tokenization. Shared, so the engine can hand the
    /// index's per-field analyzer straight in (#1134).
    analyzer: Arc<dyn Analyzer>,
}

impl std::fmt::Debug for Highlighter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Highlighter")
            .field("config", &self.config)
            .field("analyzer", &"<dyn Analyzer>")
            .finish()
    }
}

impl Highlighter {
    /// Create a new highlighter.
    pub fn new(config: HighlightConfig) -> Self {
        Highlighter {
            config,
            analyzer: Arc::new(StandardAnalyzer::new().unwrap()),
        }
    }

    /// Create a highlighter with a custom analyzer.
    pub fn with_analyzer(config: HighlightConfig, analyzer: Box<dyn Analyzer>) -> Self {
        Self::with_shared_analyzer(config, Arc::from(analyzer))
    }

    /// Create a highlighter over an analyzer shared with the index — the
    /// way the engine highlights each field with its own analyzer (#1134).
    pub fn with_shared_analyzer(config: HighlightConfig, analyzer: Arc<dyn Analyzer>) -> Self {
        Highlighter { config, analyzer }
    }

    /// Highlight text based on a query.
    ///
    /// `Q: ?Sized`, so a `&dyn Query` (the engine's resolved query) can be
    /// passed directly.
    pub fn highlight<Q: Query + ?Sized>(
        &self,
        query: &Q,
        field_name: &str,
        text: &str,
    ) -> Result<FieldHighlight> {
        // Limit text length.
        //
        // The config is named `max_analyzed_chars` but the previous code
        // compared and sliced `text.len()` — bytes — so a Japanese field
        // was cut ~3x earlier than configured and, worse, `&text[..n]`
        // panicked whenever `n` fell inside a character.
        //
        // A byte length never undershoots the character count, so
        // `text.len() <= max_analyzed_chars` already proves no truncation
        // is needed. That keeps the common path O(1).
        let text: std::borrow::Cow<'_, str> = if text.len() > self.config.max_analyzed_chars {
            std::borrow::Cow::Owned(text.chars().take(self.config.max_analyzed_chars).collect())
        } else {
            std::borrow::Cow::Borrowed(text)
        };
        let text = text.as_ref();

        // Extract terms from query
        let highlight_terms = self.extract_query_terms(query, field_name);

        if highlight_terms.is_empty() {
            return self.create_no_highlight_result(field_name, text);
        }

        // Find highlight spans
        let highlight_spans = self.find_highlight_spans(text, &highlight_terms)?;

        if highlight_spans.is_empty() {
            return self.create_no_highlight_result(field_name, text);
        }

        // Create fragments
        let fragments = self.create_fragments(text, &highlight_spans)?;

        let mut field_highlight = FieldHighlight::new(field_name.to_string());
        for fragment in fragments {
            field_highlight.add_fragment(fragment);
        }

        Ok(field_highlight)
    }

    /// Collect what `query` would highlight in `field_name` by walking the
    /// query tree (#594). With `require_field_match` on, leaves that target
    /// another field are skipped.
    fn extract_query_terms<Q: Query + ?Sized>(
        &self,
        query: &Q,
        field_name: &str,
    ) -> Vec<HighlightTerm> {
        let field = self.config.require_field_match.then_some(field_name);
        let mut terms = Vec::new();
        query.collect_highlight_terms(field, &mut terms);
        terms
    }

    /// Find highlight spans in text.
    ///
    /// One analyzer pass over `text`. Every token is probed against the
    /// exact terms (a byte-length prefilter, then a hash lookup), the
    /// prefixes, the compiled regexes and the fuzzy automata. Tokens are
    /// buffered only when a phrase is present; phrases are then matched on
    /// token positions with the same in-order, per-gap slop rule as the
    /// index-side phrase matcher, so what highlights is what searches.
    fn find_highlight_spans(
        &self,
        text: &str,
        terms: &[HighlightTerm],
    ) -> Result<Vec<HighlightSpan>> {
        let mut exact: HashSet<String> = HashSet::new();
        let mut min_term_len = usize::MAX;
        let mut max_term_len = 0usize;
        let mut phrases: Vec<(Vec<Vec<String>>, u32)> = Vec::new();
        let mut graph_phrases: Vec<(Vec<PhraseArc>, u32)> = Vec::new();
        let mut prefixes: Vec<&str> = Vec::new();
        let mut regexes: Vec<&Regex> = Vec::new();
        let mut fuzzy: Vec<&LevenshteinAutomaton> = Vec::new();
        for term in terms {
            match term {
                HighlightTerm::Exact(term) => {
                    let term = term.to_lowercase();
                    min_term_len = min_term_len.min(term.len());
                    max_term_len = max_term_len.max(term.len());
                    exact.insert(term);
                }
                HighlightTerm::Phrase { positions, slop } => {
                    let positions = positions
                        .iter()
                        .map(|alternatives| alternatives.iter().map(|t| t.to_lowercase()).collect())
                        .collect();
                    phrases.push((positions, *slop));
                }
                HighlightTerm::GraphPhrase { arcs, slop } => {
                    let arcs = arcs
                        .iter()
                        .map(|arc| {
                            let terms = arc.terms.iter().map(|t| t.to_lowercase()).collect();
                            PhraseArc::new(arc.from, arc.to, terms)
                        })
                        .collect();
                    graph_phrases.push((arcs, *slop));
                }
                HighlightTerm::Prefix(prefix) => prefixes.push(prefix),
                HighlightTerm::Regex(regex) => regexes.push(regex),
                HighlightTerm::Fuzzy(automaton) => fuzzy.push(automaton),
            }
        }

        // Exact terms are compared lowercased; patterns are tested as the
        // query wrote them, exactly as the term-dictionary enumeration does.
        // The length prefilter only gates the hash probe (#408).
        let matches_token = |candidate: &str| -> bool {
            let len = candidate.len();
            (len >= min_term_len && len <= max_term_len && exact.contains(candidate))
                || prefixes.iter().any(|prefix| candidate.starts_with(prefix))
                || regexes.iter().any(|regex| regex.is_match(candidate))
                || fuzzy.iter().any(|automaton| automaton.matches(candidate))
        };

        // Most analyzers already lowercase, so the `to_lowercase()`
        // fallback only runs for tokens that still carry uppercase (#408).
        let mut spans = Vec::new();
        let mut buffered: Vec<Token> = Vec::new();
        for token in self.analyzer.analyze(text)? {
            let matched = matches_token(&token.text)
                || (has_uppercase(&token.text) && matches_token(&token.text.to_lowercase()));
            if matched {
                let score = self.calculate_term_score(&token.text, terms.len());
                spans.push(HighlightSpan::new(
                    token.start_offset..token.end_offset,
                    true,
                    score,
                ));
            }
            if !phrases.is_empty() || !graph_phrases.is_empty() {
                buffered.push(token);
            }
        }

        // Phrases compare positions as the index stores them.
        let buffered = flatten_token_graph(buffered);
        for (phrase, slop) in &phrases {
            spans.extend(phrase_spans(&buffered, phrase, *slop));
        }
        for (arcs, slop) in &graph_phrases {
            spans.extend(graph_phrase_spans(&buffered, arcs, *slop));
        }

        // Sort spans by position
        spans.sort_by_key(|span| span.range.start);

        // Merge overlapping spans
        let merged_spans = self.merge_overlapping_spans(spans);

        Ok(merged_spans)
    }

    /// Calculate score for a term match.
    fn calculate_term_score(&self, term: &str, term_count: usize) -> f32 {
        // Simple scoring based on term length and rarity
        let base_score = 1.0;
        let length_bonus = (term.len() as f32).log2() * 0.1;
        let rarity_bonus = 1.0 / (term_count as f32).sqrt();

        base_score + length_bonus + rarity_bonus
    }

    /// Merge overlapping highlight spans.
    fn merge_overlapping_spans(&self, spans: Vec<HighlightSpan>) -> Vec<HighlightSpan> {
        let mut iter = spans.into_iter();
        let Some(mut current) = iter.next() else {
            return Vec::new();
        };

        let mut merged = Vec::new();
        for span in iter {
            if span.range.start <= current.range.end {
                // Overlapping spans - merge them
                current.range.end = current.range.end.max(span.range.end);
                current.score = current.score.max(span.score);
            } else {
                // Non-overlapping - push current and start new one
                merged.push(current);
                current = span;
            }
        }

        merged.push(current);
        merged
    }

    /// Create text fragments with highlighting.
    ///
    /// Candidates are scored and cut to `max_fragments` first; clipping the
    /// span coordinates and rendering the markup happens only for the
    /// survivors (#595). Previously every candidate was rendered and then
    /// discarded, which made this stage `O(spans × fragment_size)`.
    fn create_fragments(
        &self,
        text: &str,
        spans: &[HighlightSpan],
    ) -> Result<Vec<HighlightFragment>> {
        let mut candidates: Vec<(FragmentCandidate, f32)> = self
            .group_spans_into_fragments(text, spans)
            .into_iter()
            .map(|candidate| {
                let group = &spans[candidate.spans.clone()];
                let score = group.iter().map(|s| s.score).sum::<f32>() / group.len() as f32;
                (candidate, score)
            })
            .collect();

        // Sort by score (highest first). Stable, so equal scores keep span
        // order; then keep only the fragments that will be returned.
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(self.config.max_fragments);

        let mut fragments = Vec::with_capacity(candidates.len());
        for (FragmentCandidate { spans: run, window }, score) in candidates {
            // Adjust span coordinates relative to the fragment window.
            let group_spans: Vec<HighlightSpan> = spans[run]
                .iter()
                .map(|s| {
                    let relative_start = s.range.start.saturating_sub(window.start);
                    let relative_end = (s.range.end - window.start).min(window.len());
                    HighlightSpan::new(relative_start..relative_end, s.highlight, s.score)
                })
                .collect();

            // Defensive snap: span ends come from each token's `end_offset`
            // (#1258), which every tokenizer sets correctly. This guards
            // only against a filter that rewrites a token's text without
            // updating `end_offset` to match; snapping keeps the slice
            // total even then.
            let start = floor_boundary(text, window.start);
            let end = ceil_boundary(text, window.end).max(start);
            let fragment_text = self.apply_highlighting(&text[start..end], &group_spans, start)?;

            fragments.push(HighlightFragment::new(fragment_text, start, end, score));
        }

        Ok(fragments)
    }

    /// Group highlight spans into fragment candidates.
    ///
    /// Precondition: `spans` sorted by start with non-decreasing ends — the
    /// shape `merge_overlapping_spans` produces. Under it the spans
    /// overlapping any window form one contiguous run, so each window costs
    /// two binary searches instead of a scan over every span (#595).
    fn group_spans_into_fragments(
        &self,
        text: &str,
        spans: &[HighlightSpan],
    ) -> Vec<FragmentCandidate> {
        debug_assert!(
            spans.windows(2).all(|w| {
                w[0].range.start <= w[1].range.start && w[0].range.end <= w[1].range.end
            }),
            "spans must be sorted by start with non-decreasing ends"
        );

        let mut groups = Vec::new();
        let text_len = text.len();

        for span in spans {
            // Calculate fragment boundaries around this span
            let fragment_start = span
                .range
                .start
                .saturating_sub(self.config.fragment_size / 2);
            let fragment_end = (span.range.end + self.config.fragment_size / 2).min(text_len);

            // Adjust to word boundaries
            let fragment_start = self.find_word_boundary(text, fragment_start, false);
            let fragment_end = self.find_word_boundary(text, fragment_end, true);

            let window = fragment_start..fragment_end;

            // Overlap predicate `s.start < window.end && s.end > window.start`:
            // ends are non-decreasing, so the second half holds from `lo` on;
            // starts are non-decreasing, so the first half holds before `hi`.
            let lo = spans.partition_point(|s| s.range.end <= window.start);
            let hi = spans.partition_point(|s| s.range.start < window.end);

            if lo < hi {
                groups.push(FragmentCandidate {
                    spans: lo..hi,
                    window,
                });
            }
        }

        // Remove duplicate fragments (simple deduplication)
        groups.dedup_by(|a, b| (a.window.start as i32 - b.window.start as i32).abs() < 50);

        groups
    }

    /// Find a word boundary near byte offset `pos`.
    ///
    /// `pos` is a **byte** offset, and so is the return value: the caller
    /// (`group_spans_into_fragments`) feeds the result straight into
    /// `&text[fragment_range]`. The previous implementation collected
    /// `text.chars()` into a `Vec<char>` and indexed it with `pos`,
    /// conflating bytes with characters — for any non-ASCII text this
    /// returned a nonsense offset and could slice mid-character.
    fn find_word_boundary(&self, text: &str, pos: usize, forward: bool) -> usize {
        if forward {
            let mut pos = ceil_boundary(text, pos);
            while pos < text.len() {
                // `pos` is a char boundary by construction, so the slice
                // always yields at least one character.
                let c = text[pos..].chars().next().expect("pos < len");
                if !c.is_alphanumeric() {
                    break;
                }
                pos += c.len_utf8();
            }
            pos
        } else {
            let mut pos = floor_boundary(text, pos);
            while pos > 0 {
                let c = text[..pos].chars().next_back().expect("pos > 0");
                if !c.is_alphanumeric() {
                    break;
                }
                pos -= c.len_utf8();
            }
            pos
        }
    }

    /// Apply highlighting markup to text.
    fn apply_highlighting(
        &self,
        text: &str,
        spans: &[HighlightSpan],
        _offset: usize,
    ) -> Result<String> {
        if spans.is_empty() {
            return Ok(text.to_string());
        }

        let mut result = String::new();
        let mut last_pos = 0;

        for span in spans {
            if span.highlight {
                // Defensive snap: `span.range` came through `create_fragments`
                // relative to a fragment window whose own bounds were
                // already snapped, but token-derived offsets can still
                // misalign if a filter rewrites a token's text without
                // updating `end_offset` to match (#1258). Snapping here
                // keeps every slice below total.
                let start = floor_boundary(text, span.range.start).max(last_pos);
                let end = ceil_boundary(text, span.range.end).max(start);

                // Add text before the highlight
                result.push_str(&text[last_pos..start]);

                // Add highlighted text
                result.push_str(&self.config.opening_tag());
                result.push_str(&text[start..end]);
                result.push_str(&self.config.closing_tag());

                last_pos = end;
            }
        }

        // Add remaining text
        if last_pos < text.len() {
            result.push_str(&text[last_pos..]);
        }

        Ok(result)
    }

    /// Create result when no highlights are found.
    fn create_no_highlight_result(&self, field_name: &str, text: &str) -> Result<FieldHighlight> {
        let mut field_highlight = FieldHighlight::new(field_name.to_string());

        if self.config.return_entire_field_if_no_highlight {
            field_highlight.is_entire_field = true;
            field_highlight.add_fragment(HighlightFragment::new(
                text.to_string(),
                0,
                text.len(),
                0.0,
            ));
        }

        Ok(field_highlight)
    }
}

/// Utility for creating highlighted snippets without full query analysis.
#[derive(Debug)]
pub struct SimpleHighlighter {
    config: HighlightConfig,
}

impl SimpleHighlighter {
    /// Create a new simple highlighter.
    pub fn new(config: HighlightConfig) -> Self {
        SimpleHighlighter { config }
    }

    /// Pre-compile a slice of terms into reusable regex patterns.
    ///
    /// Each term becomes one case-insensitive word-boundary regex
    /// (`(?i)\bTERM\b`). Compilation is the dominant per-call cost in
    /// [`highlight_terms`](Self::highlight_terms): callers that highlight
    /// the same term set against many texts (e.g. one query × N search
    /// results) should compile once and feed the result to
    /// [`highlight_terms_compiled`](Self::highlight_terms_compiled),
    /// avoiding O(N × terms.len()) recompilations.
    ///
    /// Empty terms are skipped. Returned patterns preserve length-
    /// descending order so replacement is "longest match first" — this
    /// matches the implicit ordering of [`highlight_terms`].
    pub fn compile_patterns(terms: &[&str]) -> Vec<Regex> {
        let mut sorted_terms: Vec<&&str> = terms.iter().collect();
        sorted_terms.sort_by_key(|term| std::cmp::Reverse(term.len()));

        sorted_terms
            .into_iter()
            .filter(|term| !term.is_empty())
            .filter_map(|term| {
                let pattern = format!(r"(?i)\b{}\b", regex::escape(term));
                Regex::new(&pattern).ok()
            })
            .collect()
    }

    /// Highlight `text` using a pre-compiled set of regex patterns.
    ///
    /// `patterns` should typically come from
    /// [`compile_patterns`](Self::compile_patterns). The patterns are
    /// applied in the order given; for length-descending input (the
    /// `compile_patterns` default), the result matches
    /// [`highlight_terms`].
    pub fn highlight_terms_compiled(&self, text: &str, patterns: &[Regex]) -> String {
        let mut result = text.to_string();
        for regex in patterns {
            result = regex
                .replace_all(&result, |caps: &regex::Captures| {
                    format!(
                        "{}{}{}",
                        self.config.opening_tag(),
                        &caps[0],
                        self.config.closing_tag()
                    )
                })
                .to_string();
        }
        result
    }

    /// Highlight specific terms in text.
    ///
    /// One-shot convenience entry point: compiles a regex per term inline
    /// and replaces. When the same term set is reused across many
    /// highlight calls (e.g. one query × N search results), prefer the
    /// two-step [`compile_patterns`] / [`highlight_terms_compiled`] API
    /// to avoid recompiling on every invocation.
    pub fn highlight_terms(&self, text: &str, terms: &[&str]) -> String {
        let mut result = text.to_string();

        // Sort terms by length (longest first) to avoid partial replacements.
        let mut sorted_terms: Vec<&str> = terms.to_vec();
        sorted_terms.sort_by_key(|term| std::cmp::Reverse(term.len()));

        for term in sorted_terms {
            if !term.is_empty() {
                let pattern = format!(r"(?i)\b{}\b", regex::escape(term));
                if let Ok(regex) = Regex::new(&pattern) {
                    result = regex
                        .replace_all(&result, |caps: &regex::Captures| {
                            format!(
                                "{}{}{}",
                                self.config.opening_tag(),
                                &caps[0],
                                self.config.closing_tag()
                            )
                        })
                        .to_string();
                }
            }
        }

        result
    }

    /// Create a snippet of text around the first occurrence of any term.
    ///
    /// `max_length` counts **characters**. The previous implementation
    /// sliced raw bytes (`&text[..max_length]`, `&text[start..end]`),
    /// which panicked on any Japanese input.
    pub fn create_snippet(&self, text: &str, terms: &[&str], max_length: usize) -> String {
        /// Take at most `n` characters from the head of `s`.
        fn head(s: &str, n: usize) -> String {
            s.chars().take(n).collect()
        }

        let total_chars = text.chars().count();

        if terms.is_empty() || text.is_empty() {
            return if total_chars <= max_length {
                text.to_string()
            } else {
                format!("{}...", head(text, max_length))
            };
        }

        // Find the first occurrence of any term, as a character index.
        //
        // `find` reports a byte offset into the lower-cased copy; convert
        // it to a character index before doing any arithmetic. Case
        // folding can change character counts for a handful of code
        // points, so the result is clamped rather than trusted — the
        // char-iterator slicing below is total for any value anyway.
        let lowered = text.to_lowercase();
        let mut first_match_pos: Option<usize> = None;
        for term in terms {
            if let Some(byte_pos) = lowered.find(&term.to_lowercase()) {
                let char_pos = lowered[..byte_pos].chars().count().min(total_chars);
                if first_match_pos.is_none_or(|p| char_pos < p) {
                    first_match_pos = Some(char_pos);
                }
            }
        }

        let Some(match_pos) = first_match_pos else {
            // No matches found, return beginning of text
            return if total_chars <= max_length {
                text.to_string()
            } else {
                format!("{}...", head(text, max_length))
            };
        };

        // Create snippet around the match
        let start = match_pos.saturating_sub(max_length / 3);
        let end = (match_pos + max_length * 2 / 3).min(total_chars);

        let mut snippet: String = text
            .chars()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect();

        // Add ellipsis if we truncated
        if start > 0 {
            snippet = format!("...{snippet}");
        }
        if end < total_chars {
            snippet = format!("{snippet}...");
        }

        snippet
    }
}

/// Snap `pos` down to the nearest UTF-8 char boundary at or below it,
/// clamped to `text.len()`.
///
/// Highlight spans are byte offsets produced by arithmetic (fragment
/// windows, span merges, analyzer token lengths). Any of those can land
/// mid-character, which makes the subsequent `&text[a..b]` panic. Snapping
/// makes every slice in this module total.
#[inline]
fn floor_boundary(text: &str, pos: usize) -> usize {
    let mut pos = pos.min(text.len());
    while pos > 0 && !text.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Snap `pos` up to the nearest UTF-8 char boundary at or above it,
/// clamped to `text.len()`.
#[inline]
fn ceil_boundary(text: &str, pos: usize) -> usize {
    let len = text.len();
    let mut pos = pos.min(len);
    while pos < len && !text.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Spans of `phrase` within `tokens` (analyzer output in text order, laid
/// out by [`flatten_token_graph`]).
///
/// Mirrors the index-side phrase matcher: the first term anchors, each
/// following term must appear at the first position in
/// `expected..=expected + slop`, and the match then continues from that
/// position. Positions come from [`token_positions`], which is how
/// `tokens_to_analyzed_terms` numbers what the index stores: a stacked
/// synonym shares its word's position, and the analyzer's `Token::position`
/// is ignored, so a dropped stop word leaves no gap here, just as in the
/// index. Each phrase position lists alternative terms, any of which
/// matches there. A span runs from the first token's start to the last
/// token's end.
fn phrase_spans(tokens: &[Token], phrase: &[Vec<String>], slop: u32) -> Vec<HighlightSpan> {
    let Some((first, rest)) = phrase.split_first() else {
        return Vec::new();
    };
    let positions = token_positions(tokens);
    phrase_anchors(tokens, &positions, first)
        .filter_map(|(anchor, anchor_position)| {
            let mut previous = anchor_position;
            let mut end = anchor.end_offset;
            for alternatives in rest {
                let (token, position) =
                    next_phrase_token(tokens, &positions, previous, slop, alternatives)?;
                previous = position;
                end = token.end_offset;
            }
            // Phrases outrank single terms, as before.
            Some(HighlightSpan::new(anchor.start_offset..end, true, 2.0))
        })
        .collect()
}

/// Spans of a graph phrase within `tokens`: those [`phrase_spans`] finds
/// for each path through `arcs`, without listing the paths.
///
/// Each arc leaving node 0 anchors on its own tokens ([`phrase_anchors`]),
/// as that path's first position would. A state (anchor start, previous
/// position, end) moves along each arc with [`next_phrase_token`], whose
/// step depends only on the previous position, so the states some path
/// brings to a node are enough to go on from it. A state that reaches the
/// last node is a span from its anchor's start to its end.
fn graph_phrase_spans(tokens: &[Token], arcs: &[PhraseArc], slop: u32) -> Vec<HighlightSpan> {
    let Some(last) = arcs.iter().map(|arc| arc.to).max() else {
        return Vec::new();
    };
    let positions = token_positions(tokens);
    let mut states: Vec<Vec<(usize, u32, usize)>> = vec![Vec::new(); last as usize + 1];
    let mut settled = 0;
    for arc in arcs {
        let (before, after) = states.split_at_mut(arc.to as usize);
        let reached = &mut after[0];
        if arc.from == 0 {
            reached.extend(
                phrase_anchors(tokens, &positions, &arc.terms)
                    .map(|(anchor, position)| (anchor.start_offset, position, anchor.end_offset)),
            );
            continue;
        }

        // Every arc into `from` came earlier, so its states are final. The
        // end offset only matters at the last node.
        let states_at = &mut before[arc.from as usize];
        if arc.from != settled {
            settled = arc.from;
            states_at.sort_unstable();
            states_at.dedup_by_key(|&mut (start, previous, _)| (start, previous));
        }
        for &(start, previous, _) in states_at.iter() {
            if let Some((token, position)) =
                next_phrase_token(tokens, &positions, previous, slop, &arc.terms)
            {
                reached.push((start, position, token.end_offset));
            }
        }
    }

    let mut ranges: Vec<(usize, usize)> = states[last as usize]
        .iter()
        .map(|&(start, _, end)| (start, end))
        .collect();
    ranges.sort_unstable();
    ranges.dedup();
    ranges
        .into_iter()
        // Phrases outrank single terms, as before.
        .map(|(start, end)| HighlightSpan::new(start..end, true, 2.0))
        .collect()
}

/// Whether `token` is one of the lowercased `alternatives`.
fn token_is(token: &Token, alternatives: &[String]) -> bool {
    alternatives.iter().any(|term| {
        token.text == *term || (has_uppercase(&token.text) && token.text.to_lowercase() == *term)
    })
}

/// The tokens a phrase starting with `alternatives` anchors on, with their
/// positions: at each position, the first token that is one of them.
/// Tokens stacked at one position would repeat the same match.
fn phrase_anchors<'a>(
    tokens: &'a [Token],
    positions: &'a [u32],
    alternatives: &'a [String],
) -> impl Iterator<Item = (&'a Token, u32)> + 'a {
    let mut last_anchor_position = None;
    tokens
        .iter()
        .zip(positions)
        .filter_map(move |(token, &position)| {
            if last_anchor_position == Some(position) || !token_is(token, alternatives) {
                return None;
            }
            last_anchor_position = Some(position);
            Some((token, position))
        })
}

/// The token a phrase continues with after one at position `previous`:
/// the first that is one of `alternatives` at a position in `previous + 1
/// ..= previous + 1 + slop`, with its position, as `next_in_window` picks
/// it on the index side.
fn next_phrase_token<'a>(
    tokens: &'a [Token],
    positions: &[u32],
    previous: u32,
    slop: u32,
    alternatives: &[String],
) -> Option<(&'a Token, u32)> {
    let expected = previous.saturating_add(1);
    // Positions never decrease, so the first match from here is the one at
    // the smallest position.
    let from = positions.partition_point(|&p| p < expected);
    let last = expected.saturating_add(slop);
    tokens[from..]
        .iter()
        .zip(&positions[from..])
        .take_while(|&(_, &p)| p <= last)
        .find(|&(token, _)| token_is(token, alternatives))
        .map(|(token, &p)| (token, p))
}

/// Return `true` if `s` contains any upper-case character.
///
/// Fast path: ASCII-only strings are scanned byte by byte (`is_ascii`
/// itself is byte-level and short-circuits on the first non-ASCII byte,
/// after which the second scan does an ASCII upper-case check). Strings
/// containing non-ASCII characters fall back to a Unicode-aware char
/// iterator. Used by `Highlighter::find_highlight_spans` (#408) to avoid
/// allocating a lower-cased `String` per analyzer token when the token
/// is already in canonical (lower-cased) form.
#[inline]
fn has_uppercase(s: &str) -> bool {
    if s.is_ascii() {
        s.bytes().any(|b| b.is_ascii_uppercase())
    } else {
        s.chars().any(|c| c.is_uppercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::query::advanced_query::MultiFieldQuery;
    use crate::lexical::query::range::RangeQuery;
    use crate::lexical::query::span::{SpanNearQuery, SpanQuery, SpanQueryWrapper, SpanTermQuery};
    use crate::lexical::query::term::TermQuery;
    use crate::lexical::query::{
        AdvancedQuery, BooleanQuery, FuzzyQuery, PhraseQuery, PrefixQuery, RegexpQuery,
        WildcardQuery,
    };

    #[test]
    fn test_highlight_config() {
        let config = HighlightConfig::new()
            .tag("em".to_string())
            .css_class("highlight".to_string())
            .max_fragments(3)
            .fragment_size(100);

        assert_eq!(config.tag, "em");
        assert_eq!(config.css_class, Some("highlight".to_string()));
        assert_eq!(config.max_fragments, 3);
        assert_eq!(config.fragment_size, 100);

        assert_eq!(config.opening_tag(), "<em class=\"highlight\">");
        assert_eq!(config.closing_tag(), "</em>");
    }

    #[test]
    fn test_highlight_fragment() {
        let fragment = HighlightFragment::new(
            "This is a <mark>test</mark> fragment".to_string(),
            10,
            50,
            1.5,
        );

        assert_eq!(fragment.text, "This is a <mark>test</mark> fragment");
        assert_eq!(fragment.start_offset, 10);
        assert_eq!(fragment.end_offset, 50);
        assert_eq!(fragment.score, 1.5);
    }

    #[test]
    fn test_field_highlight() {
        let mut field_highlight = FieldHighlight::new("content".to_string());

        field_highlight.add_fragment(HighlightFragment::new("fragment 1".to_string(), 0, 10, 1.0));
        field_highlight.add_fragment(HighlightFragment::new(
            "fragment 2".to_string(),
            20,
            30,
            2.0,
        ));

        assert_eq!(field_highlight.fragments.len(), 2);
        assert_eq!(field_highlight.best_fragment().unwrap().score, 2.0);
        assert_eq!(
            field_highlight.combined_text(" | "),
            "fragment 1 | fragment 2"
        );
    }

    #[test]
    fn test_simple_highlighter() {
        let config = HighlightConfig::default();
        let highlighter = SimpleHighlighter::new(config);

        let text = "This is a test document with some test content.";
        let terms = vec!["test", "content"];

        let highlighted = highlighter.highlight_terms(text, &terms);
        assert!(highlighted.contains("<mark>test</mark>"));
        assert!(highlighted.contains("<mark>content</mark>"));

        let snippet = highlighter.create_snippet(text, &terms, 30);
        assert!(snippet.len() <= 35); // Account for ellipsis
        assert!(snippet.contains("test"));
    }

    /// `create_snippet` used to slice `&text[..max_length]` and
    /// `&text[start..end]` as raw byte ranges, which panicked for any
    /// Japanese text whose cut point fell inside a character.
    #[test]
    fn create_snippet_does_not_panic_on_japanese_text() {
        let config = HighlightConfig::default();
        let highlighter = SimpleHighlighter::new(config);
        let text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。".repeat(3);

        let with_match = highlighter.create_snippet(&text, &["猫"], 20);
        assert!(with_match.contains('猫'));

        let without_match = highlighter.create_snippet(&text, &["犬"], 20);
        assert!(!without_match.is_empty());
    }

    /// `max_length` counts characters: truncating 100 Japanese characters
    /// to 30 must yield exactly 30 characters (plus the `...` suffix),
    /// not an arbitrary byte-length cut.
    #[test]
    fn create_snippet_truncates_by_characters() {
        let config = HighlightConfig::default();
        let highlighter = SimpleHighlighter::new(config);
        let text = "あ".repeat(100);

        let snippet = highlighter.create_snippet(&text, &[], 30);
        assert!(snippet.ends_with("..."));
        assert_eq!(snippet.chars().count(), 33); // 30 chars + "..."
    }

    /// No-match path (terms given but none found) must also truncate on
    /// character boundaries.
    #[test]
    fn create_snippet_without_matches_truncates_japanese_head() {
        let config = HighlightConfig::default();
        let highlighter = SimpleHighlighter::new(config);
        let text = "吾輩は猫である。".repeat(20);

        let snippet = highlighter.create_snippet(&text, &["犬"], 30);
        assert!(snippet.ends_with("..."));
    }

    #[test]
    fn test_highlighter_extract_terms() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);

        let query = TermQuery::new("field", "search");
        let terms = highlighter.extract_query_terms(&query, "field");
        assert!(
            matches!(terms.as_slice(), [HighlightTerm::Exact(term)] if term == "search"),
            "the query tree yields the exact term, got {terms:?}"
        );
        assert!(
            highlighter.extract_query_terms(&query, "other").is_empty(),
            "require_field_match is on by default"
        );
    }

    #[test]
    fn test_has_uppercase() {
        assert!(!has_uppercase("rust"));
        assert!(!has_uppercase("123"));
        assert!(!has_uppercase(""));
        assert!(has_uppercase("Rust"));
        assert!(has_uppercase("rusT"));
        // Non-ASCII without case (Japanese) is treated as lowercase.
        assert!(!has_uppercase("検索"));
        // Non-ASCII upper-case (Greek capital alpha) takes the Unicode path.
        assert!(has_uppercase("Α"));
        assert!(!has_uppercase("α"));
    }

    #[test]
    fn test_find_highlight_spans_case_insensitive() {
        // Verifies that the #408 fast path (skip `to_lowercase()` when the
        // token is already lowercase) preserves case-insensitive matching:
        // an upper-cased token in the source text must still match a
        // lower-cased term.
        let highlighter = Highlighter::new(HighlightConfig::default());
        let terms = [HighlightTerm::Exact("rust".to_string())];

        let spans = highlighter
            .find_highlight_spans("learning Rust today", &terms)
            .unwrap();
        assert!(
            !spans.is_empty(),
            "uppercase 'Rust' should still match lowercased term 'rust'"
        );
    }

    #[test]
    fn test_merge_overlapping_spans() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);

        let spans = vec![
            HighlightSpan::new(0..5, true, 1.0),
            HighlightSpan::new(3..8, true, 1.5),
            HighlightSpan::new(10..15, true, 1.2),
        ];

        let merged = highlighter.merge_overlapping_spans(spans);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].range, 0..8);
        assert_eq!(merged[1].range, 10..15);
    }

    #[test]
    fn test_word_boundary_finding() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);

        let text = "The quick brown fox jumps";

        // Find word boundary before position 7 (middle of "quick")
        let boundary = highlighter.find_word_boundary(text, 7, false);
        assert_eq!(boundary, 4); // Start of "quick"

        // Find word boundary after position 7
        let boundary = highlighter.find_word_boundary(text, 7, true);
        assert_eq!(boundary, 9); // End of "quick"
    }

    /// `find_word_boundary` used to collect `text.chars()` into a
    /// `Vec<char>` and index it with a **byte** offset, conflating bytes
    /// and characters. For Japanese text the returned offset was
    /// nonsensical and often not even a valid char boundary.
    #[test]
    fn find_word_boundary_returns_char_boundaries_for_japanese() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);
        let text = "吾輩は猫である。名前はまだ無い。";

        for pos in 0..=text.len() {
            let back = highlighter.find_word_boundary(text, pos, false);
            let fwd = highlighter.find_word_boundary(text, pos, true);
            assert!(
                text.is_char_boundary(back),
                "backward boundary {back} from pos {pos} is not a char boundary"
            );
            assert!(
                text.is_char_boundary(fwd),
                "forward boundary {fwd} from pos {pos} is not a char boundary"
            );
        }
    }

    /// A mid-character byte position (not just an out-of-range one) must
    /// not panic and must snap to a real char boundary.
    #[test]
    fn find_word_boundary_snaps_a_mid_character_position() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);
        let text = "猫";
        // Byte 1 and 2 are both mid-character for a 3-byte kanji.
        let back = highlighter.find_word_boundary(text, 1, false);
        let fwd = highlighter.find_word_boundary(text, 2, true);
        assert!(text.is_char_boundary(back));
        assert!(text.is_char_boundary(fwd));
    }

    /// End-to-end: highlighting a Japanese field must not panic. This is
    /// the combination of `find_highlight_spans`, `group_spans_into_fragments`
    /// (which calls `find_word_boundary`), and `apply_highlighting`.
    #[test]
    fn highlight_japanese_text_end_to_end_does_not_panic() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);
        let query = TermQuery::new("body", "search");
        let text = "吾輩は猫である。".repeat(50);
        let result = highlighter.highlight(&query, "body", &text);
        assert!(result.is_ok(), "highlighting Japanese text must not panic");
    }

    /// `max_analyzed_chars` is named for characters; a Japanese text
    /// longer than the configured limit (in characters, not bytes) must
    /// be truncated without panicking.
    #[test]
    fn highlight_does_not_panic_when_max_analyzed_chars_cuts_a_multibyte_char() {
        let config = HighlightConfig {
            max_analyzed_chars: 10,
            ..HighlightConfig::new()
        };
        let highlighter = Highlighter::new(config);
        let query = TermQuery::new("body", "猫");
        let text = "吾輩は猫である。名前はまだ無い。".repeat(3);
        let result = highlighter.highlight(&query, "body", &text);
        assert!(result.is_ok(), "must not panic when truncating mid-field");
    }

    // --- Fragment grouping / rendering (#595) ---

    /// The pre-#595 `create_fragments` pipeline, kept as the reference:
    /// every window rescans every span, every candidate is rendered, then a
    /// stable score sort and the `max_fragments` cut. Also asserts the
    /// invariant the binary-search version relies on -- for sorted,
    /// non-overlapping spans each window's overlapping set is one
    /// contiguous index run.
    fn reference_fragments(
        h: &Highlighter,
        text: &str,
        spans: &[HighlightSpan],
    ) -> Vec<HighlightFragment> {
        let text_len = text.len();
        let mut groups: Vec<(Vec<HighlightSpan>, Range<usize>)> = Vec::new();
        for span in spans {
            let fragment_start = span.range.start.saturating_sub(h.config.fragment_size / 2);
            let fragment_end = (span.range.end + h.config.fragment_size / 2).min(text_len);
            let fragment_start = h.find_word_boundary(text, fragment_start, false);
            let fragment_end = h.find_word_boundary(text, fragment_end, true);
            let window = fragment_start..fragment_end;

            let indices: Vec<usize> = spans
                .iter()
                .enumerate()
                .filter(|(_, s)| s.range.start < window.end && s.range.end > window.start)
                .map(|(i, _)| i)
                .collect();
            assert!(
                indices.windows(2).all(|w| w[1] == w[0] + 1),
                "overlapping spans must form one contiguous run: {indices:?}"
            );

            let group: Vec<HighlightSpan> = indices
                .iter()
                .map(|&i| {
                    let s = &spans[i];
                    let relative_start = s.range.start.saturating_sub(window.start);
                    let relative_end = (s.range.end - window.start).min(window.len());
                    HighlightSpan::new(relative_start..relative_end, s.highlight, s.score)
                })
                .collect();
            if !group.is_empty() {
                groups.push((group, window));
            }
        }
        groups.dedup_by(|(_, a), (_, b)| (a.start as i32 - b.start as i32).abs() < 50);

        let mut fragments = Vec::new();
        for (group, window) in groups {
            let start = floor_boundary(text, window.start);
            let end = ceil_boundary(text, window.end).max(start);
            let rendered = h
                .apply_highlighting(&text[start..end], &group, start)
                .unwrap();
            let score = group.iter().map(|s| s.score).sum::<f32>() / group.len() as f32;
            fragments.push(HighlightFragment::new(rendered, start, end, score));
        }
        fragments.sort_by(|a, b| b.score.total_cmp(&a.score));
        fragments.truncate(h.config.max_fragments);
        fragments
    }

    fn fragment_keys(fragments: &[HighlightFragment]) -> Vec<(String, usize, usize, u32)> {
        fragments
            .iter()
            .map(|f| {
                (
                    f.text.clone(),
                    f.start_offset,
                    f.end_offset,
                    f.score.to_bits(),
                )
            })
            .collect()
    }

    /// Deterministic LCG so the generated corpora are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Text from a vocabulary mixing ASCII and multi-byte words, plus the
    /// byte ranges of its words (maximal alphanumeric runs -- the shape
    /// `find_word_boundary` snaps to).
    fn synthetic_text(rng: &mut Lcg, words: usize) -> (String, Vec<Range<usize>>) {
        const VOCAB: &[&str] = &[
            "rust",
            "search",
            "検索",
            "猫である",
            "naïve",
            "42",
            "engine",
            "x",
        ];
        const SEPARATORS: &[&str] = &[" ", ", ", "。", "\n", "  "];

        let mut text = String::new();
        for i in 0..words {
            if i > 0 {
                text.push_str(SEPARATORS[rng.below(SEPARATORS.len() as u64) as usize]);
            }
            text.push_str(VOCAB[rng.below(VOCAB.len() as u64) as usize]);
        }

        let mut ranges = Vec::new();
        let mut current: Option<usize> = None;
        for (idx, c) in text.char_indices() {
            if c.is_alphanumeric() {
                current.get_or_insert(idx);
            } else if let Some(start) = current.take() {
                ranges.push(start..idx);
            }
        }
        if let Some(start) = current {
            ranges.push(start..text.len());
        }
        (text, ranges)
    }

    /// Sorted, strictly non-overlapping spans over `word_ranges`: each word
    /// is picked with probability `per_mille`/1000, a picked word is
    /// sometimes fused with its successor into a phrase-like span, and
    /// `force_edges` guarantees the first and last words are covered.
    fn synthetic_spans(
        rng: &mut Lcg,
        word_ranges: &[Range<usize>],
        per_mille: u64,
        force_edges: bool,
    ) -> Vec<HighlightSpan> {
        let mut spans = Vec::new();
        let mut i = 0;
        while i < word_ranges.len() {
            let forced = force_edges && (i == 0 || i + 1 == word_ranges.len());
            if forced || rng.below(1000) < per_mille {
                let mut range = word_ranges[i].clone();
                if i + 2 < word_ranges.len() && rng.below(4) == 0 {
                    range.end = word_ranges[i + 1].end;
                    i += 1;
                }
                let score = 1.0 + rng.below(2000) as f32 / 1000.0;
                let highlight = rng.below(8) != 0;
                spans.push(HighlightSpan::new(range, highlight, score));
            }
            i += 1;
        }
        assert!(
            spans.windows(2).all(|w| w[0].range.end < w[1].range.start),
            "generator must produce strictly non-overlapping spans"
        );
        spans
    }

    /// `create_fragments` must produce exactly what the quadratic reference
    /// produces -- same fragments, offsets, scores and order -- across
    /// fragment sizes from 0 (window = the word itself) to far beyond the
    /// text, span densities, corpus sizes and `max_fragments` cuts.
    #[test]
    fn create_fragments_matches_the_quadratic_reference() {
        let mut rng = Lcg(0x5eed_0595);
        let mut cases = 0usize;
        let mut nonempty_cases = 0usize;

        for &fragment_size in &[0usize, 1, 7, 30, 150, 1_000, 100_000] {
            for &max_fragments in &[usize::MAX, 5, 1] {
                let h = Highlighter::new(
                    HighlightConfig::new()
                        .fragment_size(fragment_size)
                        .max_fragments(max_fragments),
                );
                for &per_mille in &[20u64, 300, 1000] {
                    for &words in &[1usize, 3, 60, 800] {
                        for &force_edges in &[false, true] {
                            let (text, word_ranges) = synthetic_text(&mut rng, words);
                            let spans =
                                synthetic_spans(&mut rng, &word_ranges, per_mille, force_edges);

                            let expected = reference_fragments(&h, &text, &spans);
                            let actual = h.create_fragments(&text, &spans).unwrap();
                            assert_eq!(
                                fragment_keys(&actual),
                                fragment_keys(&expected),
                                "fragment_size={fragment_size} max_fragments={max_fragments} \
                                 per_mille={per_mille} words={words} force_edges={force_edges}"
                            );

                            cases += 1;
                            if !expected.is_empty() {
                                nonempty_cases += 1;
                            }
                        }
                    }
                }
            }
        }

        assert!(
            nonempty_cases > cases / 2,
            "the generator must mostly exercise non-empty fragment sets ({nonempty_cases}/{cases})"
        );
    }

    /// Same equivalence, but on spans from the real `find_highlight_spans`
    /// pipeline rather than the synthetic generator.
    #[test]
    fn create_fragments_matches_the_quadratic_reference_on_analyzer_spans() {
        let text = "alpha rust beta rust gamma delta rust ".repeat(40);
        let terms = [HighlightTerm::Exact("rust".to_string())];

        for &fragment_size in &[10usize, 150] {
            for &max_fragments in &[usize::MAX, 5] {
                let h = Highlighter::new(
                    HighlightConfig::new()
                        .fragment_size(fragment_size)
                        .max_fragments(max_fragments),
                );
                let spans = h.find_highlight_spans(&text, &terms).unwrap();
                assert!(spans.len() >= 100, "the corpus must yield many spans");

                let expected = reference_fragments(&h, &text, &spans);
                let actual = h.create_fragments(&text, &spans).unwrap();
                assert_eq!(
                    fragment_keys(&actual),
                    fragment_keys(&expected),
                    "fragment_size={fragment_size} max_fragments={max_fragments}"
                );
            }
        }
    }

    /// Only `max_fragments` fragments come back, best score first, and
    /// equal scores keep span order (the sort is stable).
    #[test]
    fn create_fragments_keeps_tie_order_and_truncates() {
        // "rust" + 60 spaces, five times: spans at 0, 64, 128, 192, 256, so
        // no two windows start within 50 bytes and dedup never fires.
        let text = format!("rust{}", " ".repeat(60)).repeat(5);
        let spans: Vec<HighlightSpan> = [1.0f32, 3.0, 1.0, 3.0, 2.0]
            .iter()
            .enumerate()
            .map(|(i, &score)| HighlightSpan::new(i * 64..i * 64 + 4, true, score))
            .collect();

        let h = Highlighter::new(HighlightConfig::new().fragment_size(4).max_fragments(3));
        let fragments = h.create_fragments(&text, &spans).unwrap();
        assert_eq!(fragments.len(), 3);
        assert_eq!(
            fragments.iter().map(|f| f.start_offset).collect::<Vec<_>>(),
            vec![62, 190, 254],
            "the two 3.0s keep span order, then the 2.0"
        );
        assert_eq!(
            fragments.iter().map(|f| f.score).collect::<Vec<_>>(),
            vec![3.0, 3.0, 2.0]
        );
        assert_eq!(fragments[0].text, "  <mark>rust</mark>  ");

        let none = Highlighter::new(HighlightConfig::new().fragment_size(4).max_fragments(0));
        assert!(none.create_fragments(&text, &spans).unwrap().is_empty());
    }

    #[test]
    fn merge_overlapping_spans_empty_input() {
        let h = Highlighter::new(HighlightConfig::default());
        assert!(h.merge_overlapping_spans(Vec::new()).is_empty());
    }

    // --- Query-tree term extraction (#594) ---

    /// Every `<mark>…</mark>` run across `fragments`, in order.
    fn marked(fragments: &[HighlightFragment]) -> Vec<String> {
        let mut out = Vec::new();
        for fragment in fragments {
            let mut rest = fragment.text.as_str();
            while let Some(start) = rest.find("<mark>") {
                let after = &rest[start + "<mark>".len()..];
                let end = after.find("</mark>").expect("closing tag");
                out.push(after[..end].to_string());
                rest = &after[end + "</mark>".len()..];
            }
        }
        out
    }

    fn highlight_marks<Q: Query>(query: &Q, field: &str, text: &str) -> Vec<String> {
        let highlighter = Highlighter::new(HighlightConfig::default());
        marked(&highlighter.highlight(query, field, text).unwrap().fragments)
    }

    #[test]
    fn term_query_highlights_its_term() {
        assert_eq!(
            highlight_marks(
                &TermQuery::new("body", "rust"),
                "body",
                "Learning Rust today"
            ),
            ["Rust"]
        );
    }

    #[test]
    fn boolean_query_skips_must_not() {
        let mut query = BooleanQuery::new();
        query.add_must(Box::new(TermQuery::new("body", "rust")));
        query.add_must_not(Box::new(TermQuery::new("body", "java")));
        query.add_should(Box::new(TermQuery::new("body", "search")));
        assert_eq!(
            highlight_marks(&query, "body", "rust search java"),
            ["rust", "search"]
        );
    }

    #[test]
    fn phrase_query_highlights_only_adjacent_occurrences() {
        let query = PhraseQuery::new("body", vec!["hello".into(), "world".into()]);
        assert_eq!(
            highlight_marks(&query, "body", "hello world. world hello. hello big world"),
            ["hello world"]
        );
    }

    #[test]
    fn phrase_query_honours_slop_like_phrase_matcher() {
        let phrase = |slop: u32| {
            PhraseQuery::new("body", vec!["hello".into(), "world".into()]).with_slop(slop)
        };
        assert_eq!(
            highlight_marks(&phrase(1), "body", "hello big world"),
            ["hello big world"]
        );
        assert!(highlight_marks(&phrase(0), "body", "hello big world").is_empty());
    }

    /// A dropped stop word leaves no gap in the positions the index stores:
    /// the index matches `"hello world"`
    /// against "hello the world" at slop 0, and the highlighter must too.
    #[test]
    fn phrase_across_a_dropped_stop_word_highlights_at_slop_0() {
        let phrase = |slop: u32| {
            PhraseQuery::new("body", vec!["hello".into(), "world".into()]).with_slop(slop)
        };
        assert_eq!(
            highlight_marks(&phrase(0), "body", "hello the world"),
            ["hello the world"]
        );
        // `big` survives analysis, so it is a real gap.
        assert!(highlight_marks(&phrase(0), "body", "hello the big world").is_empty());
        assert_eq!(
            highlight_marks(&phrase(1), "body", "hello the big world"),
            ["hello the big world"]
        );
    }

    /// `UnicodeWordTokenizer` numbers whitespace and punctuation before
    /// dropping them, so `hello` and `world` come out at `Token::position`
    /// 0 and 3. The index ignores `Token::position`, and so must the
    /// highlighter.
    #[test]
    fn unicode_word_tokenizer_phrase_highlights_at_slop_0() {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::tokenizer::Tokenizer;
        use crate::analysis::tokenizer::unicode_word::UnicodeWordTokenizer;

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(UnicodeWordTokenizer::new());
        let highlighter = Highlighter::with_analyzer(
            HighlightConfig::default(),
            Box::new(PipelineAnalyzer::new(tokenizer)),
        );
        let query = PhraseQuery::new("body", vec!["hello".into(), "world".into()]);
        let fragments = highlighter
            .highlight(&query, "body", "hello, world")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["hello, world"]);
    }

    /// A highlighter whose analyzer stacks `large` on `big` (and the
    /// reverse), as `SynonymGraphFilter` does.
    fn synonym_highlighter() -> Highlighter {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::synonym::dictionary::SynonymDictionary;
        use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);
        Highlighter::with_analyzer(
            HighlightConfig::default(),
            Box::new(
                PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
                    .add_filter(Arc::new(SynonymGraphFilter::new(dict, true))),
            ),
        )
    }

    /// #1252: the index puts a stacked synonym at its word's position, so
    /// `"a large dog"` matches "a big dog" at slop 0 there, and the
    /// highlighter must count positions the same way.
    #[test]
    fn phrase_through_a_stacked_synonym_highlights_at_slop_0() {
        let query = PhraseQuery::new("body", vec!["a".into(), "large".into(), "dog".into()]);
        let fragments = synonym_highlighter()
            .highlight(&query, "body", "a big dog barks")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["a big dog"]);
    }

    /// Any alternative matches at its position, as in the phrase matcher,
    /// and tokens stacked at one position anchor one span, not two.
    #[test]
    fn phrase_alternatives_highlight_whichever_occurs() {
        let alternatives = || {
            PhraseQuery::from_positions(
                "body",
                vec![
                    vec!["red".into()],
                    vec!["big".into(), "large".into()],
                    vec!["dog".into()],
                ],
            )
        };
        assert_eq!(
            highlight_marks(
                &alternatives(),
                "body",
                "red large dog. red big dog. red cat"
            ),
            ["red large dog", "red big dog"]
        );

        let stacked_first = PhraseQuery::from_positions(
            "body",
            vec![vec!["big".into(), "large".into()], vec!["dog".into()]],
        );
        let highlighter = synonym_highlighter();
        let text = "a big dog";
        let tokens: Vec<Token> = highlighter.analyzer.analyze(text).unwrap().collect();
        assert_eq!(phrase_spans(&tokens, stacked_first.positions(), 0).len(), 1);
    }

    /// A stacked token takes no position of its own, so it does not widen
    /// the slop window: "big dog" is two positions, not three.
    #[test]
    fn a_stacked_synonym_does_not_count_against_the_slop() {
        let query = PhraseQuery::new("body", vec!["a".into(), "dog".into()]);
        let fragments = synonym_highlighter()
            .highlight(&query, "body", "a big dog")
            .unwrap()
            .fragments;
        assert!(marked(&fragments).is_empty());
        let fragments = synonym_highlighter()
            .highlight(&query.with_slop(1), "body", "a big dog")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["a big dog"]);
    }

    /// #1258: a synonym token carries its matched word's `start_offset` and
    /// `end_offset`, but not its own `text.len()` — "large" (5 bytes) stands
    /// in for "big" (3 bytes). A span computed from `text.len()` overruns
    /// into the following word; `end_offset` gives the real extent.
    #[test]
    fn a_term_matching_through_a_synonym_highlights_only_the_original_word() {
        let fragments = synonym_highlighter()
            .highlight(&TermQuery::new("body", "large"), "body", "a big dog barks")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["big"]);
    }

    /// #1258: same overrun, at a phrase's end.
    #[test]
    fn a_phrase_ending_on_a_synonym_ends_at_the_original_words_end() {
        let query = PhraseQuery::new("body", vec!["a".into(), "large".into()]);
        let fragments = synonym_highlighter()
            .highlight(&query, "body", "a big dog")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["a big"]);
    }

    // ---- Graph phrases (#1271) ----

    fn whitespace_analyzer() -> Box<dyn Analyzer> {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        Box::new(PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new())))
    }

    /// Stacks {a, b c} and {d, c} as `SynonymGraphFilter` does, so tokens
    /// share positions and synonyms span their words' offsets.
    fn stacking_analyzer() -> Box<dyn Analyzer> {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::synonym::dictionary::SynonymDictionary;
        use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["a".to_string(), "b c".to_string()]);
        dict.add_synonym_group(vec!["d".to_string(), "c".to_string()]);
        Box::new(
            PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
                .add_filter(Arc::new(SynonymGraphFilter::new(dict, true))),
        )
    }

    /// The tokens of `text` laid out as phrases are matched on them.
    fn phrase_tokens(analyzer: &dyn Analyzer, text: &str) -> Vec<Token> {
        flatten_token_graph(analyzer.analyze(text).unwrap().collect())
    }

    /// The distinct ranges of `spans`, sorted: what they mark once merged.
    fn ranges(spans: Vec<HighlightSpan>) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = spans
            .into_iter()
            .map(|span| (span.range.start, span.range.end))
            .collect();
        ranges.sort_unstable();
        ranges.dedup();
        ranges
    }

    /// The walk marks exactly what one phrase per path marks, anchored on
    /// the same tokens, over plain and stacked tokens at every slop.
    #[test]
    fn graph_phrase_spans_equal_the_union_of_path_spans() {
        use crate::lexical::query::graph_phrase::test_support::{
            Lcg as GraphLcg, paths, random_graph, random_texts,
        };

        let mut rng = GraphLcg(1271);
        let analyzers = [whitespace_analyzer(), stacking_analyzer()];
        let mut compared = 0;
        let mut marked = 0;
        for text in random_texts(&mut rng, 500) {
            for analyzer in &analyzers {
                let tokens = phrase_tokens(analyzer.as_ref(), &text);
                let Some(arcs) = random_graph(&mut rng) else {
                    continue;
                };
                for slop in 0..3 {
                    let expected = ranges(
                        paths(&arcs)
                            .iter()
                            .flat_map(|path| phrase_spans(&tokens, path, slop))
                            .collect(),
                    );
                    assert_eq!(
                        ranges(graph_phrase_spans(&tokens, &arcs, slop)),
                        expected,
                        "slop {slop}, arcs {arcs:?}, text {text:?}"
                    );
                    compared += 1;
                    marked += usize::from(!expected.is_empty());
                }
            }
        }
        assert!(compared > 1000, "only {compared} comparisons");
        assert!(marked > compared / 4, "only {marked} of {compared} marked");
    }

    #[test]
    fn graph_phrase_on_a_chain_equals_phrase_spans() {
        use crate::lexical::query::graph_phrase::test_support::{arc, paths};

        let arcs = [
            arc(0, 1, &["a"]),
            arc(1, 2, &["big", "large"]),
            arc(2, 3, &["dog"]),
        ];
        let tokens = phrase_tokens(
            whitespace_analyzer().as_ref(),
            "a big dog or a very large dog",
        );
        for slop in 0..2 {
            let expected = ranges(phrase_spans(&tokens, &paths(&arcs)[0], slop));
            assert!(!expected.is_empty());
            assert_eq!(ranges(graph_phrase_spans(&tokens, &arcs, slop)), expected);
        }
    }

    /// Each path that occurs is marked from its first word to its last.
    #[test]
    fn graph_phrase_query_highlights_each_path_that_occurs() {
        use crate::lexical::query::GraphPhraseQuery;
        use crate::lexical::query::graph_phrase::test_support::arc;

        let query = GraphPhraseQuery::from_arcs(
            "body",
            vec![
                arc(0, 1, &["machine"]),
                arc(0, 2, &["ml"]),
                arc(1, 2, &["learning"]),
                arc(2, 3, &["is"]),
            ],
        );
        let highlighter =
            Highlighter::with_analyzer(HighlightConfig::default(), whitespace_analyzer());
        let fragments = highlighter
            .highlight(&query, "body", "ml is fun, machine learning is fun")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["ml is", "machine learning is"]);
    }

    #[test]
    fn prefix_query_highlights_prefixed_tokens() {
        assert_eq!(
            highlight_marks(
                &PrefixQuery::new("body", "rust"),
                "body",
                "rust rustacean trust"
            ),
            ["rust", "rustacean"]
        );
    }

    #[test]
    fn wildcard_query_highlights_matching_tokens() {
        let query = WildcardQuery::new("body", "r?st").unwrap();
        assert_eq!(
            highlight_marks(&query, "body", "rust rest roast"),
            ["rust", "rest"]
        );
    }

    #[test]
    fn regexp_query_highlights_matching_tokens() {
        let query = RegexpQuery::new("body", "^ru.*").unwrap();
        assert_eq!(
            highlight_marks(&query, "body", "rust trust rusty"),
            ["rust", "rusty"]
        );
    }

    #[test]
    fn fuzzy_query_highlights_tokens_within_edit_distance() {
        let query = FuzzyQuery::new("body", "rust").max_edits(1);
        assert_eq!(
            highlight_marks(&query, "body", "rust rusty roast"),
            ["rust", "rusty"]
        );
    }

    #[test]
    fn span_query_wrapper_highlights_span_terms() {
        let clauses: Vec<Box<dyn SpanQuery>> = vec![
            Box::new(SpanTermQuery::new("body", "rust")),
            Box::new(SpanTermQuery::new("body", "safety")),
        ];
        let query = SpanQueryWrapper::new(Box::new(SpanNearQuery::new("body", clauses, 3, true)));
        assert_eq!(
            highlight_marks(&query, "body", "rust brings safety"),
            ["rust", "safety"]
        );
    }

    #[test]
    fn multi_field_query_highlights_configured_fields_only() {
        let query = MultiFieldQuery::new("rust".to_string()).add_field("title".to_string(), 1.0);
        assert_eq!(highlight_marks(&query, "title", "rust title"), ["rust"]);
        assert!(highlight_marks(&query, "body", "rust body").is_empty());
    }

    #[test]
    fn advanced_query_skips_negative_filters() {
        let query = AdvancedQuery::new(Box::new(TermQuery::new("body", "rust")))
            .with_negative_filter(Box::new(TermQuery::new("body", "slow")));
        assert_eq!(
            highlight_marks(&query, "body", "rust is not slow"),
            ["rust"]
        );
    }

    #[test]
    fn range_query_highlights_nothing() {
        let query = RangeQuery::new("body", Some("a".to_string()), Some("z".to_string()));
        assert!(highlight_marks(&query, "body", "anything at all").is_empty());
    }

    #[test]
    fn require_field_match_filters_other_fields() {
        let query = TermQuery::new("title", "rust");
        let text = "rust everywhere";

        assert!(
            highlight_marks(&query, "body", text).is_empty(),
            "a title term must not highlight the body by default"
        );

        let cross_field = Highlighter::new(HighlightConfig::default().require_field_match(false));
        let fragments = cross_field
            .highlight(&query, "body", text)
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["rust"]);
    }

    /// Terms are matched against the highlighter's own analyzer output, so a
    /// character-level analyzer lets a single kanji term highlight inside
    /// running Japanese text (the default `\w+` tokenizer would keep the
    /// whole run as one token).
    #[test]
    fn japanese_unigram_analyzer_highlights_kanji_term() {
        use std::sync::Arc;

        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::tokenizer::Tokenizer;
        use crate::analysis::tokenizer::ngram::NgramTokenizer;

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(NgramTokenizer::new(1, 1).unwrap());
        let highlighter = Highlighter::with_analyzer(
            HighlightConfig::default(),
            Box::new(PipelineAnalyzer::new(tokenizer)),
        );
        let query = TermQuery::new("body", "猫");
        let fragments = highlighter
            .highlight(&query, "body", "吾輩は猫である。")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["猫"]);
    }

    // --- Engine integration surface (#1134) ---

    /// The engine holds its resolved query as `Box<dyn Query>`; `highlight`
    /// must accept the unsized trait object directly.
    #[test]
    fn highlight_accepts_an_unsized_dyn_query() {
        let query: Box<dyn Query> = Box::new(TermQuery::new("body", "rust"));
        let highlighter = Highlighter::new(HighlightConfig::default());
        let fragments = highlighter
            .highlight(query.as_ref(), "body", "Learning Rust today")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["Rust"]);
    }

    /// An `Arc<dyn Analyzer>` shared with the index drives tokenisation the
    /// same way a boxed one does.
    #[test]
    fn with_shared_analyzer_uses_the_given_analyzer() {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::tokenizer::Tokenizer;
        use crate::analysis::tokenizer::ngram::NgramTokenizer;

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(NgramTokenizer::new(1, 1).unwrap());
        let analyzer: Arc<dyn Analyzer> = Arc::new(PipelineAnalyzer::new(tokenizer));
        let highlighter = Highlighter::with_shared_analyzer(HighlightConfig::default(), analyzer);
        let fragments = highlighter
            .highlight(&TermQuery::new("body", "猫"), "body", "吾輩は猫である。")
            .unwrap()
            .fragments;
        assert_eq!(marked(&fragments), ["猫"]);
    }
}
