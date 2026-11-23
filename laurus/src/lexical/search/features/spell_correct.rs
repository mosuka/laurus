//! Spell-corrected search functionality that integrates spelling correction with search.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::query::LexicalSearchResults;
use crate::lexical::search::searcher::LexicalSearchRequest;
use crate::lexical::store::LexicalStore;
use crate::spelling::corrector::{
    CorrectionResult, CorrectorConfig, DidYouMean, SpellingCorrector,
};

/// Search results with spelling correction information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellCorrectedSearchResults {
    /// The original search results.
    pub results: LexicalSearchResults,
    /// Spelling correction information.
    pub correction: CorrectionResult,
    /// Whether the search was performed with the corrected query.
    pub used_correction: bool,
    /// "Did you mean?" suggestion if available.
    pub did_you_mean: Option<String>,
}

impl SpellCorrectedSearchResults {
    /// Create new spell-corrected search results.
    pub fn new(results: LexicalSearchResults, correction: CorrectionResult) -> Self {
        SpellCorrectedSearchResults {
            results,
            correction,
            used_correction: false,
            did_you_mean: None,
        }
    }

    /// Get the query that was actually used for search.
    pub fn effective_query(&self) -> &str {
        self.correction.query()
    }

    /// Check if any spelling corrections were suggested.
    pub fn has_suggestions(&self) -> bool {
        self.correction.has_suggestions()
    }

    /// Check if auto-correction was applied.
    pub fn was_auto_corrected(&self) -> bool {
        self.correction.auto_corrected
    }

    /// Check if a "Did you mean?" suggestion should be shown.
    pub fn should_show_did_you_mean(&self) -> bool {
        self.did_you_mean.is_some() || self.correction.should_show_did_you_mean()
    }

    /// Get the correction confidence score.
    pub fn correction_confidence(&self) -> f64 {
        self.correction.confidence
    }
}

/// Strategy controlling when [`SpellCorrectedSearchEngine`] invokes the
/// expensive [`SpellingCorrector::correct`] edit-distance scan (#411).
///
/// The corrector walks the term dictionary on every call; for queries
/// that already return plenty of hits the work is wasted. The default
/// `OnLowResults { threshold: 10 }` skips the corrector entirely for
/// healthy queries and runs it only when the original search returns
/// fewer than 10 hits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpellCorrectStrategy {
    /// Always invoke the corrector. Byte-identical to pre-#411 behaviour.
    Always,
    /// Invoke the corrector only when the original search returned
    /// fewer than `threshold` hits. The default strategy.
    OnLowResults { threshold: u64 },
    /// Never invoke the corrector. The "no spell correction" path,
    /// useful for benchmarking and for users who prefer the
    /// engine-level `enabled = false` toggle's semantics without
    /// paying the wrapper construction cost.
    Never,
}

impl Default for SpellCorrectStrategy {
    fn default() -> Self {
        // Skip the corrector for any query that already returns at
        // least 10 hits — the typo-free common case in production
        // workloads.
        SpellCorrectStrategy::OnLowResults { threshold: 10 }
    }
}

/// Configuration for spell-corrected search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellCorrectedSearchConfig {
    /// Whether to enable spell correction.
    pub enabled: bool,
    /// Configuration for the spelling corrector.
    pub corrector_config: CorrectorConfig,
    /// Whether to retry search with corrected query if original query has no results.
    pub retry_with_correction: bool,
    /// Whether to show "Did you mean?" for poor results with low confidence.
    pub show_did_you_mean: bool,
    /// Minimum number of results before suggesting corrections.
    pub min_results_for_suggestions: usize,
    /// Gating strategy for [`SpellingCorrector::correct`] (#411).
    /// Defaults to `OnLowResults { threshold: 10 }`.
    #[serde(default)]
    pub strategy: SpellCorrectStrategy,
}

impl Default for SpellCorrectedSearchConfig {
    fn default() -> Self {
        SpellCorrectedSearchConfig {
            enabled: true,
            corrector_config: CorrectorConfig::default(),
            retry_with_correction: true,
            show_did_you_mean: true,
            min_results_for_suggestions: 2,
            strategy: SpellCorrectStrategy::default(),
        }
    }
}

/// A search engine wrapper that provides spell correction capabilities.
pub struct SpellCorrectedSearchEngine {
    /// The underlying search engine.
    engine: LexicalStore,
    /// The spelling corrector.
    corrector: SpellingCorrector,
    /// Configuration for spell-corrected search.
    config: SpellCorrectedSearchConfig,
    /// "Did you mean?" functionality.
    did_you_mean: DidYouMean,
}

impl SpellCorrectedSearchEngine {
    /// Create a new spell-corrected search engine.
    pub fn new(engine: LexicalStore) -> Self {
        let corrector = SpellingCorrector::new();
        let config = SpellCorrectedSearchConfig::default();
        let did_you_mean = DidYouMean::new(SpellingCorrector::new());

        SpellCorrectedSearchEngine {
            engine,
            corrector,
            config,
            did_you_mean,
        }
    }

    /// Create a new spell-corrected search engine with custom configuration.
    pub fn with_config(engine: LexicalStore, config: SpellCorrectedSearchConfig) -> Self {
        let mut corrector = SpellingCorrector::new();
        corrector.set_config(config.corrector_config.clone());
        let did_you_mean = DidYouMean::new(SpellingCorrector::new());

        SpellCorrectedSearchEngine {
            engine,
            corrector,
            config,
            did_you_mean,
        }
    }

    /// Get the underlying search engine.
    pub fn engine(&self) -> &LexicalStore {
        &self.engine
    }

    /// Get mutable access to the underlying search engine.
    pub fn engine_mut(&mut self) -> &mut LexicalStore {
        &mut self.engine
    }

    /// Update the spell correction configuration.
    pub fn set_spell_config(&mut self, config: SpellCorrectedSearchConfig) {
        self.corrector.set_config(config.corrector_config.clone());
        self.config = config;
    }

    /// Search with spell correction for a query string.
    pub fn search_with_correction(
        &mut self,
        query_str: &str,
        default_field: &str,
    ) -> Result<SpellCorrectedSearchResults> {
        use crate::lexical::query::parser::LexicalQueryParser;

        // Get analyzer from engine
        let analyzer = self.engine.analyzer()?;

        if !self.config.enabled {
            // Spell correction disabled, perform normal search
            let parser = LexicalQueryParser::new(analyzer).with_default_field(default_field);
            let query = parser.parse(query_str)?;
            let results = self.engine.search(LexicalSearchRequest::new(query))?;
            let correction = CorrectionResult::new(query_str.to_string());
            return Ok(SpellCorrectedSearchResults::new(results, correction));
        }

        // Run the original query **first** so the strategy gate
        // below can decide whether the corrector is even worth
        // invoking (#411). The corrector walks the term dictionary on
        // every call, which is wasted work whenever the original
        // query already returns enough hits.
        let parser = LexicalQueryParser::new(analyzer).with_default_field(default_field);
        let query = parser.parse(query_str)?;
        let original_results = self.engine.search(LexicalSearchRequest::new(query))?;

        let should_correct = self.should_invoke_corrector(original_results.total_hits);
        let correction = if should_correct {
            self.corrector.correct(query_str)
        } else {
            CorrectionResult::new(query_str.to_string())
        };

        // Decide whether to use correction
        let should_use_correction = self.should_use_correction(&original_results, &correction);

        let (final_results, used_correction) = if should_use_correction {
            // Use the corrected query
            let corrected_query = correction.query();
            let query = parser.parse(corrected_query)?;
            let corrected_results = self.engine.search(LexicalSearchRequest::new(query))?;
            (corrected_results, true)
        } else {
            (original_results, false)
        };

        // Generate "Did you mean?" suggestion if appropriate, but only
        // when we actually ran the corrector — otherwise the engine
        // never produced suggestions to surface.
        let did_you_mean = if should_correct && self.config.show_did_you_mean && !used_correction {
            self.did_you_mean.suggest(query_str)
        } else {
            None
        };

        let mut spell_results = SpellCorrectedSearchResults::new(final_results, correction);
        spell_results.used_correction = used_correction;
        spell_results.did_you_mean = did_you_mean;

        Ok(spell_results)
    }

    /// Search with spell correction for a field-specific query.
    pub fn search_field_with_correction(
        &mut self,
        field: &str,
        query_str: &str,
    ) -> Result<SpellCorrectedSearchResults> {
        use crate::lexical::query::parser::LexicalQueryParser;

        // Get analyzer from engine
        let analyzer = self.engine.analyzer()?;

        if !self.config.enabled {
            // Spell correction disabled, perform normal search
            let parser = LexicalQueryParser::new(analyzer);
            let query = parser.parse_field(field, query_str)?;
            let results = self.engine.search(LexicalSearchRequest::new(query))?;
            let correction = CorrectionResult::new(query_str.to_string());
            return Ok(SpellCorrectedSearchResults::new(results, correction));
        }

        // Run the original query first; gate the corrector on the
        // result count (see `search_with_correction` for the full
        // rationale, #411).
        let parser = LexicalQueryParser::new(analyzer);
        let query = parser.parse_field(field, query_str)?;
        let original_results = self.engine.search(LexicalSearchRequest::new(query))?;

        let should_correct = self.should_invoke_corrector(original_results.total_hits);
        let correction = if should_correct {
            self.corrector.correct(query_str)
        } else {
            CorrectionResult::new(query_str.to_string())
        };

        // Decide whether to use correction
        let should_use_correction = self.should_use_correction(&original_results, &correction);

        let (final_results, used_correction) = if should_use_correction {
            // Use the corrected query
            let corrected_query = correction.query();
            let query = parser.parse_field(field, corrected_query)?;
            let corrected_results = self.engine.search(LexicalSearchRequest::new(query))?;
            (corrected_results, true)
        } else {
            (original_results, false)
        };

        // Generate "Did you mean?" suggestion if appropriate.
        let did_you_mean = if should_correct && self.config.show_did_you_mean && !used_correction {
            self.did_you_mean.suggest(query_str)
        } else {
            None
        };

        let mut spell_results = SpellCorrectedSearchResults::new(final_results, correction);
        spell_results.used_correction = used_correction;
        spell_results.did_you_mean = did_you_mean;

        Ok(spell_results)
    }

    /// Decide whether to invoke the (expensive) spelling corrector
    /// based on the configured [`SpellCorrectStrategy`] and the
    /// original query's hit count (#411).
    fn should_invoke_corrector(&self, original_total_hits: u64) -> bool {
        match self.config.strategy {
            SpellCorrectStrategy::Always => true,
            SpellCorrectStrategy::Never => false,
            SpellCorrectStrategy::OnLowResults { threshold } => original_total_hits < threshold,
        }
    }

    /// Check if a word is correctly spelled.
    pub fn is_word_correct(&self, word: &str) -> bool {
        self.corrector.is_correct(word)
    }

    /// Get spelling suggestions for a word.
    pub fn suggest_word(&self, word: &str) -> Vec<crate::spelling::suggest::Suggestion> {
        self.corrector.suggest_word(word)
    }

    /// Learn from the index terms to improve spelling correction.
    ///
    /// This method extracts terms from the search index and adds them to the
    /// spelling correction dictionary. This allows the corrector to prioritize
    /// terms that actually exist in the index when making suggestions.
    ///
    /// # Implementation Note
    ///
    /// Currently this is a placeholder. To implement this functionality:
    ///
    /// ```ignore
    /// // 1. Get the index reader
    /// let reader = self.engine.reader()?;
    ///
    /// // 2. Enumerate all terms from all fields
    /// for field in ["title", "content", "tags"] {
    ///     if let Some(terms) = reader.terms(field)? {
    ///         let mut iter = terms.iterator()?;
    ///         let term_pairs = std::iter::from_fn(|| {
    ///             iter.next().ok().flatten().map(|stats| {
    ///                 (stats.term, stats.doc_freq as u32)
    ///             })
    ///         });
    ///         self.corrector.learn_from_terms(term_pairs)?;
    ///     }
    /// }
    /// ```
    pub fn learn_from_index(&mut self) -> Result<()> {
        // TODO: Implement term enumeration from the index
        // This requires access to LexicalIndexReader.terms() API which returns
        // an iterator over all terms in a field.
        Ok(())
    }

    /// Get statistics about the spelling corrector.
    pub fn corrector_stats(&self) -> crate::spelling::corrector::CorrectorStats {
        self.corrector.stats()
    }

    /// Clear the query learning history.
    pub fn clear_query_history(&mut self) {
        self.corrector.clear_query_history();
    }

    /// Decide whether to use the spelling correction based on search results and correction quality.
    fn should_use_correction(
        &self,
        original_results: &LexicalSearchResults,
        correction: &CorrectionResult,
    ) -> bool {
        // Don't use correction if auto-correction is disabled and no corrections suggested
        if !correction.has_suggestions() {
            return false;
        }

        // If auto-correction was applied, use it
        if correction.auto_corrected {
            return true;
        }

        // If retry with correction is enabled and original query has poor results
        if self.config.retry_with_correction {
            // Use correction if original query has few results and correction confidence is high
            let has_few_results =
                original_results.total_hits < self.config.min_results_for_suggestions as u64;
            let high_confidence = correction.confidence > 0.7;

            if has_few_results && high_confidence {
                return true;
            }
        }

        false
    }
}

/// Utility functions for spell-corrected search.
pub struct SpellSearchUtils;

impl SpellSearchUtils {
    /// Extract search terms from a query string for spell checking.
    pub fn extract_search_terms(query_str: &str) -> Vec<String> {
        // Common stop words to filter out
        let stop_words = [
            "and", "or", "not", "the", "is", "a", "an", "in", "on", "at", "to", "for", "of",
            "with", "by",
        ];

        // Simple extraction - split on common query operators and whitespace
        query_str
            .split(&[':', '(', ')', '"', '+', '-', ' ', '\t', '\n'][..])
            .filter_map(|term| {
                let cleaned = term.trim().to_lowercase();
                if cleaned.len() > 2
                    && cleaned.chars().all(|c| c.is_alphabetic())
                    && !stop_words.contains(&cleaned.as_str())
                {
                    Some(cleaned)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Create a corrected query string from the original and correction results.
    pub fn build_corrected_query(original: &str, correction: &CorrectionResult) -> String {
        if let Some(corrected) = &correction.corrected {
            corrected.clone()
        } else {
            // Build a partially corrected query
            let mut result = original.to_string();

            for (original_word, suggestions) in &correction.word_suggestions {
                if let Some(best_suggestion) = suggestions.first()
                    && best_suggestion.score > 0.6
                {
                    result = result.replace(original_word, &best_suggestion.word);
                }
            }

            result
        }
    }

    /// Format "Did you mean?" suggestion for display.
    pub fn format_did_you_mean(_original: &str, suggestion: &str) -> String {
        format!("Did you mean: \"{suggestion}\"?")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::store::config::LexicalIndexConfig;
    use crate::storage::file::{FileStorage, FileStorageConfig};
    use std::sync::Arc;
    use tempfile::TempDir;

    /// `should_invoke_corrector` must honour each [`SpellCorrectStrategy`]
    /// (#411). Validated without spinning up a real engine — the
    /// helper depends only on `self.config.strategy` and the input
    /// `total_hits` value, so we synthesise a wrapper directly.
    #[test]
    fn strategy_gates_corrector_invocation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let mut wrapper = SpellCorrectedSearchEngine::new(engine);

        wrapper.config.strategy = SpellCorrectStrategy::Always;
        assert!(wrapper.should_invoke_corrector(0));
        assert!(wrapper.should_invoke_corrector(1_000_000));

        wrapper.config.strategy = SpellCorrectStrategy::Never;
        assert!(!wrapper.should_invoke_corrector(0));
        assert!(!wrapper.should_invoke_corrector(1_000_000));

        wrapper.config.strategy = SpellCorrectStrategy::OnLowResults { threshold: 10 };
        assert!(wrapper.should_invoke_corrector(0)); // below threshold
        assert!(wrapper.should_invoke_corrector(9));
        assert!(!wrapper.should_invoke_corrector(10)); // at threshold (exclusive)
        assert!(!wrapper.should_invoke_corrector(50));
    }

    #[test]
    fn default_strategy_is_on_low_results_threshold_10() {
        // Documents the chosen default for the audit's
        // "default strategy decided in the PR review" criterion.
        let cfg = SpellCorrectedSearchConfig::default();
        assert_eq!(
            cfg.strategy,
            SpellCorrectStrategy::OnLowResults { threshold: 10 }
        );
    }

    #[allow(dead_code)]
    #[test]
    fn test_spell_corrected_search_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();
        let spell_engine = SpellCorrectedSearchEngine::new(engine);

        assert!(spell_engine.config.enabled);
        assert!(spell_engine.config.retry_with_correction);
    }

    #[test]
    fn test_spell_corrected_search_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let engine_config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, engine_config).unwrap();

        let spell_config = SpellCorrectedSearchConfig {
            enabled: false,
            ..Default::default()
        };

        let mut spell_engine = SpellCorrectedSearchEngine::with_config(engine, spell_config);

        let results = spell_engine
            .search_with_correction("hello world", "title")
            .unwrap();

        assert!(!results.has_suggestions());
        assert!(!results.used_correction);
        assert_eq!(results.effective_query(), "hello world");
    }

    #[test]
    fn test_spell_corrected_search_with_typos() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();
        let mut spell_engine = SpellCorrectedSearchEngine::new(engine);

        // Test with a query that might have typos
        let results = spell_engine
            .search_with_correction("helo wrld", "title")
            .unwrap();

        // Should have some correction information
        assert_eq!(results.correction.original, "helo wrld");
        // Exact behavior depends on the dictionary and suggestion quality
    }

    #[test]
    fn test_word_correction_check() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();
        let spell_engine = SpellCorrectedSearchEngine::new(engine);

        // Test with common words
        assert!(spell_engine.is_word_correct("hello")); // Should be in built-in dictionary
        assert!(spell_engine.is_word_correct("the")); // Should be in built-in dictionary

        // Test with likely typos
        let suggestions = spell_engine.suggest_word("helo");
        // Should get some suggestions (exact results depend on dictionary)
        assert!(!suggestions.is_empty() || !spell_engine.is_word_correct("hello"));
    }

    #[test]
    fn test_spell_search_utils() {
        let terms = SpellSearchUtils::extract_search_terms("title:hello AND body:world");
        assert!(terms.contains(&"title".to_string()));
        assert!(terms.contains(&"hello".to_string()));
        assert!(terms.contains(&"body".to_string()));
        assert!(terms.contains(&"world".to_string()));
        assert!(!terms.contains(&"and".to_string())); // Should be filtered out

        let corrected = SpellSearchUtils::build_corrected_query(
            "original query",
            &CorrectionResult::new("original query".to_string()),
        );
        assert_eq!(corrected, "original query");

        let did_you_mean = SpellSearchUtils::format_did_you_mean("helo", "hello");
        assert_eq!(did_you_mean, "Did you mean: \"hello\"?");
    }

    #[test]
    fn test_spell_corrected_results() {
        use crate::lexical::query::LexicalSearchResults;

        let results = LexicalSearchResults {
            hits: vec![],
            total_hits: 0,
            max_score: 0.0,
            // search_time field doesn't exist in SearchResults
        };

        let correction = CorrectionResult::new("test query".to_string());
        let spell_results = SpellCorrectedSearchResults::new(results, correction);

        assert_eq!(spell_results.effective_query(), "test query");
        assert!(!spell_results.has_suggestions());
        assert!(!spell_results.was_auto_corrected());
        assert!(!spell_results.used_correction);
        assert_eq!(spell_results.correction_confidence(), 1.0);
    }

    #[test]
    fn test_corrector_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();
        let spell_engine = SpellCorrectedSearchEngine::new(engine);

        let stats = spell_engine.corrector_stats();
        assert!(stats.dictionary_words > 0);
        assert!(stats.dictionary_total_frequency > 0);
        assert_eq!(stats.queries_learned, 0); // Initially empty
    }
}
