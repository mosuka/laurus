//! Main spelling corrector that integrates all spelling correction functionality.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::spelling::dictionary::{BuiltinDictionary, SpellingDictionary};
use crate::spelling::suggest::{Suggestion, SuggestionConfig, SuggestionEngine};

/// Configuration for the spelling corrector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectorConfig {
    /// Maximum edit distance for suggestions.
    pub max_distance: usize,
    /// Maximum number of suggestions to return.
    pub max_suggestions: usize,
    /// Minimum frequency threshold for suggestions.
    pub min_frequency: u32,
    /// Whether to enable automatic correction.
    pub auto_correct: bool,
    /// Threshold for automatic correction (0.0 to 1.0).
    pub auto_correct_threshold: f64,
    /// Whether to use index terms for dictionary.
    pub use_index_terms: bool,
    /// Whether to learn from user queries.
    pub learn_from_queries: bool,
}

impl Default for CorrectorConfig {
    fn default() -> Self {
        CorrectorConfig {
            max_distance: 2,
            max_suggestions: 5,
            min_frequency: 1,
            auto_correct: false,
            auto_correct_threshold: 0.8,
            use_index_terms: true,
            learn_from_queries: true,
        }
    }
}

/// Result of spelling correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionResult {
    /// Original query.
    pub original: String,
    /// Corrected query (if auto-correction was applied).
    pub corrected: Option<String>,
    /// Suggestions for each word.
    pub word_suggestions: HashMap<String, Vec<Suggestion>>,
    /// Overall confidence score.
    pub confidence: f64,
    /// Whether correction was applied automatically.
    pub auto_corrected: bool,
}

impl CorrectionResult {
    /// Create a new correction result.
    pub fn new(original: String) -> Self {
        CorrectionResult {
            original,
            corrected: None,
            word_suggestions: HashMap::new(),
            confidence: 1.0,
            auto_corrected: false,
        }
    }

    /// Check if any corrections were suggested.
    pub fn has_suggestions(&self) -> bool {
        !self.word_suggestions.is_empty()
    }

    /// Get the best suggestion for a specific word.
    pub fn best_suggestion(&self, word: &str) -> Option<&Suggestion> {
        self.word_suggestions.get(word)?.first()
    }

    /// Get the corrected query or original if no correction.
    pub fn query(&self) -> &str {
        self.corrected.as_ref().unwrap_or(&self.original)
    }

    /// Check if the result suggests a "Did you mean?" prompt.
    ///
    /// Returns `true` when all of the following conditions hold:
    /// - At least one word has suggestions ([`has_suggestions`](Self::has_suggestions)).
    /// - The result was **not** auto-corrected (`auto_corrected == false`).
    /// - The overall `confidence` score is below `0.7`.
    pub fn should_show_did_you_mean(&self) -> bool {
        self.has_suggestions() && !self.auto_corrected && self.confidence < 0.7
    }
}

/// Main spelling corrector that integrates dictionary lookup and suggestion generation.
///
/// `SpellingCorrector` combines a [`SuggestionEngine`] (backed by a [`SpellingDictionary`])
/// with configurable correction policies to provide query-level spelling correction.
///
/// # Correction algorithm
///
/// 1. The input query is split into individual words via
///    `extract_query_words`, which:
///    - Strips non-alphabetic characters from each token.
///    - Converts tokens to lowercase.
///    - Discards tokens shorter than 2 characters.
///    - Removes common English stop words (e.g. "the", "and", "or").
/// 2. Each remaining word is looked up in the underlying dictionary via the
///    suggestion engine.
/// 3. Words not found in the dictionary are considered potential misspellings.
///    The engine generates candidate corrections by computing edit-distance neighbours
///    (insertions, deletions, substitutions, transpositions) up to the configured
///    maximum edit distance and scoring them by a combination of edit distance and
///    dictionary frequency.
/// 4. If the [`CorrectorConfig::auto_correct`] flag is enabled and the best suggestion
///    exceeds [`CorrectorConfig::auto_correct_threshold`], the corrected query is
///    returned automatically.
/// 5. The corrector can optionally learn from indexed terms and past queries to
///    improve future suggestions.
pub struct SpellingCorrector {
    engine: SuggestionEngine,
    config: CorrectorConfig,
    query_history: HashMap<String, u32>,
}

impl SpellingCorrector {
    /// Create a new spelling corrector with built-in dictionary.
    pub fn new() -> Self {
        let dictionary = BuiltinDictionary::english();
        let config = CorrectorConfig::default();
        let suggestion_config = SuggestionConfig {
            max_distance: config.max_distance,
            max_suggestions: config.max_suggestions,
            min_frequency: config.min_frequency,
            ..Default::default()
        };
        let engine = SuggestionEngine::with_config(dictionary, suggestion_config);

        SpellingCorrector {
            engine,
            config,
            query_history: HashMap::new(),
        }
    }

    /// Create a new spelling corrector with custom dictionary.
    pub fn with_dictionary(dictionary: SpellingDictionary) -> Self {
        let config = CorrectorConfig::default();
        let suggestion_config = SuggestionConfig {
            max_distance: config.max_distance,
            max_suggestions: config.max_suggestions,
            min_frequency: config.min_frequency,
            ..Default::default()
        };
        let engine = SuggestionEngine::with_config(dictionary, suggestion_config);

        SpellingCorrector {
            engine,
            config,
            query_history: HashMap::new(),
        }
    }

    /// Create a new spelling corrector with custom configuration.
    pub fn with_config(dictionary: SpellingDictionary, config: CorrectorConfig) -> Self {
        let suggestion_config = SuggestionConfig {
            max_distance: config.max_distance,
            max_suggestions: config.max_suggestions,
            min_frequency: config.min_frequency,
            ..Default::default()
        };
        let engine = SuggestionEngine::with_config(dictionary, suggestion_config);

        SpellingCorrector {
            engine,
            config,
            query_history: HashMap::new(),
        }
    }

    /// Update the corrector configuration.
    pub fn set_config(&mut self, config: CorrectorConfig) {
        let suggestion_config = SuggestionConfig {
            max_distance: config.max_distance,
            max_suggestions: config.max_suggestions,
            min_frequency: config.min_frequency,
            ..Default::default()
        };
        self.engine.set_config(suggestion_config);
        self.config = config;
    }

    /// Correct a query string.
    pub fn correct(&mut self, query: &str) -> CorrectionResult {
        let mut result = CorrectionResult::new(query.to_string());

        // Learn from this query if enabled
        if self.config.learn_from_queries {
            self.learn_query(query);
        }

        // Extract words from the query
        let words = self.extract_query_words(query);
        let mut corrected_words = Vec::new();
        let mut total_confidence = 0.0;
        let mut corrections_made = 0;

        for word in &words {
            if self.engine.is_correct(word) {
                corrected_words.push(word.clone());
                total_confidence += 1.0;
            } else {
                let suggestions = self.engine.suggest(word);

                if !suggestions.is_empty() {
                    result
                        .word_suggestions
                        .insert(word.clone(), suggestions.clone());

                    let best_suggestion = &suggestions[0];
                    total_confidence += best_suggestion.score;

                    // Apply auto-correction if enabled and confidence is high enough
                    if self.config.auto_correct
                        && best_suggestion.score >= self.config.auto_correct_threshold
                    {
                        corrected_words.push(best_suggestion.word.clone());
                        corrections_made += 1;
                        result.auto_corrected = true;
                    } else {
                        corrected_words.push(word.clone());
                    }
                } else {
                    corrected_words.push(word.clone());
                    total_confidence += 0.5; // Unknown word gets neutral score
                }
            }
        }

        // Calculate overall confidence
        result.confidence = if words.is_empty() {
            1.0
        } else {
            total_confidence / words.len() as f64
        };

        // Create corrected query if any corrections were made
        if corrections_made > 0 {
            result.corrected = Some(corrected_words.join(" "));
        }

        result
    }

    /// Get suggestions for a single word.
    pub fn suggest_word(&self, word: &str) -> Vec<Suggestion> {
        self.engine.suggest(word)
    }

    /// Check if a word is correctly spelled.
    pub fn is_correct(&self, word: &str) -> bool {
        self.engine.is_correct(word)
    }

    /// Add terms to the dictionary from an iterator.
    ///
    /// This method allows learning from any source of terms (index, corpus, etc.)
    /// by accepting an iterator of (term, frequency) pairs.
    ///
    /// # Arguments
    ///
    /// * `terms` - Iterator yielding (term, frequency) pairs
    ///
    /// # Example
    ///
    /// ```ignore
    /// let terms = vec![
    ///     ("hello".to_string(), 100),
    ///     ("world".to_string(), 50),
    /// ];
    /// corrector.learn_from_terms(terms.into_iter())?;
    /// ```
    pub fn learn_from_terms<I>(&mut self, terms: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, u32)>,
    {
        if !self.config.use_index_terms {
            return Ok(());
        }

        for (term, frequency) in terms {
            // Only learn terms that look valid (alphabetic, reasonable length)
            if term.len() >= 2 && term.chars().all(|c| c.is_alphabetic() || c == '-') {
                self.engine.add_word(&term, frequency);
            }
        }

        Ok(())
    }

    /// Learn from a user query.
    fn learn_query(&mut self, query: &str) {
        let words = self.extract_query_words(query);

        for word in words {
            // Only learn words that look like they could be correct
            if word.len() >= 3 && word.chars().all(|c| c.is_alphabetic()) {
                let entry = self.query_history.entry(word.clone()).or_insert(0);
                *entry = entry.saturating_add(1);
                let count = *entry;

                // If we've seen this word enough times, consider it correct
                if count >= 5 {
                    self.engine.add_word(&word, count);
                }
            }
        }
    }

    /// Extract words from a query string.
    fn extract_query_words(&self, query: &str) -> Vec<String> {
        // Common stop words to filter out
        let stop_words = [
            "is", "a", "an", "the", "and", "or", "not", "in", "on", "at", "to", "for", "of",
            "with", "by",
        ];

        query
            .split_whitespace()
            .filter_map(|word| {
                // Clean up the word (remove punctuation, etc.)
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
                    .to_lowercase();

                if cleaned.len() >= 2 && !stop_words.contains(&cleaned.as_str()) {
                    Some(cleaned)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get statistics about the corrector.
    pub fn stats(&self) -> CorrectorStats {
        let (dict_words, dict_frequency) = self.engine.dictionary_stats();

        CorrectorStats {
            dictionary_words: dict_words,
            dictionary_total_frequency: dict_frequency,
            queries_learned: self.query_history.len(),
            total_query_frequency: self.query_history.values().sum(),
        }
    }

    /// Clear the query learning history.
    pub fn clear_query_history(&mut self) {
        self.query_history.clear();
    }
}

impl Default for SpellingCorrector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the spelling corrector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectorStats {
    /// Number of words in the dictionary.
    pub dictionary_words: usize,
    /// Total frequency count in dictionary.
    pub dictionary_total_frequency: u64,
    /// Number of unique queries learned.
    pub queries_learned: usize,
    /// Total frequency of learned queries.
    pub total_query_frequency: u32,
}

/// "Did you mean?" feature implementation.
pub struct DidYouMean {
    corrector: SpellingCorrector,
}

impl DidYouMean {
    /// Create a new "Did you mean?" instance.
    pub fn new(corrector: SpellingCorrector) -> Self {
        DidYouMean { corrector }
    }

    /// Generate a "Did you mean?" suggestion for a query.
    pub fn suggest(&mut self, query: &str) -> Option<String> {
        let result = self.corrector.correct(query);

        if result.should_show_did_you_mean() {
            // Generate the best possible correction
            let words = self.corrector.extract_query_words(query);
            let mut corrected_words = Vec::new();
            let mut made_corrections = false;

            for word in words {
                if let Some(suggestions) = result.word_suggestions.get(&word) {
                    if let Some(best) = suggestions.first() {
                        if best.score > 0.5 {
                            corrected_words.push(best.word.clone());
                            made_corrections = true;
                        } else {
                            corrected_words.push(word);
                        }
                    } else {
                        corrected_words.push(word);
                    }
                } else {
                    corrected_words.push(word);
                }
            }

            if made_corrections {
                Some(corrected_words.join(" "))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if a "Did you mean?" suggestion should be shown.
    pub fn should_suggest(&mut self, query: &str) -> bool {
        let result = self.corrector.correct(query);
        result.should_show_did_you_mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spelling::dictionary::BuiltinDictionary;

    #[test]
    fn test_corrector_creation() {
        let corrector = SpellingCorrector::new();
        let stats = corrector.stats();
        assert!(stats.dictionary_words > 0);
    }

    #[test]
    fn test_corrector_with_custom_dictionary() {
        let dict = BuiltinDictionary::minimal();
        let corrector = SpellingCorrector::with_dictionary(dict);

        assert!(corrector.is_correct("hello"));
        assert!(corrector.is_correct("search"));
    }

    #[test]
    fn test_word_extraction() {
        let corrector = SpellingCorrector::new();
        let words = corrector.extract_query_words("Hello, world! This is a test.");

        assert!(words.contains(&"hello".to_string()));
        assert!(words.contains(&"world".to_string()));
        assert!(words.contains(&"test".to_string()));
        // "is" and "a" should be filtered out as too short
        assert!(!words.contains(&"is".to_string()));
    }

    #[test]
    fn test_correction_result() {
        let dict = BuiltinDictionary::minimal();
        let mut corrector = SpellingCorrector::with_dictionary(dict);

        // Test correct query
        let result = corrector.correct("hello world");
        assert!(!result.has_suggestions());
        assert_eq!(result.query(), "hello world");
        assert!(!result.should_show_did_you_mean());

        // Test query with typos
        let result = corrector.correct("helo wrld");
        // Should find some suggestions (depending on dictionary)
        assert_eq!(result.query(), "helo wrld"); // No auto-correction by default
    }

    #[test]
    fn test_auto_correction() {
        let dict = BuiltinDictionary::minimal();
        let config = CorrectorConfig {
            auto_correct: true,
            auto_correct_threshold: 0.5, // Low threshold for testing
            ..Default::default()
        };
        let mut corrector = SpellingCorrector::with_config(dict, config);

        let _result = corrector.correct("helo");
        // Might auto-correct if "hello" is found with high confidence
        // The exact behavior depends on the dictionary and suggestion quality
    }

    #[test]
    fn test_suggestion_for_single_word() {
        let dict = BuiltinDictionary::minimal();
        let dict_clone = dict.clone();
        let corrector = SpellingCorrector::with_dictionary(dict);

        let suggestions = corrector.suggest_word("helo");
        // Should return suggestions (may include "hello" if in minimal dictionary)
        assert!(!suggestions.is_empty() || !dict_clone.contains("hello")); // Either suggestions or "hello" not in dict
    }

    #[test]
    fn test_did_you_mean() {
        let dict = BuiltinDictionary::minimal();
        let corrector = SpellingCorrector::with_dictionary(dict);
        let mut dym = DidYouMean::new(corrector);

        // Test with correct query
        assert!(!dym.should_suggest("hello world"));
        assert!(dym.suggest("hello world").is_none());

        // Test with potentially incorrect query
        let _should_suggest = dym.should_suggest("helo wrld");
        // The exact behavior depends on the dictionary contents
    }

    #[test]
    fn test_corrector_stats() {
        let corrector = SpellingCorrector::new();
        let stats = corrector.stats();

        assert!(stats.dictionary_words > 0);
        assert!(stats.dictionary_total_frequency > 0);
        assert_eq!(stats.queries_learned, 0); // Initially empty
        assert_eq!(stats.total_query_frequency, 0);
    }

    #[test]
    fn test_config_update() {
        let mut corrector = SpellingCorrector::new();

        let new_config = CorrectorConfig {
            max_suggestions: 10,
            auto_correct: true,
            ..Default::default()
        };

        corrector.set_config(new_config.clone());
        assert_eq!(corrector.config.max_suggestions, 10);
        assert!(corrector.config.auto_correct);
    }

    #[test]
    fn test_learning_queries() {
        let dict = BuiltinDictionary::minimal();
        let config = CorrectorConfig {
            learn_from_queries: true,
            ..Default::default()
        };
        let mut corrector = SpellingCorrector::with_config(dict, config);

        corrector.correct("hello world test");
        corrector.correct("hello programming test");

        let stats = corrector.stats();
        assert!(stats.queries_learned > 0);
        assert!(stats.total_query_frequency > 0);
    }
}
