//! Stop filter implementation.
//!
//! This module provides a filter that removes common words (stop words) that
//! typically don't contribute to search relevance. Includes default stop word
//! lists for English and Japanese, with support for custom word lists.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::token_filter::Filter;
//! use laurus::analysis::token_filter::stop::StopFilter;
//! use laurus::analysis::token::Token;
//!
//! let filter = StopFilter::new(); // Uses default English stop words
//! let tokens = vec![
//!     Token::new("the", 0),
//!     Token::new("quick", 1),
//!     Token::new("brown", 2)
//! ];
//!
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! // "the" is removed as a stop word
//! assert_eq!(result.len(), 2);
//! assert_eq!(result[0].text, "quick");
//! assert_eq!(result[1].text, "brown");
//! ```

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use crate::analysis::token::{Token, TokenStream, remove_tokens};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// Default English stop words list.
///
/// Common English words that are typically filtered out during indexing.
const DEFAULT_ENGLISH_STOP_WORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

const DEFAULT_JAPANESE_STOP_WORDS: &[&str] = &[
    "の",
    "に",
    "は",
    "を",
    "た",
    "が",
    "で",
    "て",
    "と",
    "し",
    "れ",
    "さ",
    "ある",
    "いる",
    "も",
    "する",
    "から",
    "な",
    "こと",
    "として",
    "い",
    "や",
    "れる",
    "など",
    "なっ",
    "ない",
    "この",
    "ため",
    "その",
    "あっ",
    "よう",
    "また",
    "もの",
    "という",
    "あり",
    "まで",
    "られ",
    "なる",
    "へ",
    "か",
    "だ",
    "これ",
    "によって",
    "により",
    "おり",
    "より",
    "による",
    "ず",
    "なり",
    "られる",
    "において",
    "ば",
    "なかっ",
    "なく",
    "しかし",
    "について",
    "せ",
    "だっ",
    "その後",
    "できる",
    "それ",
    "う",
    "ので",
    "なお",
    "のみ",
    "でき",
    "き",
    "つ",
    "における",
    "および",
    "いう",
    "さらに",
    "でも",
    "ら",
    "たり",
    "その他",
    "に関する",
    "たち",
    "ます",
    "ん",
    "なら",
    "に対して",
    "特に",
    "せる",
    "及び",
    "これら",
    "とき",
    "では",
    "にて",
    "ほか",
    "ながら",
    "うち",
    "そして",
    "とともに",
    "ただし",
    "かつて",
    "それぞれ",
    "または",
    "お",
    "ほど",
    "ものの",
    "に対する",
    "ほとんど",
    "と共に",
    "といった",
    "です",
    "とも",
    "ところ",
    "ここ",
];

/// Default English stop words as a HashSet.
pub static DEFAULT_ENGLISH_STOP_WORDS_SET: LazyLock<HashSet<String>> = LazyLock::new(|| {
    DEFAULT_ENGLISH_STOP_WORDS
        .iter()
        .map(|&s| s.to_string())
        .collect()
});

/// Default Japanese stop words as a HashSet.
pub static DEFAULT_JAPANESE_STOP_WORDS_SET: LazyLock<HashSet<String>> = LazyLock::new(|| {
    DEFAULT_JAPANESE_STOP_WORDS
        .iter()
        .map(|&s| s.to_string())
        .collect()
});

/// A filter that removes stop words from the token stream.
///
/// Stop words are common words (like "the", "is", "at") that are often
/// filtered out during text analysis because they typically don't contribute
/// to search relevance. This filter can either remove stop words entirely
/// or mark them as stopped while keeping them in the stream.
///
/// A removed stop word leaves no position. After [`SynonymGraphFilter`],
/// removing one keeps the token graph well formed: a member of a synonym
/// group loses the word and stays a path of its own, and a synonym stacked
/// on the word keeps its position.
///
/// [`SynonymGraphFilter`]: crate::analysis::token_filter::synonym_graph::SynonymGraphFilter
///
/// # Default Stop Word Lists
///
/// - English: 33 common words (articles, prepositions, conjunctions)
/// - Japanese: 127 common particles and auxiliary verbs
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use laurus::analysis::token_filter::Filter;
/// use laurus::analysis::token_filter::stop::StopFilter;
/// use laurus::analysis::token::Token;
///
/// let filter = StopFilter::new();
/// let tokens = vec![
///     Token::new("this", 0),
///     Token::new("is", 1),
///     Token::new("test", 2)
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// // Only "test" remains
/// assert_eq!(result.len(), 1);
/// assert_eq!(result[0].text, "test");
/// ```
///
/// ## Custom Stop Words
///
/// ```
/// use laurus::analysis::token_filter::stop::StopFilter;
///
/// let filter = StopFilter::from_words(vec!["custom", "words", "list"]);
/// ```
///
/// ## Preserve Stopped Tokens
///
/// ```
/// use laurus::analysis::token_filter::Filter;
/// use laurus::analysis::token_filter::stop::StopFilter;
/// use laurus::analysis::token::Token;
///
/// // Mark as stopped but don't remove
/// let filter = StopFilter::from_words(vec!["the"]).remove_stopped(false);
/// let tokens = vec![Token::new("the", 0), Token::new("quick", 1)];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result.len(), 2);
/// assert!(result[0].is_stopped());  // Marked as stopped
/// assert!(!result[1].is_stopped());
/// ```
#[derive(Clone, Debug)]
pub struct StopFilter {
    /// The set of stop words to remove
    stop_words: Arc<HashSet<String>>,
    /// Whether to remove stopped tokens entirely or just mark them as stopped
    remove_stopped: bool,
}

impl StopFilter {
    /// Create a new stop filter with the default English stop words.
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::token_filter::stop::StopFilter;
    ///
    /// let filter = StopFilter::new();
    /// assert!(filter.is_stop_word("the"));
    /// assert!(!filter.is_stop_word("hello"));
    /// ```
    pub fn new() -> Self {
        Self::with_stop_words(DEFAULT_ENGLISH_STOP_WORDS_SET.clone())
    }

    /// Create a new stop filter with custom stop words.
    ///
    /// # Arguments
    ///
    /// * `stop_words` - A set of words to filter out
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use laurus::analysis::token_filter::stop::StopFilter;
    ///
    /// let mut words = HashSet::new();
    /// words.insert("custom".to_string());
    /// words.insert("stop".to_string());
    ///
    /// let filter = StopFilter::with_stop_words(words);
    /// assert!(filter.is_stop_word("custom"));
    /// ```
    pub fn with_stop_words(stop_words: HashSet<String>) -> Self {
        StopFilter {
            stop_words: Arc::new(stop_words),
            remove_stopped: true,
        }
    }

    /// Create a new stop filter from a list of stop words.
    ///
    /// # Arguments
    ///
    /// * `words` - An iterator of words to filter out
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::token_filter::stop::StopFilter;
    ///
    /// let filter = StopFilter::from_words(vec!["foo", "bar", "baz"]);
    /// assert_eq!(filter.len(), 3);
    /// ```
    pub fn from_words<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stop_words = words.into_iter().map(|s| s.into()).collect();
        Self::with_stop_words(stop_words)
    }

    /// Set whether to remove stopped tokens entirely or just mark them as stopped.
    ///
    /// # Arguments
    ///
    /// * `remove` - If `true`, remove stopped tokens; if `false`, mark them as stopped
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::token_filter::stop::StopFilter;
    ///
    /// // Keep stopped tokens but mark them
    /// let filter = StopFilter::new().remove_stopped(false);
    /// ```
    pub fn remove_stopped(mut self, remove: bool) -> Self {
        self.remove_stopped = remove;
        self
    }

    /// Check if a word is a stop word.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to check
    ///
    /// # Returns
    ///
    /// `true` if the word is in the stop word set, `false` otherwise
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(word)
    }

    /// Get the number of stop words.
    pub fn len(&self) -> usize {
        self.stop_words.len()
    }

    /// Check if the stop word set is empty.
    pub fn is_empty(&self) -> bool {
        self.stop_words.is_empty()
    }
}

impl Default for StopFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter for StopFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let is_new_stop_word =
            |token: &Token| !token.is_stopped() && self.is_stop_word(&token.text);
        let filtered_tokens: Vec<Token> = if self.remove_stopped {
            remove_tokens(tokens.collect(), |token| !is_new_stop_word(token))
        } else {
            tokens
                .map(|token| {
                    if is_new_stop_word(&token) {
                        token.stop()
                    } else {
                        token
                    }
                })
                .collect()
        };

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "stop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_stop_filter() {
        let filter = StopFilter::from_words(vec!["the", "and", "or"]);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("the", 1),
            Token::new("world", 2),
            Token::new("and", 3),
            Token::new("test", 4),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
        assert_eq!(result[2].text, "test");
    }

    #[test]
    fn test_stop_filter_preserve_stopped() {
        let filter = StopFilter::from_words(vec!["the", "and"]).remove_stopped(false);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("the", 1),
            Token::new("world", 2),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "hello");
        assert!(!result[0].is_stopped());
        assert_eq!(result[1].text, "the");
        assert!(result[1].is_stopped());
        assert_eq!(result[2].text, "world");
        assert!(!result[2].is_stopped());
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(StopFilter::new().name(), "stop");
    }

    fn arc(text: &str, increment: usize, length: usize) -> Token {
        Token::new(text, 0)
            .with_position_increment(increment)
            .with_position_length(length)
    }

    /// Each token as an arc `(text, from, to)` of the token graph.
    fn arcs(tokens: &[Token]) -> Vec<(&str, u32, u32)> {
        tokens
            .iter()
            .zip(crate::analysis::token::token_positions(tokens))
            .map(|(t, from)| (t.text.as_str(), from, from + t.position_length as u32))
            .collect()
    }

    fn stop(words: &[&str], tokens: Vec<Token>) -> Vec<Token> {
        StopFilter::from_words(words.to_vec())
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect()
    }

    /// #1259: "a", stacked on the stop word "the", takes the stop word's
    /// node instead of moving back onto the word before it.
    #[test]
    fn a_synonym_stacked_on_a_stop_word_keeps_its_node() {
        let tokens = vec![
            arc("big", 1, 1),
            arc("the", 1, 1),
            arc("a", 0, 1),
            arc("dog", 1, 1),
        ];
        assert_eq!(
            arcs(&stop(&["the"], tokens)),
            vec![("big", 0, 1), ("a", 1, 2), ("dog", 2, 3)]
        );
    }

    /// #1259: "statue of liberty" with "lady liberty". "of", the only word
    /// on its node, is contracted: its node merges with the next, and
    /// "lady liberty" keeps an inner node of its own.
    #[test]
    fn a_stop_word_inside_a_member_is_contracted() {
        let graph = vec![
            arc("statue", 1, 1),
            arc("lady", 0, 3),
            arc("of", 1, 1),
            arc("liberty", 1, 2),
            arc("liberty", 1, 1),
        ];
        assert_eq!(
            arcs(&stop(&["of"], graph)),
            vec![
                ("statue", 0, 1),
                ("lady", 0, 2),
                ("liberty", 1, 3),
                ("liberty", 2, 3)
            ]
        );
    }

    /// "america" with "the usa": without "the", "usa" leaves the start
    /// node and ends where "america" does.
    #[test]
    fn a_stop_word_starting_a_member_is_contracted() {
        let graph = vec![
            arc("america", 1, 2),
            arc("the", 0, 1),
            arc("usa", 1, 1),
            arc("rocks", 1, 1),
        ];
        assert_eq!(
            arcs(&stop(&["the"], graph)),
            vec![("america", 0, 1), ("usa", 0, 1), ("rocks", 1, 2)]
        );
    }

    /// "liberty" with "statue of": without "of", "statue" ends where
    /// "liberty" does.
    #[test]
    fn a_stop_word_ending_a_member_is_contracted() {
        let graph = vec![
            arc("liberty", 1, 2),
            arc("statue", 0, 1),
            arc("of", 1, 1),
            arc("rocks", 1, 1),
        ];
        assert_eq!(
            arcs(&stop(&["of"], graph)),
            vec![("liberty", 0, 1), ("statue", 0, 1), ("rocks", 1, 2)]
        );
    }

    /// "a ml" with "machine learning": "ml" spans the same nodes as the
    /// other member, so only its arc goes, and "machine learning" stays
    /// after "a".
    #[test]
    fn a_stop_word_member_beside_another_member_is_dropped() {
        let graph = vec![
            arc("a", 1, 1),
            arc("ml", 1, 2),
            arc("machine", 0, 1),
            arc("learning", 1, 1),
        ];
        assert_eq!(
            arcs(&stop(&["ml"], graph)),
            vec![("a", 0, 1), ("machine", 1, 2), ("learning", 2, 3)]
        );
    }
}
