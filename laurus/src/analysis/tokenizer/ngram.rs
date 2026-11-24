//! N-gram tokenizer implementation.
//!
//! This module provides a tokenizer that generates character-level n-grams
//! from input text. N-grams are particularly useful for CJK languages,
//! substring matching, and fuzzy search applications.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::tokenizer::Tokenizer;
//! use laurus::analysis::tokenizer::ngram::NgramTokenizer;
//!
//! // Create a bigram tokenizer (n=2)
//! let tokenizer = NgramTokenizer::bigram();
//! let tokens: Vec<_> = tokenizer.tokenize("hello").unwrap()
//!     .map(|t| t.text.to_string())
//!     .collect();
//! assert_eq!(tokens, vec!["he", "el", "ll", "lo"]);
//! ```

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::{LaurusError, Result};

/// A tokenizer that generates character n-grams.
///
/// Character n-grams are contiguous sequences of n characters from text.
/// This tokenizer supports both fixed-size n-grams (e.g., bigrams) and
/// variable-size ranges (e.g., 2-3 character grams).
///
/// N-grams are useful for:
/// - CJK (Chinese, Japanese, Korean) language processing
/// - Substring matching
/// - Fuzzy search
/// - Spell correction
///
/// # Examples
///
/// ```
/// use laurus::analysis::tokenizer::ngram::NgramTokenizer;
/// use laurus::analysis::tokenizer::Tokenizer;
///
/// // Bigram (n=2)
/// let tokenizer = NgramTokenizer::new(2, 2).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("hello").unwrap()
///     .map(|t| t.text.to_string())
///     .collect();
/// assert_eq!(tokens, vec!["he", "el", "ll", "lo"]);
///
/// // Trigram (n=3)
/// let tokenizer = NgramTokenizer::new(3, 3).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("hello").unwrap()
///     .map(|t| t.text.to_string())
///     .collect();
/// assert_eq!(tokens, vec!["hel", "ell", "llo"]);
///
/// // Variable length (2-3)
/// let tokenizer = NgramTokenizer::new(2, 3).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("abc").unwrap()
///     .map(|t| t.text.to_string())
///     .collect();
/// assert_eq!(tokens, vec!["ab", "abc", "bc"]);
/// ```
#[derive(Clone, Debug)]
pub struct NgramTokenizer {
    /// Minimum n-gram size
    min_gram: usize,
    /// Maximum n-gram size
    max_gram: usize,
}

impl NgramTokenizer {
    /// Create a new n-gram tokenizer.
    ///
    /// # Arguments
    ///
    /// * `min_gram` - Minimum n-gram size (must be >= 1)
    /// * `max_gram` - Maximum n-gram size (must be >= min_gram)
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] if:
    /// - `min_gram` is 0
    /// - `max_gram` is less than `min_gram`
    pub fn new(min_gram: usize, max_gram: usize) -> Result<Self> {
        if min_gram == 0 {
            return Err(LaurusError::invalid_argument(
                "min_gram must be at least 1".to_string(),
            ));
        }
        if max_gram < min_gram {
            return Err(LaurusError::invalid_argument(format!(
                "max_gram ({}) must be >= min_gram ({})",
                max_gram, min_gram
            )));
        }
        Ok(Self { min_gram, max_gram })
    }

    /// Create a bigram tokenizer (n=2).
    ///
    /// Convenience constructor for 2-character n-grams.
    pub fn bigram() -> Self {
        Self {
            min_gram: 2,
            max_gram: 2,
        }
    }

    /// Create a trigram tokenizer (n=3).
    ///
    /// Convenience constructor for 3-character n-grams.
    pub fn trigram() -> Self {
        Self {
            min_gram: 3,
            max_gram: 3,
        }
    }
}

impl Tokenizer for NgramTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::new();
        let mut token_position = 0;

        // Generate n-grams
        for start in 0..chars.len() {
            for gram_size in self.min_gram..=self.max_gram {
                let end = start + gram_size;
                if end > chars.len() {
                    break;
                }

                // Extract n-gram
                let ngram: String = chars[start..end].iter().collect();

                // Calculate byte offsets in the original string
                let start_offset: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
                let end_offset: usize = chars[..end].iter().map(|c| c.len_utf8()).sum();

                tokens.push(Token::with_offsets(
                    &ngram,
                    token_position,
                    start_offset,
                    end_offset,
                ));
                token_position += 1;
            }
        }

        Ok(Box::new(tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "ngram"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_creation() {
        let tokenizer = NgramTokenizer::new(2, 3);
        assert!(tokenizer.is_ok());

        let tokenizer = NgramTokenizer::new(0, 2);
        assert!(tokenizer.is_err());

        let tokenizer = NgramTokenizer::new(3, 2);
        assert!(tokenizer.is_err());
    }

    #[test]
    fn test_bigram() {
        let tokenizer = NgramTokenizer::bigram();
        let tokens: Vec<Token> = tokenizer.tokenize("hello").unwrap().collect();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "he");
        assert_eq!(tokens[1].text, "el");
        assert_eq!(tokens[2].text, "ll");
        assert_eq!(tokens[3].text, "lo");
    }

    #[test]
    fn test_trigram() {
        let tokenizer = NgramTokenizer::trigram();
        let tokens: Vec<Token> = tokenizer.tokenize("hello").unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hel");
        assert_eq!(tokens[1].text, "ell");
        assert_eq!(tokens[2].text, "llo");
    }

    #[test]
    fn test_variable_ngram() {
        let tokenizer = NgramTokenizer::new(2, 3).unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("abc").unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "ab"); // 2-gram from position 0
        assert_eq!(tokens[1].text, "abc"); // 3-gram from position 0
        assert_eq!(tokens[2].text, "bc"); // 2-gram from position 1
    }

    #[test]
    fn test_unicode_support() {
        // Test with Hiragana/Kanji
        let tokenizer = NgramTokenizer::bigram();
        let tokens: Vec<Token> = tokenizer.tokenize("日本語").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "日本");
        assert_eq!(tokens[1].text, "本語");

        // Check byte offsets are correct for multi-byte UTF-8 characters
        // "日" = 3 bytes, "本" = 3 bytes, "語" = 3 bytes in UTF-8
        assert_eq!(tokens[0].start_offset, 0); // "日本" starts at byte 0
        assert_eq!(tokens[0].end_offset, 6); // "日本" ends at byte 6 (0 + 3 + 3)
        assert_eq!(tokens[1].start_offset, 3); // "本語" starts at byte 3 (skipping "日")
        assert_eq!(tokens[1].end_offset, 9); // "本語" ends at byte 9 (3 + 3 + 3)

        // Test with Katakana
        let tokens: Vec<Token> = tokenizer.tokenize("ゴジラ").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "ゴジ");
        assert_eq!(tokens[1].text, "ジラ");

        // Check byte offsets for Katakana (also 3 bytes per character in UTF-8)
        // "ゴ" = 3 bytes, "ジ" = 3 bytes, "ラ" = 3 bytes in UTF-8
        assert_eq!(tokens[0].start_offset, 0); // "ゴジ" starts at byte 0
        assert_eq!(tokens[0].end_offset, 6); // "ゴジ" ends at byte 6 (0 + 3 + 3)
        assert_eq!(tokens[1].start_offset, 3); // "ジラ" starts at byte 3 (skipping "ゴ")
        assert_eq!(tokens[1].end_offset, 9); // "ジラ" ends at byte 9 (3 + 3 + 3)

        // Test with NFKD normalized Katakana (dakuten separated)
        use unicode_normalization::UnicodeNormalization;
        let nfkd_text = "ゴジラ".nfkd().collect::<String>();
        // NFKD: "ゴジラ" → "コ\u{3099}シ\u{3099}ラ" (base chars + combining dakuten U+3099)
        let tokens: Vec<Token> = tokenizer.tokenize(&nfkd_text).unwrap().collect();

        // NFKD produces 5 characters: ['コ', '\u{3099}', 'シ', '\u{3099}', 'ラ']
        // Bigrams: ["コ\u{3099}", "\u{3099}シ", "シ\u{3099}", "\u{3099}ラ"]
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "コ\u{3099}"); // "コ" + combining voiced mark
        assert_eq!(tokens[1].text, "\u{3099}シ"); // combining mark + "シ"
        assert_eq!(tokens[2].text, "シ\u{3099}"); // "シ" + combining voiced mark
        assert_eq!(tokens[3].text, "\u{3099}ラ"); // combining mark + "ラ"
    }

    #[test]
    fn test_short_text() {
        let tokenizer = NgramTokenizer::new(3, 5).unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("ab").unwrap().collect();

        // Text is too short for any n-grams
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_exact_length() {
        let tokenizer = NgramTokenizer::new(3, 3).unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("abc").unwrap().collect();

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "abc");
    }

    #[test]
    fn test_tokenizer_name() {
        let tokenizer = NgramTokenizer::bigram();
        assert_eq!(tokenizer.name(), "ngram");
    }
}
