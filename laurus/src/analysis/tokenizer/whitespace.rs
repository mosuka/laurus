//! Whitespace tokenizer implementation.
//!
//! This module provides a tokenizer that splits text on whitespace characters
//! (spaces, tabs, newlines, etc.). It includes SIMD-optimized processing for
//! ASCII text and automatic token type detection.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::tokenizer::Tokenizer;
//! use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
//!
//! let tokenizer = WhitespaceTokenizer::new();
//! let tokens: Vec<_> = tokenizer.tokenize("Hello   world\ttest").unwrap().collect();
//!
//! assert_eq!(tokens.len(), 3);
//! assert_eq!(tokens[0].text, "Hello");
//! assert_eq!(tokens[1].text, "world");
//! assert_eq!(tokens[2].text, "test");
//! ```

use crate::analysis::token::{Token, TokenStream, TokenType};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;
use crate::util::simd;

/// A tokenizer that splits text on whitespace characters.
///
/// This tokenizer uses SIMD-optimized processing for ASCII text longer than
/// 32 bytes, providing improved performance for large documents. It also
/// automatically detects token types (alphanumeric, numeric, CJK, etc.) based
/// on character content.
///
/// # Performance
///
/// - ASCII text ≥32 bytes: Uses SIMD-optimized path
/// - Other text: Falls back to standard Unicode processing
///
/// # Token Types Detected
///
/// - Numeric values
/// - CJK characters
/// - Katakana/Hiragana (Japanese)
/// - Hangul (Korean)
/// - Alphanumeric ASCII
/// - Punctuation
///
/// # Examples
///
/// ```
/// use laurus::analysis::tokenizer::Tokenizer;
/// use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
///
/// let tokenizer = WhitespaceTokenizer::new();
/// let tokens: Vec<_> = tokenizer.tokenize("hello world").unwrap().collect();
/// assert_eq!(tokens.len(), 2);
/// ```
#[derive(Clone, Debug, Default)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        WhitespaceTokenizer
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        // Use SIMD-optimized whitespace detection for ASCII text
        if text.is_ascii() && text.len() >= 32 {
            self.tokenize_simd(text)
        } else {
            self.tokenize_fallback(text)
        }
    }

    fn name(&self) -> &'static str {
        "whitespace"
    }
}

impl WhitespaceTokenizer {
    /// Detect token type based on the content of the word.
    ///
    /// Analyzes the characters in the word to determine its type:
    /// - All digits → Num
    /// - Contains CJK characters → Cjk
    /// - All Katakana → Katakana
    /// - All Hiragana → Hiragana
    /// - Contains Hangul → Hangul
    /// - ASCII alphanumeric → Alphanum
    /// - All punctuation → Punctuation
    /// - Otherwise → Other
    fn detect_token_type(word: &str) -> TokenType {
        if word.is_empty() {
            return TokenType::Other;
        }

        // Check if all characters are numeric
        if word.chars().all(|c| c.is_ascii_digit()) {
            return TokenType::Num;
        }

        // Check if it contains CJK characters
        if word.chars().any(|c| {
            matches!(c,
                '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
                '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
                '\u{20000}'..='\u{2A6DF}' | // CJK Extension B
                '\u{2A700}'..='\u{2B73F}' | // CJK Extension C
                '\u{2B740}'..='\u{2B81F}' | // CJK Extension D
                '\u{2B820}'..='\u{2CEAF}'   // CJK Extension E
            )
        }) {
            return TokenType::Cjk;
        }

        // Check if it's Katakana
        if word.chars().all(|c| matches!(c, '\u{30A0}'..='\u{30FF}')) {
            return TokenType::Katakana;
        }

        // Check if it's Hiragana
        if word.chars().all(|c| matches!(c, '\u{3040}'..='\u{309F}')) {
            return TokenType::Hiragana;
        }

        // Check if it's Hangul
        if word
            .chars()
            .any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}'))
        {
            return TokenType::Hangul;
        }

        // Check if it's alphanumeric (ASCII)
        if word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return TokenType::Alphanum;
        }

        // Check if it's all punctuation
        if word.chars().all(|c| c.is_ascii_punctuation()) {
            return TokenType::Punctuation;
        }

        // Default to Other
        TokenType::Other
    }

    /// SIMD-optimized tokenization for ASCII text.
    ///
    /// Uses SIMD instructions to quickly find whitespace boundaries in ASCII text,
    /// providing significant performance improvements for large documents.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text (must be ASCII)
    fn tokenize_simd(&self, text: &str) -> Result<TokenStream> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut position = 0;
        let mut start = 0;

        // Skip leading whitespace
        while start < bytes.len() && bytes[start].is_ascii_whitespace() {
            start += 1;
        }

        while start < bytes.len() {
            // Find the end of the current word using SIMD
            let word_end = match simd::ascii::find_whitespace_simd(&bytes[start..]) {
                Some(offset) => start + offset,
                None => bytes.len(),
            };

            if word_end > start {
                // Extract the word
                let word = &text[start..word_end];
                let token_type = Self::detect_token_type(word);
                let token = Token::with_offsets(word, position, start, word_end)
                    .with_token_type(token_type);
                tokens.push(token);
                position += 1;
            }

            // Skip whitespace to find the next word
            start = word_end;
            while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                start += 1;
            }
        }

        Ok(Box::new(tokens.into_iter()))
    }

    /// Fallback implementation for non-ASCII or short text.
    ///
    /// Uses standard Unicode string splitting for text that doesn't meet
    /// the criteria for SIMD optimization.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text
    fn tokenize_fallback(&self, text: &str) -> Result<TokenStream> {
        // `split_whitespace` yields substrings that are slices of `text`, so
        // the pointer difference gives each word's real byte offset. This
        // works for repeated words and words that occur inside an earlier
        // word, unlike a `text.find(word)` lookup which always returns the
        // first occurrence.
        let text_start = text.as_ptr() as usize;
        let tokens: Vec<Token> = text
            .split_whitespace()
            .enumerate()
            .map(|(position, word)| {
                let start_offset = word.as_ptr() as usize - text_start;
                let end_offset = start_offset + word.len();
                let token_type = Self::detect_token_type(word);
                Token::with_offsets(word, position, start_offset, end_offset)
                    .with_token_type(token_type)
            })
            .collect();

        Ok(Box::new(tokens.into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize("hello  world\ttest").unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }

    #[test]
    fn test_tokenizer_name() {
        assert_eq!(WhitespaceTokenizer::new().name(), "whitespace");
    }

    /// Every token's offsets must slice back to its own text, not to some
    /// other occurrence of the same text elsewhere in the input.
    fn assert_offsets_match(text: &str, tokens: &[Token]) {
        for token in tokens {
            assert_eq!(
                &text[token.start_offset..token.end_offset],
                token.text,
                "offset mismatch for token {:?} in {:?}",
                token.text,
                text
            );
        }
    }

    #[test]
    fn test_fallback_repeated_word_offsets() {
        let tokenizer = WhitespaceTokenizer::new();
        let text = "a b a";
        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_offsets_match(text, &tokens);
        assert_eq!((tokens[2].start_offset, tokens[2].end_offset), (4, 5));
    }

    #[test]
    fn test_fallback_word_inside_earlier_word_offsets() {
        let tokenizer = WhitespaceTokenizer::new();
        let text = "ab b";
        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_offsets_match(text, &tokens);
        assert_eq!((tokens[1].start_offset, tokens[1].end_offset), (3, 4));
    }

    #[test]
    fn test_fallback_non_ascii_repeated_word_offsets() {
        let tokenizer = WhitespaceTokenizer::new();
        let text = "猫 犬 猫";
        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_offsets_match(text, &tokens);
        assert_ne!(tokens[0].start_offset, tokens[2].start_offset);
    }
}
