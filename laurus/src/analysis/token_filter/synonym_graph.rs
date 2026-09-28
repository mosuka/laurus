//! Synonym graph filter for token expansion with multi-word synonym support.
//!
//! This filter applies synonyms from a dictionary to an incoming token stream,
//! producing a correct graph output similar to Lucene's SynonymGraphFilter.

use crate::analysis::synonym::dictionary::SynonymDictionary;
use crate::analysis::synonym::graph_builder::SynonymGraphBuilder;
use crate::analysis::token::{Token, TokenStream};
use crate::analysis::token_filter::Filter;
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;

/// Synonym graph filter that applies synonyms while maintaining correct token graph structure.
///
/// This filter produces a token graph with proper `position_increment` and `position_length`
/// attributes, enabling correct phrase query matching with multi-word synonyms.
pub struct SynonymGraphFilter {
    builder: SynonymGraphBuilder,
}

impl SynonymGraphFilter {
    /// Create a new synonym graph filter.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `keep_original` - If true, keep original tokens alongside synonyms
    pub fn new(dictionary: SynonymDictionary, keep_original: bool) -> Self {
        let builder = SynonymGraphBuilder::new(dictionary, keep_original);
        Self { builder }
    }

    /// Create a new synonym graph filter with a tokenizer.
    ///
    /// The tokenizer will be used to tokenize multi-word synonyms, which is essential
    /// for languages like Japanese where words are not separated by whitespace.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `tokenizer` - Tokenizer to split synonym terms (e.g., for "機械学習" → ["機械", "学習"])
    /// * `keep_original` - If true, keep original tokens alongside synonyms
    pub fn with_tokenizer(
        dictionary: SynonymDictionary,
        tokenizer: Box<dyn Tokenizer>,
        keep_original: bool,
    ) -> Self {
        let builder = SynonymGraphBuilder::with_tokenizer(dictionary, tokenizer, keep_original);
        Self { builder }
    }

    /// Create a synonym graph filter from a dictionary file.
    pub fn from_file(path: &str, keep_original: bool) -> Result<Self> {
        let dictionary = SynonymDictionary::load_from_file(path)?;
        Ok(Self::new(dictionary, keep_original))
    }

    /// Create a synonym graph filter from a dictionary file with a tokenizer.
    pub fn from_file_with_tokenizer(
        path: &str,
        tokenizer: Box<dyn Tokenizer>,
        keep_original: bool,
    ) -> Result<Self> {
        let dictionary = SynonymDictionary::load_from_file(path)?;
        Ok(Self::with_tokenizer(dictionary, tokenizer, keep_original))
    }

    /// Set the boost multiplier for synonym tokens.
    ///
    /// This allows you to adjust the weight of synonym matches relative to original terms.
    /// For example, a boost of 0.8 means synonyms will have 80% of the weight of exact matches.
    ///
    /// # Arguments
    /// * `boost` - Boost multiplier for synonyms (typically 0.5-1.0)
    ///
    /// # Example
    /// ```
    /// use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
    /// use laurus::analysis::synonym::dictionary::SynonymDictionary;
    ///
    /// let mut dict = SynonymDictionary::new(None).unwrap();
    /// dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    ///
    /// let filter = SynonymGraphFilter::new(dict, true)
    ///     .with_boost(0.8); // Synonyms get 80% weight
    /// ```
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.builder = self.builder.with_boost(boost);
        self
    }

    /// Try to match a synonym starting at the given position in the token buffer.
    ///
    /// Returns (matched_phrase, matched_length, synonyms) if a match is found.
    fn try_match_synonym(
        &self,
        tokens: &[Token],
        start: usize,
    ) -> Option<(String, usize, Vec<String>)> {
        self.builder.try_match_synonym(tokens, start)
    }

    /// Build graph tokens from matched synonyms.
    fn build_graph_tokens(
        &self,
        original_tokens: &[Token],
        match_start: usize,
        match_length: usize,
        synonyms: &[String],
    ) -> Vec<Token> {
        self.builder
            .build_graph_tokens(original_tokens, match_start, match_length, synonyms)
    }
}

impl Filter for SynonymGraphFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let input_tokens: Vec<Token> = tokens.collect();
        let mut output_tokens = Vec::new();
        let mut i = 0;

        while i < input_tokens.len() {
            // Try to match a synonym at the current position
            if let Some((_, match_length, synonyms)) = self.try_match_synonym(&input_tokens, i) {
                // Build and add graph tokens
                let graph_tokens =
                    self.build_graph_tokens(&input_tokens, i, match_length, &synonyms);
                output_tokens.extend(graph_tokens);
                i += match_length;
            } else {
                // No match, add the original token
                output_tokens.push(input_tokens[i].clone());
                i += 1;
            }
        }

        Ok(Box::new(output_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "synonym_graph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_dictionary_basic() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "big".to_string(),
            "large".to_string(),
            "huge".to_string(),
        ]);

        let synonyms = dict.get_synonyms("big").unwrap();
        assert!(synonyms.contains(&"large".to_string()));
        assert!(synonyms.contains(&"huge".to_string()));
        assert!(!synonyms.contains(&"big".to_string()));
    }

    #[test]
    fn test_synonym_dictionary_multi_word() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "ml".to_string(),
            "machine learning".to_string(),
            "machine-learning".to_string(),
        ]);

        assert_eq!(dict.max_phrase_length(), 2);

        let synonyms = dict.get_synonyms("machine learning").unwrap();
        assert!(synonyms.contains(&"ml".to_string()));
        assert!(synonyms.contains(&"machine-learning".to_string()));
    }

    #[test]
    fn test_synonym_graph_filter_single_word() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        let tokens = vec![
            Token::new("the", 0),
            Token::new("big", 1),
            Token::new("cat", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "the", "big" (original), "large" (synonym), "cat"
        assert!(result.len() >= 4);
        assert_eq!(result[0].text, "the");
        assert_eq!(result[1].text, "big");
        assert_eq!(result[2].text, "large");
        assert_eq!(result[2].position_increment, 0); // Same position as "big"
        assert_eq!(result[3].text, "cat");
    }

    /// Around a multi-word synonym, the token after the match lands on the
    /// match's end node, both when the synonym is longer than the match and
    /// when it is shorter.
    #[test]
    fn the_token_after_a_match_follows_the_longest_path() {
        use crate::analysis::token::token_positions;
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
        let filter = SynonymGraphFilter::with_tokenizer(dict, Box::new(WhitespaceTokenizer), true);

        for (input, expected) in [
            (
                vec!["ml", "is", "fun"],
                vec![
                    ("ml", 0),
                    ("machine", 0),
                    ("learning", 1),
                    ("is", 2),
                    ("fun", 3),
                ],
            ),
            (
                vec!["machine", "learning", "is"],
                vec![("machine", 0), ("ml", 0), ("learning", 1), ("is", 2)],
            ),
        ] {
            let tokens: Vec<Token> = input
                .iter()
                .enumerate()
                .map(|(i, text)| Token::new(*text, i))
                .collect();
            let result: Vec<Token> = filter
                .filter(Box::new(tokens.into_iter()))
                .unwrap()
                .collect();
            let actual: Vec<(&str, u32)> = result
                .iter()
                .map(|t| t.text.as_str())
                .zip(token_positions(&result))
                .collect();
            assert_eq!(actual, expected, "{input:?}");
        }
    }

    #[test]
    fn test_synonym_graph_filter_multi_word() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "data science".to_string(),
            "ds".to_string(),
            "dataset".to_string(),
        ]);

        let filter = SynonymGraphFilter::new(dict, true);

        let tokens = vec![
            Token::new("data", 0),
            Token::new("science", 1),
            Token::new("intro", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should include original "data", "science" and synonyms "ds", "dataset"
        assert!(result.len() >= 4);

        // Find the "ds" token
        let ds_token = result.iter().find(|t| t.text == "ds");
        assert!(ds_token.is_some());
        let ds = ds_token.unwrap();
        assert_eq!(ds.position_increment, 0); // Stacked on position 0
        assert_eq!(ds.position_length, 2); // Spans 2 positions
    }

    #[test]
    fn test_synonym_dictionary_load_from_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary file with synonym data
        let mut file = NamedTempFile::new().unwrap();
        let json_content = r#"[
            ["ml", "machine learning"],
            ["学習", "勉強"]
        ]"#;
        write!(file, "{}", json_content).unwrap();
        let path = file.path().to_str().unwrap();

        let dict = SynonymDictionary::load_from_file(path).unwrap();

        // Test English synonyms
        let ml_synonyms = dict.get_synonyms("ml");
        assert!(ml_synonyms.is_some());
        let ml_synonyms = ml_synonyms.unwrap();
        assert!(ml_synonyms.contains(&"machine learning".to_string()));

        // Test Japanese synonyms
        let learning_synonyms = dict.get_synonyms("学習");
        assert!(learning_synonyms.is_some());
    }

    #[test]
    fn test_synonym_graph_filter_with_tokenizer() {
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        // Create a simple synonym dictionary
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        // Create filter with tokenizer
        let tokenizer = Box::new(WhitespaceTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        let tokens = vec![
            Token::new("ml", 0),
            Token::new("is", 1),
            Token::new("cool", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "ml", "machine", "learning", "is", "cool"
        assert!(result.len() >= 4);

        // Find "machine" and "learning" tokens
        let has_machine = result.iter().any(|t| t.text == "machine");
        let has_learning = result.iter().any(|t| t.text == "learning");

        assert!(has_machine, "Expected 'machine' token");
        assert!(has_learning, "Expected 'learning' token");
    }

    #[test]
    fn test_synonym_graph_filter_japanese_with_mock_tokenizer() {
        // Mock tokenizer that splits Japanese properly
        struct JapaneseTokenizer;
        impl crate::analysis::tokenizer::Tokenizer for JapaneseTokenizer {
            fn tokenize(
                &self,
                text: &str,
            ) -> crate::error::Result<crate::analysis::token::TokenStream> {
                // Simple mock: split "機械学習" into "機械" and "学習"
                let tokens: Vec<Token> = if text == "機械学習" {
                    vec![Token::new("機械", 0), Token::new("学習", 1)]
                } else {
                    vec![Token::new(text, 0)]
                };
                Ok(Box::new(tokens.into_iter()))
            }

            fn name(&self) -> &'static str {
                "japanese_tokenizer"
            }
        }

        // Create synonym dictionary with Japanese
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "機械学習".to_string()]);

        // Create filter with Japanese tokenizer
        let tokenizer = Box::new(JapaneseTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        let tokens = vec![Token::new("ml", 0)];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "ml", "機械", "学習"
        assert!(result.len() >= 3);

        // Find Japanese tokens
        let has_kikai = result.iter().any(|t| t.text == "機械");
        let has_gakushu = result.iter().any(|t| t.text == "学習");

        assert!(has_kikai, "Expected '機械' token from tokenized '機械学習'");
        assert!(
            has_gakushu,
            "Expected '学習' token from tokenized '機械学習'"
        );
    }

    #[test]
    fn test_synonym_match_without_spaces() {
        // Mock tokenizer that splits Japanese
        struct JapaneseTokenizer;
        impl crate::analysis::tokenizer::Tokenizer for JapaneseTokenizer {
            fn tokenize(
                &self,
                text: &str,
            ) -> crate::error::Result<crate::analysis::token::TokenStream> {
                let tokens: Vec<Token> = match text {
                    "機械学習" => vec![Token::new("機械", 0), Token::new("学習", 1)],
                    "マシンラーニング" => {
                        vec![Token::new("マシン", 0), Token::new("ラーニング", 1)]
                    }
                    other => vec![Token::new(other, 0)],
                };
                Ok(Box::new(tokens.into_iter()))
            }

            fn name(&self) -> &'static str {
                "japanese_tokenizer"
            }
        }

        // Dictionary with Japanese entry (no spaces)
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["機械学習".to_string(), "マシンラーニング".to_string()]);

        let tokenizer = Box::new(JapaneseTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        // Input: tokens that were already tokenized by Japanese tokenizer
        // ["機械", "学習"] should match "機械学習" in dictionary
        let tokens = vec![Token::new("機械", 0), Token::new("学習", 1)];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "機械", "学習" (original) and "マシン", "ラーニング" (synonym)
        assert!(
            result.len() >= 4,
            "Expected at least 4 tokens, got {}",
            result.len()
        );

        let has_machine = result.iter().any(|t| t.text == "マシン");
        let has_learning = result.iter().any(|t| t.text == "ラーニング");

        assert!(
            has_machine,
            "Expected 'マシン' token from synonym 'マシンラーニング'"
        );
        assert!(
            has_learning,
            "Expected 'ラーニング' token from synonym 'マシンラーニング'"
        );

        // Verify that the synonym tokens are at the same position as original
        let machine_token = result.iter().find(|t| t.text == "マシン").unwrap();
        assert_eq!(
            machine_token.position_increment, 0,
            "Synonym should be at same position"
        );
    }

    #[test]
    fn test_byte_offset_continuity_check() {
        // Test that tokens separated by whitespace in original text are not matched together
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["入門 本".to_string(), "beginner book".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens with non-continuous byte offsets (simulating "入門 本" with space)
        let mut token1 = Token::new("入門", 0);
        token1.start_offset = 15;
        token1.end_offset = 21;

        let mut token2 = Token::new("本", 1);
        token2.start_offset = 22; // Not continuous (21 != 22, there's a space)
        token2.end_offset = 25;

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should NOT match "入門 本" as a synonym because byte offsets are not continuous
        // Should only have the original 2 tokens
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "入門");
        assert_eq!(result[1].text, "本");

        // Verify no synonym tokens were added
        let has_beginner = result.iter().any(|t| t.text == "beginner");
        assert!(!has_beginner, "Should not match non-continuous tokens");
    }

    #[test]
    fn test_byte_offset_continuous_tokens_do_match() {
        // Test that tokens that ARE continuous in original text do match
        // Mock tokenizer for splitting Japanese synonyms
        struct JapaneseTokenizer;
        impl crate::analysis::tokenizer::Tokenizer for JapaneseTokenizer {
            fn tokenize(
                &self,
                text: &str,
            ) -> crate::error::Result<crate::analysis::token::TokenStream> {
                let tokens: Vec<Token> = if text == "マシンラーニング" {
                    vec![Token::new("マシン", 0), Token::new("ラーニング", 1)]
                } else {
                    vec![Token::new(text, 0)]
                };
                Ok(Box::new(tokens.into_iter()))
            }

            fn name(&self) -> &'static str {
                "japanese_tokenizer"
            }
        }

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["機械学習".to_string(), "マシンラーニング".to_string()]);

        let tokenizer = Box::new(JapaneseTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        // Create tokens with continuous byte offsets (no space between)
        let mut token1 = Token::new("機械", 0);
        token1.start_offset = 0;
        token1.end_offset = 6;

        let mut token2 = Token::new("学習", 1);
        token2.start_offset = 6; // Continuous (6 == 6)
        token2.end_offset = 12;

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should match "機械学習" and add synonym tokens
        assert!(
            result.len() >= 4,
            "Expected at least 4 tokens (2 original + 2 synonym), got {}",
            result.len()
        );

        // Verify synonym tokens were added
        let has_machine = result.iter().any(|t| t.text == "マシン");
        let has_learning = result.iter().any(|t| t.text == "ラーニング");
        assert!(
            has_machine,
            "Should match continuous tokens and add synonyms"
        );
        assert!(
            has_learning,
            "Should match continuous tokens and add synonyms"
        );
    }

    #[test]
    fn test_alphanum_with_spaces() {
        // Test that ALPHANUM tokens with spaces between them still match
        use crate::analysis::token::{Token, TokenMetadata, TokenType};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["machine learning".to_string(), "ml".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens with ALPHANUM type and spaces between them
        let mut token1 = Token::new("machine", 0);
        token1.start_offset = 0;
        token1.end_offset = 7;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Alphanum),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("learning", 1);
        token2.start_offset = 8; // Not continuous (space at position 7)
        token2.end_offset = 16;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Alphanum),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should match "machine learning" even though there's a space (both ALPHANUM)
        let has_ml = result.iter().any(|t| t.text == "ml");
        assert!(has_ml, "ALPHANUM tokens with spaces should still match");
    }

    #[test]
    fn test_mixed_alphanum_cjk_continuous() {
        // Test mixed ALPHANUM + CJK tokens that are continuous
        use crate::analysis::token::{Token, TokenMetadata, TokenType};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "100円ショップ".to_string(),
            "hundred yen shop".to_string(),
        ]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens: "100" (ALPHANUM) + "円" (CJK) + "ショップ" (CJK)
        let mut token1 = Token::new("100", 0);
        token1.start_offset = 0;
        token1.end_offset = 3;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Num),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("円", 1);
        token2.start_offset = 3; // Continuous
        token2.end_offset = 6;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token3 = Token::new("ショップ", 2);
        token3.start_offset = 6; // Continuous
        token3.end_offset = 18;
        token3.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2, token3];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should match "100円ショップ" because all are continuous
        let has_synonym = result.iter().any(|t| t.text == "hundred");
        assert!(
            has_synonym,
            "Mixed ALPHANUM+CJK continuous tokens should match"
        );
    }

    #[test]
    fn test_mixed_alphanum_cjk_non_continuous() {
        // Test mixed ALPHANUM + CJK tokens that are NOT continuous
        use crate::analysis::token::{Token, TokenMetadata, TokenType};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["machine 学習".to_string(), "test synonym".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens: "machine" (ALPHANUM) + "学習" (CJK) with space
        let mut token1 = Token::new("machine", 0);
        token1.start_offset = 0;
        token1.end_offset = 7;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Alphanum),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("学習", 1);
        token2.start_offset = 8; // Not continuous (space at position 7)
        token2.end_offset = 14;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should NOT match because mixed types with space
        let has_synonym = result.iter().any(|t| t.text == "test");
        assert!(
            !has_synonym,
            "Mixed ALPHANUM+CJK with space should NOT match"
        );
    }

    #[test]
    fn test_cjk_non_continuous() {
        // Test CJK tokens that are NOT continuous (space between)
        use crate::analysis::token::{Token, TokenMetadata, TokenType};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["入門 本".to_string(), "beginner book".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens: "入門" + "本" with space
        let mut token1 = Token::new("入門", 0);
        token1.start_offset = 0;
        token1.end_offset = 6;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("本", 1);
        token2.start_offset = 7; // Not continuous (space at position 6)
        token2.end_offset = 10;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should NOT match because CJK with space
        let has_synonym = result.iter().any(|t| t.text == "beginner");
        assert!(!has_synonym, "CJK tokens with space should NOT match");
    }

    #[test]
    fn test_single_to_multi_word_synonym_position_length() {
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        // Test Case 1: Single word → Multi-word synonym expansion
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
        dict.add_synonym_group(vec![
            "ai".to_string(),
            "artificial intelligence".to_string(),
        ]);

        let tokenizer = Box::new(WhitespaceTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        let tokens = vec![
            Token::new("ml", 0),
            Token::new("and", 1),
            Token::new("ai", 2),
            Token::new("tutorial", 3),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Verify "machine" token (first token of multi-word synonym)
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some(), "Expected 'machine' token");
        let machine = machine_token.unwrap();
        assert_eq!(machine.position, 0);
        assert_eq!(machine.position_increment, 0);
        assert_eq!(machine.position_length, 1);
        let ml = result.iter().find(|t| t.text == "ml").unwrap();
        assert_eq!(
            ml.position_length, 2,
            "the original spans both words of its multi-word synonym"
        );

        // Verify "learning" token (second token of multi-word synonym)
        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(learning_token.is_some(), "Expected 'learning' token");
        let learning = learning_token.unwrap();
        assert_eq!(learning.position, 1);
        assert_eq!(learning.position_increment, 1);
        assert_eq!(learning.position_length, 1);

        // Verify "artificial" token (first token of multi-word synonym)
        let artificial_token = result.iter().find(|t| t.text == "artificial");
        assert!(artificial_token.is_some(), "Expected 'artificial' token");
        let artificial = artificial_token.unwrap();
        assert_eq!(artificial.position, 2);
        assert_eq!(artificial.position_increment, 0);
        assert_eq!(artificial.position_length, 1);
        let ai = result.iter().find(|t| t.text == "ai").unwrap();
        assert_eq!(ai.position_length, 2);

        // Verify "intelligence" token (second token of multi-word synonym)
        let intelligence_token = result.iter().find(|t| t.text == "intelligence");
        assert!(
            intelligence_token.is_some(),
            "Expected 'intelligence' token"
        );
        let intelligence = intelligence_token.unwrap();
        assert_eq!(intelligence.position, 3);
        assert_eq!(intelligence.position_increment, 1);
        assert_eq!(intelligence.position_length, 1);
    }

    #[test]
    fn test_multi_to_single_word_synonym_position_length() {
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        // Test Case 2: Multi-word → Single word synonym expansion
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let tokenizer = Box::new(WhitespaceTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        let tokens = vec![
            Token::new("machine", 0),
            Token::new("learning", 1),
            Token::new("tutorial", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Verify "ml" token (single-word synonym for multi-word phrase)
        let ml_token = result.iter().find(|t| t.text == "ml");
        assert!(ml_token.is_some(), "Expected 'ml' token");
        let ml = ml_token.unwrap();
        assert_eq!(ml.position, 0);
        assert_eq!(ml.position_increment, 0);
        assert_eq!(
            ml.position_length, 2,
            "Single-word synonym for multi-word phrase should have pos_len=2"
        );

        // Verify original tokens are kept
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some(), "Expected original 'machine' token");

        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(
            learning_token.is_some(),
            "Expected original 'learning' token"
        );
    }

    #[test]
    fn test_japanese_synonym_expansion_with_lindera() {
        use crate::analysis::tokenizer::lindera::LinderaTokenizer;

        // Test Japanese synonym expansion with LinderaTokenizer
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "機械学習".to_string()]);
        dict.add_synonym_group(vec!["ai".to_string(), "人工知能".to_string()]);

        let lindera_tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();
        let tokenizer = Box::new(lindera_tokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        // Case 1: English → Japanese
        let en_tokens = vec![
            Token::new("ml", 0),
            Token::new("and", 1),
            Token::new("ai", 2),
        ];

        let result = filter
            .filter(Box::new(en_tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Verify "機械" token (first token of Japanese multi-word synonym)
        let kikai_token = result.iter().find(|t| t.text == "機械");
        assert!(kikai_token.is_some(), "Expected '機械' token");
        let kikai = kikai_token.unwrap();
        assert_eq!(kikai.position, 0);
        assert_eq!(kikai.position_increment, 0);
        assert_eq!(kikai.position_length, 1);

        // Verify "学習" token (second token of Japanese multi-word synonym)
        let gakushu_token = result.iter().find(|t| t.text == "学習");
        assert!(gakushu_token.is_some(), "Expected '学習' token");
        let gakushu = gakushu_token.unwrap();
        assert_eq!(gakushu.position, 1);
        assert_eq!(gakushu.position_increment, 1);
        assert_eq!(gakushu.position_length, 1);

        // Verify "人工" token (first token of Japanese multi-word synonym)
        let jinko_token = result.iter().find(|t| t.text == "人工");
        assert!(jinko_token.is_some(), "Expected '人工' token");
        let jinko = jinko_token.unwrap();
        assert_eq!(jinko.position, 2);
        assert_eq!(jinko.position_increment, 0);
        assert_eq!(jinko.position_length, 1);

        // Verify "知能" token (second token of Japanese multi-word synonym)
        let chino_token = result.iter().find(|t| t.text == "知能");
        assert!(chino_token.is_some(), "Expected '知能' token");
        let chino = chino_token.unwrap();
        assert_eq!(chino.position, 3);
        assert_eq!(chino.position_increment, 1);
        assert_eq!(chino.position_length, 1);
    }

    #[test]
    fn test_synonym_graph_filter_with_boost() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
        dict.add_synonym_group(vec![
            "ai".to_string(),
            "artificial intelligence".to_string(),
        ]);

        let filter = SynonymGraphFilter::new(dict, true).with_boost(0.8);

        let tokens = vec![Token::new("ml", 0), Token::new("tutorial", 1)];

        let result: Vec<Token> = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect();

        // Should have: ml, machine, learning, tutorial
        assert!(result.len() >= 4);

        // Original token should have boost = 1.0
        let ml_token = result.iter().find(|t| t.text == "ml");
        assert!(ml_token.is_some());
        assert_eq!(ml_token.unwrap().boost, 1.0);

        // Synonym tokens should have boost applied
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some());
        // First token of multi-word synonym: 0.9 * 0.8 = 0.72
        assert!((machine_token.unwrap().boost - 0.72).abs() < 0.001);

        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(learning_token.is_some());
        // Second token: 0.8 * 0.8 = 0.64
        assert!((learning_token.unwrap().boost - 0.64).abs() < 0.001);
    }
}
