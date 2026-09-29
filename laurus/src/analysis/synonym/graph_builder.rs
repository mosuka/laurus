//! Token graph builder for constructing synonym graphs.
//!
//! This module provides the core logic for building token graphs with proper
//! position_increment and position_length attributes for synonym expansion.

use crate::analysis::synonym::dictionary::SynonymDictionary;
use crate::analysis::token::{Token, TokenType};
use crate::analysis::tokenizer::Tokenizer;

/// Builds token graphs with synonym expansion.
pub struct SynonymGraphBuilder {
    dictionary: SynonymDictionary,
    tokenizer: Option<Box<dyn Tokenizer>>,
    keep_original: bool,
    /// Boost multiplier for synonym tokens (None means no boost adjustment)
    synonym_boost: Option<f32>,
}

impl SynonymGraphBuilder {
    /// Create a new graph builder.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `keep_original` - Whether to keep original tokens alongside synonyms
    pub fn new(dictionary: SynonymDictionary, keep_original: bool) -> Self {
        Self {
            dictionary,
            tokenizer: None,
            keep_original,
            synonym_boost: None,
        }
    }

    /// Create a new graph builder with a tokenizer.
    ///
    /// The tokenizer will be used to split multi-word synonyms into individual tokens.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `tokenizer` - Tokenizer for splitting synonym terms
    /// * `keep_original` - Whether to keep original tokens alongside synonyms
    pub fn with_tokenizer(
        dictionary: SynonymDictionary,
        tokenizer: Box<dyn Tokenizer>,
        keep_original: bool,
    ) -> Self {
        Self {
            dictionary,
            tokenizer: Some(tokenizer),
            keep_original,
            synonym_boost: None,
        }
    }

    /// Set the boost multiplier for synonym tokens.
    ///
    /// # Arguments
    /// * `boost` - Boost multiplier (e.g., 0.8 to reduce synonym weight to 80%)
    ///
    /// # Example
    /// ```
    /// use laurus::analysis::synonym::dictionary::SynonymDictionary;
    /// use laurus::analysis::synonym::graph_builder::SynonymGraphBuilder;
    ///
    /// let mut dict = SynonymDictionary::new(None).unwrap();
    /// dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    ///
    /// let builder = SynonymGraphBuilder::new(dict, true)
    ///     .with_boost(0.8); // Synonyms get 80% of original weight
    /// ```
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.synonym_boost = Some(boost);
        self
    }

    /// Build graph tokens from matched synonyms.
    ///
    /// The matched words (with `keep_original`) and each synonym form one
    /// path each, from the match's start node to one shared end node; paths
    /// with the same words are one path. A token is an arc from its node to
    /// its node + `position_length`. A path of `k` words has `k - 1` inner
    /// nodes of its own, allocated after the start node path by path, so no
    /// two paths share an arc and a quoted value follows each path exactly,
    /// as with Lucene's `SynonymGraphFilter`. The end node follows the last
    /// inner node.
    ///
    /// Tokens come out in node order: every path's first word at the start
    /// node, then each path's other words. The first token at the start
    /// node keeps the matched word's increment, the others there are
    /// stacked on it with increment 0, and each inner node's token has
    /// increment 1, so the token after the match lands on the end node. A
    /// token after the match that is itself stacked (increment 0, as an
    /// earlier synonym filter can leave it) lands on the last inner node.
    ///
    /// The index stores no `position_length`, so indexing lays the graph
    /// out on its longest path ([`flatten_token_graph`]), where the paths'
    /// words share positions again.
    ///
    /// [`flatten_token_graph`]: crate::analysis::token::flatten_token_graph
    pub fn build_graph_tokens(
        &self,
        original_tokens: &[Token],
        match_start: usize,
        match_length: usize,
        synonyms: &[String],
    ) -> Vec<Token> {
        let matched = &original_tokens[match_start..match_start + match_length];
        let first = &matched[0];
        let match_start_offset = first.start_offset;
        let match_end_offset = matched[match_length - 1].end_offset;

        let synonym_paths: Vec<Vec<Token>> = synonyms
            .iter()
            .map(|synonym| self.split_synonym(synonym))
            .filter(|words| !words.is_empty())
            .map(|words| {
                let single_word = words.len() == 1;
                words
                    .iter()
                    .enumerate()
                    .map(|(i, word)| {
                        let mut token = Token::new(word, first.position + i)
                            .with_token_type(TokenType::Synonym);
                        token.start_offset = match_start_offset;
                        token.end_offset = match_end_offset;
                        if let Some(boost) = self.synonym_boost {
                            // A single word standing for several gets a little more
                            // weight, as does the first word of a multi-word synonym.
                            let base_boost = match (single_word, i) {
                                (true, _) if match_length > 1 => 0.9,
                                (true, _) => 0.8,
                                (false, 0) => 0.9,
                                (false, _) => 0.8,
                            };
                            token = token.with_boost(base_boost * boost);
                        }
                        token
                    })
                    .collect()
            })
            .collect();

        // A group with no other member leaves nothing to replace the match
        // with, so the matched words stay even without `keep_original`.
        let mut paths: Vec<Vec<Token>> = Vec::with_capacity(synonym_paths.len() + 1);
        if self.keep_original || synonym_paths.is_empty() {
            paths.push(matched.to_vec());
        }
        for path in synonym_paths {
            let same_words = |kept: &Vec<Token>| {
                kept.len() == path.len() && kept.iter().zip(&path).all(|(a, b)| a.text == b.text)
            };
            if !paths.iter().any(same_words) {
                paths.push(path);
            }
        }

        // The first inner node of each path; one past the last is the end.
        let mut next_node = 1;
        let first_inner_nodes: Vec<usize> = paths
            .iter()
            .map(|path| {
                let first_inner = next_node;
                next_node += path.len() - 1;
                first_inner
            })
            .collect();
        let end_node = next_node;

        let mut result = Vec::with_capacity(paths.iter().map(Vec::len).sum());
        for (i, (path, &first_inner)) in paths.iter().zip(&first_inner_nodes).enumerate() {
            let mut token = path[0].clone();
            token.position_increment = if i == 0 { first.position_increment } else { 0 };
            token.position_length = if path.len() == 1 {
                end_node
            } else {
                first_inner
            };
            result.push(token);
        }
        for (path, &first_inner) in paths.iter().zip(&first_inner_nodes) {
            for (k, word) in path.iter().enumerate().skip(1) {
                let node = first_inner + k - 1;
                let mut token = word.clone();
                token.position_increment = 1;
                token.position_length = if k + 1 == path.len() {
                    end_node - node
                } else {
                    1
                };
                result.push(token);
            }
        }

        result
    }

    /// Split a synonym into words with the configured tokenizer, or on
    /// whitespace when there is none or it fails.
    fn split_synonym(&self, synonym: &str) -> Vec<String> {
        if let Some(tokenizer) = &self.tokenizer
            && let Ok(tokens) = tokenizer.tokenize(synonym)
        {
            return tokens.map(|t| t.text).collect();
        }
        synonym.split_whitespace().map(|s| s.to_string()).collect()
    }

    /// Try to match a synonym starting at the given position in the token buffer.
    ///
    /// Returns (matched_phrase, matched_length, synonyms) if a match is found.
    pub fn try_match_synonym(
        &self,
        tokens: &[Token],
        start: usize,
    ) -> Option<(String, usize, Vec<String>)> {
        let max_len = (tokens.len() - start).min(self.dictionary.max_phrase_length());

        // Try longest match first (greedy matching)
        for len in (1..=max_len).rev() {
            // Check byte offset continuity for multi-token phrases
            if len > 1 {
                // Check if all tokens are ALPHANUM type
                let all_alphanum = tokens[start..start + len].iter().all(|t| {
                    matches!(
                        t.metadata.as_ref().and_then(|m| m.token_type),
                        Some(TokenType::Alphanum) | Some(TokenType::Num)
                    )
                });

                // If not all ALPHANUM, check byte offset continuity
                if !all_alphanum {
                    let mut is_continuous = true;
                    for i in 0..len - 1 {
                        let current = &tokens[start + i];
                        let next = &tokens[start + i + 1];
                        if current.end_offset != next.start_offset {
                            is_continuous = false;
                            break;
                        }
                    }
                    // Skip this length if tokens are not continuous
                    if !is_continuous {
                        continue;
                    }
                }
            }

            let token_texts: Vec<&str> = tokens[start..start + len]
                .iter()
                .map(|t| t.text.as_str())
                .collect();

            // Try with space separator first (for English, etc.)
            let phrase_with_space = token_texts.join(" ");
            if let Some(synonyms) = self.dictionary.get_synonyms(&phrase_with_space) {
                return Some((phrase_with_space, len, synonyms.clone()));
            }

            // Try without space separator (for Japanese, Chinese, etc.)
            // Only do this for multi-token phrases
            if len > 1 {
                let phrase_no_space = token_texts.join("");
                if let Some(synonyms) = self.dictionary.get_synonyms(&phrase_no_space) {
                    return Some((phrase_no_space, len, synonyms.clone()));
                }
            }
        }

        None
    }

    /// Get a reference to the dictionary.
    pub fn dictionary(&self) -> &SynonymDictionary {
        &self.dictionary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::{flatten_token_graph, token_positions};
    use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

    /// Each token as an arc `(text, from, to)` of the token graph, plus the
    /// increments, which must be 1 for the first token at each node.
    fn arcs(tokens: &[Token]) -> (Vec<(String, u32, u32)>, Vec<usize>) {
        let arcs = tokens
            .iter()
            .zip(token_positions(tokens))
            .map(|(t, from)| (t.text.clone(), from, from + t.position_length as u32))
            .collect();
        let increments = tokens.iter().map(|t| t.position_increment).collect();
        (arcs, increments)
    }

    fn arc(text: &str, from: u32, to: u32) -> (String, u32, u32) {
        (text.to_string(), from, to)
    }

    fn words(texts: &[&str]) -> Vec<Token> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| Token::new(*text, i))
            .collect()
    }

    fn synonyms(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|s| s.to_string()).collect()
    }

    fn builder(keep_original: bool) -> SynonymGraphBuilder {
        let dict = SynonymDictionary::new(None).unwrap();
        SynonymGraphBuilder::with_tokenizer(dict, Box::new(WhitespaceTokenizer), keep_original)
    }

    /// The original spans the whole match when a synonym is longer, and
    /// every path ends at node 2.
    #[test]
    fn one_word_to_two_words_is_a_graph_with_one_end() {
        let result = builder(true).build_graph_tokens(
            &words(&["ml"]),
            0,
            1,
            &synonyms(&["machine learning"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![arc("ml", 0, 2), arc("machine", 0, 1), arc("learning", 1, 2)],
                vec![1, 0, 1]
            )
        );
    }

    /// The second original word takes the next position instead of being
    /// stacked on the first.
    #[test]
    fn two_words_to_one_word_keeps_the_originals_in_sequence() {
        let result = builder(true).build_graph_tokens(
            &words(&["machine", "learning"]),
            0,
            2,
            &synonyms(&["ml"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![arc("machine", 0, 1), arc("ml", 0, 2), arc("learning", 1, 2)],
                vec![1, 0, 1]
            )
        );
    }

    /// Two paths of two words each get an inner node of their own, even
    /// when their second words are the same.
    #[test]
    fn two_words_to_two_words_gives_each_its_own_inner_node() {
        let result = builder(true).build_graph_tokens(
            &words(&["machine", "learning"]),
            0,
            2,
            &synonyms(&["deep learning"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![
                    arc("machine", 0, 1),
                    arc("deep", 0, 2),
                    arc("learning", 1, 3),
                    arc("learning", 2, 3)
                ],
                vec![1, 0, 1, 1]
            )
        );
    }

    /// Paths of 1, 2 and 3 words all end at node 4, after the inner nodes
    /// of "machine learning" (1) and "statistical machine learning" (2, 3),
    /// so no arc holds words of two members.
    #[test]
    fn paths_of_different_lengths_end_at_the_same_node() {
        let result = builder(true).build_graph_tokens(
            &words(&["ml"]),
            0,
            1,
            &synonyms(&["machine learning", "statistical machine learning"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![
                    arc("ml", 0, 4),
                    arc("machine", 0, 1),
                    arc("statistical", 0, 2),
                    arc("learning", 1, 4),
                    arc("machine", 2, 3),
                    arc("learning", 3, 4)
                ],
                vec![1, 0, 0, 1, 1, 1]
            )
        );
    }

    #[test]
    fn without_the_original_each_multi_word_synonym_has_its_own_nodes() {
        let result = builder(false).build_graph_tokens(
            &words(&["ml"]),
            0,
            1,
            &synonyms(&["machine learning", "statistical machine learning"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![
                    arc("machine", 0, 1),
                    arc("statistical", 0, 2),
                    arc("learning", 1, 4),
                    arc("machine", 2, 3),
                    arc("learning", 3, 4)
                ],
                vec![1, 0, 1, 1, 1]
            )
        );
    }

    /// The matched words are a path like any other: a three-word match
    /// takes inner nodes 1 and 2, and "machine learning" takes node 3.
    #[test]
    fn a_three_word_match_keeps_its_words_on_their_own_nodes() {
        let result = builder(true).build_graph_tokens(
            &words(&["statistical", "machine", "learning"]),
            0,
            3,
            &synonyms(&["ml", "machine learning"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![
                    arc("statistical", 0, 1),
                    arc("ml", 0, 4),
                    arc("machine", 0, 3),
                    arc("machine", 1, 2),
                    arc("learning", 2, 4),
                    arc("learning", 3, 4)
                ],
                vec![1, 0, 0, 1, 1, 1]
            )
        );
    }

    /// A synonym that splits into the same words as another path would
    /// repeat that path's phrase.
    #[test]
    fn a_synonym_with_the_words_of_another_path_is_one_path() {
        let result = builder(true).build_graph_tokens(
            &words(&["machine", "learning"]),
            0,
            2,
            &synonyms(&["machine  learning", "ml", "ml"]),
        );
        assert_eq!(
            arcs(&result),
            (
                vec![arc("machine", 0, 1), arc("ml", 0, 2), arc("learning", 1, 2)],
                vec![1, 0, 1]
            )
        );
    }

    /// The flat layout the builder emitted before #1262: the `k`-th word of
    /// every path on node `k`.
    fn shared_inner_nodes(paths: &[Vec<&str>]) -> Vec<(String, usize, usize)> {
        let longest = paths.iter().map(Vec::len).max().unwrap_or(0);
        let mut layout = Vec::new();
        for node in 0..longest {
            let mut first_at_node = true;
            for path in paths {
                let Some(word) = path.get(node) else {
                    continue;
                };
                let length = if node + 1 == path.len() {
                    longest - node
                } else {
                    1
                };
                layout.push((word.to_string(), usize::from(first_at_node), length));
                first_at_node = false;
            }
        }
        layout
    }

    /// Laid out for the index, the graph is exactly the flat layout the
    /// builder emitted before #1262, so the stored positions do not change.
    #[test]
    fn flattened_the_graph_shares_the_inner_nodes_again() {
        let members = [
            "ml",
            "machine learning",
            "statistical machine learning",
            "deep vision",
            "a b c d",
        ];
        for matched in ["ml", "machine learning", "statistical machine learning"] {
            let synonyms: Vec<String> = members
                .iter()
                .filter(|member| **member != matched)
                .map(|member| member.to_string())
                .collect();
            let matched_words: Vec<&str> = matched.split(' ').collect();
            for keep_original in [true, false] {
                for count in 0..=synonyms.len() {
                    let synonyms = &synonyms[..count];
                    let result = builder(keep_original).build_graph_tokens(
                        &words(&matched_words),
                        0,
                        matched_words.len(),
                        synonyms,
                    );

                    let mut paths = Vec::new();
                    if keep_original || synonyms.is_empty() {
                        paths.push(matched_words.clone());
                    }
                    paths.extend(synonyms.iter().map(|s| s.split(' ').collect::<Vec<_>>()));
                    let flat: Vec<(String, usize, usize)> = flatten_token_graph(result)
                        .into_iter()
                        .map(|t| (t.text, t.position_increment, t.position_length))
                        .collect();
                    assert_eq!(
                        flat,
                        shared_inner_nodes(&paths),
                        "{matched:?} → {synonyms:?}, keep_original = {keep_original}"
                    );
                }
            }
        }
    }

    /// Without the original, the alternatives are still stacked.
    #[test]
    fn without_the_original_the_synonyms_are_stacked() {
        let result = builder(false).build_graph_tokens(
            &words(&["big"]),
            0,
            1,
            &synonyms(&["large", "huge"]),
        );
        assert_eq!(
            arcs(&result),
            (vec![arc("large", 0, 1), arc("huge", 0, 1)], vec![1, 0])
        );
    }

    /// A group with one member has no synonyms; the matched word must
    /// survive even without `keep_original`.
    #[test]
    fn a_match_without_synonyms_keeps_the_original() {
        let result = builder(false).build_graph_tokens(&words(&["big"]), 0, 1, &[]);
        assert_eq!(arcs(&result), (vec![arc("big", 0, 1)], vec![1]));
    }

    #[test]
    fn a_synonym_with_no_words_is_skipped() {
        let result =
            builder(true).build_graph_tokens(&words(&["big"]), 0, 1, &synonyms(&[" ", "large"]));
        assert_eq!(
            arcs(&result),
            (vec![arc("big", 0, 1), arc("large", 0, 1)], vec![1, 0])
        );
    }

    /// A match after the first token keeps that token's increment and
    /// the synonyms' `position` values.
    #[test]
    fn a_match_keeps_the_first_word_increment_and_positions() {
        let tokens = words(&["the", "ml"]);
        let result =
            builder(true).build_graph_tokens(&tokens, 1, 1, &synonyms(&["machine learning"]));
        let positions: Vec<usize> = result.iter().map(|t| t.position).collect();
        assert_eq!(positions, vec![1, 1, 2]);
        assert_eq!(result[0].position_increment, 1);
    }

    #[test]
    fn test_build_graph_tokens_single_word_synonym() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        let builder = SynonymGraphBuilder::new(dict, true);

        let original_tokens = vec![Token::new("big", 0)];
        let synonyms = vec!["large".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Should have original + synonym
        assert!(result.len() >= 2);

        // Find synonym token
        let large_token = result.iter().find(|t| t.text == "large");
        assert!(large_token.is_some());
        let large = large_token.unwrap();
        assert_eq!(large.position_increment, 0);
        assert_eq!(large.position_length, 1);
    }

    #[test]
    fn test_build_graph_tokens_multi_word_synonym() {
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let tokenizer = Box::new(WhitespaceTokenizer);
        let builder = SynonymGraphBuilder::with_tokenizer(dict, tokenizer, true);

        let original_tokens = vec![Token::new("ml", 0)];
        let synonyms = vec!["machine learning".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find "machine" token (first of multi-word synonym)
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some());
        let machine = machine_token.unwrap();
        assert_eq!(machine.position_increment, 0);
        assert_eq!(machine.position_length, 1);

        // The original "ml" spans both positions of "machine learning".
        let ml = result.iter().find(|t| t.text == "ml").unwrap();
        assert_eq!(ml.position_length, 2);

        // Find "learning" token (second of multi-word synonym)
        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(learning_token.is_some());
        let learning = learning_token.unwrap();
        assert_eq!(learning.position_increment, 1);
        assert_eq!(learning.position_length, 1);
    }

    #[test]
    fn test_try_match_synonym() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let builder = SynonymGraphBuilder::new(dict, true);

        let tokens = vec![Token::new("ml", 0), Token::new("tutorial", 1)];

        let result = builder.try_match_synonym(&tokens, 0);
        assert!(result.is_some());
        let (phrase, len, synonyms) = result.unwrap();
        assert_eq!(phrase, "ml");
        assert_eq!(len, 1);
        assert!(synonyms.contains(&"machine learning".to_string()));
    }

    #[test]
    fn test_boost_single_word_synonym() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        let builder = SynonymGraphBuilder::new(dict, true).with_boost(0.8);

        let original_tokens = vec![Token::new("big", 0)];
        let synonyms = vec!["large".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find synonym token
        let large_token = result.iter().find(|t| t.text == "large");
        assert!(large_token.is_some());
        let large = large_token.unwrap();

        // Boost should be applied: 0.8 (base) * 0.8 (multiplier) = 0.64
        assert!((large.boost - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_boost_multi_word_synonym() {
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let tokenizer = Box::new(WhitespaceTokenizer);
        let builder = SynonymGraphBuilder::with_tokenizer(dict, tokenizer, true).with_boost(0.8);

        let original_tokens = vec![Token::new("ml", 0)];
        let synonyms = vec!["machine learning".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find "machine" token (first of multi-word synonym)
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some());
        let machine = machine_token.unwrap();

        // First token boost: 0.9 (base) * 0.8 (multiplier) = 0.72
        assert!((machine.boost - 0.72).abs() < 0.001);

        // Find "learning" token (second of multi-word synonym)
        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(learning_token.is_some());
        let learning = learning_token.unwrap();

        // Second token boost: 0.8 (base) * 0.8 (multiplier) = 0.64
        assert!((learning.boost - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_boost_not_applied_when_not_configured() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        // No boost configured
        let builder = SynonymGraphBuilder::new(dict, true);

        let original_tokens = vec![Token::new("big", 0)];
        let synonyms = vec!["large".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find synonym token
        let large_token = result.iter().find(|t| t.text == "large");
        assert!(large_token.is_some());
        let large = large_token.unwrap();

        // Default boost should be 1.0
        assert_eq!(large.boost, 1.0);
    }
}
