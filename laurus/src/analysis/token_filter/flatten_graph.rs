//! Flatten graph filter for converting token graphs to linear streams.
//!
//! This filter converts an incoming graph token stream (such as one from SynonymGraphFilter)
//! into a flat form so that all nodes form a single linear chain with no side paths.
//!
//! Laurus's index already lays a token graph out this way when it stores the
//! positions, because it does not save position_length. An engine parses
//! queries with the same analyzer, so adding this filter to it flattens query-time
//! graphs too, and a quoted value then also matches mixes of the words of a group's
//! multi-word members.

use std::collections::VecDeque;

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// Holds all tokens leaving a given input position.
#[derive(Debug, Clone)]
struct InputNode {
    /// Tokens leaving from this input position
    tokens: Vec<Token>,
    /// Our input node ID
    node: usize,
    /// Maximum destination node for all tokens leaving here
    max_to_node: usize,
    /// Minimum destination node for all tokens leaving here
    min_to_node: usize,
    /// Where we currently map to in the output
    output_node: usize,
}

impl InputNode {
    fn new(node: usize) -> Self {
        Self {
            tokens: Vec::new(),
            node,
            max_to_node: 0,
            min_to_node: usize::MAX,
            output_node: 0,
        }
    }
}

/// Gathers merged input positions into a single output position.
#[derive(Debug, Clone)]
struct OutputNode {
    /// Input node IDs that map to this output node
    input_nodes: Vec<usize>,
    /// Start offset of tokens leaving this node
    start_offset: usize,
    /// End offset of tokens arriving to this node
    end_offset: usize,
}

impl OutputNode {
    fn new(_node: usize) -> Self {
        Self {
            input_nodes: Vec::new(),
            start_offset: 0,
            end_offset: 0,
        }
    }
}

/// Flatten graph filter that converts graph token streams to linear form.
///
/// The index cannot preserve the position_length attribute, so it flattens
/// graph token streams (e.g., from SynonymGraphFilter) itself; an analyzer
/// does not need this filter for indexing. Since the same analyzer parses
/// queries, adding it there loses the graph a quoted value is matched
/// through.
pub struct FlattenGraphFilter;

impl FlattenGraphFilter {
    /// Create a new flatten graph filter.
    pub fn new() -> Self {
        Self
    }

    /// Process tokens and flatten the graph structure.
    fn flatten_tokens(&self, input_tokens: Vec<Token>) -> Result<Vec<Token>> {
        if input_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let mut input_nodes: VecDeque<InputNode> = VecDeque::new();
        let mut output_nodes: VecDeque<OutputNode> = VecDeque::new();
        let mut output_tokens = Vec::new();

        // Initialize first nodes
        input_nodes.push_back(InputNode::new(0));
        let mut output_node = OutputNode::new(0);
        output_node.input_nodes.push(0);
        output_nodes.push_back(output_node);

        let mut current_input_pos = 0;
        let mut last_start_offset = 0;

        // Process all input tokens
        for token in input_tokens {
            // Calculate input position based on position_increment
            current_input_pos += token.position_increment;

            // Ensure we have an input node for this position
            while input_nodes.len() <= current_input_pos {
                input_nodes.push_back(InputNode::new(input_nodes.len()));
            }

            // Calculate destination position
            let input_to = current_input_pos + token.position_length;

            // Ensure we have an input node for the destination
            while input_nodes.len() <= input_to {
                input_nodes.push_back(InputNode::new(input_nodes.len()));
            }

            // Process source node
            let src_output_node = {
                let src = &mut input_nodes[current_input_pos];

                // Assign output node if not yet assigned
                if src.output_node == 0 && current_input_pos > 0 {
                    src.output_node = output_nodes.len();
                    while output_nodes.len() <= src.output_node {
                        output_nodes.push_back(OutputNode::new(output_nodes.len()));
                    }
                    output_nodes[src.output_node]
                        .input_nodes
                        .push(current_input_pos);
                }

                // Update start offset
                if output_nodes[src.output_node].start_offset == 0
                    || token.start_offset > output_nodes[src.output_node].start_offset
                {
                    output_nodes[src.output_node].start_offset = token.start_offset;
                }

                // Store the token
                src.tokens.push(token.clone());
                src.max_to_node = src.max_to_node.max(input_to);
                src.min_to_node = src.min_to_node.min(input_to);

                src.output_node
            };

            // Update destination node
            let dest = &mut input_nodes[input_to];
            if dest.node == 0 && input_to > 0 {
                dest.node = input_to;
            }

            // Assign output node for destination
            let output_end_node = src_output_node + 1;
            if output_end_node > dest.output_node {
                if dest.output_node > 0 {
                    // Remove from previous output node
                    if let Some(prev_out) = output_nodes.get_mut(dest.output_node) {
                        prev_out.input_nodes.retain(|&x| x != input_to);
                    }
                }
                while output_nodes.len() <= output_end_node {
                    output_nodes.push_back(OutputNode::new(output_nodes.len()));
                }
                output_nodes[output_end_node].input_nodes.push(input_to);
                dest.output_node = output_end_node;
            }

            // Update end offset
            if output_nodes[dest.output_node].end_offset == 0
                || token.end_offset < output_nodes[dest.output_node].end_offset
            {
                output_nodes[dest.output_node].end_offset = token.end_offset;
            }
        }

        // Now generate the output tokens by traversing the flattened graph
        let mut last_output_pos = 0;

        for (out_pos, out_node) in output_nodes.iter().enumerate() {
            if out_node.input_nodes.is_empty() {
                continue;
            }

            for &input_node_id in &out_node.input_nodes {
                if input_node_id >= input_nodes.len() {
                    continue;
                }

                let input_node = &input_nodes[input_node_id];

                for token in &input_node.tokens {
                    let mut output_token = token.clone();

                    // Calculate the destination input position
                    let to_input_pos = input_node.node + token.position_length;
                    let to_output_pos = if to_input_pos < input_nodes.len() {
                        input_nodes[to_input_pos].output_node
                    } else {
                        output_nodes.len()
                    };

                    // Correct position increment
                    output_token.position_increment = out_pos.saturating_sub(last_output_pos);

                    // Correct position length
                    output_token.position_length = if to_output_pos > out_pos {
                        to_output_pos - out_pos
                    } else {
                        1
                    };

                    // Correct offsets
                    let start = last_start_offset.max(out_node.start_offset);
                    let end = if to_output_pos < output_nodes.len() {
                        start.max(output_nodes[to_output_pos].end_offset)
                    } else {
                        start.max(token.end_offset)
                    };

                    output_token.start_offset = start;
                    output_token.end_offset = end;
                    last_start_offset = start;

                    output_tokens.push(output_token);
                    last_output_pos = out_pos;
                }
            }
        }

        Ok(output_tokens)
    }
}

impl Default for FlattenGraphFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter for FlattenGraphFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let input_tokens: Vec<Token> = tokens.collect();
        let output_tokens = self.flatten_tokens(input_tokens)?;
        Ok(Box::new(output_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "flatten_graph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_graph_simple() {
        let filter = FlattenGraphFilter::new();

        // Simple token stream without graph structure
        let tokens = vec![
            Token::new("the", 0),
            Token::new("cat", 1),
            Token::new("sat", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "the");
        assert_eq!(result[1].text, "cat");
        assert_eq!(result[2].text, "sat");
    }

    #[test]
    fn test_flatten_graph_with_synonyms() {
        let filter = FlattenGraphFilter::new();

        // Token stream with synonym at position 1 (posInc=0)
        let mut tokens = vec![
            Token::new("the", 0),
            Token::new("big", 1), // Original
            Token::new("cat", 2),
        ];

        // Add synonym "large" at same position as "big"
        let mut synonym = Token::new("large", 1);
        synonym.position_increment = 0; // Same position as "big"
        synonym.position_length = 1;
        tokens.insert(2, synonym);

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have all tokens in linear form
        assert!(result.len() >= 3);
        assert_eq!(result[0].text, "the");
    }

    #[test]
    fn test_flatten_graph_multi_word_synonym() {
        let filter = FlattenGraphFilter::new();

        // Original: "new york" -> synonym: "ny" (1 token spanning 2 positions)
        let mut tokens = vec![Token::new("new", 0), Token::new("york", 1)];

        // Add single-token synonym "ny" that spans 2 positions
        let mut synonym = Token::new("ny", 0);
        synonym.position_increment = 0; // Same position as "new"
        synonym.position_length = 2; // Spans "new york"
        tokens.insert(1, synonym);

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        assert!(result.len() >= 2);
        assert_eq!(result[0].text, "new");
    }

    #[test]
    fn test_flatten_graph_empty() {
        let filter = FlattenGraphFilter::new();
        let tokens: Vec<Token> = vec![];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_integration_with_synonym_graph_filter() {
        use crate::analysis::synonym::dictionary::SynonymDictionary;
        use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;

        // Create synonym dictionary
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["machine learning".to_string(), "ml".to_string()]);

        // Create filters
        let synonym_filter = SynonymGraphFilter::new(dict, true);
        let flatten_filter = FlattenGraphFilter::new();

        // Input tokens: "machine learning is fun"
        let tokens = vec![
            Token::new("machine", 0),
            Token::new("learning", 1),
            Token::new("is", 2),
            Token::new("fun", 3),
        ];

        // Apply synonym filter (creates graph)
        let graph_tokens = synonym_filter.filter(Box::new(tokens.into_iter())).unwrap();

        // Apply flatten filter (converts to linear)
        let result = flatten_filter
            .filter(graph_tokens)
            .unwrap()
            .collect::<Vec<_>>();

        // Should have flattened the graph structure
        assert!(result.len() >= 4);

        // Verify that "ml" synonym is present
        let has_ml = result.iter().any(|t| t.text == "ml");
        assert!(has_ml, "Expected to find 'ml' synonym in flattened output");

        // Verify that original tokens are present (keep_original=true)
        let has_machine = result.iter().any(|t| t.text == "machine");
        let has_learning = result.iter().any(|t| t.text == "learning");
        assert!(
            has_machine && has_learning,
            "Expected original tokens to be preserved"
        );
    }

    /// `SynonymGraphFilter` gives each multi-word member inner nodes of its
    /// own. This filter lays that graph out on the same positions as the
    /// indexer's `flatten_token_graph`, for paths of 1, 2 and 3 words.
    #[test]
    fn synonym_graph_output_flattens_to_the_indexed_positions() {
        use crate::analysis::synonym::dictionary::SynonymDictionary;
        use crate::analysis::token::{flatten_token_graph, token_positions};
        use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "ml".to_string(),
            "machine learning".to_string(),
            "statistical machine learning".to_string(),
        ]);
        let synonym_filter =
            SynonymGraphFilter::with_tokenizer(dict, Box::new(WhitespaceTokenizer), true);

        for input in [
            vec!["ml", "is", "fun"],
            vec!["machine", "learning", "is"],
            vec!["we", "like", "statistical", "machine", "learning"],
        ] {
            let tokens: Vec<Token> = input
                .iter()
                .enumerate()
                .map(|(i, text)| Token::new(*text, i))
                .collect();
            let graph: Vec<Token> = synonym_filter
                .filter(Box::new(tokens.into_iter()))
                .unwrap()
                .collect();
            let flat: Vec<Token> = FlattenGraphFilter::new()
                .filter(Box::new(graph.clone().into_iter()))
                .unwrap()
                .collect();
            let positions = |tokens: &[Token]| -> Vec<(u32, String)> {
                let mut positions: Vec<(u32, String)> = token_positions(tokens)
                    .into_iter()
                    .zip(tokens.iter().map(|t| t.text.clone()))
                    .collect();
                positions.sort();
                positions
            };
            assert_eq!(
                positions(&flat),
                positions(&flatten_token_graph(graph)),
                "{input:?}"
            );
        }
    }
}
