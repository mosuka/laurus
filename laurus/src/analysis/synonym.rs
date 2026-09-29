//! Core synonym functionality shared across token filters and query expansion.
//!
//! This module provides the fundamental building blocks for synonym handling
//! in text analysis. Synonyms allow matching semantically equivalent terms
//! (e.g., "quick" and "fast") to improve search recall.
//!
//! # Components
//!
//! - [`dictionary`] - Synonym dictionary management and parsing
//! - [`graph_builder`] - Token graph construction with synonym paths
//!
//! # Token Graph Concept
//!
//! When synonyms are inserted into a token stream, they create a graph structure
//! where multiple tokens can occupy the same position. For example:
//!
//! ```text
//! Input: "quick brown fox"
//! With synonym: quick → fast
//!
//! Token Graph:
//!   Position 0: "quick" ──┐
//!                         ├──> Position 1: "brown" -> Position 2: "fox"
//!   Position 0: "fast"  ──┘
//! ```
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::synonym::dictionary::SynonymDictionary;
//!
//! let mut dict = SynonymDictionary::new(None).unwrap();
//! dict.add_synonym_group(vec!["quick".to_string(), "fast".to_string(), "rapid".to_string()]);
//! dict.add_synonym_group(vec!["big".to_string(), "large".to_string(), "huge".to_string()]);
//!
//! assert!(dict.get_synonyms("quick").is_some());
//! assert!(dict.get_synonyms("big").is_some());
//! ```

pub mod dictionary;
pub mod graph_builder;
