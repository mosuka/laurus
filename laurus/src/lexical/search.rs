//! Full-text search execution and result processing.
//!
//! This module handles all search execution and result processing:
//! - Query execution
//! - Result collection and processing
//! - Faceting and aggregation
//! - Highlighting and spell correction
//!
//! # Module Structure
//!
//! - `features`: Search features (faceting, highlighting, spell correction)
//! - `searcher`: Query execution
//!
//! Note: BM25 / similarity scoring lives in [`crate::lexical::query::scorer`]
//! (the production path used by [`crate::lexical::index::inverted::searcher`])
//! — the `scoring` submodule that previously lived here was a parallel
//! plug-in scoring API that no production caller invoked. `result_processor`
//! (a similarly unreachable plug-in result-shaping API) was removed for the
//! same reason once `Engine::search` grew its own highlighting (#1134).

pub mod features;
pub mod searcher;
