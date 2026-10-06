//! Vector search implementation using approximate nearest neighbor algorithms.
//!
//! This module provides vector (semantic) search functionality through
//! various index structures (Flat, HNSW, IVF), supporting cosine similarity,
//! Euclidean distance, and other distance metrics.
//!
//! # Module Structure
//!
//! - `core`: Core data structures (vector, distance, quantization)
//! - `index`: Index management (config, factory, traits, flat, hnsw, ivf)
//! - `search`: Search execution (similarity, ranking, result processing)
//! - `store`: High-level store interface
//! - `writer`: Index writer trait

// Internal modules
pub mod core;
pub mod index;
pub mod query;
pub mod search;

pub mod reader;
pub mod store;
pub mod writer;

// Re-exports
pub use core::distance::DistanceMetric;
pub use core::field::{FieldOption, FlatOption, HnswOption, IvfOption, MultiVectorOption};
pub use core::multi_vector::MultiVectorStorage;
pub use core::vector::{StoredVector, Vector};
pub use index::config::FlatIndexConfig;
pub use index::config::{HnswIndexConfig, IvfIndexConfig};
pub use index::flat::searcher::FlatVectorSearcher;
pub use index::flat::writer::FlatIndexWriter;
pub use index::hnsw::reader::HnswIndexReader;
pub use index::hnsw::searcher::HnswSearcher;
pub use index::hnsw::writer::HnswIndexWriter;
pub use index::ivf::searcher::IvfSearcher;
pub use query::VectorQueryParser;
pub use query::VectorSearchRequestBuilder;
pub use search::searcher::{VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher};
pub use search::searcher::{VectorSearchQuery, VectorSearchRequest};
pub use store::VectorStore;
pub use store::config::{VectorFieldConfig, VectorIndexConfig};
pub use store::request::{QueryVector, VectorScoreMode};
pub use store::response::VectorSearchResults;
pub use writer::{VectorIndexWriter, VectorIndexWriterConfig};
