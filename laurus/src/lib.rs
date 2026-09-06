//! # Laurus
//!
//! Laurus is a Rust library for building search engines with support for lexical,
//! vector, and hybrid search. It provides a flexible and efficient way to create
//! powerful search applications.
//!
//! ## Features
//!
//! ### Lexical Search (BM25-based Inverted Index)
//! - BM25 scoring over an inverted index
//! - Multiple query types (term, phrase, boolean, fuzzy, range, prefix, wildcard, etc.)
//! - Field-level boosting
//!
//! ### Vector Search (HNSW, Flat, IVF Indexes)
//! - HNSW index for fast approximate nearest-neighbor lookup
//! - Flat index for exact brute-force search
//! - IVF index for inverted-file based approximate search
//! - Configurable distance metrics and quantization
//!
//! ### Hybrid Search with Configurable Fusion (RRF, WeightedSum)
//! - Combine lexical and vector results with a configurable fusion algorithm
//! - Reciprocal Rank Fusion (RRF) for rank-based merging
//! - Weighted Sum fusion with automatic min-max score normalization
//! - Unified query DSL that mixes lexical and vector clauses in a single string
//!
//! ### Embedding Integration (Local BERT/CLIP via candle, OpenAI API)
//! - Local text embeddings via candle (BERT models)
//! - OpenAI API embeddings
//! - Multimodal CLIP embeddings (text + image)
//! - Per-field embedder routing via [`PerFieldEmbedder`]
//!
//! ### Write-Ahead Log (WAL) for Durability
//! - WAL-backed durability with crash recovery
//! - Automatic replay of uncommitted changes on engine startup
//! - Rollback on partial indexing failures to maintain consistency
//!
//! ### Pluggable Storage (In-memory, File-based with mmap)
//! - In-memory storage ([`MemoryStorage`](storage::memory::MemoryStorage)) for testing and ephemeral indexes
//! - File-based storage ([`FileStorage`](storage::file::FileStorage)) with mmap-backed reads
//! - Prefixed storage for logical partitioning within a single backend
//!
//! ### Text Analysis Pipeline (Tokenizers, Filters, Char Filters, Synonyms)
//! - Pluggable analyzers: Standard, Simple, Keyword, per-field, and language-specific (English, Japanese)
//! - Tokenizers: Unicode word, whitespace, regex, n-gram, Lindera (Japanese morphological analysis)
//! - Token filters: lowercase, stop words, stemming (Porter, simple), synonym graph, boost, strip, limit
//! - Char filters: Unicode normalization, character mapping, pattern replace, Japanese iteration marks
//! - Custom analysis pipelines via [`PipelineAnalyzer`](analysis::analyzer::pipeline::PipelineAnalyzer)
//!
//! ### Column Storage for Filtering
//! - Column-oriented storage for efficient filtering and range queries on scalar fields
//!
//! ### Spelling Correction / "Did You Mean?" Suggestions
//! - Levenshtein-distance based spelling correction
//! - "Did you mean?" suggestions powered by index term frequencies
//! - Configurable auto-correction thresholds and maximum edit distances
//! - Query history learning for improved suggestion quality

// Core modules
pub mod analysis;
mod data;
pub mod embedding;
mod engine;
mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod index_dir;
pub mod lexical;
mod maintenance;
pub mod spelling;
pub mod storage;
pub mod store;
pub mod util;
pub mod vector;

// Re-exports for the public API
pub use analysis::analyzer::analyzer::Analyzer;
pub use data::{DataValue, Document, GeoEcefPoint, GeoPoint};
#[cfg(feature = "embeddings-candle")]
pub use embedding::candle_bert_embedder::CandleBertEmbedder;
#[cfg(feature = "embeddings-multimodal")]
pub use embedding::candle_clip_embedder::CandleClipEmbedder;
pub use embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
#[cfg(feature = "embeddings-openai")]
pub use embedding::openai_embedder::OpenAIEmbedder;
pub use embedding::per_field::PerFieldEmbedder;
pub use embedding::precomputed::PrecomputedEmbedder;
pub use engine::CommitPolicy;
pub use engine::Engine;
pub use engine::EngineBuilder;
pub use engine::EngineStats;
pub use engine::PqCodebookInfo;
pub use engine::SchemaPersistHook;
pub use engine::json_document::json_to_document;
pub use engine::query::UnifiedQueryParser;
pub use engine::schema::analyzer::{
    AnalyzerDefinition, AnalyzerSpec, BuiltinAnalyzerSpec, CharFilterConfig, TokenFilterConfig,
    TokenizerConfig,
};
pub use engine::schema::embedder::EmbedderDefinition;
pub use engine::schema::{DynamicFieldPolicy, FieldOption, Schema};
pub use engine::search::{
    FusionAlgorithm, HybridMode, LexicalSearchOptions, SearchQuery, SearchRequest,
    SearchRequestBuilder, SearchResult, VectorSearchOptions, VectorSearchQuery,
};
pub use engine::type_inference::{InferredValue, infer_from_json, infer_option_from_data_value};
pub use error::{LaurusError, Result};
pub use lexical::core::field::{
    BooleanOption, BytesOption, DateTimeOption, FloatOption, Geo3dOption, GeoOption, IntegerOption,
    TextOption,
};
pub use lexical::search::searcher::{
    LexicalSearchParams, LexicalSearchQuery, LexicalSearchRequest, SortField, SortOrder,
};
pub use maintenance::deletion::DeletionConfig;
pub use storage::{Storage, StorageConfig, StorageFactory};
pub use store::log::{DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
pub use vector::core::distance::DistanceMetric;
pub use vector::core::field::{FlatOption, HnswOption, IvfOption};
pub use vector::core::quantization::QuantizationMethod;
pub use vector::core::rerank::RerankStorageKind;
pub use vector::store::request::{
    QueryPayload, QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};

/// The crate version string, populated at compile time from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
