//! Text and multimodal embedding support for Laurus vector search.
//!
//! - Core traits: `Embedder`, `PerFieldEmbedder`, `PrecomputedEmbedder`
//! - Feature flags: `embeddings-candle`, `embeddings-openai`, `embeddings-multimodal`, `embeddings-all`
//! - Vector dimensions must be specified explicitly in field definitions and are not inferred from the embedder
//!
// Unified embedder trait (analogous to Analyzer in lexical module)
pub mod embedder;

// Per-field embedder support (analogous to PerFieldAnalyzer)
pub mod per_field;

// Query-time embedding cache shared by the engine and the vector query
// parser (Issue #678).
pub mod cache;

// Embedder for pre-computed vectors (analogous to NoOpAnalyzer)
pub mod precomputed;

// Embedder registry for creating embedders from schema definitions
pub mod registry;

// Hugging Face Hub access shared by the candle embedders
#[cfg(any(feature = "embeddings-candle", feature = "embeddings-multimodal"))]
pub(crate) mod candle_hub;

// Candle implementation (requires feature flag)
#[cfg(feature = "embeddings-candle")]
pub mod candle_bert_embedder;

// ColBERT token-level embedder on candle (requires feature flag)
#[cfg(feature = "embeddings-candle")]
pub mod candle_colbert_embedder;

// OpenAI implementation (requires feature flag)
#[cfg(feature = "embeddings-openai")]
pub mod openai_embedder;

// Multimodal embedding (requires embeddings-multimodal feature flag)
#[cfg(feature = "embeddings-multimodal")]
pub mod candle_clip_embedder;
