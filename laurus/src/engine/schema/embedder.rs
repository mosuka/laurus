//! Configuration types for embedder definitions within a schema.
//!
//! These types allow users to declaratively define embedding models
//! in the schema's `embedders` section. Each definition is referenced
//! by name from vector field options (e.g. `HnswOption::embedder`).
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "type": "candle_bert",
//!   "model": "sentence-transformers/all-MiniLM-L6-v2"
//! }
//! ```

use serde::{Deserialize, Serialize};

/// A declarative embedder definition stored in the schema.
///
/// Each variant maps to a concrete [`Embedder`](crate::embedding::embedder::Embedder)
/// implementation. The `type` tag selects the variant; additional fields
/// provide type-specific configuration.
///
/// # API Key Handling
///
/// For embedders that require API keys (e.g. OpenAI), the key is read
/// from an environment variable at engine initialization time, **not**
/// stored in the schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbedderDefinition {
    /// Pre-computed vectors — no embedding is performed.
    /// Use this when vectors are computed externally and passed directly.
    Precomputed,

    /// Candle-based BERT embedder for text embedding.
    /// Requires the `embeddings-candle` feature.
    CandleBert {
        /// HuggingFace model ID
        /// (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
        model: String,
    },

    /// Candle-based CLIP multimodal embedder for text and image embedding.
    /// Requires the `embeddings-multimodal` feature.
    CandleClip {
        /// HuggingFace model ID
        /// (e.g. `"openai/clip-vit-base-patch32"`).
        model: String,
    },

    /// OpenAI API embedder for text embedding.
    /// Requires the `embeddings-openai` feature.
    /// The API key is read from the `OPENAI_API_KEY` environment variable.
    Openai {
        /// OpenAI model name (e.g. `"text-embedding-3-small"`).
        model: String,
    },

    /// Candle-based ColBERT embedder producing one vector per token, for a
    /// multi-vector field (Issue #1349).
    /// Requires the `embeddings-candle` feature.
    ///
    /// Unset lengths come from the checkpoint's `artifact.metadata`, then
    /// from colbert-ai's defaults (32 and 220).
    CandleColbert {
        /// HuggingFace model ID of a BERT-based ColBERT checkpoint
        /// (e.g. `"colbert-ir/colbertv2.0"`).
        model: String,
        /// Branch, tag or commit to download. Pin a commit so that
        /// re-embedding (e.g. replaying the write-ahead log) reproduces the
        /// same vectors.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<String>,
        /// Number of tokens every query is padded or truncated to.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query_maxlen: Option<usize>,
        /// Maximum number of tokens of a document.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        doc_maxlen: Option<usize>,
    },
}

/// What an embedder definition produces, which decides the fields it can
/// serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderOutput {
    /// Nothing: documents supply the vectors. Fits any vector field.
    Precomputed,
    /// One vector per input, for a single-vector (HNSW, Flat, IVF) field.
    Vector,
    /// One vector per token, for a multi-vector field.
    TokenVectors,
}

impl EmbedderDefinition {
    /// What embedders built from this definition produce.
    pub fn output(&self) -> EmbedderOutput {
        match self {
            Self::Precomputed => EmbedderOutput::Precomputed,
            Self::CandleBert { .. } | Self::CandleClip { .. } | Self::Openai { .. } => {
                EmbedderOutput::Vector
            }
            Self::CandleColbert { .. } => EmbedderOutput::TokenVectors,
        }
    }

    /// The `type` tag of this definition (e.g. `"candle_colbert"`).
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Precomputed => "precomputed",
            Self::CandleBert { .. } => "candle_bert",
            Self::CandleClip { .. } => "candle_clip",
            Self::Openai { .. } => "openai",
            Self::CandleColbert { .. } => "candle_colbert",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precomputed_serde_roundtrip() {
        let json = r#"{"type": "precomputed"}"#;
        let def: EmbedderDefinition = serde_json::from_str(json).unwrap();
        assert!(matches!(def, EmbedderDefinition::Precomputed));
        let serialized = serde_json::to_string(&def).unwrap();
        let _roundtrip: EmbedderDefinition = serde_json::from_str(&serialized).unwrap();
    }

    #[test]
    fn test_candle_bert_serde_roundtrip() {
        let json = r#"{"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}"#;
        let def: EmbedderDefinition = serde_json::from_str(json).unwrap();
        if let EmbedderDefinition::CandleBert { model } = &def {
            assert_eq!(model, "sentence-transformers/all-MiniLM-L6-v2");
        } else {
            panic!("Expected CandleBert");
        }
        let serialized = serde_json::to_string(&def).unwrap();
        let _roundtrip: EmbedderDefinition = serde_json::from_str(&serialized).unwrap();
    }

    #[test]
    fn test_candle_clip_serde_roundtrip() {
        let json = r#"{"type": "candle_clip", "model": "openai/clip-vit-base-patch32"}"#;
        let def: EmbedderDefinition = serde_json::from_str(json).unwrap();
        assert!(matches!(def, EmbedderDefinition::CandleClip { .. }));
    }

    #[test]
    fn test_candle_colbert_serde_roundtrip() {
        let json = r#"{"type": "candle_colbert", "model": "colbert-ir/colbertv2.0"}"#;
        let def: EmbedderDefinition = serde_json::from_str(json).unwrap();
        assert!(matches!(
            &def,
            EmbedderDefinition::CandleColbert {
                model,
                revision: None,
                query_maxlen: None,
                doc_maxlen: None,
            } if model == "colbert-ir/colbertv2.0"
        ));
        assert_eq!(def.output(), EmbedderOutput::TokenVectors);
        // Unset options are left out when serialized.
        assert_eq!(serde_json::to_string(&def).unwrap(), json.replace(' ', ""));

        let toml = r#"
            type = "candle_colbert"
            model = "answerdotai/answerai-colbert-small-v1"
            revision = "abc123"
            query_maxlen = 32
            doc_maxlen = 300
        "#;
        let def: EmbedderDefinition = toml::from_str(toml).unwrap();
        assert!(matches!(
            def,
            EmbedderDefinition::CandleColbert {
                revision: Some(_),
                query_maxlen: Some(32),
                doc_maxlen: Some(300),
                ..
            }
        ));
    }

    #[test]
    fn test_output_kinds() {
        assert_eq!(
            EmbedderDefinition::Precomputed.output(),
            EmbedderOutput::Precomputed
        );
        let bert = EmbedderDefinition::CandleBert {
            model: "m".to_string(),
        };
        assert_eq!(bert.output(), EmbedderOutput::Vector);
        assert_eq!(bert.type_name(), "candle_bert");
    }

    #[test]
    fn test_openai_serde_roundtrip() {
        let json = r#"{"type": "openai", "model": "text-embedding-3-small"}"#;
        let def: EmbedderDefinition = serde_json::from_str(json).unwrap();
        if let EmbedderDefinition::Openai { model } = &def {
            assert_eq!(model, "text-embedding-3-small");
        } else {
            panic!("Expected Openai");
        }
    }
}
