//! Unified embedder trait for vector indexing.
//!
//! This module provides the `Embedder` trait, which serves as a unified interface
//! for embedders used with `VectorIndex`. It is analogous to how `Analyzer` is
//! used with `LexicalIndex` in the lexical module.
//!
//! # Design
//!
//! The `Embedder` trait abstracts over different embedding strategies:
//! - Text-only embedders (e.g., BERT, sentence-transformers)
//! - Image-only embedders
//! - Multimodal embedders (e.g., CLIP) that handle both text and images
//! - Per-field embedders that route to different underlying embedders based on field name
//!
//! # Symmetry with Analyzer
//!
//! This design mirrors `Analyzer` in the lexical module:
//!
//! | Lexical | Vector |
//! |---------|--------|
//! | `Analyzer` | `Embedder` |
//! | `PerFieldAnalyzer` | `PerFieldEmbedder` |
//! | `NoOpAnalyzer` | `NoOpEmbedder` |
//! | `analyze(text) -> TokenStream` | `embed(input) -> Vector` |
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use laurus::embedding::embedder::{Embedder, EmbedInput};
//! use laurus::embedding::per_field::PerFieldEmbedder;
//! use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
//! use laurus::embedding::precomputed::PrecomputedEmbedder;
//! use std::sync::Arc;
//!
//! # async fn example() -> laurus::Result<()> {
//! let text_embedder = Arc::new(
//!     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! // Document-level embedder (common case)
//! let embedder: Arc<dyn Embedder> = Arc::new(PrecomputedEmbedder::new());
//!
//! // Embed text
//! let vector = embedder.embed(&EmbedInput::Text("Hello, world!")).await?;
//! # Ok(())
//! # }
//! # }
//! ```

use std::any::Any;
use std::fmt::Debug;

use async_trait::async_trait;

use crate::error::Result;
use crate::vector::core::vector::Vector;

/// Input types for embedding operations.
///
/// This enum represents the different types of input that can be embedded.
/// Embedders declare which input types they support via `supported_input_types()`.
#[derive(Debug, Clone)]
pub enum EmbedInput<'a> {
    /// Text input for text embedding.
    Text(&'a str),

    /// Raw bytes for embedding.
    /// The optional string is a MIME type hint (e.g., "image/png", "text/plain").
    /// When the MIME type is `None` or does not start with `"text/"`,
    /// [`input_type()`](EmbedInput::input_type) defaults to [`EmbedInputType::Image`].
    Bytes(&'a [u8], Option<&'a str>),
}

impl<'a> EmbedInput<'a> {
    /// Get the input type of this input.
    pub fn input_type(&self) -> EmbedInputType {
        match self {
            EmbedInput::Text(_) => EmbedInputType::Text,
            EmbedInput::Bytes(_, mime) => {
                if let Some(mime) = mime
                    && mime.starts_with("text/")
                {
                    return EmbedInputType::Text;
                }
                EmbedInputType::Image
            }
        }
    }

    /// Check if this is a text input.
    pub fn is_text(&self) -> bool {
        match self {
            EmbedInput::Text(_) => true,
            EmbedInput::Bytes(_, mime) => mime.is_some_and(|m| m.starts_with("text/")),
        }
    }

    /// Check if this is an image input.
    pub fn is_image(&self) -> bool {
        match self {
            EmbedInput::Bytes(_, Some(mime)) => mime.starts_with("image/"),
            _ => false,
        }
    }

    /// Get the text content if this is a text input.
    pub fn as_text(&self) -> Option<&'a str> {
        match self {
            EmbedInput::Text(text) => Some(text),
            _ => None,
        }
    }
}

/// Types of input that an embedder can support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedInputType {
    /// Text input support.
    Text,

    /// Image input support.
    Image,
}

/// The side of a retrieval pair an input is embedded for.
///
/// Late-interaction models such as ColBERT encode queries and documents
/// differently (different marker tokens, query padding), so a
/// [`TokenEmbedder`] needs to know which one it is embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedRole {
    /// A search query.
    Query,
    /// A document being indexed.
    Document,
}

/// Token-level (multi-vector) embedding for late-interaction models such as
/// ColBERT (Issue #1349).
///
/// An input becomes one vector per token instead of one vector in total.
/// Embedders that can do this expose it through
/// [`Embedder::as_token_embedder`]; a multi-vector field embeds text only
/// through this trait, never by wrapping [`Embedder::embed`], so a
/// single-vector model on such a field is an error rather than a silent
/// one-token document.
#[async_trait]
pub trait TokenEmbedder: Send + Sync + Debug {
    /// Embed every input into its token vectors.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The inputs to embed.
    /// * `role` - Whether the inputs are queries or documents.
    ///
    /// # Returns
    ///
    /// One list of token vectors per input, in input order. Every vector
    /// has [`token_dimension`](Self::token_dimension) elements.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`](crate::error::LaurusError::InvalidArgument)
    /// for an input the embedder cannot handle, and another error kind when
    /// the model itself fails.
    async fn embed_tokens(
        &self,
        inputs: &[EmbedInput<'_>],
        role: EmbedRole,
    ) -> Result<Vec<Vec<Vector>>>;

    /// Dimension of every token vector this embedder produces.
    fn token_dimension(&self) -> usize;
}

/// Unified embedder trait for vector indexing.
///
/// This trait provides a common interface for embedders that can be used
/// with `VectorIndex`, similar to how `Analyzer` is used with `LexicalIndex`.
///
/// # Supported Input Types
///
/// Embedders declare which input types they support:
/// - Text-only embedders: `[EmbedInputType::Text]`
/// - Image-only embedders: `[EmbedInputType::Image]`
/// - Multimodal embedders (e.g., CLIP): `[EmbedInputType::Text, EmbedInputType::Image]`
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent embedding
/// operations across multiple threads.
///
/// # Example
///
/// ```
/// use async_trait::async_trait;
/// use laurus::embedding::embedder::{Embedder, EmbedInput, EmbedInputType};
/// use laurus::{LaurusError, Result};
/// use laurus::vector::core::vector::Vector;
///
/// #[derive(Debug)]
/// struct MyTextEmbedder {
///     dimension: usize,
/// }
///
/// #[async_trait]
/// impl Embedder for MyTextEmbedder {
///     async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
///         match input {
///             EmbedInput::Text(text) => {
///                 // Generate embedding from text
///                 Ok(Vector::new(vec![0.0; self.dimension]))
///             }
///             _ => Err(LaurusError::invalid_argument(
///                 "this embedder only supports text input"
///             )),
///         }
///     }
///     fn supported_input_types(&self) -> Vec<EmbedInputType> {
///         vec![EmbedInputType::Text]
///     }
///
///     fn name(&self) -> &str {
///         "my-text-embedder"
///     }
///
///     fn as_any(&self) -> &dyn std::any::Any {
///         self
///     }
/// }
/// ```
#[async_trait]
pub trait Embedder: Send + Sync + Debug {
    /// Generate an embedding vector for the given input.
    ///
    /// The supported input variants are:
    /// - [`EmbedInput::Text`] - a text string to embed.
    /// - [`EmbedInput::Bytes`] - raw bytes with an optional MIME type hint.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to embed. See [`EmbedInput`] for the available
    ///   variants.
    ///
    /// # Returns
    ///
    /// A vector representation of the input, or an error if the input type
    /// is not supported or embedding fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input type is not supported by this embedder
    /// - The embedding operation fails (e.g., model error)
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector>;

    /// Generate embeddings for multiple inputs in batch.
    ///
    /// The default implementation calls `embed` sequentially.
    /// Override this method for better performance with batch processing.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of inputs to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input
    async fn embed_batch(&self, inputs: &[EmbedInput<'_>]) -> Result<Vec<Vector>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.embed(input).await?);
        }
        Ok(results)
    }

    /// Get the input types supported by this embedder.
    ///
    /// # Returns
    ///
    /// A vector of supported input types
    fn supported_input_types(&self) -> Vec<EmbedInputType>;

    /// Check if this embedder supports the given input type.
    ///
    /// # Arguments
    ///
    /// * `input_type` - The input type to check
    ///
    /// # Returns
    ///
    /// `true` if the embedder supports this input type
    fn supports(&self, input_type: EmbedInputType) -> bool {
        self.supported_input_types().contains(&input_type)
    }

    /// Check if this embedder supports text input.
    fn supports_text(&self) -> bool {
        self.supports(EmbedInputType::Text)
    }

    /// Check if this embedder supports image input.
    fn supports_image(&self) -> bool {
        self.supports(EmbedInputType::Image)
    }

    /// Check if this embedder is multimodal (supports both text and image).
    fn is_multimodal(&self) -> bool {
        self.supports_text() && self.supports_image()
    }

    /// Get the name/identifier of this embedder.
    ///
    /// This is useful for logging and debugging purposes.
    ///
    /// # Returns
    ///
    /// A string identifying the embedder (e.g., model name)
    fn name(&self) -> &str {
        "unknown"
    }

    /// The token-level embedding capability of this embedder, if it has one.
    ///
    /// Defaults to `None`. A late-interaction embedder returns `Some(self)`;
    /// see [`TokenEmbedder`].
    fn as_token_embedder(&self) -> Option<&dyn TokenEmbedder> {
        None
    }

    /// Downcast support for runtime type identification.
    ///
    /// This enables downcasting to concrete types when needed.
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use crate::error::LaurusError;

    use super::*;

    #[derive(Debug)]
    struct MockTextEmbedder {
        dimension: usize,
    }

    #[async_trait]
    impl Embedder for MockTextEmbedder {
        async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
            match input {
                EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
                _ => Err(LaurusError::invalid_argument(
                    "this embedder only supports text input",
                )),
            }
        }

        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }

        fn name(&self) -> &str {
            "mock-text"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug)]
    struct MockMultimodalEmbedder;

    #[async_trait]
    impl Embedder for MockMultimodalEmbedder {
        async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
            match input {
                EmbedInput::Text(_) | EmbedInput::Bytes(_, _) => Ok(Vector::new(vec![0.0; 3])),
            }
        }

        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text, EmbedInputType::Image]
        }

        fn name(&self) -> &str {
            "mock-multimodal"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_embed_input_type() {
        assert_eq!(EmbedInput::Text("hello").input_type(), EmbedInputType::Text);
        assert_eq!(
            EmbedInput::Bytes(&[0, 1, 2], None).input_type(),
            EmbedInputType::Image
        );
        assert_eq!(
            EmbedInput::Bytes(&[0, 1, 2], Some("text/plain")).input_type(),
            EmbedInputType::Text
        );
    }

    #[test]
    fn test_text_embedder_supports() {
        let embedder = MockTextEmbedder { dimension: 384 };

        assert!(embedder.supports_text());
        assert!(!embedder.supports_image());
        assert!(!embedder.is_multimodal());
    }

    #[test]
    fn test_multimodal_embedder_supports() {
        let embedder = MockMultimodalEmbedder;

        assert!(embedder.supports_text());
        assert!(embedder.supports_image());
        assert!(embedder.is_multimodal());
    }

    #[tokio::test]
    async fn test_text_embedder_embed() {
        let embedder = MockTextEmbedder { dimension: 384 };

        // Text input should work
        let result = embedder.embed(&EmbedInput::Text("hello")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.len(), 384);

        // Image input should fail
        let result = embedder.embed(&EmbedInput::Bytes(&[], None)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_multimodal_embedder_embed() {
        let embedder = MockMultimodalEmbedder;

        // Both text and image should work
        let text_result = embedder.embed(&EmbedInput::Text("hello")).await;
        assert!(text_result.is_ok());

        let image_result = embedder.embed(&EmbedInput::Bytes(&[], None)).await;
        assert!(image_result.is_ok());
    }
}
