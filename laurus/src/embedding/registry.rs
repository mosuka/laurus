//! Embedder registry for creating embedders from schema definitions.
//!
//! Maps [`EmbedderDefinition`] variants to concrete [`Embedder`] instances.
//! Feature-gated embedders return clear error messages when the required
//! feature is not enabled.

use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::embedding::precomputed::PrecomputedEmbedder;
use crate::engine::schema::embedder::EmbedderDefinition;
use crate::error::{LaurusError, Result};

/// Create an embedder from a schema definition.
///
/// Constructs the appropriate [`Embedder`] implementation based on the
/// given [`EmbedderDefinition`]. For OpenAI, the API key is read from
/// the `OPENAI_API_KEY` environment variable.
///
/// # Arguments
///
/// * `name` - A label for the embedder (used in error messages).
/// * `definition` - The embedder definition from the schema.
///
/// # Returns
///
/// An `Arc<dyn Embedder>` wrapping the constructed embedder.
///
/// # Errors
///
/// Returns an error if:
/// - The required feature is not compiled in.
/// - Model initialization fails (e.g. download failure, invalid model name).
/// - The `OPENAI_API_KEY` environment variable is not set (for OpenAI).
pub async fn create_embedder_from_definition(
    _name: &str,
    definition: &EmbedderDefinition,
) -> Result<Arc<dyn Embedder>> {
    match definition {
        EmbedderDefinition::Precomputed => Ok(Arc::new(PrecomputedEmbedder::new())),

        #[cfg(feature = "embeddings-candle")]
        EmbedderDefinition::CandleBert { model } => {
            use crate::embedding::candle_bert_embedder::CandleBertEmbedder;
            let embedder = CandleBertEmbedder::new(model)?;
            Ok(Arc::new(embedder))
        }
        #[cfg(not(feature = "embeddings-candle"))]
        EmbedderDefinition::CandleBert { .. } => Err(LaurusError::not_implemented(
            "candle_bert embedder requires the 'embeddings-candle' feature to be enabled",
        )),

        #[cfg(feature = "embeddings-multimodal")]
        EmbedderDefinition::CandleClip { model } => {
            use crate::embedding::candle_clip_embedder::CandleClipEmbedder;
            let embedder = CandleClipEmbedder::new(model)?;
            Ok(Arc::new(embedder))
        }
        #[cfg(not(feature = "embeddings-multimodal"))]
        EmbedderDefinition::CandleClip { .. } => Err(LaurusError::not_implemented(
            "candle_clip embedder requires the 'embeddings-multimodal' feature to be enabled",
        )),

        #[cfg(feature = "embeddings-openai")]
        EmbedderDefinition::Openai { model } => {
            use crate::embedding::openai_embedder::OpenAIEmbedder;
            let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                LaurusError::invalid_config(
                    "OpenAI embedder requires the OPENAI_API_KEY environment variable to be set",
                )
            })?;
            let embedder = OpenAIEmbedder::new(api_key, model.clone()).await?;
            Ok(Arc::new(embedder))
        }
        #[cfg(not(feature = "embeddings-openai"))]
        EmbedderDefinition::Openai { .. } => Err(LaurusError::not_implemented(
            "openai embedder requires the 'embeddings-openai' feature to be enabled",
        )),

        #[cfg(feature = "embeddings-candle")]
        EmbedderDefinition::CandleColbert {
            model,
            revision,
            query_maxlen,
            doc_maxlen,
        } => {
            use crate::embedding::candle_colbert_embedder::{
                CandleColbertEmbedder, CandleColbertOptions,
            };
            let model = model.clone();
            let options = CandleColbertOptions {
                revision: revision.clone(),
                query_maxlen: *query_maxlen,
                doc_maxlen: *doc_maxlen,
            };
            // Downloading and memory-mapping the checkpoint blocks, so keep
            // it off the async worker threads.
            let embedder = tokio::task::spawn_blocking(move || {
                CandleColbertEmbedder::with_options(&model, options)
            })
            .await
            .map_err(|e| {
                LaurusError::internal(format!("loading the ColBERT model failed: {e}"))
            })??;
            Ok(Arc::new(embedder))
        }
        #[cfg(not(feature = "embeddings-candle"))]
        EmbedderDefinition::CandleColbert { .. } => Err(LaurusError::not_implemented(
            "candle_colbert embedder requires the 'embeddings-candle' feature to be enabled",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_precomputed() {
        let def = EmbedderDefinition::Precomputed;
        let embedder = create_embedder_from_definition("test", &def).await.unwrap();
        assert_eq!(embedder.name(), "PrecomputedEmbedder");
    }

    #[tokio::test]
    async fn test_precomputed_serde_roundtrip() {
        let json = r#"{"type": "precomputed"}"#;
        let def: EmbedderDefinition = serde_json::from_str(json).unwrap();
        let embedder = create_embedder_from_definition("test", &def).await.unwrap();
        assert_eq!(embedder.name(), "PrecomputedEmbedder");
    }
}
