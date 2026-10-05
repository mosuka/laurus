//! Candle-based BERT embedder implementation.
//!
//! This module provides a text embedder using HuggingFace Candle framework.
//! Requires the `embeddings-candle` feature to be enabled.

#[cfg(feature = "embeddings-candle")]
use std::any::Any;

#[cfg(feature = "embeddings-candle")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "embeddings-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "embeddings-candle")]
use candle_transformers::models::bert::{BertModel, Config};
#[cfg(feature = "embeddings-candle")]
use tokenizers::Tokenizer;

#[cfg(feature = "embeddings-candle")]
use crate::embedding::candle_hub::{HubModel, legacy_cache_dir};
#[cfg(feature = "embeddings-candle")]
use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
#[cfg(feature = "embeddings-candle")]
use crate::error::{LaurusError, Result};
#[cfg(feature = "embeddings-candle")]
use crate::vector::core::vector::Vector;

/// Candle-based BERT embedder using BERT models from HuggingFace.
///
/// This embedder uses the Candle framework to run BERT models locally,
/// providing high-quality embeddings without external API dependencies.
///
/// # Features
///
/// - Offline inference (no API calls)
/// - GPU acceleration support
/// - Multiple BERT model support
/// - Fast inference with Rust performance
///
/// # Examples
///
/// ```no_run
/// use laurus::embedding::embedder::{Embedder, EmbedInput};
/// use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
///
/// # async fn example() -> laurus::Result<()> {
/// // Create embedder with a sentence-transformers model
/// let embedder = CandleBertEmbedder::new(
///     "sentence-transformers/all-MiniLM-L6-v2"
/// )?;
///
/// // Generate embedding
/// let vector = embedder.embed(&EmbedInput::Text("Rust is awesome!")).await?;
///
/// // Batch processing
/// let inputs = vec![EmbedInput::Text("Hello"), EmbedInput::Text("World")];
/// let vectors = embedder.embed_batch(&inputs).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-candle")]
pub struct CandleBertEmbedder {
    /// The BERT model for generating embeddings.
    model: BertModel,
    /// Tokenizer for converting text to token IDs.
    tokenizer: Tokenizer,
    /// Device to run the model on (CPU or GPU).
    device: Device,
    /// Dimension of the output embeddings.
    dim: usize,
    /// Name of the HuggingFace model.
    model_name: String,
}

#[cfg(feature = "embeddings-candle")]
impl std::fmt::Debug for CandleBertEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleBertEmbedder")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dim)
            .finish()
    }
}

#[cfg(feature = "embeddings-candle")]
impl CandleBertEmbedder {
    /// Create a new Candle-based BERT embedder from a HuggingFace model.
    ///
    /// The model will be automatically downloaded from HuggingFace Hub if not cached.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    ///
    /// # Returns
    ///
    /// A new `CandleBertEmbedder` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model download fails
    /// - Model loading fails
    /// - Device initialization fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
    ///
    /// # fn example() -> laurus::Result<()> {
    /// // Small and fast model
    /// let embedder = CandleBertEmbedder::new(
    ///     "sentence-transformers/all-MiniLM-L6-v2"
    /// )?;
    ///
    /// // Larger, more accurate model
    /// let embedder = CandleBertEmbedder::new(
    ///     "sentence-transformers/all-mpnet-base-v2"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_name: &str) -> Result<Self> {
        // Setup device (prefer GPU if available)
        let device = Device::cuda_if_available(0)
            .map_err(|e| LaurusError::InvalidOperation(format!("Device setup failed: {}", e)))?;

        // Download model from HuggingFace Hub
        let repo = HubModel::open(model_name, None, Some(legacy_cache_dir()))?;

        // Load config
        let config_filename = repo.file("config.json")?;
        let config_str = std::fs::read_to_string(config_filename)
            .map_err(|e| LaurusError::InvalidOperation(format!("Config read failed: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| LaurusError::InvalidOperation(format!("Config parse failed: {}", e)))?;

        // Load weights
        let weights_filename = repo.file("model.safetensors")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device).map_err(
                |e| LaurusError::InvalidOperation(format!("VarBuilder creation failed: {}", e)),
            )?
        };

        // Load model
        let model = BertModel::load(vb, &config)
            .map_err(|e| LaurusError::InvalidOperation(format!("Model load failed: {}", e)))?;

        // Load tokenizer
        let tokenizer_filename = repo.file("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenizer load failed: {}", e)))?;

        let dim = config.hidden_size;

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            model_name: model_name.to_string(),
        })
    }

    /// Embed text directly (internal implementation).
    ///
    /// The candle inference pipeline is synchronous.  We use `block_in_place`
    /// so the tokio runtime can schedule other tasks on other threads while
    /// this thread is blocked on CPU-bound model inference.
    async fn embed_text(&self, text: &str) -> Result<Vector> {
        let text = text.to_string();
        tokio::task::block_in_place(|| self.embed_text_sync(&text))
    }

    /// Synchronous embedding implementation.
    fn embed_text_sync(&self, text: &str) -> Result<Vector> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenization failed: {}", e)))?;

        let token_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert to tensors
        let token_ids_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tensor creation failed: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Forward pass
        let embeddings = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor, None)
            .map_err(|e| LaurusError::InvalidOperation(format!("Model forward failed: {}", e)))?;

        // Mean pooling
        let pooled = self.mean_pool(&embeddings, &attention_mask_tensor)?;

        // Normalize (L2 normalization)
        let norm = pooled
            .sqr()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .sum_all()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .sqrt()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .to_scalar::<f32>()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Divide by norm to normalize
        let normalized = pooled
            .affine((1.0 / norm) as f64, 0.0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Convert to Vector
        let vector_data: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .to_vec1()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        Ok(Vector::new(vector_data))
    }

    /// Perform mean pooling over token embeddings.
    fn mean_pool(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match embedding dimensions
        let mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .expand(embeddings.shape())
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .to_dtype(embeddings.dtype())
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Multiply embeddings by mask
        let masked_embeddings = embeddings
            .mul(&mask_expanded)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Sum across sequence dimension
        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Sum mask values
        let sum_mask = mask_expanded
            .sum(1)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Divide to get mean
        let mean = sum_embeddings
            .div(&sum_mask)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        Ok(mean)
    }
}

#[cfg(feature = "embeddings-candle")]
#[async_trait]
impl Embedder for CandleBertEmbedder {
    /// Generate an embedding vector for the given input.
    ///
    /// Only text input is supported. Image input will return an error.
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => self.embed_text(text).await,
            _ => Err(LaurusError::invalid_argument(
                "CandleBertEmbedder only supports text input",
            )),
        }
    }

    /// Get the supported input types.
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    /// Get the name/identifier of this embedder.
    fn name(&self) -> &str {
        &self.model_name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
