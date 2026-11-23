//! OpenAI API-based text embedder implementation.
//!
//! This module provides a text embedder using OpenAI's Embeddings API.
//! Requires the `embeddings-openai` feature to be enabled.

#[cfg(feature = "embeddings-openai")]
use std::any::Any;

#[cfg(feature = "embeddings-openai")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-openai")]
use reqwest::Client;
#[cfg(feature = "embeddings-openai")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "embeddings-openai")]
use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
#[cfg(feature = "embeddings-openai")]
use crate::error::{LaurusError, Result};
#[cfg(feature = "embeddings-openai")]
use crate::vector::core::vector::Vector;

/// Request structure for OpenAI Embeddings API.
#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    /// Model identifier to use for embeddings.
    model: String,
    /// Input texts to embed (batch).
    input: Vec<String>,
    /// Optional custom dimension (only for newer models).
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// Response structure from OpenAI Embeddings API.
#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    /// List of embedding data objects.
    data: Vec<EmbeddingData>,
}

/// Individual embedding data from API response.
#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    /// The embedding vector.
    embedding: Vec<f32>,
}

/// OpenAI API-based embedder.
///
/// This embedder uses OpenAI's Embeddings API to generate high-quality
/// embeddings. Requires an API key and internet connection.
///
/// # Features
///
/// - State-of-the-art embedding quality
/// - Multiple model options
/// - Batch processing support
/// - Easy to use (no model management)
///
/// # Cost Considerations
///
/// OpenAI charges per token processed. Consider using `CandleBertEmbedder`
/// for cost-sensitive applications or when processing large volumes.
///
/// # Examples
///
/// ```no_run
/// use laurus::embedding::embedder::{Embedder, EmbedInput};
/// use laurus::embedding::openai_embedder::OpenAIEmbedder;
///
/// # async fn example() -> laurus::Result<()> {
/// // Create embedder with API key
/// let embedder = OpenAIEmbedder::new(
///     std::env::var("OPENAI_API_KEY").unwrap(),
///     "text-embedding-3-small".to_string()
/// ).await?;
///
/// // Generate embedding
/// let vector = embedder.embed(&EmbedInput::Text("Rust is awesome!")).await?;
///
/// // Batch processing (more efficient)
/// let inputs = vec![EmbedInput::Text("Hello"), EmbedInput::Text("World")];
/// let vectors = embedder.embed_batch(&inputs).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-openai")]
pub struct OpenAIEmbedder {
    /// HTTP client for making API requests.
    client: Client,
    /// OpenAI API key for authentication.
    api_key: String,
    /// OpenAI model name (e.g., "text-embedding-3-small").
    model: String,
    /// Dimension of the output embeddings.
    dimension: usize,
}

#[cfg(feature = "embeddings-openai")]
impl std::fmt::Debug for OpenAIEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbedder")
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

#[cfg(feature = "embeddings-openai")]
impl OpenAIEmbedder {
    /// Create a new OpenAI embedder.
    ///
    /// Validates the API key and model availability by making a request to OpenAI's API.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key (get from <https://platform.openai.com/api-keys>)
    /// * `model` - Model name to use
    ///
    /// # Returns
    ///
    /// A new `OpenAIEmbedder` instance if validation succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key is invalid
    /// - The model does not exist or is not available
    /// - Network request fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use laurus::embedding::openai_embedder::OpenAIEmbedder;
    ///
    /// # async fn example() -> laurus::Result<()> {
    /// // Small model (recommended for most use cases)
    /// let embedder = OpenAIEmbedder::new(
    ///     "sk-...".to_string(),
    ///     "text-embedding-3-small".to_string()
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(api_key: String, model: String) -> Result<Self> {
        let client = Client::new();

        // Validate model existence via API
        let url = format!("https://api.openai.com/v1/models/{}", model);
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("Failed to connect to OpenAI API: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(LaurusError::InvalidOperation(format!(
                "Failed to validate OpenAI model '{}'. Status: {}. Response: {}",
                model, status, text
            )));
        }

        let dimension = Self::default_dimension(&model);

        Ok(Self {
            client,
            api_key,
            model,
            dimension,
        })
    }

    /// Create an embedder with a custom dimension.
    ///
    /// OpenAI's newer models support custom dimensions for reduced cost and storage.
    /// Validates the API key and model availability via API.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `model` - Model name
    /// * `dimension` - Custom dimension size (must be supported by the model)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use laurus::embedding::openai_embedder::OpenAIEmbedder;
    ///
    /// # async fn example() -> laurus::Result<()> {
    /// // Use smaller dimension for cost savings
    /// let embedder = OpenAIEmbedder::with_dimension(
    ///     "sk-...".to_string(),
    ///     "text-embedding-3-small".to_string(),
    ///     512  // Reduced from 1536
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_dimension(api_key: String, model: String, dimension: usize) -> Result<Self> {
        // Reuse validation logic from new() by calling it and then modifying the dimension
        // This is slightly inefficient as it creates a client twice, but clean for now.
        // Alternatively, refactor validation into a private helper.
        // For simplicity, let's just duplicate the validation logic or better yet,
        // extract validation to a helper. Given constraints, let's reuse new().

        let mut embedder = Self::new(api_key, model).await?;
        embedder.dimension = dimension;
        Ok(embedder)
    }

    /// Get the default dimension for a given model.
    ///
    /// Returns the standard embedding dimension for each OpenAI model.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    ///
    /// # Returns
    ///
    /// Default dimension: 1536 for small/ada-002, 3072 for large
    fn default_dimension(model: &str) -> usize {
        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // fallback
        }
    }

    /// Embed text using OpenAI API (internal implementation).
    async fn embed_text(&self, text: &str) -> Result<Vector> {
        let dimensions = if self.dimension == Self::default_dimension(&self.model) {
            None
        } else {
            Some(self.dimension)
        };

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: vec![text.to_string()],
            dimensions,
        };

        let http_response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("OpenAI API request failed: {}", e))
            })?;

        let status = http_response.status();
        let response_text = http_response.text().await.map_err(|e| {
            LaurusError::InvalidOperation(format!("Failed to read response text: {}", e))
        })?;

        if !status.is_success() {
            return Err(LaurusError::InvalidOperation(format!(
                "OpenAI API error (status {}): {}",
                status, response_text
            )));
        }

        let response: EmbeddingResponse = serde_json::from_str(&response_text).map_err(|e| {
            LaurusError::InvalidOperation(format!(
                "Failed to parse OpenAI response: {}. Response text: {}",
                e, response_text
            ))
        })?;

        let embedding = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| LaurusError::InvalidOperation("No embedding in response".to_string()))?
            .embedding;

        Ok(Vector::new(embedding))
    }

    /// Embed multiple texts in a single batch request (internal implementation).
    async fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let dimensions = if self.dimension == Self::default_dimension(&self.model) {
            None
        } else {
            Some(self.dimension)
        };

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
            dimensions,
        };

        let http_response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("OpenAI API request failed: {}", e))
            })?;

        let status = http_response.status();
        let response_text = http_response.text().await.map_err(|e| {
            LaurusError::InvalidOperation(format!("Failed to read response text: {}", e))
        })?;

        if !status.is_success() {
            return Err(LaurusError::InvalidOperation(format!(
                "OpenAI API error (status {}): {}",
                status, response_text
            )));
        }

        let response: EmbeddingResponse = serde_json::from_str(&response_text).map_err(|e| {
            LaurusError::InvalidOperation(format!(
                "Failed to parse OpenAI response: {}. Response text: {}",
                e, response_text
            ))
        })?;

        Ok(response
            .data
            .into_iter()
            .map(|d| Vector::new(d.embedding))
            .collect())
    }
}

#[cfg(feature = "embeddings-openai")]
#[async_trait]
impl Embedder for OpenAIEmbedder {
    /// Generate an embedding vector for the given input.
    ///
    /// Only text input is supported. Image input will return an error.
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => self.embed_text(text).await,
            _ => Err(LaurusError::invalid_argument(
                "OpenAIEmbedder only supports text input",
            )),
        }
    }

    /// Generate embeddings for multiple inputs in a batch.
    ///
    /// For text inputs, this makes a single API request which is more efficient.
    /// All inputs must be text; image inputs will cause an error.
    async fn embed_batch(&self, inputs: &[EmbedInput<'_>]) -> Result<Vec<Vector>> {
        // Extract text from all inputs, fail if any are not text
        let texts: Vec<&str> = inputs
            .iter()
            .map(|input| match input {
                EmbedInput::Text(text) => Ok(*text),
                _ => Err(LaurusError::invalid_argument(
                    "OpenAIEmbedder only supports text input",
                )),
            })
            .collect::<Result<Vec<_>>>()?;

        self.embed_text_batch(&texts).await
    }

    /// Get the supported input types.
    ///
    /// OpenAIEmbedder only supports text input.
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    /// Get the name/identifier of this embedder.
    fn name(&self) -> &str {
        &self.model
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
