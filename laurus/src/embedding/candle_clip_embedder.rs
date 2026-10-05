//! Candle-based CLIP embedder implementation.
//!
//! This module provides a multimodal embedder using HuggingFace Candle framework
//! with CLIP (Contrastive Language-Image Pre-Training) models.
//! Requires the `embeddings-multimodal` feature to be enabled.

#[cfg(feature = "embeddings-multimodal")]
use std::any::Any;

#[cfg(feature = "embeddings-multimodal")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-multimodal")]
use candle_core::{DType, Device, Module, Tensor};
#[cfg(feature = "embeddings-multimodal")]
use candle_nn::{Linear, VarBuilder};
#[cfg(feature = "embeddings-multimodal")]
use candle_transformers::models::clip;
#[cfg(feature = "embeddings-multimodal")]
use tokenizers::Tokenizer;

#[cfg(feature = "embeddings-multimodal")]
use crate::embedding::candle_hub::{HubModel, legacy_cache_dir};
#[cfg(feature = "embeddings-multimodal")]
use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
#[cfg(feature = "embeddings-multimodal")]
use crate::error::{LaurusError, Result};
#[cfg(feature = "embeddings-multimodal")]
use crate::vector::core::vector::Vector;

/// Candle-based CLIP embedder using CLIP models from HuggingFace.
///
/// This embedder uses the Candle framework to run CLIP models locally,
/// providing embeddings for both text and images in the same vector space.
///
/// # Features
///
/// - Offline inference (no API calls)
/// - GPU acceleration support
/// - Cross-modal search (text-to-image, image-to-image)
/// - Multiple CLIP model variants support
///
/// # Supported Models
///
/// Note: Use models from the HuggingFace model hub that have CLIP architecture.
/// The default config currently supports `vit-base-patch32` architecture.
///
/// Example repositories:
/// - Models with pre-trained CLIP weights compatible with Candle
/// - Custom fine-tuned CLIP models
///
/// # Examples
///
/// ## Text-to-Image Search
///
/// ```no_run
/// use laurus::embedding::embedder::{Embedder, EmbedInput};
/// use laurus::embedding::candle_clip_embedder::CandleClipEmbedder;
/// use std::fs;
///
/// # async fn example() -> laurus::Result<()> {
/// let embedder = CandleClipEmbedder::new(
///     "openai/clip-vit-base-patch32"
/// )?;
///
/// // Embed text query
/// let text_vec = embedder.embed(&EmbedInput::Text("a photo of a cat")).await?;
///
/// // Embed images
/// // Loading files is now the responsibility of the caller
/// let cat_bytes = fs::read("cat.jpg")?;
/// let img1 = embedder.embed(&EmbedInput::Bytes(&cat_bytes, None)).await?;
///
/// let dog_bytes = fs::read("dog.jpg")?;
/// let img2 = embedder.embed(&EmbedInput::Bytes(&dog_bytes, None)).await?;
///
/// // Text and images are in the same vector space
/// # Ok(())
/// # }
/// ```
///
/// ## Image-to-Image Search
///
/// ```no_run
/// use laurus::embedding::embedder::{Embedder, EmbedInput};
/// use laurus::embedding::candle_clip_embedder::CandleClipEmbedder;
/// use std::fs;
///
/// # async fn example() -> laurus::Result<()> {
/// let embedder = CandleClipEmbedder::new(
///     "openai/clip-vit-base-patch32"
/// )?;
///
/// // Find similar images
/// let query_bytes = fs::read("query.jpg")?;
/// let query = embedder.embed(&EmbedInput::Bytes(&query_bytes, None)).await?;
///
/// let img1_bytes = fs::read("img1.jpg")?;
/// let img2_bytes = fs::read("img2.jpg")?;
///
/// let inputs = vec![
///     EmbedInput::Bytes(&img1_bytes, None),
///     EmbedInput::Bytes(&img2_bytes, None),
/// ];
/// let images = embedder.embed_batch(&inputs).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-multimodal")]
pub struct CandleClipEmbedder {
    /// CLIP text transformer model.
    text_model: clip::text_model::ClipTextTransformer,
    /// CLIP vision transformer model.
    vision_model: clip::vision_model::ClipVisionTransformer,
    /// Linear projection layer for text embeddings.
    text_projection: Linear,
    /// Linear projection layer for vision embeddings.
    vision_projection: Linear,
    /// Tokenizer for text input.
    tokenizer: Tokenizer,
    /// Device to run models on (CPU or GPU).
    device: Device,
    /// Dimension of the shared embedding space.
    dimension: usize,
    /// Name of the HuggingFace CLIP model.
    model_name: String,
    /// Expected image size (width/height in pixels).
    image_size: usize,
}

#[cfg(feature = "embeddings-multimodal")]
impl std::fmt::Debug for CandleClipEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleClipEmbedder")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .field("image_size", &self.image_size)
            .finish()
    }
}

#[cfg(feature = "embeddings-multimodal")]
impl CandleClipEmbedder {
    /// Create a new Candle-based CLIP embedder from a HuggingFace CLIP model.
    ///
    /// The model will be automatically downloaded from HuggingFace Hub if not cached.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace CLIP model identifier (e.g., "openai/clip-vit-base-patch32")
    ///
    /// # Returns
    ///
    /// A new `CandleClipEmbedder` instance
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
    /// use laurus::embedding::candle_clip_embedder::CandleClipEmbedder;
    ///
    /// # fn example() -> laurus::Result<()> {
    /// // Fast and efficient
    /// let embedder = CandleClipEmbedder::new(
    ///     "openai/clip-vit-base-patch32"
    /// )?;
    ///
    /// // Higher quality
    /// let embedder = CandleClipEmbedder::new(
    ///     "openai/clip-vit-large-patch14"
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
        // Currently defaults to ViT-B/32 configuration. To support other CLIP variants
        // (e.g. ViT-L/14), this should parse config.json from the model repository.
        let config = clip::ClipConfig::vit_base_patch32();

        // Load weights - try safetensors first, fall back to pytorch
        let weights_filename = repo
            .file("model.safetensors")
            .or_else(|_| repo.file("pytorch_model.bin"))?;

        let vb = if weights_filename.to_string_lossy().ends_with(".safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)
                    .map_err(|e| {
                        LaurusError::InvalidOperation(format!("VarBuilder creation failed: {}", e))
                    })?
            }
        } else {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device).map_err(|e| {
                LaurusError::InvalidOperation(format!("VarBuilder creation failed: {}", e))
            })?
        };

        // Load text model
        let text_model =
            clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config.text_config)
                .map_err(|e| {
                    LaurusError::InvalidOperation(format!("Text model load failed: {}", e))
                })?;

        // Load vision model
        let vision_model = clip::vision_model::ClipVisionTransformer::new(
            vb.pp("vision_model"),
            &config.vision_config,
        )
        .map_err(|e| LaurusError::InvalidOperation(format!("Vision model load failed: {}", e)))?;

        // Load projection layers
        let projection_dim = config.text_config.projection_dim;

        // CLIP models use linear layers without bias
        let text_projection = candle_nn::linear_no_bias(
            config.text_config.embed_dim,
            projection_dim,
            vb.pp("text_projection"),
        )
        .map_err(|e| {
            LaurusError::InvalidOperation(format!("Text projection load failed: {}", e))
        })?;

        let vision_projection = candle_nn::linear_no_bias(
            config.vision_config.embed_dim,
            projection_dim,
            vb.pp("visual_projection"),
        )
        .map_err(|e| {
            LaurusError::InvalidOperation(format!("Vision projection load failed: {}", e))
        })?;

        // Load tokenizer
        let tokenizer_filename = repo.file("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenizer load failed: {}", e)))?;

        let dimension = projection_dim;
        let image_size = config.vision_config.image_size;

        Ok(Self {
            text_model,
            vision_model,
            text_projection,
            vision_projection,
            tokenizer,
            device,
            dimension,
            model_name: model_name.to_string(),
            image_size,
        })
    }

    /// Embed text using CLIP text encoder.
    async fn embed_text(&self, text: &str) -> Result<Vector> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenization failed: {}", e)))?;

        let token_ids = encoding.get_ids();

        // Convert to tensor
        let token_ids_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tensor creation failed: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Forward pass through text model
        let text_features = self.text_model.forward(&token_ids_tensor).map_err(|e| {
            LaurusError::InvalidOperation(format!("Text model forward failed: {}", e))
        })?;

        // Project to common embedding space
        let projected = self
            .text_projection
            .forward(&text_features)
            .map_err(|e| LaurusError::InvalidOperation(format!("Text projection failed: {}", e)))?;

        // Normalize
        let normalized = self.normalize(&projected)?;

        // Convert to Vector
        let vector_data: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .to_vec1()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        Ok(Vector::new(vector_data))
    }

    /// Embed image bytes using CLIP vision encoder.
    async fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vector> {
        // Preprocess image from bytes
        let image_tensor = self.preprocess_image_bytes(bytes)?;

        // Forward pass through vision model
        let vision_features = self.vision_model.forward(&image_tensor).map_err(|e| {
            LaurusError::InvalidOperation(format!("Vision model forward failed: {}", e))
        })?;

        // Project to common embedding space
        let projected = self
            .vision_projection
            .forward(&vision_features)
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("Vision projection failed: {}", e))
            })?;

        // Normalize
        let normalized = self.normalize(&projected)?;

        // Convert to Vector
        let vector_data: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .to_vec1()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        Ok(Vector::new(vector_data))
    }

    /// Preprocess image bytes to the format expected by CLIP vision model.
    fn preprocess_image_bytes(&self, bytes: &[u8]) -> Result<Tensor> {
        use image::ImageReader;
        use std::io::Cursor;

        // Load image from bytes
        let img_reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("Image format guess failed: {}", e))
            })?;

        let img = img_reader
            .decode()
            .map_err(|e| LaurusError::InvalidOperation(format!("Image decode failed: {}", e)))?;

        // Resize to model's expected size
        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Triangle,
        );

        // Convert to RGB
        let img = match img {
            image::DynamicImage::ImageRgb8(img) => img,
            img => img.to_rgb8(),
        };

        // Convert to tensor (C, H, W format)
        let img_data = img.into_raw();
        let img_tensor = Tensor::from_vec(
            img_data,
            (self.image_size, self.image_size, 3),
            &self.device,
        )
        .map_err(|e| LaurusError::InvalidOperation(format!("Tensor creation failed: {}", e)))?;

        // Normalize: (pixel / 255.0 - mean) / std
        // CLIP uses ImageNet normalization
        let mean = Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &self.device)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .reshape((1, 1, 3))
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;
        let std = Tensor::new(&[0.2686295_f32, 0.2613026, 0.2757771], &self.device)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .reshape((1, 1, 3))
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Scale to [0, 1] and normalize
        let normalized = img_tensor
            .to_dtype(DType::F32)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .affine(1.0 / 255.0, 0.0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .broadcast_sub(&mean)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .broadcast_div(&std)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        // Permute to (C, H, W)
        let normalized = normalized
            .permute((2, 0, 1))
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        Ok(normalized)
    }

    /// Normalize embeddings using L2 normalization.
    ///
    /// Divides each embedding vector by its L2 norm, ensuring all vectors
    /// have unit length. This is standard for CLIP embeddings to enable
    /// cosine similarity via dot product.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Embeddings to normalize (shape: [batch_size, dimension])
    ///
    /// # Returns
    ///
    /// L2-normalized embeddings with the same shape
    fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let norm = tensor
            .sqr()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?
            .sqrt()
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))?;

        tensor
            .broadcast_div(&norm)
            .map_err(|e| LaurusError::InvalidOperation(e.to_string()))
    }
}

#[cfg(feature = "embeddings-multimodal")]
#[async_trait]
impl Embedder for CandleClipEmbedder {
    /// Generate an embedding vector for the given input.
    ///
    /// Supports both text and image inputs. Text and images are embedded into
    /// the same vector space, enabling cross-modal similarity search.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to embed (text or image)
    ///
    /// # Returns
    ///
    /// A normalized vector representation in CLIP's shared embedding space
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => self.embed_text(text).await,
            EmbedInput::Bytes(bytes, mime) => {
                if let Some(mime_type) = mime
                    && mime_type.starts_with("text/")
                {
                    let text = std::str::from_utf8(bytes).map_err(|e| {
                        LaurusError::invalid_argument(format!(
                            "invalid utf-8 for text bytes: {}",
                            e
                        ))
                    })?;
                    return self.embed_text(text).await;
                }
                // Default to image processing for other mime types or if no mime type is provided
                self.embed_image_bytes(bytes).await
            }
        }
    }

    /// Get the supported input types.
    ///
    /// CandleClipEmbedder supports both text and image inputs.
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text, EmbedInputType::Image]
    }

    /// Get the name/identifier of this embedder.
    ///
    /// Returns the HuggingFace CLIP model identifier.
    fn name(&self) -> &str {
        &self.model_name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
