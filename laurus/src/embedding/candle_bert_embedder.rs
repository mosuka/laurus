//! Candle-based BERT embedder implementation.
//!
//! This module provides a text embedder using HuggingFace Candle framework.
//! Requires the `embeddings-candle` feature to be enabled.

#[cfg(feature = "embeddings-candle")]
use std::any::Any;

#[cfg(feature = "embeddings-candle")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-candle")]
use candle_core::{D, DType, Device, Tensor};
#[cfg(feature = "embeddings-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "embeddings-candle")]
use candle_transformers::models::bert::{BertModel, Config};
#[cfg(feature = "embeddings-candle")]
use tokenizers::{Tokenizer, TruncationParams};

#[cfg(feature = "embeddings-candle")]
use crate::embedding::candle_hub::HubModel;
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
        Self::with_options(model_name, CandleBertOptions::default())
    }

    /// Create a BERT embedder from a HuggingFace model with options, for
    /// example a pinned model revision.
    ///
    /// Inputs are truncated to the model's `max_seq_length` from
    /// `sentence_bert_config.json` when the repository has one, as
    /// sentence-transformers does (Issue #1340); otherwise to the
    /// truncation length in `tokenizer.json`, otherwise to
    /// `max_position_embeddings`.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn with_options(model_name: &str, options: CandleBertOptions) -> Result<Self> {
        // Setup device (prefer GPU if available)
        let device = Device::cuda_if_available(0)
            .map_err(|e| LaurusError::InvalidOperation(format!("Device setup failed: {}", e)))?;

        // Download model from HuggingFace Hub
        let repo = HubModel::open(model_name, options.revision.as_deref(), None)?;

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
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenizer load failed: {}", e)))?;

        // Truncate and pad like sentence-transformers rather than as
        // tokenizer.json says: a fixed padding (128 for all-MiniLM-L6-v2)
        // would feed the model pad tokens it was never meant to see.
        let sentence_bert_max_length = match repo.optional_file("sentence_bert_config.json")? {
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    LaurusError::InvalidOperation(format!(
                        "sentence_bert_config.json read failed: {e}"
                    ))
                })?;
                serde_json::from_str::<SentenceBertConfig>(&text)
                    .map_err(|e| {
                        LaurusError::InvalidOperation(format!(
                            "sentence_bert_config.json parse failed: {e}"
                        ))
                    })?
                    .max_seq_length
            }
            None => None,
        };
        let max_length = max_sequence_length(
            sentence_bert_max_length,
            tokenizer.get_truncation().map(|t| t.max_length),
            config.max_position_embeddings,
        );
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("Tokenizer truncation setup failed: {e}"))
            })?;

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
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenization failed: {}", e)))?;
        let vector = pooled_embedding(
            &self.model,
            &self.device,
            encoding.get_ids(),
            encoding.get_type_ids(),
            encoding.get_attention_mask(),
        )
        .map_err(|e| LaurusError::InvalidOperation(format!("Model forward failed: {}", e)))?;
        Ok(Vector::new(vector))
    }
}

/// Options for [`CandleBertEmbedder::with_options`].
#[cfg(feature = "embeddings-candle")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CandleBertOptions {
    /// Branch, tag or commit of the model repository; `None` is the
    /// default branch.
    pub revision: Option<String>,
}

#[cfg(feature = "embeddings-candle")]
impl CandleBertOptions {
    /// Download the model at this branch, tag or commit.
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }
}

/// The part of a sentence-transformers `sentence_bert_config.json` that
/// affects the encoding.
#[cfg(feature = "embeddings-candle")]
#[derive(Debug, serde::Deserialize)]
struct SentenceBertConfig {
    max_seq_length: Option<usize>,
}

/// Longest input in tokens, special tokens included: the
/// sentence-transformers `max_seq_length`, else the tokenizer's truncation
/// length, else the model's position limit, and never past that limit.
#[cfg(feature = "embeddings-candle")]
fn max_sequence_length(
    sentence_bert: Option<usize>,
    tokenizer: Option<usize>,
    max_position_embeddings: usize,
) -> usize {
    sentence_bert
        .filter(|&n| n > 0)
        .or(tokenizer.filter(|&n| n > 0))
        .unwrap_or(max_position_embeddings)
        .min(max_position_embeddings)
}

/// Mean-pooled, L2-normalized embedding of one tokenized input.
///
/// Like sentence-transformers: the type ids and the attention mask go to
/// the model in their own slots, the mean is taken over the attended
/// tokens (divided by at least 1e-9), and the result is divided by
/// `max(‖x‖, 1e-12)`.
#[cfg(feature = "embeddings-candle")]
fn pooled_embedding(
    model: &BertModel,
    device: &Device,
    ids: &[u32],
    type_ids: &[u32],
    attention_mask: &[u32],
) -> candle_core::Result<Vec<f32>> {
    let row = |values: &[u32]| Tensor::new(values, device)?.unsqueeze(0);
    let (ids, type_ids, attention_mask) = (row(ids)?, row(type_ids)?, row(attention_mask)?);
    let hidden = model.forward(&ids, &type_ids, Some(&attention_mask))?;

    let mask = attention_mask.to_dtype(hidden.dtype())?.unsqueeze(2)?;
    let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
    let counts = mask.sum(1)?.maximum(1e-9)?;
    let mean = summed.broadcast_div(&counts)?;
    let norms = mean.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
    mean.broadcast_div(&norms)?.squeeze(0)?.to_vec1()
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

#[cfg(all(test, feature = "embeddings-candle"))]
mod tests {
    use candle_nn::VarMap;
    use candle_transformers::models::bert::HiddenAct;

    use super::*;

    #[test]
    fn test_max_sequence_length_prefers_sentence_transformers() {
        // all-MiniLM-L6-v2: max_seq_length 256 over tokenizer.json's 128.
        assert_eq!(max_sequence_length(Some(256), Some(128), 512), 256);
        assert_eq!(max_sequence_length(None, Some(128), 512), 128);
        assert_eq!(max_sequence_length(None, None, 512), 512);
        // Never past the position limit, and zero means unset.
        assert_eq!(max_sequence_length(Some(1024), None, 512), 512);
        assert_eq!(max_sequence_length(Some(0), Some(0), 512), 512);
    }

    /// A tiny BERT with random weights.
    fn tiny_bert() -> BertModel {
        let config = Config {
            vocab_size: 128,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 32,
            hidden_act: HiddenAct::Gelu,
            max_position_embeddings: 64,
            ..Config::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        BertModel::load(vb, &config).unwrap()
    }

    /// Issue #1340: masked padding must not change the embedding. It did
    /// when the mask was passed as `token_type_ids` and no attention mask
    /// was given, so the real tokens attended to the pads.
    #[test]
    fn test_padding_does_not_change_the_embedding() {
        let model = tiny_bert();
        let device = Device::Cpu;
        let alone = pooled_embedding(&model, &device, &[101, 7, 8, 102], &[0; 4], &[1; 4]).unwrap();
        let padded = pooled_embedding(
            &model,
            &device,
            &[101, 7, 8, 102, 0, 0, 0],
            &[0; 7],
            &[1, 1, 1, 1, 0, 0, 0],
        )
        .unwrap();

        let diff = alone
            .iter()
            .zip(&padded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "padding changed the embedding by {diff}");
        let norm = alone.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
    }

    #[test]
    fn test_options_builder() {
        let options = CandleBertOptions::default().revision("abc");
        assert_eq!(options.revision.as_deref(), Some("abc"));
    }

    // Parity with sentence-transformers, the reference implementation. The
    // fixtures in `tests/fixtures/sentence_transformers/` come from
    // `scripts/sentence_transformers_reference.py`. These tests download
    // the models, so they are ignored by default:
    //
    //     cargo test -p laurus --features embeddings-candle --lib -- \
    //         --ignored bert_parity --nocapture

    #[derive(serde::Deserialize)]
    struct Fixture {
        model: String,
        revision: String,
        max_seq_length: usize,
        /// Whether sentence-transformers normalizes this model's output;
        /// CandleBertEmbedder always does.
        normalize: bool,
        dim: usize,
        inputs: Vec<FixtureInput>,
    }

    #[derive(serde::Deserialize)]
    struct FixtureInput {
        text: String,
        input_ids: Vec<u32>,
        /// Little-endian f32 values, base64.
        vector: String,
    }

    fn decode(encoded: &str) -> Vec<f32> {
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let (words, rest) = bytes.as_chunks::<4>();
        assert!(rest.is_empty());
        words.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    fn l2_normalized(vector: &[f32]) -> Vec<f32> {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        vector.iter().map(|x| x / norm).collect()
    }

    async fn check_parity(file: &str) {
        let path = format!(
            "{}/tests/fixtures/sentence_transformers/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let (model, revision) = (fixture.model.clone(), fixture.revision.clone());
        let embedder = tokio::task::spawn_blocking(move || {
            CandleBertEmbedder::with_options(
                &model,
                CandleBertOptions::default().revision(revision),
            )
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            embedder.tokenizer.get_truncation().map(|t| t.max_length),
            Some(fixture.max_seq_length)
        );

        let (mut max_abs, mut min_cosine) = (0.0f32, 1.0f32);
        for input in &fixture.inputs {
            let encoding = embedder
                .tokenizer
                .encode(input.text.as_str(), true)
                .unwrap();
            assert_eq!(encoding.get_ids(), input.input_ids, "{}", input.text);

            let ours = embedder
                .embed(&EmbedInput::Text(&input.text))
                .await
                .unwrap();
            let mut reference = decode(&input.vector);
            assert_eq!(reference.len(), fixture.dim);
            if !fixture.normalize {
                reference = l2_normalized(&reference);
            }
            for (a, b) in ours.data.iter().zip(&reference) {
                max_abs = max_abs.max((a - b).abs());
            }
            // Both sides are unit vectors.
            let cosine: f32 = ours.data.iter().zip(&reference).map(|(a, b)| a * b).sum();
            min_cosine = min_cosine.min(cosine);
        }

        println!("{file}: max |Δ| {max_abs:.2e}, min cosine {min_cosine:.8}");
        assert!(max_abs <= 1e-5, "max |Δ| {max_abs}");
        assert!(min_cosine >= 0.99999, "min cosine {min_cosine}");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads sentence-transformers/all-MiniLM-L6-v2"]
    async fn bert_parity_all_minilm_l6_v2() {
        check_parity("all-MiniLM-L6-v2.json").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    async fn bert_parity_paraphrase_multilingual_minilm_l12_v2() {
        check_parity("paraphrase-multilingual-MiniLM-L12-v2.json").await;
    }
}
