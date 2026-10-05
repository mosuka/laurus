//! Candle-based ColBERT token-level embedder (Issue #1349).
//!
//! Runs a BERT-based ColBERT checkpoint (for example
//! `colbert-ir/colbertv2.0` or `answerdotai/answerai-colbert-small-v1`)
//! locally and turns text into one vector per token, encoding queries and
//! documents the way colbert-ai's `Checkpoint` does:
//!
//! ```text
//! query:    [CLS] [unused0] w1 … wn [SEP] [MASK] … [MASK]   exactly query_maxlen tokens
//! document: [CLS] [unused1] w1 … wn [SEP]                   at most doc_maxlen tokens
//! ```
//!
//! - The query's `[MASK]` padding is not attended to (unless the checkpoint
//!   sets `attend_to_mask_tokens`), but its vectors are kept: they expand
//!   the query.
//! - Document vectors of punctuation tokens are dropped (`mask_punctuation`).
//! - `token_type_ids` are all zero. The last hidden state goes through the
//!   checkpoint's `linear` projection (no bias) and every vector is
//!   L2-normalized.
//!
//! Requires the `embeddings-candle` feature.

use std::any::Any;
use std::collections::HashSet;
use std::fmt::Display;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::embedding::candle_hub::HubModel;
use crate::embedding::embedder::{EmbedInput, EmbedInputType, EmbedRole, Embedder, TokenEmbedder};
use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;

/// colbert-ai's default `query_maxlen`, used when the checkpoint does not
/// set one.
const DEFAULT_QUERY_MAXLEN: usize = 32;

/// colbert-ai's default `doc_maxlen`, used when the checkpoint does not set
/// one.
const DEFAULT_DOC_MAXLEN: usize = 220;

/// Longest query accepted: a late-interaction rescore takes at most 1,024
/// query vectors.
const MAX_QUERY_MAXLEN: usize = 1024;

/// Shortest length that leaves room for one content token besides
/// `[CLS]`, the marker and `[SEP]`.
const MIN_MAXLEN: usize = 4;

/// Inputs per forward pass.
const BATCH_SIZE: usize = 32;

/// Python's `string.punctuation`; colbert-ai drops the document vectors of
/// these characters' tokens.
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Options for [`CandleColbertEmbedder::with_options`].
///
/// Unset lengths come from the checkpoint's `artifact.metadata`, then from
/// colbert-ai's defaults (32 and 220).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CandleColbertOptions {
    /// Branch, tag or commit of the model repository; `None` is the
    /// default branch. Pin a commit so that re-embedding (for example when
    /// the write-ahead log is replayed) reproduces the same vectors.
    pub revision: Option<String>,
    /// Number of tokens every query is padded or truncated to, markers
    /// included.
    pub query_maxlen: Option<usize>,
    /// Maximum number of tokens of a document, markers included.
    pub doc_maxlen: Option<usize>,
}

impl CandleColbertOptions {
    /// Download the model at this branch, tag or commit.
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Override the query length.
    pub fn query_maxlen(mut self, query_maxlen: usize) -> Self {
        self.query_maxlen = Some(query_maxlen);
        self
    }

    /// Override the maximum document length.
    pub fn doc_maxlen(mut self, doc_maxlen: usize) -> Self {
        self.doc_maxlen = Some(doc_maxlen);
        self
    }
}

/// The parts of a checkpoint's `artifact.metadata` (colbert-ai's saved
/// configuration) that affect the encoding.
#[derive(Debug, Default, Deserialize)]
struct ArtifactMetadata {
    query_maxlen: Option<usize>,
    doc_maxlen: Option<usize>,
    attend_to_mask_tokens: Option<bool>,
    mask_punctuation: Option<bool>,
    query_token_id: Option<String>,
    doc_token_id: Option<String>,
}

/// One input laid out for the model.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Encoded {
    ids: Vec<u32>,
    /// 1 where the position is attended to, 0 otherwise.
    attention: Vec<u32>,
    /// Positions whose output vectors are returned.
    keep: Vec<bool>,
}

/// How ColBERT lays out queries and documents, independent of the model
/// weights.
#[derive(Debug, Clone)]
struct Layout {
    cls: u32,
    sep: u32,
    pad: u32,
    mask: u32,
    query_marker: u32,
    doc_marker: u32,
    query_maxlen: usize,
    doc_maxlen: usize,
    attend_to_mask_tokens: bool,
    /// Token ids whose document vectors are dropped.
    skiplist: HashSet<u32>,
}

impl Layout {
    /// Resolve the layout from the tokenizer, the checkpoint metadata and
    /// the caller's overrides.
    fn new(
        tokenizer: &Tokenizer,
        config: &Config,
        metadata: &ArtifactMetadata,
        options: &CandleColbertOptions,
    ) -> Result<Self> {
        let token = |name: &str| {
            tokenizer.token_to_id(name).ok_or_else(|| {
                LaurusError::InvalidOperation(format!("the tokenizer has no '{name}' token"))
            })
        };

        let query_maxlen = options
            .query_maxlen
            .or(metadata.query_maxlen)
            .unwrap_or(DEFAULT_QUERY_MAXLEN);
        let doc_maxlen = options
            .doc_maxlen
            .or(metadata.doc_maxlen)
            .unwrap_or(DEFAULT_DOC_MAXLEN);
        let positions = config.max_position_embeddings;
        let query_limit = positions.min(MAX_QUERY_MAXLEN);
        if !(MIN_MAXLEN..=query_limit).contains(&query_maxlen) {
            return Err(LaurusError::invalid_argument(format!(
                "query_maxlen must be between {MIN_MAXLEN} and {query_limit}, got {query_maxlen}"
            )));
        }
        if !(MIN_MAXLEN..=positions).contains(&doc_maxlen) {
            return Err(LaurusError::invalid_argument(format!(
                "doc_maxlen must be between {MIN_MAXLEN} and {positions}, got {doc_maxlen}"
            )));
        }

        // colbert-ai keys the skiplist by the first token of each character.
        let mut skiplist = HashSet::new();
        if metadata.mask_punctuation.unwrap_or(true) {
            for symbol in PUNCTUATION.chars() {
                let encoding = tokenizer
                    .encode(symbol.to_string().as_str(), false)
                    .map_err(tokenizer_error)?;
                if let Some(&id) = encoding.get_ids().first() {
                    skiplist.insert(id);
                }
            }
        }

        Ok(Self {
            cls: token("[CLS]")?,
            sep: token("[SEP]")?,
            pad: token("[PAD]")?,
            mask: token("[MASK]")?,
            query_marker: token(metadata.query_token_id.as_deref().unwrap_or("[unused0]"))?,
            doc_marker: token(metadata.doc_token_id.as_deref().unwrap_or("[unused1]"))?,
            query_maxlen,
            doc_maxlen,
            attend_to_mask_tokens: metadata.attend_to_mask_tokens.unwrap_or(false),
            skiplist,
        })
    }

    /// Lay out the content tokens (no special tokens) of one input.
    fn encode(&self, content: &[u32], role: EmbedRole) -> Encoded {
        match role {
            EmbedRole::Query => self.query(content),
            EmbedRole::Document => self.document(content),
        }
    }

    fn query(&self, content: &[u32]) -> Encoded {
        let content = &content[..content.len().min(self.query_maxlen - 3)];
        let mut ids = Vec::with_capacity(self.query_maxlen);
        ids.extend([self.cls, self.query_marker]);
        ids.extend_from_slice(content);
        ids.push(self.sep);
        let mut attention = vec![1; ids.len()];
        ids.resize(self.query_maxlen, self.mask);
        attention.resize(self.query_maxlen, u32::from(self.attend_to_mask_tokens));
        // colbert-ai turns every [PAD] into [MASK], including one that was
        // written in the text (which stays attended to).
        for id in &mut ids {
            if *id == self.pad {
                *id = self.mask;
            }
        }
        Encoded {
            keep: vec![true; ids.len()],
            ids,
            attention,
        }
    }

    fn document(&self, content: &[u32]) -> Encoded {
        let content = &content[..content.len().min(self.doc_maxlen - 3)];
        let mut ids = Vec::with_capacity(content.len() + 3);
        ids.extend([self.cls, self.doc_marker]);
        ids.extend_from_slice(content);
        ids.push(self.sep);
        let keep = ids
            .iter()
            .map(|id| *id != self.pad && !self.skiplist.contains(id))
            .collect();
        Encoded {
            attention: vec![1; ids.len()],
            keep,
            ids,
        }
    }
}

/// The BERT encoder and the ColBERT projection.
struct Network {
    bert: BertModel,
    linear: Linear,
    pad: u32,
    device: Device,
}

impl Network {
    /// Token vectors of each input, in input order.
    ///
    /// Inputs are sorted by length and run in chunks of [`BATCH_SIZE`], so
    /// a chunk pads as little as possible.
    fn embed(&self, encoded: &[Encoded]) -> Result<Vec<Vec<Vector>>> {
        let mut order: Vec<usize> = (0..encoded.len()).collect();
        order.sort_by_key(|&i| encoded[i].ids.len());
        let mut out = vec![Vec::new(); encoded.len()];
        for chunk in order.chunks(BATCH_SIZE) {
            let batch: Vec<&Encoded> = chunk.iter().map(|&i| &encoded[i]).collect();
            for (&i, tokens) in chunk.iter().zip(self.forward(&batch).map_err(model_error)?) {
                out[i] = tokens;
            }
        }
        Ok(out)
    }

    /// One forward pass over a padded batch.
    fn forward(&self, batch: &[&Encoded]) -> candle_core::Result<Vec<Vec<Vector>>> {
        let len = batch.iter().map(|e| e.ids.len()).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(batch.len() * len);
        let mut attention = Vec::with_capacity(batch.len() * len);
        for encoded in batch {
            ids.extend_from_slice(&encoded.ids);
            ids.resize(ids.len() + len - encoded.ids.len(), self.pad);
            attention.extend_from_slice(&encoded.attention);
            attention.resize(attention.len() + len - encoded.attention.len(), 0);
        }
        let shape = (batch.len(), len);
        let ids = Tensor::from_vec(ids, shape, &self.device)?;
        let attention = Tensor::from_vec(attention, shape, &self.device)?;
        let token_types = ids.zeros_like()?;

        let hidden = self.bert.forward(&ids, &token_types, Some(&attention))?;
        let projected = self.linear.forward(&hidden)?;
        // F.normalize(p=2, dim=-1): divide by max(norm, 1e-12).
        let norms = projected
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .maximum(1e-12)?;
        let rows: Vec<Vec<Vec<f32>>> = projected.broadcast_div(&norms)?.to_vec3()?;

        Ok(batch
            .iter()
            .zip(rows)
            .map(|(encoded, vectors)| {
                // Zipping with `keep` also drops the batch padding.
                vectors
                    .into_iter()
                    .zip(&encoded.keep)
                    .filter(|(_, keep)| **keep)
                    .map(|(vector, _)| Vector::new(vector))
                    .collect()
            })
            .collect())
    }
}

/// Everything inference needs, shared with the blocking inference tasks.
struct ColbertModel {
    network: Network,
    tokenizer: Tokenizer,
    layout: Layout,
}

impl ColbertModel {
    fn embed(&self, texts: &[String], role: EmbedRole) -> Result<Vec<Vec<Vector>>> {
        let encoded = texts
            .iter()
            .map(|text| {
                let encoding = self
                    .tokenizer
                    .encode(text.as_str(), false)
                    .map_err(tokenizer_error)?;
                Ok(self.layout.encode(encoding.get_ids(), role))
            })
            .collect::<Result<Vec<_>>>()?;
        self.network.embed(&encoded)
    }
}

/// ColBERT token-level embedder running a BERT-based checkpoint with
/// candle (Issue #1349).
///
/// It is a [`TokenEmbedder`]: use it for a multi-vector field, where it
/// embeds documents when they are indexed and late-interaction rescore
/// queries when they are searched. It cannot produce a single vector, so
/// [`Embedder::embed`] returns an error.
///
/// Inference runs on the CPU in a blocking task, so it does not stall the
/// async runtime.
///
/// # Examples
///
/// ```no_run
/// use laurus::embedding::candle_colbert_embedder::CandleColbertEmbedder;
/// use laurus::{EmbedInput, EmbedRole, Embedder};
///
/// # async fn example() -> laurus::Result<()> {
/// let embedder = CandleColbertEmbedder::new("colbert-ir/colbertv2.0")?;
/// let tokens = embedder
///     .as_token_embedder()
///     .expect("a ColBERT embedder embeds tokens")
///     .embed_tokens(&[EmbedInput::Text("what is late interaction?")], EmbedRole::Query)
///     .await?;
/// assert_eq!(tokens[0].len(), 32); // query_maxlen
/// # Ok(())
/// # }
/// ```
pub struct CandleColbertEmbedder {
    model: Arc<ColbertModel>,
    model_name: String,
    dimension: usize,
}

impl std::fmt::Debug for CandleColbertEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let layout = &self.model.layout;
        f.debug_struct("CandleColbertEmbedder")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .field("query_maxlen", &layout.query_maxlen)
            .field("doc_maxlen", &layout.doc_maxlen)
            .finish()
    }
}

impl CandleColbertEmbedder {
    /// Load a ColBERT checkpoint from the Hugging Face Hub with its own
    /// settings.
    ///
    /// # Errors
    ///
    /// See [`Self::with_options`].
    pub fn new(model_name: &str) -> Result<Self> {
        Self::with_options(model_name, CandleColbertOptions::default())
    }

    /// Load a ColBERT checkpoint from the Hugging Face Hub.
    ///
    /// Downloads (or reuses from hf-hub's cache) `config.json`,
    /// `tokenizer.json`, `model.safetensors` and, when present,
    /// `artifact.metadata`. This blocks; call it from a blocking context.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when the checkpoint is not
    /// BERT-based or a length is out of range, and
    /// [`LaurusError::InvalidOperation`] when a download or loading the
    /// model fails.
    pub fn with_options(model_name: &str, options: CandleColbertOptions) -> Result<Self> {
        let repo = HubModel::open(model_name, options.revision.as_deref(), None)?;

        let config_text = read_file(&repo.file("config.json")?)?;
        let raw_config: serde_json::Value = serde_json::from_str(&config_text)
            .map_err(|e| LaurusError::InvalidOperation(format!("config.json parse failed: {e}")))?;
        match raw_config.get("model_type").and_then(|t| t.as_str()) {
            Some("bert") => {}
            other => {
                return Err(LaurusError::invalid_argument(format!(
                    "'{model_name}' is not a BERT-based checkpoint (model_type {other:?}); \
                     CandleColbertEmbedder supports BERT-based ColBERT checkpoints such as \
                     colbert-ir/colbertv2.0"
                )));
            }
        }
        let config: Config = serde_json::from_value(raw_config)
            .map_err(|e| LaurusError::InvalidOperation(format!("config.json parse failed: {e}")))?;

        let metadata = match repo.optional_file("artifact.metadata")? {
            Some(path) => serde_json::from_str(&read_file(&path)?).map_err(|e| {
                LaurusError::InvalidOperation(format!("artifact.metadata parse failed: {e}"))
            })?,
            None => ArtifactMetadata::default(),
        };

        // Lengths, markers and padding are handled by `Layout`, not by
        // whatever tokenizer.json configures.
        let mut tokenizer = Tokenizer::from_file(repo.file("tokenizer.json")?)
            .map_err(|e| LaurusError::InvalidOperation(format!("tokenizer load failed: {e}")))?;
        tokenizer.with_truncation(None).map_err(tokenizer_error)?;
        tokenizer.with_padding(None);
        let layout = Layout::new(&tokenizer, &config, &metadata, &options)?;

        let device = Device::Cpu;
        let weights = repo.file("model.safetensors")?;
        // SAFETY: the file is a downloaded snapshot that nothing modifies
        // while it is mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .map_err(model_error)?;
        let bert = BertModel::load(vb.clone(), &config).map_err(model_error)?;
        let weight = vb.pp("linear").get_unchecked("weight").map_err(|e| {
            LaurusError::InvalidOperation(format!(
                "'{model_name}' has no ColBERT projection ('linear.weight'): {e}"
            ))
        })?;
        let (dimension, hidden) = weight.dims2().map_err(model_error)?;
        if hidden != config.hidden_size {
            return Err(LaurusError::InvalidOperation(format!(
                "'{model_name}' projects {hidden}-d states, but its hidden size is {}",
                config.hidden_size
            )));
        }

        Ok(Self {
            model: Arc::new(ColbertModel {
                network: Network {
                    bert,
                    linear: Linear::new(weight, None),
                    pad: layout.pad,
                    device,
                },
                tokenizer,
                layout,
            }),
            model_name: model_name.to_string(),
            dimension,
        })
    }
}

#[async_trait]
impl TokenEmbedder for CandleColbertEmbedder {
    async fn embed_tokens(
        &self,
        inputs: &[EmbedInput<'_>],
        role: EmbedRole,
    ) -> Result<Vec<Vec<Vector>>> {
        let texts = inputs
            .iter()
            .map(|input| {
                input.as_text().map(str::to_string).ok_or_else(|| {
                    LaurusError::invalid_argument("CandleColbertEmbedder only supports text input")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || model.embed(&texts, role))
            .await
            .map_err(|e| LaurusError::internal(format!("ColBERT inference task failed: {e}")))?
    }

    fn token_dimension(&self) -> usize {
        self.dimension
    }
}

#[async_trait]
impl Embedder for CandleColbertEmbedder {
    /// Always an error: a ColBERT model embeds tokens, not whole inputs.
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(LaurusError::invalid_argument(format!(
            "'{}' is a ColBERT token-level embedder; use it for a multi-vector field",
            self.model_name
        )))
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        &self.model_name
    }

    fn as_token_embedder(&self) -> Option<&dyn TokenEmbedder> {
        Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn read_file(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| {
        LaurusError::InvalidOperation(format!("failed to read '{}': {e}", path.display()))
    })
}

fn model_error(err: impl Display) -> LaurusError {
    LaurusError::InvalidOperation(format!("ColBERT model error: {err}"))
}

fn tokenizer_error(err: impl Display) -> LaurusError {
    LaurusError::InvalidOperation(format!("ColBERT tokenizer error: {err}"))
}

#[cfg(test)]
mod tests {
    use candle_nn::VarMap;
    use candle_transformers::models::bert::HiddenAct;

    use super::*;

    const CLS: u32 = 101;
    const SEP: u32 = 102;
    const MASK: u32 = 103;
    const PAD: u32 = 0;
    const COMMA: u32 = 50;
    const PERIOD: u32 = 51;

    fn layout(attend_to_mask_tokens: bool) -> Layout {
        Layout {
            cls: CLS,
            sep: SEP,
            pad: PAD,
            mask: MASK,
            query_marker: 1,
            doc_marker: 2,
            query_maxlen: 8,
            doc_maxlen: 8,
            attend_to_mask_tokens,
            skiplist: HashSet::from([COMMA, PERIOD]),
        }
    }

    #[test]
    fn test_query_is_marked_and_expanded_with_mask() {
        let encoded = layout(false).encode(&[7, 8], EmbedRole::Query);
        assert_eq!(encoded.ids, [CLS, 1, 7, 8, SEP, MASK, MASK, MASK]);
        assert_eq!(encoded.attention, [1, 1, 1, 1, 1, 0, 0, 0]);
        assert!(
            encoded.keep.iter().all(|k| *k),
            "every query vector is kept"
        );

        let attended = layout(true).encode(&[7, 8], EmbedRole::Query);
        assert_eq!(attended.attention, [1; 8]);
    }

    #[test]
    fn test_query_is_truncated_to_leave_room_for_the_special_tokens() {
        let encoded = layout(false).encode(&[10, 11, 12, 13, 14, 15, 16], EmbedRole::Query);
        assert_eq!(encoded.ids, [CLS, 1, 10, 11, 12, 13, 14, SEP]);
        assert_eq!(encoded.attention, [1; 8]);
    }

    #[test]
    fn test_query_pad_in_the_text_becomes_an_attended_mask() {
        let encoded = layout(false).encode(&[7, PAD], EmbedRole::Query);
        assert_eq!(encoded.ids, [CLS, 1, 7, MASK, SEP, MASK, MASK, MASK]);
        assert_eq!(encoded.attention, [1, 1, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_document_is_marked_and_drops_punctuation() {
        let encoded = layout(false).encode(&[7, COMMA, 8, PERIOD], EmbedRole::Document);
        assert_eq!(encoded.ids, [CLS, 2, 7, COMMA, 8, PERIOD, SEP]);
        assert_eq!(encoded.attention, [1; 7]);
        assert_eq!(encoded.keep, [true, true, true, false, true, false, true]);
    }

    #[test]
    fn test_document_is_truncated_and_never_padded() {
        let encoded = layout(false).encode(&[10, 11, 12, 13, 14, 15, 16], EmbedRole::Document);
        assert_eq!(encoded.ids, [CLS, 2, 10, 11, 12, 13, 14, SEP]);

        let empty = layout(false).encode(&[], EmbedRole::Document);
        assert_eq!(empty.ids, [CLS, 2, SEP]);
        assert_eq!(empty.keep, [true; 3]);
    }

    /// A tiny BERT with random weights: enough to check batching,
    /// padding and normalization without downloading a model.
    fn network() -> Network {
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
        Network {
            bert: BertModel::load(vb.pp("bert"), &config).unwrap(),
            linear: candle_nn::linear_no_bias(16, 6, vb.pp("linear")).unwrap(),
            pad: PAD,
            device: Device::Cpu,
        }
    }

    fn max_abs_diff(a: &[Vector], b: &[Vector]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| x.data.iter().zip(y.data.iter()).map(|(p, q)| (p - q).abs()))
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_network_keeps_the_marked_rows_with_unit_norm() {
        let layout = layout(false);
        let network = network();
        let encoded = [
            layout.encode(&[7, COMMA, 8], EmbedRole::Document),
            layout.encode(&[7, 8], EmbedRole::Query),
        ];
        let tokens = network.embed(&encoded).unwrap();
        assert_eq!(tokens[0].len(), 5, "[CLS] [D] 7 8 [SEP] without the comma");
        assert_eq!(tokens[1].len(), 8, "query_maxlen vectors");
        for vector in tokens.iter().flatten() {
            assert_eq!(vector.dimension(), 6);
            let norm: f32 = vector.data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        }
    }

    /// Padding a document in a batch must not change its vectors, and the
    /// output must come back in input order.
    #[test]
    fn test_batched_documents_match_one_by_one() {
        let layout = layout(false);
        let network = network();
        let long = layout.encode(&[7, 8, 9, 10], EmbedRole::Document);
        let short = layout.encode(&[11], EmbedRole::Document);

        let together = network.embed(&[long.clone(), short.clone()]).unwrap();
        let long_alone = network.embed(&[long]).unwrap();
        let short_alone = network.embed(&[short]).unwrap();
        assert!(max_abs_diff(&together[0], &long_alone[0]) < 1e-5);
        assert!(max_abs_diff(&together[1], &short_alone[0]) < 1e-5);
    }

    #[test]
    fn test_options_builder() {
        let options = CandleColbertOptions::default()
            .revision("abc")
            .query_maxlen(16)
            .doc_maxlen(300);
        assert_eq!(options.revision.as_deref(), Some("abc"));
        assert_eq!(options.query_maxlen, Some(16));
        assert_eq!(options.doc_maxlen, Some(300));
    }

    // Parity with colbert-ai, the reference implementation. The fixtures in
    // `tests/fixtures/colbert/` come from `scripts/colbert_reference.py`.
    // These tests download the models, so they are ignored by default:
    //
    //     cargo test -p laurus --features embeddings-candle --lib -- \
    //         --ignored colbert_parity --nocapture

    #[derive(Deserialize)]
    struct Fixture {
        model: String,
        revision: String,
        config: FixtureConfig,
        queries: Vec<FixtureQuery>,
        documents: Vec<FixtureDocument>,
        /// MaxSim of every (query, document) pair.
        scores: Vec<Vec<f32>>,
    }

    #[derive(Deserialize)]
    struct FixtureConfig {
        query_maxlen: usize,
        doc_maxlen: usize,
        dim: usize,
    }

    #[derive(Deserialize)]
    struct FixtureQuery {
        text: String,
        input_ids: Vec<u32>,
        attention_mask: Vec<u32>,
        rows: usize,
        vectors: String,
    }

    #[derive(Deserialize)]
    struct FixtureDocument {
        text: String,
        input_ids: Vec<u32>,
        kept: Vec<usize>,
        rows: usize,
        /// Rows whose vectors `vectors` holds (all but the middle of a long
        /// document).
        stored_rows: Vec<usize>,
        vectors: String,
    }

    /// Little-endian f32 rows from base64.
    fn decode_rows(encoded: &str, dim: usize) -> Vec<Vec<f32>> {
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let (words, rest) = bytes.as_chunks::<4>();
        assert!(rest.is_empty());
        let values: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
        values.chunks(dim).map(<[f32]>::to_vec).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Largest element difference and smallest cosine seen so far.
    struct Drift {
        max_abs: f32,
        min_cosine: f32,
    }

    impl Drift {
        fn add(&mut self, ours: &[f32], reference: &[f32]) {
            assert_eq!(ours.len(), reference.len());
            for (a, b) in ours.iter().zip(reference) {
                self.max_abs = self.max_abs.max((a - b).abs());
            }
            let cosine =
                dot(ours, reference) / (dot(ours, ours).sqrt() * dot(reference, reference).sqrt());
            self.min_cosine = self.min_cosine.min(cosine);
        }
    }

    async fn check_parity(file: &str) {
        let path = format!(
            "{}/tests/fixtures/colbert/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let (model, revision) = (fixture.model.clone(), fixture.revision.clone());
        let embedder = tokio::task::spawn_blocking(move || {
            CandleColbertEmbedder::with_options(
                &model,
                CandleColbertOptions::default().revision(revision),
            )
        })
        .await
        .unwrap()
        .unwrap();
        let dim = fixture.config.dim;
        let layout = &embedder.model.layout;
        assert_eq!(layout.query_maxlen, fixture.config.query_maxlen);
        assert_eq!(layout.doc_maxlen, fixture.config.doc_maxlen);
        assert_eq!(embedder.token_dimension(), dim);

        // Token ids, attention and kept positions match exactly.
        let content = |text: &str| {
            embedder
                .model
                .tokenizer
                .encode(text, false)
                .unwrap()
                .get_ids()
                .to_vec()
        };
        for query in &fixture.queries {
            let encoded = layout.encode(&content(&query.text), EmbedRole::Query);
            assert_eq!(encoded.ids, query.input_ids, "{}", query.text);
            assert_eq!(encoded.attention, query.attention_mask, "{}", query.text);
        }
        for document in &fixture.documents {
            let encoded = layout.encode(&content(&document.text), EmbedRole::Document);
            assert_eq!(encoded.ids, document.input_ids, "{}", document.text);
            let kept: Vec<usize> = (0..encoded.keep.len())
                .filter(|&i| encoded.keep[i])
                .collect();
            assert_eq!(kept, document.kept, "{}", document.text);
        }

        let query_inputs: Vec<EmbedInput<'_>> = fixture
            .queries
            .iter()
            .map(|q| EmbedInput::Text(&q.text))
            .collect();
        let queries = embedder
            .embed_tokens(&query_inputs, EmbedRole::Query)
            .await
            .unwrap();
        let document_inputs: Vec<EmbedInput<'_>> = fixture
            .documents
            .iter()
            .map(|d| EmbedInput::Text(&d.text))
            .collect();
        let documents = embedder
            .embed_tokens(&document_inputs, EmbedRole::Document)
            .await
            .unwrap();

        let mut drift = Drift {
            max_abs: 0.0,
            min_cosine: 1.0,
        };
        for (ours, query) in queries.iter().zip(&fixture.queries) {
            assert_eq!(ours.len(), query.rows, "{}", query.text);
            for (vector, reference) in ours.iter().zip(decode_rows(&query.vectors, dim)) {
                drift.add(&vector.data, &reference);
            }
        }
        for (ours, document) in documents.iter().zip(&fixture.documents) {
            assert_eq!(ours.len(), document.rows, "{}", document.text);
            for (&row, reference) in document
                .stored_rows
                .iter()
                .zip(decode_rows(&document.vectors, dim))
            {
                drift.add(&ours[row].data, &reference);
            }
        }

        let mut max_score_diff = 0.0f32;
        for (query, expected) in queries.iter().zip(&fixture.scores) {
            for (document, expected) in documents.iter().zip(expected) {
                let score: f32 = query
                    .iter()
                    .map(|q| {
                        document
                            .iter()
                            .map(|d| dot(&q.data, &d.data))
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .sum();
                max_score_diff = max_score_diff.max((score - expected).abs());
            }
        }

        println!(
            "{file}: max |Δ| {:.2e}, min cosine {:.8}, max |ΔMaxSim| {:.2e}",
            drift.max_abs, drift.min_cosine, max_score_diff
        );
        // Observed on Apple M4 (candle 0.11 vs. torch 2.14, fp32): max |Δ|
        // below 1e-6 and max |ΔMaxSim| below 2e-5. The limits leave room for
        // other CPUs' summation order; encoding bugs (attending to [MASK], a
        // wrong marker or token type, no normalization) are far above them.
        assert!(drift.max_abs <= 1e-5, "max |Δ| {}", drift.max_abs);
        assert!(
            drift.min_cosine >= 0.99999,
            "min cosine {}",
            drift.min_cosine
        );
        assert!(max_score_diff <= 2e-4, "max |ΔMaxSim| {max_score_diff}");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads colbert-ir/colbertv2.0"]
    async fn colbert_parity_colbertv2() {
        check_parity("colbertv2.0.json").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads answerdotai/answerai-colbert-small-v1"]
    async fn colbert_parity_answerai_small() {
        check_parity("answerai-colbert-small-v1.json").await;
    }
}
