//! Hugging Face Hub access shared by the candle embedders.

use std::path::PathBuf;

use hf_hub::{HFClientBuilder, HFError, HFRepositorySync, RepoTypeModel, split_id};

use crate::error::{LaurusError, Result};

/// A model repository on the Hugging Face Hub, optionally pinned to a
/// revision.
pub(crate) struct HubModel {
    repo: HFRepositorySync<RepoTypeModel>,
    revision: Option<String>,
    model_id: String,
}

impl HubModel {
    /// Open the repository `model_id` (`"owner/name"`).
    ///
    /// # Arguments
    ///
    /// * `revision` - Branch, tag or commit to download from; `None` means
    ///   the default branch.
    /// * `cache_dir` - Cache directory override; `None` keeps hf-hub's
    ///   default (`HF_HUB_CACHE`, then `$HF_HOME/hub`, then
    ///   `~/.cache/huggingface/hub`).
    pub(crate) fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let mut builder = HFClientBuilder::new();
        if let Some(dir) = cache_dir {
            builder = builder.cache_dir(dir);
        }
        let client = builder.build_sync().map_err(|e| {
            LaurusError::InvalidOperation(format!("HF API initialization failed: {e}"))
        })?;
        let (owner, name) = split_id(model_id);
        Ok(Self {
            repo: client.model(owner, name),
            revision: revision.map(str::to_string),
            model_id: model_id.to_string(),
        })
    }

    /// Path of `filename`, downloaded unless already cached.
    pub(crate) fn file(&self, filename: &str) -> Result<PathBuf> {
        self.fetch(filename)
            .map_err(|e| self.download_error(filename, e))
    }

    /// Like [`Self::file`], but `Ok(None)` when the repository has no such
    /// file. Every other failure is still an error, so a network problem
    /// never passes for a missing file.
    #[cfg(feature = "embeddings-candle")]
    pub(crate) fn optional_file(&self, filename: &str) -> Result<Option<PathBuf>> {
        match self.fetch(filename) {
            Ok(path) => Ok(Some(path)),
            Err(HFError::EntryNotFound { .. }) => Ok(None),
            Err(e) => Err(self.download_error(filename, e)),
        }
    }

    fn fetch(&self, filename: &str) -> std::result::Result<PathBuf, HFError> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
    }

    fn download_error(&self, filename: &str, err: HFError) -> LaurusError {
        LaurusError::InvalidOperation(format!(
            "Failed to download '{filename}' from '{}': {err}",
            self.model_id
        ))
    }
}

/// The cache directory `CandleBertEmbedder` and `CandleClipEmbedder` have
/// always used: `$HF_HOME`, then `~/.cache/huggingface`, then
/// `/tmp/huggingface` (not hf-hub's `.../hub` default, see Issue #1355).
pub(crate) fn legacy_cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HOME").map(|home| format!("{home}/.cache/huggingface")))
        .unwrap_or_else(|_| "/tmp/huggingface".to_string())
        .into()
}
