//! JavaScript callback embedder for WASM environments.
//!
//! Bridges the [`Embedder`] trait to a JavaScript function, enabling
//! in-engine automatic embedding powered by browser-side models
//! (e.g. Transformers.js). This gives WASM users the same Unified Query
//! DSL experience as native platforms.

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use wasm_bindgen::prelude::*;

use laurus::embedding::embedder::{EmbedInput, EmbedInputType, EmbedRole, Embedder, TokenEmbedder};
use laurus::vector::core::vector::Vector;
use laurus::{LaurusError, Result};

// ---------------------------------------------------------------------------
// Send wrapper for !Send futures (safe in single-threaded WASM)
// ---------------------------------------------------------------------------

/// A wrapper that marks a `!Send` future as `Send`.
///
/// # Safety
///
/// This is only safe in single-threaded environments (i.e. `wasm32-unknown-unknown`).
/// The future will never actually be sent across threads.
struct AssertSend<F>(F);

// SAFETY: WASM is single-threaded. The future will only ever be polled
// on the main (and only) thread.
unsafe impl<F> Send for AssertSend<F> {}

impl<F: Future> Future for AssertSend<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner future after pinning.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// JsFunction wrapper (Send + Sync for single-threaded WASM)
// ---------------------------------------------------------------------------

struct JsFunction(js_sys::Function);

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsFunction {}
unsafe impl Sync for JsFunction {}

// ---------------------------------------------------------------------------
// JsCallbackEmbedder
// ---------------------------------------------------------------------------

/// An [`Embedder`] implementation that delegates to a JavaScript callback.
///
/// The JS function receives a string and must return a `Promise<number[]>`
/// (an array of floats representing the embedding vector).
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { pipeline } from '@huggingface/transformers';
///
/// const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
///
/// schema.addEmbedder("my-bert", {
///   type: "callback",
///   embed: async (text) => {
///     const output = await model(text, { pooling: 'mean', normalize: true });
///     return Array.from(output.data);
///   }
/// });
/// ```
pub struct JsCallbackEmbedder {
    func: Arc<JsFunction>,
    name: String,
}

impl fmt::Debug for JsCallbackEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsCallbackEmbedder")
            .field("name", &self.name)
            .finish()
    }
}

impl JsCallbackEmbedder {
    /// Create a new JS callback embedder.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for this embedder (used in logging).
    /// * `func` - A JS function `(text: string) => Promise<number[]>`.
    pub fn new(name: String, func: js_sys::Function) -> Self {
        Self {
            func: Arc::new(JsFunction(func)),
            name,
        }
    }

    /// Call the JS function and await the result.
    async fn embed_text(&self, text: &str) -> Result<Vector> {
        // Call: func(text) -> Promise<number[]>
        let js_text = JsValue::from_str(text);
        let promise = self
            .func
            .0
            .call1(&JsValue::NULL, &js_text)
            .map_err(|e| LaurusError::internal(format!("JS embedder call failed: {e:?}")))?;

        // Await the Promise
        let js_result = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| LaurusError::internal(format!("JS embedder promise rejected: {e:?}")))?;

        // Convert number[] to Vec<f32>
        let js_array = js_sys::Array::from(&js_result);
        let vec: Vec<f32> = js_array
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if vec.is_empty() {
            return Err(LaurusError::internal(
                "JS embedder returned an empty vector",
            ));
        }

        Ok(Vector::new(vec))
    }
}

// Manual `Embedder` trait implementation.
//
// We do NOT use `#[async_trait]` here because the macro generates
// `Pin<Box<dyn Future + Send>>` and `JsFuture` is `!Send`.
// Instead we manually return `AssertSend`-wrapped futures, which is
// safe because WASM is single-threaded.
impl Embedder for JsCallbackEmbedder {
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn embed<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        input: &'life1 EmbedInput<'life2>,
    ) -> Pin<Box<dyn Future<Output = Result<Vector>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        let text = input.as_text().map(|s| s.to_string());
        Box::pin(AssertSend(async move {
            let text = text.ok_or_else(|| {
                LaurusError::invalid_argument("JsCallbackEmbedder only supports text input")
            })?;
            self.embed_text(&text).await
        }))
    }
}

// ---------------------------------------------------------------------------
// JsTokenCallbackEmbedder
// ---------------------------------------------------------------------------

/// A token-level embedder that delegates to a JavaScript callback, serving
/// a multi-vector field and the late-interaction rescore (Issue #1351).
///
/// The JS function receives the text and the role (`"query"` or
/// `"document"`) and returns the token vectors as `number[][]`, or a
/// Promise of them.
///
/// # Example (JavaScript)
///
/// ```javascript
/// schema.addEmbedder("colbert", {
///   type: "token_callback",
///   embed: async (text, role) => myColbert.encode(text, role),
///   dimension: 128,
/// });
/// schema.addMultiVectorField("body_colbert", 128, "cosine", "colbert");
/// ```
pub struct JsTokenCallbackEmbedder {
    func: Arc<JsFunction>,
    name: String,
    dimension: usize,
}

impl fmt::Debug for JsTokenCallbackEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsTokenCallbackEmbedder")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl JsTokenCallbackEmbedder {
    /// Create a new JS token callback embedder.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for this embedder (used in logging).
    /// * `func` - A JS function
    ///   `(text: string, role: "query" | "document") => number[][] | Promise<number[][]>`.
    /// * `dimension` - The dimension of every token vector the function
    ///   returns, checked against the field when the index opens.
    pub fn new(name: String, func: js_sys::Function, dimension: usize) -> Self {
        Self {
            func: Arc::new(JsFunction(func)),
            name,
            dimension,
        }
    }

    /// Call the JS function on one text and await its token vectors.
    async fn embed_text(&self, text: &str, role: EmbedRole) -> Result<Vec<Vector>> {
        let role = match role {
            EmbedRole::Query => "query",
            EmbedRole::Document => "document",
        };
        let returned = self
            .func
            .0
            .call2(
                &JsValue::NULL,
                &JsValue::from_str(text),
                &JsValue::from_str(role),
            )
            .map_err(|e| LaurusError::internal(format!("JS token embedder call failed: {e:?}")))?;
        // `Promise.resolve` accepts a plain return value as well as a Promise.
        let resolved = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&returned))
            .await
            .map_err(|e| {
                LaurusError::internal(format!("JS token embedder promise rejected: {e:?}"))
            })?;
        token_vectors_from_js(&resolved)
    }
}

/// Read the `number[][]` a JS token embedder returned. A non-numeric
/// element is an error rather than a silent 0.
fn token_vectors_from_js(value: &JsValue) -> Result<Vec<Vector>> {
    let not_token_vectors = || LaurusError::internal("JS token embedder must return number[][]");
    if !js_sys::Array::is_array(value) {
        return Err(not_token_vectors());
    }
    js_sys::Array::from(value)
        .iter()
        .enumerate()
        .map(|(i, row)| {
            if !js_sys::Array::is_array(&row) {
                return Err(not_token_vectors());
            }
            js_sys::Array::from(&row)
                .iter()
                .map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                        LaurusError::internal(format!(
                            "JS token embedder returned a non-number in token vector {i}"
                        ))
                    })
                })
                .collect::<Result<Vec<f32>>>()
                .map(Vector::new)
        })
        .collect()
}

// Manual trait implementations, for the same `!Send` reason as
// `JsCallbackEmbedder` above.
impl TokenEmbedder for JsTokenCallbackEmbedder {
    fn embed_tokens<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        inputs: &'life1 [EmbedInput<'life2>],
        role: EmbedRole,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<Vector>>>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        let texts: Option<Vec<String>> = inputs
            .iter()
            .map(|input| input.as_text().map(str::to_string))
            .collect();
        Box::pin(AssertSend(async move {
            let texts = texts.ok_or_else(|| {
                LaurusError::invalid_argument("JsTokenCallbackEmbedder only supports text input")
            })?;
            let mut out = Vec::with_capacity(texts.len());
            for text in &texts {
                out.push(self.embed_text(text, role).await?);
            }
            Ok(out)
        }))
    }

    fn token_dimension(&self) -> usize {
        self.dimension
    }
}

impl Embedder for JsTokenCallbackEmbedder {
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_token_embedder(&self) -> Option<&dyn TokenEmbedder> {
        Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Always an error: this embedder produces token vectors, not one
    /// vector per input.
    fn embed<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        _input: &'life1 EmbedInput<'life2>,
    ) -> Pin<Box<dyn Future<Output = Result<Vector>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        let name = self.name.clone();
        Box::pin(async move {
            Err(LaurusError::invalid_argument(format!(
                "'{name}' is a token-level embedder; use it for a multi-vector field"
            )))
        })
    }
}
