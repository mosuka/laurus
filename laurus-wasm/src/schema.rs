//! WASM wrapper for the Laurus [`Schema`] type.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use laurus::{
    Analyzer, BooleanOption, BytesOption, DateTimeOption, DistanceMetric, DynamicFieldPolicy,
    EmbedderDefinition, FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption,
    IntegerOption, IvfOption, QuantizationMethod, RerankStorageKind, Schema, TextOption,
};
use wasm_bindgen::prelude::*;

use crate::analysis::WasmJapaneseAnalyzer;
use crate::embedder::JsCallbackEmbedder;

/// Parse a distance metric string into [`DistanceMetric`].
///
/// # Arguments
///
/// * `s` - Distance metric name: "cosine", "euclidean", "dot_product"/"dot", "manhattan", "angular".
///
/// # Returns
///
/// The corresponding [`DistanceMetric`] variant.
fn parse_distance(s: &str) -> Result<DistanceMetric, JsValue> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(JsValue::from_str(&format!(
            "Unknown distance metric: '{other}'. Valid: cosine, euclidean, dot_product, manhattan, angular"
        ))),
    }
}

/// Parse a quantizer name plus optional `subvectorCount` into a
/// [`QuantizationMethod`].
///
/// Accepts `"scalar_8bit"` / `"scalar"` (the default when `name` is
/// `None`) and `"product_quantization"` / `"pq"`. Product quantization
/// requires a `subvector_count` (which must divide the field dimension —
/// validated by the core at index-build time); supplying it for any other
/// quantizer is rejected so an incoherent configuration cannot silently
/// reach the core.
fn parse_quantizer(
    name: Option<&str>,
    subvector_count: Option<usize>,
) -> Result<QuantizationMethod, JsValue> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None | Some("scalar_8bit") | Some("scalar") => {
            if subvector_count.is_some() {
                return Err(JsValue::from_str(
                    "subvectorCount is only valid with quantizer='product_quantization'",
                ));
            }
            Ok(QuantizationMethod::Scalar8Bit)
        }
        Some("product_quantization") | Some("pq") => {
            let subvector_count = subvector_count.ok_or_else(|| {
                JsValue::from_str(
                    "quantizer='product_quantization' requires subvectorCount \
                     (must divide the field dimension)",
                )
            })?;
            Ok(QuantizationMethod::ProductQuantization { subvector_count })
        }
        Some(other) => Err(JsValue::from_str(&format!(
            "Unknown quantizer: '{other}'. Valid: scalar_8bit, product_quantization"
        ))),
    }
}

/// Parse a rerank-storage name into an optional [`RerankStorageKind`].
///
/// `None` (the default) keeps the Stage-1 int8-only segment; `"f32"`
/// enables the Stage-2 full-precision rerank sidecar (`*.hnsw.f32`).
fn parse_rerank_storage(name: Option<&str>) -> Result<Option<RerankStorageKind>, JsValue> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None => Ok(None),
        Some("f32") => Ok(Some(RerankStorageKind::F32)),
        Some(other) => Err(JsValue::from_str(&format!(
            "Unknown rerank_storage: '{other}'. Valid: f32"
        ))),
    }
}

/// Schema builder for defining index fields and embedders.
///
/// ```javascript
/// import { WasmSchema } from "laurus-wasm";
///
/// const schema = new WasmSchema();
/// schema.addTextField("title");
/// schema.addHnswField("embedding", 384);
/// schema.setDefaultFields(["title"]);
/// ```
#[wasm_bindgen(js_name = "Schema")]
pub struct WasmSchema {
    pub(crate) inner: Schema,
    /// JS callback embedders registered via `addEmbedder({ type: "callback" })`.
    /// Stored separately because they can't be serialized into `EmbedderDefinition`.
    pub(crate) js_embedders: HashMap<String, JsCallbackEmbedder>,
    /// Pre-constructed analyzers registered via `addAnalyzer(...)`.
    /// These are injected into the Engine at build time as runtime
    /// analyzers because they hold non-serializable state (e.g. a
    /// Lindera dictionary loaded from raw bytes).
    pub(crate) runtime_analyzers: HashMap<String, Arc<dyn Analyzer>>,
}

#[wasm_bindgen(js_class = "Schema")]
impl WasmSchema {
    /// Create a new empty schema.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Schema::new(),
            js_embedders: HashMap::new(),
            runtime_analyzers: HashMap::new(),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the original value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    /// * `term_vectors` - Whether term positions are stored, required by
    ///   phrase and span queries over this field (default `true`).
    /// * `analyzer` - Optional analyzer name. Pass a parameter-less
    ///   built-in directly: `"standard"`, `"english"`, `"keyword"`,
    ///   `"simple"`, `"noop"`. For the Japanese analyzer, build it
    ///   from raw IPADIC bytes via [`JapaneseAnalyzer.fromBytes`] and
    ///   register it on the schema via [`Schema.addAnalyzer`], then
    ///   reference it here by the registered name.
    ///
    /// [`JapaneseAnalyzer.fromBytes`]: crate::analysis::WasmJapaneseAnalyzer::from_bytes
    /// [`Schema.addAnalyzer`]: WasmSchema::add_analyzer
    #[wasm_bindgen(js_name = "addTextField")]
    pub fn add_text_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
        term_vectors: Option<bool>,
        analyzer: Option<String>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::Text(TextOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                term_vectors: term_vectors.unwrap_or(true),
                analyzer: analyzer.map(laurus::AnalyzerSpec::Named),
            }),
        );
    }

    /// Add an integer (i64) field.
    #[wasm_bindgen(js_name = "addIntegerField")]
    pub fn add_integer_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
        multi_valued: Option<bool>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::Integer(IntegerOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                multi_valued: multi_valued.unwrap_or(false),
            }),
        );
    }

    /// Add a float (f64) field.
    #[wasm_bindgen(js_name = "addFloatField")]
    pub fn add_float_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
        multi_valued: Option<bool>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::Float(FloatOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                multi_valued: multi_valued.unwrap_or(false),
            }),
        );
    }

    /// Add a boolean field.
    #[wasm_bindgen(js_name = "addBooleanField")]
    pub fn add_boolean_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Boolean(BooleanOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a date/time field.
    #[wasm_bindgen(js_name = "addDatetimeField")]
    pub fn add_datetime_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::DateTime(DateTimeOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a geographic coordinate field (latitude, longitude).
    #[wasm_bindgen(js_name = "addGeoField")]
    pub fn add_geo_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Geo(GeoOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a 3D ECEF Cartesian point field (x, y, z in meters).
    ///
    /// Values are submitted as a `{ x, y, z }` JSON object and are
    /// queryable via `searchGeo3dDistance`, `searchGeo3dBoundingBox`,
    /// and `searchGeo3dNearest` on `Index`. See the conceptual docs at
    /// `docs/src/concepts/geo3d.md`.
    #[wasm_bindgen(js_name = "addGeo3dField")]
    pub fn add_geo3d_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Geo3d(Geo3dOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a binary data field.
    #[wasm_bindgen(js_name = "addBytesField")]
    pub fn add_bytes_field(&mut self, name: String, stored: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Bytes(BytesOption {
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add an HNSW approximate nearest-neighbor vector index field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default "cosine").
    /// * `m` - HNSW branching factor (default 16).
    /// * `ef_construction` - Build-time expansion factor (default 200).
    /// * `defaultEfSearch` - Schema-level default for the search-time
    ///   `ef_search` candidate-list size (Issue #644). When omitted, the
    ///   searcher uses an internal fallback of 50. Per-query overrides
    ///   via the search request still take precedence.
    /// * `embedder` - Optional embedder name registered via `addEmbedder`.
    /// * `quantizer` - Vector quantizer — "scalar_8bit" (default) or
    ///   "product_quantization" (requires `subvectorCount`).
    /// * `subvectorCount` - Number of PQ sub-vectors. Required when
    ///   `quantizer` is "product_quantization" and must divide `dimension`;
    ///   rejected for other quantizers.
    /// * `rerankStorage` - Stage-2 rerank sidecar — omitted (default) keeps
    ///   the int8-only segment, "f32" stores full-precision vectors in a
    ///   `*.hnsw.f32` sidecar for exact rerank distances.
    /// * `pqCodebookPath` - Storage-relative file name of a shared PQ
    ///   codebook (Issue #631), trained once via the
    ///   `laurus train pq-codebook` CLI command. Only meaningful when
    ///   `quantizer` is "product_quantization"; commits then encode against
    ///   the pre-trained codebook instead of re-training k-means per
    ///   segment. Omitted (default) keeps per-segment training.
    /// * `baseWeight` - This field's relative scoring priority when
    ///   searched alongside other vector fields (Issue #1084). Defaults to
    ///   1.0. Only matters when a query targets two or more specific
    ///   vector fields at once; has no effect on the lexical-vs-vector
    ///   balance of a hybrid search.
    #[wasm_bindgen(js_name = "addHnswField")]
    #[allow(clippy::too_many_arguments)]
    pub fn add_hnsw_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        m: Option<u32>,
        ef_construction: Option<u32>,
        default_ef_search: Option<u32>,
        embedder: Option<String>,
        quantizer: Option<String>,
        subvector_count: Option<u32>,
        rerank_storage: Option<String>,
        pq_codebook_path: Option<String>,
        base_weight: Option<f32>,
    ) -> Result<(), JsValue> {
        let opt = HnswOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            m: m.unwrap_or(16) as usize,
            ef_construction: ef_construction.unwrap_or(200) as usize,
            default_ef_search: default_ef_search.map(|v| v as usize),
            quantizer: parse_quantizer(quantizer.as_deref(), subvector_count.map(|v| v as usize))?,
            rerank_storage: parse_rerank_storage(rerank_storage.as_deref())?,
            embedder,
            pq_codebook_path,
            base_weight: base_weight.unwrap_or(1.0),
        };
        self.inner.fields.insert(name, FieldOption::Hnsw(opt));
        Ok(())
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// # Arguments
    ///
    /// * `baseWeight` - This field's relative scoring priority when
    ///   searched alongside other vector fields (Issue #1084). Defaults to
    ///   1.0. See `addHnswField` for the full contract.
    #[wasm_bindgen(js_name = "addFlatField")]
    pub fn add_flat_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        embedder: Option<String>,
        base_weight: Option<f32>,
    ) -> Result<(), JsValue> {
        let opt = FlatOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            embedder,
            base_weight: base_weight.unwrap_or(1.0),
            ..Default::default()
        };
        self.inner.fields.insert(name, FieldOption::Flat(opt));
        Ok(())
    }

    /// Add an IVF approximate nearest-neighbor vector field.
    ///
    /// # Arguments
    ///
    /// * `baseWeight` - This field's relative scoring priority when
    ///   searched alongside other vector fields (Issue #1084). Defaults to
    ///   1.0. See `addHnswField` for the full contract.
    #[wasm_bindgen(js_name = "addIvfField")]
    #[allow(clippy::too_many_arguments)]
    pub fn add_ivf_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        n_clusters: Option<u32>,
        n_probe: Option<u32>,
        embedder: Option<String>,
        base_weight: Option<f32>,
    ) -> Result<(), JsValue> {
        let opt = IvfOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            n_clusters: n_clusters.unwrap_or(100) as usize,
            n_probe: n_probe.unwrap_or(1) as usize,
            embedder,
            base_weight: base_weight.unwrap_or(1.0),
            ..Default::default()
        };
        self.inner.fields.insert(name, FieldOption::Ivf(opt));
        Ok(())
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The `config` object must have a `type` key:
    ///
    /// | type           | extra keys | description                             |
    /// |----------------|------------|-----------------------------------------|
    /// | `"precomputed"` | —         | No embedding; vectors passed directly   |
    /// | `"callback"`   | `embed`    | JS function `(text) => Promise<number[]>` |
    ///
    /// ## Example — callback embedder with Transformers.js
    ///
    /// ```javascript
    /// const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    /// schema.addEmbedder("my-bert", {
    ///   type: "callback",
    ///   embed: async (text) => {
    ///     const out = await model(text, { pooling: 'mean', normalize: true });
    ///     return Array.from(out.data);
    ///   }
    /// });
    /// ```
    #[wasm_bindgen(js_name = "addEmbedder")]
    pub fn add_embedder(&mut self, name: String, config: JsValue) -> Result<(), JsValue> {
        // Read the "type" field from the config object
        let type_key = js_sys::Reflect::get(&config, &JsValue::from_str("type"))
            .map_err(|_| JsValue::from_str("Embedder config must have a 'type' key"))?;
        let embedder_type = type_key
            .as_string()
            .ok_or_else(|| JsValue::from_str("Embedder 'type' must be a string"))?;

        match embedder_type.as_str() {
            "precomputed" => {
                self.inner
                    .embedders
                    .insert(name, EmbedderDefinition::Precomputed);
            }
            "callback" => {
                // Extract the "embed" function from the config
                let embed_fn =
                    js_sys::Reflect::get(&config, &JsValue::from_str("embed")).map_err(|_| {
                        JsValue::from_str("Callback embedder config must have an 'embed' key")
                    })?;
                let func = embed_fn.dyn_into::<js_sys::Function>().map_err(|_| {
                    JsValue::from_str(
                        "'embed' must be a function: (text: string) => Promise<number[]>",
                    )
                })?;

                // Register as precomputed in the schema (so the engine creates the field)
                // and store the actual JS embedder separately.
                self.inner
                    .embedders
                    .insert(name.clone(), EmbedderDefinition::Precomputed);
                self.js_embedders
                    .insert(name.clone(), JsCallbackEmbedder::new(name, func));
            }
            other => {
                return Err(JsValue::from_str(&format!(
                    "Unsupported embedder type: '{other}'. Valid: 'precomputed', 'callback'"
                )));
            }
        }

        Ok(())
    }

    /// Register a pre-built analyzer under a name.
    ///
    /// The registered analyzer takes precedence over the parameter-less
    /// built-in names (`"standard"`, `"english"`, `"keyword"`, `"simple"`,
    /// `"noop"`) and over `schema.analyzers` definitions when text fields
    /// reference an analyzer by `Named` form.
    ///
    /// Currently only Japanese analyzers built via
    /// [`JapaneseAnalyzer.fromBytes`] are supported here. The runtime
    /// registry is the only practical way to use the Japanese analyzer in
    /// browser WASM, where the `{ "language": "japanese", "dict": ... }`
    /// preset cannot resolve a filesystem path.
    ///
    /// ```javascript
    /// const ja = JapaneseAnalyzer.fromBytes(...);
    /// schema.addAnalyzer("ja-ipadic", ja);
    /// schema.addTextField("body", undefined, undefined, undefined, "ja-ipadic");
    /// ```
    ///
    /// [`JapaneseAnalyzer.fromBytes`]: crate::analysis::WasmJapaneseAnalyzer::from_bytes
    #[wasm_bindgen(js_name = "addAnalyzer")]
    pub fn add_analyzer(&mut self, name: String, analyzer: &WasmJapaneseAnalyzer) {
        self.runtime_analyzers.insert(name, analyzer.analyzer());
    }

    /// Set the default fields used when no field is specified in a query.
    #[wasm_bindgen(js_name = "setDefaultFields")]
    pub fn set_default_fields(&mut self, fields: Vec<String>) {
        self.inner.default_fields = fields;
    }

    /// Set the policy for fields that are not declared in this schema.
    ///
    /// Accepted values (case-insensitive): `"strict"`, `"dynamic"`,
    /// `"ignore"`. Behaviour:
    ///
    /// - `"strict"`: reject documents containing undeclared fields.
    /// - `"dynamic"` (default): infer a type for each undeclared field and
    ///   add it to the schema during ingestion. **Warning**: integer fields
    ///   silently truncate incoming float values (e.g. `3.14` → `3`).
    /// - `"ignore"`: silently drop undeclared fields.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error string if `policy` is not one of the
    /// accepted names.
    #[wasm_bindgen(js_name = "setDynamicFieldPolicy")]
    pub fn set_dynamic_field_policy(&mut self, policy: String) -> Result<(), JsValue> {
        let parsed =
            DynamicFieldPolicy::from_str(&policy).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner.dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (`"strict"` / `"dynamic"` / `"ignore"`).
    #[wasm_bindgen(js_name = "dynamicFieldPolicy")]
    pub fn dynamic_field_policy(&self) -> String {
        match self.inner.dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict".into(),
            DynamicFieldPolicy::Dynamic => "dynamic".into(),
            DynamicFieldPolicy::Ignore => "ignore".into(),
        }
    }

    /// Return the list of field names defined in this schema.
    #[wasm_bindgen(js_name = "fieldNames")]
    pub fn field_names(&self) -> Vec<String> {
        self.inner.fields.keys().cloned().collect()
    }

    /// Return a string representation of this schema, listing the declared
    /// field names. Convenient for `console.log` / `String(schema)`.
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_repr(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.fields.keys().collect::<Vec<_>>()
        )
    }
}
