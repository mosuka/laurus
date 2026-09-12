//! Ruby wrapper for the Laurus [`Schema`] type.

use std::cell::RefCell;
use std::str::FromStr;

use laurus::{
    BooleanOption, BytesOption, DateTimeOption, DistanceMetric, DynamicFieldPolicy,
    EmbedderDefinition, FieldOption, FloatOption, Geo3dOption, GeoOption, HnswOption,
    IntegerOption, IvfOption, QuantizationMethod, RerankStorageKind, Schema, TextOption,
};
use magnus::prelude::*;
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Error, RArray, RHash, RModule, Ruby, TryConvert, Value};

/// Parse a distance metric string into [`DistanceMetric`].
fn parse_distance(s: &str) -> Result<DistanceMetric, Error> {
    let ruby = Ruby::get().expect("called from Ruby thread");
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(Error::new(
            ruby.exception_arg_error(),
            format!(
                "Unknown distance metric: '{}'. Valid: cosine, euclidean, dot_product, manhattan, angular",
                other
            ),
        )),
    }
}

/// Extract `base_weight:` from a splat `RHash` of otherwise-unrecognized
/// keyword arguments, defaulting to `1.0` when absent.
///
/// `base_weight` is pulled out via the `get_kwargs` `Splat` slot (an
/// `RHash` of everything not already named in `optional`) rather than
/// being added directly to the `Opt` tuple, because `magnus::scan_args`'s
/// `ScanArgsOpt` is only implemented for tuples up to 9 elements
/// (Issue #1084) and the HNSW field builder's existing option list
/// already uses all 9.
fn base_weight_from_splat(splat: RHash) -> Result<f32, Error> {
    let ruby = Ruby::get().expect("called from Ruby thread");
    let value: Option<f64> = splat
        .get(ruby.to_symbol("base_weight"))
        .map(TryConvert::try_convert)
        .transpose()?;
    Ok(value.unwrap_or(1.0) as f32)
}

/// Parse a quantizer name plus optional `subvector_count` into a
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
) -> Result<QuantizationMethod, Error> {
    let ruby = Ruby::get().expect("called from Ruby thread");
    match name.map(|s| s.to_lowercase()).as_deref() {
        None | Some("scalar_8bit") | Some("scalar") => {
            if subvector_count.is_some() {
                return Err(Error::new(
                    ruby.exception_arg_error(),
                    "subvector_count is only valid with quantizer: 'product_quantization'",
                ));
            }
            Ok(QuantizationMethod::Scalar8Bit)
        }
        Some("product_quantization") | Some("pq") => {
            let subvector_count = subvector_count.ok_or_else(|| {
                Error::new(
                    ruby.exception_arg_error(),
                    "quantizer: 'product_quantization' requires subvector_count \
                     (must divide the field dimension)",
                )
            })?;
            Ok(QuantizationMethod::ProductQuantization { subvector_count })
        }
        Some(other) => Err(Error::new(
            ruby.exception_arg_error(),
            format!("Unknown quantizer: '{other}'. Valid: scalar_8bit, product_quantization"),
        )),
    }
}

/// Parse a rerank-storage name into an optional [`RerankStorageKind`].
///
/// `None` (the default) keeps the Stage-1 int8-only segment; `"f32"`
/// enables the Stage-2 full-precision rerank sidecar (`*.hnsw.f32`).
fn parse_rerank_storage(name: Option<&str>) -> Result<Option<RerankStorageKind>, Error> {
    let ruby = Ruby::get().expect("called from Ruby thread");
    match name.map(|s| s.to_lowercase()).as_deref() {
        None => Ok(None),
        Some("f32") => Ok(Some(RerankStorageKind::F32)),
        Some(other) => Err(Error::new(
            ruby.exception_arg_error(),
            format!("Unknown rerank_storage: '{other}'. Valid: f32"),
        )),
    }
}

/// Ruby-facing schema builder (`Laurus::Schema`).
///
/// Uses `RefCell` for interior mutability since magnus methods receive `&self`.
#[magnus::wrap(class = "Laurus::Schema")]
pub struct RbSchema {
    pub inner: RefCell<Schema>,
}

impl RbSchema {
    /// Create a new empty schema.
    fn new() -> Self {
        Self {
            inner: RefCell::new(Schema::new()),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the original value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `term_vectors:` (bool, default true): Whether term positions are
    ///     stored, required by phrase and span queries over this field.
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues, the column-oriented store
    ///     sort/facet/aggregation read from. Takes effect only when
    ///     `stored:` is also true.
    ///   - `analyzer:` (String, optional): Analyzer name. For
    ///     parameter-less built-ins (`"standard"`, `"english"`,
    ///     `"keyword"`, `"simple"`, `"noop"`) pass the name directly.
    ///     For parameterized presets such as the Japanese analyzer
    ///     (which needs a Lindera dictionary path), register a custom
    ///     analyzer via `add_analyzer` and reference it by name.
    fn add_text_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs = get_kwargs::<
            _,
            (),
            (
                Option<bool>,
                Option<bool>,
                Option<bool>,
                Option<bool>,
                Option<Option<String>>,
            ),
            (),
        >(
            args.keywords,
            &[],
            &[
                "stored",
                "indexed",
                "term_vectors",
                "doc_values",
                "analyzer",
            ],
        )?;
        let (stored, indexed, term_vectors, doc_values, analyzer) = kwargs.optional;
        let stored = stored.unwrap_or(true);
        let indexed = indexed.unwrap_or(true);
        let term_vectors = term_vectors.unwrap_or(true);
        let doc_values = doc_values.unwrap_or(true);
        let analyzer = analyzer.flatten().map(laurus::AnalyzerSpec::Named);
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Text(TextOption {
                indexed,
                stored,
                term_vectors,
                doc_values,
                analyzer,
            }),
        );
        Ok(())
    }

    /// Add an integer (i64) field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `multi_valued:` (bool, default false): When true, the field
    ///     accepts arrays of integers and range queries match if any value
    ///     satisfies the predicate (Lucene-style "any match").
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues. Takes effect only when `stored:` is
    ///     also true.
    fn add_integer_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs =
            get_kwargs::<_, (), (Option<bool>, Option<bool>, Option<bool>, Option<bool>), ()>(
                args.keywords,
                &[],
                &["stored", "indexed", "multi_valued", "doc_values"],
            )?;
        let (stored, indexed, multi_valued, doc_values) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Integer(IntegerOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                multi_valued: multi_valued.unwrap_or(false),
                doc_values: doc_values.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add a float (f64) field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `multi_valued:` (bool, default false): When true, the field
    ///     accepts arrays of floats and range queries match if any value
    ///     satisfies the predicate (Lucene-style "any match").
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues. Takes effect only when `stored:` is
    ///     also true.
    fn add_float_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs =
            get_kwargs::<_, (), (Option<bool>, Option<bool>, Option<bool>, Option<bool>), ()>(
                args.keywords,
                &[],
                &["stored", "indexed", "multi_valued", "doc_values"],
            )?;
        let (stored, indexed, multi_valued, doc_values) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Float(FloatOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                multi_valued: multi_valued.unwrap_or(false),
                doc_values: doc_values.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add a boolean field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues. Takes effect only when `stored:` is
    ///     also true.
    fn add_boolean_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<bool>, Option<bool>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["stored", "indexed", "doc_values"],
        )?;
        let (stored, indexed, doc_values) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Boolean(BooleanOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                doc_values: doc_values.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add a date/time field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues. Takes effect only when `stored:` is
    ///     also true.
    fn add_datetime_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<bool>, Option<bool>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["stored", "indexed", "doc_values"],
        )?;
        let (stored, indexed, doc_values) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::DateTime(DateTimeOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                doc_values: doc_values.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add a geographic coordinate field (latitude, longitude).
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues. Takes effect only when `stored:` is
    ///     also true.
    fn add_geo_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<bool>, Option<bool>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["stored", "indexed", "doc_values"],
        )?;
        let (stored, indexed, doc_values) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Geo(GeoOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                doc_values: doc_values.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add a 3D ECEF Cartesian point field (x, y, z in meters).
    ///
    /// Values are submitted as a Hash `{ "x" => ..., "y" => ..., "z" => ... }`
    /// and are queryable via `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`,
    /// and `Geo3dNearestQuery`. See the conceptual docs at
    /// `docs/src/concepts/geo3d.md`.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    ///   - `indexed:` (bool, default true): Whether the field is searchable.
    ///   - `doc_values:` (bool, default true): Whether the value is also
    ///     copied into DocValues. Takes effect only when `stored:` is
    ///     also true.
    fn add_geo3d_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<bool>, Option<bool>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["stored", "indexed", "doc_values"],
        )?;
        let (stored, indexed, doc_values) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Geo3d(Geo3dOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                doc_values: doc_values.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add a binary data field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `stored:` (bool, default true): Whether the value is retrievable.
    fn add_bytes_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (name,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<bool>,), ()>(args.keywords, &[], &["stored"])?;
        let (stored,) = kwargs.optional;
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Bytes(BytesOption {
                stored: stored.unwrap_or(true),
            }),
        );
        Ok(())
    }

    /// Add an HNSW approximate nearest-neighbor vector index field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `dimension` (usize): Vector dimensionality.
    ///   - `distance:` (String, default "cosine"): Distance metric.
    ///   - `m:` (usize, default 16): HNSW branching factor.
    ///   - `ef_construction:` (usize, default 200): Build-time expansion factor.
    ///   - `default_ef_search:` (usize, optional): Schema-level default for the
    ///     search-time `ef_search` candidate-list size (Issue #644). When
    ///     omitted, the searcher uses an internal fallback of 50. Per-query
    ///     overrides via the search request still take precedence.
    ///   - `embedder:` (String, optional): Embedder name registered via `add_embedder`.
    ///   - `quantizer:` (String, optional): Vector quantizer — "scalar_8bit"
    ///     (default) or "product_quantization" (requires `subvector_count`).
    ///   - `subvector_count:` (usize, optional): Number of PQ sub-vectors.
    ///     Required when `quantizer:` is "product_quantization" and must
    ///     divide `dimension`; rejected for other quantizers.
    ///   - `rerank_storage:` (String, optional): Stage-2 rerank sidecar —
    ///     omitted (default) keeps the int8-only segment, "f32" stores
    ///     full-precision vectors in a `*.hnsw.f32` sidecar for exact rerank.
    ///   - `pq_codebook_path:` (String, optional): Storage-relative file
    ///     name of a shared PQ codebook (Issue #631), trained once via the
    ///     `laurus train pq-codebook` CLI command. Only meaningful with
    ///     `quantizer:` "product_quantization"; commits then encode against
    ///     the pre-trained codebook instead of re-training k-means per
    ///     segment. Omitted (default) keeps per-segment training.
    ///   - `base_weight:` (Float, default 1.0): This field's relative
    ///     scoring priority when searched alongside other vector fields
    ///     (Issue #1084). Only matters when a query targets two or more
    ///     specific vector fields at once; has no effect on the
    ///     lexical-vs-vector balance of a hybrid search.
    fn add_hnsw_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String, usize), (), (), (), RHash, ()>(args)?;
        let (name, dimension) = args.required;
        let kwargs = get_kwargs::<
            _,
            (),
            (
                Option<String>,
                Option<usize>,
                Option<usize>,
                Option<usize>,
                Option<Option<String>>,
                Option<String>,
                Option<usize>,
                Option<String>,
                Option<String>,
            ),
            RHash,
        >(
            args.keywords,
            &[],
            &[
                "distance",
                "m",
                "ef_construction",
                "default_ef_search",
                "embedder",
                "quantizer",
                "subvector_count",
                "rerank_storage",
                "pq_codebook_path",
            ],
        )?;
        let (
            distance,
            m,
            ef_construction,
            default_ef_search,
            embedder,
            quantizer,
            subvector_count,
            rerank_storage,
            pq_codebook_path,
        ) = kwargs.optional;
        let distance_str = distance.as_deref().unwrap_or("cosine");
        let opt = HnswOption {
            dimension,
            distance: parse_distance(distance_str)?,
            m: m.unwrap_or(16),
            ef_construction: ef_construction.unwrap_or(200),
            default_ef_search,
            quantizer: parse_quantizer(quantizer.as_deref(), subvector_count)?,
            rerank_storage: parse_rerank_storage(rerank_storage.as_deref())?,
            embedder: embedder.flatten(),
            pq_codebook_path,
            base_weight: base_weight_from_splat(kwargs.splat)?,
        };
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Hnsw(opt));
        Ok(())
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `dimension` (usize): Vector dimensionality.
    ///   - `distance:` (String, default "cosine"): Distance metric.
    ///   - `embedder:` (String, optional): Embedder name registered via `add_embedder`.
    ///   - `base_weight:` (Float, default 1.0): This field's relative
    ///     scoring priority when searched alongside other vector fields
    ///     (Issue #1084). See `add_hnsw_field` for the full contract.
    fn add_flat_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String, usize), (), (), (), RHash, ()>(args)?;
        let (name, dimension) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<String>, Option<Option<String>>, Option<f64>), ()>(
            args.keywords,
            &[],
            &["distance", "embedder", "base_weight"],
        )?;
        let (distance, embedder, base_weight) = kwargs.optional;
        let distance_str = distance.as_deref().unwrap_or("cosine");
        let opt = laurus::FlatOption {
            dimension,
            distance: parse_distance(distance_str)?,
            embedder: embedder.flatten(),
            base_weight: base_weight.unwrap_or(1.0) as f32,
            ..Default::default()
        };
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Flat(opt));
        Ok(())
    }

    /// Add an IVF (Inverted File Index) approximate nearest-neighbor vector field.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `name` (String): Field name.
    ///   - `dimension` (usize): Vector dimensionality.
    ///   - `distance:` (String, default "cosine"): Distance metric.
    ///   - `n_clusters:` (usize, default 100): Number of Voronoi clusters.
    ///   - `n_probe:` (usize, default 1): Number of clusters to probe at search time.
    ///   - `embedder:` (String, optional): Embedder name registered via `add_embedder`.
    ///   - `base_weight:` (Float, default 1.0): This field's relative
    ///     scoring priority when searched alongside other vector fields
    ///     (Issue #1084). See `add_hnsw_field` for the full contract.
    fn add_ivf_field(&self, args: &[Value]) -> Result<(), Error> {
        let args = scan_args::<(String, usize), (), (), (), RHash, ()>(args)?;
        let (name, dimension) = args.required;
        let kwargs = get_kwargs::<
            _,
            (),
            (
                Option<String>,
                Option<usize>,
                Option<usize>,
                Option<Option<String>>,
                Option<f64>,
            ),
            (),
        >(
            args.keywords,
            &[],
            &[
                "distance",
                "n_clusters",
                "n_probe",
                "embedder",
                "base_weight",
            ],
        )?;
        let (distance, n_clusters, n_probe, embedder, base_weight) = kwargs.optional;
        let distance_str = distance.as_deref().unwrap_or("cosine");
        let opt = IvfOption {
            dimension,
            distance: parse_distance(distance_str)?,
            n_clusters: n_clusters.unwrap_or(100),
            n_probe: n_probe.unwrap_or(1),
            embedder: embedder.flatten(),
            base_weight: base_weight.unwrap_or(1.0) as f32,
            ..Default::default()
        };
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Ivf(opt));
        Ok(())
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The `config` Hash must have a `"type"` key selecting the backend:
    ///
    /// | type              | required keys | feature flag            |
    /// |-------------------|---------------|-------------------------|
    /// | `"precomputed"`   | —             | (always available)      |
    /// | `"candle_bert"`   | `"model"`     | `embeddings-candle`     |
    /// | `"candle_clip"`   | `"model"`     | `embeddings-multimodal` |
    /// | `"openai"`        | `"model"`     | `embeddings-openai`     |
    ///
    /// # Arguments
    ///
    /// * `name` - Unique embedder name referenced from vector fields.
    /// * `config` - Hash describing the embedder.
    fn add_embedder(&self, name: String, config: RHash) -> Result<(), Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let embedder_type: Option<Value> = config.get(ruby.str_new("type"));
        let embedder_type: String = embedder_type
            .ok_or_else(|| {
                Error::new(
                    ruby.exception_arg_error(),
                    "embedder config must have a 'type' key",
                )
            })
            .and_then(magnus::TryConvert::try_convert)?;

        let definition = match embedder_type.as_str() {
            "precomputed" => EmbedderDefinition::Precomputed,
            "candle_bert" => {
                let model_val: Option<Value> = config.get(ruby.str_new("model"));
                let model: String = model_val
                    .ok_or_else(|| {
                        Error::new(
                            ruby.exception_arg_error(),
                            "candle_bert embedder requires a 'model' key",
                        )
                    })
                    .and_then(magnus::TryConvert::try_convert)?;
                EmbedderDefinition::CandleBert { model }
            }
            "candle_clip" => {
                let model_val: Option<Value> = config.get(ruby.str_new("model"));
                let model: String = model_val
                    .ok_or_else(|| {
                        Error::new(
                            ruby.exception_arg_error(),
                            "candle_clip embedder requires a 'model' key",
                        )
                    })
                    .and_then(magnus::TryConvert::try_convert)?;
                EmbedderDefinition::CandleClip { model }
            }
            "openai" => {
                let model_val: Option<Value> = config.get(ruby.str_new("model"));
                let model: String = model_val
                    .ok_or_else(|| {
                        Error::new(
                            ruby.exception_arg_error(),
                            "openai embedder requires a 'model' key",
                        )
                    })
                    .and_then(magnus::TryConvert::try_convert)?;
                EmbedderDefinition::Openai { model }
            }
            other => {
                return Err(Error::new(
                    ruby.exception_arg_error(),
                    format!(
                        "Unknown embedder type: '{}'. Valid types: precomputed, candle_bert, candle_clip, openai",
                        other
                    ),
                ));
            }
        };

        self.inner.borrow_mut().embedders.insert(name, definition);
        Ok(())
    }

    /// Set the default fields used when no field is specified in a query.
    ///
    /// # Arguments
    ///
    /// * `fields` - Array of field name strings.
    fn set_default_fields(&self, fields: RArray) -> Result<(), Error> {
        let field_names: Vec<String> = fields.to_vec()?;
        self.inner.borrow_mut().default_fields = field_names;
        Ok(())
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
    /// # Arguments
    ///
    /// * `policy` - One of `"strict"`, `"dynamic"`, `"ignore"`.
    ///
    /// # Errors
    ///
    /// Raises a Ruby `ArgumentError` if `policy` is not one of the accepted names.
    fn set_dynamic_field_policy(&self, policy: String) -> Result<(), Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let parsed = DynamicFieldPolicy::from_str(&policy)
            .map_err(|e| Error::new(ruby.exception_arg_error(), e.to_string()))?;
        self.inner.borrow_mut().dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (`"strict"` / `"dynamic"` / `"ignore"`).
    fn dynamic_field_policy(&self) -> String {
        match self.inner.borrow().dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict".to_string(),
            DynamicFieldPolicy::Dynamic => "dynamic".to_string(),
            DynamicFieldPolicy::Ignore => "ignore".to_string(),
        }
    }

    /// Return the list of field names defined in this schema.
    fn field_names(&self) -> Vec<String> {
        self.inner.borrow().fields.keys().cloned().collect()
    }

    /// Return a string representation of this schema.
    fn inspect(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.borrow().fields.keys().collect::<Vec<_>>()
        )
    }
}

/// Register the `Laurus::Schema` class and its methods.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module to define the class under.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let class = module.define_class("Schema", ruby.class_object())?;
    class.define_singleton_method("new", magnus::function!(RbSchema::new, 0))?;
    class.define_method(
        "add_text_field",
        magnus::method!(RbSchema::add_text_field, -1),
    )?;
    class.define_method(
        "add_integer_field",
        magnus::method!(RbSchema::add_integer_field, -1),
    )?;
    class.define_method(
        "add_float_field",
        magnus::method!(RbSchema::add_float_field, -1),
    )?;
    class.define_method(
        "add_boolean_field",
        magnus::method!(RbSchema::add_boolean_field, -1),
    )?;
    class.define_method(
        "add_datetime_field",
        magnus::method!(RbSchema::add_datetime_field, -1),
    )?;
    class.define_method(
        "add_geo_field",
        magnus::method!(RbSchema::add_geo_field, -1),
    )?;
    class.define_method(
        "add_geo3d_field",
        magnus::method!(RbSchema::add_geo3d_field, -1),
    )?;
    class.define_method(
        "add_bytes_field",
        magnus::method!(RbSchema::add_bytes_field, -1),
    )?;
    class.define_method(
        "add_hnsw_field",
        magnus::method!(RbSchema::add_hnsw_field, -1),
    )?;
    class.define_method(
        "add_flat_field",
        magnus::method!(RbSchema::add_flat_field, -1),
    )?;
    class.define_method(
        "add_ivf_field",
        magnus::method!(RbSchema::add_ivf_field, -1),
    )?;
    class.define_method("add_embedder", magnus::method!(RbSchema::add_embedder, 2))?;
    class.define_method(
        "set_default_fields",
        magnus::method!(RbSchema::set_default_fields, 1),
    )?;
    class.define_method(
        "set_dynamic_field_policy",
        magnus::method!(RbSchema::set_dynamic_field_policy, 1),
    )?;
    class.define_method(
        "dynamic_field_policy",
        magnus::method!(RbSchema::dynamic_field_policy, 0),
    )?;
    class.define_method("field_names", magnus::method!(RbSchema::field_names, 0))?;
    class.define_method("inspect", magnus::method!(RbSchema::inspect, 0))?;
    class.define_method("to_s", magnus::method!(RbSchema::inspect, 0))?;
    Ok(())
}
